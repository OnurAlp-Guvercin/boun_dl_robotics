import argparse
import json
import traceback
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast
from collections.abc import Sized

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from homework4 import Hw5Env, bezier


TensorDict = Dict[str, torch.Tensor]


TRAJECTORY_DIM = 5  # [t, e_y, e_z, o_y, o_z]
QUERY_DIM = 1
CONDITION_DIM = 1  # object height
TARGET_DIM = 4  # [e_y, e_z, o_y, o_z]

DEFAULT_DATA_PATH = "data/hw4"
DEFAULT_RUN_DIR = "runs/hw4/cnmp"
DEFAULT_NUM_TRAJECTORIES = 200
DEFAULT_STEPS = 100
DEFAULT_WORKERS = 1
DEFAULT_RENDER_MODE = "offscreen"
DEFAULT_EPOCHS = 500
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_NUM_HIDDEN_LAYERS = 3
DEFAULT_MIN_STD = 0.03
DEFAULT_MAX_CONTEXT = 20
DEFAULT_MAX_TARGET = 40
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_N_TESTS = 200
DEFAULT_SEED = 42
DEFAULT_DEVICE = "auto"

DATASET_FILE = "hw4_dataset.pt"
SPLIT_TRAIN_FILE = "hw4_train.pt"
SPLIT_VAL_FILE = "hw4_val.pt"
SPLIT_TEST_FILE = "hw4_test.pt"
SPLIT_META_FILE = "hw4_split_meta.json"

CMD_COLLECT = "collect"
CMD_TRAIN = "train"
CMD_TEST = "test"
DEFAULT_COMMAND = CMD_TRAIN


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != DEFAULT_DEVICE:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_existing_path(path_str: str) -> Path:
    raw = Path(path_str).expanduser()
    if raw.is_absolute():
        if raw.exists():
            return raw
        raise FileNotFoundError(f"Path does not exist: {raw}")

    src_dir = Path(__file__).resolve().parent
    candidates: List[Path] = []
    for candidate in ((Path.cwd() / raw).resolve(), (src_dir / raw).resolve(), (src_dir.parent / raw).resolve()):
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(f"Path not found: '{path_str}'. Searched:\n{searched}")


def trajectory_points_from_states(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if states.ndim != 2 or states.shape[1] != 5:
        raise ValueError(f"Expected states with shape (steps, 5), got {states.shape}")
    steps = states.shape[0]
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32).reshape(steps, 1)
    trajectory = np.concatenate([t, states[:, :4].astype(np.float32)], axis=1)
    height = np.array([states[0, 4]], dtype=np.float32)
    return trajectory, height


def sample_demonstration(env: Hw5Env, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    env.reset()
    p_1 = np.array([0.5, 0.3, 1.04], dtype=np.float32)
    p_2 = np.array([0.5, 0.15, np.random.uniform(1.04, 1.4)], dtype=np.float32)
    p_3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)], dtype=np.float32)
    p_4 = np.array([0.5, -0.3, 1.04], dtype=np.float32)
    curve = bezier(np.stack([p_1, p_2, p_3, p_4], axis=0), steps=steps)

    env._set_ee_in_cartesian(curve[0], rotation=[-90, 0, 180], n_splits=100, max_iters=100, threshold=0.05)
    states: List[np.ndarray] = []
    for point in curve:
        env._set_ee_pose(point, rotation=[-90, 0, 180], max_iters=10)
        states.append(env.high_level_state().astype(np.float32))
    return trajectory_points_from_states(np.stack(states, axis=0))


def _collect_worker(
    worker_id: int,
    n_trajectories: int,
    steps: int,
    out_dir: str,
    seed: int,
    render_mode: str,
) -> None:
    set_seeds(seed + worker_id)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    error_log = out_path / f"hw4_worker_{worker_id:02d}.error.log"

    try:
        env = Hw5Env(render_mode=render_mode)
        trajectories = torch.zeros((n_trajectories, steps, TRAJECTORY_DIM), dtype=torch.float32)
        heights = torch.zeros((n_trajectories, CONDITION_DIM), dtype=torch.float32)

        for i in tqdm(range(n_trajectories), desc=f"collect[w{worker_id}]", leave=False):
            trajectory, height = sample_demonstration(env, steps=steps)
            trajectories[i] = torch.from_numpy(trajectory)
            heights[i] = torch.from_numpy(height)

        torch.save(
            {
                "trajectories": trajectories,
                "heights": heights,
                "steps": int(steps),
            },
            out_path / f"hw4_shard_{worker_id:02d}.pt",
        )
        error_log.unlink(missing_ok=True)
    except Exception:
        error_log.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def merge_shards(data_dir: Path, cleanup: bool = False) -> Path:
    shard_paths = sorted(data_dir.glob("hw4_shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found under {data_dir}")

    shards: List[TensorDict] = [torch.load(p, map_location="cpu") for p in shard_paths]
    trajectories = torch.cat([s["trajectories"] for s in shards], dim=0)
    heights = torch.cat([s["heights"] for s in shards], dim=0)
    dataset = {
        "trajectories": trajectories,
        "heights": heights,
        "steps": int(trajectories.shape[1]),
    }
    merged_path = data_dir / DATASET_FILE
    torch.save(dataset, merged_path)

    if cleanup:
        for path in shard_paths:
            path.unlink(missing_ok=True)
    return merged_path


def save_dataset_splits(
    dataset_path: Path,
    seed: int,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> Dict[str, int]:
    data = torch.load(dataset_path, map_location="cpu")
    n_total = int(data["trajectories"].shape[0])
    n_val = max(1, int(n_total * val_ratio))
    n_test = max(1, int(n_total * test_ratio))
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough trajectories for train split.")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    def subset(indices: torch.Tensor) -> TensorDict:
        return {
            "trajectories": data["trajectories"][indices],
            "heights": data["heights"][indices],
        }

    data_dir = dataset_path.parent
    torch.save(subset(train_idx), data_dir / SPLIT_TRAIN_FILE)
    torch.save(subset(val_idx), data_dir / SPLIT_VAL_FILE)
    torch.save(subset(test_idx), data_dir / SPLIT_TEST_FILE)

    meta = {
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "n_total": int(n_total),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
    }
    with open(data_dir / SPLIT_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def collect_dataset(
    num_trajectories: int,
    steps: int,
    workers: int,
    out_dir: Path,
    seed: int,
    render_mode: str,
    cleanup: bool = False,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if num_trajectories < 1:
        raise ValueError("num_trajectories must be >= 1")
    if steps < 2:
        raise ValueError("steps must be >= 2")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if num_trajectories < workers:
        workers = num_trajectories

    per_worker = [num_trajectories // workers] * workers
    for i in range(num_trajectories % workers):
        per_worker[i] += 1

    if workers == 1:
        _collect_worker(
            worker_id=0,
            n_trajectories=per_worker[0],
            steps=steps,
            out_dir=str(out_dir),
            seed=seed,
            render_mode=render_mode,
        )
    else:
        procs: List[Tuple[int, Process]] = []
        for worker_id, n_worker_trajectories in enumerate(per_worker):
            proc = Process(
                target=_collect_worker,
                args=(worker_id, n_worker_trajectories, steps, str(out_dir), seed, render_mode),
            )
            proc.start()
            procs.append((worker_id, proc))
        for worker_id, proc in procs:
            proc.join()
            if proc.exitcode != 0:
                err_path = out_dir / f"hw4_worker_{worker_id:02d}.error.log"
                if err_path.exists():
                    raise RuntimeError(
                        f"Collector worker failed with exit code {proc.exitcode}.\n"
                        f"Worker traceback:\n{err_path.read_text(encoding='utf-8')}"
                    )
                raise RuntimeError(f"Collector worker failed with exit code {proc.exitcode}")

    merged_path = merge_shards(out_dir, cleanup=cleanup)
    save_dataset_splits(
        dataset_path=merged_path,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    return merged_path


class Hw4TrajectoryDataset(Dataset[TensorDict]):
    def __init__(self, data: TensorDict) -> None:
        self.trajectories = data["trajectories"].float()
        self.heights = data["heights"].float()
        if self.trajectories.ndim != 3 or self.trajectories.shape[-1] != TRAJECTORY_DIM:
            raise ValueError(f"Expected trajectories with shape (N, T, {TRAJECTORY_DIM})")
        if self.heights.ndim != 2 or self.heights.shape[-1] != CONDITION_DIM:
            raise ValueError(f"Expected heights with shape (N, {CONDITION_DIM})")
        if self.trajectories.shape[0] != self.heights.shape[0]:
            raise ValueError("trajectories and heights have different lengths")

    def __len__(self) -> int:
        return int(self.trajectories.shape[0])

    def __getitem__(self, idx: int) -> TensorDict:
        return {
            "trajectory": self.trajectories[idx],
            "height": self.heights[idx],
        }


@dataclass
class SplitLoaders:
    train: DataLoader[TensorDict]
    val: DataLoader[TensorDict]
    test: DataLoader[TensorDict]


def load_hw4_dataset(data_path: Path) -> Hw4TrajectoryDataset:
    if data_path.is_dir():
        dataset_path = data_path / DATASET_FILE
        if not dataset_path.exists():
            dataset_path = merge_shards(data_path)
    else:
        dataset_path = data_path
    data = torch.load(dataset_path, map_location="cpu")
    return Hw4TrajectoryDataset(data)


def build_loaders(
    dataset: Dataset[TensorDict],
    batch_size: int,
    seed: int,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> SplitLoaders:
    dataset_sized = cast(Sized, dataset)
    if len(dataset_sized) < 10:
        raise ValueError("Dataset is too small. Collect at least 10 trajectories.")

    n_total = len(dataset_sized)
    n_val = max(1, int(n_total * val_ratio))
    n_test = max(1, int(n_total * test_ratio))
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough trajectories for train split.")

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=generator)
    return SplitLoaders(
        train=DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        val=DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0),
        test=DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def load_split_loaders(
    data_path: Path,
    batch_size: int,
    seed: int,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> SplitLoaders:
    if data_path.is_dir():
        train_path = data_path / SPLIT_TRAIN_FILE
        val_path = data_path / SPLIT_VAL_FILE
        test_path = data_path / SPLIT_TEST_FILE
        if train_path.exists() and val_path.exists() and test_path.exists():
            return SplitLoaders(
                train=DataLoader(Hw4TrajectoryDataset(torch.load(train_path, map_location="cpu")), batch_size=batch_size, shuffle=True),
                val=DataLoader(Hw4TrajectoryDataset(torch.load(val_path, map_location="cpu")), batch_size=batch_size, shuffle=False),
                test=DataLoader(Hw4TrajectoryDataset(torch.load(test_path, map_location="cpu")), batch_size=batch_size, shuffle=False),
            )

    dataset = load_hw4_dataset(data_path)
    return build_loaders(
        dataset=dataset,
        batch_size=batch_size,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


def compute_normalizer(loader: DataLoader[TensorDict]) -> TensorDict:
    trajectories = []
    heights = []
    for batch in loader:
        trajectories.append(batch["trajectory"])
        heights.append(batch["height"])
    traj = torch.cat(trajectories, dim=0)
    height = torch.cat(heights, dim=0)
    target = traj[:, :, 1:]
    return {
        "t_mean": traj[:, :, :1].mean(dim=(0, 1)),
        "t_std": traj[:, :, :1].std(dim=(0, 1)).clamp_min(1e-6),
        "y_mean": target.mean(dim=(0, 1)),
        "y_std": target.std(dim=(0, 1)).clamp_min(1e-6),
        "h_mean": height.mean(dim=0),
        "h_std": height.std(dim=0).clamp_min(1e-6),
    }


def normalizer_to_device(normalizer: TensorDict, device: torch.device) -> TensorDict:
    return {key: value.to(device=device, dtype=torch.float32) for key, value in normalizer.items()}


def normalizer_to_jsonable(normalizer: TensorDict) -> Dict[str, List[float]]:
    return {key: value.detach().cpu().view(-1).tolist() for key, value in normalizer.items()}


def normalize_t(t: torch.Tensor, normalizer: TensorDict) -> torch.Tensor:
    return (t - normalizer["t_mean"]) / normalizer["t_std"]


def normalize_y(y: torch.Tensor, normalizer: TensorDict) -> torch.Tensor:
    return (y - normalizer["y_mean"]) / normalizer["y_std"]


def denormalize_y(y: torch.Tensor, normalizer: TensorDict) -> torch.Tensor:
    return y * normalizer["y_std"] + normalizer["y_mean"]


def normalize_h(h: torch.Tensor, normalizer: TensorDict) -> torch.Tensor:
    return (h - normalizer["h_mean"]) / normalizer["h_std"]


class CNMP(nn.Module):
    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_hidden_layers: int = DEFAULT_NUM_HIDDEN_LAYERS,
        min_std: float = DEFAULT_MIN_STD,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.min_std = float(min_std)

        encoder_layers: List[nn.Module] = [
            nn.Linear(QUERY_DIM + TARGET_DIM, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_hidden_layers - 1):
            encoder_layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        encoder_layers.append(nn.Linear(hidden_size, hidden_size))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = [
            nn.Linear(hidden_size + QUERY_DIM + CONDITION_DIM, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_hidden_layers - 1):
            decoder_layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        decoder_layers.append(nn.Linear(hidden_size, 2 * TARGET_DIM))
        self.decoder = nn.Sequential(*decoder_layers)

    def aggregate(self, encoded_context: torch.Tensor, context_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if context_mask is None:
            return encoded_context.mean(dim=1)
        masked = encoded_context * context_mask.unsqueeze(-1)
        normalizer = context_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return masked.sum(dim=1) / normalizer

    def forward(
        self,
        context: torch.Tensor,
        target_t: torch.Tensor,
        target_h: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_context = self.encoder(context)
        representation = self.aggregate(encoded_context, context_mask=context_mask)
        n_targets = int(target_t.shape[1])
        representation = representation.unsqueeze(1).expand(-1, n_targets, -1)
        decoder_input = torch.cat([representation, target_t, target_h], dim=-1)
        decoder_output = self.decoder(decoder_input)
        mean = decoder_output[..., :TARGET_DIM]
        raw_std = decoder_output[..., TARGET_DIM:]
        std = torch.nn.functional.softplus(raw_std) + self.min_std
        return mean, std

    def nll_loss(
        self,
        context: torch.Tensor,
        target_t: torch.Tensor,
        target_h: torch.Tensor,
        target_y: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        target_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        mean, std = self.forward(context, target_t, target_h, context_mask=context_mask)
        nll = -torch.distributions.Normal(mean, std).log_prob(target_y)
        if target_mask is None:
            return nll.mean()
        masked_nll = nll * target_mask.unsqueeze(-1)
        denom = (target_mask.sum() * TARGET_DIM).clamp_min(1.0)
        return masked_nll.sum() / denom


def sample_cnmp_batch(
    trajectories: torch.Tensor,
    heights: torch.Tensor,
    max_context: int,
    max_target: int,
    normalizer: TensorDict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    trajectories = trajectories.to(device=device, dtype=torch.float32)
    heights = heights.to(device=device, dtype=torch.float32)
    batch_size, steps, _ = trajectories.shape
    max_context = min(max(1, max_context), steps)
    max_target = min(max(1, max_target), steps)

    context = torch.zeros((batch_size, max_context, QUERY_DIM + TARGET_DIM), device=device)
    context_mask = torch.zeros((batch_size, max_context), device=device)
    target_t = torch.zeros((batch_size, max_target, QUERY_DIM), device=device)
    target_h = torch.zeros((batch_size, max_target, CONDITION_DIM), device=device)
    target_y = torch.zeros((batch_size, max_target, TARGET_DIM), device=device)
    target_y_raw = torch.zeros((batch_size, max_target, TARGET_DIM), device=device)
    target_mask = torch.zeros((batch_size, max_target), device=device)

    for batch_idx in range(batch_size):
        n_context = int(np.random.randint(1, max_context + 1))
        n_target = int(np.random.randint(1, max_target + 1))
        context_idx = torch.randperm(steps, device=device)[:n_context]
        target_idx = torch.randperm(steps, device=device)[:n_target]

        context_points = trajectories[batch_idx, context_idx]
        target_points = trajectories[batch_idx, target_idx]

        context_t = normalize_t(context_points[:, :1], normalizer)
        context_y = normalize_y(context_points[:, 1:], normalizer)
        context[batch_idx, :n_context] = torch.cat([context_t, context_y], dim=-1)
        context_mask[batch_idx, :n_context] = 1.0

        target_t[batch_idx, :n_target] = normalize_t(target_points[:, :1], normalizer)
        target_h[batch_idx, :n_target] = normalize_h(heights[batch_idx], normalizer).unsqueeze(0).expand(n_target, -1)
        target_y[batch_idx, :n_target] = normalize_y(target_points[:, 1:], normalizer)
        target_y_raw[batch_idx, :n_target] = target_points[:, 1:]
        target_mask[batch_idx, :n_target] = 1.0

    return context, target_t, target_h, target_y, context_mask, target_mask, target_y_raw


def evaluate_nll(
    model: CNMP,
    loader: DataLoader[TensorDict],
    normalizer: TensorDict,
    max_context: int,
    max_target: int,
    device: torch.device,
    desc: str,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            cnmp_batch = sample_cnmp_batch(
                trajectories=batch["trajectory"],
                heights=batch["height"],
                max_context=max_context,
                max_target=max_target,
                normalizer=normalizer,
                device=device,
            )
            context, target_t, target_h, target_y, context_mask, target_mask, _ = cnmp_batch
            loss = model.nll_loss(context, target_t, target_h, target_y, context_mask, target_mask)
            total_loss += float(loss.item())
            total_batches += 1
    return total_loss / max(1, total_batches)


def mse_tests(
    model: CNMP,
    test_loader: DataLoader[TensorDict],
    normalizer: TensorDict,
    max_context: int,
    max_target: int,
    n_tests: int,
    device: torch.device,
) -> Dict[str, object]:
    test_batches = list(test_loader)
    if not test_batches:
        raise ValueError("Empty test loader")

    model.eval()
    ee_errors: List[float] = []
    obj_errors: List[float] = []

    with torch.no_grad():
        progress = tqdm(total=n_tests, desc="test", leave=False)
        while len(ee_errors) < n_tests:
            batch = test_batches[np.random.randint(0, len(test_batches))]
            trajectories = batch["trajectory"]
            heights = batch["height"]
            batch_size = int(trajectories.shape[0])
            take = min(batch_size, n_tests - len(ee_errors))
            if take < batch_size:
                indices = torch.randperm(batch_size)[:take]
                trajectories = trajectories[indices]
                heights = heights[indices]

            cnmp_batch = sample_cnmp_batch(
                trajectories=trajectories,
                heights=heights,
                max_context=max_context,
                max_target=max_target,
                normalizer=normalizer,
                device=device,
            )
            context, target_t, target_h, _, context_mask, target_mask, target_y_raw = cnmp_batch
            mean_norm, _ = model(context, target_t, target_h, context_mask=context_mask)
            pred_raw = denormalize_y(mean_norm, normalizer)
            sq_error = (pred_raw - target_y_raw).pow(2) * target_mask.unsqueeze(-1)
            denom = target_mask.sum(dim=1).clamp_min(1.0)
            ee_batch = sq_error[:, :, :2].sum(dim=(1, 2)) / (denom * 2.0)
            obj_batch = sq_error[:, :, 2:].sum(dim=(1, 2)) / (denom * 2.0)
            ee_errors.extend(ee_batch.detach().cpu().tolist())
            obj_errors.extend(obj_batch.detach().cpu().tolist())
            progress.update(take)
        progress.close()

    ee_arr = np.array(ee_errors[:n_tests], dtype=np.float64)
    obj_arr = np.array(obj_errors[:n_tests], dtype=np.float64)
    return {
        "n_tests": int(n_tests),
        "end_effector": {
            "mean_mse": float(ee_arr.mean()),
            "std_mse": float(ee_arr.std(ddof=0)),
            "values": ee_arr.tolist(),
        },
        "object": {
            "mean_mse": float(obj_arr.mean()),
            "std_mse": float(obj_arr.std(ddof=0)),
            "values": obj_arr.tolist(),
        },
    }


def save_mse_bar_plot(metrics: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ee = cast(Dict[str, object], metrics["end_effector"])
    obj = cast(Dict[str, object], metrics["object"])
    labels = ["Object", "End-effector"]
    means = [float(obj["mean_mse"]), float(ee["mean_mse"])]
    stds = [float(obj["std_mse"]), float(ee["std_mse"])]

    plt.figure(figsize=(6, 4))
    colors = ["#3A7CA5", "#D95D39"]
    plt.bar(labels, means, yerr=stds, capsize=8, color=colors, edgecolor="black", linewidth=0.8)
    plt.ylabel("MSE")
    plt.title("CNMP Prediction Error")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_training_plots(history: List[Dict[str, float]], out_dir: Path) -> None:
    if not history:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(item["epoch"]) for item in history]
    train_nll = [float(item["train_nll"]) for item in history]
    val_nll = [float(item["val_nll"]) for item in history]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_nll, label="train_nll", linewidth=1.5)
    plt.plot(epochs, val_nll, label="val_nll", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.title("CNMP Train/Val Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_plot.png", dpi=150)
    plt.close()


def train(
    data_path: str = DEFAULT_DATA_PATH,
    run_dir: str = DEFAULT_RUN_DIR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_hidden_layers: int = DEFAULT_NUM_HIDDEN_LAYERS,
    min_std: float = DEFAULT_MIN_STD,
    max_context: int = DEFAULT_MAX_CONTEXT,
    max_target: int = DEFAULT_MAX_TARGET,
    seed: int = DEFAULT_SEED,
    device: str = DEFAULT_DEVICE,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> Dict[str, float]:
    set_seeds(seed)
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)

    loaders = load_split_loaders(
        data_path=Path(data_path),
        batch_size=batch_size,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    normalizer = normalizer_to_device(compute_normalizer(loaders.train), dev)

    model = CNMP(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        min_std=min_std,
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    history: List[Dict[str, float]] = []
    for epoch in tqdm(range(1, epochs + 1), desc="epochs", leave=True):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch in tqdm(loaders.train, desc=f"train e{epoch}", leave=False):
            cnmp_batch = sample_cnmp_batch(
                trajectories=batch["trajectory"],
                heights=batch["height"],
                max_context=max_context,
                max_target=max_target,
                normalizer=normalizer,
                device=dev,
            )
            context, target_t, target_h, target_y, context_mask, target_mask, _ = cnmp_batch
            loss = model.nll_loss(context, target_t, target_h, target_y, context_mask, target_mask)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_batches += 1

        train_nll = train_loss_sum / max(1, train_batches)
        val_nll = evaluate_nll(
            model=model,
            loader=loaders.val,
            normalizer=normalizer,
            max_context=max_context,
            max_target=max_target,
            device=dev,
            desc=f"val e{epoch}",
        )
        history.append({"epoch": float(epoch), "train_nll": float(train_nll), "val_nll": float(val_nll)})
        print(f"[CNMP] epoch={epoch}/{epochs} train_nll={train_nll:.6f} val_nll={val_nll:.6f}")

        if val_nll < best_val:
            best_val = val_nll
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "normalizer": {key: value.detach().cpu() for key, value in normalizer.items()},
                    "config": {
                        "hidden_size": int(hidden_size),
                        "num_hidden_layers": int(num_hidden_layers),
                        "min_std": float(min_std),
                        "max_context": int(max_context),
                        "max_target": int(max_target),
                    },
                },
                out_dir / "best.pt",
            )

    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_nll": float(best_val),
                "history": history,
                "normalizer": normalizer_to_jsonable(normalizer),
            },
            f,
            indent=2,
        )
    save_training_plots(history=history, out_dir=out_dir)
    print("[CNMP] training completed. Best checkpoint: best.pt")
    return {"best_val_nll": float(best_val)}


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[CNMP, TensorDict, Dict[str, object]]:
    ckpt_path = resolve_existing_path(checkpoint_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get("config", {})
    model = CNMP(
        hidden_size=int(config.get("hidden_size", DEFAULT_HIDDEN_SIZE)),
        num_hidden_layers=int(config.get("num_hidden_layers", DEFAULT_NUM_HIDDEN_LAYERS)),
        min_std=float(config.get("min_std", DEFAULT_MIN_STD)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    normalizer = normalizer_to_device(checkpoint["normalizer"], device)
    return model, normalizer, dict(config)


def test(
    data_path: str = DEFAULT_DATA_PATH,
    checkpoint_path: str = f"{DEFAULT_RUN_DIR}/best.pt",
    run_dir: str = DEFAULT_RUN_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_tests: int = DEFAULT_N_TESTS,
    max_context: Optional[int] = None,
    max_target: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    device: str = DEFAULT_DEVICE,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> Dict[str, object]:
    set_seeds(seed)
    dev = resolve_device(device)
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, normalizer, config = load_checkpoint(checkpoint_path=checkpoint_path, device=dev)
    eval_max_context = int(max_context if max_context is not None else config.get("max_context", DEFAULT_MAX_CONTEXT))
    eval_max_target = int(max_target if max_target is not None else config.get("max_target", DEFAULT_MAX_TARGET))

    loaders = load_split_loaders(
        data_path=Path(data_path),
        batch_size=batch_size,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    metrics = mse_tests(
        model=model,
        test_loader=loaders.test,
        normalizer=normalizer,
        max_context=eval_max_context,
        max_target=eval_max_target,
        n_tests=max(DEFAULT_N_TESTS, n_tests),
        device=dev,
    )
    save_mse_bar_plot(metrics=metrics, out_path=out_dir / "hw4_cnmp_mse_bar.png")

    with open(out_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "config": {
                    "checkpoint_path": str(resolve_existing_path(checkpoint_path)),
                    "data_path": data_path,
                    "batch_size": int(batch_size),
                    "max_context": int(eval_max_context),
                    "max_target": int(eval_max_target),
                    "seed": int(seed),
                    "device": str(dev),
                },
            },
            f,
            indent=2,
        )

    ee = cast(Dict[str, object], metrics["end_effector"])
    obj = cast(Dict[str, object], metrics["object"])
    print(
        "[CNMP] test MSE "
        f"end_effector={float(ee['mean_mse']):.8f} +/- {float(ee['std_mse']):.8f}, "
        f"object={float(obj['mean_mse']):.8f} +/- {float(obj['std_mse']):.8f}"
    )
    return metrics


def collect(
    num_trajectories: int = DEFAULT_NUM_TRAJECTORIES,
    steps: int = DEFAULT_STEPS,
    workers: int = DEFAULT_WORKERS,
    out_dir: str = DEFAULT_DATA_PATH,
    seed: int = DEFAULT_SEED,
    render_mode: str = DEFAULT_RENDER_MODE,
    cleanup: bool = False,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> Path:
    set_seeds(seed)
    print(
        f"[collect] num_trajectories={num_trajectories}, steps={steps}, workers={workers}, "
        f"out_dir={out_dir}, render_mode={render_mode}"
    )
    merged_path = collect_dataset(
        num_trajectories=num_trajectories,
        steps=steps,
        workers=workers,
        out_dir=Path(out_dir),
        seed=seed,
        render_mode=render_mode,
        cleanup=cleanup,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    data = torch.load(merged_path, map_location="cpu")
    print(f"Saved dataset to {merged_path}")
    print(f"num_trajectories={data['trajectories'].shape[0]}, steps={data['trajectories'].shape[1]}")
    split_meta_path = Path(out_dir) / SPLIT_META_FILE
    if split_meta_path.exists():
        meta = json.loads(split_meta_path.read_text(encoding="utf-8"))
        print(f"split counts: train={meta['n_train']} val={meta['n_val']} test={meta['n_test']}")
    return merged_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Homework 4 - CNMP learning from demonstrations.")
    sub = parser.add_subparsers(dest="command", required=False)

    p_collect = sub.add_parser(CMD_COLLECT)
    p_collect.add_argument("--num-trajectories", type=int, default=DEFAULT_NUM_TRAJECTORIES)
    p_collect.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_collect.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p_collect.add_argument("--out-dir", type=str, default=DEFAULT_DATA_PATH)
    p_collect.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_collect.add_argument("--render-mode", type=str, default=DEFAULT_RENDER_MODE, choices=["offscreen", "gui"])
    p_collect.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    p_collect.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    p_collect.add_argument("--cleanup", action="store_true")

    p_train = sub.add_parser(CMD_TRAIN)
    p_train.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    p_train.add_argument("--run-dir", type=str, default=DEFAULT_RUN_DIR)
    p_train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p_train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_train.add_argument("--lr", type=float, default=DEFAULT_LR)
    p_train.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p_train.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    p_train.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    p_train.add_argument("--num-hidden-layers", type=int, default=DEFAULT_NUM_HIDDEN_LAYERS)
    p_train.add_argument("--min-std", type=float, default=DEFAULT_MIN_STD)
    p_train.add_argument("--max-context", type=int, default=DEFAULT_MAX_CONTEXT)
    p_train.add_argument("--max-target", type=int, default=DEFAULT_MAX_TARGET)
    p_train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_train.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p_train.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    p_train.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)

    p_test = sub.add_parser(CMD_TEST)
    p_test.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    p_test.add_argument("--checkpoint-path", type=str, default=f"{DEFAULT_RUN_DIR}/best.pt")
    p_test.add_argument("--run-dir", type=str, default=DEFAULT_RUN_DIR)
    p_test.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_test.add_argument("--n-tests", type=int, default=DEFAULT_N_TESTS)
    p_test.add_argument("--max-context", type=int, default=None)
    p_test.add_argument("--max-target", type=int, default=None)
    p_test.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_test.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p_test.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    p_test.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)

    args = parser.parse_args()
    if args.command is None:
        args.command = DEFAULT_COMMAND

    if args.command == CMD_COLLECT:
        collect(
            num_trajectories=getattr(args, "num_trajectories", DEFAULT_NUM_TRAJECTORIES),
            steps=getattr(args, "steps", DEFAULT_STEPS),
            workers=getattr(args, "workers", DEFAULT_WORKERS),
            out_dir=getattr(args, "out_dir", DEFAULT_DATA_PATH),
            seed=getattr(args, "seed", DEFAULT_SEED),
            render_mode=getattr(args, "render_mode", DEFAULT_RENDER_MODE),
            cleanup=getattr(args, "cleanup", False),
            val_ratio=getattr(args, "val_ratio", DEFAULT_VAL_RATIO),
            test_ratio=getattr(args, "test_ratio", DEFAULT_TEST_RATIO),
        )
    elif args.command == CMD_TRAIN:
        train(
            data_path=getattr(args, "data_path", DEFAULT_DATA_PATH),
            run_dir=getattr(args, "run_dir", DEFAULT_RUN_DIR),
            epochs=getattr(args, "epochs", DEFAULT_EPOCHS),
            batch_size=getattr(args, "batch_size", DEFAULT_BATCH_SIZE),
            lr=getattr(args, "lr", DEFAULT_LR),
            weight_decay=getattr(args, "weight_decay", DEFAULT_WEIGHT_DECAY),
            grad_clip=getattr(args, "grad_clip", DEFAULT_GRAD_CLIP),
            hidden_size=getattr(args, "hidden_size", DEFAULT_HIDDEN_SIZE),
            num_hidden_layers=getattr(args, "num_hidden_layers", DEFAULT_NUM_HIDDEN_LAYERS),
            min_std=getattr(args, "min_std", DEFAULT_MIN_STD),
            max_context=getattr(args, "max_context", DEFAULT_MAX_CONTEXT),
            max_target=getattr(args, "max_target", DEFAULT_MAX_TARGET),
            seed=getattr(args, "seed", DEFAULT_SEED),
            device=getattr(args, "device", DEFAULT_DEVICE),
            val_ratio=getattr(args, "val_ratio", DEFAULT_VAL_RATIO),
            test_ratio=getattr(args, "test_ratio", DEFAULT_TEST_RATIO),
        )
    else:
        test(
            data_path=getattr(args, "data_path", DEFAULT_DATA_PATH),
            checkpoint_path=getattr(args, "checkpoint_path", f"{DEFAULT_RUN_DIR}/best.pt"),
            run_dir=getattr(args, "run_dir", DEFAULT_RUN_DIR),
            batch_size=getattr(args, "batch_size", DEFAULT_BATCH_SIZE),
            n_tests=getattr(args, "n_tests", DEFAULT_N_TESTS),
            max_context=getattr(args, "max_context", None),
            max_target=getattr(args, "max_target", None),
            seed=getattr(args, "seed", DEFAULT_SEED),
            device=getattr(args, "device", DEFAULT_DEVICE),
            val_ratio=getattr(args, "val_ratio", DEFAULT_VAL_RATIO),
            test_ratio=getattr(args, "test_ratio", DEFAULT_TEST_RATIO),
        )


if __name__ == "__main__":
    main()
