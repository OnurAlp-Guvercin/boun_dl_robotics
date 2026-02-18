import argparse
import json
import math
import traceback
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from homework1 import Hw1Env

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


N_ACTIONS = 4
IMG_SIZE = 128
IMG_SHAPE = (3, IMG_SIZE, IMG_SIZE)

DEFAULT_NUM_SAMPLES = 1000
DEFAULT_WORKERS = 128
DEFAULT_DATA_PATH = "data/hw1"
DEFAULT_RUN_DIR = "runs/hw1/mlp_pos"
DEFAULT_EPOCHS = 25
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_HUBER_BETA = 0.05
DEFAULT_LR_SCHED_FACTOR = 0.5
DEFAULT_LR_SCHED_PATIENCE = 3
DEFAULT_SEED = 42
DEFAULT_DEVICE = "auto"

CMD_COLLECT = "collect"
CMD_TRAIN = "train"
CMD_TEST = "test"
DEFAULT_COMMAND = CMD_TEST


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collect_worker(worker_id: int, n_samples: int, out_dir: str, seed: int) -> None:
    set_seeds(seed + worker_id)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    error_log = out_path / f"hw1_worker_{worker_id:02d}.error.log"
    try:
        env = Hw1Env(render_mode="offscreen")
        imgs_before = torch.zeros((n_samples, *IMG_SHAPE), dtype=torch.uint8)
        actions = torch.zeros((n_samples,), dtype=torch.uint8)
        pos_after = torch.zeros((n_samples, 2), dtype=torch.float32)
        imgs_after = torch.zeros((n_samples, *IMG_SHAPE), dtype=torch.uint8)

        for i in tqdm(range(n_samples), desc=f"collect[w{worker_id}]", leave=False):
            env.reset()
            _, img_before = env.state()
            action_id = int(np.random.randint(N_ACTIONS))
            env.step(action_id)
            obj_pos_after, img_after = env.state()

            imgs_before[i] = img_before
            actions[i] = action_id
            pos_after[i] = torch.tensor(obj_pos_after, dtype=torch.float32)
            imgs_after[i] = img_after

        shard = {
            "imgs_before": imgs_before,
            "actions": actions,
            "pos_after": pos_after,
            "imgs_after": imgs_after,
        }
        torch.save(shard, out_path / f"hw1_shard_{worker_id:02d}.pt")
        error_log.unlink(missing_ok=True)
    except Exception:
        tb = traceback.format_exc()
        error_log.write_text(tb, encoding="utf-8")
        raise


def merge_shards(data_dir: Path, cleanup: bool = False) -> Path:
    shard_paths = sorted(data_dir.glob("hw1_shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found under {data_dir}")

    shards: List[Dict[str, torch.Tensor]] = [torch.load(p, map_location="cpu") for p in shard_paths]
    merged = {
        "imgs_before": torch.cat([s["imgs_before"] for s in shards], dim=0),
        "actions": torch.cat([s["actions"] for s in shards], dim=0),
        "pos_after": torch.cat([s["pos_after"] for s in shards], dim=0),
        "imgs_after": torch.cat([s["imgs_after"] for s in shards], dim=0),
    }
    merged_path = data_dir / "hw1_dataset.pt"
    torch.save(merged, merged_path)

    if cleanup:
        for p in shard_paths:
            p.unlink(missing_ok=True)
    return merged_path


def collect_dataset(num_samples: int, workers: int, out_dir: Path, seed: int, cleanup: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if num_samples < workers:
        workers = num_samples

    per_worker = [num_samples // workers] * workers
    for i in range(num_samples % workers):
        per_worker[i] += 1

    if workers == 1:
        _collect_worker(worker_id=0, n_samples=per_worker[0], out_dir=str(out_dir), seed=seed)
    else:
        procs: List[Tuple[int, Process]] = []
        for i, n in enumerate(per_worker):
            p = Process(target=_collect_worker, args=(i, n, str(out_dir), seed))
            p.start()
            procs.append((i, p))
        for worker_id, p in procs:
            p.join()
            if p.exitcode != 0:
                err_path = out_dir / f"hw1_worker_{worker_id:02d}.error.log"
                if err_path.exists():
                    err_text = err_path.read_text(encoding="utf-8")
                    raise RuntimeError(
                        f"Collector worker failed with exit code {p.exitcode}.\n"
                        f"Worker traceback:\n{err_text}"
                    )
                raise RuntimeError(f"Collector worker failed with exit code {p.exitcode}")

    return merge_shards(out_dir, cleanup=cleanup)


class Hw1Dataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]) -> None:
        self.imgs_before = data["imgs_before"].float() / 255.0
        self.actions = data["actions"].long()
        self.pos_after = data["pos_after"].float()
        self.imgs_after = data["imgs_after"].float() / 255.0

    def __len__(self) -> int:
        return self.actions.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        action_id = self.actions[idx]
        action_onehot = F.one_hot(action_id, num_classes=N_ACTIONS).float()
        return {
            "img_before": self.imgs_before[idx],
            "action_id": action_id,
            "action_onehot": action_onehot,
            "pos_after": self.pos_after[idx],
            "img_after": self.imgs_after[idx],
        }


@dataclass
class SplitLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def build_loaders(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> SplitLoaders:
    if len(dataset) < 10:
        raise ValueError("Dataset is too small. Collect at least 10 samples.")

    n_total = len(dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_test = max(1, int(n_total * test_ratio))
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough samples for train split.")

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    return SplitLoaders(
        train=DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        val=DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0),
        test=DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def load_hw1_dataset(data_path: Path) -> Hw1Dataset:
    if data_path.is_dir():
        dataset_path = data_path / "hw1_dataset.pt"
        if not dataset_path.exists():
            dataset_path = merge_shards(data_path)
    else:
        dataset_path = data_path
    data = torch.load(dataset_path, map_location="cpu")
    return Hw1Dataset(data)


class PositionMLP(nn.Module):
    def __init__(self, hidden_dims: Tuple[int, int, int] = (256, 128, 64)) -> None:
        super().__init__()
        in_dim = int(np.prod(IMG_SHAPE)) + N_ACTIONS
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 2),
        )

    def forward(self, img_before: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        x = img_before.flatten(start_dim=1)
        x = torch.cat([x, action_onehot], dim=1)
        return self.net(x)


def evaluate(model: nn.Module, loader, device: torch.device, desc: str = "eval") -> Dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            img_before = batch["img_before"].to(device)
            action_onehot = batch["action_onehot"].to(device)
            target = batch["pos_after"].to(device)
            pred = model(img_before, action_onehot)
            b = img_before.size(0)
            total_mse += F.mse_loss(pred, target, reduction="sum").item()
            total_mae += F.l1_loss(pred, target, reduction="sum").item()
            total_count += b

    mse = total_mse / (total_count * 2)
    mae = total_mae / (total_count * 2)
    rmse = math.sqrt(mse)
    return {"mse": mse, "mae": mae, "rmse": rmse}


def save_loss_plots(
    out_dir: Path,
    step_history: Optional[List[Dict[str, float]]],
    history: Optional[List[Dict[str, float]]],
) -> None:
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    if step_history:
        steps = [int(x["step"]) for x in step_history]
        losses = [float(x["train_mse"]) for x in step_history]
        plt.figure(figsize=(8, 4))
        plt.plot(steps, losses, linewidth=1.2)
        plt.xlabel("Step")
        plt.ylabel("Train MSE")
        plt.title("MLP Train Loss (Step)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "loss_step_plot.png", dpi=150)
        plt.close()

    if history:
        epochs = [int(x["epoch"]) for x in history]
        train_vals = [float(x["train_mse"]) for x in history]
        val_vals = [float(x["val_mse"]) for x in history]
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, train_vals, label="train_mse", linewidth=1.5)
        plt.plot(epochs, val_vals, label="val_mse", linewidth=1.5)
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("MLP Train/Val Loss (Epoch)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "loss_epoch_plot.png", dpi=150)
        plt.close()


def train(
    data_path: str = DEFAULT_DATA_PATH,
    run_dir: str = DEFAULT_RUN_DIR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    huber_beta: float = DEFAULT_HUBER_BETA,
    lr_sched_factor: float = DEFAULT_LR_SCHED_FACTOR,
    lr_sched_patience: int = DEFAULT_LR_SCHED_PATIENCE,
    seed: int = DEFAULT_SEED,
    device: str = DEFAULT_DEVICE,
) -> Dict[str, float]:
    set_seeds(seed)
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)

    dataset = load_hw1_dataset(Path(data_path))
    loaders = build_loaders(dataset, batch_size=batch_size, seed=seed)

    model = PositionMLP().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_sched_factor,
        patience=lr_sched_patience,
    )
    loss_fn = nn.SmoothL1Loss(beta=huber_beta)

    best_val = float("inf")
    history: List[Dict[str, float]] = []
    step_history: List[Dict[str, float]] = []
    global_step = 0

    for epoch in tqdm(range(1, epochs + 1), desc="epochs", leave=True):
        model.train()
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_count = 0

        for batch in tqdm(loaders.train, desc=f"train e{epoch}", leave=False):
            img_before = batch["img_before"].to(dev)
            action_onehot = batch["action_onehot"].to(dev)
            target = batch["pos_after"].to(dev)
            pred = model(img_before, action_onehot)
            loss = loss_fn(pred, target)
            batch_mse = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            global_step += 1

            train_loss_sum += loss.item() * img_before.size(0)
            train_mse_sum += batch_mse.item() * img_before.size(0)
            train_count += img_before.size(0)
            step_history.append(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "train_loss": float(loss.item()),
                    "train_mse": float(batch_mse.item()),
                }
            )

        train_loss = train_loss_sum / max(train_count, 1)
        train_mse = train_mse_sum / max(train_count, 1)
        val_metrics = evaluate(model, loaders.val, dev, desc=f"val e{epoch}")
        scheduler.step(val_metrics["mse"])
        curr_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "lr": curr_lr,
                "train_loss": train_loss,
                "train_mse": train_mse,
                "val_mse": val_metrics["mse"],
            }
        )
        print(
            f"[MLP] epoch={epoch}/{epochs} train_loss={train_loss:.6f} train_mse={train_mse:.6f} "
            f"val_mse={val_metrics['mse']:.6f} val_rmse={val_metrics['rmse']:.6f} lr={curr_lr:.2e}"
        )

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    test_metrics = test(
        data_path=data_path,
        checkpoint_path=str(out_dir / "best.pt"),
        batch_size=batch_size,
        seed=seed,
        device=device,
        run_dir=run_dir,
        history=history,
        step_history=step_history,
    )
    return test_metrics


def test(
    data_path: str = DEFAULT_DATA_PATH,
    checkpoint_path: str = f"{DEFAULT_RUN_DIR}/best.pt",
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    device: str = DEFAULT_DEVICE,
    run_dir: str = DEFAULT_RUN_DIR,
    history: Optional[List[Dict[str, float]]] = None,
    step_history: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, float]:
    set_seeds(seed)
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)

    dataset = load_hw1_dataset(Path(data_path))
    loaders = build_loaders(dataset, batch_size=batch_size, seed=seed)

    model = PositionMLP().to(dev)
    model.load_state_dict(torch.load(checkpoint_path, map_location=dev))
    metrics = evaluate(model, loaders.test, dev, desc="test")
    print(f"[MLP] test metrics: {metrics}")

    payload: Dict[str, object] = {"test": metrics}
    if history is not None:
        payload["history"] = history
    if step_history is not None:
        payload["step_history"] = step_history
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    save_loss_plots(out_dir=out_dir, step_history=step_history, history=history)
    return metrics


def collect(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    workers: int = DEFAULT_WORKERS,
    out_dir: str = DEFAULT_DATA_PATH,
    seed: int = DEFAULT_SEED,
    cleanup: bool = False,
) -> Path:
    set_seeds(seed)
    print(f"[collect] num_samples={num_samples}, workers={workers}, out_dir={out_dir}")
    merged_path = collect_dataset(
        num_samples=num_samples,
        workers=workers,
        out_dir=Path(out_dir),
        seed=seed,
        cleanup=cleanup,
    )
    data = torch.load(merged_path, map_location="cpu")
    print(f"Saved dataset to {merged_path}")
    print(f"num_samples={data['actions'].shape[0]}")
    return merged_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Homework 1 - MLP position prediction.")
    sub = parser.add_subparsers(dest="command", required=False)

    p_collect = sub.add_parser(CMD_COLLECT)
    p_collect.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    p_collect.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p_collect.add_argument("--out-dir", type=str, default=DEFAULT_DATA_PATH)
    p_collect.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_collect.add_argument("--cleanup", action="store_true")

    p_train = sub.add_parser(CMD_TRAIN)
    p_train.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    p_train.add_argument("--run-dir", type=str, default=DEFAULT_RUN_DIR)
    p_train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p_train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_train.add_argument("--lr", type=float, default=DEFAULT_LR)
    p_train.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p_train.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    p_train.add_argument("--huber-beta", type=float, default=DEFAULT_HUBER_BETA)
    p_train.add_argument("--lr-sched-factor", type=float, default=DEFAULT_LR_SCHED_FACTOR)
    p_train.add_argument("--lr-sched-patience", type=int, default=DEFAULT_LR_SCHED_PATIENCE)
    p_train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_train.add_argument("--device", type=str, default=DEFAULT_DEVICE)

    p_test = sub.add_parser(CMD_TEST)
    p_test.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    p_test.add_argument("--checkpoint-path", type=str, default=f"{DEFAULT_RUN_DIR}/best.pt")
    p_test.add_argument("--run-dir", type=str, default=DEFAULT_RUN_DIR)
    p_test.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_test.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_test.add_argument("--device", type=str, default=DEFAULT_DEVICE)

    args = parser.parse_args()
    if args.command is None:
        args.command = DEFAULT_COMMAND

    if args.command == CMD_COLLECT:
        collect(
            num_samples=getattr(args, "num_samples", DEFAULT_NUM_SAMPLES),
            workers=getattr(args, "workers", DEFAULT_WORKERS),
            out_dir=getattr(args, "out_dir", DEFAULT_DATA_PATH),
            seed=getattr(args, "seed", DEFAULT_SEED),
            cleanup=getattr(args, "cleanup", False),
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
            huber_beta=getattr(args, "huber_beta", DEFAULT_HUBER_BETA),
            lr_sched_factor=getattr(args, "lr_sched_factor", DEFAULT_LR_SCHED_FACTOR),
            lr_sched_patience=getattr(args, "lr_sched_patience", DEFAULT_LR_SCHED_PATIENCE),
            seed=getattr(args, "seed", DEFAULT_SEED),
            device=getattr(args, "device", DEFAULT_DEVICE),
        )
    else:
        test(
            data_path=getattr(args, "data_path", DEFAULT_DATA_PATH),
            checkpoint_path=getattr(args, "checkpoint_path", f"{DEFAULT_RUN_DIR}/best.pt"),
            run_dir=getattr(args, "run_dir", DEFAULT_RUN_DIR),
            batch_size=getattr(args, "batch_size", DEFAULT_BATCH_SIZE),
            seed=getattr(args, "seed", DEFAULT_SEED),
            device=getattr(args, "device", DEFAULT_DEVICE),
        )


if __name__ == "__main__":
    main()
