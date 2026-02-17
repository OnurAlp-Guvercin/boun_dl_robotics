import argparse
import json
import math
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


N_ACTIONS = 4
IMG_SIZE = 128
IMG_SHAPE = (3, IMG_SIZE, IMG_SIZE)


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

    env = Hw1Env(render_mode="offscreen")
    imgs_before = torch.zeros((n_samples, *IMG_SHAPE), dtype=torch.uint8)
    actions = torch.zeros((n_samples,), dtype=torch.uint8)
    pos_after = torch.zeros((n_samples, 2), dtype=torch.float32)
    imgs_after = torch.zeros((n_samples, *IMG_SHAPE), dtype=torch.uint8)

    for i in range(n_samples):
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
        procs: List[Process] = []
        for i, n in enumerate(per_worker):
            p = Process(target=_collect_worker, args=(i, n, str(out_dir), seed))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
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


def evaluate(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
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


def train(
    data_path: str = "data/hw1",
    run_dir: str = "runs/hw1/mlp_pos",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "auto",
) -> Dict[str, float]:
    set_seeds(seed)
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)

    dataset = load_hw1_dataset(Path(data_path))
    loaders = build_loaders(dataset, batch_size=batch_size, seed=seed)

    model = PositionMLP().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in loaders.train:
            img_before = batch["img_before"].to(dev)
            action_onehot = batch["action_onehot"].to(dev)
            target = batch["pos_after"].to(dev)
            pred = model(img_before, action_onehot)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * img_before.size(0)
            train_count += img_before.size(0)

        train_loss = train_loss_sum / max(train_count, 1)
        val_metrics = evaluate(model, loaders.val, dev)
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_metrics["mse"]})
        print(
            f"[MLP] epoch={epoch}/{epochs} train_mse={train_loss:.6f} "
            f"val_mse={val_metrics['mse']:.6f} val_rmse={val_metrics['rmse']:.6f}"
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
    )
    return test_metrics


def test(
    data_path: str = "data/hw1",
    checkpoint_path: str = "runs/hw1/mlp_pos/best.pt",
    batch_size: int = 32,
    seed: int = 42,
    device: str = "auto",
    run_dir: str = "runs/hw1/mlp_pos",
    history: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, float]:
    set_seeds(seed)
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)

    dataset = load_hw1_dataset(Path(data_path))
    loaders = build_loaders(dataset, batch_size=batch_size, seed=seed)

    model = PositionMLP().to(dev)
    model.load_state_dict(torch.load(checkpoint_path, map_location=dev))
    metrics = evaluate(model, loaders.test, dev)
    print(f"[MLP] test metrics: {metrics}")

    payload: Dict[str, object] = {"test": metrics}
    if history is not None:
        payload["history"] = history
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return metrics


def collect(
    num_samples: int = 1000,
    workers: int = 4,
    out_dir: str = "data/hw1",
    seed: int = 42,
    cleanup: bool = False,
) -> Path:
    set_seeds(seed)
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
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect")
    p_collect.add_argument("--num-samples", type=int, default=1000)
    p_collect.add_argument("--workers", type=int, default=4)
    p_collect.add_argument("--out-dir", type=str, default="data/hw1")
    p_collect.add_argument("--seed", type=int, default=42)
    p_collect.add_argument("--cleanup", action="store_true")

    p_train = sub.add_parser("train")
    p_train.add_argument("--data-path", type=str, default="data/hw1")
    p_train.add_argument("--run-dir", type=str, default="runs/hw1/mlp_pos")
    p_train.add_argument("--epochs", type=int, default=25)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--device", type=str, default="auto")

    p_test = sub.add_parser("test")
    p_test.add_argument("--data-path", type=str, default="data/hw1")
    p_test.add_argument("--checkpoint-path", type=str, default="runs/hw1/mlp_pos/best.pt")
    p_test.add_argument("--run-dir", type=str, default="runs/hw1/mlp_pos")
    p_test.add_argument("--batch-size", type=int, default=32)
    p_test.add_argument("--seed", type=int, default=42)
    p_test.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    if args.command == "collect":
        collect(
            num_samples=args.num_samples,
            workers=args.workers,
            out_dir=args.out_dir,
            seed=args.seed,
            cleanup=args.cleanup,
        )
    elif args.command == "train":
        train(
            data_path=args.data_path,
            run_dir=args.run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
        )
    else:
        test(
            data_path=args.data_path,
            checkpoint_path=args.checkpoint_path,
            run_dir=args.run_dir,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
