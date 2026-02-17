import argparse
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Homework 1 dataset collector.")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="data/hw1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cleanup", action="store_true", help="Delete shard files after merging.")
    args = parser.parse_args()

    set_seeds(args.seed)
    merged_path = collect_dataset(
        num_samples=args.num_samples,
        workers=args.workers,
        out_dir=Path(args.out_dir),
        seed=args.seed,
        cleanup=args.cleanup,
    )
    data = torch.load(merged_path, map_location="cpu")
    print(f"Saved dataset to {merged_path}")
    print(f"num_samples={data['actions'].shape[0]}")


if __name__ == "__main__":
    main()
