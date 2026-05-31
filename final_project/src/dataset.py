"""
Dataset for behaviour cloning.

TrajectoryDataset
  - Loads saved trajectory *.pt files
  - Extracts sliding windows: (gt_bbox[t], ee_pos[t]) → ee_pos[t+1..t+H]
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from utils import bbox_to_input, normalise_ee   # noqa: E402

HORIZON = 5


def _load_traj_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("traj_*.pt"))


class TrajectoryDataset(Dataset):
    """
    Each item is one sliding window:
      x : (7,)         float32  – [gt_bbox(4), ee_norm(3)]
      y : (HORIZON*3,) float32  – next HORIZON normalised EE positions, flattened
    """

    def __init__(
        self,
        data_dir: str,
        horizon: int = HORIZON,
    ) -> None:
        self.horizon = horizon
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []

        data_path = Path(data_dir)
        paths = _load_traj_paths(data_path)
        if not paths:
            raise FileNotFoundError(f"No traj_*.pt files in {data_path}")

        for path in paths:
            traj = torch.load(path, map_location="cpu")
            if not traj.get("success", False):
                continue

            ee_positions = traj["ee_positions"].numpy()   # (T, 3)
            bboxes       = traj["gt_bboxes"].numpy()      # (T, 4)
            T            = ee_positions.shape[0]

            if T < horizon + 1:
                continue

            for t in range(T - horizon):
                x = bbox_to_input(bboxes[t], ee_positions[t])
                future = ee_positions[t + 1 : t + 1 + horizon]
                y = np.stack([normalise_ee(p) for p in future]).astype(np.float32).flatten()
                self.samples.append((x, y))

        if not self.samples:
            raise RuntimeError(
                "No training samples extracted. "
                "Check that trajectories were saved with success=True "
                "and have at least horizon+1 steps."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def build_split_loaders(
    data_dir:   str,
    batch_size: int   = 64,
    val_ratio:  float = 0.1,
    test_ratio: float = 0.1,
    seed:       int   = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load dataset, split, return (train, val, test) DataLoaders."""
    ds = TrajectoryDataset(data_dir)

    n       = len(ds)
    n_val   = max(1, int(n * val_ratio))
    n_test  = max(1, int(n * test_ratio))
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test], generator=generator
    )
    print(f"[dataset] samples: train={n_train}, val={n_val}, test={n_test}")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0),
    )
