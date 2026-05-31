"""
Training: behaviour cloning with MSE loss on normalised EE delta actions.

Usage
-----
  python final_project/src/train.py \\
    --data-dir  final_project/data/trajectories \\
    --run-dir   final_project/runs/navigation
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from model   import NavigationMLP  # noqa: E402
from utils   import bbox_to_input                # noqa: E402

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR    = "final_project/data/trajectories"
DEFAULT_RUN_DIR     = "final_project/runs/navigation"
DEFAULT_EPOCHS      = 200
DEFAULT_BATCH_SIZE  = 128
DEFAULT_LR          = 3e-4
DEFAULT_WEIGHT_DECAY= 1e-5
DEFAULT_GRAD_CLIP   = 1.0
DEFAULT_WARMUP      = 10
DEFAULT_VAL_RATIO   = 0.1
DEFAULT_TEST_RATIO  = 0.1
DEFAULT_SEED        = 42
DEFAULT_DEVICE      = "auto"
DEFAULT_HORIZONS    = [1]


def _load_traj_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("traj_*.pt"))


def _target_stratum(target_name: str) -> str:
    """Map 'red_sphere_3' -> 'red_sphere' for stratified trajectory splits."""
    parts = target_name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else target_name


class TrajectoryDataset(Dataset):
    """
    Sliding multi-step samples:
      x: (7,) float32 = [fixed initial bbox(4), ee_norm(3)]
      y: (3*HORIZON,) float32 = raw EE deltas in metres
    """

    def __init__(
        self,
        data_dir: str,
        traj_paths: Optional[list[Path]] = None,
        horizon: int = 1,
    ) -> None:
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []

        data_path = Path(data_dir)
        paths = traj_paths if traj_paths is not None else _load_traj_paths(data_path)
        if not paths:
            raise FileNotFoundError(f"No trajectory files selected from {data_path}")

        for path in paths:
            traj = torch.load(path, map_location="cpu")
            if not traj.get("success", False):
                continue

            ee_positions = traj["ee_positions"].numpy()
            gt_bboxes = traj["gt_bboxes"].numpy()
            if ee_positions.shape[0] < horizon + 1:
                continue

            fixed_bbox = gt_bboxes[0]

            for t in range(ee_positions.shape[0] - horizon):
                x = bbox_to_input(fixed_bbox, ee_positions[t])
                # Extract horizon deltas and stack
                deltas = []
                for h in range(horizon):
                    delta = (ee_positions[t + h + 1] - ee_positions[t + h]).astype(np.float32)
                    deltas.append(delta)
                y = np.concatenate(deltas, axis=0).astype(np.float32)
                self.samples.append((x, y))

        if not self.samples:
            raise RuntimeError(
                f"No training samples extracted. Check that trajectories were "
                f"saved with success=True and have at least {horizon + 1} steps."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def _successful_paths_by_stratum(data_path: Path) -> dict[str, list[Path]]:
    """Load successful trajectory paths grouped by target color+shape."""
    groups: dict[str, list[Path]] = {}
    for path in _load_traj_paths(data_path):
        traj = torch.load(path, map_location="cpu")
        if not traj.get("success", False):
            continue
        key = _target_stratum(str(traj["target_name"]))
        groups.setdefault(key, []).append(path)
    return {k: sorted(v) for k, v in sorted(groups.items())}


def _split_one_group(
    paths: list[Path],
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split one stratum while keeping at least one train trajectory when possible."""
    shuffled = list(paths)
    rng.shuffle(shuffled)
    n = len(shuffled)

    if n <= 1:
        return shuffled, [], []
    if n == 2:
        if test_ratio > val_ratio:
            return shuffled[:1], [], shuffled[1:]
        return shuffled[:1], shuffled[1:], []

    n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
    n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 else 0

    while n - n_val - n_test < 1:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break

    val = shuffled[:n_val]
    test = shuffled[n_val:n_val + n_test]
    train = shuffled[n_val + n_test:]
    return train, val, test


def stratified_trajectory_split(
    data_dir: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path], dict[str, dict[str, int]]]:
    """
    Split successful trajectories by target color+shape.

    Samples from the same trajectory never cross train/val/test boundaries.
    """
    data_path = Path(data_dir)
    groups = _successful_paths_by_stratum(data_path)
    if not groups:
        raise RuntimeError(f"No successful trajectories found in {data_path}")

    rng = random.Random(seed)
    train_paths: list[Path] = []
    val_paths: list[Path] = []
    test_paths: list[Path] = []
    counts: dict[str, dict[str, int]] = {}

    for key, paths in groups.items():
        tr, va, te = _split_one_group(paths, val_ratio, test_ratio, rng)
        train_paths.extend(tr)
        val_paths.extend(va)
        test_paths.extend(te)
        counts[key] = {"train": len(tr), "val": len(va), "test": len(te), "total": len(paths)}

    # Very small datasets can have all strata with a single trajectory. Keep the
    # loaders usable while still preventing sample leakage between splits.
    if not val_paths and len(train_paths) > 1:
        val_paths.append(train_paths.pop())
    if not test_paths and len(train_paths) > 1:
        test_paths.append(train_paths.pop())

    return sorted(train_paths), sorted(val_paths), sorted(test_paths), counts


def build_split_loaders(
    data_dir: str,
    batch_size: int = 64,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    horizon: int = 1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load dataset, split, return (train, val, test) DataLoaders."""
    print("[dataset] Using initial ground-truth bboxes.")

    train_paths, val_paths, test_paths, split_counts = stratified_trajectory_split(
        data_dir=data_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    train_ds = TrajectoryDataset(
        data_dir, traj_paths=train_paths, horizon=horizon
    )
    val_ds = TrajectoryDataset(
        data_dir, traj_paths=val_paths, horizon=horizon
    )
    test_ds = TrajectoryDataset(
        data_dir, traj_paths=test_paths, horizon=horizon
    )

    print(
        "[dataset] trajectories: "
        f"train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}"
    )
    print(f"[dataset] samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"[dataset] strata: {len(split_counts)} color/shape groups")
    print(f"[dataset] horizon: {horizon}")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size,   shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=batch_size,  shuffle=False, num_workers=0),
    )


def resolve_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    total_mse = total_l1 = total_n = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            b    = x.size(0)
            total_mse += F.mse_loss(pred, y, reduction="sum").item()
            total_l1  += F.l1_loss(pred, y, reduction="sum").item()
            total_n   += b
    denom = max(total_n * y.shape[-1], 1)
    mse   = total_mse / denom
    return {"mse": mse, "mae": total_l1 / denom, "rmse": math.sqrt(mse)}


def save_loss_plot(history: list[dict], out_dir: Path) -> None:
    epochs     = [h["epoch"] for h in history]
    train_loss = [float(h["train_loss"] if "train_loss" in h else h["train_mse"]) for h in history]
    val_mse    = [h["val_mse"]   for h in history]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, np.maximum(train_loss, 1e-12), label="train_mse", linewidth=1.5)
    plt.plot(epochs, np.maximum(val_mse, 1e-12),    label="val_mse",   linewidth=1.5)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log scale)")
    plt.title("Navigation MLP – Train / Val Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_plot.png", dpi=150)
    plt.close()


def train(
    data_dir:     str   = DEFAULT_DATA_DIR,
    run_dir:      str   = DEFAULT_RUN_DIR,
    epochs:       int   = DEFAULT_EPOCHS,
    batch_size:   int   = DEFAULT_BATCH_SIZE,
    lr:           float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    grad_clip:    float = DEFAULT_GRAD_CLIP,
    warmup_epochs:int   = DEFAULT_WARMUP,
    val_ratio:    float = DEFAULT_VAL_RATIO,
    test_ratio:   float = DEFAULT_TEST_RATIO,
    seed:         int   = DEFAULT_SEED,
    device:       str   = DEFAULT_DEVICE,
    horizon:      int   = 1,
) -> dict:
    set_seeds(seed)
    dev     = resolve_device(device)
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_split_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        horizon=horizon,
    )

    model     = NavigationMLP(horizon=horizon).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = nn.MSELoss()

    # Warmup + cosine LR schedule
    warmup    = max(0, min(warmup_epochs, epochs - 1))
    cosine_ep = max(1, epochs - warmup)
    min_lr    = lr * 0.05
    if warmup > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, 0.2, 1.0, warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_ep, min_lr),
            ],
            milestones=[warmup],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_ep, min_lr)

    best_val  = float("inf")
    history   = []
    print(f"[train] model params: {model.n_parameters():,}  device: {dev}")

    for epoch in tqdm(range(1, epochs + 1), desc="epochs"):
        model.train()
        tr_sum = tr_n = 0

        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            pred  = model(x)
            loss  = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            tr_sum += loss.item() * x.size(0)
            tr_n   += x.size(0)

        train_loss = tr_sum / max(tr_n, 1)
        val_met   = evaluate(model, val_loader, dev)
        curr_lr   = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch":     epoch,
            "lr":        curr_lr,
            "train_loss": train_loss,
            "val_mse":   val_met["mse"],
        })

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"[train] epoch={epoch:4d}/{epochs}  "
                f"train_loss={train_loss:.6f}  val_mse={val_met['mse']:.6f}  "
                f"lr={curr_lr:.2e}"
            )

        if val_met["mse"] < best_val:
            best_val = val_met["mse"]
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "horizon": horizon,
                "policy_type": "residual_delta_mlp",
            }, out_dir / "best.pt")

        scheduler.step()

    # ── final test evaluation ─────────────────────────────────────────────────
    ckpt = torch.load(out_dir / "best.pt", map_location=dev)
    model.load_state_dict(ckpt["model_state"])
    test_met = evaluate(model, test_loader, dev)
    print(f"\n[train] Best val MSE = {best_val:.6f}  (epoch {ckpt['epoch']})")
    print(f"[train] Test  MSE = {test_met['mse']:.6f}  RMSE = {test_met['rmse']:.6f}")

    # ── save artefacts ────────────────────────────────────────────────────────
    metrics = {
        "best_val_mse": best_val,
        "best_epoch":   int(ckpt["epoch"]),
        "test_mse":     test_met["mse"],
        "test_mae":     test_met["mae"],
        "test_rmse":    test_met["rmse"],
        "policy_type":  "residual_delta_mlp",
        "horizon":      horizon,
        "target":       "delta_ee_meters",
        "loss":         "MSELoss",
        "history":      history,
    }
    with open(out_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    save_loss_plot(history, out_dir)
    print(f"[train] Artefacts saved to {out_dir}")
    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Train navigation MLP.")
    p.add_argument("--data-dir",       type=str,            default=DEFAULT_DATA_DIR)
    p.add_argument("--run-dir",        type=str,            default=DEFAULT_RUN_DIR)
    p.add_argument("--epochs",         type=int,            default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size",     type=int,            default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr",             type=float,          default=DEFAULT_LR)
    p.add_argument("--weight-decay",   type=float,          default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--grad-clip",      type=float,          default=DEFAULT_GRAD_CLIP)
    p.add_argument("--warmup-epochs",  type=int,            default=DEFAULT_WARMUP)
    p.add_argument("--val-ratio",      type=float,          default=DEFAULT_VAL_RATIO)
    p.add_argument("--test-ratio",     type=float,          default=DEFAULT_TEST_RATIO)
    p.add_argument("--seed",           type=int,            default=DEFAULT_SEED)
    p.add_argument("--device",         type=str,            default=DEFAULT_DEVICE)
    p.add_argument(
        "--horizon",
        type=int,
        nargs="+",
        default=DEFAULT_HORIZONS,
        help="Planning horizon(s), e.g. --horizon 1 or --horizon 1 2 3 4 5",
    )
    args = p.parse_args()

    horizons = args.horizon

    if len(horizons) > 1:
        print(f"\n{'='*70}")
        print(f"Training multiple horizons: {horizons}")
        print(f"{'='*70}\n")
        results = {}

        for h in horizons:
            run_dir = f"{args.run_dir.replace('navigation', f'nav_h{h}')}"
            print(f"\n{'='*70}")
            print(f"Training HORIZON={h}")
            print(f"{'='*70}\n")

            train_kwargs = {
                "data_dir": args.data_dir,
                "run_dir": run_dir,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "warmup_epochs": args.warmup_epochs,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "seed": args.seed,
                "device": args.device,
                "horizon": h,
            }
            metrics = train(**train_kwargs)
            results[h] = {
                "status": "success",
                "checkpoint": f"{run_dir}/best.pt",
                "test_mse": metrics.get("test_mse"),
                "test_rmse": metrics.get("test_rmse"),
            }

        # Summary
        print(f"\n{'='*70}")
        print("TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"{'Horizon':<10} {'Test MSE':<15} {'Test RMSE':<15}")
        print("-" * 40)
        for h in sorted(results.keys()):
            res = results[h]
            if res["status"] == "success":
                print(f"{h:<10} {res['test_mse']:<15.6f} {res['test_rmse']:<15.6f}")

        # Save summary
        summary_path = Path(args.run_dir).parent / "train_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary saved to {summary_path}\n")
    else:
        # Single horizon
        train_kwargs = vars(args)
        train_kwargs["horizon"] = horizons[0]
        train(**train_kwargs)


if __name__ == "__main__":
    main()
