from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
import matplotlib.patches as mpatches  # pyright: ignore[reportMissingModuleSource]
from matplotlib.axes import Axes  # pyright: ignore[reportMissingModuleSource]
from matplotlib.patches import Rectangle  # pyright: ignore[reportMissingModuleSource]
from typing import cast

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

IMG_H = IMG_W = 128


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_trajectories(data_dir: Path, n: Optional[int] = None) -> list[dict]:
    paths = sorted(data_dir.glob("traj_*.pt"))
    if n is not None:
        paths = random.sample(paths, min(n, len(paths)))
    return [torch.load(p, map_location="cpu") for p in paths]


def _draw_bbox(ax, bbox_norm: np.ndarray, color: str = "lime", label: str = "") -> None:
    """Draw a (cx, cy, w, h) normalised bbox on a matplotlib Axes."""
    cx, cy, w, h = bbox_norm * IMG_W  # denormalise to pixels
    x0 = cx - w / 2
    y0 = cy - h / 2
    rect = Rectangle((x0, y0), w, h, linewidth=2, edgecolor=color,
                      facecolor="none", label=label)
    ax.add_patch(rect)
    ax.plot(cx, cy, "+", color=color, markersize=8, linewidth=2)


def _tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    """(3, H, W) uint8 → (H, W, 3) uint8 numpy."""
    return image.permute(1, 2, 0).numpy()


# ── mode: trajectories ────────────────────────────────────────────────────────

def vis_trajectories(
    data_dir:  Path,
    n_samples: int,
    n_frames:  int = 6,
    out_dir:   Optional[Path] = None,
) -> None:
    """Show evenly spaced frames from N random successful trajectories."""
    paths = sorted(data_dir.glob("traj_*.pt"))
    succ  = [p for p in paths
             if torch.load(p, map_location="cpu").get("success", False)]

    if not succ:
        print("[vis] No successful trajectories found.")
        return

    chosen = random.sample(succ, min(n_samples, len(succ)))

    for traj_path in chosen:
        traj   = torch.load(traj_path, map_location="cpu")
        images = traj["images"]             # (T, 3, H, W)
        bboxes = traj["gt_bboxes"].numpy()  # (T, 4)
        T      = images.shape[0]
        idxs   = np.linspace(0, T - 1, min(n_frames, T), dtype=int)

        fig, axes = plt.subplots(1, len(idxs), figsize=(3 * len(idxs), 3.5))
        axes_list = [cast(Axes, ax) for ax in np.atleast_1d(axes).ravel()]

        title_parts = [
            traj["target_name"],
            f"steps={T}",
            "SUCCESS" if traj["success"] else "FAIL",
        ]
        fig.suptitle(" | ".join(title_parts), fontsize=10)

        for ax, t in zip(axes_list, idxs):
            ax.imshow(_tensor_to_rgb(images[t]))
            _draw_bbox(ax, bboxes[t], color="lime")
            ax.set_title(f"step {t}", fontsize=8)
            ax.axis("off")

        plt.tight_layout()
        if out_dir is not None:
            fname = traj_path.stem + "_frames.png"
            plt.savefig(out_dir / fname, dpi=120)
            plt.close()
        else:
            plt.show()

    print(f"[vis] Trajectories plotted ({len(chosen)} trajectories).")


# ── mode: training curves ─────────────────────────────────────────────────────

def vis_training(run_dir: Path, out_dir: Optional[Path] = None) -> None:
    metrics_path = run_dir / "train_metrics.json"
    if not metrics_path.exists():
        print(f"[vis] {metrics_path} not found.")
        return

    with open(metrics_path) as f:
        m = json.load(f)

    history    = m["history"]
    epochs     = [h["epoch"]     for h in history]
    train_loss = [h.get("train_loss", h.get("train_mse")) for h in history]
    val_mse    = [h["val_mse"]   for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, np.maximum(train_loss, 1e-12), label="train", linewidth=1.5)
    axes[0].plot(epochs, np.maximum(val_mse, 1e-12),    label="val",   linewidth=1.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (log scale)")
    axes[0].set_title("Train / Val MSE")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    lrs = [h["lr"] for h in history]
    axes[1].plot(epochs, lrs, color="orange", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("LR Schedule")
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"Best val MSE={m['best_val_mse']:.6f} (ep {m['best_epoch']})  |  "
        f"Test MSE={m['test_mse']:.6f}  RMSE={m['test_rmse']:.6f}"
    )
    plt.tight_layout()

    if out_dir is not None:
        plt.savefig(out_dir / "training_curves.png", dpi=150)
        plt.close()
        print(f"[vis] Saved training curves → {out_dir / 'training_curves.png'}")
    else:
        plt.show()



# ── mode: episode trajectories (from eval JSON) ──────────────────────────────

def vis_episodes(
    eval_json: Path,
    n_samples: int,
    out_dir:   Optional[Path] = None,
) -> None:
    """
    Plot per-episode EE trajectory + predicted waypoints from eval_results.json.
    Shows a 2-D top-down view (x, y) of the workspace.
    """
    with open(eval_json) as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    if not episodes:
        print("[vis] No episodes found in JSON.")
        return

    chosen = random.sample(episodes, min(n_samples, len(episodes)))

    for ep in chosen:
        traj   = ep.get("trajectory", [])
        if not traj:
            continue

        fig, ax = plt.subplots(figsize=(5, 5))
        status  = "SUCCESS" if ep["success"] else "FAIL"
        color   = "green" if ep["success"] else "red"

        ax.set_title(
            f"ep{ep['episode']:03d} | {ep['target_name']} | {status} | "
            f"steps={ep['n_steps']} | dist={ep['final_ee_dist']:.3f}m",
            fontsize=8,
        )

        # Collect all EE positions and predicted poses across blocks
        ee_xs, ee_ys = [], []
        for block in traj:
            ex, ey = block["ee_pos"][0], block["ee_pos"][1]
            ee_xs.append(ex)
            ee_ys.append(ey)

            # Predicted waypoints (denormalised)
            for pose in block.get("predicted_poses", []):
                ax.plot(pose[0], pose[1], ".", color="orange", markersize=4, alpha=0.5)

            # Bbox centre projected to workspace (cx maps to x roughly)
            bbox = block["bbox"]
            ax.plot(bbox[0], bbox[1], "x", color="purple", markersize=6,
                    label="bbox (norm)" if block is traj[0] else "")

        # EE path
        ax.plot(ee_xs, ee_ys, "-o", color=color, markersize=5, linewidth=1.5,
                label="EE path")
        ax.plot(ee_xs[0],  ee_ys[0],  "o", color="blue",  markersize=8, label="start")
        ax.plot(ee_xs[-1], ee_ys[-1], "s", color=color,   markersize=8, label="end")

        ax.set_xlabel("EE x (m)")
        ax.set_ylabel("EE y (m)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if out_dir is not None:
            plt.savefig(out_dir / f"ep{ep['episode']:03d}_traj.png", dpi=120)
            plt.close()
        else:
            plt.show()

    print(f"[vis] Episode trajectories plotted ({len(chosen)} episodes).")


# ── mode: summary stats ───────────────────────────────────────────────────────

def vis_eval_summary(eval_json: Path, out_dir: Optional[Path] = None) -> None:
    with open(eval_json) as f:
        data = json.load(f)
    summ = data["summary"]
    eps  = data["episodes"]

    dists  = [e["final_ee_dist"] for e in eps]
    steps  = [e["n_steps"] for e in eps]
    labels = ["success" if e["success"] else "fail" for e in eps]
    colors = ["green" if l == "success" else "red" for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(len(dists)), dists, color=colors)
    axes[0].axhline(summ["mean_final_dist"], color="black", linestyle="--",
                    label=f"mean={summ['mean_final_dist']:.3f}m")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Final EE-to-object distance (m)")
    axes[0].set_title(f"Final Distance  |  Success={summ['success_rate']*100:.1f}%")
    axes[0].legend(fontsize=8)
    green_patch = mpatches.Patch(color="green", label="success")
    red_patch   = mpatches.Patch(color="red",   label="fail")
    axes[0].legend(handles=[green_patch, red_patch] + axes[0].get_legend_handles_labels()[0])

    axes[1].hist(steps, bins=20, color="steelblue", edgecolor="white")
    axes[1].set_xlabel("Steps taken")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Episode Length Distribution")
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"Success rate: {summ['success_rate']*100:.1f}%  |  "
        f"Mean dist: {summ['mean_final_dist']:.3f}m"
    )
    plt.tight_layout()

    if out_dir is not None:
        plt.savefig(out_dir / "eval_summary.png", dpi=150)
        plt.close()
        print(f"[vis] Saved eval summary → {out_dir / 'eval_summary.png'}")
    else:
        plt.show()


# ── mode: horizon comparison ──────────────────────────────────────────────────

def vis_horizon_comparison(runs_dir: Path, out_dir: Optional[Path] = None) -> None:
    """Compare success rate, mean distance, and mean steps across horizons.

    Looks for vis_h{N}/eval_results.json under runs_dir.
    """
    horizon_dirs = sorted(runs_dir.glob("vis_h*"), key=lambda p: int(p.name[5:]))
    if not horizon_dirs:
        print(f"[vis] No vis_h* directories found under {runs_dir}")
        return

    horizons, success_rates, mean_dists, mean_steps = [], [], [], []

    for hdir in horizon_dirs:
        json_path = hdir / "eval_results.json"
        if not json_path.exists():
            print(f"[vis] Skipping {hdir.name}: eval_results.json not found")
            continue
        with open(json_path) as f:
            data = json.load(f)
        s = data["summary"]
        h = int(hdir.name[5:])
        horizons.append(h)
        success_rates.append(s["success_rate"] * 100)
        mean_dists.append(s["mean_final_dist"])
        succ_steps = s.get("mean_steps_success")
        if succ_steps is None:
            eps = data.get("episodes", [])
            succ_eps = [e["n_steps"] for e in eps if e["success"]]
            succ_steps = float(np.mean(succ_eps)) if succ_eps else 0.0
        mean_steps.append(succ_steps)

    if not horizons:
        print("[vis] No valid horizon data found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Success rate
    axes[0].plot(horizons, success_rates, "o-", color="steelblue", linewidth=2, markersize=7)
    for h, v in zip(horizons, success_rates):
        axes[0].annotate(f"{v:.1f}%", (h, v), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    axes[0].set_xlabel("Horizon")
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Success Rate vs Horizon")
    axes[0].set_xticks(horizons)
    axes[0].set_ylim(0, 105)
    axes[0].grid(alpha=0.3)

    # Mean final distance
    axes[1].plot(horizons, mean_dists, "o-", color="tomato", linewidth=2, markersize=7)
    for h, v in zip(horizons, mean_dists):
        axes[1].annotate(f"{v:.3f}", (h, v), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    axes[1].set_xlabel("Horizon")
    axes[1].set_ylabel("Mean Final Distance (m)")
    axes[1].set_title("Mean Final EE Distance vs Horizon")
    axes[1].set_xticks(horizons)
    axes[1].grid(alpha=0.3)

    # Mean steps (success only)
    axes[2].plot(horizons, mean_steps, "o-", color="seagreen", linewidth=2, markersize=7)
    for h, v in zip(horizons, mean_steps):
        axes[2].annotate(f"{v:.1f}", (h, v), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    axes[2].set_xlabel("Horizon")
    axes[2].set_ylabel("Mean Steps (success only)")
    axes[2].set_title("Mean Steps vs Horizon")
    axes[2].set_xticks(horizons)
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Horizon Comparison  |  horizons={horizons}", fontsize=12)
    plt.tight_layout()

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "horizon_comparison.png", dpi=150)
        plt.close()
        print(f"[vis] Saved horizon comparison → {out_dir / 'horizon_comparison.png'}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Visualisation tool.")
    p.add_argument("--mode", type=str, required=True,
                   choices=["trajectories", "training", "eval-summary", "episodes",
                            "horizon-compare"])
    p.add_argument("--data-dir",   type=str, default="final_project/data/trajectories")
    p.add_argument("--run-dir",    type=str, default="final_project/runs/navigation")
    p.add_argument("--eval-json",  type=str, default="")
    p.add_argument("--n-samples",  type=int, default=5)
    p.add_argument("--n-frames",   type=int, default=6)
    p.add_argument("--out-dir",    type=str, default="",
                   help="Save figures here instead of plt.show()")
    p.add_argument("--seed",       type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out_dir) if args.out_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    if args.mode == "trajectories":
        vis_trajectories(Path(args.data_dir), args.n_samples, args.n_frames, out)

    elif args.mode == "training":
        vis_training(Path(args.run_dir), out)

    elif args.mode == "eval-summary":
        if not args.eval_json:
            p.error("--eval-json required for eval-summary mode")
        vis_eval_summary(Path(args.eval_json), out)

    elif args.mode == "episodes":
        if not args.eval_json:
            p.error("--eval-json required for episodes mode")
        vis_episodes(Path(args.eval_json), args.n_samples, out)

    elif args.mode == "horizon-compare":
        vis_horizon_comparison(Path(args.run_dir), out)


if __name__ == "__main__":
    main()
