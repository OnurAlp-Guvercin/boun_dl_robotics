"""Generate comparison charts (GT bbox vs VLM bbox) for the presentation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 3, 4, 5]


def load(kind: str):
    sr, dist, steps = [], [], []
    for h in HORIZONS:
        with open(RUNS / kind / f"vis_h{h}" / "eval_results.json") as f:
            s = json.load(f)["summary"]
        sr.append(s["success_rate"] * 100)
        dist.append(s["mean_final_dist"])
        steps.append(s["mean_steps_success"])
    return sr, dist, steps


gt_sr, gt_dist, gt_steps = load("gt_bbox")
vlm_sr, vlm_dist, vlm_steps = load("vlm_bbox")

GT_C = "#2a7de1"
VLM_C = "#e8590c"

# ---- Chart 1: success rate + mean final distance side by side ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

ax = axes[0]
ax.plot(HORIZONS, gt_sr, "o-", color=GT_C, lw=2.5, ms=9, label="GT bbox (oracle)")
ax.plot(HORIZONS, vlm_sr, "s--", color=VLM_C, lw=2.5, ms=9, label="VLM bbox (Qwen3-VL)")
for h, v in zip(HORIZONS, gt_sr):
    ax.annotate(f"{v:.0f}%", (h, v), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color=GT_C, fontweight="bold")
for h, v in zip(HORIZONS, vlm_sr):
    ax.annotate(f"{v:.0f}%", (h, v), textcoords="offset points", xytext=(0, -16),
                ha="center", fontsize=9, color=VLM_C, fontweight="bold")
ax.set_xlabel("Planning horizon H", fontsize=11)
ax.set_ylabel("Success rate (%)", fontsize=11)
ax.set_title("Reaching success rate vs horizon", fontsize=12, fontweight="bold")
ax.set_xticks(HORIZONS)
ax.set_ylim(55, 105)
ax.grid(alpha=0.3)
ax.legend(fontsize=10, loc="lower left")

ax = axes[1]
ax.plot(HORIZONS, gt_dist, "o-", color=GT_C, lw=2.5, ms=9, label="GT bbox (oracle)")
ax.plot(HORIZONS, vlm_dist, "s--", color=VLM_C, lw=2.5, ms=9, label="VLM bbox (Qwen3-VL)")
ax.set_xlabel("Planning horizon H", fontsize=11)
ax.set_ylabel("Mean final EE–object distance (m)", fontsize=11)
ax.set_title("Final distance vs horizon (lower is better)", fontsize=12, fontweight="bold")
ax.set_xticks(HORIZONS)
ax.grid(alpha=0.3)
ax.legend(fontsize=10, loc="upper left")

fig.suptitle("Ground-truth bbox vs VLM-predicted bbox", fontsize=14, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig(OUT / "gt_vs_vlm.png", dpi=160)
plt.close()

# ---- Chart 2: grouped bar of success rate (clean headline figure) ------------
fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(HORIZONS))
w = 0.38
b1 = ax.bar(x - w / 2, gt_sr, w, color=GT_C, label="GT bbox (oracle)")
b2 = ax.bar(x + w / 2, vlm_sr, w, color=VLM_C, label="VLM bbox (Qwen3-VL)")
ax.bar_label(b1, fmt="%.0f%%", fontsize=9, padding=2)
ax.bar_label(b2, fmt="%.0f%%", fontsize=9, padding=2)
ax.set_xticks(x)
ax.set_xticklabels([f"H={h}" for h in HORIZONS])
ax.set_ylabel("Success rate (%)", fontsize=11)
ax.set_ylim(0, 112)
ax.set_title("Success rate: GT bbox vs VLM bbox across horizons",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(OUT / "success_bars.png", dpi=160)
plt.close()

print("charts written to", OUT)
print("GT  sr:", [round(v) for v in gt_sr])
print("VLM sr:", [round(v) for v in vlm_sr])
