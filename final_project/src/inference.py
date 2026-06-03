from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import matplotlib
# Use non-interactive backend to avoid Tkinter GUI calls from worker threads
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from env     import MultiObjectEnv           # noqa: E402
from model   import NavigationMLP  # noqa: E402
from utils   import compute_gt_bbox, bbox_to_input  # noqa: E402

DEFAULT_CHECKPOINT  = "final_project/runs/navigation/best.pt"
DEFAULT_N_EPISODES  = 50
DEFAULT_MAX_STEPS   = 150
DEFAULT_VLM_URL     = "http://localhost:8000"
DEFAULT_VLM_MODEL   = "Qwen3"
DEFAULT_N_OBJ_MIN   = 2
DEFAULT_N_OBJ_MAX   = 4
DEFAULT_SEED        = 100
DEFAULT_RENDER_MODE = "offscreen"
DEFAULT_DEVICE      = "auto"
DEFAULT_HORIZONS    = [1]
DEFAULT_MAX_DELTA   = 0.05


def _save_episode_frames(
    ep_result: dict,
    vis_dir:   Path,
    n_frames:  int = 10,
    n_cols:    int = 5,
) -> None:
    """Save evenly-sampled episode frames as PNG grid with VLM bbox overlay."""
    traj   = ep_result["trajectory"]
    ep_num = ep_result["episode"]
    target = ep_result["target_name"]

    # Flatten all step images across policy steps.
    all_frames = []   # (image, bbox, step_idx, is_vlm_query, action_idx)
    step_idx   = 0
    for action_idx, block in enumerate(traj):
        bbox      = np.array(block["bbox"], dtype=np.float32)
        step_imgs = block.get("step_images", [])
        for i, img in enumerate(step_imgs):
            is_vlm = (action_idx == 0 and i == 0)
            all_frames.append((img, bbox, step_idx, is_vlm, action_idx))
            step_idx += 1

    if not all_frames:
        return

    # Evenly sample n_frames from the full episode
    total   = len(all_frames)
    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    frames  = [all_frames[i] for i in indices]

    n    = len(frames)
    cols = min(n, n_cols)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(rows, cols)

    # Nesne adını renkten ve şekilden ayır: "red_box_0" → "Red Box"
    parts       = target.split("_")
    pretty_name = " ".join(p.capitalize() for p in parts[:-1])  # son indeksi at
    status      = "SUCCESS" if ep_result["success"] else "FAIL"
    color_title = "green" if ep_result["success"] else "red"

    fig.suptitle(
        f"Hedef: {pretty_name}  |  {status}  |  "
        f"{ep_result['n_steps']} adım  |  "
        f"Son mesafe: {ep_result['final_ee_dist']:.3f}m",
        fontsize=10,
        color=color_title,
        fontweight="bold",
    )

    for idx, (img, bbox, sidx, is_query, action_idx) in enumerate(frames):
        r, c = divmod(idx, cols)
        ax   = axes[r, c]

        # Ensure image is torch.Tensor and on CPU
        if isinstance(img, np.ndarray):
            img_np = img
        else:
            img_np = img.cpu().numpy() if isinstance(img, torch.Tensor) else np.array(img)

        # Handle format: (C, H, W) → (H, W, C)
        if img_np.ndim == 3 and img_np.shape[0] in (3, 4):
            img_np = img_np.transpose(1, 2, 0)

        ax.imshow(img_np)
        cx, cy, w, h = bbox * 128
        color = "yellow" if is_query else "red"
        rect  = Rectangle((cx - w/2, cy - h/2), w, h,
                           linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.plot(cx, cy, "+", color=color, markersize=6)
        ax.set_title(f"adım {action_idx}", fontsize=7)
        ax.axis("off")

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    fname = f"ep{ep_num:03d}_{target}_frames.png"
    plt.savefig(vis_dir / fname, dpi=120)
    plt.close()


def resolve_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_for_horizon(checkpoint: str, horizon: int) -> str:
    """Resolve the checkpoint path for a horizon sweep."""
    ckpt_path = Path(checkpoint)
    if ckpt_path.parent.name.startswith("nav_h"):
        return str(ckpt_path.parent.parent / f"nav_h{horizon}" / ckpt_path.name)
    if ckpt_path.parent.name == "navigation":
        return str(ckpt_path.parent.parent / f"nav_h{horizon}" / ckpt_path.name)
    return checkpoint


# -- single episode ------------------------------------------------------------

def run_episode(
    env:          MultiObjectEnv,
    model:        NavigationMLP,
    target_name:  str,
    device:       torch.device,
    max_steps:    int,
    vlm_client    = None,
    use_gt_bbox:  bool = False,
    horizon:      int = 1,
    max_delta:    float = DEFAULT_MAX_DELTA,
) -> dict:
    """
    Run one reaching episode with HORIZON-step predictions.

    Returns
    -------
    dict with keys: success, n_steps, final_ee_dist, trajectory
    """
    model.eval()

    steps_done = 0
    success    = False
    trajectory = []

    ee_pos, _, image = env.state()
    if use_gt_bbox or vlm_client is None:
        fixed_bbox = compute_gt_bbox(env.model, env.data, target_name)
        bbox_source = "gt"
    else:
        fixed_bbox = vlm_client.get_bbox(image, target_name)
        bbox_source = "vlm"
        if fixed_bbox is None:
            fixed_bbox = compute_gt_bbox(env.model, env.data, target_name)
            bbox_source = "gt_fallback"

    while steps_done < max_steps:
        ee_pos, _, image = env.state()

        # -- predict HORIZON deltas ---------------------------------------------
        x_np  = bbox_to_input(fixed_bbox, ee_pos)
        x_t   = torch.from_numpy(x_np).unsqueeze(0).to(device)

        with torch.no_grad():
            delta_flat = model.predict_delta(x_t).squeeze(0).cpu().numpy()  # (3*HORIZON,)

        # Reshape to (HORIZON, 3)
        deltas = delta_flat.reshape(horizon, 3)

        # Apply HORIZON deltas sequentially
        predicted_poses = []
        step_images = [image]

        for h in range(horizon):
            if steps_done >= max_steps:
                break

            predicted_delta = deltas[h]
            if max_delta > 0:
                predicted_delta = np.clip(predicted_delta, -max_delta, max_delta)
            predicted_pose = ee_pos + predicted_delta

            env.move_ee_to_pose(predicted_pose)
            steps_done += 1
            ee_pos, _, step_img = env.state()
            step_images.append(step_img)
            predicted_poses.append(predicted_pose.tolist())

            # Check contact OR distance-based success
            # If contact is registered right now or occurred during intermediate
            # mujoco steps inside move_ee_to_pose, count it as success.
            if env.check_contact(target_name) or env.recent_contact_with_body(target_name):
                success = True
                break
            else:
                obj_pos = env.get_object_pos(target_name)
                dist_to_obj = float(np.linalg.norm(ee_pos[:2] - obj_pos[:2]))
                if dist_to_obj < 0.05:
                    success = True
                    break

        traj_step = {
            "ee_pos":               ee_pos.tolist(),
            "bbox":                 fixed_bbox.tolist(),
            "bbox_source":          bbox_source,
            "predicted_delta":      np.clip(deltas[0], -max_delta, max_delta).tolist()
                                      if max_delta > 0 else deltas[0].tolist(),
            "predicted_poses":      predicted_poses,
            "step_images":          step_images,
            "n_steps_this_block":   len(predicted_poses),
        }
        trajectory.append(traj_step)

        if success or steps_done >= max_steps:
            break

    # Final distance (EE to object centre)
    final_ee  = env.data.site(env._ee_site).xpos.copy()
    final_obj = env.get_object_pos(target_name)
    dist      = float(np.linalg.norm(final_ee[:2] - final_obj[:2]))

    return {
        "success":       success,
        "n_steps":       steps_done,
        "final_ee_dist": dist,
        "trajectory":    trajectory,
    }


# -- full evaluation -----------------------------------------------------------

def evaluate(
    checkpoint:  str,
    n_episodes:  int   = DEFAULT_N_EPISODES,
    max_steps:   int   = DEFAULT_MAX_STEPS,
    vlm_url:     str   = DEFAULT_VLM_URL,
    vlm_model:   str   = DEFAULT_VLM_MODEL,
    use_gt_bbox: bool  = False,
    n_obj_min:   int   = DEFAULT_N_OBJ_MIN,
    n_obj_max:   int   = DEFAULT_N_OBJ_MAX,
    seed:        int   = DEFAULT_SEED,
    render_mode: str   = DEFAULT_RENDER_MODE,
    device:      str   = DEFAULT_DEVICE,
    out_dir:     str   = "",
    save_vis:    bool  = False,
    horizon:     int   = 1,
    max_delta:   float = DEFAULT_MAX_DELTA,
) -> dict:
    dev = resolve_device(device)

    # -- load model ------------------------------------------------------------
    ckpt  = torch.load(checkpoint, map_location=dev)
    # Load horizon from checkpoint if available, otherwise use parameter
    horizon = ckpt.get("horizon", horizon)
    model = NavigationMLP(horizon=horizon).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[eval] Loaded checkpoint: {checkpoint}  (epoch {ckpt.get('epoch','?')}, horizon={horizon})")

    # -- VLM availability check ------------------------------------------------
    use_vlm = False
    if not use_gt_bbox:
        try:
            from vlm_client import VLMClient   # noqa: E402
            probe_client = VLMClient(base_url=vlm_url, model_name=vlm_model)
            if probe_client.is_available():
                print(f"[eval] VLM server available at {vlm_url}")
                use_vlm = True
            else:
                print("[eval] VLM server not reachable → falling back to GT bboxes")
        except Exception as e:
            print(f"[eval] VLM init failed ({e}) → falling back to GT bboxes")

    out_path = Path(out_dir) / "eval_results.json" if out_dir else None
    vis_dir = None
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        if save_vis:
            vis_dir = Path(out_dir) / "vis_episodes"
            vis_dir.mkdir(exist_ok=True)

    def _run_episode_worker(ep: int) -> dict:
        """Worker function for parallel episode execution."""
        env = MultiObjectEnv(
            n_objects_range=(n_obj_min, n_obj_max),
            seed=seed + ep * 997,
            render_mode=render_mode,
        )
        env.reset()

        obj_names = env.get_object_names()
        rng = np.random.default_rng(seed + ep * 997)
        target = obj_names[int(rng.integers(len(obj_names)))]

        vlm_client = None
        if use_vlm:
            from vlm_client import VLMClient  # noqa: E402
            vlm_client = VLMClient(base_url=vlm_url, model_name=vlm_model)

        ep_result = run_episode(
            env=env,
            model=model,
            target_name=target,
            device=dev,
            max_steps=max_steps,
            vlm_client=vlm_client,
            use_gt_bbox=(not use_vlm) or use_gt_bbox,
            horizon=horizon,
            max_delta=max_delta,
        )
        ep_result["episode"] = ep
        ep_result["target_name"] = target

        # NOTE: do NOT call _save_episode_frames here — matplotlib is not
        # thread-safe and will raise Done/renderer errors under ThreadPoolExecutor.
        # Visualisation is handled in the main thread after fut.result().

        return ep_result

    results = []
    n_success = 0

    # Run episodes in parallel with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_run_episode_worker, ep): ep for ep in range(n_episodes)}

        for fut in tqdm(as_completed(futures), total=n_episodes, desc="episodes"):
            ep_result = fut.result()

            # Save visualisation in the main thread (matplotlib is not thread-safe)
            if vis_dir is not None:
                _save_episode_frames(ep_result, vis_dir)

            for step_data in ep_result["trajectory"]:
                step_data.pop("step_images", None)

            results.append(ep_result)
            if ep_result["success"]:
                n_success += 1

            # -- save after every episode --------------------------------------
            if out_path:
                all_dists  = [r["final_ee_dist"] for r in results]
                succ_steps = [r["n_steps"] for r in results if r["success"]]
                partial_summary = {
                    "n_episodes":         len(results),
                    "n_success":          n_success,
                    "success_rate":       n_success / len(results),
                    "mean_final_dist":    float(np.mean(all_dists)),
                    "std_final_dist":     float(np.std(all_dists)),
                    "mean_steps_success": float(np.mean(succ_steps)) if succ_steps else None,
                    "vlm_used":           use_vlm and (not use_gt_bbox),
                    "max_delta":          max_delta,
                }
                with open(out_path, "w") as f:
                    json.dump({"summary": partial_summary, "episodes": results}, f, indent=2)

    # Sort results by episode number
    results.sort(key=lambda r: r["episode"])

    # -- final aggregate metrics -----------------------------------------------
    success_rate = n_success / max(n_episodes, 1)
    all_dists    = [r["final_ee_dist"] for r in results]
    succ_steps   = [r["n_steps"] for r in results if r["success"]]

    summary = {
        "n_episodes":         n_episodes,
        "n_success":          n_success,
        "success_rate":       success_rate,
        "mean_final_dist":    float(np.mean(all_dists)),
        "std_final_dist":     float(np.std(all_dists)),
        "mean_steps_success": float(np.mean(succ_steps)) if succ_steps else None,
        "vlm_used":           use_vlm and (not use_gt_bbox),
        "max_delta":          max_delta,
    }

    print("\n-- Evaluation Results ------------------------------------------")
    print(f"  Episodes       : {n_episodes}")
    print(f"  Success rate   : {success_rate*100:.1f}%  ({n_success}/{n_episodes})")
    print(f"  Mean final dist: {summary['mean_final_dist']:.4f} m")
    print(f"  Mean steps (success): {summary['mean_steps_success']}")
    print("----------------------------------------------------------------")

    if out_path:
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "episodes": results}, f, indent=2)
        print(f"[eval] Results saved to {out_path}")

    return summary


# -- CLI -----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Closed-loop inference + evaluation.")
    p.add_argument("--checkpoint",  type=str,   default=DEFAULT_CHECKPOINT)
    p.add_argument("--n-episodes",  type=int,   default=DEFAULT_N_EPISODES)
    p.add_argument("--max-steps",   type=int,   default=DEFAULT_MAX_STEPS)
    p.add_argument("--vlm-url",     type=str,   default=DEFAULT_VLM_URL)
    p.add_argument("--vlm-model",   type=str,   default=DEFAULT_VLM_MODEL)
    p.add_argument("--use-gt-bbox", action="store_true",
                   help="Skip VLM; use ground-truth bboxes (ablation / no-VLM mode)")
    p.add_argument("--n-obj-min",   type=int,   default=DEFAULT_N_OBJ_MIN)
    p.add_argument("--n-obj-max",   type=int,   default=DEFAULT_N_OBJ_MAX)
    p.add_argument("--seed",        type=int,   default=DEFAULT_SEED)
    p.add_argument("--render-mode", type=str,   default=DEFAULT_RENDER_MODE,
                   choices=["offscreen", "gui"])
    p.add_argument("--device",      type=str,   default=DEFAULT_DEVICE)
    p.add_argument(
        "--horizon",
        type=int,
        nargs="+",
        default=DEFAULT_HORIZONS,
        help="Planning horizon(s), e.g. --horizon 1 or --horizon 1 2 3 4 5",
    )
    p.add_argument("--out-dir",     type=str,   default="",
                   help="Directory to save eval_results.json (optional)")
    p.add_argument("--save-vis",    action="store_true",
                   help="Save per-episode frame PNGs to out-dir/vis_episodes/")
    p.add_argument("--max-delta",   type=float, default=DEFAULT_MAX_DELTA,
                   help="Clip each predicted EE delta component in metres; use 0 to disable.")
    args = p.parse_args()

    horizons = args.horizon

    if len(horizons) > 1:
        print(f"\n{'='*70}")
        print(f"Running inference with multiple horizons: {horizons}")
        print(f"{'='*70}\n")
        results = {}

        for h in horizons:
            checkpoint = checkpoint_for_horizon(args.checkpoint, h)
            out_dir = f"{args.out_dir}/vis_h{h}" if args.out_dir else f"final_project/runs/vis_h{h}"

            print(f"\n{'='*70}")
            print(f"Inference HORIZON={h}")
            print(f"{'='*70}\n")

            eval_kwargs = {
                "checkpoint": checkpoint,
                "n_episodes": args.n_episodes,
                "max_steps": args.max_steps,
                "vlm_url": args.vlm_url,
                "vlm_model": args.vlm_model,
                "use_gt_bbox": args.use_gt_bbox,
                "n_obj_min": args.n_obj_min,
                "n_obj_max": args.n_obj_max,
                "seed": args.seed,
                "render_mode": args.render_mode,
                "device": args.device,
                "out_dir": out_dir,
                "save_vis": args.save_vis,
                "horizon": h,
                "max_delta": args.max_delta,
            }
            summary = evaluate(**eval_kwargs)
            results[h] = {
                "success_rate": summary.get("success_rate"),
                "mean_final_dist": summary.get("mean_final_dist"),
                "mean_steps_success": summary.get("mean_steps_success"),
            }

        # Summary
        print(f"\n{'='*70}")
        print("INFERENCE SUMMARY")
        print(f"{'='*70}")
        print(f"{'Horizon':<10} {'Success%':<12} {'Mean Dist':<12} {'Mean Steps':<12}")
        print("-" * 50)
        for h in sorted(results.keys()):
            res = results[h]
            sr = res.get("success_rate", 0) * 100
            md = res.get("mean_final_dist", 0)
            ms = res.get("mean_steps_success", 0)
            print(f"{h:<10} {sr:<12.1f} {md:<12.4f} {ms:<12.1f}")

        # Save summary
        summary_path = Path(args.out_dir).parent / "inference_summary.json" if args.out_dir else Path("final_project/runs/inference_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary saved to {summary_path}\n")
    else:
        # Single horizon
        eval_kwargs = vars(args)
        eval_kwargs["horizon"] = horizons[0]
        evaluate(**eval_kwargs)

if __name__ == "__main__":
    main()
