"""
make_videos.py  –  Replay successful episodes and export as MP4.

Episodes are replayed by re-creating the MuJoCo env with the same seed and
executing the saved predicted_poses from eval_results.json.  No changes to
inference.py are required.

Usage
-----
python final_project/src/make_videos.py \
    --eval-json  final_project/runs/vlm_runs/vis_h1/eval_results.json \
    --n          5 \
    --out-dir    final_project/runs/vlm_runs/videos \
    --fps        15 \
    --seed       100
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Select the first EGL device explicitly so MuJoCo can init without DRI access.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_DEVICE_ID", "0")

import numpy as np
import imageio
import mujoco

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from env import MultiObjectEnv  # noqa: E402

DEFAULT_FPS       = 15
DEFAULT_N         = 5
DEFAULT_SEED      = 100   # must match the seed used during inference
DEFAULT_N_OBJ_MIN = 2
DEFAULT_N_OBJ_MAX = 4
DEFAULT_RENDER_SIZE = 480   # native render resolution for video (EGL framebuffer limit)


def _capture(renderer: mujoco.Renderer, data: mujoco.MjData) -> np.ndarray:
    """Render a frame at the renderer's native resolution. Returns (H, W, 3) uint8."""
    renderer.update_scene(data, camera="topdown")
    return renderer.render().copy()


def collect_frames(
    ep_data:  dict,
    env:      MultiObjectEnv,
    renderer: mujoco.Renderer,
    base_seed: int,
) -> list[np.ndarray]:
    """
    Reset the shared env to the episode's seed, replay predicted_poses,
    and return frames. env and renderer are reused across episodes to keep
    the GL context alive.
    """
    ep_seed = base_seed + ep_data["episode"] * 997
    env._object_seed = ep_seed
    env.reset()

    # After reset the model is reloaded; update the renderer's scene binding.
    # (mujoco.Renderer is already bound to env.model by reference — just render.)
    frames: list[np.ndarray] = [_capture(renderer, env.data)]

    for block in ep_data["trajectory"]:
        for pose in block.get("predicted_poses", []):
            env.move_ee_to_pose(np.array(pose, dtype=np.float64))
            frames.append(_capture(renderer, env.data))

    return frames


def draw_bbox_on_frame(
    frame: np.ndarray,
    bbox:  list[float],
    color: tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw (cx, cy, w, h) normalised bbox onto an (H, W, 3) uint8 frame."""
    frame = frame.copy()
    H, W  = frame.shape[:2]
    cx, cy, w, h = bbox
    x0 = int((cx - w / 2) * W)
    y0 = int((cy - h / 2) * H)
    x1 = int((cx + w / 2) * W)
    y1 = int((cy + h / 2) * H)
    x0, x1 = max(0, x0), min(W - 1, x1)
    y0, y1 = max(0, y0), min(H - 1, y1)

    # Top / bottom
    frame[y0:y0 + thickness, x0:x1] = color
    frame[y1 - thickness:y1, x0:x1] = color
    # Left / right
    frame[y0:y1, x0:x0 + thickness] = color
    frame[y0:y1, x1 - thickness:x1] = color
    return frame


def make_video(
    eval_json:   str,
    n:           int   = DEFAULT_N,
    out_dir:     str   = "",
    fps:         int   = DEFAULT_FPS,
    seed:        int   = DEFAULT_SEED,
    n_obj_min:   int   = DEFAULT_N_OBJ_MIN,
    n_obj_max:   int   = DEFAULT_N_OBJ_MAX,
    only_success:bool  = True,
    draw_bbox:   bool  = True,
    render_size: int   = DEFAULT_RENDER_SIZE,
) -> None:
    with open(eval_json) as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    if only_success:
        episodes = [e for e in episodes if e["success"]]

    if not episodes:
        print("[video] No episodes match the filter.")
        return

    chosen = random.sample(episodes, min(n, len(episodes)))
    chosen.sort(key=lambda e: e["episode"])

    out = Path(out_dir) if out_dir else Path(eval_json).parent / "videos"
    out.mkdir(parents=True, exist_ok=True)

    # Create env and renderer ONCE — reusing them keeps the GL context alive
    # across all episodes (avoids gladLoadGL errors on the 2nd+ env creation).
    first_seed = seed + chosen[0]["episode"] * 997
    env = MultiObjectEnv(
        n_objects_range=(n_obj_min, n_obj_max),
        seed=first_seed,
        render_mode="offscreen",
    )
    renderer = mujoco.Renderer(env.model, height=render_size, width=render_size)

    for ep_data in chosen:
        ep_idx  = ep_data["episode"]
        target  = ep_data["target_name"]
        status  = "success" if ep_data["success"] else "fail"
        n_steps = ep_data["n_steps"]

        print(f"[video] ep={ep_idx:03d}  target={target}  {status}  steps={n_steps}")

        frames = collect_frames(
            ep_data   = ep_data,
            env       = env,
            renderer  = renderer,
            base_seed = seed,
        )

        if draw_bbox and ep_data["trajectory"]:
            bbox = ep_data["trajectory"][0]["bbox"]
            frames = [draw_bbox_on_frame(f, bbox) for f in frames]

        fname = out / f"ep{ep_idx:03d}_{target}_{status}.mp4"
        writer = imageio.get_writer(str(fname), fps=fps, codec="libx264",
                                    quality=8, pixelformat="yuv420p")
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        h, w = frames[0].shape[:2]
        print(f"         → {fname}  ({len(frames)} frames, {w}×{h})")

    print(f"\n[video] Done. {len(chosen)} video(s) saved to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate videos from eval episodes.")
    p.add_argument("--eval-json",    type=str, required=True,
                   help="Path to eval_results.json")
    p.add_argument("--n",            type=int, default=DEFAULT_N,
                   help="Number of episodes to render")
    p.add_argument("--out-dir",      type=str, default="",
                   help="Output directory (default: eval_json/../videos/)")
    p.add_argument("--fps",          type=int, default=DEFAULT_FPS)
    p.add_argument("--seed",         type=int, default=DEFAULT_SEED,
                   help="Base seed used during inference (default: 100)")
    p.add_argument("--n-obj-min",    type=int, default=DEFAULT_N_OBJ_MIN)
    p.add_argument("--n-obj-max",    type=int, default=DEFAULT_N_OBJ_MAX)
    p.add_argument("--all-episodes", action="store_true",
                   help="Include failed episodes too (default: success only)")
    p.add_argument("--no-bbox",      action="store_true",
                   help="Skip bbox overlay on frames")
    p.add_argument("--render-size",  type=int, default=DEFAULT_RENDER_SIZE,
                   help=f"Native render resolution (default: {DEFAULT_RENDER_SIZE})")
    p.add_argument("--rng-seed",     type=int, default=0,
                   help="Seed for episode sampling randomness")
    args = p.parse_args()

    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    make_video(
        eval_json    = args.eval_json,
        n            = args.n,
        out_dir      = args.out_dir,
        fps          = args.fps,
        seed         = args.seed,
        n_obj_min    = args.n_obj_min,
        n_obj_max    = args.n_obj_max,
        only_success = not args.all_episodes,
        draw_bbox    = not args.no_bbox,
        render_size  = args.render_size,
    )


if __name__ == "__main__":
    main()
