"""
Data collection: generate reaching trajectories for every object in every scene.

Each saved trajectory (*.pt) contains:
  target_name  : str
  success      : bool
  scene_id     : int
  n_steps      : int
  images       : (T, 3, 128, 128) uint8
  ee_positions : (T, 3)          float32  – world frame
  gt_bboxes    : (T, 4)          float32  – (cx,cy,w,h) normalised [0,1]
  obj_positions: (T, 3)          float32  – live object position at each step

Usage
-----
  python final_project/src/collect.py --n-scenes 100 --n-workers 4 \
    --out-dir final_project/data/trajectories
"""
import argparse
import ctypes.util
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ── path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from env import MultiObjectEnv, EE_Z          # noqa: E402
from utils import compute_gt_bbox             # noqa: E402

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_N_SCENES    = 100
DEFAULT_OUT_DIR     = "final_project/data/trajectories"
DEFAULT_MAX_STEPS   = 80
DEFAULT_STEP_SIZE   = 0.04
DEFAULT_N_OBJ_MIN   = 2
DEFAULT_N_OBJ_MAX   = 4
DEFAULT_SEED        = 42
DEFAULT_RENDER_MODE = "offscreen"
DEFAULT_N_WORKERS   = 64

# ── worker-process globals (set once per worker via initializer) ───────────────
_worker_env = None
_worker_cfg: dict = {}


def _init_worker(cfg: dict) -> None:
    """Runs once in each worker process: set GL backend and create env."""
    global _worker_env, _worker_cfg

    if "MUJOCO_GL" not in os.environ:
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_display:
            os.environ["MUJOCO_GL"] = "egl" if ctypes.util.find_library("EGL") else "osmesa"

    # Re-insert path (spawn starts a fresh interpreter)
    src = Path(__file__).resolve().parent
    hw_src = src.parent.parent / "boun_dl_robotics" / "cmpe591.github.io" / "src"
    for p in (str(src), str(hw_src)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from env import MultiObjectEnv  # noqa: E402 (re-import in worker)

    _worker_cfg = cfg
    _worker_env = MultiObjectEnv(
        n_objects_range=(cfg["n_obj_min"], cfg["n_obj_max"]),
        seed=cfg["seed"],
        render_mode=cfg["render_mode"],
    )


def _run_scene(scene_id: int) -> list[dict]:
    """Called in a worker process for a single scene."""
    global _worker_env, _worker_cfg
    from utils import compute_gt_bbox  # noqa: E402

    cfg = _worker_cfg
    env = _worker_env
    env._object_seed = cfg["seed"] + scene_id * 1000
    env.reset()
    return collect_scene(
        env=env,
        scene_id=scene_id,
        max_steps=cfg["max_steps"],
        step_size=cfg["step_size"],
        out_dir=Path(cfg["out_dir"]),
        verbose=cfg["verbose"],
    )


# ── single-scene collection ───────────────────────────────────────────────────

def collect_scene(
    env: MultiObjectEnv,
    scene_id: int,
    max_steps: int,
    step_size: float,
    out_dir: Path,
    verbose: bool = False,
) -> list[dict]:
    """
    Collect one trajectory per object in the current scene.
    Returns list of metadata dicts (one per trajectory attempted).
    """
    obj_infos = env.get_object_info()
    results   = []

    for obj in obj_infos:
        target_name = obj["name"]

        env.reset()

        target_xy  = env.get_object_pos(target_name)[:2]
        target_pos = np.array([target_xy[0], target_xy[1], EE_Z], dtype=np.float64)

        images        = []
        ee_positions  = []
        gt_bboxes     = []
        obj_positions = []

        success = False
        step    = 0

        while step < max_steps:
            ee_pos, _, image = env.state()
            bbox    = compute_gt_bbox(env.model, env.data, target_name)
            obj_pos = env.get_object_pos(target_name)

            images.append(image)
            ee_positions.append(torch.from_numpy(ee_pos))
            gt_bboxes.append(torch.from_numpy(bbox))
            obj_positions.append(torch.from_numpy(obj_pos.astype(np.float32)))

            if env.check_contact(target_name):
                success = True
                break

            env.move_ee_step(target_pos, step_size=step_size)
            step += 1

        n_steps = len(images)
        if verbose:
            print(f"  scene {scene_id:04d} | {target_name:25s} | "
                  f"{'✓' if success else '✗'} steps={n_steps}")

        if not success:
            results.append({"scene_id": scene_id, "target_name": target_name,
                            "success": False, "n_steps": n_steps, "file": None})
            continue

        traj_path = out_dir / f"traj_{scene_id:06d}_{target_name}.pt"
        torch.save({
            "target_name":   target_name,
            "success":       True,
            "scene_id":      scene_id,
            "n_steps":       n_steps,
            "images":        torch.stack(images),
            "ee_positions":  torch.stack(ee_positions),
            "gt_bboxes":     torch.stack(gt_bboxes),
            "obj_positions": torch.stack(obj_positions),
        }, traj_path)

        results.append({
            "scene_id":    scene_id,
            "target_name": target_name,
            "success":     True,
            "n_steps":     n_steps,
            "file":        str(traj_path),
        })

    return results


# ── metadata helper ──────────────────────────────────────────────────────────

def _save_meta(out: Path, all_results: list[dict], n_scenes: int,
               max_steps: int, step_size: float,
               n_obj_min: int, n_obj_max: int, seed: int) -> None:
    n_success = sum(1 for r in all_results if r["success"])
    meta = {
        "n_scenes":       n_scenes,
        "n_trajectories": len(all_results),
        "n_success":      n_success,
        "max_steps":      max_steps,
        "step_size":      step_size,
        "n_obj_range":    [n_obj_min, n_obj_max],
        "seed":           seed,
        "trajectories":   all_results,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


# ── main collect ──────────────────────────────────────────────────────────────

def collect(
    n_scenes:    int   = DEFAULT_N_SCENES,
    out_dir:     str   = DEFAULT_OUT_DIR,
    max_steps:   int   = DEFAULT_MAX_STEPS,
    step_size:   float = DEFAULT_STEP_SIZE,
    n_obj_min:   int   = DEFAULT_N_OBJ_MIN,
    n_obj_max:   int   = DEFAULT_N_OBJ_MAX,
    seed:        int   = DEFAULT_SEED,
    render_mode: str   = DEFAULT_RENDER_MODE,
    verbose:     bool  = False,
    n_workers:   int   = DEFAULT_N_WORKERS,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    meta_kw = dict(n_scenes=n_scenes, max_steps=max_steps, step_size=step_size,
                   n_obj_min=n_obj_min, n_obj_max=n_obj_max, seed=seed)

    if n_workers <= 1:
        # ── single process ────────────────────────────────────────────────────
        env = MultiObjectEnv(
            n_objects_range=(n_obj_min, n_obj_max),
            seed=seed,
            render_mode=render_mode,
        )
        for scene_id in tqdm(range(n_scenes), desc="scenes"):
            env._object_seed = seed + scene_id * 1000
            env.reset()
            all_results.extend(
                collect_scene(env, scene_id, max_steps, step_size, out, verbose)
            )
            _save_meta(out, all_results, **meta_kw)
    else:
        # ── multiprocessing ───────────────────────────────────────────────────
        worker_cfg = dict(
            out_dir=str(out),
            max_steps=max_steps,
            step_size=step_size,
            n_obj_min=n_obj_min,
            n_obj_max=n_obj_max,
            seed=seed,
            render_mode=render_mode,
            verbose=verbose,
        )
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            n_workers,
            initializer=_init_worker,
            initargs=(worker_cfg,),
        ) as pool:
            for scene_results in tqdm(
                pool.imap_unordered(_run_scene, range(n_scenes)),
                total=n_scenes,
                desc="scenes",
            ):
                all_results.extend(scene_results)
                _save_meta(out, all_results, **meta_kw)

        all_results.sort(key=lambda r: (r["scene_id"], r["target_name"]))
        _save_meta(out, all_results, **meta_kw)  # final write with sorted order

    n_success = sum(1 for r in all_results if r["success"])
    rate = n_success / max(len(all_results), 1) * 100
    print(f"\nCollection done.")
    print(f"  Total attempted : {len(all_results)}")
    print(f"  Successful      : {n_success}  ({rate:.1f}%)")
    print(f"  Saved to        : {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect reaching trajectories.")
    parser.add_argument("--n-scenes",    type=int,   default=DEFAULT_N_SCENES)
    parser.add_argument("--out-dir",     type=str,   default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-steps",   type=int,   default=DEFAULT_MAX_STEPS)
    parser.add_argument("--step-size",   type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--n-obj-min",   type=int,   default=DEFAULT_N_OBJ_MIN)
    parser.add_argument("--n-obj-max",   type=int,   default=DEFAULT_N_OBJ_MAX)
    parser.add_argument("--seed",        type=int,   default=DEFAULT_SEED)
    parser.add_argument("--render-mode", type=str,   default=DEFAULT_RENDER_MODE,
                        choices=["offscreen", "gui"])
    parser.add_argument("--n-workers",   type=int,   default=DEFAULT_N_WORKERS,
                        help="Parallel worker processes (default: 1)")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()
    collect(**vars(args))


if __name__ == "__main__":
    main()
