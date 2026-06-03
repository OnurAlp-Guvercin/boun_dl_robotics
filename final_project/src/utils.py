"""
Shared utilities: camera projection, coordinate normalisation, bbox helpers.
"""
import sys
from pathlib import Path
from typing import Optional

_HW_SRC = Path(__file__).resolve().parent.parent.parent / "boun_dl_robotics" / "cmpe591.github.io" / "src"
sys.path.insert(0, str(_HW_SRC))

import numpy as np
import mujoco  # noqa: E402

# -- EE workspace bounds (used for normalisation) ------------------------------
EE_BOUNDS = np.array([
    [0.20, 1.00],   # x
    [-0.40, 0.40],  # y
    [0.90, 1.50],   # z
], dtype=np.float32)   # shape (3, 2)


def normalise_ee(pos: np.ndarray) -> np.ndarray:
    """Map EE position from world space to [0, 1]^3."""
    lo, hi = EE_BOUNDS[:, 0], EE_BOUNDS[:, 1]
    return np.clip((np.array(pos, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0)


def denormalise_ee(norm: np.ndarray) -> np.ndarray:
    """Inverse of normalise_ee."""
    lo, hi = EE_BOUNDS[:, 0], EE_BOUNDS[:, 1]
    return np.array(norm, dtype=np.float32) * (hi - lo) + lo


# -- Camera projection ---------------------------------------------------------

def world_to_pixel(
    model,
    data,
    world_pos: np.ndarray,
    cam_name: str = "topdown",
    img_h: int = 128,
    img_w: int = 128,
) -> Optional[tuple[float, float]]:
    """
    Project a 3-D world point to pixel (u, v) using the named MuJoCo camera.

    Returns None if the point is behind the camera.

    Convention (MuJoCo):
      cam_xmat is the camera-to-world rotation matrix.
      World points must be rotated by cam_xmat.T to enter camera frame.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{cam_name}' not found in model")

    cam_pos = data.cam_xpos[cam_id].copy()                 # (3,)
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()   # camera-to-world rotation

    # Vector from camera to point, in camera frame
    dp    = np.array(world_pos, dtype=np.float64) - cam_pos
    p_cam = cam_mat.T @ dp   # (3,): x=right, y=up, z=away-from-scene (MuJoCo convention)

    # MuJoCo camera z-axis points AWAY from the scene, so objects in front have p_cam[2] < 0.
    depth = -p_cam[2]
    if depth <= 1e-6:
        return None

    fovy_deg = float(model.cam_fovy[cam_id])
    f = (img_h / 2.0) / np.tan(np.deg2rad(fovy_deg / 2.0))

    u = f * p_cam[0] / depth + img_w / 2.0
    v = img_h / 2.0 - f * p_cam[1] / depth

    return float(u), float(v)


def compute_gt_bbox(
    model,
    data,
    obj_name: str,
    half_size: float = 0.025,
    cam_name: str = "topdown",
    img_h: int = 128,
    img_w: int = 128,
) -> np.ndarray:
    """
    Compute ground-truth bounding box for an object as (cx, cy, w, h)
    normalised to [0, 1] in image coordinates.

    Falls back to centre-only (with fixed small size) if projection fails.
    """
    obj_pos = data.body(obj_name).xpos.copy()

    centre = world_to_pixel(model, data, obj_pos, cam_name, img_h, img_w)
    if centre is None:
        return np.array([0.5, 0.5, 0.05, 0.05], dtype=np.float32)

    cx_px, cy_px = centre

    # Project corners to estimate pixel size
    offsets = half_size * np.array([
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    ])
    pixel_corners = [
        world_to_pixel(model, data, obj_pos + off, cam_name, img_h, img_w)
        for off in offsets
    ]
    valid = [p for p in pixel_corners if p is not None]

    if valid:
        xs = [p[0] for p in valid]
        ys = [p[1] for p in valid]
        w_px = max(xs) - min(xs)
        h_px = max(ys) - min(ys)
    else:
        w_px = h_px = 8.0  # fallback ~6% of 128

    return np.array([
        cx_px / img_w,
        cy_px / img_h,
        max(w_px / img_w, 0.02),
        max(h_px / img_h, 0.02),
    ], dtype=np.float32)


def bbox_to_input(bbox: np.ndarray, ee_pos: np.ndarray) -> np.ndarray:
    """
    Concatenate normalised bbox (4) and normalised ee_pos (3) → (7,) float32.
    """
    return np.concatenate([
        np.array(bbox, dtype=np.float32),
        normalise_ee(np.array(ee_pos, dtype=np.float32)),
    ]).astype(np.float32)
