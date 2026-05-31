"""
Multi-object tabletop environment.

Random boxes and spheres at random positions with distinct colors.
Supports step-by-step EE motion and MuJoCo contact detection.
"""
import os
import sys
import ctypes.util
from pathlib import Path
from typing import Any, Optional, cast

# Set MUJOCO_GL before mujoco is imported (it reads the env var at import time).
if os.name == "nt":
    # Windows cannot use egl here; force a safe backend even if the shell
    # inherited an incompatible MUJOCO_GL value.
    os.environ["MUJOCO_GL"] = "glfw"
elif "MUJOCO_GL" not in os.environ:
    _has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not _has_display:
        os.environ["MUJOCO_GL"] = "egl" if ctypes.util.find_library("EGL") else "osmesa"

_SRC = Path(__file__).resolve().parent
_HW_SRC = _SRC.parent.parent / "boun_dl_robotics" / "cmpe591.github.io" / "src"
sys.path.insert(0, str(_HW_SRC))

import numpy as np
import torch
import torchvision.transforms.functional as TF
import mujoco

from environment import BaseEnv, create_tabletop_scene, create_object  # noqa: E402

# ── Object catalogue ──────────────────────────────────────────────────────────
SHAPES = ["box", "sphere"]
COLORS: dict[str, list[float]] = {
    "red":    [0.90, 0.10, 0.10, 1.0],
    "green":  [0.10, 0.80, 0.10, 1.0],
    "blue":   [0.10, 0.10, 0.90, 1.0],
    "yellow": [0.90, 0.80, 0.00, 1.0],
    "purple": [0.60, 0.00, 0.80, 1.0],
    "cyan":   [0.00, 0.80, 0.80, 1.0],
}
COLOR_NAMES = list(COLORS.keys())

# ── Table workspace ───────────────────────────────────────────────────────────
TABLE_X = (0.35, 0.75)
TABLE_Y = (-0.25, 0.25)
TABLE_Z = 1.10          # object spawn height
EE_Z    = 1.065         # working height for EE

OBJ_HALF = 0.025        # half-size of objects
MIN_SEP  = 0.10         # minimum centre-to-centre distance between objects

IMG_H = IMG_W = 128
IMG_SHAPE = (3, IMG_H, IMG_W)


class MultiObjectEnv(BaseEnv):
    """Tabletop env with randomly placed coloured boxes and spheres."""

    def __init__(
        self,
        n_objects_range: tuple[int, int] = (2, 4),
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        # Must be set BEFORE super().__init__ because reset() is called there.
        self._n_objects_range = n_objects_range
        self._object_seed = seed
        self._obj_info: list[dict] = []
        super().__init__(**kwargs)

    # ── Scene creation ────────────────────────────────────────────────────────

    def _create_scene(self, seed: Optional[int] = None):
        rng_seed = seed if seed is not None else self._object_seed
        if rng_seed is not None:
            np.random.seed(rng_seed)

        scene = create_tabletop_scene()

        n = int(np.random.randint(self._n_objects_range[0], self._n_objects_range[1] + 1))

        # Unique colour and random shape per object
        color_idxs = np.random.choice(len(COLOR_NAMES), size=n, replace=False)
        shapes = np.random.choice(SHAPES, size=n)

        self._obj_info = []
        placed: list[np.ndarray] = []

        for i in range(n):
            # Rejection sampling for non-overlapping positions
            for _ in range(200):
                x = float(np.random.uniform(*TABLE_X))
                y = float(np.random.uniform(*TABLE_Y))
                pos = np.array([x, y, TABLE_Z], dtype=np.float32)
                if all(np.linalg.norm(pos[:2] - p[:2]) >= MIN_SEP for p in placed):
                    break
            placed.append(pos)

            color_name = COLOR_NAMES[int(color_idxs[i])]
            shape      = shapes[i]
            name       = f"{color_name}_{shape}_{i}"
            rgba       = COLORS[color_name]
            size       = [OBJ_HALF, OBJ_HALF, OBJ_HALF]

            create_object(scene, shape, pos=pos.tolist(), quat=[0, 0, 0, 1],
                          size=size, rgba=rgba, name=name)

            self._obj_info.append({
                "name":   name,
                "shape":  shape,
                "color":  color_name,
                "spawn_pos": pos.copy(),
            })

        return scene

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_object_names(self) -> list[str]:
        return [o["name"] for o in self._obj_info]

    def get_object_info(self) -> list[dict]:
        return list(self._obj_info)

    def get_object_pos(self, name: str) -> np.ndarray:
        """Current 3-D position of the object (live from MuJoCo data)."""
        return self.data.body(name).xpos.copy()

    # ── State ─────────────────────────────────────────────────────────────────

    def state(self) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """
        Returns
        -------
        ee_pos       : (3,) float32  – end-effector xyz in world frame
        joint_angles : (7,) float32  – normalised joint positions
        image        : (3, 128, 128) uint8 torch tensor (0-255)
        """
        ee_pos       = self.data.site(self._ee_site).xpos.astype(np.float32).copy()
        joint_angles = self._get_joint_position().astype(np.float32)
        viewer = cast(Any, self.viewer)

        if self._render_mode == "offscreen":
            viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = viewer.read_pixels(camid=1)[0].copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            crop_size = min(pixels.shape[1:])
            pixels = TF.center_crop(pixels, [crop_size, crop_size])
            pixels = TF.resize(pixels, [IMG_H, IMG_W])

        return ee_pos, joint_angles, pixels

    # ── Motion ────────────────────────────────────────────────────────────────

    def move_ee_step(
        self,
        target_pos: np.ndarray,
        step_size: float = 0.04,
        n_splits: int = 10,
    ) -> np.ndarray:
        """
        Move EE one step (at most *step_size* metres) towards *target_pos*.
        Returns new EE position.
        """
        ee_pos  = self.data.site(self._ee_site).xpos.copy()
        delta   = np.array(target_pos) - ee_pos
        dist    = float(np.linalg.norm(delta))
        if dist < 1e-4:
            return ee_pos
        move    = delta / dist * min(step_size, dist)
        new_pos = ee_pos + move
        new_pos[2] = max(new_pos[2], EE_Z)
        self._set_ee_in_cartesian(new_pos, rotation=[-90, 0, 180],
                                  n_splits=n_splits, threshold=0.03)
        return self.data.site(self._ee_site).xpos.astype(np.float32).copy()

    def move_ee_to_pose(
        self,
        target_pos: np.ndarray,
        n_splits: int = 10,
    ) -> np.ndarray:
        """
        Move EE directly to *target_pos* (predicted waypoint from MLP).
        Returns new EE position.
        """
        pos = np.array(target_pos, dtype=np.float64)
        pos[2] = max(float(pos[2]), EE_Z)
        self._set_ee_in_cartesian(pos, rotation=[-90, 0, 180],
                                  n_splits=n_splits, threshold=0.03)
        return self.data.site(self._ee_site).xpos.astype(np.float32).copy()

    def reset_ee(self) -> None:
        """Return arm to its home (initial joint) configuration."""
        self._set_joint_position(
            {i: angle for i, angle in enumerate(self._init_position)},
            max_iters=2000,
            threshold=0.05,
        )

    # ── Contact detection ─────────────────────────────────────────────────────

    def check_contact(self, target_name: str) -> bool:
        """True if any robot/gripper geom is in contact with *target_name*."""
        # Cache geom sets on first call (they don't change per episode)
        if not hasattr(self, "_robot_geom_ids"):
            self._robot_geom_ids = self._get_robot_geom_ids()

        target_geom_ids = self._get_body_geom_ids(target_name)

        for c_idx in range(int(self.data.ncon)):
            g1 = int(self.data.contact[c_idx].geom1)
            g2 = int(self.data.contact[c_idx].geom2)
            if (g1 in self._robot_geom_ids and g2 in target_geom_ids) or \
               (g2 in self._robot_geom_ids and g1 in target_geom_ids):
                return True
        return False

    def recent_contact_with_body(self, body_name: str) -> bool:
        """Return True if any recent contact involved the robot and the given body.

        This looks at contact pairs recorded by BaseEnv.pop_recent_contacts() and
        tests whether any pair contains one robot geom and one geom belonging to
        the named body.
        """
        # Ensure robot geom ids are cached
        if not hasattr(self, "_robot_geom_ids"):
            self._robot_geom_ids = self._get_robot_geom_ids()

        target_geom_ids = self._get_body_geom_ids(body_name)

        # pop recent contacts (non-destructive check would be fine too, but we
        # use pop to avoid repeated detection of the same contact)
        if not hasattr(self, "pop_recent_contacts"):
            return False
        pairs = self.pop_recent_contacts()
        for pair in pairs:
            ids = set(pair)
            if (ids & self._robot_geom_ids) and (ids & target_geom_ids):
                return True
        return False

    def _get_body_geom_ids(self, body_name: str) -> set[int]:
        body_id = int(self.model.body(body_name).id)
        return {i for i in range(self.model.ngeom)
                if int(np.asarray(self.model.geom(i).bodyid).flat[0]) == body_id}

    def _get_robot_geom_ids(self) -> set[int]:
        ids: set[int] = set()
        for i in range(self.model.nbody):
            bname = self.model.body(i).name
            if any(kw in bname for kw in ("ur5e", "robotiq", "gripper", "finger", "pad")):
                ids |= {j for j in range(self.model.ngeom)
                        if int(np.asarray(self.model.geom(j).bodyid).flat[0]) == i}
        return ids
