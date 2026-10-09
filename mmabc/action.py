"""MiVerse action layout.

Raw keys and the 75-d model vector use the same order. A pose is stored as
``[x, y, z, qw, qx, qy, qz]`` and becomes position (3) plus rotation-6D (6)
inside the model vector. Manipulation owns ``[0, 56)``; body motion owns ``[56, 75)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from mmabc.canonical import rotation as rot

ACTION_DIM = 75
VIEWS = ("primary", "wrist_left", "wrist_right")
CAMERAS = {
    "primary": "cam_head",
    "wrist_left": "cam_left_wrist",
    "wrist_right": "cam_right_wrist",
}

KEYS: tuple[tuple[str, int, str], ...] = (
    ("left_arm_joint_states", 7, "vec"),
    ("right_arm_joint_states", 7, "vec"),
    ("left_ee_joint_states", 12, "vec"),
    ("right_ee_joint_states", 12, "vec"),
    ("left_base_ee_poses", 7, "pose"),
    ("right_base_ee_poses", 7, "pose"),
    ("waist_joint_states", 2, "vec"),
    ("leg_joint_states", 2, "vec"),
    ("head_base_poses", 7, "pose"),
    ("root_linear_velocity", 3, "vec"),
    ("root_angular_velocity", 3, "vec"),
)

KEY_ALIASES = {"head_base_poses": ("base_head_poses",)}


@dataclass(frozen=True)
class KeySlot:
    key: str
    raw_dim: int
    kind: str
    start: int
    end: int


def _slots() -> tuple[KeySlot, ...]:
    slots, cursor = [], 0
    for key, dim, kind in KEYS:
        width = 9 if kind == "pose" else dim
        slots.append(KeySlot(key, dim, kind, cursor, cursor + width))
        cursor += width
    return tuple(slots)


SLOTS = _slots()
assert SLOTS[-1].end == ACTION_DIM


def singular(key: str) -> str:
    if key.endswith("_states"):
        return key[: -len("_states")] + "_state"
    if key.endswith("_poses"):
        return key[: -len("_poses")] + "_pose"
    return key


def pose_to_model(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    r6 = rot.matrix_to_rot6d(rot.quat_wxyz_to_matrix(pose[..., 3:7]))
    return np.concatenate([pose[..., :3], r6], axis=-1)


def model_to_pose(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    q = rot.matrix_to_quat_wxyz(rot.rot6d_to_matrix(vec[..., 3:9]))
    return np.concatenate([vec[..., :3], q], axis=-1)


def _lookup(values: Mapping[str, object], key: str):
    names = [key, singular(key), *KEY_ALIASES.get(key, ())]
    names += [singular(a) for a in KEY_ALIASES.get(key, ())]
    for cand in names:
        for name in (cand, f"state.{cand}", f"state/{cand}"):
            if name in values:
                return values[name]
    return None


def pack(values: Mapping[str, object], *, allow_missing: bool = False) -> np.ndarray:
    """Raw per-key arrays -> ``(..., 75)`` model vector."""
    lead = None
    for slot in SLOTS:
        raw = _lookup(values, slot.key)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=np.float64)
        if arr.shape[-1] != slot.raw_dim:
            raise ValueError(f"{slot.key}: expected last dim {slot.raw_dim}, got {arr.shape}")
        lead = arr.shape[:-1]
        break
    if lead is None:
        if not allow_missing:
            raise KeyError(f"missing MiVerse key {SLOTS[0].key!r}")
        lead = ()

    parts = []
    for slot in SLOTS:
        raw = _lookup(values, slot.key)
        if raw is None:
            if not allow_missing:
                raise KeyError(f"missing MiVerse key {slot.key!r} (or {singular(slot.key)!r})")
            neutral = np.zeros(lead + (slot.raw_dim,), dtype=np.float64)
            if slot.kind == "pose":
                neutral[..., 3] = 1.0
            raw = neutral
        arr = np.asarray(raw, dtype=np.float64).reshape(lead + (slot.raw_dim,))
        parts.append(pose_to_model(arr) if slot.kind == "pose" else arr)
    return np.concatenate(parts, axis=-1)


def unpack(vec: np.ndarray, *, key_style: str = "singular") -> dict[str, np.ndarray]:
    """``(..., 75)`` model vector -> raw per-key arrays. Poses return to wxyz quaternions."""
    vec = np.asarray(vec, dtype=np.float64)
    if vec.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected last dim {ACTION_DIM}, got {vec.shape}")
    out = {}
    for slot in SLOTS:
        part = vec[..., slot.start : slot.end]
        value = model_to_pose(part) if slot.kind == "pose" else part
        name = singular(slot.key) if key_style == "singular" else slot.key
        out[name] = value.astype(np.float32)
    return out
