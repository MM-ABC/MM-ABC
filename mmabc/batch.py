"""Tensor batches for ``MMABCPolicy``.

Inference images are uint8 ``(B, V, H, W, 3)``. Training images add a time axis,
``(B, V, T, H, W, 3)``, with the current frame at index 0 and one frame per
distinct future offset after it. State and action are normalised ``action_dim`` vectors.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from mmabc.action import pack


def _tensor(value, *, dtype, device) -> torch.Tensor:
    if isinstance(value, Mapping):
        value = pack(value)
    if not torch.is_tensor(value):
        value = torch.as_tensor(np.asarray(value), dtype=dtype)
    else:
        value = value.to(dtype=dtype)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    return value.to(device)


def _images(images, *, device) -> torch.Tensor:
    if not torch.is_tensor(images):
        images = torch.as_tensor(np.asarray(images))
    if images.dtype != torch.uint8:
        images = images.to(torch.uint8)
    if images.ndim == 4:
        images = images.unsqueeze(0)
    if images.ndim == 5:
        images = images.unsqueeze(2)
    if images.ndim != 6:
        raise ValueError(f"images must be (B, V, H, W, 3) or (B, V, T, H, W, 3), got {tuple(images.shape)}")
    return images.to(device)


def make_batch(
    policy,
    images,
    state,
    instructions,
    *,
    view_mask=None,
    target=None,
    target_mask=None,
    future_valid=None,
    device=None,
) -> dict:
    """Build the dict consumed by ``forward`` and ``predict_action``."""
    if device is None:
        device = state.device if torch.is_tensor(state) else torch.device("cpu")
    device = torch.device(device)
    spec = policy.input_spec()
    images = _images(images, device=device)
    state = _tensor(state, dtype=torch.float32, device=device)
    batch_size = state.shape[0]
    if images.shape[0] != batch_size:
        raise ValueError(f"images batch {images.shape[0]} != state batch {batch_size}")
    if isinstance(instructions, str):
        instructions = [instructions] * batch_size
    if len(instructions) != batch_size:
        raise ValueError("one instruction is required per sample")

    dim = spec["state_dim"]
    if state.shape[-1] != dim:
        raise ValueError(f"state last dim {state.shape[-1]} != {dim}")
    views = images.shape[1]
    if view_mask is None:
        view_mask = torch.ones(batch_size, views, dtype=torch.float32, device=device)
    else:
        view_mask = _tensor(view_mask, dtype=torch.float32, device=device)

    batch = {
        "images": images,
        "state": state,
        "state_mask": torch.ones(batch_size, dim, dtype=torch.float32, device=device),
        "aux_active": torch.ones(batch_size, dtype=torch.float32, device=device),
        "view_mask": view_mask,
        "prompt": list(instructions),
    }
    if target is not None:
        target = _tensor(target, dtype=torch.float32, device=device)
        if target_mask is None:
            target_mask = torch.ones_like(target)
        else:
            target_mask = _tensor(target_mask, dtype=torch.float32, device=device)
        batch["target"] = target
        batch["target_mask"] = target_mask
    if future_valid is not None:
        batch["future_valid"] = _tensor(future_valid, dtype=torch.float32, device=device)
    return batch


def synthetic_batch(policy, batch_size: int = 1, *, train: bool = False, device="cpu", seed: int = 0) -> dict:
    """Random tensors with the shapes ``policy.input_spec()`` describes."""
    spec = policy.input_spec()
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    views = len(spec["views"])
    height = width = spec["image_size"]
    dim = spec["state_dim"]
    if train:
        frames = spec["train_images"][1]
        images = torch.randint(0, 256, (batch_size, views, frames, height, width, 3), dtype=torch.uint8, generator=g)
        target = torch.randn(batch_size, spec["chunk_size"], dim, generator=g)
        horizons = len(policy.future_offsets) if policy.future_enabled else 0
        future_valid = torch.ones(batch_size, horizons) if horizons else None
    else:
        images = torch.randint(0, 256, (batch_size, views, height, width, 3), dtype=torch.uint8, generator=g)
        target = None
        future_valid = None
    state = torch.randn(batch_size, dim, generator=g)
    return make_batch(
        policy,
        images,
        state,
        ["pick up the cup and place it on the tray"] * batch_size,
        target=target,
        future_valid=future_valid,
        device=device,
    )
