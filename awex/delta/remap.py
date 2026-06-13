# Licensed under the Apache License, Version 2.0
"""Delta index remapping for TP mismatch scenarios.

When training TP != inference TP, flat delta indices computed on the training
shard must be filtered and remapped to the inference shard's coordinate space.
Uses AWEX's CommunicationOperation (train_slices / inf_slices) to define
the overlap region for each (train_rank, infer_rank) pair.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from awex.delta.patch import SparseWeightPatch

logger = logging.getLogger(__name__)


def remap_delta_indices(
    patch: SparseWeightPatch,
    train_shape: tuple[int, ...],
    train_slices: tuple[slice, ...],
    inf_slices: tuple[slice, ...],
    infer_shape: tuple[int, ...],
) -> Optional[SparseWeightPatch]:
    """Remap flat delta indices from training shard space to inference shard space.

    Uses the overlap region defined by CommunicationOperation to:
    1. Unflatten training flat indices → multi-dim
    2. Filter: keep only indices within train_slices
    3. Remap: translate from train_slices coords to inf_slices coords
    4. Flatten for inference shard shape

    Args:
        patch: sparse patch with indices in training shard flat space
        train_shape: shape of the training shard tensor
        train_slices: slice ranges defining the overlap in training shard
        inf_slices: slice ranges defining where to write in inference shard
        infer_shape: shape of the inference shard tensor

    Returns:
        New SparseWeightPatch with remapped indices, or None if no overlap.
    """
    indices = patch.indices.long()
    values = patch.values
    ndim = len(train_shape)

    if ndim == 0 or indices.numel() == 0:
        return None

    # Step 1: unflatten to multi-dim indices
    if ndim == 1:
        multi_idx = (indices,)
    elif ndim == 2:
        multi_idx = (indices // train_shape[1], indices % train_shape[1])
    else:
        multi_idx = _unravel_index(indices, train_shape)

    # Step 2: filter — keep indices within train_slices
    mask = torch.ones(indices.numel(), dtype=torch.bool, device=indices.device)
    for dim in range(ndim):
        s = train_slices[dim]
        if s == slice(None):
            continue
        start = s.start or 0
        stop = s.stop or train_shape[dim]
        mask &= (multi_idx[dim] >= start) & (multi_idx[dim] < stop)

    if not mask.any():
        return None

    # Step 3: remap — train_slices space → inf_slices space
    remapped = []
    for dim in range(ndim):
        dim_idx = multi_idx[dim][mask]
        t_start = (train_slices[dim].start or 0) if train_slices[dim] != slice(None) else 0
        i_start = (inf_slices[dim].start or 0) if inf_slices[dim] != slice(None) else 0
        remapped.append(dim_idx - t_start + i_start)

    # Step 4: flatten for inference shard
    new_flat = _ravel_multi_index(remapped, infer_shape)

    return SparseWeightPatch(
        name=patch.name,
        indices=new_flat.to(torch.int32),
        values=values[mask],
    )


def remap_patches_for_operation(
    patches: list[SparseWeightPatch],
    train_shapes: dict[str, tuple[int, ...]],
    infer_shapes: dict[str, tuple[int, ...]],
    operations: list,
) -> list[SparseWeightPatch]:
    """Remap all patches for a set of CommunicationOperations.

    Args:
        patches: sparse patches from delta detection (in training shard space)
        train_shapes: {param_name: train_shard_shape}
        infer_shapes: {param_name: infer_shard_shape}
        operations: list of CommunicationOperation defining overlaps

    Returns:
        List of remapped patches for the target inference rank.
    """
    patch_map = {p.name: p for p in patches}
    remapped = []

    for op in operations:
        name = op.send_shard_meta.name
        if name not in patch_map:
            continue
        if name not in train_shapes:
            continue

        result = remap_delta_indices(
            patch=patch_map[name],
            train_shape=train_shapes[name],
            train_slices=op.train_slices,
            inf_slices=op.inf_slices,
            infer_shape=infer_shapes.get(name, train_shapes[name]),
        )
        if result is not None and result.num_updates > 0:
            remapped.append(result)

    return remapped


def _unravel_index(
    flat_indices: torch.Tensor, shape: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
    """Convert flat indices to multi-dimensional indices."""
    result = []
    remaining = flat_indices
    for dim in range(len(shape) - 1):
        stride = 1
        for s in shape[dim + 1:]:
            stride *= s
        dim_idx = remaining // stride
        remaining = remaining % stride
        result.append(dim_idx)
    result.append(remaining)
    return tuple(result)


def _ravel_multi_index(
    multi_idx: list[torch.Tensor], shape: tuple[int, ...]
) -> torch.Tensor:
    """Convert multi-dimensional indices to flat indices."""
    flat = torch.zeros_like(multi_idx[0], dtype=torch.long)
    stride = 1
    for dim in range(len(shape) - 1, -1, -1):
        flat += multi_idx[dim] * stride
        stride *= shape[dim]
    return flat
