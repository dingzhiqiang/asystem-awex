# Licensed under the Apache License, Version 2.0
"""Delta index remapping for TP mismatch scenarios.

When training TP != inference TP, flat delta indices computed on the training
shard must be filtered and remapped to the inference shard's coordinate space.
Uses AWEX's CommunicationOperation (train_slices / inf_slices) to define
the overlap region for each (train_rank, infer_rank) pair.
"""

from __future__ import annotations

import logging

import torch

from awex.delta.patch import SparseWeightPatch

logger = logging.getLogger(__name__)


def remap_delta_indices(
    patch: SparseWeightPatch,
    train_shape: tuple[int, ...],
    train_slices: tuple[slice, ...],
    inf_slices: tuple[slice, ...],
    infer_shape: tuple[int, ...],
) -> SparseWeightPatch | None:
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
        _assert_unit_step(s, dim, patch.name)
        start = s.start or 0
        stop = s.stop if s.stop is not None else train_shape[dim]
        mask &= (multi_idx[dim] >= start) & (multi_idx[dim] < stop)

    if not mask.any():
        return None

    # Step 3: remap — train_slices space → inf_slices space
    remapped = []
    for dim in range(ndim):
        dim_idx = multi_idx[dim][mask]
        t_start = (train_slices[dim].start or 0) if train_slices[dim] != slice(None) else 0
        i_start = (inf_slices[dim].start or 0) if inf_slices[dim] != slice(None) else 0
        if train_slices[dim] != slice(None):
            _assert_unit_step(train_slices[dim], dim, patch.name)
        if inf_slices[dim] != slice(None):
            _assert_unit_step(inf_slices[dim], dim, patch.name)
        remapped.append(dim_idx - t_start + i_start)

    # Step 4: flatten for inference shard
    new_flat = _ravel_multi_index(remapped, infer_shape)
    _assert_int32_safe(new_flat, infer_shape, patch.name)

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


# int32 flat index ceiling: a single inference shard must stay below 2**31
# elements, otherwise the int32 patch indices overflow.
_MAX_INT32_NUMEL = 2**31


def _assert_unit_step(s: slice, dim: int, name: str) -> None:
    """Remap math assumes contiguous (step==1) slices. AWEX transfer plans only
    build ``slice(start, stop)`` (step is None == 1); a strided slice would
    silently miscompute the overlap, so fail loud instead."""
    if s.step not in (None, 1):
        raise NotImplementedError(
            f"remap_delta_indices: strided slice step={s.step} on dim {dim} of "
            f"'{name}' is unsupported (transfer plans are expected contiguous)."
        )


def _assert_int32_safe(flat: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    numel = 1
    for d in shape:
        numel *= d
    if numel >= _MAX_INT32_NUMEL:
        raise ValueError(
            f"remap_delta_indices: inference shard '{name}' has {numel} elements "
            f">= 2**31; int32 flat indices overflow. Fall back to dense for this param."
        )


def remap_mask_for_op(
    name: str,
    changed_mask: torch.Tensor,
    values_source: torch.Tensor,
    train_shape: tuple[int, ...],
    op,
) -> SparseWeightPatch | None:
    """Per-op entry point for the transport layer.

    Computes the sparse patch for a single CommunicationOperation directly from
    a boolean change mask, without first materializing a full-shard patch. This
    is what the P2P send path calls once per op: the same param's mask is shared
    across all ops (computed once), each op projects its own overlap sub-region
    into inference-shard index space.

    Args:
        name: HF parameter name (``op.send_shard_meta.name``).
        changed_mask: bool tensor, same numel/shape as the train-shard param,
            True where the bf16 element changed this step.
        values_source: the train-shard param tensor (new values are gathered
            from it at the changed positions).
        train_shape: shape of the train-shard tensor.
        op: CommunicationOperation, provides ``train_slices`` / ``inf_slices``
            and ``recv_shard_meta.shape`` (inference-shard shape).

    Returns:
        SparseWeightPatch with indices in the inference-shard flat space and the
        gathered values, or None if no changed element falls in this op's
        overlap region.
    """
    flat_mask = changed_mask.reshape(-1)
    idx = flat_mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return None
    vals = values_source.reshape(-1).index_select(0, idx)
    patch = SparseWeightPatch(name=name, indices=idx.to(torch.int32), values=vals)
    return remap_delta_indices(
        patch=patch,
        train_shape=tuple(train_shape),
        train_slices=op.train_slices,
        inf_slices=op.inf_slices,
        infer_shape=tuple(op.recv_shard_meta.shape),
    )

