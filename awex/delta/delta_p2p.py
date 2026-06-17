# Licensed under the Apache License, Version 2.0
"""Variable-length sparse payload protocol for the colocate P2P transport.

The dense colocate transport (``nccl_stream_batch.update_weights_in_colocate_mode``)
sends a fixed-size sliced tensor per CommunicationOperation. A delta payload is
*variable length* — each op carries only the elements that changed, so the
receiver cannot pre-size its recv buffers from the static transfer plan alone.

Following vLLM PR #40096, we split control plane from data plane:

- **control plane**: a per-op ``nnz`` (number of changed elements) is exchanged
  first (one int32 per op — fixed size, derivable from the plan's op count, so
  it rides the existing symmetric recursive-partition round without breaking the
  deadlock invariant).
- **data plane**: for each op the sender broadcasts ``indices`` (int32, in the
  *inference* shard's flat space — already remapped) and ``values``; the
  receiver, now knowing ``nnz``, pre-allocates and scatters into its live
  parameter view.

This module holds the **pure, CPU-testable** packing/unpacking logic. The actual
NCCL send/recv is wired in ``nccl_stream_batch`` and is not tested here (no
distributed context on CPU); what is tested is the round-trip:
``build_send_patches -> (transmit) -> allocate_recv_buffers -> scatter`` equals a
dense copy, including the ``nnz == 0`` symmetry that is the #1 deadlock hazard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from awex.delta.codec import apply_sparse_patch_
from awex.delta.patch import SparseWeightPatch
from awex.delta.remap import remap_mask_for_op

logger = logging.getLogger(__name__)


@dataclass
class OpDeltaPayload:
    """Per-op delta payload, parallel to one CommunicationOperation.

    ``nnz == 0`` is a valid, required state: an op whose overlap region had no
    changed element this version still occupies a slot (empty index/value
    tensors), so both peers iterate the *same* op sequence and stay symmetric.
    Dropping a zero-nnz op on one side only would desync the recursive-partition
    schedule and hang.
    """

    op: object  # CommunicationOperation
    indices: torch.Tensor  # int32, inference-shard flat space (possibly empty)
    values: torch.Tensor  # param dtype (possibly empty)

    @property
    def nnz(self) -> int:
        return int(self.indices.numel())


def build_send_patches(
    ops: list,
    masks: dict[str, torch.Tensor],
    send_params: dict[str, torch.Tensor],
) -> list[OpDeltaPayload]:
    """Build one OpDeltaPayload per op (in input order), preserving zero-nnz ops.

    Args:
        ops: CommunicationOperations for one peer (send direction).
        masks: ``{hf_name: bool change mask}`` over the train-shard param;
            computed once per param, shared across that param's ops.
        send_params: ``{hf_name: train-shard tensor}`` (value source).

    Returns:
        List parallel to ``ops``; every op gets a payload, empty if no overlap.
    """
    payloads: list[OpDeltaPayload] = []
    for op in ops:
        name = op.send_shard_meta.name
        mask = masks.get(name)
        src = send_params.get(name)
        patch: SparseWeightPatch | None = None
        if mask is not None and src is not None:
            patch = remap_mask_for_op(name, mask, src, tuple(src.shape), op)
        if patch is None:
            # zero-nnz slot: keep the op so both peers stay symmetric.
            dtype = send_params[name].dtype if name in send_params else torch.bfloat16
            payloads.append(
                OpDeltaPayload(
                    op=op,
                    indices=torch.empty(0, dtype=torch.int32),
                    values=torch.empty(0, dtype=dtype),
                )
            )
        else:
            payloads.append(
                OpDeltaPayload(op=op, indices=patch.indices, values=patch.values)
            )
    return payloads


def nnz_vector(payloads: list[OpDeltaPayload]) -> torch.Tensor:
    """Control-plane vector: one int32 nnz per op, in op order (fixed size)."""
    return torch.tensor([p.nnz for p in payloads], dtype=torch.int32)


def allocate_recv_buffers(
    nnz_list: list[int],
    value_dtype: torch.dtype,
    device="cpu",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Receiver pre-allocates (idx, val) buffers from the exchanged nnz vector.

    One (possibly empty) buffer pair per op, in the same order the sender packs,
    so the data-plane recv stays aligned and symmetric.
    """
    buffers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for nnz in nnz_list:
        buffers.append(
            (
                torch.empty(nnz, dtype=torch.int32, device=device),
                torch.empty(nnz, dtype=value_dtype, device=device),
            )
        )
    return buffers


@torch.no_grad()
def scatter_recv_into(
    recv_params: dict[str, torch.Tensor],
    ops: list,
    recv_buffers: list[tuple[torch.Tensor, torch.Tensor]],
    slice_fn,
) -> int:
    """Scatter received (idx, val) buffers into the live inference params.

    Args:
        recv_params: ``{hf_name: inference param}`` (write-through view).
        ops: CommunicationOperations (recv direction), parallel to recv_buffers.
        recv_buffers: (idx, val) per op, as filled by the data-plane recv.
        slice_fn: ``slice_fn(tensor, op) -> inference-shard view`` (the op's
            ``inf_slices`` region of the param; may be non-contiguous —
            ``apply_sparse_patch_`` handles that).

    Returns:
        Number of ops that actually applied a non-empty patch (for logging).
    """
    applied = 0
    for op, (idx, val) in zip(ops, recv_buffers, strict=True):
        if idx.numel() == 0:
            continue  # zero-nnz op: nothing to write, but it still had a slot
        name = op.recv_shard_meta.name
        target = slice_fn(recv_params[name], op)
        apply_sparse_patch_(target, idx, val)
        applied += 1
    return applied
