# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor

import torch
import torch.distributed as dist

from awex import logging
from awex.transfer.nccl_comm import (
    detect_hang,
    execute_tensors_to_copy,
    validate_rank_mappings,
)
from awex.transfer.transfer_plan import slice_tensor
from awex.util import device as device_util

logger = logging.getLogger(__name__)
hang_detector = ThreadPoolExecutor(max_workers=1)

_EXPERT_PARAM_MARKER = ".mlp.experts."


def _is_expert_transfer_op(op) -> bool:
    """Return whether an op transfers a canonical routed-expert parameter."""
    send_meta = getattr(op, "send_shard_meta", None)
    recv_meta = getattr(op, "recv_shard_meta", None)
    send_name = getattr(send_meta, "name", "")
    recv_name = getattr(recv_meta, "name", "")
    return (
        _EXPERT_PARAM_MARKER in send_name
        and _EXPERT_PARAM_MARKER in recv_name
        and send_name == recv_name
    )


def _wire_dtype(op, fallback: torch.dtype) -> torch.dtype:
    """Resolve the dtype posted by the receiver for a transfer operation."""
    recv_meta = getattr(op, "recv_shard_meta", None)
    return getattr(recv_meta, "dtype", None) or fallback


def _partition_expert_p2p_entries(
    entries: list[tuple[object, torch.Tensor]], pack_experts: bool
) -> tuple[
    list[tuple[object, torch.Tensor]],
    list[tuple[torch.dtype, list[tuple[object, torch.Tensor]]]],
]:
    """Split entries into direct ops and deterministic expert dtype buckets."""
    if not pack_experts:
        return entries, []
    direct_entries = []
    expert_buckets = {}
    for op, tensor in entries:
        if _is_expert_transfer_op(op):
            wire_dtype = _wire_dtype(op, tensor.dtype)
            expert_buckets.setdefault(wire_dtype, []).append((op, tensor))
        else:
            direct_entries.append((op, tensor))
    packed_entries = [
        (wire_dtype, expert_buckets[wire_dtype])
        for wire_dtype in sorted(expert_buckets, key=str)
    ]
    return direct_entries, packed_entries


def _packed_recv_staging_bytes(ops: list[object]) -> int:
    """Return staging bytes needed to pack the expert operations in ``ops``."""
    total_bytes = 0
    for op in ops:
        if not _is_expert_transfer_op(op):
            continue
        send_meta = getattr(op, "send_shard_meta", None)
        fallback_dtype = getattr(send_meta, "dtype", torch.float32)
        wire_dtype = _wire_dtype(op, fallback_dtype)
        numel = math.prod(getattr(op, "overlap_shape", ()) or ())
        total_bytes += numel * torch.empty((), dtype=wire_dtype).element_size()
    return total_bytes


@torch.no_grad()
def _pack_p2p_send_tensors(
    tensors: list[torch.Tensor], wire_dtype: torch.dtype
) -> torch.Tensor:
    """Copy tensors into one dense, independent P2P send buffer."""
    if not tensors:
        raise ValueError("Cannot pack an empty tensor list")
    device = tensors[0].device
    total_numel = sum(tensor.numel() for tensor in tensors)
    packed = torch.empty(total_numel, dtype=wire_dtype, device=device)
    offset = 0
    for tensor in tensors:
        if tensor.device != device:
            raise ValueError(
                f"Packed P2P tensors must share a device: {device} != {tensor.device}"
            )
        numel = tensor.numel()
        packed.narrow(0, offset, numel).view(tensor.shape).copy_(tensor)
        offset += numel
    return packed


def _prepare_packed_p2p_recv_tensor(
    tensors: list[torch.Tensor], wire_dtype: torch.dtype
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Allocate one packed recv buffer and views to copy into target tensors."""
    if not tensors:
        raise ValueError("Cannot prepare an empty packed recv tensor list")
    device = tensors[0].device
    total_numel = sum(tensor.numel() for tensor in tensors)
    packed = torch.empty(total_numel, dtype=wire_dtype, device=device)
    copyback_pairs = []
    offset = 0
    for tensor in tensors:
        if tensor.device != device:
            raise ValueError(
                f"Packed P2P tensors must share a device: {device} != {tensor.device}"
            )
        numel = tensor.numel()
        packed_view = packed.narrow(0, offset, numel).view(tensor.shape)
        copyback_pairs.append((tensor, packed_view))
        offset += numel
    return packed, copyback_pairs


def _clone_p2p_send_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a dense tensor suitable for torch.distributed P2P send."""
    return tensor.clone(memory_format=torch.contiguous_format)


def _prepare_p2p_recv_tensor(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Return a dense recv buffer and an optional copyback pair."""
    if tensor.is_contiguous():
        return tensor, None
    recv_buffer = torch.empty(
        tuple(tensor.shape),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return recv_buffer, (tensor, recv_buffer)


@torch.no_grad()
def _sync_p2p_recv_tensor_pairs(
    recv_tensor_pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    if not recv_tensor_pairs:
        return
    for original_tensor, recv_buffer in recv_tensor_pairs:
        original_tensor.copy_(recv_buffer)
    recv_tensor_pairs.clear()


class NcclColocateStreamBatchTransport:
    MAX_STREAMS = 64

    def __init__(self, transfer_rank, world_size):
        self.transfer_rank = transfer_rank
        self.world_size = world_size
        # Initialize a fixed pool of device streams (CUDA/NPU)
        self._stream_pool = [
            device_util.create_stream()
            for _ in range(min(self.MAX_STREAMS, world_size))
        ]

    def update_weights_in_colocate_mode(
        self,
        train_to_infer_device_mapping,
        infer_to_train_device_mapping,
        transfer_rank,
        rank_coordinate,
        world_size,
        send_transfer_plan,
        recv_transfer_plan,
        weights_update_group,
        send_parameters,
        recv_parameters,
        *,
        step_id=-1,
        async_op=True,
        **kwargs,
    ):
        logger.info("Using RECURSIVE PARTITION batch_isend_irecv with O(log N) rounds")
        task_id = f"{rank_coordinate}-{step_id}"
        validate_rank_mappings(
            train_to_infer_device_mapping, infer_to_train_device_mapping
        )
        start_time = time.time()

        # Get send/recv operations dict
        send_ops = dict(send_transfer_plan.operations)
        recv_ops = dict(recv_transfer_plan.operations)
        num_sends = sum(len(ops) for ops in send_ops.values())
        num_recvs = sum(len(ops) for ops in recv_ops.values())
        logger.info(
            f"Start to execute weights update for {task_id}, "
            f"num_sends {num_sends}, num_recvs {num_recvs}"
        )

        chunk_mb = int(os.environ.get("AWEX_CHUNK_MB", "0") or "0")

        if chunk_mb > 0:
            self._run_chunked(
                task_id=task_id,
                step_id=step_id,
                train_to_infer_device_mapping=train_to_infer_device_mapping,
                infer_to_train_device_mapping=infer_to_train_device_mapping,
                transfer_rank=transfer_rank,
                rank_coordinate=rank_coordinate,
                world_size=world_size,
                send_ops=send_ops,
                recv_ops=recv_ops,
                recv_transfer_plan=recv_transfer_plan,
                weights_update_group=weights_update_group,
                send_parameters=send_parameters,
                recv_parameters=recv_parameters,
                async_op=async_op,
                chunk_bytes=chunk_mb * 1024 * 1024,
            )
            duration = time.time() - start_time
            logger.info(
                f"Finished CHUNKED weights update for {task_id}, took {duration:.4f}s "
                f"(chunk_mb={chunk_mb})"
            )
            return

        # === LEGACY PATH (one-shot clone-then-transfer) ===
        # Build P2P operations with sliced tensors
        all_send_p2p_ops = {}  # peer_rank -> List[(plan_op, p2p_op)]
        all_recv_p2p_ops = {}  # peer_rank -> List[(plan_op, p2p_op)]
        tensors_to_copy = []
        recv_tensor_pairs = []
        train_slice_context = {}

        # Process send operations
        for peer_rank, ops in send_ops.items():
            # Map training rank to inference rank in colocate mode
            mapped_peer_rank = train_to_infer_device_mapping.get(peer_rank, peer_rank)
            if mapped_peer_rank == transfer_rank:
                # Self-copy operations
                for op in ops:
                    send_tensor = send_parameters[op.send_shard_meta.name]
                    tensor_sliced = slice_tensor(
                        send_tensor, op, True, slice_context=train_slice_context
                    )
                    tensors_to_copy.append(tensor_sliced)
            else:
                # P2P send operations
                p2p_ops = []
                for op in ops:
                    send_tensor = send_parameters[op.send_shard_meta.name]
                    tensor_sliced = slice_tensor(
                        send_tensor, op, True, slice_context=train_slice_context
                    )
                    # Use mapped inference rank for P2P operation
                    recv_rank = train_to_infer_device_mapping.get(
                        op.recv_rank, op.recv_rank
                    )
                    cloned = _clone_p2p_send_tensor(tensor_sliced)
                    # Wire-size parity with the receiver's dtype (see the
                    # chunked path / Problem 69: bf16 gate.weight into an fp32
                    # recv slot wedges the receiver forever).
                    recv_dtype = getattr(op.recv_shard_meta, "dtype", None)
                    if recv_dtype is not None and cloned.dtype != recv_dtype:
                        cloned = cloned.to(recv_dtype)
                    if not cloned.is_contiguous():
                        cloned = cloned.contiguous()
                    p2p_op = dist.P2POp(
                        dist.isend if async_op else dist.send,
                        cloned,
                        recv_rank,
                        group=weights_update_group,
                    )
                    p2p_ops.append((op, p2p_op))
                all_send_p2p_ops[mapped_peer_rank] = p2p_ops

        # Process recv operations
        for send_rank, ops in recv_ops.items():
            recv_from_rank = train_to_infer_device_mapping[send_rank]
            if recv_from_rank == transfer_rank:
                # Skip self-recv (handled by tensors_to_copy)
                continue
            p2p_ops = []
            for op in ops:
                recv_tensor = recv_parameters[op.recv_shard_meta.name]
                tensor_sliced = slice_tensor(recv_tensor, op, False)
                tensor_sliced, copyback_pair = _prepare_p2p_recv_tensor(tensor_sliced)
                if copyback_pair is not None:
                    recv_tensor_pairs.append(copyback_pair)
                p2p_op = dist.P2POp(
                    dist.irecv if async_op else dist.recv,
                    tensor_sliced,
                    recv_from_rank,
                    group=weights_update_group,
                )
                p2p_ops.append((op, p2p_op))
            all_recv_p2p_ops[recv_from_rank] = p2p_ops

        # Execute self-copy operations
        if len(tensors_to_copy) > 0:
            send_rank = infer_to_train_device_mapping[transfer_rank]
            execute_tensors_to_copy(
                tensors_to_copy,
                recv_transfer_plan.operations[send_rank],
                recv_parameters,
                f"tensor copy for {task_id}",
            )
        else:
            logger.info(f"No tensors to copy for {task_id}")

        future = Future()
        total_send_ops = sum(len(ops) for ops in all_send_p2p_ops.values())
        total_recv_ops = sum(len(ops) for ops in all_recv_p2p_ops.values())
        msg = f"[{os.getpid()}] execute {total_send_ops} sends {total_recv_ops} recvs with recursive partition for {task_id}"
        hang_detector.submit(detect_hang, future, msg, [], timeout=60)

        # Recursive-partition butterfly with per-peer batch_isend_irecv (see
        # _execute_ops_concurrent). The phase structure is symmetric and the
        # data layer is verified fully consistent; the earlier deadlock was
        # purely from submitting a whole half's ops in one batch. Issuing one
        # batch per peer caps in-flight P2P at O(1) peer and stays
        # deadlock-free.
        self.execute_recursive_partition_stream_transfer(
            transfer_rank,
            world_size,
            all_send_p2p_ops,
            all_recv_p2p_ops,
            weights_update_group,
            rank_coordinate,
            step_id,
        )

        device_util.synchronize()
        if recv_tensor_pairs:
            logger.info(
                f"Syncing {len(recv_tensor_pairs)} non-contiguous recv buffers for {task_id}"
            )
            _sync_p2p_recv_tensor_pairs(recv_tensor_pairs)
            device_util.synchronize()
        future.set_result(True)
        duration = time.time() - start_time
        logger.info(
            f"Finished executing weights update for {task_id}, took {duration:.4f} seconds"
        )

    def execute_recursive_partition_stream_transfer(
        self,
        transfer_rank,
        world_size,
        all_send_p2p_ops,  # Dict[peer_rank] -> List[(plan_op, p2p_op)]
        all_recv_p2p_ops,  # Dict[peer_rank] -> List[(plan_op, p2p_op)]
        weights_update_group,
        rank_coordinate,
        step_id,
        *,
        synchronize_at_end=True,
    ):
        """
        Execute P2P transfer using recursive partition algorithm.

        Algorithm:
        - Round 1: partition_size=world_size, split into [0, world_size/2) and [world_size/2, world_size)
          - First half sends to second half
          - Second half recvs from first half
          - First half recvs from second half
          - Second half sends to first half

        - Round 2: partition_size=world_size/2, operate on each half independently
        - ...
        - Continue until partition_size=2

        Total rounds: log2(world_size)
        Each rank sends/recvs to/from ALL ranks in the other half of its partition.
        """
        num_rounds = int(math.log2(world_size))
        prefix = f"[{os.getpid()}] [{rank_coordinate}] [step {step_id}]"
        start_time = time.time()
        logger.info(
            f"{prefix} Starting recursive partition transfer with {num_rounds} rounds"
        )
        for round_idx in range(num_rounds):
            partition_size = world_size // (2**round_idx)
            half = partition_size // 2

            # Determine my partition base (which partition I'm in)
            partition_base = (transfer_rank // partition_size) * partition_size
            partition_end = partition_base + partition_size
            offset_in_partition = transfer_rank - partition_base

            # Determine if I'm in first half or second half of my partition
            in_first_half = offset_in_partition < half
            # Determine the range of ranks in the other half
            if in_first_half:
                other_half_start = partition_base + half
                other_half_end = partition_end
            else:
                other_half_start = partition_base
                other_half_end = partition_base + half
            logger.info(
                f"{prefix} Round {round_idx}: partition_size={partition_size}, "
                f"partition=[{partition_base}, {partition_end}), half={half}, "
                f"in_first_half={in_first_half}, other_half=[{other_half_start}, {other_half_end})"
            )

            round_start = time.time()
            # === PHASE 1: First half sends to second half, second half receives from first half ===
            if in_first_half:
                # Execute all send operations to ranks in the other half with concurrent execution
                num_ops = self._execute_ops_concurrent(
                    all_send_p2p_ops, range(other_half_start, other_half_end)
                )
            else:
                # Execute all recv operations from ranks in the other half with concurrent execution
                num_ops = self._execute_ops_concurrent(
                    all_recv_p2p_ops, range(other_half_start, other_half_end)
                )
            logger.info(
                f"{prefix} Round {round_idx} Phase 1: enqueued {num_ops} "
                f"{'sends' if in_first_half else 'recvs'}"
            )
            # === PHASE 2: First half receives from second half, second half sends to first half ===
            if in_first_half:
                # Execute all recv operations from ranks in the other half with concurrent execution
                num_ops2 = self._execute_ops_concurrent(
                    all_recv_p2p_ops, range(other_half_start, other_half_end)
                )
            else:
                # Execute all send operations to ranks in the other half with concurrent execution
                num_ops2 = self._execute_ops_concurrent(
                    all_send_p2p_ops, range(other_half_start, other_half_end)
                )
            logger.info(
                f"{prefix} Round {round_idx} Phase 2: enqueued {num_ops2} "
                f"{'recvs' if in_first_half else 'sends'}"
            )
            round_duration = time.time() - round_start
            logger.info(
                f"[{os.getpid()}] Round {round_idx} completed: "
                f"phase1={num_ops} ops, phase2={num_ops2} ops, "
                f"took {round_duration:.4f}s"
            )
        # The chunked caller synchronizes once at its chunk boundary after
        # this function returns.  Avoid doing the same device-wide sync here
        # and immediately again in _run_chunked; the legacy path keeps the
        # original default for safety.
        if synchronize_at_end:
            device_util.synchronize()
        duration = time.time() - start_time
        logger.info(f"{prefix} All {num_rounds} rounds completed in {duration:.4f}s")

    def _execute_ops_concurrent(self, ops_dict, peer_ranks):
        """
        Execute ops from multiple peers with interleaved execution for better concurrency.

        Instead of executing all ops for one peer sequentially (peer1_all_ops, peer2_all_ops, ...),
        this method interleaves operations in a round-robin fashion (peer1_op1, peer2_op1, ...,
        peer1_op2, peer2_op2, ...). This allows operations from different peers to overlap and
        execute concurrently on the GPU.

        Each peer rank consistently uses the same CUDA stream to maintain ordering within
        that peer's operations, while different peers use different streams (up to max)
        for concurrent execution.

        Args:
            ops_dict: Dictionary mapping peer_rank to list of (plan_op, p2p_op) tuples
            peer_ranks: Range or iterable of peer ranks to process

        Returns:
            Total number of ops executed
        """
        # Per-peer batch_isend_irecv. Submitting a WHOLE half's ops in one
        # batch_isend_irecv (the previous behaviour) deadlocks at 32-rank /
        # PP4->PP1 asymmetric scale: round 0's other-half has 16 peers and
        # thousands of ops, and flooding NCCL with that many concurrent P2P
        # channels exhausts them so the GPU drain never completes (data layer
        # verified fully symmetric — the failure is purely runtime concurrency
        # scale). Instead we walk peers in ascending rank order and issue ONE
        # batch_isend_irecv per peer, capping in-flight P2P at O(1) peer.
        #
        # This is deadlock-free because a recursive-partition phase is
        # single-direction: in phase 1 every first-half rank only SENDS and
        # every second-half rank only RECVS (phase 2 is the mirror). A
        # (sender, receiver) pair's per-peer batch is matched by NCCL group
        # on (src, dst, group); serializing peers cannot form a wait cycle
        # since no rank both sends and receives within the same phase. (This
        # is exactly why recursive partition's symmetric phases are safe and
        # the circle-shift directed ring was not.)
        #
        # Both sides must walk peers in the SAME (ascending) order so the
        # k-th batch on a sender pairs with the corresponding recv on the
        # receiver. peer_ranks is already an ascending range here.
        trace = os.environ.get("AWEX_P2P_TRACE", "").strip() in ("1", "true", "True")
        my_rank = self.transfer_rank
        sync_peer_groups = os.environ.get("AWEX_P2P_GROUP_SYNC", "1").strip() in (
            "1",
            "true",
            "True",
        )
        try:
            peer_group_size = max(
                1, int(os.environ.get("AWEX_P2P_PEER_GROUP_SIZE", "1") or "1")
            )
        except ValueError:
            peer_group_size = 1

        # Keep the group boundaries identical on every rank.  In particular,
        # do not group only ``ops_dict``'s non-empty peers: senders and
        # receivers can have different sparse peer sets.  Fixed peer-rank
        # ranges preserve the same FIFO ordering while allowing a small,
        # bounded amount of overlap.  The default remains one peer per batch,
        # which is the deadlock-safe behavior established in 8bc8fc1.
        peer_ranks = list(peer_ranks)
        total_ops = 0
        for group_start in range(0, len(peer_ranks), peer_group_size):
            peer_group = peer_ranks[group_start : group_start + peer_group_size]
            grouped_ops = []
            grouped_peer_counts = []
            for peer_rank in peer_group:
                ops = ops_dict.get(peer_rank)
                if not ops:
                    grouped_peer_counts.append((peer_rank, 0))
                    continue
                p2p_ops = [p2p_op for _, p2p_op in ops]
                if not p2p_ops:
                    grouped_peer_counts.append((peer_rank, 0))
                    continue
                grouped_peer_counts.append((peer_rank, len(p2p_ops)))
                grouped_ops.extend(p2p_ops)

            if not grouped_ops:
                continue
            if trace:
                logger.info(
                    f"[P2P-TRACE rank={my_rank}] peers={peer_group} "
                    f"counts={grouped_peer_counts} nops={len(grouped_ops)} "
                    f"group_size={peer_group_size} -> batch_isend_irecv (pre-wait)"
                )
            works = dist.batch_isend_irecv(grouped_ops)
            for work in works:
                work.wait()
            # ``work.wait()`` still drains the NCCL work handle.  The extra
            # device-wide synchronize is retained as the default safety mode,
            # but can be disabled for performance experiments; the chunk
            # boundary below always performs a final device synchronize.
            if (
                sync_peer_groups
                and hasattr(torch, "cuda")
                and torch.cuda.is_available()
            ):
                torch.cuda.synchronize()
            if trace:
                logger.info(
                    f"[P2P-TRACE rank={my_rank}] peers={peer_group} "
                    f"synchronize={'done' if sync_peer_groups else 'skipped'}"
                )
            total_ops += len(grouped_ops)
        return total_ops

    def _run_chunked(
        self,
        *,
        task_id,
        step_id,
        train_to_infer_device_mapping,
        infer_to_train_device_mapping,
        transfer_rank,
        rank_coordinate,
        world_size,
        send_ops,
        recv_ops,
        recv_transfer_plan,
        weights_update_group,
        send_parameters,
        recv_parameters,
        async_op,
        chunk_bytes,
    ):
        """Chunked send/recv for AWEX colocation.

        Cross-rank determinism: chunk N takes ops[N*step:(N+1)*step] from each
        peer's per-peer ops list. plan_builder.build_local_transfer_plan
        already sorts each peer's ops by (send_shard_meta.name, send_offset,
        recv_offset) (transfer_plan.py:571), so rank A's send_ops[B] and rank
        B's recv_ops[A] are aligned index-by-index. Same step_size on every
        rank means matching send/recv pairs always land in the same chunk.
        NCCL P2P pairs FIFO within (group, src, dst) so this preserves
        protocol semantics.

        step_size is derived from chunk_bytes by sampling the per-op nbytes
        from a representative op so that one chunk's clones approach but do
        not exceed chunk_bytes.

        Local self-copy (tensors_to_copy) and self-recv-from-other-trains do
        not consume clone memory and are emitted once up front.

        ``AWEX_PACK_EXPERT_P2P=1`` preserves the canonical per-expert plan but
        coalesces each peer/chunk/wire-dtype group into one NCCL message. This
        isolates P2P launch/work overhead from converter and sharding changes.
        The receiver stages packed messages and copies their views into the
        original SGLang parameter slices after the chunk has completed.
        """
        train_slice_context = {}

        local_train_rank = infer_to_train_device_mapping.get(transfer_rank)
        tensors_to_copy = []
        local_self_recv_collected = []

        send_per_peer = {}
        for peer_rank, ops in send_ops.items():
            mapped_peer_rank = train_to_infer_device_mapping.get(peer_rank, peer_rank)
            if mapped_peer_rank == transfer_rank:
                for op in ops:
                    op_send_rank = getattr(op, "send_rank", None)
                    if (
                        local_train_rank is not None
                        and op_send_rank is not None
                        and op_send_rank != local_train_rank
                    ):
                        local_self_recv_collected.append(op)
                    else:
                        if op.send_shard_meta.name not in send_parameters:
                            raise KeyError(op.send_shard_meta.name)
                        send_tensor = send_parameters[op.send_shard_meta.name]
                        tensor_sliced = slice_tensor(
                            send_tensor, op, True, slice_context=train_slice_context
                        )
                        tensors_to_copy.append(tensor_sliced)
            else:
                missing = [
                    op.send_shard_meta.name
                    for op in ops
                    if op.send_shard_meta.name not in send_parameters
                ]
                if missing:
                    raise KeyError(missing[0])
                send_per_peer[mapped_peer_rank] = list(ops)

        recv_per_peer = {}
        for send_rank, ops in recv_ops.items():
            recv_from_rank = train_to_infer_device_mapping[send_rank]
            if recv_from_rank == transfer_rank:
                continue
            recv_per_peer[recv_from_rank] = list(ops)

        local_self_recv_built = []
        for op in local_self_recv_collected:
            recv_buf = recv_parameters[op.recv_shard_meta.name]
            recv_sliced = slice_tensor(recv_buf, op, False)
            recv_sliced, copyback_pair = _prepare_p2p_recv_tensor(recv_sliced)
            actual_send_rank = train_to_infer_device_mapping.get(
                op.send_rank, op.send_rank
            )
            p2p_op = dist.P2POp(
                dist.irecv,
                recv_sliced,
                actual_send_rank,
                group=weights_update_group,
            )
            local_self_recv_built.append((actual_send_rank, op, p2p_op, copyback_pair))

        pack_expert_p2p = os.environ.get("AWEX_PACK_EXPERT_P2P", "0").strip() in (
            "1",
            "true",
            "True",
        )
        # The defensive local-self receive path posts operations outside the
        # normal per-peer chunk index. Packing only one side would change the
        # P2P FIFO, so disable packing globally if any rank needs that path.
        pack_safe = pack_expert_p2p and not local_self_recv_built
        if dist.is_initialized():
            pack_safe_tensor = torch.tensor(
                [int(pack_safe)],
                device=device_util.get_torch_device(),
                dtype=torch.int32,
            )
            dist.all_reduce(
                pack_safe_tensor,
                op=dist.ReduceOp.MIN,
                group=weights_update_group,
            )
            pack_safe = bool(pack_safe_tensor.item())
        if pack_expert_p2p and not pack_safe:
            logger.warning(
                f"[CHUNKED {task_id}] Expert P2P packing disabled because at "
                "least one rank requires the local-self receive fallback"
            )
        pack_expert_p2p = pack_safe
        logger.info(f"[CHUNKED {task_id}] expert_p2p_packing={pack_expert_p2p}")

        if len(tensors_to_copy) > 0:
            send_rank_for_self = infer_to_train_device_mapping[transfer_rank]
            execute_tensors_to_copy(
                tensors_to_copy,
                recv_transfer_plan.operations[send_rank_for_self],
                recv_parameters,
                f"tensor copy for {task_id}",
            )
        else:
            logger.info(f"No tensors to copy for {task_id}")

        sample_op = None
        for peer_rank in sorted(send_per_peer.keys()):
            ops = send_per_peer[peer_rank]
            if ops:
                sample_op = ops[0]
                break
        if sample_op is None:
            for peer_rank in sorted(recv_per_peer.keys()):
                ops = recv_per_peer[peer_rank]
                if ops:
                    sample_op = ops[0]
                    break

        if sample_op is None:
            step_size = 1
        else:
            shape = sample_op.send_shard_meta.shape
            elem_size = 2
            try:
                from awex.util.tensor_util import dtype_to_size as _dtype_size

                elem_size = _dtype_size(sample_op.send_shard_meta.dtype)
            except Exception:
                pass
            sliced_numel = 1
            try:
                for s in sample_op.train_slices or []:
                    span = s.stop - s.start if s.stop is not None else 0
                    sliced_numel *= max(span, 1)
            except Exception:
                sliced_numel = 1
                for d in shape:
                    sliced_numel *= d
            per_op_bytes = max(sliced_numel * elem_size, 1)
            step_size = max(1, chunk_bytes // per_op_bytes)
            logger.info(
                f"[CHUNKED {task_id}] local sample shape={shape} per_op_bytes={per_op_bytes} "
                f"chunk_bytes={chunk_bytes} step_size_local={step_size}"
            )

        env_force = os.environ.get("AWEX_CHUNK_OPS", "").strip()
        if env_force:
            try:
                forced = max(1, int(env_force))
                step_size = forced
                logger.info(f"[CHUNKED {task_id}] AWEX_CHUNK_OPS override={forced}")
            except ValueError:
                pass
        else:
            try:
                if dist.is_initialized():
                    t = torch.tensor(
                        [int(step_size)],
                        device=device_util.get_torch_device(),
                        dtype=torch.int64,
                    )
                    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=weights_update_group)
                    new_step = int(t.item())
                    if new_step != step_size:
                        logger.info(
                            f"[CHUNKED {task_id}] step_size aligned via all_reduce "
                            f"local={step_size} -> global_min={new_step}"
                        )
                    step_size = max(1, new_step)
            except Exception as e:
                logger.warning(
                    f"[CHUNKED {task_id}] step_size all_reduce failed: {e}; "
                    f"using local={step_size} (risk of cross-rank chunk drift)"
                )

        max_send_len = max((len(v) for v in send_per_peer.values()), default=0)
        max_recv_len = max((len(v) for v in recv_per_peer.values()), default=0)
        n_chunks = max(
            1,
            (max(max_send_len, max_recv_len) + step_size - 1) // step_size,
        )
        # n_chunks must be globally consistent; otherwise ranks with fewer
        # chunks exit the loop early and the others hang in batch_isend_irecv
        # waiting for peers that already left. step_size is already MIN-reduced
        # above, but n_chunks depends on per-rank max_send/recv lengths which
        # diverge across ranks. Take MAX to ensure every rank runs the same
        # number of chunk iterations (empty chunks are no-ops).
        try:
            if dist.is_initialized():
                t = torch.tensor(
                    [int(n_chunks)],
                    device=device_util.get_torch_device(),
                    dtype=torch.int64,
                )
                dist.all_reduce(t, op=dist.ReduceOp.MAX, group=weights_update_group)
                new_n = int(t.item())
                if new_n != n_chunks:
                    logger.info(
                        f"[CHUNKED {task_id}] n_chunks aligned via all_reduce "
                        f"local={n_chunks} -> global_max={new_n}"
                    )
                    n_chunks = new_n
        except Exception as e:
            logger.warning(
                f"[CHUNKED {task_id}] n_chunks all_reduce failed: {e}; "
                f"using local={n_chunks} (risk of cross-rank chunk drift / hang)"
            )
        logger.info(
            f"[CHUNKED {task_id}] n_chunks={n_chunks} step_size={step_size} "
            f"max_send_per_peer={max_send_len} max_recv_per_peer={max_recv_len}"
        )

        # Packed receive buffers remain alive for every peer until the chunk
        # finishes and its views have been copied back.  Bound their combined
        # peak before allocating any of them.  If one rank cannot pack safely,
        # every rank must use the direct protocol to keep P2P FIFO identical.
        if pack_expert_p2p:
            local_max_recv_stage_bytes = max(
                (
                    sum(
                        _packed_recv_staging_bytes(ops[start : start + step_size])
                        for ops in recv_per_peer.values()
                    )
                    for start in range(0, n_chunks * step_size, step_size)
                ),
                default=0,
            )
            cap_mb_raw = os.environ.get("AWEX_PACK_RECV_STAGING_MB", "").strip()
            try:
                recv_stage_cap_bytes = (
                    max(1, int(float(cap_mb_raw) * 1024 * 1024))
                    if cap_mb_raw
                    else chunk_bytes
                )
            except ValueError:
                logger.warning(
                    f"[CHUNKED {task_id}] Invalid AWEX_PACK_RECV_STAGING_MB="
                    f"{cap_mb_raw!r}; using chunk_bytes={chunk_bytes}"
                )
                recv_stage_cap_bytes = chunk_bytes

            global_max_recv_stage_bytes = local_max_recv_stage_bytes
            global_recv_stage_cap_bytes = recv_stage_cap_bytes
            if dist.is_initialized():
                stage_and_cap = torch.tensor(
                    [local_max_recv_stage_bytes, recv_stage_cap_bytes],
                    device=device_util.get_torch_device(),
                    dtype=torch.int64,
                )
                dist.all_reduce(
                    stage_and_cap[:1],
                    op=dist.ReduceOp.MAX,
                    group=weights_update_group,
                )
                dist.all_reduce(
                    stage_and_cap[1:],
                    op=dist.ReduceOp.MIN,
                    group=weights_update_group,
                )
                global_max_recv_stage_bytes = int(stage_and_cap[0].item())
                global_recv_stage_cap_bytes = int(stage_and_cap[1].item())

            if global_max_recv_stage_bytes > global_recv_stage_cap_bytes:
                pack_expert_p2p = False
                logger.warning(
                    f"[CHUNKED {task_id}] Expert P2P packing disabled before "
                    "allocation: global_recv_stage_mb="
                    f"{global_max_recv_stage_bytes / 1024 / 1024:.1f} exceeds "
                    "cap_mb="
                    f"{global_recv_stage_cap_bytes / 1024 / 1024:.1f}"
                )
            else:
                logger.info(
                    f"[CHUNKED {task_id}] packed recv staging preflight: "
                    "global_peak_mb="
                    f"{global_max_recv_stage_bytes / 1024 / 1024:.1f} "
                    f"cap_mb={global_recv_stage_cap_bytes / 1024 / 1024:.1f}"
                )

        total_clone_bytes = 0
        total_send_logical_expert_ops = 0
        total_send_packed_expert_ops = 0
        total_recv_logical_expert_ops = 0
        total_recv_packed_expert_ops = 0
        total_recv_staging_bytes = 0
        # Preserve the established per-chunk cache trim by default. Packed
        # staging adds another large temporary allocation, so changing this
        # safety behavior should remain an explicit experiment.
        trim_cuda_cache = os.environ.get("AWEX_CHUNK_EMPTY_CACHE", "1").strip() in (
            "1",
            "true",
            "True",
        )

        for chunk_idx in range(n_chunks):
            start = chunk_idx * step_size
            end = start + step_size

            logger.warning(
                f"[CHUNKED-DIAG {task_id}] chunk_idx={chunk_idx}/{n_chunks} ENTER "
                f"slice=[{start},{end})"
            )
            chunk_send_p2p_ops = {}
            chunk_recv_p2p_ops = {}
            chunk_recv_tensor_pairs = []
            chunk_clone_bytes = 0
            chunk_send_logical_expert_ops = 0
            chunk_send_packed_expert_ops = 0
            chunk_recv_logical_expert_ops = 0
            chunk_recv_packed_expert_ops = 0
            chunk_recv_staging_bytes = 0

            for mapped_peer_rank, ops in send_per_peer.items():
                sub = ops[start:end]
                if not sub:
                    continue
                p2p_ops = []
                transfer_entries = []
                for op in sub:
                    send_tensor = send_parameters[op.send_shard_meta.name]
                    tensor_sliced = slice_tensor(
                        send_tensor, op, True, slice_context=train_slice_context
                    )
                    recv_rank = train_to_infer_device_mapping.get(
                        op.recv_rank, op.recv_rank
                    )
                    if recv_rank != mapped_peer_rank:
                        raise ValueError(
                            "Transfer-plan peer does not match operation target: "
                            f"mapped_peer={mapped_peer_rank}, target={recv_rank}"
                        )
                    transfer_entries.append((op, tensor_sliced))

                direct_entries, packed_expert_entries = _partition_expert_p2p_entries(
                    transfer_entries, pack_expert_p2p
                )
                for op, tensor_sliced in direct_entries:
                    wire_dtype = _wire_dtype(op, tensor_sliced.dtype)
                    cloned = _clone_p2p_send_tensor(tensor_sliced)
                    # Wire-size parity: the receiver posts irecv with ITS shard
                    # dtype. 961 plan ops (mlp.gate.weight, 124 edges) are bf16
                    # on the train side but fp32 on the sglang side; sending
                    # bf16 bytes into an fp32-sized recv leaves the receiver
                    # waiting forever (deterministic chunk-7 deadlock,
                    # Problem 69). Cast the clone to the receiver's dtype.
                    if cloned.dtype != wire_dtype:
                        cloned = cloned.to(wire_dtype)
                    if not cloned.is_contiguous():
                        cloned = cloned.contiguous()
                    p2p_op = dist.P2POp(
                        dist.isend if async_op else dist.send,
                        cloned,
                        mapped_peer_rank,
                        group=weights_update_group,
                    )
                    p2p_ops.append((op, p2p_op))
                    chunk_clone_bytes += cloned.numel() * cloned.element_size()

                for wire_dtype, bucket in packed_expert_entries:
                    packed = _pack_p2p_send_tensors(
                        [tensor for _, tensor in bucket], wire_dtype
                    )
                    p2p_op = dist.P2POp(
                        dist.isend if async_op else dist.send,
                        packed,
                        mapped_peer_rank,
                        group=weights_update_group,
                    )
                    p2p_ops.append((bucket[0][0], p2p_op))
                    chunk_clone_bytes += packed.numel() * packed.element_size()
                    chunk_send_logical_expert_ops += len(bucket)
                    chunk_send_packed_expert_ops += 1
                chunk_send_p2p_ops[mapped_peer_rank] = p2p_ops

            for recv_from_rank, ops in recv_per_peer.items():
                sub = ops[start:end]
                if not sub:
                    continue
                p2p_ops = []
                transfer_entries = []
                for op in sub:
                    recv_tensor = recv_parameters[op.recv_shard_meta.name]
                    tensor_sliced = slice_tensor(recv_tensor, op, False)
                    transfer_entries.append((op, tensor_sliced))

                direct_entries, packed_expert_entries = _partition_expert_p2p_entries(
                    transfer_entries, pack_expert_p2p
                )
                for op, tensor_sliced in direct_entries:
                    wire_dtype = _wire_dtype(op, tensor_sliced.dtype)
                    tensor_sliced, copyback_pair = _prepare_p2p_recv_tensor(
                        tensor_sliced
                    )
                    if copyback_pair is not None:
                        chunk_recv_tensor_pairs.append(copyback_pair)
                    p2p_op = dist.P2POp(
                        dist.irecv if async_op else dist.recv,
                        tensor_sliced,
                        recv_from_rank,
                        group=weights_update_group,
                    )
                    p2p_ops.append((op, p2p_op))

                for wire_dtype, bucket in packed_expert_entries:
                    packed, copyback_pairs = _prepare_packed_p2p_recv_tensor(
                        [tensor for _, tensor in bucket], wire_dtype
                    )
                    p2p_op = dist.P2POp(
                        dist.irecv if async_op else dist.recv,
                        packed,
                        recv_from_rank,
                        group=weights_update_group,
                    )
                    p2p_ops.append((bucket[0][0], p2p_op))
                    chunk_recv_tensor_pairs.extend(copyback_pairs)
                    chunk_recv_logical_expert_ops += len(bucket)
                    chunk_recv_packed_expert_ops += 1
                    chunk_recv_staging_bytes += packed.numel() * packed.element_size()
                chunk_recv_p2p_ops[recv_from_rank] = p2p_ops

            if chunk_idx == 0 and local_self_recv_built:
                for (
                    actual_send_rank,
                    op,
                    p2p_op,
                    copyback_pair,
                ) in local_self_recv_built:
                    chunk_recv_p2p_ops.setdefault(actual_send_rank, []).append(
                        (op, p2p_op)
                    )
                    if copyback_pair is not None:
                        chunk_recv_tensor_pairs.append(copyback_pair)

            self.execute_recursive_partition_stream_transfer(
                transfer_rank,
                world_size,
                chunk_send_p2p_ops,
                chunk_recv_p2p_ops,
                weights_update_group,
                rank_coordinate,
                step_id,
                synchronize_at_end=False,
            )
            device_util.synchronize()
            if chunk_recv_tensor_pairs:
                logger.info(
                    f"[CHUNKED {task_id}] syncing {len(chunk_recv_tensor_pairs)} "
                    f"staged recv buffers for chunk {chunk_idx}"
                )
                _sync_p2p_recv_tensor_pairs(chunk_recv_tensor_pairs)
                device_util.synchronize()
            logger.warning(
                f"[CHUNKED-DIAG {task_id}] chunk_idx={chunk_idx}/{n_chunks} EXIT "
                f"send_peers={len(chunk_send_p2p_ops)} recv_peers={len(chunk_recv_p2p_ops)} "
                f"clone_mb={chunk_clone_bytes / 1024 / 1024:.1f} "
                f"recv_stage_mb={chunk_recv_staging_bytes / 1024 / 1024:.1f} "
                f"expert_send_ops={chunk_send_logical_expert_ops}->"
                f"{chunk_send_packed_expert_ops} "
                f"expert_recv_ops={chunk_recv_logical_expert_ops}->"
                f"{chunk_recv_packed_expert_ops}"
            )

            chunk_send_p2p_ops = None
            chunk_recv_p2p_ops = None
            chunk_recv_tensor_pairs = None
            import gc as _gc

            _gc.collect()
            if trim_cuda_cache and hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

            total_clone_bytes += chunk_clone_bytes
            total_send_logical_expert_ops += chunk_send_logical_expert_ops
            total_send_packed_expert_ops += chunk_send_packed_expert_ops
            total_recv_logical_expert_ops += chunk_recv_logical_expert_ops
            total_recv_packed_expert_ops += chunk_recv_packed_expert_ops
            total_recv_staging_bytes += chunk_recv_staging_bytes

        logger.info(
            f"CHUNKED transfer done {task_id}: chunks={n_chunks} step_size={step_size} "
            f"total_clone_mb={total_clone_bytes / 1024 / 1024:.2f} "
            f"total_recv_stage_mb={total_recv_staging_bytes / 1024 / 1024:.2f} "
            f"expert_send_ops={total_send_logical_expert_ops}->"
            f"{total_send_packed_expert_ops} "
            f"expert_recv_ops={total_recv_logical_expert_ops}->"
            f"{total_recv_packed_expert_ops}"
        )
