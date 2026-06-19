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


def _op_key(op):
    """Delegate to awex.delta.delta_p2p.op_key (single source of truth)."""
    from awex.delta.delta_p2p import op_key

    return op_key(op)


class _PlanView:
    """Lightweight transfer-plan view exposing a filtered ``operations`` dict.

    transfer_delta_in_colocate_mode only reads ``plan.operations`` (peer -> ops),
    so a filtered sub-view is enough to scope a two-round P2P transfer to a
    single dtype group without copying the whole TransferPlan.
    """

    def __init__(self, operations):
        self.operations = operations


def _filter_plan_by_dtype(plan, dtype, *, is_send):
    """Filter a transfer plan's operations down to ops of one parameter dtype.

    For mixed-precision delta: each uniform-dtype group runs its own two-round
    P2P (round 1 carries only nnz, so the recv side needs a single val dtype).
    Send-side ops are matched on ``send_shard_meta.dtype``, recv-side on
    ``recv_shard_meta.dtype``; the same parameter has identical dtype on both
    ends, so a sender's "dtype X" group pairs with the receiver's "dtype X"
    group. Empty peers are dropped (zero-op peers are no-ops in the schedule).
    """
    meta_attr = "send_shard_meta" if is_send else "recv_shard_meta"
    filtered = {}
    for peer_rank, ops in plan.operations.items():
        kept = [op for op in ops if getattr(op, meta_attr).dtype == dtype]
        if kept:
            filtered[peer_rank] = kept
    return _PlanView(filtered)


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

        # Chunked path (production): split each peer's ops into AWEX_CHUNK_MB
        # sized chunks so a single batch_isend_irecv never floods NCCL P2P
        # channels (the 32-rank / PP4->PP1 asymmetric one-shot path deadlocks
        # in _coalescing_manager). Ported from the known-good Asystem
        # feat-sglang-plugin transport. Falls through to the legacy one-shot
        # path only when AWEX_CHUNK_MB is unset/0.
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
                f"Finished CHUNKED weights update for {task_id}, took "
                f"{duration:.4f}s (chunk_mb={chunk_mb})"
            )
            return

        # === LEGACY PATH (one-shot clone-then-transfer) ===
        # Build P2P operations with sliced tensors
        all_send_p2p_ops = {}  # peer_rank -> List[(plan_op, p2p_op)]
        all_recv_p2p_ops = {}  # peer_rank -> List[(plan_op, p2p_op)]
        tensors_to_copy = []
        train_slice_context = {}
        non_contiguous_tensor_pairs = []

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
                    p2p_op = dist.P2POp(
                        dist.isend if async_op else dist.send,
                        tensor_sliced.clone(),
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
                if not tensor_sliced.is_contiguous():
                    original_tensor = tensor_sliced
                    tensor_sliced = tensor_sliced.contiguous()
                    non_contiguous_tensor_pairs.append((original_tensor, tensor_sliced))
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

        # Execute recursive partition transfer. The asymmetric-p2p deadlock the
        # old FIXME warned about is fixed in _execute_ops_concurrent (per-peer
        # batch_isend_irecv, O(1) in-flight peers); both dense and the delta
        # two-round path go through the same symmetric schedule.
        self.execute_recursive_partition_stream_transfer(
            transfer_rank,
            world_size,
            all_send_p2p_ops,
            all_recv_p2p_ops,
            weights_update_group,
            rank_coordinate,
            step_id,
        )
        if non_contiguous_tensor_pairs:
            with torch.no_grad():
                for original_tensor, recv_tensor in non_contiguous_tensor_pairs:
                    original_tensor.copy_(recv_tensor)
                non_contiguous_tensor_pairs.clear()
                del non_contiguous_tensor_pairs
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
        device_util.synchronize()
        duration = time.time() - start_time
        logger.info(f"{prefix} All {num_rounds} rounds completed in {duration:.4f}s")

    def _execute_ops_concurrent(self, ops_dict, peer_ranks):
        """Execute one recursive-partition phase, walking peers in ascending order.

        Per-peer ``batch_isend_irecv``. The previous behaviour (round-robin the
        WHOLE half's ops across multiple CUDA streams, then wait on all handles
        at once) deadlocks at 32-rank / PP4->PP1 asymmetric scale: round 0's
        other-half has 16 peers and thousands of ops, and flooding NCCL with
        that many concurrent P2P channels exhausts them so the GPU drain never
        completes (the data layer is verified fully symmetric -- the failure is
        purely runtime concurrency scale; this is the P60-73 hang). Walking
        peers in ascending rank order and issuing ONE batch_isend_irecv per
        peer caps in-flight P2P at O(1) peer.

        Deadlock-free because a recursive-partition phase is single-direction:
        in phase 1 every first-half rank only SENDS and every second-half rank
        only RECVS (phase 2 is the mirror), so serializing peers cannot form a
        wait cycle. Both sides MUST walk peers in the same (ascending) order so
        the k-th per-peer batch on a sender pairs with the matching recv on the
        receiver; ``peer_ranks`` is already an ascending range here.

        Ported from the known-good Asystem path (awex feat-sglang-plugin); the
        multi-stream flood version this replaces was the delta branch's own
        ``FIXME: batch_isend_irecv hang ... asymmetric p2p`` left unfixed.

        Args:
            ops_dict: Dictionary mapping peer_rank to list of (plan_op, p2p_op).
            peer_ranks: Ascending range/iterable of peer ranks to process.

        Returns:
            Total number of ops executed.
        """
        trace = os.environ.get("AWEX_P2P_TRACE", "").strip() in ("1", "true", "True")
        my_rank = self.transfer_rank
        total_ops = 0
        for peer_rank in peer_ranks:
            ops = ops_dict.get(peer_rank)
            if not ops:
                continue
            p2p_ops = [p2p_op for _, p2p_op in ops]
            if not p2p_ops:
                continue
            if trace:
                logger.info(
                    f"[P2P-TRACE rank={my_rank}] peer={peer_rank} "
                    f"nops={len(p2p_ops)} -> batch_isend_irecv (pre-wait)"
                )
            works = dist.batch_isend_irecv(p2p_ops)
            for work in works:
                work.wait()
            # work.wait() only blocks the CPU until the CUDA event records
            # 'enqueued', not actual NCCL kernel completion; syncing per peer
            # keeps in-flight P2P bounded to one peer and surfaces any hang at
            # the offending peer rather than at a later boundary.
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            if trace:
                logger.info(
                    f"[P2P-TRACE rank={my_rank}] peer={peer_rank} "
                    f"batch drained ({len(p2p_ops)} ops)"
                )
            total_ops += len(p2p_ops)
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
            actual_send_rank = train_to_infer_device_mapping.get(
                op.send_rank, op.send_rank
            )
            p2p_op = dist.P2POp(
                dist.irecv,
                recv_sliced,
                actual_send_rank,
                group=weights_update_group,
            )
            local_self_recv_built.append((actual_send_rank, op, p2p_op))

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
                for s in (sample_op.train_slices or []):
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
                        device=device_util.current_device(),
                        dtype=torch.int64,
                    )
                    dist.all_reduce(
                        t, op=dist.ReduceOp.MIN, group=weights_update_group
                    )
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
                    device=device_util.current_device(),
                    dtype=torch.int64,
                )
                dist.all_reduce(
                    t, op=dist.ReduceOp.MAX, group=weights_update_group
                )
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

        total_clone_bytes = 0

        for chunk_idx in range(n_chunks):
            start = chunk_idx * step_size
            end = start + step_size

            logger.warning(
                f"[CHUNKED-DIAG {task_id}] chunk_idx={chunk_idx}/{n_chunks} ENTER "
                f"slice=[{start},{end})"
            )
            chunk_send_p2p_ops = {}
            chunk_recv_p2p_ops = {}
            chunk_clone_bytes = 0

            for mapped_peer_rank, ops in send_per_peer.items():
                sub = ops[start:end]
                if not sub:
                    continue
                p2p_ops = []
                for op in sub:
                    send_tensor = send_parameters[op.send_shard_meta.name]
                    tensor_sliced = slice_tensor(
                        send_tensor, op, True, slice_context=train_slice_context
                    )
                    recv_rank = train_to_infer_device_mapping.get(
                        op.recv_rank, op.recv_rank
                    )
                    cloned = tensor_sliced.clone()
                    # Wire-size parity: the receiver posts irecv with ITS shard
                    # dtype. 961 plan ops (mlp.gate.weight, 124 edges) are bf16
                    # on the train side but fp32 on the sglang side; sending
                    # bf16 bytes into an fp32-sized recv leaves the receiver
                    # waiting forever (deterministic chunk-7 deadlock,
                    # Problem 69). Cast the clone to the receiver's dtype.
                    recv_dtype = getattr(op.recv_shard_meta, "dtype", None)
                    if recv_dtype is not None and cloned.dtype != recv_dtype:
                        cloned = cloned.to(recv_dtype)
                    p2p_op = dist.P2POp(
                        dist.isend if async_op else dist.send,
                        cloned,
                        recv_rank,
                        group=weights_update_group,
                    )
                    p2p_ops.append((op, p2p_op))
                    chunk_clone_bytes += cloned.numel() * cloned.element_size()
                chunk_send_p2p_ops[mapped_peer_rank] = p2p_ops

            for recv_from_rank, ops in recv_per_peer.items():
                sub = ops[start:end]
                if not sub:
                    continue
                p2p_ops = []
                for op in sub:
                    recv_tensor = recv_parameters[op.recv_shard_meta.name]
                    tensor_sliced = slice_tensor(recv_tensor, op, False)
                    p2p_op = dist.P2POp(
                        dist.irecv if async_op else dist.recv,
                        tensor_sliced,
                        recv_from_rank,
                        group=weights_update_group,
                    )
                    p2p_ops.append((op, p2p_op))
                chunk_recv_p2p_ops[recv_from_rank] = p2p_ops

            if chunk_idx == 0 and local_self_recv_built:
                for actual_send_rank, op, p2p_op in local_self_recv_built:
                    chunk_recv_p2p_ops.setdefault(actual_send_rank, []).append(
                        (op, p2p_op)
                    )

            self.execute_recursive_partition_stream_transfer(
                transfer_rank,
                world_size,
                chunk_send_p2p_ops,
                chunk_recv_p2p_ops,
                weights_update_group,
                rank_coordinate,
                step_id,
            )
            device_util.synchronize()
            logger.warning(
                f"[CHUNKED-DIAG {task_id}] chunk_idx={chunk_idx}/{n_chunks} EXIT "
                f"send_peers={len(chunk_send_p2p_ops)} recv_peers={len(chunk_recv_p2p_ops)} "
                f"clone_mb={chunk_clone_bytes/1024/1024:.1f}"
            )

            chunk_send_p2p_ops = None
            chunk_recv_p2p_ops = None
            import gc as _gc
            _gc.collect()
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

            total_clone_bytes += chunk_clone_bytes

        logger.info(
            f"CHUNKED transfer done {task_id}: chunks={n_chunks} step_size={step_size} "
            f"total_clone_mb={total_clone_bytes / 1024 / 1024:.2f}"
        )

    # ------------------------------------------------------------------
    # Delta (sparse) transfer: two-round variable-length P2P.
    # GPU/multi-process validation pending (no distributed ctx on CPU).
    # Reuses execute_recursive_partition_stream_transfer unchanged so the
    # deadlock-safe symmetric schedule is preserved; only the per-op tensors
    # differ (round 1: nnz int32[1]; round 2: idx int32[nnz] + val[nnz]).
    # ------------------------------------------------------------------
    def transfer_delta_in_colocate_mode(
        self,
        train_to_infer_device_mapping,
        infer_to_train_device_mapping,
        transfer_rank,
        rank_coordinate,
        world_size,
        send_transfer_plan,
        recv_transfer_plan,
        weights_update_group,
        send_payloads_by_op,  # {send_shard_meta.name+train_slices key -> OpDeltaPayload}
        recv_parameters,
        value_dtype,
        *,
        step_id=-1,
    ):
        """Cross-rank P2P delta transfer (bandwidth-critical path).

        send_payloads_by_op: per-op OpDeltaPayload (indices already remapped to
            inference-shard flat space, values gathered), keyed per send op.
        recv_parameters: live inference params (self.parameters, write-through).
        Local (self-copy) ops are NOT handled here — the caller reconstructs
        them locally (design §5.4 first version).
        """
        send_ops = dict(send_transfer_plan.operations)
        recv_ops = dict(recv_transfer_plan.operations)

        # ---- Round 1: exchange nnz (fixed 1 int32 per op, symmetric) ----
        nnz_send_p2p = {}  # peer -> [(op, P2POp)]
        nnz_recv_p2p = {}
        recv_nnz_buf = {}  # peer -> [(op, int32[1] tensor)]
        # get_torch_device() returns a torch.device; current_device() returns a
        # bare int which torch interprets as a CUDA ordinal and silently maps to
        # CPU/raises on the NPU(hccl) backend -> nnz/idx/val built on the wrong
        # device and the cross-rank isend fails. Match the rest of the codebase.
        device = device_util.get_torch_device()

        for peer_rank, ops in send_ops.items():
            mapped_peer = train_to_infer_device_mapping.get(peer_rank, peer_rank)
            if mapped_peer == transfer_rank:
                continue  # self-copy handled by caller
            p2p = []
            for op in ops:
                payload = send_payloads_by_op[_op_key(op)]
                nnz_t = torch.tensor([payload.nnz], dtype=torch.int32, device=device)
                recv_rank = train_to_infer_device_mapping.get(
                    op.recv_rank, op.recv_rank
                )
                p2p.append(
                    (
                        op,
                        dist.P2POp(
                            dist.isend, nnz_t, recv_rank, group=weights_update_group
                        ),
                    )
                )
            nnz_send_p2p[mapped_peer] = p2p

        for send_rank, ops in recv_ops.items():
            recv_from = train_to_infer_device_mapping[send_rank]
            if recv_from == transfer_rank:
                continue
            p2p = []
            bufs = []
            for op in ops:
                nnz_t = torch.empty(1, dtype=torch.int32, device=device)
                p2p.append(
                    (
                        op,
                        dist.P2POp(
                            dist.irecv, nnz_t, recv_from, group=weights_update_group
                        ),
                    )
                )
                bufs.append((op, nnz_t))
            nnz_recv_p2p[recv_from] = p2p
            recv_nnz_buf[recv_from] = bufs

        self.execute_recursive_partition_stream_transfer(
            transfer_rank,
            world_size,
            nnz_send_p2p,
            nnz_recv_p2p,
            weights_update_group,
            rank_coordinate,
            step_id,
        )
        device_util.synchronize()

        # ---- Allocate recv idx/val buffers from received nnz ----
        recv_payload_bufs = {}  # peer -> [(op, idx_buf, val_buf)]
        for peer, bufs in recv_nnz_buf.items():
            entries = []
            for op, nnz_t in bufs:
                n = int(nnz_t.item())
                entries.append(
                    (
                        op,
                        torch.empty(n, dtype=torch.int32, device=device),
                        torch.empty(n, dtype=value_dtype, device=device),
                    )
                )
            recv_payload_bufs[peer] = entries

        # ---- Round 2: exchange idx + val (sizes now known both sides) ----
        # Two P2POps per op (idx then val); symmetric on both sides.
        pay_send_p2p = {}
        pay_recv_p2p = {}
        for peer_rank, ops in send_ops.items():
            mapped_peer = train_to_infer_device_mapping.get(peer_rank, peer_rank)
            if mapped_peer == transfer_rank:
                continue
            p2p = []
            for op in ops:
                payload = send_payloads_by_op[_op_key(op)]
                recv_rank = train_to_infer_device_mapping.get(
                    op.recv_rank, op.recv_rank
                )
                # copy=True guarantees an independent buffer for the async isend
                # (parity with the dense path's tensor_sliced.clone()): .to(device)
                # alone is a no-op when already on-device, so without copy the
                # send would alias the payload buffer and risk a write-after-read
                # race if it is reused before the isend drains.
                idx = payload.indices.to(device=device, copy=True).contiguous()
                val = payload.values.to(device=device, copy=True).contiguous()
                p2p.append(
                    (
                        op,
                        dist.P2POp(
                            dist.isend, idx, recv_rank, group=weights_update_group
                        ),
                    )
                )
                p2p.append(
                    (
                        op,
                        dist.P2POp(
                            dist.isend, val, recv_rank, group=weights_update_group
                        ),
                    )
                )
            pay_send_p2p[mapped_peer] = p2p

        for peer, entries in recv_payload_bufs.items():
            p2p = []
            for op, idx_buf, val_buf in entries:
                p2p.append(
                    (
                        op,
                        dist.P2POp(
                            dist.irecv, idx_buf, peer, group=weights_update_group
                        ),
                    )
                )
                p2p.append(
                    (
                        op,
                        dist.P2POp(
                            dist.irecv, val_buf, peer, group=weights_update_group
                        ),
                    )
                )
            pay_recv_p2p[peer] = p2p

        self.execute_recursive_partition_stream_transfer(
            transfer_rank,
            world_size,
            pay_send_p2p,
            pay_recv_p2p,
            weights_update_group,
            rank_coordinate,
            step_id,
        )
        device_util.synchronize()

        # ---- Scatter received patches into live inference params ----
        from awex.delta.codec import apply_sparse_patch_

        applied = 0
        for entries in recv_payload_bufs.values():
            for op, idx_buf, val_buf in entries:
                if idx_buf.numel() == 0:
                    continue
                target = slice_tensor(
                    recv_parameters[op.recv_shard_meta.name], op, False
                )
                apply_sparse_patch_(target, idx_buf, val_buf)
                applied += 1
        logger.info(
            "[%s] delta transfer step %s: applied %d non-empty patches",
            rank_coordinate,
            step_id,
            applied,
        )
        return applied

    def apply_delta_colocate(
        self,
        train_to_infer_device_mapping,
        infer_to_train_device_mapping,
        transfer_rank,
        rank_coordinate,
        world_size,
        send_transfer_plan,
        recv_transfer_plan,
        weights_update_group,
        send_full_params,  # {hf_name: reconstructed full train-shard tensor}
        masks,  # {hf_name: bool change mask over the train-shard}
        recv_parameters,  # self.parameters (live inference view)
        value_dtype,
        *,
        step_id=-1,
    ):
        """Reader-facing entry: self-copy locally (full) + cross-rank delta.

        Design §5.4 first version: the local self-copy segment is not the
        bandwidth bottleneck, so it reuses the dense path on a locally
        reconstructed full tensor; only the cross-rank P2P segment ships sparse
        deltas. ``send_full_params`` is the full reconstructed train-shard (from
        ``_delta_base`` + delta applied), used for self-copy; ``masks`` drives
        the per-op sparse projection for cross-rank ops.
        """
        from awex.delta.delta_p2p import build_send_payloads_by_op

        # --- Self-copy segment (local, full, dense logic) ---
        send_ops = dict(send_transfer_plan.operations)
        train_slice_context = {}
        tensors_to_copy = []
        for peer_rank, ops in send_ops.items():
            mapped_peer = train_to_infer_device_mapping.get(peer_rank, peer_rank)
            if mapped_peer != transfer_rank:
                continue  # cross-rank handled below
            for op in ops:
                send_tensor = send_full_params[op.send_shard_meta.name]
                tensors_to_copy.append(
                    slice_tensor(
                        send_tensor, op, True, slice_context=train_slice_context
                    )
                )
        if tensors_to_copy:
            local_send_rank = infer_to_train_device_mapping[transfer_rank]
            execute_tensors_to_copy(
                tensors_to_copy,
                recv_transfer_plan.operations[local_send_rank],
                recv_parameters,
                f"delta self-copy for {rank_coordinate}-{step_id}",
            )

        # --- Cross-rank segment (sparse delta over P2P), grouped by dtype ---
        # The two-round protocol's round 1 carries only nnz (not per-op dtype),
        # so the recv side pre-allocates val buffers from ONE dtype. To support
        # mixed-precision models (bf16 body + fp32 MoE router) we partition the
        # cross-rank ops by their parameter dtype and run one full two-round
        # transfer per uniform-dtype group. Each group is internally uniform, so
        # the existing transfer_delta_in_colocate_mode is reused unchanged.
        #
        # Deadlock symmetry: the dtype group set must be identical and same-order
        # on every rank. The model structure is identical across ranks, so we
        # derive the group set from recv_parameters (the full live inference
        # view) and iterate sorted(by str). Every rank enters every group's
        # collective unconditionally (empty group -> zero-nnz, still symmetric),
        # mirroring the dense path's unconditional transfer call.
        cross_ops = []
        for peer_rank, ops in send_ops.items():
            mapped_peer = train_to_infer_device_mapping.get(peer_rank, peer_rank)
            if mapped_peer == transfer_rank:
                continue
            cross_ops.extend(ops)

        # Cross-rank op -> param dtype (use the actual send tensor dtype; equals
        # send_shard_meta.dtype but is the ground truth the payload is built from).
        def _op_dtype(op):
            return send_full_params[op.send_shard_meta.name].dtype

        ops_by_dtype = {}
        for op in cross_ops:
            ops_by_dtype.setdefault(_op_dtype(op), []).append(op)

        # All-rank-consistent group set + order (model structure identical).
        all_dtypes = sorted(
            {t.dtype for t in recv_parameters.values()}, key=str
        )

        applied = 0
        for dt in all_dtypes:
            group_ops = ops_by_dtype.get(dt, [])
            send_payloads_by_op = build_send_payloads_by_op(
                group_ops, masks, send_full_params
            )
            sub_send_plan = _filter_plan_by_dtype(send_transfer_plan, dt, is_send=True)
            sub_recv_plan = _filter_plan_by_dtype(recv_transfer_plan, dt, is_send=False)
            # Unconditional entry per group keeps the recursive-partition
            # schedule symmetric across ranks even when this rank has no op in
            # this dtype group (a peer may still send to us).
            applied += self.transfer_delta_in_colocate_mode(
                train_to_infer_device_mapping,
                infer_to_train_device_mapping,
                transfer_rank,
                rank_coordinate,
                world_size,
                sub_send_plan,
                sub_recv_plan,
                weights_update_group,
                send_payloads_by_op,
                recv_parameters,
                dt,
                step_id=step_id,
            )
        return applied
