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

        # Execute recursive partition transfer
        # FIXME: batch_isend_irecv hang sometimes, seems `batch_isend_irecv` can't handle asymmetric p2p communication.
        # so we use send/recv directly
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
        # Collect ops from all peers that have operations, along with their peer_rank
        peer_ops_with_rank = []
        active_peer_ranks = []
        for peer_rank in peer_ranks:
            if peer_rank in ops_dict:
                peer_ops_with_rank.append((peer_rank, ops_dict[peer_rank]))
                active_peer_ranks.append(peer_rank)

        if not peer_ops_with_rank:
            return 0

        # Submit ALL phase-local P2P ops as a single batch_isend_irecv. The
        # multi-stream round-robin original (commented out below) hangs when
        # the transfer plan is asymmetric across ranks: per-rank stream index
        # assignment based on local enumerate(active_peer_ranks) creates
        # different stream allocations on the two sides of a peer pair, and
        # NCCL P2P FIFO ordering on a stream then deadlocks. Submitting all
        # ops in one batch lets NCCL coordinate ordering globally; pairs are
        # matched by (src, dst, group) regardless of stream.
        all_p2p_ops = []
        for peer_rank, ops in peer_ops_with_rank:
            for _, p2p_op in ops:
                all_p2p_ops.append(p2p_op)
        if not all_p2p_ops:
            return 0
        works = dist.batch_isend_irecv(all_p2p_ops)
        for work in works:
            work.wait()
        # Force GPU completion before returning. work.wait() only blocks
        # CPU thread until CUDA event records 'enqueued', not actual NCCL
        # kernel completion. Without this sync, downstream chunk steps may
        # race ahead and the per-chunk torch.cuda.synchronize() at the end
        # of _run_chunked exposes the resulting NCCL kernel hang only at
        # chunk boundary instead of at the actual offending phase.
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        return len(all_p2p_ops)

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
