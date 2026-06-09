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
        # Direction 4: dump the device mappings (validate_rank_mappings only
        # checks they are mutual inverses, not the actual rank correspondence).
        # train->infer maps train-domain ranks (>=infer_world_size) to the
        # colocated infer rank; identity would mean no domain shift (a bug).
        t2i_identity = all(
            k == v for k, v in train_to_infer_device_mapping.items()
        )
        i2t_identity = all(
            k == v for k, v in infer_to_train_device_mapping.items()
        )
        logger.info(
            f"[{rank_coordinate}] train_to_infer_device_mapping="
            f"{dict(sorted(train_to_infer_device_mapping.items()))} "
            f"(identity={t2i_identity})"
        )
        logger.info(
            f"[{rank_coordinate}] infer_to_train_device_mapping="
            f"{dict(sorted(infer_to_train_device_mapping.items()))} "
            f"(identity={i2t_identity})"
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
                    # Drop the unconditional clone: a contiguous slice of the
                    # IPC-shared training buffer is sent zero-copy. The
                    # handshake (train blocks on write_finished until every
                    # infer rank put weights_update_finished) plus the post-wait
                    # synchronize bound the async isend flight window, so the
                    # train side cannot mutate the IPC base mid-flight.
                    # contiguous() is a no-op for contiguous slices and only
                    # copies the non-contiguous ones (NCCL send also requires
                    # contiguous memory).
                    p2p_op = dist.P2POp(
                        dist.isend if async_op else dist.send,
                        tensor_sliced.contiguous(),
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
        is_pow2 = world_size > 0 and (world_size & (world_size - 1)) == 0
        if not is_pow2:
            logger.warning(
                f"{prefix} world_size={world_size} is NOT a power of 2; recursive "
                f"partition only runs {num_rounds} rounds and will leave some "
                f"rank pairs uncovered -> hang"
            )
        all_send_keys = set(all_send_p2p_ops.keys())
        all_recv_keys = set(all_recv_p2p_ops.keys())
        covered_send_peers = set()
        covered_recv_peers = set()
        logger.info(
            f"{prefix} Starting recursive partition transfer with {num_rounds} rounds; "
            f"send_peers={sorted(all_send_keys)} recv_peers={sorted(all_recv_keys)}"
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

            # Butterfly coverage: each round this rank both sends to and recvs
            # from every peer in other_half. Accumulate which send/recv peer
            # keys fall inside this round's range; any plan op whose peer key is
            # never covered across all rounds will never be enqueued -> the
            # opposite rank waits forever (hang).
            round_other_half = set(range(other_half_start, other_half_end))
            round_send_active = round_other_half & all_send_keys
            round_recv_active = round_other_half & all_recv_keys
            covered_send_peers |= round_send_active
            covered_recv_peers |= round_recv_active
            logger.info(
                f"{prefix} Round {round_idx} coverage: "
                f"send_active={sorted(round_send_active)} "
                f"recv_active={sorted(round_recv_active)}"
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
            # Cross-rank round barrier: the butterfly only matches NCCL P2P ops
            # by (src,dst) FIFO order (no MPI tag). Under extreme per-edge load
            # imbalance, a lightly-loaded rank can finish this round's work.wait()
            # and race into the next round's smaller partition, enqueuing P2P ops
            # on a pipe whose peer is still draining THIS round -> cross-round FIFO
            # contention -> circular wait -> deadlock. A barrier here forces every
            # rank to finish round_idx before any rank starts round_idx+1, so each
            # (src,dst) pipe only ever carries ops from a single round at a time.
            dist.barrier(
                group=weights_update_group,
                device_ids=[device_util.current_device()],
            )
        device_util.synchronize()
        duration = time.time() - start_time
        uncovered_send = all_send_keys - covered_send_peers
        uncovered_recv = all_recv_keys - covered_recv_peers
        if uncovered_send or uncovered_recv:
            logger.warning(
                f"{prefix} BUTTERFLY COVERAGE GAP after {num_rounds} rounds: "
                f"UNCOVERED_SEND={sorted(uncovered_send)} "
                f"UNCOVERED_RECV={sorted(uncovered_recv)} "
                f"(these peer ops were never enqueued -> opposite rank will hang); "
                f"all_send={sorted(all_send_keys)} covered_send={sorted(covered_send_peers)}; "
                f"all_recv={sorted(all_recv_keys)} covered_recv={sorted(covered_recv_peers)}"
            )
        else:
            logger.info(
                f"{prefix} BUTTERFLY COVERAGE OK: all {len(all_send_keys)} send + "
                f"{len(all_recv_keys)} recv peers covered across {num_rounds} rounds"
            )
        logger.info(f"{prefix} All {num_rounds} rounds completed in {duration:.4f}s")

    def _execute_ops_concurrent(self, ops_dict, peer_ranks):
        """
        Issue all P2P ops for the given peers as one batched NCCL group call.

        Why batch_isend_irecv instead of manual multi-stream posting: the previous
        implementation posted tens of thousands of isend/irecv individually across
        many CUDA streams and then work.wait()-ed all of them. At 32-rank scale with
        highly imbalanced per-peer op counts this overwhelmed NCCL's P2P channels and
        deadlocked ~half the ranks in work.wait() (Round 0 Phase 2). Wrapping every op
        in a single dist.batch_isend_irecv (ncclGroupStart/End) lets NCCL aggregate and
        schedule the transfers internally. It stays deadlock-safe as long as per-(src,dst)
        FIFO order is preserved, which it is: ops for each peer are appended in their
        original op_idx order.

        Args:
            ops_dict: Dictionary mapping peer_rank to list of (plan_op, p2p_op) tuples
            peer_ranks: Range or iterable of peer ranks to process

        Returns:
            Total number of ops executed
        """
        p2p_ops = []
        for peer_rank in peer_ranks:
            if peer_rank in ops_dict:
                for _, p2p_op in ops_dict[peer_rank]:
                    p2p_ops.append(p2p_op)

        if not p2p_ops:
            return 0

        works = dist.batch_isend_irecv(p2p_ops)
        for work in works:
            work.wait()

        return len(p2p_ops)
