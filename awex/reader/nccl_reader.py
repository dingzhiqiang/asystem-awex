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

import gc
import os
import time

import torch
import torch.distributed as dist

from awex import logging
from awex.reader.weights_reader import WorkerWeightsReader
from awex.transfer.nccl_comm import batch_send_recv, nccl_build_recv_ops
from awex.transfer.transfer_plan import (
    TransferPlanBuilder,
    compute_transfer_plan_hash,
    compute_transfer_plan_stats,
)
from awex.util import device as device_util
from awex.util.common import (
    compute_statistics,
    get_free_port,
    get_ip_address,
)
from awex.util.gpu import get_gpu_status, print_current_gpu_status
from awex.util.system_util import count_open_fds
from awex.util.tensor_util import (
    cuda_ipc_deserialize,
    ipc_deserialize,
    reconstruct_tensors_from_groups,
)

logger = logging.getLogger(__name__)


class NCCLWorkerWeightsReader(WorkerWeightsReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transfer_plan = None
        self.weights_update_group = None
        self.send_ranks = None
        self.send_ranks_sample = None
        self.num_to_recvs = None
        self.rank_coordinate = None

    def initialize(self):
        super().initialize()
        plan_builder = TransferPlanBuilder(
            self.infer_world_size,
            self.training_world_size,
            self.num_engines,
            self.enable_debug_mode,
        )
        self.transfer_plan = plan_builder.build_local_transfer_plan(
            self.parameters_meta,
            self.training_params_meta,
            self.transfer_rank,
        )
        inter_hash = compute_transfer_plan_hash(self.transfer_plan)
        logger.info(
            "Reader rank %s inter plan hash: %s",
            self.transfer_rank,
            inter_hash,
        )
        logger.info(
            "Reader rank %s transfer plan stats: %s",
            self.transfer_rank,
            compute_transfer_plan_stats(self.transfer_plan),
        )
        if self.transfer_rank == 0:
            master_address = get_ip_address()
            master_port = get_free_port()
            master_info = (master_address, master_port)
            self.meta_server_client.put_object("master_info", master_info)
            logger.info(
                f"Put master info to meta server for rank {self.transfer_rank}: {master_info}"
            )
        else:
            master_info = self.meta_server_client.get_object(
                "master_info", timeout=self.timeout
            )
            master_address, master_port = master_info
            logger.info(
                f"Get master info from meta server for rank {self.transfer_rank}: {master_info}"
            )
        logger.info(
            f"Start to initialize NCCL weights writer for rank {self.transfer_rank}"
        )

        self.master_address = master_address
        self.master_port = master_port
        self.world_size = (
            self.infer_world_size
            if self.enable_colocate_mode
            else self.transfer_world_size
        )

        self._set_device()
        self._init_weights_exchange_process_group()
        self._shake_hands_with_writer()

        self.send_ranks = list(self.transfer_plan.operations.keys())
        self.send_ranks_sample = (
            self.send_ranks[:8] + ["..."] + self.send_ranks[-8:]
            if len(self.send_ranks) > 16
            else self.send_ranks
        )
        self.num_to_recvs = sum(
            len(operations) for operations in self.transfer_plan.operations.values()
        )
        self.rank_coordinate = (
            f"{self.engine_rank}-{self.rank_info.global_rank}-{self.transfer_rank}"
        )
        if self.enable_colocate_mode:
            self._init_reader_in_colocate_mode()
        self.deserialized_weights = {}
        # Delta transfer base (env-gated, see _maybe_reconstruct_delta): CPU
        # copy of the last fully-synced train-shard payload + its version.
        # Sparse payloads are reconstructed against this base before the
        # (unchanged) NCCL reshard transport runs.
        self._delta_base: dict = {}
        self._delta_base_version = None
        # Per-param change mask for the current delta step ({name: bool tensor}),
        # or None on a dense/anchor step. Drives the cross-rank sparse P2P in
        # _update_weights_in_colocate_mode; reset every collect.
        self._delta_masks = None
        logger.info(
            f"Created NCCL weights reader for rank {self.rank_info.global_rank}, engine rank {self.engine_rank}"
        )

    def _shake_hands_with_writer(self):
        from awex.util.process_group import (
            setup_batch_isend_irecv,
        )

        if self.transfer_rank == 0:
            logger.info(
                f"Start to test NCCL ready for rank {self.transfer_rank}, world size {self.transfer_world_size}"
            )
            dist.recv(
                self.ready_tensor,
                src=self.world_size - 1,
                group=self.weights_update_group,
            )
            logger.info(
                f"NCCL ready: recv tensor from rank 0 for rank {self.transfer_rank}"
            )
        if (
            self.enable_colocate_mode
            and self.transfer_rank == self.infer_world_size - 1
        ):
            dist.send(
                self.ready_tensor,
                dst=0,
                group=self.weights_update_group,
            )
        setup_batch_isend_irecv(
            self.weights_update_group, self.transfer_rank, self.world_size
        )

    def _set_device(self):
        gpu_id = getattr(self.scheduler, "gpu_id", None) or getattr(
            self.scheduler, "local_rank", None
        )
        if gpu_id is None:
            gpu_id = int(os.environ.get("LOCAL_RANK", 0))
        device_type = device_util.get_device_type()
        device_count = device_util.device_count() or 1
        if device_type == "cuda":
            prev_device = torch.cuda.current_device()
            logger.info(
                f"[NCCLWeightsReader] Set device to {gpu_id} for rank {self.transfer_rank}, "
                f"device env is {os.environ.get('DEVICE')}, "
                f"previous device is {prev_device}, "
                f"device_count is {device_count}, "
                f"CUDA_VISIBLE_DEVICES env is {os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
            torch.cuda.set_device(gpu_id)
            self.barrier_device = torch.cuda.current_device()
            self.backend = "nccl"
            self.ready_tensor = torch.tensor(1).cuda()
        else:
            logger.info(
                f"[NCCLWeightsReader] Set device to {gpu_id} for rank {self.transfer_rank}, "
                f"device env is {os.environ.get('DEVICE')}, "
                f"previous device is {device_util.current_device()}, "
                f"device_count is {device_count}, "
                f"{'/'.join(device_util.visible_devices_env_names())} env is {device_util.visible_devices_env_value() or '(unset)'}"
            )
            device_util.set_device(gpu_id)
            self.barrier_device = device_util.current_device()
            self.backend = self.comm_backend
            self.ready_tensor = torch.tensor(1, device=device_util.get_torch_device())

    def _init_weights_exchange_process_group(self):
        if self.already_initialized:
            return
        from awex.util.process_group import (
            init_weights_update_group,
        )

        self.weights_update_group = init_weights_update_group(
            master_address=self.master_address,
            master_port=self.master_port,
            rank=self.transfer_rank,
            world_size=self.world_size,
            group_name="weights_exchange",
            backend=self.backend,
            role="inference",
        )
        logger.info(
            f"Initialized NCCL weights reader for rank {self.transfer_rank}, engine rank {self.engine_rank}"
        )
        # Add a barrier to ensure all processes are ready
        dist.barrier(group=self.weights_update_group, device_ids=[self.barrier_device])
        logger.info(f"Barrier passed for weights reader with rank {self.transfer_rank}")
        self.already_initialized = True

    def _destroy_weights_exchange_process_group(self):
        # reduce the impact of process group to avoid oom in infer
        if self.destroy_pg_after_update and self.backend == "hccl":
            self.already_initialized = False
            torch.distributed.destroy_process_group(self.weights_update_group)
            torch.npu.synchronize()
            torch.npu.empty_cache()

    def _init_reader_in_colocate_mode(self):
        self.meta_server_client.add_object_to_set(
            "inference_device_rank_entries",
            (get_ip_address(), device_util.current_device(), self.transfer_rank),
        )
        self.meta_server_client.wait_set_until_size(
            "inference_device_rank_entries", self.infer_world_size, timeout=self.timeout
        )
        self.inference_device_mapping = self.meta_server_client.get_set(
            "inference_device_rank_entries"
        )
        self.inference_device_mapping = {
            (ip_address, device_id): transfer_rank
            for ip_address, device_id, transfer_rank in self.inference_device_mapping
        }

        self.meta_server_client.wait_set_until_size(
            "training_device_rank_entries",
            self.training_world_size,
            timeout=self.timeout,
        )
        device_rank_entries = self.meta_server_client.get_set(
            "training_device_rank_entries"
        )
        self.training_device_mapping = {
            (ip_address, device_id): transfer_rank
            for ip_address, device_id, transfer_rank in device_rank_entries
        }
        self.train_to_infer_device_mapping = {}
        self.infer_to_train_device_mapping = {}
        for ip_address, device_id, transfer_rank in device_rank_entries:
            infer_rank = self.inference_device_mapping[(ip_address, device_id)]
            self.train_to_infer_device_mapping[transfer_rank] = infer_rank
            self.infer_to_train_device_mapping[infer_rank] = transfer_rank
        plan_builder = TransferPlanBuilder(
            self.infer_world_size,
            self.training_world_size,
            self.num_engines,
            self.enable_debug_mode,
        )
        self.send_transfer_plan = plan_builder.build_local_transfer_plan(
            self.parameters_meta,
            self.training_params_meta,
            self.infer_to_train_device_mapping[self.transfer_rank],
        )
        from awex.transfer.nccl_stream_batch import NcclColocateStreamBatchTransport

        self.colocate_transport = NcclColocateStreamBatchTransport(
            self.transfer_rank, self.infer_world_size
        )
        logger.info(
            f"Initialized NCCL weights reader for rank {self.transfer_rank} in colocate mode"
        )

    def pre_update_weights(self, step_id, **kwargs):
        pass

    def collect_training_weights(self, step_id, **kwargs):
        if not self.enable_colocate_mode:
            return
        # Can't serialize IPC tensors at initialization since every step, the memory address for weights will change
        # because we use offloading for moving GPU tensors to CPU and back later
        # We'll get serialized weights from meta server each step instead
        # Get serialized weights from meta server
        ip_address = get_ip_address()
        device_id = device_util.current_device()
        key = f"training_serialized_weights_{ip_address}_{device_id}_{step_id}"
        logger.info(
            f"Start to get serialized ipc weights {key} for rank {self.rank_coordinate}"
        )
        self.send_rank, self.send_rank_info, serialized_weights = (
            self.meta_server_client.get_object(key, timeout=self.timeout)
        )
        logger.info(
            f"Finished getting serialized ipc weights {key} for rank {self.rank_coordinate}"
        )
        logger.info(
            f"GPU status before deserialization:\n{get_gpu_status()} for rank {self.rank_coordinate}"
        )
        logger.info(f"Open fds before deserialization: {count_open_fds()}")
        # Deserialize weights into tensors
        if self.ipc_backend in ("cpu", "npu"):
            group_shared, metadata, names = ipc_deserialize(serialized_weights)
            group_shared = [t.to(device_id) for t in group_shared]
        else:
            group_shared, metadata, names = cuda_ipc_deserialize(serialized_weights)
        device_util.synchronize(device_id=device_util.current_device())
        tensors = reconstruct_tensors_from_groups(group_shared, metadata)
        device_util.synchronize(device_id=device_util.current_device())
        named_tensors = dict(zip(names, tensors))
        self.deserialized_weights = self._maybe_reconstruct_delta(
            named_tensors, step_id, device_id
        )
        logger.info(
            f"Deserialized {len(self.deserialized_weights)} parameters and {len(group_shared)} groups"
        )
        logger.info(
            f"GPU status after deserialization for rank {self.rank_coordinate}:\n{get_gpu_status()}"
        )
        logger.info(f"Open fds after deserialization: {count_open_fds()}")

    def _maybe_reconstruct_delta(self, named_tensors, step_id, device_id):
        """Reconstruct the full train-shard from a delta + derive change masks.

        Env-gated by ``AWEX_DELTA_TRANSFER``. On a delta payload this rebuilds
        the full local train-shard from the CPU ``_delta_base`` (needed for the
        local self-copy segment and as the base for the next version) AND
        derives the per-param change mask, stored on ``self._delta_masks``, which
        drives the *cross-rank* sparse P2P in ``_update_weights_in_colocate_mode``
        (only the changed elements cross the wire — the v1 mistake was sending
        the reconstructed full tensor cross-rank).

        Payloads are self-describing (a ``__awex_delta_header__`` tensor marks a
        delta); dense full-sync payloads (re)seed the base and clear the masks.
        Returns the full (dense-shaped) named tensor dict either way.
        """
        self._delta_masks = None  # default: dense/anchor step
        delta_enabled = os.environ.get("AWEX_DELTA_TRANSFER", "0") == "1"
        from awex.delta import (
            decode_delta_payload,
            is_delta_payload,
            reconstruct_against_base,
        )

        if not is_delta_payload(named_tensors):
            if delta_enabled:
                # Dense full sync: snapshot as the new base for later deltas.
                self._delta_base = {
                    name: t.detach().to("cpu", copy=True)
                    for name, t in named_tensors.items()
                }
                self._delta_base_version = step_id
                logger.info(
                    "Delta: seeded base from dense full sync at step %d (%d params)",
                    step_id,
                    len(self._delta_base),
                )
            return named_tensors

        # From here on the payload IS a delta.
        if not delta_enabled:
            raise RuntimeError(
                "Received a delta weight payload but AWEX_DELTA_TRANSFER is not "
                "enabled on the inference side; writer/reader env mismatch."
            )
        decoded = decode_delta_payload(named_tensors)
        header = decoded.header
        if self._delta_base_version is None or not self._delta_base:
            self._request_full_sync(step_id, "reader_base_missing")
            raise RuntimeError(
                f"Delta payload at step {step_id} but reader has no base "
                f"(base_version={self._delta_base_version}); requested full sync."
            )
        if header.base_version != self._delta_base_version:
            self._request_full_sync(step_id, "version_chain_broken")
            raise RuntimeError(
                f"Delta version chain broken at step {step_id}: payload base="
                f"{header.base_version}, reader base={self._delta_base_version}; "
                f"requested full sync."
            )

        start = time.time()
        try:
            result, counts = reconstruct_against_base(
                self._delta_base, decoded, device_id
            )
        except ValueError as exc:
            # e.g. a sparse patch for a name absent from the base: the full
            # shape is unknown and must not be dead-reckoned. Treat as a
            # broken chain and request a full sync.
            self._request_full_sync(step_id, "reconstruct_failed")
            raise RuntimeError(
                f"Delta reconstruction failed at step {step_id}: {exc}; "
                f"requested full sync."
            ) from exc
        self._delta_base_version = header.payload_version
        device_util.synchronize(device_id=device_util.current_device())
        # Build per-param change masks (for cross-rank sparse P2P). dense-
        # fallback params changed in full; sparse params changed at scattered
        # positions; unchanged params get no mask entry (False everywhere).
        masks = {}
        for name, full in result.items():
            if name in decoded.dense:
                masks[name] = torch.ones(
                    full.shape, dtype=torch.bool, device=full.device
                )
            elif name in decoded.sparse:
                indices, _ = decoded.sparse[name]
                m = torch.zeros(full.numel(), dtype=torch.bool, device=full.device)
                if indices.numel() > 0:
                    m[indices.to(full.device).long()] = True
                masks[name] = m.view(full.shape)
        self._delta_masks = masks
        logger.info(
            "Delta: reconstructed step %d (base=%d) sparse=%d dense=%d "
            "unchanged=%d, took %.3fs",
            step_id,
            header.base_version,
            counts["sparse"],
            counts["dense"],
            counts["unchanged"],
            time.time() - start,
        )
        return result

    def _full_sync_marker_key(self):
        """Per-(ip, device) marker key so each colocate pair is independent
        and the paired writer can delete it without racing other writers."""
        return (
            f"awex_delta_require_full_sync_{get_ip_address()}_"
            f"{device_util.current_device()}"
        )

    def _request_full_sync(self, step_id, reason):
        """Signal the paired writer to fall back to a dense full sync."""
        try:
            self.meta_server_client.put_object(
                self._full_sync_marker_key(),
                (self.rank_coordinate, step_id, reason),
            )
            logger.error(
                "Delta: requested full sync at step %d (reason=%s)", step_id, reason
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Delta: failed to request full sync: %s", exc)

    @staticmethod
    def _sync_non_contiguous_tensor_pairs(non_contiguous_tensor_pairs):
        if not non_contiguous_tensor_pairs:
            return
        with torch.no_grad():
            for original_tensor, recv_tensor in non_contiguous_tensor_pairs:
                original_tensor.copy_(recv_tensor)
            non_contiguous_tensor_pairs.clear()
            del non_contiguous_tensor_pairs

    def _update_weights(self, step_id, **kwargs):
        """
        Asynchronously receive weights from training ranks using torch.distributed.irecv.

        This method implements a pipelined approach where:
        1. For each sender rank, we maintain a queue of operations to receive
        2. We start irecv operations in parallel from all sender ranks
        3. When a receive completes, we immediately start the next receive from that rank
        4. We continue until all operations from all sender ranks are completed

        Args:
            step_id: The training step ID
            **kwargs: Additional keyword arguments (unused)
        """
        logger.info(
            f"Start to update weights using NCCL for step {step_id} from "
            f"{len(self.transfer_plan.operations)} ranks({self.send_ranks_sample}) "
            f"for rank {self.rank_coordinate}."
        )
        self._init_weights_exchange_process_group()
        start_time = time.time()

        # Build receive ops once for logging, then execute them via
        # batch_send_recv to keep scheduling consistent with the writer.
        p2p_op_list, non_contiguous_tensor_pairs, recv_traj_list = nccl_build_recv_ops(
            self.parameters,
            self.transfer_plan,
            self.weights_update_group,
            self.use_batch_send_recv,
        )
        logger.info(
            f"Reader: Built {len(p2p_op_list)} recv operations from "
            f"{len(self.transfer_plan.operations)} training ranks"
        )

        logger.info(
            f"Reader: Executing {len(p2p_op_list)} recv ops via batch_send_recv"
        )
        if self.use_batch_send_recv:
            batch_send_recv(
                send_ops=[], recv_ops=p2p_op_list, blocking=True, use_group=True
            )
        else:
            self._send_recv_one_by_one(p2p_op_list, recv_traj_list)

        self._sync_non_contiguous_tensor_pairs(non_contiguous_tensor_pairs)
        device_util.synchronize(device_id=device_util.current_device())
        duration = time.time() - start_time
        logger.info(
            f"Finished receiving weights for step {step_id} using NCCL "
            f"from {len(self.transfer_plan.operations)} ranks({self.send_ranks_sample}) "
            f"to rank {self.rank_coordinate} with {self.num_to_recvs} receives, took {duration:.4f} seconds"
        )
        compute_statistics(
            self._history_update_weights_time,
            step_id,
            duration,
            "Receive weights using NCCL",
        )
        dist.barrier(
            group=self.weights_update_group, device_ids=[device_util.current_device()]
        )
        logger.info(
            f"Barrier passed for reader step {step_id} with rank {self.transfer_rank}"
        )
        if p2p_op_list is not None:
            p2p_op_list.clear()
            del p2p_op_list
        self._destroy_weights_exchange_process_group()
        gc.collect()

    def _send_recv_one_by_one(self, p2p_op_list, recv_traj_list):
        # it's useful for debug or insufficient memory if infer with closed sleep mode
        # it's useful for the hardware diff scene which using batch_send_recv will be error,such as 910B2 and 910B1
        for (param_name, send_rank), op_dist in zip(recv_traj_list, p2p_op_list):
            logger.debug(
                f"Reader {self.transfer_rank} start to receive from {send_rank} for {param_name}"
            )
            non_contiguous_tensor_pair = None
            if not op_dist.tensor.is_contiguous():
                original_tensor = op_dist.tensor
                op_dist.tensor = original_tensor.contiguous()
                non_contiguous_tensor_pair = (original_tensor, op_dist.tensor)

            op_task = op_dist.op(
                op_dist.tensor,
                group=op_dist.group,
                tag=op_dist.tag,
                group_src=op_dist.group_peer,
            )
            op_task.wait()
            if not op_task.is_completed():
                device_util.synchronize(device_id=device_util.current_device())
                assert op_task.is_completed()
            if non_contiguous_tensor_pair is not None:
                with torch.no_grad():
                    non_contiguous_tensor_pair[0].copy_(non_contiguous_tensor_pair[1])
                del non_contiguous_tensor_pair
                del op_dist.tensor
            logger.debug(
                f"Reader {self.transfer_rank} end to receive from {send_rank} for {param_name}"
            )

    def _update_weights_in_colocate_mode(self, step_id, **kwargs):
        assert self.enable_colocate_mode, "Colocate mode is not enabled"
        self.collect_training_weights(step_id, **kwargs)
        logger.info(
            f"Start to update weights using NCCL for step {step_id} from {len(self.transfer_plan.operations)} "
            f"ranks({self.send_ranks_sample}) for rank {self.rank_coordinate}."
        )
        start_time = time.time()
        if self._delta_masks is not None:
            # Delta step: self-copy full locally + cross-rank sparse P2P.
            # Mixed-dtype models (bf16 body + fp32 MoE router) are supported by
            # grouping cross-rank ops by dtype inside apply_delta_colocate: each
            # uniform-dtype group runs its own two-round P2P (round 1 carries
            # only nnz, so the recv side must pre-allocate val buffers from a
            # single dtype per group). value_dtype=None tells apply_delta to
            # derive the per-group dtype itself.
            self.colocate_transport.apply_delta_colocate(
                self.train_to_infer_device_mapping,
                self.infer_to_train_device_mapping,
                self.transfer_rank,
                self.rank_coordinate,
                self.infer_world_size,
                self.send_transfer_plan,
                self.transfer_plan,
                self.weights_update_group,
                self.deserialized_weights,
                self._delta_masks,
                self.parameters,
                None,
                step_id=step_id,
            )
            self._delta_masks = None
        else:
            self.colocate_transport.update_weights_in_colocate_mode(
                self.train_to_infer_device_mapping,
                self.infer_to_train_device_mapping,
                self.transfer_rank,
                self.rank_coordinate,
                self.infer_world_size,
                self.send_transfer_plan,
                self.transfer_plan,
                self.weights_update_group,
                self.deserialized_weights,
                self.parameters,
                step_id=step_id,
            )
        print_current_gpu_status(
            f"after weights update using NCCL for rank {self.rank_coordinate}"
        )
        self.deserialized_weights = None
        duration = time.time() - start_time
        compute_statistics(
            self._history_update_weights_time,
            step_id,
            duration,
            "Receive weights using NCCL",
        )
        ip_address = get_ip_address()
        device_id = device_util.current_device()
        key_suffix = f"_{ip_address}_{device_id}_{step_id}"
        # Signal completion to training process
        update_finished_key = f"weights_update_finished{key_suffix}"
        self.meta_server_client.put_object(update_finished_key, True)
        dist.barrier(
            group=self.weights_update_group, device_ids=[device_util.current_device()]
        )
        logger.info(
            f"Barrier passed for reader step {step_id} with rank {self.transfer_rank}"
        )
        gc.collect()
        if device_util.get_device_type() == "cuda":
            torch.cuda.empty_cache()
        write_finished_key = f"write_finished{key_suffix}"
        self.meta_server_client.get_object_then_delete(write_finished_key)
        logger.info(
            f"Finished updating weights in colocate mode for rank {self.transfer_rank}"
        )
