# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.
#
# SPDX-License-Identifier: Apache-2.0

"""Awex SGLang adapter for colocated weight transfer.

This adapter runs inside the SGLang Scheduler process and provides:
- Weight metadata collection (with SGLang fused param unfusing)
- Colocated weight update via CUDA IPC + NCCL reshard
- Memory release/resume delegating to SGLang's native TMS
"""

from __future__ import annotations

import gc
import logging
import math
import os
import time
from typing import Any, Optional

import torch
import torch.distributed as dist

from awex.meta.weight_meta import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.models import get_infer_weights_converter
from awex.sharding.param_sharding import ShardingType
from awex.sharding.rank_info import RankInfo
from awex.sharding.sglang_sharding import (
    get_sglang_rank_info,
    get_sglang_sharding_strategy,
)
from awex.transfer.nccl_stream_batch import NcclColocateStreamBatchTransport
from awex.transfer.transfer_plan import TransferPlanBuilder
from awex.util.tensor_util import (
    cuda_ipc_deserialize,
    reconstruct_tensors_from_groups,
)

logger = logging.getLogger(__name__)


class AwexSGLangAdapter:
    """Awex inference-side adapter for colocated SGLang schedulers.

    Provides weight metadata collection, colocate IPC-based weight transfer,
    and memory management via SGLang's native release/resume mechanism.
    """

    def __init__(self, scheduler: Any):
        self._scheduler = scheduler
        self._transfer_rank: int | None = None
        self._rank_info: RankInfo | None = None
        self._parameters: dict[str, torch.Tensor] | None = None
        self._released_tags: set[str] = set()
        self._colocate_transport: NcclColocateStreamBatchTransport | None = None
        self._weights_update_group = None
        self._train_to_infer_device_mapping: dict | None = None
        self._infer_to_train_device_mapping: dict | None = None
        self._kv_store_url: str | None = None
        self._pair_name: str | None = None
        self._timeout_s: float = 120.0
        self._http_client = None
        self._weight_converter = None

    def _get_model(self) -> torch.nn.Module:
        return self._scheduler.tp_worker.model_runner.model

    def _get_model_arch_name(self) -> str:
        model = self._get_model()
        config = getattr(model, "config", None)
        architectures = getattr(config, "architectures", None)
        if architectures:
            return architectures[0]
        return type(model).__name__

    def _get_weight_converter(self):
        if self._weight_converter is None:
            rank_info = self._rank_info or self._build_rank_info()
            self._weight_converter = get_infer_weights_converter(
                "sglang",
                self._get_model_arch_name(),
                self._get_model().config,
                rank_info,
                self._scheduler.server_args,
            )
        return self._weight_converter

    def _get_model_context(self) -> dict[str, Any]:
        server_args = self._scheduler.server_args
        tp_size = int(getattr(server_args, "tp_size", 1))
        pp_size = int(getattr(server_args, "pp_size", 1))
        dp_size = int(getattr(server_args, "dp_size", 1))

        if dist.is_available() and dist.is_initialized():
            world_size = int(dist.get_world_size())
            global_rank = int(dist.get_rank())
        else:
            world_size = tp_size * pp_size
            global_rank = int(getattr(self._scheduler, "tp_rank", 0))

        return {
            "scheduler": self._scheduler,
            "tp_rank": int(getattr(self._scheduler, "tp_rank", 0)),
            "tp_size": tp_size,
            "pp_rank": int(getattr(self._scheduler, "pp_rank", 0)),
            "pp_size": pp_size,
            "dp_size": dp_size,
            "world_size": world_size,
            "global_rank": global_rank,
            "attn_tp_rank": int(
                getattr(self._scheduler, "attn_tp_rank",
                        getattr(self._scheduler, "tp_rank", 0))
            ),
            "attn_tp_size": int(getattr(self._scheduler, "attn_tp_size", tp_size)),
        }

    @property
    def parallelism_strategy(self) -> dict:
        ctx = self._get_model_context()
        server_args = self._scheduler.server_args
        return {
            "world_size": ctx["world_size"],
            "tp_size": int(getattr(server_args, "tp_size", ctx["tp_size"])),
            "pp_size": int(getattr(server_args, "pp_size", ctx["pp_size"])),
            "dp_size": int(getattr(server_args, "dp_size", ctx["dp_size"])),
            "ep_size": int(getattr(server_args, "ep_size", 1)),
            "num_engines": 1,
        }

    def _unfuse_params(
        self, name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Split SGLang fused parameters (qkv_proj, gate_up_proj, MoE experts)."""
        if self._get_model_arch_name() == "BailingMoeV3ForCausalLM":
            return self._get_weight_converter().convert_param(name, tensor)
        if "qkv_proj" in name:
            cfg = self._get_model().config
            num_heads = cfg.num_attention_heads
            num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
            total_head_units = num_heads + 2 * num_kv_heads
            dim0 = tensor.shape[0]
            q_size = dim0 * num_heads // total_head_units
            kv_size = dim0 * num_kv_heads // total_head_units
            return [
                (name.replace("qkv_proj", "q_proj"), tensor.narrow(0, 0, q_size)),
                (name.replace("qkv_proj", "k_proj"), tensor.narrow(0, q_size, kv_size)),
                (name.replace("qkv_proj", "v_proj"), tensor.narrow(0, q_size + kv_size, kv_size)),
            ]
        if "gate_up_proj" in name:
            half = tensor.shape[0] // 2
            return [
                (name.replace("gate_up_proj", "gate_proj"), tensor.narrow(0, 0, half)),
                (name.replace("gate_up_proj", "up_proj"), tensor.narrow(0, half, half)),
            ]
        return [(name, tensor)]

    def _build_rank_info(self) -> RankInfo:
        model_context = self._get_model_context()
        return get_sglang_rank_info(model_context, engine_rank=0)

    def _build_sharding_strategy(self, rank_info: RankInfo):
        model = self._get_model()
        model_config = getattr(model, "config", None)
        model_name = None
        if model_config is not None:
            architectures = getattr(model_config, "architectures", None)
            if architectures and len(architectures) > 0:
                model_name = architectures[0]
        if model_name is None:
            model_name = type(model).__name__
        return get_sglang_sharding_strategy(
            model_name, self._scheduler.server_args, rank_info
        )

    def get_weight_metadata(self) -> list[ParameterMeta]:
        rank_info = self._build_rank_info()
        strategy = self._build_sharding_strategy(rank_info)
        self._rank_info = rank_info

        metadata: list[ParameterMeta] = []
        for name, param in self._get_model().named_parameters():
            for hf_name, local_tensor in self._unfuse_params(name, param.data):
                local_shape = tuple(local_tensor.shape)
                sharding_type, sharding_dim, num_shards = (
                    strategy.get_sharding_strategy(hf_name)
                )

                global_offset = [0] * len(local_shape)
                if sharding_type == ShardingType.TP_SHARDING:
                    rank_pos = rank_info.tp_rank
                elif sharding_type == ShardingType.DP_TP_SHARDING:
                    rank_pos = rank_info.attn_tp_rank
                elif sharding_type == ShardingType.EP_SHARDING:
                    rank_pos = rank_info.ep_rank
                elif sharding_type == ShardingType.EP_TP_SHARDING:
                    rank_pos = rank_info.ep_tp_rank
                else:
                    rank_pos = 0

                if (
                    sharding_type != ShardingType.NO_SHARDING
                    and 0 <= sharding_dim < len(local_shape)
                ):
                    global_offset[sharding_dim] = int(rank_pos) * int(
                        local_shape[sharding_dim]
                    )

                global_shape = list(local_shape)
                if (
                    sharding_type != ShardingType.NO_SHARDING
                    and 0 <= sharding_dim < len(global_shape)
                ):
                    global_shape[sharding_dim] = (
                        int(local_shape[sharding_dim]) * int(num_shards)
                    )

                shard_meta = ParameterShardMeta(
                    tp_rank=rank_info.tp_rank,
                    attn_tp_rank=rank_info.attn_tp_rank,
                    pp_rank=rank_info.pp_rank,
                    ep_rank=rank_info.ep_rank,
                    ep_tp_rank=rank_info.ep_tp_rank,
                    global_rank=rank_info.global_rank,
                    world_size=rank_info.world_size,
                    engine_rank=rank_info.engine_rank,
                    cp_rank=rank_info.cp_rank,
                    cp_size=rank_info.cp_size,
                    cp_mode=rank_info.cp_mode,
                    name=hf_name,
                    shape=local_shape,
                    numel=int(local_tensor.numel()),
                    dtype=local_tensor.dtype,
                    global_offset=tuple(global_offset),
                    sharding_type=sharding_type,
                    num_shards=int(num_shards),
                    sharding_dim=int(sharding_dim),
                )
                replica = ParameterReplicaMeta(shards=[shard_meta])
                metadata.append(
                    ParameterMeta(
                        name=hf_name,
                        global_numel=math.prod(global_shape) if global_shape else 1,
                        global_shape=tuple(global_shape),
                        dtype=local_tensor.dtype,
                        shards=[shard_meta],
                        replicas=[replica],
                    )
                )
        return metadata

    def get_local_shard_parameters(
        self, required_names: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        required = set(required_names) if required_names else None
        local_params: dict[str, torch.Tensor] = {}
        for name, param in self._get_model().named_parameters():
            for hf_name, hf_tensor in self._unfuse_params(name, param.data):
                if required is None or hf_name in required:
                    local_params[hf_name] = hf_tensor
        self._parameters = local_params
        return local_params

    def init_colocate_weight_update(
        self,
        pair_name: str,
        kv_store_url: str,
        transfer_rank: int,
        infer_world_size: int,
        train_world_size: int,
        num_engines: int = 1,
        master_port: int = 29600,
        timeout_s: float = 120.0,
    ) -> None:
        if infer_world_size != train_world_size:
            raise ValueError(
                f"Colocate mode requires infer_world_size == train_world_size, "
                f"got {infer_world_size} vs {train_world_size}"
            )
        self._pair_name = pair_name
        self._kv_store_url = kv_store_url
        self._transfer_rank = transfer_rank
        self._timeout_s = timeout_s

        try:
            import httpx
            self._http_client = httpx.Client()
        except ImportError:
            import urllib.request
            self._http_client = None

        from awex.meta.meta_server import MetaServerClient
        infer_meta_key = f"infer_params_meta_{pair_name}"
        train_meta_key = f"training_params_meta_{pair_name}"

        meta_client = MetaServerClient(kv_store_url)
        infer_meta = meta_client.get_object(infer_meta_key)
        train_meta = meta_client.get_object(train_meta_key)

        builder = TransferPlanBuilder(
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            num_infer_engines=num_engines,
        )

        train_to_infer = {}
        infer_to_train = {}
        for i in range(min(infer_world_size, train_world_size)):
            train_rank = infer_world_size + i
            train_to_infer[train_rank] = i
            infer_to_train[i] = train_rank
        self._train_to_infer_device_mapping = train_to_infer
        self._infer_to_train_device_mapping = infer_to_train

        self._send_transfer_plan = builder.build_local_transfer_plan(
            infer_meta, train_meta,
            global_transfer_rank=infer_to_train[transfer_rank],
        )
        self._recv_transfer_plan = builder.build_local_transfer_plan(
            infer_meta, train_meta,
            global_transfer_rank=transfer_rank,
        )

        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
        from awex.transfer.nccl_comm import init_weights_update_group
        self._weights_update_group = init_weights_update_group(
            master_address="127.0.0.1",
            master_port=master_port,
            rank=transfer_rank,
            world_size=infer_world_size,
            group_name=f"awex_colocate_{pair_name}",
        )
        self._colocate_transport = NcclColocateStreamBatchTransport(
            transfer_rank, infer_world_size
        )
        logger.info(
            "Initialized colocate weight update: pair=%s, transfer_rank=%d, "
            "infer_world_size=%d",
            pair_name, transfer_rank, infer_world_size,
        )

    def execute_colocate_weight_update(self, version: int) -> None:
        assert self._kv_store_url is not None, (
            "init_colocate_weight_update must be called first"
        )
        paired_train_rank = self._infer_to_train_device_mapping[self._transfer_rank]
        kv_key = f"colocate_weights_rank{paired_train_rank}_{version}"

        from awex.meta.meta_server import MetaServerClient
        meta_client = MetaServerClient(self._kv_store_url)
        serialized_weights = meta_client.get_object(kv_key, timeout=self._timeout_s)

        group_shared, metadata, names = cuda_ipc_deserialize(serialized_weights)
        torch.cuda.synchronize()
        tensors = reconstruct_tensors_from_groups(group_shared, metadata)
        torch.cuda.synchronize()
        deserialized_weights = dict(zip(names, tensors))

        recv_parameters = self.get_local_shard_parameters()
        rank_info = self._build_rank_info()
        rank_coordinate = f"infer_{rank_info.global_rank}"

        self._colocate_transport.update_weights_in_colocate_mode(
            self._train_to_infer_device_mapping,
            self._infer_to_train_device_mapping,
            self._transfer_rank,
            rank_coordinate,
            len(self._infer_to_train_device_mapping),
            self._send_transfer_plan,
            self._recv_transfer_plan,
            self._weights_update_group,
            deserialized_weights,
            recv_parameters,
            step_id=version,
        )

        done_key = f"colocate_done_rank{paired_train_rank}_{version}"
        meta_client.put_object(done_key, True)

        del deserialized_weights, group_shared, tensors
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(
            "Colocate weight update completed: version=%d, rank=%d",
            version, self._transfer_rank,
        )

    def release_memory(self, tags: Optional[list[str]] = None) -> None:
        from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput
        tags = tags or ["kv_cache"]
        native_tags = [t for t in tags if t not in self._released_tags]
        if native_tags:
            req = ReleaseMemoryOccupationReqInput(tags=native_tags)
            self._scheduler.release_memory_occupation(req)
            self._released_tags.update(native_tags)
        logger.info("release_memory completed: tags=%s", tags)

    def resume_memory(self, tags: Optional[list[str]] = None) -> None:
        from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput
        tags = tags or ["kv_cache"]
        resume_tags = [t for t in tags if t in self._released_tags]
        if resume_tags:
            req = ResumeMemoryOccupationReqInput(tags=resume_tags)
            self._scheduler.resume_memory_occupation(req)
            self._released_tags.difference_update(resume_tags)
        logger.info("resume_memory completed: tags=%s", tags)

    def teardown(self) -> None:
        if self._weights_update_group is not None and dist.is_initialized():
            dist.destroy_process_group(self._weights_update_group)
        self._weights_update_group = None
        self._colocate_transport = None
        self._parameters = None
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None
