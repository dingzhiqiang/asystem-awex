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

import os
import queue

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import awex.transfer.nccl_stream_batch as stream_batch
from awex.meta.weight_meta import ParameterShardMeta
from awex.sharding.param_sharding import ShardingType
from awex.transfer.nccl_stream_batch import (
    NcclColocateStreamBatchTransport,
    _clone_p2p_send_tensor,
    _is_expert_transfer_op,
    _pack_p2p_send_tensors,
    _packed_recv_staging_bytes,
    _partition_expert_p2p_entries,
    _prepare_p2p_recv_tensor,
    _prepare_packed_p2p_recv_tensor,
    _sync_p2p_recv_tensor_pairs,
)
from awex.transfer.transfer_plan import CommunicationOperation, TransferPlan


def test_prepare_p2p_recv_tensor_uses_dense_buffer_for_noncontiguous_view():
    base = torch.zeros(4, 4)
    view = base[:, 1]
    assert not view.is_contiguous()

    recv_tensor, copyback_pair = _prepare_p2p_recv_tensor(view)

    assert recv_tensor.is_contiguous()
    assert copyback_pair is not None
    recv_tensor.copy_(torch.arange(4, dtype=base.dtype))

    _sync_p2p_recv_tensor_pairs([copyback_pair])

    torch.testing.assert_close(base[:, 1], torch.arange(4, dtype=base.dtype))
    torch.testing.assert_close(base[:, 0], torch.zeros(4))


def test_prepare_p2p_recv_tensor_reuses_contiguous_tensor():
    tensor = torch.zeros(4)

    recv_tensor, copyback_pair = _prepare_p2p_recv_tensor(tensor)

    assert recv_tensor is tensor
    assert copyback_pair is None


def test_clone_p2p_send_tensor_returns_contiguous_clone():
    base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    view = base[:, 1]
    assert not view.is_contiguous()

    cloned = _clone_p2p_send_tensor(view)

    assert cloned.is_contiguous()
    assert cloned.data_ptr() != view.data_ptr()
    torch.testing.assert_close(cloned, view)


def test_pack_p2p_send_tensors_is_dense_independent_and_casts_dtype():
    base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    noncontiguous = base[:, 1]
    other = torch.tensor([20.0, 21.0], dtype=torch.float32)

    packed = _pack_p2p_send_tensors([noncontiguous, other], wire_dtype=torch.float64)

    assert packed.is_contiguous()
    assert packed.dtype == torch.float64
    torch.testing.assert_close(
        packed,
        torch.tensor([1.0, 5.0, 9.0, 13.0, 20.0, 21.0], dtype=torch.float64),
    )
    base.fill_(-1)
    other.fill_(-1)
    torch.testing.assert_close(
        packed,
        torch.tensor([1.0, 5.0, 9.0, 13.0, 20.0, 21.0], dtype=torch.float64),
    )


def test_prepare_packed_p2p_recv_tensor_copies_back_to_mixed_views():
    base = torch.zeros(4, 4)
    noncontiguous = base[:, 1]
    contiguous = torch.zeros(2)

    packed, copyback_pairs = _prepare_packed_p2p_recv_tensor(
        [noncontiguous, contiguous], wire_dtype=torch.float32
    )
    packed.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0, 8.0, 9.0]))

    _sync_p2p_recv_tensor_pairs(copyback_pairs)

    torch.testing.assert_close(base[:, 1], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(contiguous, torch.tensor([8.0, 9.0]))
    torch.testing.assert_close(base[:, 0], torch.zeros(4))


def _make_transfer_op(
    name: str,
    recv_dtype: torch.dtype = torch.float32,
    *,
    numel: int = 2,
    send_rank: int = 1,
    recv_rank: int = 0,
) -> CommunicationOperation:
    send_shard = ParameterShardMeta(
        tp_rank=0,
        attn_tp_rank=0,
        pp_rank=0,
        ep_rank=0,
        ep_tp_rank=0,
        global_rank=0,
        world_size=1,
        engine_rank=0,
        name=name,
        shape=(numel,),
        numel=numel,
        dtype=torch.float32,
        sharding_type=ShardingType.NO_SHARDING,
    )
    recv_shard = ParameterShardMeta(
        tp_rank=0,
        attn_tp_rank=0,
        pp_rank=0,
        ep_rank=0,
        ep_tp_rank=0,
        global_rank=0,
        world_size=1,
        engine_rank=0,
        name=name,
        shape=(numel,),
        numel=numel,
        dtype=recv_dtype,
        sharding_type=ShardingType.NO_SHARDING,
    )
    return CommunicationOperation(
        send_rank=send_rank,
        send_shard_meta=send_shard,
        send_offset=(0,),
        recv_rank=recv_rank,
        recv_shard_meta=recv_shard,
        recv_offset=(0,),
        overlap_shape=(numel,),
        train_slices=(slice(0, numel),),
        inf_slices=(slice(0, numel),),
    )


def test_is_expert_transfer_op_requires_matching_routed_expert_names():
    expert_name = "model.layers.2.mlp.experts.7.down_proj.weight"
    expert_op = _make_transfer_op(expert_name)
    dense_op = _make_transfer_op("model.layers.2.mlp.down_proj.weight")

    assert _is_expert_transfer_op(expert_op)
    assert not _is_expert_transfer_op(dense_op)


def test_partition_expert_p2p_entries_preserves_direct_order_and_sorts_dtypes():
    dense_op = _make_transfer_op("model.layers.2.mlp.down_proj.weight")
    expert_fp64 = _make_transfer_op(
        "model.layers.2.mlp.experts.8.down_proj.weight", torch.float64
    )
    expert_fp32_a = _make_transfer_op(
        "model.layers.2.mlp.experts.7.gate_proj.weight", torch.float32
    )
    expert_fp32_b = _make_transfer_op(
        "model.layers.2.mlp.experts.7.up_proj.weight", torch.float32
    )
    entries = [
        (expert_fp64, torch.zeros(2)),
        (dense_op, torch.ones(2)),
        (expert_fp32_a, torch.full((2,), 2.0)),
        (expert_fp32_b, torch.full((2,), 3.0)),
    ]

    direct, packed = _partition_expert_p2p_entries(entries, pack_experts=True)

    assert [op for op, _ in direct] == [dense_op]
    assert [dtype for dtype, _ in packed] == [torch.float32, torch.float64]
    assert [op for op, _ in packed[0][1]] == [expert_fp32_a, expert_fp32_b]
    assert [op for op, _ in packed[1][1]] == [expert_fp64]

    direct, packed = _partition_expert_p2p_entries(entries, pack_experts=False)
    assert direct == entries
    assert packed == []


def test_packed_recv_staging_bytes_counts_only_experts_at_wire_dtype():
    expert_fp32 = _make_transfer_op(
        "model.layers.2.mlp.experts.7.gate_proj.weight",
        torch.float32,
        numel=3,
    )
    expert_fp64 = _make_transfer_op(
        "model.layers.2.mlp.experts.8.down_proj.weight",
        torch.float64,
        numel=5,
    )
    dense = _make_transfer_op(
        "model.layers.2.mlp.down_proj.weight", torch.float64, numel=11
    )

    assert _packed_recv_staging_bytes([expert_fp32, expert_fp64, dense]) == 52


def _gloo_packed_transfer_worker(rank, init_file, staging_cap_mb, result_queue):
    os.environ["AWEX_DEVICE_TYPE"] = "cpu"
    os.environ["AWEX_PACK_EXPERT_P2P"] = "1"
    os.environ["AWEX_CHUNK_OPS"] = "4"
    os.environ["AWEX_PACK_RECV_STAGING_MB"] = staging_cap_mb
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )

    send_pack_calls = []
    recv_pack_calls = []
    original_send_pack = stream_batch._pack_p2p_send_tensors
    original_recv_pack = stream_batch._prepare_packed_p2p_recv_tensor

    def tracked_send_pack(tensors, wire_dtype):
        send_pack_calls.append((str(wire_dtype), tuple(t.numel() for t in tensors)))
        return original_send_pack(tensors, wire_dtype)

    def tracked_recv_pack(tensors, wire_dtype):
        recv_pack_calls.append((str(wire_dtype), tuple(t.numel() for t in tensors)))
        return original_recv_pack(tensors, wire_dtype)

    stream_batch._pack_p2p_send_tensors = tracked_send_pack
    stream_batch._prepare_packed_p2p_recv_tensor = tracked_recv_pack
    try:
        specs = [
            ("model.layers.2.mlp.experts.0.gate_proj.weight", torch.float32, 2),
            ("model.layers.2.mlp.experts.0.up_proj.weight", torch.float32, 3),
            ("model.layers.2.mlp.experts.1.down_proj.weight", torch.float64, 4),
            ("model.layers.2.mlp.down_proj.weight", torch.float32, 7),
            ("model.layers.3.mlp.experts.2.down_proj.weight", torch.float32, 5),
            ("model.layers.3.mlp.down_proj.weight", torch.float32, 6),
        ]
        ops = [
            _make_transfer_op(
                name,
                recv_dtype,
                numel=numel,
                send_rank=0,
                recv_rank=1,
            )
            for name, recv_dtype, numel in specs
        ]
        send_parameters = {
            op.send_shard_meta.name: torch.arange(
                op.send_shard_meta.numel, dtype=torch.float32
            )
            + index * 10
            for index, op in enumerate(ops)
        }
        recv_parameters = {
            op.recv_shard_meta.name: torch.zeros(
                op.recv_shard_meta.numel, dtype=op.recv_shard_meta.dtype
            )
            for op in ops
        }

        transport = NcclColocateStreamBatchTransport(rank, 2)
        transport._run_chunked(
            task_id="gloo-pack-test",
            step_id=0,
            train_to_infer_device_mapping={0: 0, 1: 1},
            infer_to_train_device_mapping={0: 0, 1: 1},
            transfer_rank=rank,
            rank_coordinate=f"rank-{rank}",
            world_size=2,
            send_ops={1: ops} if rank == 0 else {},
            recv_ops={0: ops} if rank == 1 else {},
            recv_transfer_plan=TransferPlan(operations={0: ops} if rank == 1 else {}),
            weights_update_group=dist.group.WORLD,
            send_parameters=send_parameters if rank == 0 else {},
            recv_parameters=recv_parameters if rank == 1 else {},
            async_op=True,
            chunk_bytes=1024 * 1024,
        )
        result_queue.put(
            {
                "rank": rank,
                "send_pack_calls": send_pack_calls,
                "recv_pack_calls": recv_pack_calls,
                "received": {
                    name: tensor.tolist() for name, tensor in recv_parameters.items()
                }
                if rank == 1
                else {},
            }
        )
    finally:
        stream_batch._pack_p2p_send_tensors = original_send_pack
        stream_batch._prepare_packed_p2p_recv_tensor = original_recv_pack
        dist.destroy_process_group()


def _run_two_rank_gloo_transfer(tmp_path, staging_cap_mb):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    init_file = str(tmp_path / "gloo-store")
    processes = [
        context.Process(
            target=_gloo_packed_transfer_worker,
            args=(rank, init_file, staging_cap_mb, result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
    hanging = [process for process in processes if process.is_alive()]
    for process in hanging:
        process.terminate()
        process.join(timeout=5)
    assert not hanging, "Two-rank packed transfer hung"
    assert [process.exitcode for process in processes] == [0, 0]

    try:
        results = [result_queue.get(timeout=5) for _ in range(2)]
    except queue.Empty:
        pytest.fail("Two-rank packed transfer did not return both results")
    return {result["rank"]: result for result in results}


def test_run_chunked_gloo_preserves_mixed_dtype_fifo_across_chunks(tmp_path):
    results = _run_two_rank_gloo_transfer(tmp_path, staging_cap_mb="1")
    expected_calls = [
        ("torch.float32", (2, 3)),
        ("torch.float64", (4,)),
        ("torch.float32", (5,)),
    ]
    assert results[0]["send_pack_calls"] == expected_calls
    assert results[1]["recv_pack_calls"] == expected_calls
    assert results[0]["recv_pack_calls"] == []
    assert results[1]["send_pack_calls"] == []

    for index, values in enumerate(results[1]["received"].values()):
        assert values == pytest.approx(
            [index * 10 + offset for offset in range(len(values))]
        )


def test_run_chunked_gloo_disables_packing_globally_above_staging_cap(tmp_path):
    results = _run_two_rank_gloo_transfer(tmp_path, staging_cap_mb="0.000001")
    assert results[0]["send_pack_calls"] == []
    assert results[0]["recv_pack_calls"] == []
    assert results[1]["send_pack_calls"] == []
    assert results[1]["recv_pack_calls"] == []

    for index, values in enumerate(results[1]["received"].values()):
        assert values == pytest.approx(
            [index * 10 + offset for offset in range(len(values))]
        )
