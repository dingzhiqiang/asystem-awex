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

from types import SimpleNamespace

import torch
from transformers import PretrainedConfig

from awex.models import get_infer_weights_converter, get_sharding_strategy
from awex.models.registry import get_train_weights_converter
from awex.sharding.param_sharding import ShardingType
from awex.sharding.rank_info import RankInfo


def _make_rank_info() -> RankInfo:
    return RankInfo(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        dp_size=1,
        dp_rank=0,
        ep_rank=0,
        ep_size=1,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_dp_rank=0,
        world_size=1,
        global_rank=0,
        local_rank=0,
        engine_rank=0,
        is_infer=False,
    )


def _make_bailing_v3_config() -> PretrainedConfig:
    cfg = PretrainedConfig()
    cfg.architectures = ["BailingMoeV3ForCausalLM"]
    cfg.quantization_config = {}
    cfg.num_hidden_layers = 4
    cfg.hidden_size = 8
    cfg.num_attention_heads = 2
    cfg.num_key_value_heads = 2
    cfg.head_dim = 2
    cfg.v_head_dim = 2
    cfg.layer_group_size = 4
    cfg.num_experts = 4
    return cfg


def test_bailing_moe_train_converter_accepts_tf_config():
    cfg = PretrainedConfig()
    cfg.quantization_config = {}
    rank_info = _make_rank_info()
    infer_conf = {"infer_atten_tp_size": 1}
    tf_config = SimpleNamespace()

    converter = get_train_weights_converter(
        "mcore",
        "BailingMoeForCausalLM",
        cfg,
        rank_info,
        infer_conf,
        tf_config=tf_config,
    )

    assert converter.tf_config is tf_config


def test_bailing_linear_train_converter_accepts_tf_config():
    cfg = PretrainedConfig()
    cfg.quantization_config = {}
    rank_info = _make_rank_info()
    infer_conf = {"infer_atten_tp_size": 1}
    tf_config = SimpleNamespace(layer_group_size=4)

    converter = get_train_weights_converter(
        "mcore",
        "BailingMoeV2_5ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
        tf_config=tf_config,
    )

    assert converter.tf_config is tf_config


def test_bailing_linear_sharding_strategy_handles_mla_exceptions():
    rank_info = _make_rank_info()
    rank_info.tp_size = 2
    strategy_cls = get_sharding_strategy("BailingMoeV2_5ForCausalLM")
    strategy = strategy_cls(
        engine_name="sglang",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=2,
        tp_size=2,
        ep_size=1,
        ep_tp_size=1,
        rank_info=rank_info,
    )

    assert strategy.get_sharding_strategy("model.layers.0.attention.g_norm.weight") == (
        ShardingType.TP_SHARDING,
        0,
        2,
    )
    assert strategy.get_sharding_strategy(
        "model.layers.0.attention.kv_a_proj_with_mqa.weight"
    ) == (ShardingType.NO_SHARDING, 0, 1)


def test_bailing_v3_train_converter_accepts_tf_config():
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    infer_conf = {"infer_atten_tp_size": 1}
    tf_config = SimpleNamespace(layer_group_size=4)

    converter = get_train_weights_converter(
        "mcore",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
        tf_config=tf_config,
    )

    assert converter.tf_config is tf_config
    assert converter.layer_group_size == 4


def test_bailing_v3_train_converter_splits_kda_attention(monkeypatch):
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    infer_conf = {"infer_atten_tp_size": 1}
    tf_config = SimpleNamespace(layer_group_size=4)
    converter = get_train_weights_converter(
        "mcore",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
        tf_config=tf_config,
    )
    monkeypatch.setattr(
        "awex.models.ling_v3.get_full_tensor",
        lambda tensor, dim=0: tensor,
    )

    parameter = torch.arange(40, dtype=torch.float32).reshape(20, 2)
    converted = converter.convert_param(
        "decoder.layers.0.self_attention.in_proj.weight",
        parameter,
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.attention.q_proj.weight",
        "model.layers.0.attention.k_proj.weight",
        "model.layers.0.attention.v_proj.weight",
        "model.layers.0.attention.f_proj.weight",
        "model.layers.0.attention.g_proj.weight",
    ]
    for idx, (_, tensor) in enumerate(converted):
        torch.testing.assert_close(tensor, parameter[idx * 4 : (idx + 1) * 4])


def test_bailing_v3_sglang_converter_splits_fused_kda_attention():
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    infer_conf = SimpleNamespace(tp_size=1, ep_size=1)
    converter = get_infer_weights_converter(
        "sglang",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
    )

    parameter = torch.arange(40, dtype=torch.float32).reshape(20, 2)
    converted = converter.convert_param(
        "model.layers.0.attention.fused_qkvbfg_proj.weight",
        parameter,
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.attention.q_proj.weight",
        "model.layers.0.attention.k_proj.weight",
        "model.layers.0.attention.v_proj.weight",
        "model.layers.0.attention.f_proj.weight",
        "model.layers.0.attention.g_proj.weight",
    ]
    for idx, (_, tensor) in enumerate(converted):
        torch.testing.assert_close(tensor, parameter[idx * 4 : (idx + 1) * 4])


def test_bailing_v3_sglang_converter_splits_rel_fused_kda_attention_with_beta():
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    infer_conf = SimpleNamespace(tp_size=1, ep_size=1)
    converter = get_infer_weights_converter(
        "sglang",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
    )

    parameter = torch.arange(44, dtype=torch.float32).reshape(22, 2)
    converted = converter.convert_param(
        "model.layers.0.attention.fused_qkvbfg_proj.weight",
        parameter,
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.attention.q_proj.weight",
        "model.layers.0.attention.k_proj.weight",
        "model.layers.0.attention.v_proj.weight",
        "model.layers.0.attention.b_proj.weight",
        "model.layers.0.attention.f_proj.weight",
        "model.layers.0.attention.g_proj.weight",
    ]
    starts_and_ends = [(0, 4), (4, 8), (8, 12), (12, 14), (14, 18), (18, 22)]
    for (_, tensor), (start, end) in zip(converted, starts_and_ends):
        torch.testing.assert_close(tensor, parameter[start:end])


def test_bailing_v3_sglang_converter_splits_local_qkv_conv_tp():
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    rank_info.tp_size = 2
    infer_conf = SimpleNamespace(tp_size=2, ep_size=1)
    converter = get_infer_weights_converter(
        "sglang",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
    )

    parameter = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    converted = converter.convert_param(
        "model.layers.0.attention.qkv_conv1d.weight",
        parameter,
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.attention.q_conv1d.weight",
        "model.layers.0.attention.k_conv1d.weight",
        "model.layers.0.attention.v_conv1d.weight",
    ]
    for idx, (_, tensor) in enumerate(converted):
        torch.testing.assert_close(tensor, parameter[idx * 2 : (idx + 1) * 2])


def test_bailing_v3_sglang_converter_splits_fused_qkvbfg_a_tp():
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    rank_info.tp_size = 2
    infer_conf = SimpleNamespace(tp_size=2, ep_size=1)
    converter = get_infer_weights_converter(
        "sglang",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
    )

    parameter = torch.arange(22, dtype=torch.float32).reshape(11, 2)
    converted = converter.convert_param(
        "model.layers.0.attention.fused_qkvbfg_a_proj.weight",
        parameter,
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.attention.q_proj.weight",
        "model.layers.0.attention.k_proj.weight",
        "model.layers.0.attention.v_proj.weight",
        "model.layers.0.attention.b_proj.weight",
        "model.layers.0.attention.f_a_proj.weight",
        "model.layers.0.attention.g_a_proj.weight",
    ]
    offsets = [0, 2, 4, 6, 7, 9]
    sections = [2, 2, 2, 1, 2, 2]
    for (_, tensor), offset, section in zip(converted, offsets, sections):
        torch.testing.assert_close(tensor, parameter[offset : offset + section])


def test_bailing_v3_sglang_converter_splits_fused_fg_b():
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    infer_conf = SimpleNamespace(tp_size=1, ep_size=1)
    converter = get_infer_weights_converter(
        "sglang",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
    )

    parameter = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    converted = converter.convert_param(
        "model.layers.0.attention.fused_fg_b_proj.weight_scale_inv",
        parameter,
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.attention.f_b_proj.weight_scale_inv",
        "model.layers.0.attention.g_b_proj.weight_scale_inv",
    ]
    torch.testing.assert_close(converted[0][1], parameter[0])
    torch.testing.assert_close(converted[1][1], parameter[1])


def test_bailing_v3_sglang_converter_normalizes_word_embeddings_and_a_log():
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    infer_conf = SimpleNamespace(tp_size=1, ep_size=1)
    converter = get_infer_weights_converter(
        "sglang",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
    )

    embedding = torch.randn(4, 3)
    assert converter.convert_param("model.word_embeddings.weight", embedding) == [
        ("model.word_embeddings.weight", embedding)
    ]

    a_log = torch.randn(1, 1, 2, 1)
    converted = converter.convert_param("model.layers.0.attention.A_log", a_log)
    assert converted[0][0] == "model.layers.0.attention.A_log"
    torch.testing.assert_close(converted[0][1], a_log.squeeze())


def test_bailing_v3_sglang_converter_maps_o_proj_by_layer_type():
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    infer_conf = SimpleNamespace(tp_size=1, ep_size=1)
    converter = get_infer_weights_converter(
        "sglang",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
    )

    parameter = torch.randn(4, 4)

    kda_converted = converter.convert_param(
        "model.layers.0.attention.o_proj.weight",
        parameter,
    )
    assert kda_converted == [("model.layers.0.attention.o_proj.weight", parameter)]

    mla_converted = converter.convert_param(
        "model.layers.3.attention.o_proj.weight",
        parameter,
    )
    assert mla_converted == [("model.layers.3.attention.dense.weight", parameter)]

    scale_converted = converter.convert_param(
        "model.layers.3.attention.o_proj.weight_scale_inv",
        parameter,
    )
    assert scale_converted == [
        ("model.layers.3.attention.dense.weight_scale_inv", parameter)
    ]


def test_bailing_v3_sharding_strategy_handles_kda_and_mla_params():
    rank_info = _make_rank_info()
    rank_info.tp_size = 2
    strategy_cls = get_sharding_strategy("BailingMoeV3ForCausalLM")
    strategy = strategy_cls(
        engine_name="sglang",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=2,
        tp_size=2,
        ep_size=1,
        ep_tp_size=1,
        rank_info=rank_info,
    )

    assert strategy.get_sharding_strategy(
        "model.layers.0.attention.fused_qkvbfg_proj.weight"
    ) == (ShardingType.TP_SHARDING, 0, 2)
    assert strategy.get_sharding_strategy(
        "model.layers.0.attention.qkv_conv1d.weight"
    ) == (ShardingType.TP_SHARDING, 0, 2)
    assert strategy.get_sharding_strategy(
        "model.layers.0.attention.f_a_proj.weight"
    ) == (ShardingType.NO_SHARDING, 0, 1)
    assert strategy.get_sharding_strategy(
        "model.layers.0.attention.g_a_proj.weight"
    ) == (ShardingType.NO_SHARDING, 0, 1)
    assert strategy.get_sharding_strategy(
        "model.layers.0.attention.f_b_proj.weight"
    ) == (ShardingType.TP_SHARDING, 0, 2)
    assert strategy.get_sharding_strategy(
        "model.layers.3.attention.kv_a_proj_with_mqa.weight"
    ) == (ShardingType.NO_SHARDING, 0, 1)


def test_bailing_v3_sglang_adapter_uses_registry_converter():
    from awex.sglang_awex_adapter import AwexSGLangAdapter

    cfg = _make_bailing_v3_config()
    server_args = SimpleNamespace(tp_size=1, pp_size=1, dp_size=1, ep_size=1)
    model = SimpleNamespace(config=cfg)
    scheduler = SimpleNamespace(
        server_args=server_args,
        tp_rank=0,
        pp_rank=0,
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=model,
            ),
        ),
    )
    adapter = AwexSGLangAdapter(scheduler)
    adapter._rank_info = _make_rank_info()

    parameter = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    converted = adapter._unfuse_params(
        "model.layers.0.attention.fused_fg_b_proj.weight",
        parameter,
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.attention.f_b_proj.weight",
        "model.layers.0.attention.g_b_proj.weight",
    ]
    torch.testing.assert_close(converted[0][1], parameter[0])
    torch.testing.assert_close(converted[1][1], parameter[1])
