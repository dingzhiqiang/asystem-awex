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

import pytest
import torch

from awex.models.glm5_next import (
    Glm5NextShardingStrategy,
    SGlangToHFWeightConverterGlm5Next,
    build_mcore_converter,
)
from awex.models.registry import get_infer_weights_converter
from awex.sharding.param_sharding import ShardingType


@pytest.fixture
def config():
    return SimpleNamespace(
        architectures=["Glm5NextForConditionalGeneration"],
        text_config=SimpleNamespace(
            num_attention_heads=64,
            num_key_value_heads=64,
            n_routed_experts=288,
            num_hidden_layers=45,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            moe_router_dtype="float32",
            linear_attn_config={"num_heads": 64, "head_dim": 128},
            layer_types=[
                "full_attention" if i % 4 == 3 else "linear_attention"
                for i in range(45)
            ],
        ),
        quantization_config={},
    )


def rank(tp=4, ep=32, **overrides):
    return SimpleNamespace(
        **{
            "tp_size": tp,
            "tp_rank": 0,
            "attn_tp_size": tp,
            "attn_tp_rank": 0,
            "ep_size": ep,
            "ep_rank": 0,
            "ep_tp_size": 1,
            "ep_tp_rank": 0,
            "pp_size": 1,
            "pp_rank": 0,
            "cp_size": 1,
            **overrides,
        }
    )


def train(config, ranks=None, **infer):
    return build_mcore_converter()(
        config, ranks or rank(), {"infer_atten_tp_size": 8, **infer}, SimpleNamespace()
    )


def infer(config, tp=8, ep=8, **ranks):
    return SGlangToHFWeightConverterGlm5Next(
        config,
        SimpleNamespace(tp_size=tp, ep_size=ep, device_backend="cpu"),
        rank(tp, ep, **ranks),
    )


def strategy(engine, ranks, **options):
    return Glm5NextShardingStrategy(
        engine_name=engine,
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=1,
        tp_size=ranks.tp_size,
        ep_size=ranks.ep_size,
        ep_tp_size=1,
        rank_info=ranks,
        **options,
    )


def fp8_config(**overrides):
    return {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
        **overrides,
    }


def test_registry_selects_glm_converter_for_nested_config(config):
    converter = get_infer_weights_converter(
        "sglang",
        "Glm5NextForConditionalGeneration",
        config,
        rank(8, 8),
        SimpleNamespace(tp_size=8, ep_size=8, device_backend="cpu"),
    )
    assert isinstance(converter, SGlangToHFWeightConverterGlm5Next)
    assert converter.model_config.num_experts == 288


@pytest.mark.parametrize("site", ["attn", "ffn"])
def test_mhc_scale_keeps_name_three_fp32_values_and_receiver_storage(config, site):
    source = "self_attention" if site == "attn" else "mlp"
    values = torch.tensor([0.12500001, -0.37500003, 0.75000006], dtype=torch.float32)
    name, value = train(config).convert_param(
        f"decoder.layers.0.{source}_hyper_connection.scale", values
    )[0]
    assert name == f"model.layers.0.hc_{site}_scale"
    receiver = torch.zeros(3, dtype=torch.float32)
    target_name, target = infer(config).convert_param(name, receiver)[0]
    assert target_name == name
    target.copy_(value)
    torch.testing.assert_close(receiver, values, rtol=0, atol=0)
    with pytest.raises(ValueError, match="three"):
        infer(config).convert_param(name, torch.zeros(1))
    with pytest.raises(ValueError, match="FP32"):
        infer(config).convert_param(name, torch.zeros(3, dtype=torch.bfloat16))


def test_kda_tp4_to_tp8_conv_views_reconstruct_semantic_rows(config):
    rows = 64 * 128
    source = torch.arange(3 * rows * 4, dtype=torch.float32).view(3 * rows, 1, 4)
    sender = dict(
        train(config).convert_param(
            "decoder.layers.0.self_attention.kda.conv1d.weight", source
        )
    )
    for tp_rank in range(8):
        receiver = torch.zeros(3 * rows // 8, 1, 4, dtype=torch.float32)
        targets = infer(config, tp_rank=tp_rank, attn_tp_rank=tp_rank).convert_param(
            "model.layers.0.self_attn.qkv_conv1d.weight", receiver
        )
        for component, (name, target) in enumerate(targets):
            target.copy_(sender[name][tp_rank * rows // 8 : (tp_rank + 1) * rows // 8])
            first = component * rows + tp_rank * rows // 8
            expected = source[first : first + rows // 8]
            torch.testing.assert_close(
                receiver[component * rows // 8 : (component + 1) * rows // 8],
                expected,
                rtol=0,
                atol=0,
            )


def test_fused_kda_receiver_preserves_replicated_gates_and_writes_original(config):
    projection, head_dim, local_heads = 1024, 128, 8
    sizes = [projection, projection, projection, local_heads, head_dim, head_dim]
    receiver = torch.zeros(sum(sizes), 2)
    pairs = infer(config).convert_param(
        "model.layers.0.self_attn.fused_qkvbfg_a_proj.weight", receiver
    )
    offset = 0
    for index, ((name, target), rows) in enumerate(zip(pairs, sizes)):
        target.fill_(index + 1)
        assert torch.all(receiver[offset : offset + rows] == index + 1)
        assert name.endswith(
            (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "b_proj.weight",
                "f_a_proj.weight",
                "g_a_proj.weight",
            )
        )
        offset += rows
    a_log = torch.zeros(1, 1, 8, 1)
    infer(config).convert_param("model.layers.0.self_attn.A_log", a_log)[0][1].fill_(9)
    assert torch.all(a_log == 9)


def test_pipeline_mapping_and_expert_ids_are_global(config):
    converter = train(
        config,
        rank(pp_size=3, pp_rank=1, ep_rank=7),
        train_pp_stage_layer_id_map={"1:0": {0: 15}},
    )
    conv_name = converter.convert_param(
        "decoder.layers.0.self_attention.kda.A_log", torch.ones(64)
    )[0][0]
    assert conv_name == "model.layers.15.linear_attn.A_log"
    expert = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    pairs = dict(
        converter.convert_param(
            "decoder.layers.0.mlp.experts.linear_fc1.weight2", expert
        )
    )
    assert set(pairs) == {
        "model.layers.15.mlp.experts.65.gate_proj.weight",
        "model.layers.15.mlp.experts.65.up_proj.weight",
    }
    torch.testing.assert_close(
        pairs["model.layers.15.mlp.experts.65.up_proj.weight"],
        expert[4:],
        rtol=0,
        atol=0,
    )


def test_infer_ep8_fused_experts_writeback_global_id(config):
    receiver = torch.zeros(36, 8, 2)
    pairs = dict(
        infer(config, ep_rank=1).convert_param(
            "model.layers.3.mlp.experts.w13_weight", receiver
        )
    )
    pairs["model.layers.3.mlp.experts.65.up_proj.weight"].fill_(7)
    assert torch.all(receiver[29, 4:] == 7)
    assert torch.all(receiver[29, :4] == 0)
    assert receiver.sum() == 56


@pytest.mark.parametrize(
    "leaf,shape,split", [("weight", (2048, 2), 1536), ("weight_scale_inv", (16, 2), 12)]
)
def test_mla_fused_down_projection_split_writes_through(config, leaf, shape, split):
    receiver = torch.zeros(shape)
    pairs = dict(
        infer(config).convert_param(
            f"model.layers.3.self_attn.fused_qkv_a_proj_with_mqa.{leaf}", receiver
        )
    )
    pairs[f"model.layers.3.self_attn.q_a_proj.{leaf}"].fill_(3)
    pairs[f"model.layers.3.self_attn.kv_a_proj_with_mqa.{leaf}"].fill_(5)
    assert torch.all(receiver[:split] == 3)
    assert torch.all(receiver[split:] == 5)


def test_fp8_exclusions_use_hf_names_and_preserve_kv_b(config):
    config.quantization_config = fp8_config(
        modules_to_not_convert=[
            "model.language_model.layers.0.self_attn.q_proj",
            "model.layers.3.self_attn.kv_b_proj",
        ]
    )
    values = torch.linspace(-1, 1, 128 * 128, dtype=torch.bfloat16).view(128, 128)
    converter = train(config)
    for source in (
        "decoder.layers.0.self_attention.kda.q_proj.weight",
        "decoder.layers.3.self_attention.linear_kv_up_proj.weight",
    ):
        pairs = converter.convert_param(source, values)
        assert len(pairs) == 1 and pairs[0][1].dtype == torch.bfloat16
    pairs = dict(
        converter.convert_param(
            "decoder.layers.0.self_attention.kda.k_proj.weight", values
        )
    )
    prefix = "model.layers.0.linear_attn.k_proj.weight"
    assert pairs[prefix].dtype == torch.float8_e4m3fn
    assert pairs[prefix + "_scale_inv"].dtype == torch.float32
    restored = pairs[prefix].float() * pairs[prefix + "_scale_inv"][0, 0]
    # E4M3 round-to-nearest has a half-ULP relative bound of 2**-4;
    # half its smallest subnormal supplies the absolute error bound.
    torch.testing.assert_close(
        restored,
        values.float(),
        rtol=2**-4,
        atol=float(pairs[prefix + "_scale_inv"][0, 0]) * 2**-10,
    )


def test_metadata_dict_without_quant_method_does_not_enable_fp8(config):
    config.quantization_config = {"modules_to_not_convert": ["lm_head"]}
    values = torch.ones(128, 128, dtype=torch.bfloat16)
    pairs = train(config).convert_param(
        "decoder.layers.0.self_attention.kda.q_proj.weight", values
    )
    assert len(pairs) == 1 and pairs[0][1].dtype == torch.bfloat16


@pytest.mark.parametrize(
    "metadata",
    [
        {"quant_method": "mxfp8"},
        fp8_config(fmt="e5m2"),
        fp8_config(weight_block_size=[32, 32]),
    ],
)
def test_unsupported_quantization_is_rejected_on_both_sides(config, metadata):
    config.quantization_config = metadata
    for constructor in (train, infer):
        with pytest.raises(ValueError):
            constructor(config)


def test_scale_alias_does_not_accept_packed_or_swizzled_layout(config):
    converter = infer(config)
    name, _ = converter.convert_param(
        "model.layers.3.self_attn.q_b_proj.weight_scale", torch.ones(2, 2)
    )[0]
    assert name.endswith("weight_scale_inv")
    value = torch.ones(2, 2)
    value.format_ue8m0 = True
    with pytest.raises(ValueError, match="FP32 block scales"):
        converter.convert_param(name, value)
    with pytest.raises(ValueError, match="FP32 block scales"):
        converter.convert_param(name, torch.ones(2, 2, dtype=torch.uint8))


@pytest.mark.parametrize(
    "suffix,dim,replicated",
    [
        ("q_proj.weight", 0, False),
        ("o_proj.weight_scale_inv", 1, False),
        ("f_a_proj.weight", 0, True),
        ("o_norm.weight", 0, True),
        ("A_log", 0, False),
    ],
)
def test_kda_sharding_distinguishes_actor_replication_from_receiver(
    config, suffix, dim, replicated
):
    name = "model.layers.0.linear_attn." + suffix
    assert strategy("mcore", rank()).get_sharding_strategy(name) == (
        ShardingType.NO_SHARDING,
        0,
        1,
    )
    expected = (
        (ShardingType.NO_SHARDING, 0, 1)
        if replicated
        else (ShardingType.DP_TP_SHARDING, dim, 8)
    )
    assert strategy("sglang", rank(8, 8)).get_sharding_strategy(name) == expected


def test_frozen_indexer_is_transferred_and_replicated(config):
    name, values = train(config).convert_param(
        "decoder.layers.3.self_attention.wk.weight", torch.ones(128, 128)
    )[0]
    assert name == "model.layers.3.self_attn.indexer.wk.weight"
    assert strategy("mcore", rank()).get_sharding_strategy(name) == (
        ShardingType.NO_SHARDING,
        0,
        1,
    )
    assert infer(config).convert_param(name, values)[0][0] == name


def test_vision_and_cp_fail_explicitly(config):
    with pytest.raises(ValueError, match="vision"):
        infer(config).convert_param("visual.patch_embed.proj.weight", torch.zeros(2, 2))
    with pytest.raises(ValueError, match="CP1"):
        infer(config, cp_size=2)
