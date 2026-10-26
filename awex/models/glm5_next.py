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

"""GLM-5.3-Flash text weight conversion for Megatron and SGLang.

The transport namespace uses ``linear_attn`` for KDA and ``self_attn`` for
MLA. KDA is replicated in the training implementation but head-sharded in
SGLang; distinct names keep that distinction available to metadata resolvers
without depending on local pipeline layer numbers. Checkpoint exclusion rules
are evaluated in the original HF namespace before quantizing transport tensors.

Use SGLang's language-only model configuration for text training. This adapter
rejects vision tensors rather than silently dropping weights that would be lost
when colocated inference releases and reallocates its parameter storage.
"""

import re
from types import SimpleNamespace

import torch

from awex.converter.sglang_converter import SGlangToHFWeightConverter
from awex.converter.weights_converter import per_block_cast_to_fp8
from awex.sharding.param_sharding import ShardingStrategy, ShardingType


def _get(config, key, default=None):
    return (
        config.get(key, default)
        if isinstance(config, dict)
        else getattr(config, key, default)
    )


def _text_config(config):
    text = _get(config, "text_config", config)
    values = dict(text) if isinstance(text, dict) else dict(vars(text))
    values["num_experts"] = _get(text, "n_routed_experts", _get(text, "num_experts"))
    return SimpleNamespace(**values)


def _hf_name(name: str) -> str:
    return name.replace("model.language_model.", "model.").replace(
        ".linear_attn.", ".self_attn."
    )


def _fp8_config(config):
    quantization = _get(config, "quantization_config", None) or {}
    method = quantization.get("quant_method")
    if method is None:
        return {}
    if method != "fp8":
        raise ValueError(f"Unsupported GLM transport quantization: {method}")
    if quantization.get("weight_block_size") != [128, 128]:
        raise ValueError("GLM FP8 transport requires 128x128 weight blocks")
    if quantization.get("fmt", "e4m3") != "e4m3":
        raise ValueError("GLM FP8 transport requires e4m3 weights and FP32 scales")
    return quantization


def _split_rows(tensor: torch.Tensor, sizes: list[int], name: str):
    if tensor.ndim == 0 or tensor.shape[0] != sum(sizes):
        raise ValueError(
            f"Invalid GLM tensor layout for {name}: {tuple(tensor.shape)}, rows={sizes}"
        )
    return tensor.split(sizes, dim=0)


class Glm5NextShardingStrategy(ShardingStrategy):
    def get_sharding_strategy(self, parameter_name, **kwargs):
        if self.enable_dp_attention or getattr(self.rank_info, "cp_size", 1) != 1:
            raise ValueError("GLM weight transfer requires CP1 and no DP attention")
        name = parameter_name.replace("_scale_inv", "")
        if ".linear_attn." in name:
            suffix = name.split(".linear_attn.", 1)[1]
            if self.engine_name == "mcore" or suffix in {
                "f_a_proj.weight",
                "g_a_proj.weight",
                "o_norm.weight",
            }:
                return ShardingType.NO_SHARDING, 0, 1
            dim = 1 if suffix == "o_proj.weight" else 0
            size = self.rank_info.attn_tp_size
            return (
                ShardingType.DP_TP_SHARDING if size > 1 else ShardingType.NO_SHARDING,
                dim,
                size,
            )
        if (
            ".hc_" in name
            or ".indexer." in name
            or name.endswith(".e_score_correction_bias")
            or name.endswith((".q_a_proj.weight", ".kv_a_proj_with_mqa.weight"))
        ):
            return ShardingType.NO_SHARDING, 0, 1
        # The unfused shared MLP follows attention TP, not routed-expert EP.
        if ".shared_experts." in name:
            return self.get_mlp_sharding_strategy(parameter_name, **kwargs)
        return super().get_sharding_strategy(parameter_name, **kwargs)


def build_mcore_converter():
    from awex.converter.mcore_converter import (
        McoreToHFWeightConverter,
        _process_mcore_pp_name,
    )

    class McoreToHFWeightConverterGlm5Next(McoreToHFWeightConverter):
        def __init__(self, hf_config, rank_info, infer_conf, tf_config):
            text = _text_config(hf_config)
            infer_conf = {
                **infer_conf,
                "router_dtype": _get(text, "moe_router_dtype", "float32").replace(
                    "float", "fp"
                ),
            }
            super().__init__(text, rank_info, infer_conf, tf_config)
            self.quantization_config = _fp8_config(hf_config)

        def _fuse_gate_up_proj(self, name):
            return False

        @staticmethod
        def _normalize_attn_name(name):
            return name

        def _convert_attention_param(self, name, parameter, layer_number):
            if name.startswith("self_attention.kda."):
                suffix = name.removeprefix("self_attention.kda.")
                if suffix == "conv1d.weight":
                    width = parameter.shape[0] // 3
                    parts = _split_rows(parameter.float(), [width] * 3, name)
                    return [
                        (f"linear_attn.{key}_conv1d.weight", part)
                        for key, part in zip("qkv", parts)
                    ]
                if suffix == "A_log":
                    parameter = parameter.reshape(-1).float()
                elif suffix == "dt_bias":
                    parameter = parameter.float()
                allowed = {
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "b_proj.weight",
                    "f_a_proj.weight",
                    "f_b_proj.weight",
                    "g_a_proj.weight",
                    "g_b_proj.weight",
                    "o_norm.weight",
                    "o_proj.weight",
                    "A_log",
                    "dt_bias",
                }
                if suffix not in allowed:
                    raise NotImplementedError(f"Unsupported GLM KDA parameter: {name}")
                return [(f"linear_attn.{suffix}", parameter)]
            suffix = name.removeprefix("self_attention.")
            indexer = {
                "wq_b.weight",
                "wk.weight",
                "k_norm.weight",
                "k_norm.bias",
                "weights_proj.weight",
                "index_kpool_compress_gate",
                "index_kpool_compress_ape",
            }
            if suffix in indexer:
                return [(f"self_attn.indexer.{suffix}", parameter)]
            mapping = {
                "linear_q_down_proj.weight": "q_a_proj.weight",
                "linear_q_up_proj.weight": "q_b_proj.weight",
                "linear_q_up_proj.layer_norm_weight": "q_a_layernorm.weight",
                "linear_kv_down_proj.weight": "kv_a_proj_with_mqa.weight",
                "linear_kv_up_proj.weight": "kv_b_proj.weight",
                "linear_kv_up_proj.layer_norm_weight": "kv_a_layernorm.weight",
                "linear_proj.weight": "o_proj.weight",
            }
            if suffix not in mapping:
                raise NotImplementedError(f"Unsupported GLM MLA parameter: {name}")
            return [(f"self_attn.{mapping[suffix]}", parameter)]

        def _convert_expert_bias_param(self, name, parameter, layer_number):
            return "mlp.gate.e_score_correction_bias", parameter.float()

        def _quantize(self, pairs):
            if not self.quantization_config:
                return pairs
            exclusions = [
                _hf_name(n)
                for n in self.quantization_config.get("modules_to_not_convert", [])
            ]
            output = []
            for name, value in pairs:
                module = _hf_name(name).removesuffix(".weight")
                excluded = any(
                    module == n
                    or module.startswith(n + ".")
                    or module.endswith("." + n)
                    for n in exclusions
                )
                quantizable = (
                    name.endswith(".weight") and value.ndim == 2 and not excluded
                )
                if quantizable:
                    weight, scale = per_block_cast_to_fp8(value, scale_ue8m0=False)
                    output.extend([(name, weight), (name + "_scale_inv", scale)])
                else:
                    output.append((name, value))
            return output

        def convert_param(self, name, parameter, vp_stage=None):
            clean = name.replace("module.", "")
            match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", clean)
            if match and (
                "_hyper_connection." in clean
                or match[2] in {"input_layernorm.weight", "pre_mlp_layernorm.weight"}
            ):
                clean = _process_mcore_pp_name(
                    clean,
                    self.rank_info,
                    self.hf_config,
                    self.tf_config,
                    vp_stage=vp_stage,
                    pp_stage_layer_id_map=self._pp_stage_layer_id_map,
                )
                _, _, layer, suffix = clean.split(".", 3)
                if "_hyper_connection." in suffix:
                    source, field = suffix.split(".", 1)
                    site = {
                        "self_attention_hyper_connection": "attn",
                        "mlp_hyper_connection": "ffn",
                    }.get(source)
                    target = {
                        "mapping_proj.weight": "fn",
                        "bias": "base",
                        "scale": "scale",
                    }.get(field)
                    if site is None or target is None:
                        raise NotImplementedError(
                            f"Unsupported GLM mHC parameter: {name}"
                        )
                    if target == "scale" and parameter.numel() != 3:
                        raise ValueError("GLM mHC scale must retain all three values")
                    return [
                        (f"model.layers.{layer}.hc_{site}_{target}", parameter.float())
                    ]
                target = suffix.replace("pre_mlp_layernorm", "post_attention_layernorm")
                return [(f"model.layers.{layer}.{target}", parameter)]
            return self._quantize(
                super().convert_param(name, parameter, vp_stage=vp_stage)
            )

    return McoreToHFWeightConverterGlm5Next


class SGlangToHFWeightConverterGlm5Next(SGlangToHFWeightConverter):
    def __init__(self, model_config, infer_engine_config, rank_info):
        quantization = _fp8_config(model_config)
        if quantization:
            # A plain FP32 scale tensor can still contain DeepGEMM's rounded
            # UE8M0 values. Require backends that retain the canonical layout;
            # dtype checks alone cannot detect all transformed representations.
            for field in ("fp8_gemm_runner_backend", "moe_runner_backend"):
                if _get(infer_engine_config, field) != "triton":
                    raise ValueError(f"GLM FP8 weight transfer requires {field}=triton")
        if (
            _get(infer_engine_config, "enable_dp_attention", False)
            or getattr(rank_info, "cp_size", 1) != 1
        ):
            raise ValueError("GLM weight transfer requires CP1 and no DP attention")
        super().__init__(_text_config(model_config), infer_engine_config, rank_info)

    def _fuse_qkv(self, name):
        return False

    def _fuse_gate_up_proj(self, name):
        return False

    def convert_param(self, name, parameter):
        name = name.replace("model.language_model.", "model.")
        if name.startswith(("visual.", "model.visual.", "vision_model.")):
            raise ValueError(
                "GLM text weight updates require language_only=True; frozen vision needs an explicit preservation contract"
            )
        # Current SGLang accepts both checkpoint scale spellings.
        if name.endswith((".weight_scale", ".w13_weight_scale", ".w2_weight_scale")):
            name += "_inv"
        if "scale_inv" in name and (
            parameter.dtype != torch.float32
            or getattr(parameter, "format_ue8m0", False)
            or "swizzled" in name
        ):
            raise ValueError("GLM transport requires ordinary FP32 block scales")
        match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
        if not match:
            return super().convert_param(name, parameter)
        layer, suffix = int(match[1]), match[2]
        prefix = f"model.layers.{layer}."
        if suffix.startswith("hc_") or suffix.endswith("e_score_correction_bias"):
            if parameter.dtype != torch.float32:
                raise ValueError(f"GLM {name} must retain FP32 precision")
            if suffix in {"hc_attn_scale", "hc_ffn_scale"} and parameter.numel() != 3:
                raise ValueError("GLM mHC scale must retain all three values")
            return [(name, parameter)]
        if suffix.startswith("self_attn."):
            field = suffix.removeprefix("self_attn.")
            kinds = self.model_config.layer_types
            if kinds[layer] == "linear_attention":
                prefix += "linear_attn."
                linear = self.model_config.linear_attn_config
                heads, width = linear["num_heads"], linear["head_dim"]
                tp = self.rank_info.attn_tp_size
                projection = heads * width // tp
                if field in {
                    "qkv_proj.weight",
                    "qkv_proj.weight_scale_inv",
                    "qkv_conv1d.weight",
                }:
                    rows = projection
                    if field.endswith("_scale_inv"):
                        if projection % 128:
                            raise ValueError(
                                "GLM KDA FP8 shards must align to 128 rows"
                            )
                        rows //= 128
                    parts = _split_rows(parameter, [rows] * 3, name)
                    leaf = (
                        "_proj.weight"
                        if field.startswith("qkv_proj")
                        else "_conv1d.weight"
                    )
                    if field.endswith("_scale_inv"):
                        leaf += "_scale_inv"
                    return [
                        (prefix + key + leaf, part) for key, part in zip("qkv", parts)
                    ]
                if field == "fused_qkvbfg_a_proj.weight":
                    sizes = [projection] * 3 + [heads // tp, width, width]
                    keys = (
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "b_proj",
                        "f_a_proj",
                        "g_a_proj",
                    )
                    return [
                        (prefix + key + ".weight", part)
                        for key, part in zip(keys, _split_rows(parameter, sizes, name))
                    ]
                if field == "fused_fg_b_proj.weight":
                    if tuple(parameter.shape) != (2, projection, width):
                        raise ValueError(
                            f"Unexpected GLM batched projection layout: {tuple(parameter.shape)}"
                        )
                    return [
                        (prefix + key + ".weight", parameter[index])
                        for index, key in enumerate(("f_b_proj", "g_b_proj"))
                    ]
                if field == "A_log":
                    parameter = parameter.view(-1)
                return [(prefix + field, parameter)]
            if field.startswith("fused_qkv_a_proj_with_mqa."):
                leaf = field.split(".", 1)[1]
                divisor = 128 if leaf.endswith("_scale_inv") else 1
                sizes = [
                    self.model_config.q_lora_rank // divisor,
                    (
                        self.model_config.kv_lora_rank
                        + self.model_config.qk_rope_head_dim
                    )
                    // divisor,
                ]
                return [
                    (prefix + "self_attn." + key + "." + leaf, part)
                    for key, part in zip(
                        ("q_a_proj", "kv_a_proj_with_mqa"),
                        _split_rows(parameter, sizes, name),
                    )
                ]
            return [(name, parameter)]
        if suffix in {"input_layernorm.weight", "post_attention_layernorm.weight"}:
            return [(name, parameter)]
        return super().convert_param(name, parameter)


CONFIG = {
    "model_name": "Glm5NextForConditionalGeneration",
    "sharding_strategy": Glm5NextShardingStrategy,
    "mcore_converter": build_mcore_converter,
    "sglang_converter": SGlangToHFWeightConverterGlm5Next,
}
