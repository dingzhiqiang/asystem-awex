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

import json
import pickle

import pytest
from transformers import PretrainedConfig

from awex.util.common import simple_hf_config


@pytest.mark.parametrize("quant_key", ["quantization_config", "compression_config"])
def test_simple_hf_config_removes_quantizer_backrefs_without_mutating(quant_key):
    text = PretrainedConfig(num_hidden_layers=45)
    config = PretrainedConfig(text_config=text)
    quant = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": ["model.language_model.layers.3.self_attn.kv_b_proj"],
        "hf_config": config,
    }
    setattr(config, quant_key, quant)
    setattr(text, quant_key, quant)
    config._name_or_path = "sglang/model"

    result = simple_hf_config(config)

    restored = json.loads(json.dumps(pickle.loads(pickle.dumps(result))))
    for value in (restored[quant_key], restored["text_config"][quant_key]):
        assert "hf_config" not in value
        assert value == {key: item for key, item in quant.items() if key != "hf_config"}
    assert result.text_config["num_hidden_layers"] == 45
    assert restored["_name_or_path"] == "sglang/model"
    assert quant["hf_config"] is config
    assert getattr(config, quant_key) is quant
    assert getattr(text, quant_key) is quant
