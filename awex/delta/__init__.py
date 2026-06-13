# Licensed under the Apache License, Version 2.0
"""awex.delta — Delta (incremental) weight transfer module."""

from awex.delta.codec import (
    CODEC_VERSION,
    DELTA_HEADER_NAME,
    DELTA_IDX_SUFFIX,
    DELTA_VAL_SUFFIX,
    DecodedDelta,
    DeltaHeader,
    DeltaTracker,
    EncodedDelta,
    apply_sparse_patch_,
    bitwise_changed_mask,
    decode_delta_payload,
    int_view,
    is_delta_payload,
    reconstruct_against_base,
)
from awex.delta.detector import DeltaWeightDetector
from awex.delta.patch import (
    DeltaResult,
    SparseWeightPatch,
    dict_to_patches,
    patches_to_dict,
)
from awex.delta.remap import remap_delta_indices, remap_patches_for_operation

__all__ = [
    "CODEC_VERSION",
    "DELTA_HEADER_NAME",
    "DELTA_IDX_SUFFIX",
    "DELTA_VAL_SUFFIX",
    "DecodedDelta",
    "DeltaHeader",
    "DeltaResult",
    "DeltaTracker",
    "DeltaWeightDetector",
    "EncodedDelta",
    "SparseWeightPatch",
    "apply_sparse_patch_",
    "bitwise_changed_mask",
    "decode_delta_payload",
    "dict_to_patches",
    "int_view",
    "is_delta_payload",
    "patches_to_dict",
    "reconstruct_against_base",
    "remap_delta_indices",
    "remap_patches_for_operation",
]
