# Licensed under the Apache License, Version 2.0
"""awex.delta — thin forwarder to the standalone dte library.

The delta algorithm now lives in `dte.core` (delta-transfer-engine). This package
re-exports it so existing `from awex.delta import ...` call sites (awex internals
+ AReaL) keep working unchanged. Wire format is identical.
"""
from dte.core import *  # noqa: F401,F403
from dte.core import (
    CODEC_VERSION,
    DELTA_HEADER_NAME,
    DELTA_IDX_SUFFIX,
    DELTA_VAL_SUFFIX,
    DecodedDelta,
    DeltaHeader,
    DeltaResult,
    DeltaTracker,
    DeltaWeightDetector,
    EncodedDelta,
    SparseWeightPatch,
    apply_sparse_patch_,
    bitwise_changed_mask,
    decode_delta_payload,
    dict_to_patches,
    int_view,
    invert_adamw,
    is_delta_payload,
    patches_to_dict,
    reconstruct_against_base,
    remap_delta_indices,
    remap_patches_for_operation,
)
