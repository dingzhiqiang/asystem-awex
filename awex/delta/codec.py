# Licensed under the Apache License, Version 2.0
"""Shim: awex.delta.codec now delegates to dte.core.codec.

The delta (incremental weight transfer) algorithm's single source of truth is
the standalone `dte` library (delta-transfer-engine). awex.delta is kept as a
thin forwarder so awex internals and AReaL keep importing `awex.delta.*`
unchanged while the implementation lives in dte. Wire format / CODEC_VERSION are
identical (dte.core.codec is the same code).
"""
from dte.core.codec import *  # noqa: F401,F403
from dte.core.codec import (  # explicit: names imported by path elsewhere
    CODEC_VERSION,
    DELTA_HEADER_NAME,
    DELTA_IDX_SUFFIX,
    DELTA_VAL_SUFFIX,
    DELTA_MAGIC,
    DecodedDelta,
    DeltaHeader,
    DeltaTracker,
    EncodedDelta,
    apply_sparse_patch_,
    bitwise_changed_mask,
    decode_delta_payload,
    int_view,
    invert_adamw,
    is_delta_payload,
    reconstruct_against_base,
)
