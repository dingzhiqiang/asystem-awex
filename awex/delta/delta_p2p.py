# Licensed under the Apache License, Version 2.0
"""Shim: awex.delta.delta_p2p delegates to dte.core.delta_p2p (see codec.py note)."""
from dte.core.delta_p2p import *  # noqa: F401,F403
from dte.core.delta_p2p import (
    OpDeltaPayload,
    allocate_recv_buffers,
    build_send_patches,
    build_send_payloads_by_op,
    nnz_vector,
    op_key,
    scatter_recv_into,
)
