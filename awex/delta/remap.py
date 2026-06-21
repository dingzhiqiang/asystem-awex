# Licensed under the Apache License, Version 2.0
"""Shim: awex.delta.remap delegates to dte.core.remap (see codec.py shim note)."""
from dte.core.remap import *  # noqa: F401,F403
from dte.core.remap import (
    remap_delta_indices,
    remap_mask_for_op,
    remap_patches_for_operation,
)
