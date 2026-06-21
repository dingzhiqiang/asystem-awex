# Licensed under the Apache License, Version 2.0
"""Shim: awex.delta.patch delegates to dte.core.patch (see codec.py shim note)."""
from dte.core.patch import *  # noqa: F401,F403
from dte.core.patch import (
    DeltaResult,
    SparseWeightPatch,
    dict_to_patches,
    patches_to_dict,
)
