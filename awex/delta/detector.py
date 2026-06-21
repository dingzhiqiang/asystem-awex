# Licensed under the Apache License, Version 2.0
"""Shim: awex.delta.detector delegates to dte.core.detector (see codec.py note)."""
from dte.core.detector import *  # noqa: F401,F403
from dte.core.detector import DeltaWeightDetector
