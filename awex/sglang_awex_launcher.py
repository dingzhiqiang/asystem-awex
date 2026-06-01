# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.
#
# SPDX-License-Identifier: Apache-2.0

"""Launch SGLang server with awex colocate support.

Usage:
    python -m awex.sglang_awex_launcher [sglang args...]

This replaces SGLang's run_scheduler_process with the awex-enabled version
before starting the server, so each scheduler process gets AwexSchedulerBridge
bound automatically.
"""

import sys


def main():
    from awex.sglang_plugin import register_awex_sglang_plugin

    register_awex_sglang_plugin()

    from sglang.launch_server import main as sglang_main

    sglang_main()


if __name__ == "__main__":
    main()
