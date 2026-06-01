# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.
#
# SPDX-License-Identifier: Apache-2.0

"""Awex SGLang plugin: SchedulerBridge + HTTP endpoints + custom scheduler process.

Usage:
    1. As a launcher: python -m awex.sglang_awex_launcher (replaces run_scheduler_process)
    2. Programmatically: call register_awex_sglang_plugin() before SGLang starts

Design (following AReaL-gh PR #1310 AwexSchedulerBridge pattern):
    - No monkeypatch of SGLang worker classes
    - Bind awex_* methods onto Scheduler instance after creation
    - Register /awex/* HTTP endpoints on the SGLang FastAPI app
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AwexSchedulerBridge:
    """Compose awex weight-update capabilities onto a SGLang Scheduler instance.

    Lifecycle:
      1. Created after Scheduler.__init__() in areal_run_scheduler_process
      2. bind() attaches awex_* methods to the scheduler via setattr
      3. SGLang's RPC dispatch finds them via getattr(scheduler, method_name)
      4. Methods delegate to AwexSGLangAdapter for actual work
    """

    def __init__(self, scheduler: Any) -> None:
        self._scheduler = scheduler
        self._adapter = None

    def bind(self) -> None:
        methods = [
            "awex_report_weight_meta",
            "awex_report_parallelism",
            "awex_init_colocate_weight_update",
            "awex_execute_colocate_weight_update",
            "awex_release_memory",
            "awex_resume_memory",
        ]
        for name in methods:
            setattr(self._scheduler, name, getattr(self, name))
        logger.info("AwexSchedulerBridge bound %d methods to scheduler", len(methods))

    def _require_adapter(self):
        if self._adapter is None:
            from awex.sglang_awex_adapter import AwexSGLangAdapter
            self._adapter = AwexSGLangAdapter(self._scheduler)
        return self._adapter

    def awex_report_weight_meta(self) -> list:
        return self._require_adapter().get_weight_metadata()

    def awex_report_parallelism(self) -> dict:
        return self._require_adapter().parallelism_strategy

    def awex_init_colocate_weight_update(self, **kwargs: Any) -> None:
        self._require_adapter().init_colocate_weight_update(**kwargs)

    def awex_execute_colocate_weight_update(self, version: int = 0) -> None:
        self._require_adapter().execute_colocate_weight_update(version)

    def awex_release_memory(self, tags: list[str] | None = None) -> None:
        self._require_adapter().release_memory(tags)

    def awex_resume_memory(self, tags: list[str] | None = None) -> None:
        self._require_adapter().resume_memory(tags)


def register_awex_endpoints(app, rpc_dispatch_fn: Callable) -> None:
    """Register /awex/* HTTP endpoints on a FastAPI app.

    Args:
        app: FastAPI application instance
        rpc_dispatch_fn: callable(method_name, **kwargs) that broadcasts
            the call to all scheduler processes
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse

    @app.post("/awex/report_weight_meta")
    async def report_weight_meta() -> JSONResponse:
        try:
            result = rpc_dispatch_fn("awex_report_weight_meta")
            return JSONResponse(content={"status": "ok", "meta": result})
        except Exception as e:
            logger.error("report_weight_meta failed: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.get("/awex/report_parallelism")
    async def report_parallelism() -> JSONResponse:
        try:
            result = rpc_dispatch_fn("awex_report_parallelism")
            return JSONResponse(content=result)
        except Exception as e:
            logger.error("report_parallelism failed: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/init_colocate_weight_update")
    async def init_colocate_weight_update(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            rpc_dispatch_fn("awex_init_colocate_weight_update", **data)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("init_colocate_weight_update failed: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/execute_colocate_weight_update")
    async def execute_colocate_weight_update(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            version = data.get("version", 0)
            rpc_dispatch_fn("awex_execute_colocate_weight_update", version=version)
            return JSONResponse(content={"status": "ok", "version": version})
        except Exception as e:
            logger.error("execute_colocate_weight_update failed: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/release_memory")
    async def release_memory(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            tags = data.get("tags")
            rpc_dispatch_fn("awex_release_memory", tags=tags)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("release_memory failed: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/resume_memory")
    async def resume_memory(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            tags = data.get("tags")
            rpc_dispatch_fn("awex_resume_memory", tags=tags)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("resume_memory failed: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    logger.info("Registered /awex/* endpoints on FastAPI app")


def areal_run_scheduler_process(
    server_args,
    port_args,
    gpu_id,
    tp_rank,
    attn_cp_rank,
    moe_dp_rank,
    moe_ep_rank,
    pp_rank,
    dp_rank,
    pipe_writer,
) -> None:
    """Drop-in replacement for sglang.srt.managers.scheduler.run_scheduler_process.

    Identical to upstream except for the AwexSchedulerBridge.bind() call
    after Scheduler creation.
    """
    import signal

    import psutil
    from sglang.srt.managers.scheduler import Scheduler, configure_scheduler
    from sglang.srt.utils import kill_itself_when_parent_died
    from sglang.utils import get_exception_traceback

    dp_rank = configure_scheduler(
        server_args, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank, dp_rank
    )

    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    try:
        from sglang.srt.utils import get_bool_env_var, set_gpu_proc_affinity
        if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
            set_gpu_proc_affinity(
                server_args.pp_size, server_args.tp_size,
                server_args.nnodes, gpu_id
            )
    except (ImportError, AttributeError):
        pass

    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )

        # ---- AWEX ADDITION ----
        AwexSchedulerBridge(scheduler).bind()
        # ---- END AWEX ----

        pipe_writer.send(scheduler.get_init_info())
        scheduler.run_event_loop()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)


def register_awex_sglang_plugin() -> None:
    """Patch SGLang to use our custom run_scheduler_process.

    Call this before SGLang's launch_server().
    """
    import sglang.srt.managers.scheduler as sched_mod
    sched_mod.run_scheduler_process = areal_run_scheduler_process
    logger.info(
        "Patched sglang.srt.managers.scheduler.run_scheduler_process "
        "with awex-enabled version"
    )
