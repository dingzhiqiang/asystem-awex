# AReaL swe/main 共卡模式实现计划

## Context

在 AReaL swe/main 分支上实现 GPU 共卡（colocate）模式，训练和推理共享同一组 GPU。使用 AWEX 的 CUDA IPC 做权重传输。

**目标仓库**:
- AReaL: `/Users/dzq/work/2026/codes/rlhf-repo/AReaL-asystem/AReaL` → 分支 `chucai.dzq/feat-colocate-awex` (基于 `swe/main`)
- awex: `/Users/dzq/work/2026/codes/gh-repo/asystem-awex` → 分支 `chucai.dzq/feat-sglang-plugin`

**设计参考**: AReaL-gh PR #1310 的接口设计（`AwexMegatronAdapter`、`AwexSGLangAdapter`、`AwexSchedulerBridge`）

---

## swe/main 现有接口

| 组件 | 现有方法 | 说明 |
|------|---------|------|
| `MegatronEngine` | `offload()` / `onload()` | TMS pause/resume（需替换为手动 offload） |
| `MegatronEngine` | `update_weights(meta)` | 支持 "xccl" 和 "disk" 两种 type |
| `MegatronEngine` | `connect_engine(engine, meta)` | 连接 rollout engine，初始化 NCCL group |
| `RemoteInfEngine` | `offload()` / `onload(tags)` | HTTP 调 SGLang `/release_memory_occupation` / `/resume_memory_occupation` |
| `RemoteInfEngine` | `pause()` / `resume()` | 暂停/恢复推理请求 |
| `RemoteInfEngine` | `update_weights_from_distributed(meta)` | HTTP 调 SGLang `/update_weights_from_distributed` |
| `PPOTrainer.train()` | `rollout.pause()` → update → `rollout.resume()` | 主循环已有 pause/resume 框架 |
| `SGLangBackend` | `get_offload_request()` / `get_onload_request(tags)` | 构造 HTTP request |
| SGLang (开源) | `/release_memory_occupation` | TMS pause，tags: weights, kv_cache, cuda_graph |
| SGLang (开源) | `/resume_memory_occupation` | TMS resume |

---

## 改动总览

### awex 仓库 (新增 3 文件)

| 文件 | 说明 |
|------|------|
| `awex/sglang_plugin.py` | `AwexSchedulerBridge` + `register_awex_endpoints(app)` — bind awex 方法到 SGLang Scheduler |
| `awex/sglang_awex_adapter.py` | `AwexSGLangAdapter` — 推理侧 colocate weight update (IPC deserialize + NCCL reshard) |
| `awex/sglang_awex_launcher.py` | 启动入口 — 替换 `run_scheduler_process` 后启动 SGLang |

### AReaL 仓库 (修改 3 文件 + 新增 1 文件)

| 文件 | 说明 |
|------|------|
| `areal/engine/megatron_engine.py` | 添加 `offload_colocate()`/`onload_colocate()` + `"awex"` weight update type |
| `areal/engine/awex_colocate.py` (新) | `AwexMegatronAdapter` — 训练侧 colocate weight update (IPC serialize + handshake) |
| `areal/trainer/rl_trainer.py` | 共卡模式主循环分支 |
| `areal/infra/launcher/sglang_server.py` | 共卡模式启动时用 awex launcher 替换默认 SGLang 启动 |

---

## Phase 1: awex SGLang Plugin

### `awex/sglang_plugin.py`

采用 AReaL-gh 的 `AwexSchedulerBridge` 模式（不 monkeypatch worker 类，而是 bind 方法到 scheduler 实例）：

```python
class AwexSchedulerBridge:
    """Bind awex_* methods onto a SGLang Scheduler instance."""
    
    def __init__(self, scheduler):
        self._scheduler = scheduler
        self._adapter = None  # lazy AwexSGLangAdapter
    
    def bind(self):
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
    
    def awex_execute_colocate_weight_update(self, version=0):
        self._require_adapter().execute_colocate_weight_update(version)
    
    def awex_release_memory(self, tags=None):
        self._require_adapter().release_memory(tags)
    
    def awex_resume_memory(self, tags=None):
        self._require_adapter().resume_memory(tags)

def register_awex_endpoints(app, rpc_dispatch_fn):
    """注册 /awex/* HTTP 端点到 SGLang FastAPI app."""
    @app.post("/awex/init_colocate_weight_update")
    async def init_colocate(request): ...
    
    @app.post("/awex/execute_colocate_weight_update")
    async def execute_colocate(request): ...
    
    @app.post("/awex/release_memory")
    async def release_memory(request): ...

def areal_run_scheduler_process(...):
    """替换 sglang.srt.managers.scheduler.run_scheduler_process"""
    from sglang.srt.managers.scheduler import Scheduler, configure_scheduler
    # ... 标准 SGLang scheduler 初始化 ...
    scheduler = Scheduler(...)
    AwexSchedulerBridge(scheduler).bind()  # <-- 唯一新增
    scheduler.run_event_loop()
```

### `awex/sglang_awex_adapter.py`

对齐 AReaL-gh 的 `AwexSGLangAdapter`，核心方法：

```python
class AwexSGLangAdapter:
    def __init__(self, scheduler): ...
    
    def get_weight_metadata(self) -> list[ParameterMeta]:
        """收集 SGLang 模型的参数元数据（含 unfuse qkv/gate_up）"""
    
    def init_colocate_weight_update(self, pair_name, kv_store_url, transfer_rank, 
                                     infer_world_size, train_world_size, ...):
        """初始化 transfer plan + NCCL group (推理 rank 间)"""
    
    def execute_colocate_weight_update(self, version):
        """从 KV store 获取 IPC handle → deserialize → NCCL reshard → write to model"""
    
    def release_memory(self, tags):
        """调 scheduler.release_memory_occupation (SGLang 原生)"""
    
    def resume_memory(self, tags):
        """调 scheduler.resume_memory_occupation (SGLang 原生)"""
```

### `awex/sglang_awex_launcher.py`

```python
def main():
    """启动 SGLang server，使用自定义 run_scheduler_process."""
    # 替换 SGLang 的 scheduler process launcher
    import sglang.srt.managers.scheduler as sched_mod
    from awex.sglang_plugin import areal_run_scheduler_process
    sched_mod.run_scheduler_process = areal_run_scheduler_process
    # 正常启动 SGLang
    from sglang.launch_server import main as sglang_main
    sglang_main()
```

---

## Phase 2: AReaL Actor 侧

### `areal/engine/awex_colocate.py` (新文件)

对齐 AReaL-gh 的 `AwexMegatronAdapter`：

```python
class AwexMegatronAdapter:
    """Actor 侧 AWEX colocate 适配器."""
    
    def __init__(self, engine: MegatronEngine):
        self._engine = engine
        self._offloaded_weights = {}
        self._offloaded_optimizer_states = {}
    
    def init_colocate_weight_update(self, pair_name, kv_store_url, transfer_rank, ...):
        """初始化 colocate 传输配置"""
    
    def execute_colocate_weight_update(self, version):
        """IPC serialize 当前权重 → PUT 到 KV store → 等 inference done"""
        params = self._get_hf_params()
        group_tensors, metadata = group_tensors_by_shape_and_dtype(params.values())
        group_shared = [t.share_memory_() for t in group_tensors]
        serialized = cuda_ipc_serialize((group_shared, metadata, list(params.keys())))
        # PUT to gateway KV store
        # poll for inference "done" signal
        # cleanup
    
    def release_memory(self, tags=["optimizer", "weights"]):
        if "optimizer" in tags: self._offload_optimizer_states()
        if "weights" in tags: self._offload_model_weights()
    
    def resume_memory(self, tags=["optimizer", "weights"]):
        if "weights" in tags: self._reload_model_weights()
        if "optimizer" in tags: self._reload_optimizer_states()
    
    def _offload_model_weights(self):
        """param.data → CPU, param.data = empty(0, cpu)"""
    
    def _offload_optimizer_states(self):
        """optimizer state tensors → CPU"""
```

### `areal/engine/megatron_engine.py` (修改)

```python
# 1. 添加 colocate_mode 属性
self._awex_adapter: AwexMegatronAdapter | None = None

# 2. connect_engine 时初始化 awex adapter
def connect_engine(self, engine, meta):
    ...
    if meta.type == "awex":
        from areal.engine.awex_colocate import AwexMegatronAdapter
        self._awex_adapter = AwexMegatronAdapter(self)
        self._awex_adapter.init_colocate_weight_update(...)

# 3. update_weights 添加 "awex" 分支
def update_weights(self, meta):
    if meta.type == "awex":
        self._awex_adapter.execute_colocate_weight_update(meta.version)
    elif meta.type == "xccl": ...

# 4. offload/onload 添加 colocate 分支（不用 TMS）
def offload(self):
    if self._awex_adapter is not None:
        self._awex_adapter.release_memory(tags=["optimizer", "weights"])
    else:
        torch_memory_saver.pause()
    self.is_offload = True

def onload(self):
    if self._awex_adapter is not None:
        self._awex_adapter.resume_memory(tags=["optimizer", "weights"])
    else:
        torch_memory_saver.resume()
    self.is_offload = False
```

---

## Phase 3: Trainer 主循环

### `areal/trainer/rl_trainer.py` (修改)

在现有 `train()` 方法的循环中，weight update 部分改为：

```python
# 现有代码 (line ~513):
if not self._should_replay_rollout_data(global_step):
    self.rollout.pause()
    
    # --- 共卡模式分支 ---
    if self.config.enable_colocate_mode:
        # Actor offload optimizer (保留 weights 供 AWEX 读)
        self.actor._awex_adapter.release_memory(tags=["optimizer"])
        # AWEX colocate weight update (Actor IPC serialize → SGLang IPC read)
        versioned_meta = self.weight_update_meta.with_version(new_version)
        self.actor.update_weights(versioned_meta)
        # Actor offload weights
        self.actor._awex_adapter.release_memory(tags=["weights"])
    else:
        # 原有 xccl/disk 路径
        versioned_meta = self.weight_update_meta.with_version(new_version)
        self.actor.update_weights(versioned_meta)
    
    self.actor.set_version(new_version)
    ...
    self.rollout.resume()
```

rollout 阶段前后增加 offload/onload：

```python
# rollout 前: 确保 SGLang 有显存
if self.config.enable_colocate_mode:
    self.rollout.onload()  # resume_memory_occupation

# rollout
rollout_batch = self.actor.prepare_batch(...)

# rollout 后: 释放 SGLang 显存给 train
if self.config.enable_colocate_mode:
    self.rollout.offload()  # release_memory_occupation

# train
if self.config.enable_colocate_mode:
    self.actor.onload()  # resume weights + optimizer from CPU
self.actor.ppo_update(adv_batch)
```

---

## Phase 4: SGLang 启动配置

### `areal/infra/launcher/sglang_server.py` (修改)

共卡模式时用 awex launcher 替换默认启动：

```python
def launch_server_cmd(server_args, ...):
    if enable_colocate_mode:
        # 使用 awex launcher (自定义 run_scheduler_process)
        cmd = [sys.executable, "-m", "awex.sglang_awex_launcher", 
               "--enable-memory-saver", ...]
    else:
        cmd = [sys.executable, "-m", "sglang.launch_server", ...]
```

### Env 隔离 (在 scheduler 或 launcher 中)

```python
# Actor worker env
actor_env = {"TMS_INIT_ENABLE": "0"}  # 不用 TMS

# Rollout worker env  
rollout_env = {"TMS_INIT_ENABLE": "1"}  # SGLang 用 TMS
```

---

## 协调器 (Gateway)

AReaL-gh 用独立 FastAPI gateway 做 KV store 和 lifecycle 协调。swe/main 可以简化为：
- 复用 awex 的 `MetaServer`（已有的 HTTP dict 服务）作为 KV store
- 或者内嵌一个轻量 KV store 在 trainer 进程中

初始阶段建议用 awex 自带的 MetaServer，与 HybridEngine 实现对齐。

---

## Verification

1. **awex plugin 单元测试**: 验证 `AwexSchedulerBridge.bind()` 正确 setattr
2. **Actor offload 测试**: offload → onload 后模型数值不变
3. **IPC 端到端**: Actor serialize → Reader deserialize，验证数值正确
4. **单卡 smoke test**: 1 GPU 跑 1 step train + weight sync
5. **多卡 e2e**: 4 GPU TP=4，5 steps 无 memory leak

---

## 实现顺序

1. awex: `sglang_awex_adapter.py` (SGLang 侧 adapter)
2. awex: `sglang_plugin.py` (SchedulerBridge + HTTP routes + launcher)
3. AReaL: `areal/engine/awex_colocate.py` (Actor 侧 adapter)
4. AReaL: `megatron_engine.py` 修改 (connect_engine + update_weights + offload/onload)
5. AReaL: `rl_trainer.py` 修改 (共卡主循环)
6. AReaL: `sglang_server.py` 修改 (启动配置)
