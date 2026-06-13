# Asystem 权重传输详解（HybridEngine + AWEX 共卡实现）

## 架构概览

```
┌──────────────────────────────────────────────────────────────────────┐
│  AReaL Trainer                                                        │
│  grpo_proxy_trainer.py                                                │
│  主循环: rollout.resume → rollout → rollout.pause → train → weight_sync│
├──────────────────────────────────────────────────────────────────────┤
│  HybridEngine (每个 Worker 进程内)                                     │
│  ├─ Training Backend (MegatronBackend)                                │
│  │   ├─ release_memory_occupation(["optimizer","weights"])            │
│  │   ├─ offload_optimizer_states() → CPU pinned                      │
│  │   ├─ offload_params() → CPU pinned                                │
│  │   └─ storage().resize_(0) 释放 GPU                                │
│  ├─ Inference Backend (SGLangBackend)                                 │
│  │   ├─ release_memory_occupation(["kv_cache","weights"])             │
│  │   │   → SGLang TorchMemorySaver.pause("kv_cache") + pause("weights")│
│  │   ├─ resume_memory_occupation(["weights"])                         │
│  │   │   → SGLang TorchMemorySaver.resume("weights")                 │
│  │   └─ resume_memory_occupation(["kv_cache"])                        │
│  │       → SGLang TorchMemorySaver.resume("kv_cache")                │
│  └─ WeightsExchangeReader (内嵌 AWEX Reader)                          │
│      ├─ 分卡: NCCL P2P recv from training ranks                       │
│      └─ 共卡: IPC collect + NCCL reshard between inference ranks      │
├──────────────────────────────────────────────────────────────────────┤
│  AWEX (权重传输底层库)                                                  │
│  ├─ MetaServer: HTTP dict 服务 (元数据交换 + 同步信号)                    │
│  ├─ Writer (训练侧): convert → IPC/NCCL send                          │
│  ├─ Reader (推理侧): IPC/NCCL recv → write to model                   │
│  └─ TransferPlan: 预计算每对 (train_rank, infer_rank) 的 overlap       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 分卡模式 (Separation)

训练和推理在**不同 GPU** 上。

### 初始化流程

```python
# 1. 推理侧收集参数元数据
# HybridEngine SGLangBackend 启动时:
self.weights_exchange_reader = get_weights_exchange_reader(self)
self.weights_exchange_reader.initialize()
# → 遍历 SGLang model.named_parameters()
# → 转 HF 统一命名 + 计算 global_offset
# → PUT "infer_params_meta" 到 MetaServer

# 2. 训练侧收集参数元数据
# HybridEngine MegatronBackend 启动时:
self.weights_exchange_writer = NCCLWeightsWriter(...)
self.weights_exchange_writer.initialize()
# → 遍历 Megatron model.named_parameters()
# → all_gather_object 汇总所有训练 rank
# → 统一 PP 层编号 (layers.0 → layers.16 for PP rank 1)
# → PUT "training_params_meta" 到 MetaServer
# → GET "infer_params_meta" from MetaServer

# 3. 双方各自计算 TransferPlan
# 训练 rank 算: "我给谁发什么"
# 推理 rank 算: "我从谁收什么"
plan = TransferPlanBuilder(...).build_local_transfer_plan(
    infer_meta, train_meta, my_rank)

# 4. 建立 NCCL process group (所有训练+推理 rank 加入)
init_weights_update_group(master_addr, port, rank, world_size)
```

### 每步传输 (Writer 侧)

```python
# awex/writer/nccl_writer.py: _write_weights()
def _write_weights(self, step_id):
    # 1. 转格式 (Megatron → HF 命名)
    parameters = self.convert_parameters()
    
    # 2. 按 TransferPlan slice + 创建 isend
    p2p_op_list, _ = nccl_build_send_ops(
        parameters, self.transfer_plan, self.weights_update_group, -1)
    
    # 3. 批量执行所有 isend
    reqs = dist.batch_isend_irecv(p2p_op_list)
    for req in reqs:
        req.wait()
    
    # 4. barrier 等所有人完成
    dist.barrier(group=self.weights_update_group)
```

### 每步传输 (Reader 侧)

```python
# HybridEngine weights_reader.py: _update_weights()
def _update_weights(self, step_id):
    # 1. 按 TransferPlan 创建 irecv (预分配接收 buffer)
    for send_rank, operations in self.transfer_plan.operations.items():
        for op in operations:
            recv_tensor = self.parameters[op.recv_shard_meta.name]
            tensor_sliced = slice_tensor(recv_tensor, op, False)
            p2p_op_list.append(
                dist.P2POp(dist.irecv, tensor_sliced, send_rank, group=...))
    
    # 2. 批量执行所有 irecv
    batch_send_recv(send_ops=None, recv_ops=p2p_op_list, blocking=True)
    
    # 3. barrier
    dist.barrier(group=self.weights_update_group)
```

---

## 共卡模式 (Colocate) — 核心逻辑

训练和推理在**同一组 GPU** 上，分时复用。

### 与分卡的关键差异

| | 分卡 | 共卡 |
|--|------|------|
| 数据通道 | NCCL P2P (跨 GPU) | CUDA IPC (同 GPU 零拷贝) + NCCL (推理 rank 间) |
| 同步机制 | NCCL barrier | MetaServer 三步握手 |
| 地址问题 | 无（每次 alloc 新 buffer） | 每步 IPC handle 变化（offload 后地址变了） |
| 显存管理 | 各自独立 | Actor offload ↔ SGLang resume 交替 |

### 初始化 (_init_reader_in_colocate_mode)

```python
# HybridEngine weights_reader.py:1053-1094
def _init_reader_in_colocate_mode(self):
    device_id = torch.cuda.current_device()
    
    # 1. 注册自己的 (ip, device_id, rank) 到 MetaServer
    self.meta_server_client.add_object_to_set(
        "inference_device_rank_entries", (ip, device_id, self.transfer_rank))
    
    # 2. 等所有推理 rank 注册完
    self.meta_server_client.wait_set_until_size(
        "inference_device_rank_entries", self.infer_world_size)
    
    # 3. 等所有训练 rank 注册完
    self.meta_server_client.wait_set_until_size(
        "training_device_rank_entries", self.training_world_size)
    
    # 4. 建立 device mapping: (ip, device_id) → rank
    #    找到同卡的训练 rank (相同 ip + device_id)
    self.train_to_infer_device_mapping = {}  # train_rank → infer_rank
    self.infer_to_train_device_mapping = {}  # infer_rank → train_rank
    for ip, device_id, train_rank in training_entries:
        infer_rank = self.inference_device_mapping[(ip, device_id)]
        self.train_to_infer_device_mapping[train_rank] = infer_rank
        self.infer_to_train_device_mapping[infer_rank] = train_rank
    
    # 5. 计算 send_transfer_plan (用同卡训练 rank 的视角)
    #    这个 plan 定义了: 同卡训练 rank 的哪些参数 → 本推理 rank 的哪些位置
    train_rank = self.infer_to_train_device_mapping[self.transfer_rank]
    self.send_transfer_plan = plan_builder.build_local_transfer_plan(
        infer_meta, train_meta, train_rank)
    
    # 6. 创建 NcclColocateStreamBatchTransport (推理 rank 间 NCCL)
    self.colocate_transport = NcclColocateStreamBatchTransport(
        self.transfer_rank, self.infer_world_size)
```

### Writer 侧 (训练进程)

```python
# awex/writer/nccl_writer.py:204-285
@torch.no_grad()
def _write_weights_in_colocate_mode(self, step_id):
    start_time = time.time()
    
    # ── Step 1: 准备参数 ──
    tensors, names = self._prepare_params_for_colocate()
    # → release_grad_memory()
    # → convert_parameters() (Megatron → HF 命名)
    # → 返回 (tensors_list, names_list)
    
    # ── Step 2: 按 shape+dtype 分组合并 (减少 IPC 句柄数) ──
    group_tensors, metadata = group_tensors_by_shape_and_dtype(tensors)
    torch.cuda.synchronize()
    
    # ── Step 3: 释放原始权重到 CPU ──
    release_tensors(tensors)
    del tensors
    self.train_engine.release_memory_occupation("weights")
    # 通知所有推理 rank: 训练侧已 offload 完毕
    self.meta_server_client.add_object_to_set(
        "all_training_offloaded_weights", self.transfer_rank)
    
    # ── Step 4: IPC 共享内存序列化 ──
    if self.ipc_backend == "cpu":
        group_shared = [tensor.cpu().share_memory_() for tensor in group_tensors]
        serialized_weights = ipc_serialize((group_shared, metadata, names))
    else:
        group_shared = [tensor.cuda().share_memory_() for tensor in group_tensors]
        serialized_weights = cuda_ipc_serialize((group_shared, metadata, names))
    
    # ── Step 5: 通过 MetaServer 发布 IPC 句柄 ──
    # key 包含 ip + device_id + step_id (确保同卡推理 rank 能定位)
    key_suffix = f"_{ip_address}_{device_id}_{step_id}"
    serialized_weights_key = f"training_serialized_weights{key_suffix}"
    self.meta_server_client.put_object(
        serialized_weights_key,
        (self.transfer_rank, self.rank_info, serialized_weights))
    
    # ── Step 6: 等推理侧读完 (三步握手第一步) ──
    update_finished_key = f"weights_update_finished{key_suffix}"
    self.meta_server_client.get_object(update_finished_key, timeout=self.timeout)
    self.meta_server_client.delete_if_exists(update_finished_key)
    
    # ── Step 7: 释放 IPC shared tensor ──
    release_tensors(group_tensors)
    release_tensors(group_shared)
    del group_tensors, group_shared
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    
    # ── Step 8: 通知推理侧 "我已释放" (三步握手第二步) ──
    write_finished_key = f"write_finished{key_suffix}"
    self.meta_server_client.put_object(write_finished_key, True)
```

### Reader 侧 (推理进程) — collect_training_weights

```python
# HybridEngine weights_reader.py:1099-1127
def collect_training_weights(self, step_id):
    """从同卡训练进程获取权重 (CUDA IPC 零拷贝)。"""
    if not self.enable_colocate_mode:
        return
    
    # ⚠️ 每步都重新获取 IPC handle
    # 因为 offload→onload 后 tensor 地址变了, 旧 handle 无效
    ip_address = get_ip_address()
    device_id = torch.cuda.current_device()
    key = f"training_serialized_weights_{ip_address}_{device_id}_{step_id}"
    
    # 1. 从 MetaServer 获取 IPC 句柄 (不是权重数据本身, 只有几百字节)
    self.send_rank, self.send_rank_info, serialized_weights = \
        self.meta_server_client.get_object(key, timeout=self.timeout)
    
    # 2. IPC 反序列化 (零拷贝, 直接映射同卡 GPU 显存)
    if self.ipc_backend == "cpu":
        group_shared, metadata, names = ipc_deserialize(serialized_weights)
        group_shared = [t.to(device_id) for t in group_shared]
    else:
        group_shared, metadata, names = cuda_ipc_deserialize(serialized_weights)
    torch.cuda.synchronize()
    
    # 3. 从分组恢复为独立 tensor
    tensors = reconstruct_tensors_from_groups(group_shared, metadata)
    torch.cuda.synchronize()
    self.deserialized_weights = dict(zip(names, tensors))
```

### Reader 侧 — _update_weights_in_colocate_mode

```python
# HybridEngine weights_reader.py:1194-1242
def _update_weights_in_colocate_mode(self, step_id):
    assert self.enable_colocate_mode
    
    # ── Step 1: IPC 收集同卡训练权重 ──
    self.collect_training_weights(step_id)
    
    # ── Step 2: 推理 rank 间 NCCL 重分发 ──
    # 用 NcclColocateStreamBatchTransport 的递归分区算法
    # - 同卡部分: tensor copy (本 GPU 训练权重 → 本 GPU 推理模型)
    # - 跨卡部分: NCCL send/recv (其他 GPU 的训练权重)
    self.colocate_transport.update_weights_in_colocate_mode(
        self.train_to_infer_device_mapping,
        self.infer_to_train_device_mapping,
        self.transfer_rank,
        self.rank_coordinate,
        self.infer_world_size,
        self.send_transfer_plan,   # "同卡训练 rank 的视角"的 plan
        self.transfer_plan,         # "本推理 rank 的视角"的 plan
        self.weights_update_group,
        self.deserialized_weights,  # IPC 拿到的 (只有同卡训练 rank 有的部分)
        self.parameters,            # 推理模型的参数 (目标)
        step_id=step_id,
    )
    
    # ── Step 3: 清理 + 通知训练侧 ──
    self.deserialized_weights = None
    
    # 三步握手: 通知训练进程 "我读完了"
    ip_address = get_ip_address()
    device_id = torch.cuda.current_device()
    key_suffix = f"_{ip_address}_{device_id}_{step_id}"
    update_finished_key = f"weights_update_finished{key_suffix}"
    self.meta_server_client.put_object(update_finished_key, True)
    
    # barrier
    dist.barrier(group=self.weights_update_group)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # 等训练进程确认已释放 IPC 共享内存
    write_finished_key = f"write_finished{key_suffix}"
    self.meta_server_client.get_object_then_delete(write_finished_key)
```

### NcclColocateStreamBatchTransport 递归分区算法

```python
# awex/transfer/nccl_stream_batch.py:51-189
def update_weights_in_colocate_mode(self, ...):
    """
    核心逻辑: 对每个参数的每个 CommunicationOperation:
    
    1. 判断是 self-copy 还是 P2P:
       - 如果 peer_rank (经 device mapping 后) == 自己 → self copy
       - 否则 → NCCL isend/irecv
    
    2. Self-copy (同卡):
       从 IPC 拿到的 deserialized_weights 中 slice 出对应区域
       → copy_ 到推理模型的对应参数位置
    
    3. P2P (跨卡):
       发送: 从 deserialized_weights slice → isend 到目标推理 rank
       接收: 从其他推理 rank irecv → 写入推理模型参数
    
    4. 递归分区 (O(log N) rounds):
       不是一次性发所有 P2P，而是分轮执行
       每轮每个 rank 和一个特定 peer 交换
       避免 NCCL 死锁和内存峰值
    """
    
    # === 构建 P2P 操作 ===
    for peer_rank, ops in send_ops.items():
        mapped_peer_rank = train_to_infer_device_mapping.get(peer_rank, peer_rank)
        if mapped_peer_rank == transfer_rank:
            # Self-copy: 同卡训练权重 → 推理模型
            for op in ops:
                send_tensor = send_parameters[op.send_shard_meta.name]
                tensor_sliced = slice_tensor(send_tensor, op, True)
                tensors_to_copy.append(tensor_sliced)
        else:
            # P2P send to another inference rank
            for op in ops:
                send_tensor = send_parameters[op.send_shard_meta.name]
                tensor_sliced = slice_tensor(send_tensor, op, True)
                p2p_op = dist.P2POp(dist.isend, tensor_sliced.clone(), 
                                     mapped_peer_rank, group=weights_update_group)
                all_send_p2p_ops[mapped_peer_rank].append((op, p2p_op))
    
    # === Self-copy 执行 ===
    execute_tensors_to_copy(tensors_to_copy, recv_ops, recv_parameters)
    
    # === 递归分区执行 P2P ===
    self.execute_recursive_partition_stream_transfer(
        transfer_rank, world_size,
        all_send_p2p_ops, all_recv_p2p_ops,
        weights_update_group, ...)
```

---

## HybridEngine 显存管理 (共卡编排)

### MegatronBackend (训练侧)

```python
# asystem_runtime/backend/megatron_backend.py:312-400
def release_memory_occupation(self, tags=None):
    tags = tags or ["optimizer", "weights"]
    
    if "optimizer" in tags:
        self.offload_optimizer_states()
        # → offload_megatron_optimizer(self.model_optimizer)
        # → 每个 state tensor .cpu() + 清空 GPU 引用
    
    if "weights" in tags:
        self.offload_params()
        # → offload_megatron_model_to_cpu(self.model_engine)
        # → 每个 param .cpu() + 清空 GPU 引用
    
    self.offloaded.update(tags)
    
    if self.enable_colocate_mode:
        dist.barrier()  # 确保所有训练 rank 同步 offload 完

def resume_memory_occupation(self, tags=None):
    tags = tags or ["optimizer", "weights"]
    
    if "weights" in tags:
        self.load_params()
        # → load_megatron_model_to_gpu(self.model_engine)
        # → 每个 param .to(device)
    
    if "optimizer" in tags:
        self.load_optimizer_states()
        # → load_megatron_optimizer(self.model_optimizer)
    
    self.offloaded.difference_update(tags)
```

### SGLangBackend (推理侧)

```python
# asystem_runtime/backend/sglang/backend.py:652-700
def release_memory_occupation(self, tags=None):
    tags = tags or ["kv_cache", "weights"]
    
    # 先保存 buffers (layernorm running stats 等)
    if "weights" in tags:
        self.stashed_model_static_state = _export_static_state(model)
    
    # 下发到 SGLang scheduler
    obj = ReleaseMemoryOccupationReqInput(tags=tags)
    self.tokenizer_manager.release_memory_occupation(obj, None)
    # → SGLang scheduler.release_memory_occupation()
    #   → memory_saver_adapter.pause("kv_cache") — 释放 KV Cache
    #   → flush_cache() — 清空 cache 索引
    #   → memory_saver_adapter.pause("weights") — 释放模型权重

def resume_memory_occupation(self, tags=None):
    tags = tags or ["kv_cache", "weights"]
    
    obj = ResumeMemoryOccupationReqInput(tags=tags)
    self.tokenizer_manager.resume_memory_occupation(obj, None)
    # → SGLang scheduler.resume_memory_occupation()
    #   → memory_saver_adapter.resume("weights") — 恢复权重 GPU 显存
    #   → 恢复 stashed buffers
    #   → memory_saver_adapter.resume("kv_cache") — 恢复 KV Cache 显存
```

---

## 共卡完整时序 (一个训练 Step)

```
训练进程 (Megatron)                            推理进程 (SGLang)
═══════════════                               ═══════════════

[GPU: weights 14GB + optimizer 56GB]           [GPU: 0]
                                               (weights + kv_cache 已释放)

─── 训练阶段 ───
resume_memory_occupation(["weights","optimizer"])
  → load params + optimizer 到 GPU
[GPU: 84GB (weights + optimizer + grad)]

forward → backward → optimizer.step()
[GPU: 84GB]

─── 权重同步阶段 ───
release_memory_occupation(["optimizer"])
  → offload optimizer 到 CPU
[GPU: 28GB (weights + grad)]

release_grad_memory()
[GPU: 14GB (weights)]

convert_parameters() (Megatron→HF)
group_tensors_by_shape_and_dtype()
[GPU: ~28GB (原始 + grouped)]

release_tensors(原始)
release_memory_occupation("weights")
  → offload weights 到 CPU
[GPU: ~14GB (只剩 grouped)]

share_memory_() + cuda_ipc_serialize()
  → IPC 句柄生成 (几百字节)
put MetaServer(serialized_weights_key)
                                               ─── 权重接收 ───
                                               (被 WeightsExchangeReader.update_weights 调用)
                                               
                                               release_memory_occupation(["weights","kv_cache"])
                                                 → TMS pause (已在之前做过)
                                               
                                               等 "all_training_offloaded_weights" 就绪
                                               resume_memory_occupation("weights")
                                                 → TMS resume weights → 分配 GPU 显存
                                               [GPU: 14GB (空的 weights buffer)]
                                               
                                               get MetaServer(serialized_weights_key)
                                               cuda_ipc_deserialize() → 零拷贝映射
                                               reconstruct_tensors_from_groups()
                                               [GPU: 14GB (IPC 映射的训练权重)]
                                               
                                               ─── NCCL Reshard (推理 rank 间) ───
                                               NcclColocateStreamBatchTransport:
                                                 self-copy: 同卡训练权重 → 推理模型
                                                 P2P: 从其他推理 rank 接收跨 PP 层
                                               [GPU: 14GB (推理模型 weights 更新完毕)]
                                               
                                               put MetaServer("weights_update_finished")

get MetaServer("weights_update_finished")
  → "推理侧读完了"
release IPC shared tensors
gc.collect() + empty_cache()
[GPU: 0]

put MetaServer("write_finished")
                                               get MetaServer("write_finished")
                                               resume_memory_occupation("kv_cache")
                                                 → TMS resume kv_cache → 分配 KV 显存
                                               [GPU: 14GB weights + 55GB kv_cache]
                                               
                                               ─── Rollout 开始 ───
```

---

## 三步握手协议

共卡模式必须有三步握手，因为两个进程共享 GPU 显存：

```
           训练进程                          推理进程
              │                                │
   ① PUT "serialized_weights" ────────→ GET "serialized_weights"
     (IPC handle, 不是数据)               (IPC 零拷贝读 GPU 显存)
              │                                │
              │                    ② PUT "weights_update_finished"
    GET "weights_update_finished" ←────────    │
     (确认推理侧已拿到数据)                     │
              │                                │
    释放 IPC shared tensor                     │
    PUT "write_finished" ─────────────→ GET "write_finished"
              │                          (确认可以继续)
              │                                │
```

为什么需要三步？
- 如果训练侧在推理侧 IPC deserialize 前就释放 shared tensor → 推理侧读到垃圾
- 如果推理侧在训练侧确认释放前就开始 rollout → 可能和训练侧的下一步产生显存竞争

---

## device mapping 的作用

共卡模式通过 `(ip_address, device_id)` 确定哪个训练 rank 和哪个推理 rank 在同一个 GPU 上：

```python
# 例: 4 GPU, 训练 PP=2 TP=2, 推理 TP=4
#
# GPU 0: train_rank=0 (PP=0,TP=0) + infer_rank=0 (TP=0)
# GPU 1: train_rank=1 (PP=0,TP=1) + infer_rank=1 (TP=1)
# GPU 2: train_rank=2 (PP=1,TP=0) + infer_rank=2 (TP=2)
# GPU 3: train_rank=3 (PP=1,TP=1) + infer_rank=3 (TP=3)
#
# train_to_infer_device_mapping = {0:0, 1:1, 2:2, 3:3}
# infer_to_train_device_mapping = {0:0, 1:1, 2:2, 3:3}
#
# 推理 rank 0 通过 IPC 从训练 rank 0 拿到 layers 0-15 的 TP shard 0
# 推理 rank 0 还需要 layers 16-31 → 通过 NCCL 从推理 rank 2 获取
# (推理 rank 2 通过 IPC 从训练 rank 2 拿到了 layers 16-31)
```

---

## 关键代码路径索引

| 功能 | 文件 | 行号/方法 |
|------|------|----------|
| Writer 分卡 NCCL send | `awex/writer/nccl_writer.py` | `_write_weights()` |
| Writer 共卡 IPC serialize | `awex/writer/nccl_writer.py` | `_write_weights_in_colocate_mode()` |
| Writer 初始化 (共卡) | `awex/writer/nccl_writer.py` | `_init_writer_in_colocate_mode()` |
| Reader 分卡 NCCL recv | `HE/weights_exchange/weights_reader.py` | `_update_weights()` |
| Reader 共卡 IPC collect | `HE/weights_exchange/weights_reader.py` | `collect_training_weights()` |
| Reader 共卡 full flow | `HE/weights_exchange/weights_reader.py` | `_update_weights_in_colocate_mode()` |
| Reader 初始化 (共卡) | `HE/weights_exchange/weights_reader.py` | `_init_reader_in_colocate_mode()` |
| NCCL reshard 递归分区 | `awex/transfer/nccl_stream_batch.py` | `update_weights_in_colocate_mode()` |
| Transfer plan 构建 | `awex/transfer/transfer_plan.py` | `TransferPlanBuilder.build_local_transfer_plan()` |
| IPC 序列化 | `awex/util/tensor_util.py` | `cuda_ipc_serialize()` / `cuda_ipc_deserialize()` |
| 参数分组 | `awex/util/tensor_util.py` | `group_tensors_by_shape_and_dtype()` |
| MetaServer | `awex/meta/meta_server.py` | `MetaServer` / `MetaServerClient` |
| Megatron offload | `HE/backend/megatron_backend.py` | `release_memory_occupation()` |
| SGLang TMS release | `HE/backend/sglang/backend.py` | `release_memory_occupation()` |
| HE 统一入口 | `HE/hybrid_engine.py` | `release_memory_occupation()` |
| HE 权重更新协调 | `HE/weights_exchange/weights_reader.py` | `update_weights()` (line 369) |

（HE = `asystem_runtime/` in Asystem-HybridEngine repo）
