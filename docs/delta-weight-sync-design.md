# Delta Weight Sync 增量权重传输设计文档

## 1. 背景与动机

### 观察

RL 训练中相邻两步之间，在 bf16 精度下 **>98% 的权重元素不发生变化**。原因：
- 单步梯度更新量小（lr ~3e-6）
- bf16 精度有限（最小可表示变化 ~2^-7 × value）
- 大部分参数的梯度为 0 或接近 0（稀疏梯度）

### 当前全量传输的代价

| 模型规模 | bf16 全量 | 共卡 IPC + NCCL reshard | 分卡 NCCL P2P |
|---------|----------|----------------------|--------------|
| 7B | 14 GB | ~0.8s | ~0.8s |
| 100B | 200 GB | ~9s | ~20s |

### Delta 传输的理论收益

如果只传 1-3% 变化的元素（indices int32 + values bf16 = 6 bytes/element vs 2 bytes/element 全量）：

| 模型 | 全量 | Delta (2% sparsity) | 有效传输量 | 加速比 |
|------|------|------|------|------|
| 7B | 14 GB | indices 1.12GB + values 0.28GB = 1.4GB | 1.4 GB | ~10x |
| 100B | 200 GB | 20 GB | 20 GB | ~10x |

注：indices 用 int32 占 4 bytes，values 用 bf16 占 2 bytes，每个变化元素 6 bytes，而全量每个元素 2 bytes。当 sparsity > 67% 时 delta 比全量小。实际 >98% sparsity 时 delta 约为全量的 1/30。

---

## 2. 业界开源方案调研

### 2.1 vLLM 官方 Sparse Weight Transfer (main 分支, 已合并)

**vLLM 已在 main 分支实现了完整的 sparse weight patch 基础设施**，是目前最成熟的参考。

核心接口 (`vllm/distributed/weight_transfer/base.py`):

```python
@dataclass
class SparseWeightPatch:
    """A sparse in-place patch for one existing parameter."""
    name: str
    indices: torch.Tensor   # int32, flat indices
    values: torch.Tensor    # same dtype as parameter

@dataclass
class WeightTransferUpdateInfo:
    update_kind: Literal["dense", "sparse_flat"] = "dense"
    num_updates_list: list[int] | None = None  # 每个参数的变化元素数
```

NCCL 实现 (`vllm/distributed/weight_transfer/nccl_engine.py`):

```python
# Trainer 侧 (rank 0 broadcast)
@staticmethod
def trainer_send_sparse_weights(iterator: Iterator[SparseWeightPatch], trainer_args):
    for patch in iterator:
        args.group.broadcast(patch.indices, src=0, stream=stream)
        args.group.broadcast(patch.values, src=0, stream=stream)

# Worker 侧 (receive + apply)
def receive_sparse_weights(self, update_info, apply_patches):
    for name, dtype_name, num_updates in zip(names, dtypes, num_updates_list):
        indices = torch.empty(num_updates, dtype=torch.int32, device=device)
        values = torch.empty(num_updates, dtype=dtype, device=device)
        self.model_update_group.broadcast(indices, src=0, stream=stream)
        self.model_update_group.broadcast(values, src=0, stream=stream)
        apply_patches([SparseWeightPatch(name=name, indices=indices, values=values)])
```

**关键特征**：
- `SparseWeightPatch`: flat indices (int32) + values (param dtype)
- NCCL broadcast: 先发 indices 再发 values，per-parameter
- Worker 侧通过 `apply_patches` 回调做 in-place 更新
- 不支持 packed mode（sparse + packed 互斥）
- `num_updates_list` 预先告知每个参数的变化数量（用于预分配 tensor）

**局限**：
- 只支持 broadcast（rank 0 → all workers），不支持 P2P 或 all-to-all
- 不处理 TP mismatch（假设 trainer 和 inference 的 shard 对齐）
- 无 index remap 能力

### 2.2 HuggingFace TRL Delta Weight Sync (PR #5417, 2026.3, Draft)

HuggingFace TRL PR #5417 实现了 delta weight sync：

### 架构

```
Trainer                          vLLM (远端)
  │                                │
  ├─ BF16ChangeDetector            │
  │   (optimizer pre/post hook)    │
  │                                │
  ├─ 计算 mask (!=)               │
  ├─ 编码 sparse safetensors       │
  │   {name}.indices (int32)       │
  │   {name}.values (bf16)         │
  │                                │
  ├─ 上传到 HF Hub Bucket ────────→ 下载 safetensors
  │                                ├─ 维护 CPU bf16 snapshot
  │                                ├─ snapshot[indices] = values
  │                                └─ load_weights(full_tensor)
```

### 2.3 方案对比

| | vLLM (官方 main) | TRL (PR #5417) | 我们 (AWEX) |
|--|--------|------|------|
| **数据格式** | `SparseWeightPatch(name, indices, values)` | sparse safetensors (`{name}.indices` + `{name}.values`) | 对齐 vLLM 格式 |
| **传输层** | NCCL broadcast | HF Hub Bucket (文件) | NCCL P2P + CUDA IPC |
| **变化检测** | 不提供（外部生成 patch） | `BF16ChangeDetector`（optimizer hook） | 对齐 TRL 检测方式 |
| **推理侧 apply** | `apply_patches` 回调（in-place） | CPU snapshot → `load_weights`（full tensor） | in-place `scatter_` |
| **TP 支持** | 仅 broadcast（假设 shard 对齐） | 单卡 | **完整支持 TP mismatch**（index remap） |
| **PP 支持** | 无 | 无 | **完整支持**（IPC delta + NCCL reshard full） |
| **EP 支持** | 无 | 无 | **完整支持**（未激活 expert delta 为空） |
| **状态** | 已合并 main | Draft PR | 设计中 |

**设计决策**：
- `SparseWeightPatch` 格式对齐 vLLM 官方（`name + flat_indices_int32 + values`），便于未来与 vLLM 生态互通
- 变化检测复用 TRL 的 `BF16ChangeDetector` 思路（optimizer pre/post hook）
- 传输层用 AWEX 的 NCCL P2P / IPC（不走文件系统），支持 5D 并行的 index remap

1. **单卡 trainer + 单卡 vLLM**：不支持 TP/PP/EP
2. **vLLM 无 sparse in-place API**：必须在 CPU 维护 full snapshot → load_weights
3. **文件传输**：HF Hub Bucket 延迟高（秒级），不适合高频同步
4. **无 NCCL 集成**：依赖文件系统做数据平面

---

## 3. 并行策略分析（核心难点）

### 3.1 当前 AWEX Colocate 的并行模型

```
训练侧 (Megatron):  TP=4, PP=2, EP=4  → 8 GPU
推理侧 (SGLang):    TP=4, PP=1         → 4 GPU (DP=2 共 8 GPU)
```

**约束**：当前 colocate 模式要求 `infer_world_size == train_world_size`。

每个推理 rank 和一个训练 rank 配对在同一个 GPU 上：
```
GPU 0: Train rank 0 (PP=0, TP=0, layers 0-15 的 shard 0) 
        + Infer rank 0 (TP=0, 需要全部 layers 的 shard 0)
GPU 4: Train rank 4 (PP=1, TP=0, layers 16-31 的 shard 0)
        + Infer rank 4 (TP=0, 需要全部 layers 的 shard 0)
```

### 3.2 全量传输的 reshard 流程

```
Step 1: IPC 获取同卡训练 rank 的权重 (零拷贝)
  Infer rank 0 ← Train rank 0 的权重 (layers 0-15, shard 0)
  Infer rank 4 ← Train rank 4 的权重 (layers 16-31, shard 0)

Step 2: NCCL reshard (推理 rank 间，获取缺失部分)
  Infer rank 0 需要 layers 16-31 ← 从 Infer rank 4 NCCL recv
  Infer rank 4 需要 layers 0-15  ← 从 Infer rank 0 NCCL recv
```

### 3.3 Delta 在不同并行策略下的行为

#### Case A: TP 相同 (训练 TP=4, 推理 TP=4)

**最简单的情况**。每个训练 rank 的 shard 和对应推理 rank 的 shard **完全对齐**。

```
Train rank 0 持有 qkv_proj.weight 的 [0:750, :]  (TP shard 0/4)
Infer rank 0 也需要 qkv_proj.weight 的 [0:750, :] (同样是 TP shard 0/4)
```

Delta 的 indices 在训练 shard 上计算 → 直接可以在推理 shard 上 scatter apply。**无需 index 转换**。

#### Case B: PP 不同 (训练 PP=2, 推理 PP=1)

PP 决定了每个 rank 持有哪些 **layers**。不同 PP rank 持有**不同的** layers，它们之间没有重叠。

```
Train PP=0: layers 0-15 (delta 在这些 layers 上计算)
Train PP=1: layers 16-31 (delta 在这些 layers 上计算)
Infer PP=0: layers 0-31 (需要所有 layers)
```

对于 IPC step（同卡）：推理 rank 从配对训练 rank 获取**该训练 rank 有的** layers 的 delta。
对于 NCCL reshard step：推理 rank 从**其他**推理 rank 获取缺失 layers 的 delta。

**关键问题**：NCCL reshard 时，发送方持有的是从 IPC 拿到的 delta（sparse），接收方需要的是这些 layers 的完整权重更新。

**两种策略**：
- **策略 1（简单）**：IPC 传 delta，推理侧先 apply 到本地 snapshot 得到 full tensor，再用现有 NCCL reshard 传 full tensor
- **策略 2（激进）**：IPC 传 delta，NCCL reshard 也传 delta（需要对方也维护 snapshot）

推荐**策略 1**：IPC 步骤省的是内存分配和 serialize 开销（同卡无带宽瓶颈），NCCL reshard 步骤走全量（跨卡需要完整数据）。

#### Case C: TP 不同 (训练 TP=4, 推理 TP=8)  ✅ 支持

训练 shard 和推理 shard 的切分粒度不同，一个训练 shard 映射到多个推理 shard。

```
Train TP=4: qkv_proj shard [0:750, :] (dim=0 切分)
Infer TP=8: qkv_proj shard [0:375, :] 和 [375:750, :]
```

**解法：复用 TransferPlan 的 overlap 信息做 index 映射。**

AWEX 的 `CommunicationOperation` 已经为每对 (训练 shard, 推理 shard) 计算了精确的重叠区域：
```python
@dataclass
class CommunicationOperation:
    train_slices: Tuple[slice, ...]  # 训练 shard 中要发送的切片
    inf_slices: Tuple[slice, ...]    # 推理 shard 中对应接收的切片
    overlap_shape: Tuple[int, ...]   # 重叠区域的 shape
```

例如 train rank 0 → infer rank 0 的 op：
```python
train_slices = (slice(0, 375), slice(None))   # 训练 shard 的前 375 行
inf_slices   = (slice(0, 375), slice(None))   # 推理 shard 的前 375 行
```

train rank 0 → infer rank 1 的 op：
```python
train_slices = (slice(375, 750), slice(None))  # 训练 shard 的后 375 行
inf_slices   = (slice(0, 375), slice(None))    # 推理 shard 的前 375 行（它只有 375 行）
```

**Delta Index 映射算法**：

```python
def remap_delta_indices(
    flat_indices: torch.Tensor,      # 训练 shard 上的 flat indices
    values: torch.Tensor,            # 对应的 values
    train_shape: tuple,              # 训练 shard 的 shape
    op: CommunicationOperation,      # 定义 overlap 的操作
) -> tuple[torch.Tensor, torch.Tensor]:
    """将训练 shard 的 flat delta indices 映射到推理 shard 的 flat indices。
    
    Steps:
    1. flat → multi-dim (unflatten)
    2. filter: 只保留落在 train_slices 范围内的 indices
    3. remap: 从 train_slices 空间映射到 inf_slices 空间
    4. multi-dim → flat (flatten for target shape)
    """
    ndim = len(train_shape)
    cols = train_shape[-1] if ndim >= 2 else 1
    
    # Step 1: unflatten to get per-dim indices
    if ndim == 2:
        rows = flat_indices // cols
        col_idx = flat_indices % cols
    elif ndim == 1:
        rows = flat_indices
        col_idx = None
    else:
        # General N-dim: use unravel_index
        multi_idx = torch.unravel_index(flat_indices, train_shape)
        rows = multi_idx[0]
        col_idx = multi_idx[1] if ndim >= 2 else None
    
    # Step 2: filter — 只保留在 train_slices 范围内的
    mask = torch.ones(len(flat_indices), dtype=torch.bool, device=flat_indices.device)
    for dim, s in enumerate(op.train_slices):
        dim_idx = multi_idx[dim] if ndim > 2 else (rows if dim == 0 else col_idx)
        if dim_idx is not None and s != slice(None):
            start = s.start or 0
            stop = s.stop or train_shape[dim]
            mask &= (dim_idx >= start) & (dim_idx < stop)
    
    if not mask.any():
        return torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=values.dtype)
    
    filtered_flat = flat_indices[mask]
    filtered_values = values[mask]
    
    # Step 3: remap — train_slices 空间 → inf_slices 空间
    # 对于 sharding dim: new_idx = old_idx - train_start + inf_start
    infer_shape = op.overlap_shape  # 或从 inf_slices 推导
    
    if ndim == 2:
        f_rows = filtered_flat // cols
        f_cols = filtered_flat % cols
        # Remap each dim
        t_s0 = op.train_slices[0]
        i_s0 = op.inf_slices[0]
        new_rows = f_rows - (t_s0.start or 0) + (i_s0.start or 0)
        t_s1 = op.train_slices[1]
        i_s1 = op.inf_slices[1]
        if t_s1 != slice(None) and i_s1 != slice(None):
            new_cols = f_cols - (t_s1.start or 0) + (i_s1.start or 0)
        else:
            new_cols = f_cols
        # Flatten for inference shard
        infer_cols = op.overlap_shape[1] if len(op.overlap_shape) > 1 else cols
        new_flat = new_rows * infer_cols + new_cols
    elif ndim == 1:
        t_s0 = op.train_slices[0]
        i_s0 = op.inf_slices[0]
        new_flat = filtered_flat - (t_s0.start or 0) + (i_s0.start or 0)
    
    return new_flat.to(torch.int32), filtered_values
```

**时间复杂度**：O(N_changed) per parameter，纯 GPU tensor 操作，无 CPU 瓶颈。

对于 100B 模型 2% sparsity：~2B total params / TP8 = 250M per shard，2% = 5M indices。GPU 上做 5M 个 index 的 filter + remap < 1ms。

#### Case D: EP (Expert Parallelism)

MoE 模型中 EP 将不同 experts 分配到不同 rank。RL 训练中，每步只有部分 experts 被激活并获得梯度更新。

**EP 对 delta 的影响**：
- 没有被激活的 expert → 该 rank 的 expert 权重 delta = 0（完全不用传）
- 被激活的 expert → 正常计算 delta

EP 实际上让 delta 更稀疏：如果只有 10/64 个 experts 在某步被激活，那 EP 维度上只有 ~15% 的 ranks 需要传 delta。

### 3.4 5D 并行兼容性总结

| 维度 | 训练 vs 推理 | Delta 支持 | 实现方式 |
|------|------------|-----------|---------|
| **TP** | 相同 | ✅ | indices 直接对齐，无需转换 |
| **TP** | 不同 | ✅ | 用 TransferPlan.CommunicationOperation 的 train_slices/inf_slices 做 index filter + remap |
| **PP** | 不同 | ✅ | 不同 layers 无重叠，IPC 传 delta，NCCL reshard 传 full |
| **DP** | - | ✅ | 所有 DP rank 权重相同，只需一个 DP head 发送 |
| **EP** | - | ✅ | 不同 experts 无重叠，未激活 expert 的 delta 为空 |
| **CP** | - | ✅ | Context parallelism 不影响权重分片 |

#### NCCL Reshard 的 Delta 策略

NCCL reshard（推理 rank 间交换跨 PP 层的权重）也支持 delta：

```
分卡模式 (训练和推理在不同 GPU):
  训练 rank → NCCL send → 推理 rank
  ✅ delta: 直接 send (remapped_indices, values)，对方 scatter apply

共卡模式 (同 GPU):
  训练 rank → IPC → 推理 rank 0 → NCCL reshard → 推理 rank 1
  IPC 步骤: ✅ delta
  NCCL reshard 步骤: 取决于模式
    - Same TP: 推理 rank 间无需 reshard（各自已有完整 shard）
    - Diff PP: 推理 rank 间交换跨 PP 层
      - ✅ 可传 delta: 发送方 apply delta 后拿到 full tensor，
            但传给接收方时可以只传 delta
            (两边都维护 snapshot，发方计算 diff 后发 sparse)
      - 或简化: 直接传 full tensor（跨卡 NVLink 带宽充足）
```

**推荐**：第一版 NCCL reshard 传 full tensor（现有路径不改），delta 只用于 IPC 和分卡 NCCL 直传。原因是 NCCL reshard 的数据量本身就只是跨 PP 层的部分参数，已经是 1/PP 的量。

---

## 4. 设计方案

### 4.1 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│  Training Rank (Megatron)                                    │
│                                                              │
│  ┌─────────────────────┐                                     │
│  │ DeltaWeightDetector  │                                     │
│  │ ├─ CPU bf16 snapshot │                                     │
│  │ ├─ compute_delta()   │→ {name: (indices, values)}         │
│  │ └─ update_snapshot() │                                     │
│  └─────────────────────┘                                     │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────────┐                                     │
│  │ DeltaSerializer      │                                     │
│  │ ├─ pack sparse       │→ IPC serialize (indices+values)    │
│  │ └─ fallback full     │   or MetaServer put                │
│  └─────────────────────┘                                     │
└────────────────┬────────────────────────────────────────────┘
                 │ IPC (same GPU) or MetaServer
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Inference Rank (SGLang)                                     │
│                                                              │
│  ┌─────────────────────┐                                     │
│  │ DeltaApplier         │                                     │
│  │ ├─ IPC deserialize   │                                     │
│  │ ├─ scatter_(indices, │← sparse in-place update            │
│  │ │        values)     │                                     │
│  │ └─ NCCL reshard      │← full tensor for cross-PP layers  │
│  └─────────────────────┘                                     │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 变化检测器 (Training 侧)

```python
class DeltaWeightDetector:
    """在 Megatron 侧检测 bf16 权重变化。
    
    维护 CPU pinned bf16 snapshot，optimizer step 后对比检测变化。
    """
    
    def __init__(self, model, converter):
        self._snapshot: dict[str, torch.Tensor] = {}  # HF name → CPU bf16
        self._converter = converter  # Megatron → HF 命名转换
        self._step_count = 0
        
        # 首次快照
        for megatron_name, param in model.named_parameters():
            hf_name = converter.convert_name(megatron_name)
            if hf_name:
                self._snapshot[hf_name] = param.data.to(torch.bfloat16).cpu().pin_memory().clone()
    
    def compute_delta(self, model) -> DeltaResult:
        """比较当前权重与 snapshot，返回变化的 sparse indices + values。
        
        Returns:
            DeltaResult with:
              - deltas: {hf_name: (indices_int32, values_bf16)}
              - stats: {total_elements, changed_elements, sparsity}
        """
        deltas = {}
        total_elements = 0
        changed_elements = 0
        
        for megatron_name, param in model.named_parameters():
            hf_name = self._converter.convert_name(megatron_name)
            if hf_name is None:
                continue
            
            current_bf16 = param.data.to(torch.bfloat16)
            old_bf16 = self._snapshot[hf_name].to(param.device)
            
            # Element-wise 比较 (bit-exact in bf16)
            mask = (current_bf16 != old_bf16)
            numel = current_bf16.numel()
            total_elements += numel
            
            if mask.any():
                indices = mask.flatten().nonzero(as_tuple=False).squeeze(1).to(torch.int32)
                values = current_bf16.flatten()[indices.long()]
                deltas[hf_name] = (indices, values)
                changed_elements += indices.numel()
                
                # 更新 snapshot
                self._snapshot[hf_name].copy_(current_bf16.cpu())
        
        self._step_count += 1
        sparsity = 1.0 - changed_elements / max(total_elements, 1)
        
        return DeltaResult(
            deltas=deltas,
            total_elements=total_elements,
            changed_elements=changed_elements,
            sparsity=sparsity,
            step=self._step_count,
        )

@dataclass
class DeltaResult:
    deltas: dict[str, tuple[torch.Tensor, torch.Tensor]]  # name → (indices, values)
    total_elements: int
    changed_elements: int
    sparsity: float
    step: int
    
    @property
    def delta_size_bytes(self) -> int:
        """Delta 的传输字节数。"""
        total = 0
        for indices, values in self.deltas.values():
            total += indices.numel() * 4  # int32
            total += values.numel() * 2   # bf16
        return total
    
    @property
    def full_size_bytes(self) -> int:
        return self.total_elements * 2  # bf16
    
    def should_use_delta(self, threshold: float = 0.5) -> bool:
        """当 delta 大小 < threshold × full 大小时使用 delta。"""
        return self.delta_size_bytes < threshold * self.full_size_bytes
```

### 4.3 传输编码

```python
# 两种编码模式

class DeltaTransferMode(Enum):
    FULL = "full"      # 全量传输 (首次或 fallback)
    SPARSE = "sparse"  # 稀疏增量传输

@dataclass
class DeltaTransferPayload:
    mode: DeltaTransferMode
    version: int
    # FULL mode: tensors dict
    # SPARSE mode: {name: (indices_int32, values_bf16)} packed
    
def serialize_delta_for_ipc(delta_result: DeltaResult) -> bytes:
    """将 delta 编码为 IPC 传输格式。
    
    格式: {
        f"{name}.indices": int32_tensor,
        f"{name}.values": bf16_tensor,
        "__meta__": metadata_tensor (version, num_params, etc.)
    }
    """
    tensors = {}
    names = []
    for name, (indices, values) in delta_result.deltas.items():
        tensors[f"{name}.indices"] = indices.cuda()  # GPU 上做 IPC
        tensors[f"{name}.values"] = values.cuda()
        names.append(name)
    
    # Metadata packed as tensor
    meta = torch.tensor([delta_result.step, len(names)], dtype=torch.int64)
    tensors["__meta__"] = meta
    tensors["__names__"] = names  # pickle-able, goes via MetaServer
    
    return cuda_ipc_serialize(tensors)
```

### 4.4 接收端 (SGLang 侧)

```python
class DeltaWeightApplier:
    """SGLang 侧的增量权重应用器。"""
    
    def apply_delta(self, model, delta_payload):
        """In-place sparse scatter update。
        
        SGLang 的 model weights 已经在 GPU 上，直接 scatter 即可。
        不需要像 TRL/vLLM 那样维护 CPU snapshot。
        """
        for name, param in model.named_parameters():
            hf_name = self._convert_name(name)
            if f"{hf_name}.indices" in delta_payload:
                indices = delta_payload[f"{hf_name}.indices"].long()
                values = delta_payload[f"{hf_name}.values"]
                # In-place scatter (关键：直接在 GPU param 上操作)
                param.data.flatten().scatter_(0, indices, values)
    
    def apply_full(self, model, full_tensors):
        """全量更新（首次或 fallback）。"""
        for name, param in model.named_parameters():
            hf_name = self._convert_name(name)
            if hf_name in full_tensors:
                param.data.copy_(full_tensors[hf_name])
```

### 4.5 分卡模式的 Delta NCCL 传输

分卡模式下（训练和推理在不同 GPU），delta 直接通过 NCCL P2P 传输，完全替代全量 broadcast：

```python
# 训练侧 (Writer)
def write_weights_delta(self, step_id, delta_result):
    """分卡模式：NCCL send delta 到推理侧。"""
    for peer_rank, ops in self.transfer_plan.operations.items():
        for op in ops:
            name = op.send_shard_meta.name
            if name not in delta_result.deltas:
                # 该参数无变化，发送空标记
                dist.isend(torch.tensor([0], dtype=torch.int32), peer_rank, group=...)
                continue
            
            indices, values = delta_result.deltas[name]
            # Remap indices to target inference shard space
            remapped_indices, remapped_values = remap_delta_indices(
                indices, values, 
                self.train_shard_shapes[name],
                op,
            )
            # 先发 size，再发 indices + values
            size_tensor = torch.tensor([remapped_indices.numel()], dtype=torch.int32)
            dist.isend(size_tensor, peer_rank, group=...)
            if remapped_indices.numel() > 0:
                dist.isend(remapped_indices, peer_rank, group=...)
                dist.isend(remapped_values, peer_rank, group=...)

# 推理侧 (Reader)
def update_weights_delta(self, step_id):
    """分卡模式：NCCL recv delta 并 scatter apply。"""
    for send_rank, ops in self.transfer_plan.operations.items():
        for op in ops:
            name = op.recv_shard_meta.name
            # Recv size
            size_tensor = torch.empty(1, dtype=torch.int32, device=self.device)
            dist.irecv(size_tensor, send_rank, group=...)
            size = size_tensor.item()
            if size == 0:
                continue
            # Recv indices + values
            indices = torch.empty(size, dtype=torch.int32, device=self.device)
            values = torch.empty(size, dtype=torch.bfloat16, device=self.device)
            dist.irecv(indices, send_rank, group=...)
            dist.irecv(values, send_rank, group=...)
            # In-place scatter apply
            param = self.parameters[name]
            target_slice = param[op.inf_slices]
            target_slice.flatten().scatter_(0, indices.long(), values)
```

**带宽收益**：分卡模式下带宽是真正的瓶颈（NVLink 300GB/s 或 IB 200Gb/s）。7B 模型 2% delta: 14GB → ~0.5GB，节省 96% 传输时间。

### 4.6 共卡 NCCL Reshard 的 Delta 处理

在 PP 不同时，推理 rank 需要从其他推理 rank 获取缺失层的权重。

**策略：IPC 用 delta，NCCL reshard 用 full tensor。**

```python
def execute_colocate_weight_update_with_delta(self, version, delta_result):
    """
    Step 1: IPC 获取同卡训练 rank 的 delta (sparse)
    Step 2: Apply delta 到本地持有的层 (in-place scatter)
    Step 3: NCCL reshard 发送更新后的 full tensor 给需要的推理 rank
    """
    # Step 1: IPC 获取 delta
    delta_payload = self._ipc_receive_delta(version)
    
    # Step 2: Apply delta 到本地权重 (只有同卡训练 rank 有的 layers)
    local_layers = self._get_layers_from_paired_train_rank()
    for name, param in self._get_model().named_parameters():
        if self._layer_of(name) in local_layers:
            if f"{name}.indices" in delta_payload:
                indices = delta_payload[f"{name}.indices"].long()
                values = delta_payload[f"{name}.values"]
                param.data.flatten().scatter_(0, indices, values)
    
    # Step 3: NCCL reshard (用现有的 recursive partition 算法)
    # 此时本地的 layers 已经是最新的 full tensor，可以正常 slice + send
    self._colocate_transport.update_weights_in_colocate_mode(
        ...,
        send_parameters=self._get_local_params(),  # 已 apply delta 的 full tensors
        recv_parameters=self._model_parameters,
        ...
    )
```

**为什么 NCCL reshard 不传 delta？**

1. NCCL reshard 的 transfer_plan 是基于 tensor shape 和 global_offset 预计算的固定 slice 操作
2. 改成 sparse 需要动态确定传输大小（每步不同），打破预计算假设
3. 跨卡（NVLink/IB）带宽足够高，全量传输本身已经很快
4. 实现复杂度大幅增加，收益不成比例

### 4.6 显存开销分析

#### Training 侧

| 组件 | 额外显存 | 说明 |
|------|---------|------|
| CPU bf16 snapshot | 0 (CPU) | pinned memory，不占 GPU |
| current_bf16 转换 | ~model_size/TP | 临时分配，立即释放 |
| mask (bool) | ~model_size/TP/8 | 临时 |
| indices (int32) | ~2% × model_size/TP × 2 | 稀疏，很小 |
| values (bf16) | ~2% × model_size/TP | 稀疏，很小 |

对于 100B 模型 TP=8：每 rank 约 12.5GB 参数。snapshot 在 CPU (12.5GB pinned)。GPU 临时开销 ~12.5GB（current_bf16 对比时）但可以分批处理避免。

#### Inference 侧

**零额外开销**。直接 in-place scatter 到已有的模型参数上。

---

## 5. 实现分阶段

### Phase 1: Delta 检测 + 统计 (验证假设)

- 实现 `DeltaWeightDetector`
- 在现有 `execute_colocate_weight_update` 中添加 delta 统计 logging
- 不改变传输路径，只 log sparsity 和 delta size
- 验证 >98% sparsity 的假设

### Phase 2: 分卡 Delta NCCL 传输

- 实现 `remap_delta_indices`（复用 TransferPlan 的 CommunicationOperation）
- 替换分卡模式的 NCCL 全量 broadcast 为 delta send/recv
- 支持所有 TP/PP/EP 组合
- 首次 anchor 全量 + 后续 delta sparse

### Phase 3: 共卡 Delta IPC 传输

- IPC serialize delta（indices + values）替代全量权重
- SGLang 侧 in-place scatter apply
- NCCL reshard 仍用 full tensor（或可选 delta）

### Phase 4: 优化

- 分批计算 delta 避免 GPU 峰值显存
- CUDA kernel 加速 bf16 比较 + nonzero + remap
- 异步 delta 计算（overlap with 其他操作）
- 全 NCCL delta（包括 reshard 步骤）

---

## 6. 配置接口

```yaml
# awex config extension
weight_transfer:
  enable_delta: true
  delta_threshold: 0.5       # delta_size < 50% full_size 时启用
  force_full_every_n: 100    # 每 100 步强制全量（防止累积误差）
  delta_batch_size: 32       # 分批计算 delta 的参数数量
```

---

## 7. 风险与缓解

| 风险 | 缓解 |
|------|------|
| bf16 比较漏掉 fp32→bf16 精度变化 | 比较的是 bf16 cast 后的值，和推理侧看到的一致 |
| 长期运行精度漂移 | force_full_every_n 定期全量同步 |
| snapshot CPU 内存开销 | pinned memory，不影响 GPU；100B/TP8 约 12.5GB per rank |
| indices int32 溢出 | 单参数 numel < 2^31 (4B elements)，100B 最大单参数 ~100M elements |
| delta 计算 GPU 峰值 | 分批处理，每次只对比一小组参数 |
| NCCL reshard 与 delta 不兼容 | 设计中 NCCL reshard 始终用 full tensor |

---

## 8. 与现有 AWEX 架构的集成点

```
AwexMegatronAdapter.execute_colocate_weight_update(version)
  │
  ├─ if version == 0: 全量 (anchor)
  │     └─ 现有 IPC serialize 路径
  │     └─ delta_detector.update_snapshot() — 建立 baseline
  │
  └─ if version > 0:
        ├─ delta_detector.compute_delta(model) → DeltaResult
        ├─ if delta_result.should_use_delta():
        │     ├─ 共卡: serialize_delta_for_ipc() → MetaServer put
        │     └─ 分卡: remap_delta_indices() per op → NCCL send
        └─ else:
              └─ 现有全量路径 (fallback)

AwexSGLangAdapter.execute_colocate_weight_update(version)
  │
  ├─ MetaServer get payload (共卡) or NCCL recv (分卡)
  ├─ if payload.mode == SPARSE:
  │     ├─ deserialize delta
  │     ├─ per-param scatter_(indices, values)  — in-place GPU update
  │     └─ NCCL reshard (full tensor for cross-PP layers)
  └─ if payload.mode == FULL:
        └─ 现有全量路径

分卡模式 NCCLWeightsWriter._write_weights(step_id)
  │
  ├─ if delta_enabled and step_id > 0:
  │     ├─ delta_detector.compute_delta()
  │     ├─ for each (peer_rank, ops) in transfer_plan:
  │     │     └─ remap_delta_indices(indices, values, op) → NCCL isend
  │     └─ barrier
  └─ else:
        └─ 现有全量 NCCL send 路径

分卡模式 NCCLWeightsReader._update_weights(step_id)
  │
  ├─ if delta_enabled and step_id > 0:
  │     ├─ for each (send_rank, ops) in transfer_plan:
  │     │     └─ NCCL irecv (size, indices, values) → scatter_ apply
  │     └─ barrier → flush_cache
  └─ else:
        └─ 现有全量 NCCL recv 路径
```

### 5D 并行下的完整数据流 (示例)

```
训练: TP=4, PP=2, EP=4 (32 GPU)
推理: TP=8, PP=1 (32 GPU, DP=4)

Train rank 0 (PP=0, TP=0): 持有 layers 0-15 的 shard 0/4
  │
  ├─ compute_delta(): 发现 layers 0-15 中 1.5% 参数变化
  │
  ├─ TransferPlan says: 
  │     op1: train_slices=(0:375,:) → infer rank 0, inf_slices=(0:375,:)
  │     op2: train_slices=(375:750,:) → infer rank 1, inf_slices=(0:375,:)
  │
  ├─ remap_delta_indices(flat_indices, op1):
  │     filter: 只保留 row < 375 的 indices
  │     remap: 不变（train offset == infer offset）
  │     → NCCL send to infer rank 0
  │
  └─ remap_delta_indices(flat_indices, op2):
        filter: 只保留 375 <= row < 750 的 indices
        remap: row -= 375 (对齐到 infer rank 1 的 local space)
        → NCCL send to infer rank 1

Infer rank 0 (TP=0): 持有 all layers 的 shard 0/8
  │
  ├─ recv delta from train rank 0: layers 0-15 的部分
  ├─ recv delta from train rank 4: layers 16-31 的部分
  │
  └─ per-param scatter_(indices, values) — done, no NCCL reshard needed
     (因为每个 infer rank 已经从对应的 train ranks 收到了所有 layers 的 delta)
```

