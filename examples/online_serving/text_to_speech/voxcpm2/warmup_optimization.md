# VoxCPM2 Warmup 优化报告

## 1. 问题现状

VoxCPM2 服务启动后，首次用户请求耗时 ~28s，第二次 ~1.8s。

```
# startup_timecost.txt 修复前首次请求
00:10:20  TTS speech request received
00:10:23  CJK split map built (19789 entries) + torch.compile applied (~3s)
00:10:47  CUDA Graph captured for scaffold + residual (~24s)
00:10:48  Response sent (200 OK)  ← 首次请求总耗时 ~28s

# 第二次请求
00:11:04  TTS speech request received
00:11:06  Response sent (200 OK)  ← 耗时 ~1.8s
```

**根本原因**：首次请求背负了全部"冷启动"开销：

| 阶段 | 耗时 | 说明 |
|------|------|------|
| CJK 分词表构建 | ~3s | 19789 条目的 tokenizer 查表扫描（serving 侧 + model 侧各一次） |
| `torch.compile` JIT 首次执行 | 包含在上述 ~3s 内 | LocDiT、feat_encoder、AudioVAE、projections 的首次 Dynamo 追踪（此环境有编译缓存） |
| scaffold/residual CUDA Graph capture | ~24s | vLLM PagedAttention 的 CUDA Graph 捕获（含 compiled 代码首次执行的 GPU kernel JIT） |
| 实际推理 | ~1s | warmup 完成后的首次推理 |

这些操作全部发生在**第一个用户请求的处理路径上**，用户体感极差。

## 2. 与 qwen3-tts 的差异

qwen3-tts 不存在同类问题，原因在于架构不同：

| | qwen3-tts | VoxCPM2 |
|---|---|---|
| CUDA Graph 目标 | 独立 tokenizer decoder | PagedAttention scaffold/residual LLM |
| 依赖 vLLM `ForwardContext` | 不依赖 | 依赖（`attn_metadata`、`slot_mapping`） |
| 预热完成时机 | 模型 `__init__` 内 | 必须等到真实推理阶段 |

qwen3-tts 的 `CUDAGraphDecoderWrapper.warmup()` 在 `__init__` 中即可完成——decoder 是一个纯 nn.Module，不依赖任何 vLLM 运行时上下文。

VoxCPM2 的 scaffold/residual 是 PagedAttention LLM，其 CUDA Graph 捕获需要 `ForwardContext`（`attn_metadata`、`slot_mapping` 等），这些元数据只在推理步骤中存在。因此必须在推理阶段——而非模型初始化阶段——完成预热。

## 3. 优化方案

### 核心思路

在服务启动阶段（对外 ready 之前）发送一个合成语音请求，走完整推理管线，将所有"冷启动"开销从首次用户请求转移到启动阶段。

### 具体实现

**serving 层**（`serving_speech.py`）：新增 `warmup()` 方法

```python
async def warmup(self) -> None:
    """Run a synthetic speech request to trigger all first-request warmup."""
    if self._tts_model_type != "voxcpm2":
        return

    warmup_req = OpenAICreateSpeechRequest(
        input="Warmup.",
        voice="default",
        response_format="wav",
        speed=1.0,
        stream=False,
        model=self.model_name,
    )
    _audio_bytes, _media_type = await self._generate_audio_bytes(
        warmup_req, request_id="speech-warmup"
    )
```

**API server 层**（`api_server.py`）：在 `omni_init_app_state` 中，speech serving 初始化完成后立即调用：

```python
state.openai_serving_speech = OmniOpenAIServingSpeech(...)
await state.openai_serving_speech.warmup()  # ← 预热请求
```

### 预热请求触发的完整链路

```
serving-layer warmup 请求
  → preprocess: _get_multichar_zh_split()（CJK 分词表）
  → _finish_prefill: _setup_cfm_buffers() + _setup_torch_compile()
      → torch.compile(LocDiT, feat_encoder, AudioVAE, projections)
      → precompute_fused_qkv(scaffold, residual)
  → 首轮 decode: _finish_decode（compiled projections 首次调用）
  → _collect_audio: audio_vae.decode（compiled VAE 首次调用）
  → 后续 decode: _capture_graph(scaffold, bs=1) + _capture_graph(residual, bs=1)
```

所有 torch.compile JIT、CUDA Graph capture 在此阶段完成。

### 为什么不在模型 `__init__` 中预热

尝试过在 `__init__` 中运行 dummy 数据预热 TTS 函数（LocDiT、feat_encoder、VAE），但遇到两个问题：

1. **QKV 融合时序错误**：`_setup_torch_compile` 中的 `precompute_fused_qkv()` 会在 scaffold 模型上融合 QKV 权重，但 `__init__` 时 vLLM 尚未调用 `load_weights()`，scaffold 权重为随机值。融合后的 `_fused_qkv_weight` 被持久化，导致后续推理使用错误权重，输出噪音。

2. **CUDA Graph 无法在 `__init__` 中捕获**：scaffold/residual 的 CUDA Graph 需要 `ForwardContext`，只能在真实推理步骤中获取。

最终方案删除了所有模型层预热代码，仅保留 serving 层的一个合成请求。

### 涉及的代码变动

| 文件 | 行数 | 说明 |
|------|------|------|
| `entrypoints/openai/serving_speech.py` | +29 | 新增 `warmup()` 方法 |
| `entrypoints/openai/api_server.py` | +2 | 启动时调用 `await warmup()` |


## 4. 优化效果

### 耗时对比

| 阶段 | 优化前 | 优化后 |
|------|--------|--------|
| CJK 分词表构建 | 首次用户请求 ~3s | 服务启动阶段 |
| `torch.compile` JIT 首次执行 | 首次用户请求（含在 ~3s 内） | 服务启动阶段 |
| CUDA Graph capture | 首次用户请求 ~24s | 服务启动阶段 |
| **首次用户请求总耗时** | **~28s** | **~2.0s** |
| **后续用户请求耗时** | ~1.8s | ~1.8s |

```
# startup_timecost.txt 修复后首次请求
00:28:04  TTS speech request received
00:28:06  Response sent (200 OK)  ← 耗时 ~2.0s（无冷启动开销）
```

### 额外启动时间

服务启动阶段增加约 **28s**（预热请求耗时），发生在 `Application startup complete` 日志之后、服务对外 ready 之前。对用户无感知。

### 补充说明

预热请求使用 `batch_size=1`，仅捕获该 batch 的 CUDA Graph。若后续出现 batch_size=2 的并发请求，首次会额外触发 CUDA Graph 录制（无 torch.compile JIT 开销）。可通过配置 `max_num_seqs` 控制最大 batch。