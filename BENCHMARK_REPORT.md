# Gemma 4 26B-A4B — Detailed Benchmark Report

**Date**: April 2026  
**Model**: `google/gemma-4-26B-A4B-it` (26B parameters, Mixture-of-Experts, ~4B active)  
**Serving Engine**: vLLM 0.19.0  
**Benchmark Tool**: `vllm bench serve` (GPU/TPU), custom Python script (Vertex AI)  
**Workload**: ~20K random input tokens → 250 output tokens (low cache hit rate)  
**Customer Target**: ~3.5s P90 E2E latency  

---

## Table of Contents

1. [Platform Specifications](#1-platform-specifications)
2. [GPU Benchmark Results (RTX Pro 6000)](#2-gpu-benchmark-results)
3. [TPU Benchmark Results (v6e-8 Trillium)](#3-tpu-benchmark-results)
4. [Vertex AI MaaS Results](#4-vertex-ai-maas-results)
5. [3-Way Comparison](#5-3-way-comparison)
6. [Scaling Analysis](#6-scaling-analysis)
7. [Conclusions](#7-conclusions)
8. [Caveats & Limitations](#8-caveats--limitations)

---

## 1. Platform Specifications

### GPU: NVIDIA RTX Pro 6000 (Blackwell)

| Spec | Value |
|------|-------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| VRAM | 96 GB GDDR7 (vLLM reports 97.9 GiB usable) |
| Compute Capability | 12.0 (sm_120, Blackwell workstation) |
| CUDA / Driver | 13.0 / 580.126.09 |
| Quantization | FP8 |
| Model weights | 25.67 GiB |
| KV cache | 53.86 GiB (235,296 tokens) |
| GPU utilization | 91.4% (89.4 / 97.9 GiB) |
| On-demand pricing | $4.50/hr |

### TPU: Cloud TPU v6e-8 (Trillium)

| Spec | Value |
|------|-------|
| Accelerator | 8× Trillium chips |
| Memory | 256 GB HBM total (32 GB/chip) |
| Precision | BF16 (no quantization) |
| Tensor Parallelism | 8 |
| Max Model Length | 32,768 tokens |
| KV cache blocks | 8,888 per layer (30 layers) |
| Docker image | `vllm/vllm-tpu:gemma4` |
| On-demand pricing | $21.60/hr ($2.70/chip × 8) |

### Vertex AI MaaS: Model-as-a-Service

| Spec | Value |
|------|-------|
| Model | `google/gemma-4-26b-a4b-it-maas` |
| API | `aiplatform.googleapis.com` (generateContent) |
| Endpoint | Global (no deployment needed) |
| Streaming | True per-token streaming |
| Input pricing | $0.15 / million tokens |
| Output pricing | $0.60 / million tokens |

---

## 2. GPU Benchmark Results

### vLLM Configuration

```bash
vllm serve google/gemma-4-26B-A4B-it \
    --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 --quantization fp8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 128000 \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 8 \
    --enable-chunked-prefill \
    --enable-prefix-caching
```

### Single Request Baseline

```
vllm bench serve --dataset-name random --random-input-len 20000 --random-output-len 250 --num-prompts 1
```

| Metric | Value |
|--------|-------|
| TTFT | 1,009ms |
| TPOT | 9.27ms |
| ITL median | 9.37ms |
| Output throughput | 75.4 tok/s |
| E2E latency | ~3.32s |
| **Within 3.5s target** | ✅ Yes |

### QPS Sweep (10 prompts per rate, seed=42)

```
vllm bench serve --dataset-name random --random-input-len 20000 --random-output-len 250 --num-prompts 10 --request-rate $QPS
```

| Target QPS | Achieved QPS | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Peak Conc | Est. E2E (s) |
|-----------|-------------|-------------|----------------|----------------|-----------|-------------|
| 0.10 | 0.10 | 24.4 | 120.5 | 10.20 | 3 | 2.66 |
| 0.15 | 0.14 | 36.2 | 91.4 | 10.27 | 4 | 2.65 |
| 0.20 | 0.19 | 47.7 | 89.6 | 10.41 | 4 | 2.69 |
| 0.25 | 0.24 | 59.0 | 88.8 | 10.34 | 4 | 2.66 |
| 0.30 | 0.28 | 70.0 | 88.4 | 10.46 | 4 | 2.70 |
| 0.40 | 0.37 | 91.3 | 90.7 | 10.50 | 4 | 2.71 |
| 0.50 | 0.45 | 111.7 | 91.8 | 10.62 | 5 | 2.74 |
| 0.70 | 0.60 | 148.8 | 94.0 | 11.16 | 5 | 2.87 |
| 1.00 | 0.78 | 195.9 | 95.1 | 12.17 | 6 | 3.13 |

> Note: Low TTFT (~90ms) at sustained QPS is due to prefix caching warming up. First requests see ~1s TTFT.

### Burst Sweep (all requests at once, seed=42)

```
vllm bench serve --dataset-name random --random-input-len 20000 --random-output-len 250 --num-prompts $N --request-rate inf
```

| N | Achieved QPS | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Peak Conc |
|---|-------------|-------------|----------------|----------------|-----------|
| 1 | 0.42 | 105.4 | 85.9 | 9.18 | 1 |
| 2 | 0.71 | 177.1 | 141.2 | 10.77 | 2 |
| 5 | 1.31 | 326.4 | 540.1 | 13.04 | 5 |
| 8 | 2.08 | 519.2 | 408.9 | 13.62 | 8 |
| 10 | 1.49 | 373.4 | 1,139.1 | 13.38 | 10 |
| 15 | 1.18 | 295.6 | 4,083.0 | 17.32 | 15 |
| 20 | 1.30 | 323.9 | 4,650.8 | 15.23 | 20 |
| 30 | 1.23 | 308.0 | 8,879.5 | 16.03 | 30 |

**Key observations:**
- Peak throughput at N=8 (519 tok/s) = `max_num_seqs`
- TTFT explodes beyond N=8 due to request queuing
- TPOT degrades ~75% from N=1 to N=30 (9ms → 16ms)

---

## 3. TPU Benchmark Results (v6e-8 Trillium)

### vLLM Configuration

```bash
# Docker: vllm/vllm-tpu:gemma4
# VLLM_ARGS:
--model google/gemma-4-26B-A4B-it \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --disable_chunked_mm_input
```

### Single Request Baseline

| Metric | Value |
|--------|-------|
| TTFT | 821ms (87ms steady-state¹) |
| TPOT | 8.60ms |
| ITL median | 8.69ms |
| Output throughput | 84.42 tok/s |
| Total throughput | 6,838 tok/s |
| E2E latency | ~2.97s |
| **Within 3.5s target** | ✅ Yes |

> ¹ First request includes XLA compilation overhead. Subsequent requests see ~87ms TTFT.

### QPS Sweep (10 prompts per rate, seed=42)

| Target QPS | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Mean ITL (ms) |
|-----------|----------------|-------------------|---------------|----------------|---------------|
| 0.10 | 614.66 | 672.50 | 683.55 | 9.59 | 9.59 |
| 0.15 | 94.48 | 94.86 | 100.20 | 8.85 | 8.85 |
| 0.20 | 92.32 | 93.48 | 99.24 | 8.71 | 8.71 |
| 0.25 | 93.18 | 94.22 | 101.98 | 8.71 | 8.71 |
| 0.30 | 92.80 | 93.54 | 98.33 | 8.79 | 8.79 |
| 0.40 | 93.83 | 96.26 | 100.37 | 8.98 | 8.98 |
| 0.50 | 92.00 | 93.45 | 99.59 | 8.87 | 8.87 |
| 0.70 | 92.11 | 92.73 | 100.22 | 8.67 | 8.67 |
| 1.00 | 91.78 | 91.96 | 99.34 | 8.79 | 8.79 |

> QPS=0.1 elevated TTFT (615ms) includes first-request XLA compilation. Steady-state TTFT is ~92ms.

### Burst Sweep (all requests at once, seed=42)

| N | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Mean ITL (ms) |
|---|----------------|-------------------|---------------|----------------|---------------|
| 1 | 86.85 | 86.85 | 86.85 | 8.89 | 8.89 |
| 2 | 148.43 | 148.43 | 155.21 | 8.92 | 8.95 |
| 5 | 298.10 | 347.99 | 349.69 | 8.79 | 8.97 |
| 8 | 477.17 | 529.30 | 532.86 | 8.77 | 9.10 |
| 10 | 554.42 | 649.08 | 652.53 | 9.09 | 9.49 |
| 15 | 1,507.77 | 1,389.78 | 3,739.50 | 17.95 | 17.95 |
| 20 | 1,685.61 | 1,210.18 | 4,265.61 | 26.59 | 26.65 |

**Key observations:**
- **Steady-state TTFT ~92ms** — comparable to GPU's ~88ms under QPS load (both benefit from warm state: GPU from prefix caching, TPU from XLA compilation)
- **Burst N=10 TTFT=554ms** vs GPU's 1,139ms — 2x faster prefill under concurrent load
- **Burst N=20 TTFT=1,686ms** vs GPU's 4,651ms — **2.8x faster** under heavy burst load
- TPOT stays flat at ~8.8ms up to N=10, then degrades at N=15+ (KV cache pressure)
- 256 GB HBM enables handling many more concurrent prefills without queuing

---

## 4. Vertex AI MaaS Results

### Configuration

- Model: `google/gemma-4-26b-a4b-it-maas` via `aiplatform.googleapis.com`
- API: Vertex AI `generateContent` (streaming mode)
- No deployment needed — global endpoint, fully managed

### Single Request Baseline

| Metric | Value |
|--------|-------|
| TTFT | 491ms |
| E2E latency | 0.80s |
| **Within 3.5s target** | ✅ Yes |

### QPS Sweep (10 requests each)

| QPS | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean Latency (s) | Req Throughput |
|-----|----------------|-------------------|---------------|-------------------|----------------|
| 0.10 | 1,222 | 1,167 | 1,438 | 1.50 | 0.109 req/s |
| 0.15 | 1,000 | 1,150 | 1,300 | 1.26 | 0.163 req/s |
| 0.20 | 916 | 1,156 | 1,213 | 1.20 | 0.215 req/s |
| 0.25 | 727 | 576 | 1,253 | 1.04 | 0.267 req/s |
| 0.30 | 680 | 460 | 1,250 | 1.00 | 0.324 req/s |
| 0.40 | 648 | 512 | 1,157 | 0.95 | 0.423 req/s |
| 0.50 | 559 | 518 | 956 | 0.86 | 0.532 req/s |
| 0.70 | 681 | 615 | 1,174 | 0.97 | 0.730 req/s |
| 1.00 | 605 | 534 | 1,091 | 0.91 | 0.995 req/s |

### Burst Sweep (all requests simultaneous)

| N | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean Latency (s) | Req Throughput |
|---|----------------|-------------------|---------------|-------------------|----------------|
| 1 | 491 | 491 | 491 | 0.80 | 1.25 req/s |
| 2 | 647 | 647 | 648 | 0.81 | 2.46 req/s |
| 5 | 769 | 639 | 1,282 | 1.08 | 3.12 req/s |
| 8 | 1,086 | 1,047 | 1,263 | 1.44 | 4.56 req/s |
| 10 | 1,234 | 1,244 | 1,284 | 1.58 | 6.15 req/s |
| 15 | 2,429 | 2,399 | 2,693 | 2.76 | 4.91 req/s |
| 20 | 3,091 | 3,062 | 3,580 | 3.42 | 4.99 req/s |
| 30 | 4,327 | 4,328 | 4,517 | 4.70 | 6.04 req/s |

**Key observations:**
- Lowest single-request E2E of all platforms (0.80s)
- TTFT improves with sustained QPS (680ms → 559ms at 0.5 QPS) — likely warmed backend
- Burst throughput peaks at N=10 (6.15 req/s), then fluctuates
- Only platform to meet 3.5s E2E target at N=20 burst (3.42s)
- No per-token TPOT available (managed API)

---

## 5. 3-Way Comparison

![Comparison Chart](plots/09_gpu_tpu_vertexai_comparison.png)

![Comparison Table](plots/10_gpu_tpu_vertexai_table.png)

### Head-to-Head: E2E Latency (Customer's Primary Metric)

All E2E values calculated as: `TTFT + (250-1) × TPOT`. Values sourced directly from data tables in Sections 2-4.

| Scenario | GPU (RTX Pro 6000) | TPU v6e-8 | MaaS | Winner |
|----------|-------------------|-----------|------|--------|
| **Single E2E** | 3.32s | 2.97s | **0.80s** | **MaaS** |
| **0.3 QPS E2E (cold)** | ⚠️ 3.61s¹ | **2.28s** | **1.00s** | **MaaS** |
| **Burst N=10 E2E** | 4.47s | **2.82s** | 1.58s | **MaaS** |
| **Burst N=20 E2E** | 8.44s | 8.31s | **3.42s** | **MaaS** |

> ¹ GPU 0.3 QPS E2E uses cold TTFT (1,009ms) + TPOT=10.46ms = 3.61s. The benchmark measured 2.70s using prefix-cached TTFT (88ms), but customer has "low cache hit rate" making cached values unrealistic for production.

### Head-to-Head: Latency Components

| Metric | GPU | TPU v6e-8 | MaaS |
|--------|-----|-----------|------|
| **Single TTFT** | 1,009ms | 821ms (87ms steady) | 491ms |
| **Single TPOT** | 9.27ms | 8.60ms | N/A† |
| **0.3 QPS TTFT (cold)** | ⚠️ 1,009ms¹ | 93ms | 680ms |
| **0.3 QPS TPOT** | 10.46ms | 8.79ms | N/A† |
| **Burst N=20 TTFT** | 4,651ms | 1,686ms | 3,091ms |
| **Burst N=20 TPOT** | 15.23ms | 26.59ms | N/A† |
| **Burst N=20 Output tok/s** | 323.9 | N/A² | ~4.99 req/s |
| **On-demand Cost** | $4.50/hr | $21.60/hr | Managed |

> ¹ GPU QPS sweep measured TTFT=88ms due to prefix caching with `seed=42`. Production cold TTFT = 1,009ms (used here). Customer states "low cache hit rate".  
> ² TPU burst output tok/s not directly measured by `vllm bench serve` in this configuration  
> † Managed APIs: per-token timing (TPOT) not measurable

---

## 6. Scaling Analysis

### Horizontal Scaling to 100 QPS

Assuming serial execution (1 request per instance at a time):

| Platform | Single E2E | Instances for 100 QPS | Hourly Cost |
|----------|-----------|----------------------|-------------|
| GPU | 3.32s | 334 | $1,503/hr |
| TPU v6e-8 | 2.97s | 297 | $6,415/hr |

> Note: TPU v6e-8 handles much higher concurrent load (N=10 at ~2.8s E2E), reducing required instances significantly for bursty workloads. Cost comparison assumes serial 1-req/instance — see Section 8 caveats.

### Cost per Million Tokens

| Platform | Cost/M output tokens | Notes |
|----------|---------------------|-------|
| **MaaS** | **$12.60** | $0.15/M input + $0.60/M output (at 20K:250 ratio = $0.00315/req)¹ |
| GPU | $16.58 | $4.50/hr ÷ 271,440 tok/hr (75.4 tok/s) |
| TPU v6e-8 | $71.07 | $21.60/hr ÷ 303,912 tok/hr (84.4 tok/s) |

> ¹ MaaS cost per request for this workload: 20,000 × $0.15/M + 250 × $0.60/M = $0.003 + $0.00015 = **$0.00315/request**. Cost/M output tokens = $0.00315 × (1M/250) = $12.60/M (includes input token cost).

> **MaaS is cheapest** per output token for this workload, even before accounting for GPU/TPU infrastructure overhead (VM management, idle time, scaling). TPU v6e-8 is 5.6x more expensive than MaaS per output token at single concurrency, though concurrent batching reduces effective cost.

---

## 7. Conclusions

### Platform Selection Guide (E2E Focused)

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **Lowest E2E (single request)** | MaaS | 0.80s E2E, zero infrastructure |
| **Lowest E2E under sustained load** | MaaS | 1.00s at 0.3 QPS; self-hosted: TPU 2.28s vs GPU 3.61s (cold) |
| **Best burst E2E (N=20)** | MaaS | 3.42s; self-hosted: GPU 8.44s ≈ TPU 8.31s |
| **Best burst TTFT (prefill)** | TPU v6e-8 | 1,686ms at N=20 vs GPU's 4,651ms (2.8x faster) |
| **Best decode (TPOT)** | TPU v6e-8 | 8.60ms single; but GPU better at high N (15.23ms vs 26.59ms at N=20) |
| **Lowest cost per output token** | MaaS | $12.60/M vs GPU $16.58/M vs TPU $71.07/M |
| **Lowest single-instance cost** | GPU | $4.50/hr vs $21.60/hr TPU |
| **Zero infrastructure** | MaaS | Fully managed, auto-scales |
| **True token streaming** | GPU or TPU | Per-token delivery supported |

### 3.5s Mean E2E Target Assessment

All E2E values calculated from `TTFT + (n-1) × TPOT` using data from Sections 2-4.

| Platform | Single Request | 0.3 QPS Sustained | Burst N=20 | Verdict |
|----------|---------------|-------------------|------------|---------|
| GPU (cold TTFT) | 3.32s ✅ | ⚠️ 3.61s ❌ | 8.44s ❌ | **Single ✅, sustained ❌, burst ❌** |
| TPU v6e-8 | 2.97s ✅ | **2.28s ✅** | 8.31s ❌ | **Sustained ✅, burst ❌** |
| MaaS | **0.80s ✅** | **1.00s ✅** | **3.42s ✅** | **All scenarios ✅** |

> **⚠️ Critical**: GPU 0.3 QPS uses cold TTFT (1,009ms) because the customer states "low cache hit rate". E2E = 1,009 + 249 × 10.46 ≈ **3.61s**, which **exceeds the 3.5s target**.

> **Important**: These are **mean** E2E estimates. The customer target is **P90 E2E**, which was not directly measured. P90 will be higher than mean. See Section 8 for details.

> Only TPU and MaaS meet the 3.5s target for sustained workloads. GPU fails at 0.3 QPS with cold TTFT (3.61s). MaaS is the only platform that meets 3.5s E2E across all scenarios.

---

## 8. Caveats & Limitations

### ⚠️ GPU 0.3 QPS E2E Is Misleading

The report shows 2.70s E2E at 0.3 QPS using prefix-cached TTFT (88ms), but the customer states they have a **"low cache hit rate"** in production. With cold TTFT (1,009ms), actual E2E ≈ **3.61s** (1,009 + 249 × 10.46ms), which **exceeds the 3.5s target**. The QPS sweep TTFT should **NOT** be used for target assessment — use single-request cold TTFT instead.

### ⚠️ P90 E2E Likely Fails Target

GPU single-request **mean** E2E = 3.32s. With typical latency variance, **P90 will exceed 3.5s**. The customer's target is **P90**, not mean. A mean of 3.32s with any meaningful standard deviation will produce a P90 above 3.5s.

### ⚠️ TPU max-model-len Mismatch

The TPU benchmark uses `--max-model-len 32768`, but the customer's production config requires **128,000 tokens**. A larger max context length may require upgrading to **v6e-16** or could cause **significant performance degradation** due to increased KV cache memory pressure on v6e-8.

### ⚠️ MaaS Uses Different Model Variant

The MaaS endpoint uses `gemma-4-26b-a4b-it-maas`, which **may differ** from the self-hosted `google/gemma-4-26B-A4B-it` model. Performance characteristics (latency, throughput, quality) may not be directly comparable across self-hosted and MaaS deployments.

### ⚠️ TPU Burst N=30 Data Missing

The raw TPU benchmark log was **truncated at N=20**. The N=30 burst data point was **not captured** and is absent from the TPU results. GPU N=30 data is available for reference but cannot be compared against TPU at that concurrency level.

### P90 E2E Not Measured

The customer's target metric is **P90 E2E latency**. This benchmark reports **mean** E2E latency, estimated via formula `TTFT + (n-1) × TPOT`. P90 E2E was not directly captured by the benchmarking tools. **Recommendation**: Re-run benchmarks with explicit P90 E2E tracking before making capacity decisions.

### ⚠️ Customer vLLM Flags Not Used

The customer's production config includes several flags that were **not used** in this GPU benchmark. **These omissions may significantly affect real-world performance:**

| Customer Flag | Used in Benchmark? | Potential Impact |
|--------------|-------------------|-----------------|
| `--performance-mode balanced` | ❌ No | May affect latency/throughput tradeoff |
| `--kv-sharing-fast-prefill` | ❌ No | May affect prefill speed |
| `--enable-auto-tool-choice` | ❌ No | Adds tool-call overhead |
| `--tool-call-parser gemma4` | ❌ No | Affects token processing |
| `--reasoning-parser gemma4` | ❌ No | Affects token processing |

**⚠️ Recommendation**: Re-benchmark with customer's exact flags before finalizing. Results without these flags may be **overly optimistic** if the flags add processing overhead, or **overly pessimistic** if flags like `--kv-sharing-fast-prefill` improve performance.

### Raw Data Files

| Platform | Raw Log File | Status |
|----------|-------------|--------|
| GPU (RTX Pro 6000) | `data/gpu-benchmark-results.txt` | ✅ Available |
| TPU v6e-8 | `data/tpu-benchmark-results.txt` | ✅ Available |
| Vertex AI MaaS | `data/maas-benchmark-results.txt` | ✅ Available |

### Prefix Caching Effect on GPU QPS Sweep

The GPU QPS sweep uses `seed=42`, generating the same random input for all requests. Combined with `--enable-prefix-caching`, subsequent requests benefit from cached prefills, resulting in artificially low TTFT (~88ms vs 1,009ms cold). Production workloads with diverse inputs would see higher TTFT.

### Scaling Analysis Simplification

Section 6 assumes 1 request per instance (serial execution). In practice, GPU can batch up to `max_num_seqs=8` concurrent requests, and TPU handles even more. Actual instance requirements would be significantly lower than shown.
