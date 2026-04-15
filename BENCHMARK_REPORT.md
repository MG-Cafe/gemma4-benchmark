# Gemma 4 26B-A4B — Detailed Benchmark Report

**Date**: April 2026  
**Model**: `google/gemma-4-26B-A4B-it` (26B parameters, Mixture-of-Experts, ~4B active)  
**Serving Engine**: vLLM 0.19.0  
**Benchmark Tool**: custom Python script (GPU/TPU), custom Python script (Vertex AI)  
**Workload**: ~20K random input tokens → 250 output tokens (unique random prompts per request)  
**Customer Target**: ~3.5s P90 E2E latency  

---

## Table of Contents

1. [Platform Specifications](#1-platform-specifications)
2. [GPU Benchmark Results (4× RTX Pro 6000, TP=4)](#2-gpu-benchmark-results)
3. [TPU Benchmark Results (v6e-4 Trillium)](#3-tpu-benchmark-results)
4. [Vertex AI MaaS Results](#4-vertex-ai-maas-results)
5. [3-Way Comparison](#5-3-way-comparison)
6. [Scaling Analysis](#6-scaling-analysis)
7. [Conclusions](#7-conclusions)
8. [Caveats & Limitations](#8-caveats--limitations)

---

## 1. Platform Specifications

### GPU: 4× NVIDIA RTX Pro 6000 (Blackwell, TP=4)

| Spec | Value |
|------|-------|
| GPU | 4× NVIDIA RTX PRO 6000 Blackwell Server Edition |
| VRAM | 384 GB GDDR7 total (96 GB × 4) |
| Machine Type | g4-standard-192 (4 GPUs, 192 vCPUs, 768GB RAM) |
| Compute Capability | 12.0 (sm_120, Blackwell) |
| CUDA / Driver | 13.0 / 580.126.09 |
| Quantization | FP8 |
| Tensor Parallelism | 4 |
| Model weights | ~6.4 GiB per GPU (25.67 GiB / 4) |
| KV cache | ~73 GiB per GPU |
| GPU utilization | ~83% per GPU |
| On-demand pricing | $18.00/hr ($4.50/GPU × 4) |

### TPU: Cloud TPU v6e-4 (Trillium)

| Spec | Value |
|------|-------|
| Accelerator | 4× Trillium chips |
| Memory | ~128 GB HBM total (32 GB/chip) |
| Precision | BF16 (no quantization) |
| Tensor Parallelism | 4 |
| Max Model Length | 128,000 tokens |
| Flash Attention | VLLM_FLASH_ATTN=1 |
| Docker image | `vllm/vllm-tpu:gemma4` |
| On-demand pricing | $10.80/hr ($2.70/chip × 4) |

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

## 2. GPU Benchmark Results (4× RTX Pro 6000, TP=4)

### vLLM Configuration

```bash
vllm serve google/gemma-4-26B-A4B-it \
    --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 --quantization fp8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 128000 \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 32 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --tensor-parallel-size 4 \
    --trust-remote-code
```

### Single Request Baseline (P90 from 10 cold runs)

| Metric | Value |
|--------|-------|
| TTFT | 1,108ms |
| TPOT | 8.83ms |
| Output throughput | 76.7 tok/s |
| E2E (mean) | 3.31s |
| E2E (P90) | **3.31s** |
| **Within 3.5s target** | ✅ Yes |

> TP=4 parallelizes the 20K-token prefill across 4 GPUs, achieving 1,108ms TTFT. Single-request P90 E2E meets the 3.5s target.

### QPS Sweep (10 prompts per rate, 10 fresh runs each — P90 E2E)

| Target QPS | P90 E2E (s) | Mean TTFT (ms) | Mean TPOT (ms) | Output tok/s |
|-----------|-------------|----------------|----------------|-------------|
| 0.10 | 3.79 | 967 | 9.90 | 24.2 |
| 0.15 | 4.11 | 1,033 | 10.64 | 35.8 |
| 0.20 | 4.54 | 1,047 | 11.79 | 47.0 |
| 0.25 | 4.67 | 1,067 | 12.65 | 57.9 |
| 0.30 | 5.04 | 1,106 | 13.66 | 68.4 |
| 0.40 | 5.45 | 1,210 | 14.87 | 88.4 |
| 0.50 | 6.44 | 1,340 | 17.69 | 105.8 |
| 0.70 | 7.67 | 1,565 | 20.68 | 138.7 |
| 1.00 | 8.79 | 2,067 | 25.14 | 170.6 |

> TPOT degrades under QPS load (9.9→25.1ms) because TP=4 with max_num_seqs=32 processes more concurrent requests, increasing per-token decode time.

### Burst Sweep (all requests at once, 10 fresh runs each — P90 E2E)

| N | P90 E2E (s) | Mean TTFT (ms) | Mean TPOT (ms) | Output tok/s |
|---|-------------|----------------|----------------|-------------|
| 1 | 3.16 | 536 | 8.69 | 95.4 |
| 2 | 3.48 | 592 | 9.73 | 169.7 |
| 5 | 5.57 | 1,797 | 11.91 | 283.3 |
| 8 | 5.68 | 1,972 | 14.75 | 351.0 |
| 10 | 5.11 | 1,589 | 13.84 | 489.7 |
| 15 | 16.65 | 3,763 | 24.89 | 405.0 |
| 20 | 21.65 | 10,851 | 43.23 | 230.9 |
| 30 | 31.28 | 15,542 | 62.91 | 239.9 |

**Key observations:**
- **Peak throughput at N=10** (489.7 tok/s)
- **N=1-2 within 3.5s target** (3.16s and 3.48s P90 E2E)
- **TTFT stays low through N=2** (592ms) but explodes at N=15+ (3.8s→15.5s)
- **TPOT degrades severely**: 8.69ms (N=1) → 62.91ms (N=30) due to memory pressure at high concurrency
- The 4-GPU setup processes all requests simultaneously (max_num_seqs=32), creating decode contention at high N

---

## 3. TPU Benchmark Results (v6e-4 Trillium)

### vLLM Configuration

```bash
# Docker: vllm/vllm-tpu:gemma4
# Environment: VLLM_FLASH_ATTN=1
# VLLM_ARGS:
--model google/gemma-4-26B-A4B-it \
    --max-model-len 128000 \
    --tensor-parallel-size 4 \
    --disable_chunked_mm_input
```

### ⚠️ Methodology: Unique Random Prompts (No Prefix Caching)

Each request uses a **freshly generated unique random prompt** (~20K tokens) to prevent any prefix caching benefits. This ensures TTFT reflects real prefill computation cost, not cached lookups.

### Single Request Baseline (10 runs, unique random prompts)

| Metric | Value |
|--------|-------|
| TTFT | 706ms |
| TPOT | 7.23ms |
| Output throughput | 138.3 tok/s |
| E2E (mean) | 2.09s |
| E2E (P90) | **2.52s** |
| **Within 3.5s target** | ✅ Yes (2.52s) |

### QPS Sweep (10 unique random prompts per rate)

| Target QPS | Mean TTFT (ms) | P90 TTFT (ms) | Mean TPOT (ms) | P90 E2E (s) |
|-----------|----------------|---------------|----------------|-------------|
| 0.10 | 707 | 708 | 7.27 | 2.53 |
| 0.15 | 706 | 707 | 7.27 | 2.50 |
| 0.20 | 706 | 707 | 7.25 | 2.52 |
| 0.25 | 707 | 707 | 7.26 | 2.53 |
| 0.30 | 706 | 707 | 7.25 | 2.52 |
| 0.40 | 706 | 707 | 7.27 | 2.53 |
| 0.50 | 712 | 718 | 8.87 | 3.15 |
| 0.70 | 715 | 719 | 10.95 | 3.76 |
| 1.00 | 725 | 733 | 16.76 | 5.15 |

### Burst Sweep (all requests at once, unique random prompts)

| N | Mean TTFT (ms) | P90 TTFT (ms) | Mean TPOT (ms) | P90 E2E (s) |
|---|----------------|---------------|----------------|-------------|
| 1 | 704 | 704 | 7.20 | 2.50 |
| 2 | 1,066 | 1,320 | 8.69 | 3.26 |
| 5 | 2,016 | 3,014 | 14.36 | 5.18 |
| 8 | 2,920 | 4,693 | 18.73 | 7.36 |
| 10 | 3,603 | 5,915 | 22.88 | 8.64 |
| 15 | 5,733 | 10,334 | 25.15 | 13.22 |
| 20 | 7,831 | 13,432 | 27.34 | 16.74 |

**Key observations:**
- **TTFT ~706ms baseline** — real prefill cost for ~20K unique random tokens on 4 Trillium chips
- **TPOT stays excellent at low concurrency**: 7.2ms (N=1) → 8.7ms (N=2), competitive with GPU
- **TPOT degrades at high concurrency**: 14.4ms (N=5) → 27.3ms (N=20)
- **Single request and low QPS (≤0.4) meet 3.5s target** — P90 E2E ~2.52s
- **Burst N≥5 exceeds 3.5s target** — prefill queuing on 4 chips causes TTFT growth (N=2 at 3.26s still within target)
- **TPU wins single-request TTFT** vs GPU (706ms vs 1,108ms) — faster prefill per chip

---

## 4. Vertex AI MaaS Results

### Configuration

- Model: `google/gemma-4-26b-a4b-it-maas` via `aiplatform.googleapis.com`
- API: Vertex AI `generateContent` (streaming mode)
- No deployment needed — global endpoint, fully managed

### Single Request Baseline (10 cold runs)

| Metric | Value |
|--------|-------|
| Mean TTFT | 1,330ms |
| P90 TTFT | 1,525ms |
| Mean E2E | 2.94s |
| P90 E2E | **3.09s** |
| **Within 3.5s target** | ✅ Yes |

### QPS Sweep (10 requests each, P90 E2E)

| QPS | Mean TTFT (ms) | P90 TTFT (ms) | Mean E2E (s) | P90 E2E (s) |
|-----|----------------|---------------|--------------|-------------|
| 0.10 | 1,275 | 1,348 | 2.90 | 2.94 |
| 0.15 | 1,397 | 1,527 | 3.02 | 3.14 |
| 0.20 | 1,311 | 1,380 | 2.94 | 3.05 |
| 0.25 | 1,284 | 1,345 | 2.95 | 3.06 |
| 0.30 | 1,292 | 1,447 | 2.91 | 3.08 |
| 0.40 | 1,335 | 1,440 | 3.01 | 3.09 |
| 0.50 | 1,284 | 1,399 | 2.93 | 3.05 |
| 0.70 | 1,277 | 1,379 | 3.27 | 3.80 |
| 1.00 | 1,303 | 1,568 | 3.37 | 3.61 |

### Burst Sweep (all requests simultaneous, P90 E2E)

| N | Mean TTFT (ms) | P90 TTFT (ms) | Mean E2E (s) | P90 E2E (s) |
|---|----------------|---------------|--------------|-------------|
| 1 | 1,484 | — | 3.81 | — |
| 2 | 1,533 | 1,533 | 3.64 | 3.79 |
| 5 | 1,655 | 2,029 | 3.93 | 4.20 |
| 8 | 1,836 | 2,116 | 4.45 | 4.95 |
| 10 | 2,392 | 2,824 | 5.12 | 6.11 |
| 15 | 3,274 | 3,703 | 6.17 | 7.30 |
| 20 | 4,201 | 4,964 | 7.42 | 8.22 |
| 30 | 5,594 | 6,377 | 9.06 | 9.90 |

**Key observations:**
- Competitive single-request P90 E2E (3.09s) — second only to TPU v6e-4 (2.52s)
- MaaS stays remarkably flat under QPS sweep: P90 E2E 2.94-3.09s from 0.1-0.5 QPS
- Burst P90 E2E scales linearly and predictably (3.79s at N=2 → 9.90s at N=30)
- Both TPU and MaaS meet 3.5s P90 at sustained QPS up to 0.5; MaaS offers zero-infrastructure advantage

---

## 5. 3-Way Comparison

![Comparison Chart](plots/09_gpu_tpu_vertexai_comparison.png)

![Comparison Table](plots/10_gpu_tpu_vertexai_table.png)

### Head-to-Head: P90 E2E Latency (Customer's Primary Metric)

| Scenario | GPU (4×RTX TP=4) | TPU v6e-4 | MaaS | Winner |
|----------|------------------|-----------|------|--------|
| **Single request** | 3.31s | **2.52s** ✅ | 3.09s ✅ | **TPU** |
| **0.3 QPS steady** | 5.04s | **2.52s** ✅ | 3.08s ✅ | **TPU** |
| **Burst N=10** | 5.11s | 8.64s | **6.11s** | **MaaS** |
| **Burst N=20** | 21.65s | 16.74s | **8.22s** | **MaaS** |

### Head-to-Head: Latency Components

| Metric | GPU (4×RTX TP=4) | TPU v6e-4 | MaaS |
|--------|------------------|-----------|------|
| **Single TTFT** | 1,108ms | **706ms** | 1,330ms |
| **Single TPOT** | 8.83ms | **7.23ms** | 6.34ms |
| **0.3 QPS TTFT** | 1,106ms | **706ms** | 1,292ms |
| **0.3 QPS TPOT** | 13.66ms | **7.25ms** | N/A† |
| **Burst N=20 TTFT** | 10,851ms | 7,831ms | **4,201ms** |
| **Burst N=20 TPOT** | 43.23ms | **27.34ms** | N/A† |
| **Peak throughput** | **489.7 tok/s** (N=10) | ~138.3 tok/s | ~3.3 req/s |
| **On-demand Cost** | $18.00/hr | **$10.80/hr** | Pay-per-token |

> † Managed APIs: per-token timing (TPOT) not measurable

---

## 6. Scaling Analysis

### Cost per Million Output Tokens

| Platform | Cost/M output tokens | Notes |
|----------|---------------------|-------|
| **MaaS** | **$12.60** | $0.15/M input + $0.60/M output (at 20K:250 ratio) |
| TPU v6e-4 | $21.69 | $10.80/hr ÷ 497,880 tok/hr (138.3 tok/s) |
| GPU 4×RTX TP=4 | $65.32 | $18.00/hr ÷ 275,612 tok/hr (76.7 tok/s) |

> MaaS is cheapest per output token. TPU v6e-4 beats GPU on cost-per-token ($21.69 vs $65.32) due to both lower hourly rate and higher single-request throughput (138.3 vs 76.7 tok/s).

---

## 7. Conclusions

### Platform Selection Guide

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **Lowest single-request latency** | TPU v6e-4 | 2.52s P90 E2E — best of all platforms |
| **Best P90 under low QPS (≤0.4)** | TPU v6e-4 | 2.52s at 0.3 QPS — beats MaaS (3.08s) and GPU (5.04s) |
| **Best single-request TTFT** | TPU v6e-4 | 706ms (unique random prompts, no caching) vs GPU 1,108ms |
| **Best burst performance (N≥5)** | MaaS | 4.20s at N=5, 8.22s at N=20 — auto-scaling handles burst |
| **Lowest cost per output token** | MaaS | $12.60/M vs TPU $21.69/M vs GPU $65.32/M |
| **Lowest hourly cost** | TPU v6e-4 | $10.80/hr vs GPU $18.00/hr |
| **Zero infrastructure** | MaaS | Fully managed, auto-scales |

### 3.5s P90 E2E Target Assessment

| Platform | Single Request | 0.3 QPS Sustained | Burst N=20 | Verdict |
|----------|---------------|-------------------|------------|---------|
| GPU (4×RTX TP=4) | 3.31s ✅ | 5.04s ❌ | 21.65s ❌ | **Single ✅, sustained ❌, burst ❌** |
| TPU v6e-4 | **2.52s ✅** | **2.52s ✅** | 16.74s ❌ | **Single ✅, sustained ✅, burst ❌** |
| MaaS | **3.09s ✅** | **3.08s ✅** | 8.22s ❌ | **Single ✅, sustained ✅, burst ❌** |

> **No platform meets 3.5s P90 E2E at burst N=20.** For single request and sustained low QPS, TPU v6e-4 delivers the best latency (2.52s). MaaS is a close second (3.08-3.09s) with the advantage of zero infrastructure and auto-scaling for burst. GPU meets the target only for single requests.

---

## 8. Caveats & Limitations

### ⚠️ TPU Platform: v6e-4 (Not v6e-8)

This benchmark uses a **v6e-4** (4 Trillium chips, ~128GB HBM), not a v6e-8. A v6e-8 (8 chips, 256GB HBM) would likely show better burst performance due to doubled memory and compute. Results should not be extrapolated to v6e-8.

### ⚠️ Unique Random Prompts (No Prefix Caching)

All TPU benchmark requests use **unique random prompts** (~20K tokens each) to prevent prefix caching. This gives realistic TTFT numbers (706ms) reflecting actual prefill computation. With prefix caching enabled and repeated prompts, TTFT would be significantly lower.

### ⚠️ MaaS Uses Different Model Variant

The MaaS endpoint uses `gemma-4-26b-a4b-it-maas`, which **may differ** from the self-hosted `google/gemma-4-26B-A4B-it` model. Performance characteristics may not be directly comparable.

### ⚠️ Customer vLLM Flags Not Used

The customer's production config includes flags not used in this benchmark:

| Customer Flag | Used? | Potential Impact |
|--------------|-------|-----------------|
| `--performance-mode balanced` | ❌ | May affect latency/throughput tradeoff |
| `--kv-sharing-fast-prefill` | ❌ | May affect prefill speed |
| `--enable-auto-tool-choice` | ❌ | Adds tool-call overhead |
| `--tool-call-parser gemma4` | ❌ | Affects token processing |
| `--reasoning-parser gemma4` | ❌ | Affects token processing |

**Recommendation**: Re-benchmark with customer's exact flags before finalizing.

### Raw Data Files

| Platform | Raw Log File | Status |
|----------|-------------|--------|
| GPU (4×RTX TP=4) | `data/gpu-benchmark-results.txt` | ✅ Available |
| GPU P90 JSON | `data/gpu-p90-results.json` | ✅ Available |
| TPU v6e-4 | `data/tpu-benchmark-p90-results.txt` | ✅ Available (128K, unique random) |
| TPU P90 JSON | `data/tpu-p90-results.json` | ✅ Available |
| TPU P90 128K JSON | `data/tpu-p90-results-128k.json` | ✅ Available |
| Vertex AI MaaS | `data/maas-benchmark-results.txt` | ✅ Available |
| MaaS P90 JSON | `data/maas-p90-results.json` | ✅ Available |
