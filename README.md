# Gemma 4 26B-A4B Inference Benchmark

> **GPU vs TPU vs Vertex AI MaaS** — 3-way performance comparison for serving Gemma 4 26B-A4B-it (50 measured data points)

![3-Way Comparison](plots/09_gpu_tpu_vertexai_comparison.png)

## Overview

This repository benchmarks **Google's Gemma 4 26B-A4B-it** (26B params, MoE with ~4B active) across three deployment options:

| Platform | Hardware | Serving Stack | Data Points | Pricing |
|----------|----------|---------------|-------------|---------|
| **GPU** | NVIDIA RTX Pro 6000 (Blackwell), 96GB GDDR7 | vLLM 0.19.0 (FP8) | 17 | $4.50/hr on-demand |
| **TPU** | Cloud TPU v6e-8 (Trillium), 8 chips, 256GB HBM | vLLM (BF16, TP=8) | 16 | $21.60/hr on-demand |
| **Vertex AI MaaS** | Global endpoint (aiplatform.googleapis.com) | Model-as-a-Service | 17 | $0.15/M in, $0.60/M out |

### Workload

- **Input**: ~20,000 tokens (random, low cache hit rate)
- **Output**: 250 tokens
- **Target**: ~3.5s P90 E2E latency

---

## Quick Results

### Single Request (Baseline)

| Metric | GPU | TPU v6e-8 | **Vertex AI MaaS** | Winner |
|--------|-----|-----------|-----------|--------|
| **TTFT** | 1,009ms | 821ms² | **491ms** | **MaaS** |
| **TPOT** | 9.27ms | **8.60ms** | N/A¹ | **TPU** |
| **E2E Latency** | 3.32s | **2.97s** | **0.80s** | **MaaS** |

### Under Load (0.3 QPS, cold TTFT)

| Metric | GPU | TPU v6e-8 | **Vertex AI MaaS** | Winner |
|--------|-----|-----------|-----------|--------|
| **Mean TTFT (cold)** | ⚠️ 1,009ms³ | 93ms | 680ms | **TPU** |
| **Mean TPOT** | 10.46ms | **8.79ms** | N/A¹ | **TPU** |
| **E2E Latency (cold)** | ⚠️ 3.61s³ | **2.28s** | **1.00s** | **MaaS** |

### Burst (20 concurrent requests)

| Metric | GPU | TPU v6e-8 | **Vertex AI MaaS** | Winner |
|--------|-----|-----------|-----------|--------|
| **Mean TTFT** | 4,651ms | **1,686ms** | 3,091ms | **TPU** |
| **Mean TPOT** | **15.23ms** | 26.59ms | N/A¹ | **GPU** |
| **Throughput** | 323.9 tok/s | N/A⁴ | **~5.0 req/s** | **GPU** (tok/s) |
| **Mean E2E** | 8.44s | 8.31s | **3.42s** | **MaaS** |

> ¹ Managed APIs: per-token timing (TPOT) not measurable  
> ² First-request TTFT includes XLA compilation overhead; steady-state TTFT is ~87ms  
> ³ GPU QPS sweep measured 88ms TTFT due to prefix caching with `seed=42`, but customer states "low cache hit rate". Cold TTFT (1,009ms) used here; cold E2E = 1,009 + 249×10.46 = **3.61s, exceeding 3.5s target**  
> ⁴ TPU burst output tok/s not directly measured in this configuration

---

## Key Findings

1. **🏆 MaaS wins on E2E latency** across all scenarios: 0.80s single, 1.00s at 0.3 QPS, 3.42s burst N=20
2. **⚠️ GPU FAILS 3.5s target at 0.3 QPS** with cold TTFT: E2E = 3.61s (customer has "low cache hit rate")
3. **TPU meets 3.5s target** for single (2.97s) and sustained (2.28s) workloads; GPU only meets it for single requests (3.32s)
4. **TPU wins burst TTFT**: N=20 TTFT=1.7s vs GPU's 4.7s (**2.8x faster**) — 256GB HBM reduces prefill queuing
5. **GPU wins burst decode**: TPOT=15.23ms at N=20 vs TPU's 26.59ms — GPU decode degrades less under load
6. **Under burst N=20, E2E is similar**: GPU 8.44s ≈ TPU 8.31s (TPU's TTFT advantage offset by higher TPOT)
7. **Self-hosted GPU** wins for streaming use cases (real token-by-token output at 9-16ms/token)
8. **⚠️ P90 E2E not measured**: All E2E values are mean estimates — customer's P90 target needs separate validation

---

## Plots

### GPU vs TPU (vLLM direct)

| Plot | Description |
|------|-------------|
| ![TPOT](plots/01_tpot_vs_concurrency.png) | Decode latency degrades with concurrency |
| ![TTFT](plots/02_ttft_vs_concurrency.png) | Prefill queuing explodes beyond max_num_seqs |
| ![E2E](plots/03_e2e_vs_concurrency.png) | End-to-end latency vs concurrency |
| ![Throughput](plots/04_throughput_vs_concurrency.png) | Throughput peaks at max_num_seqs=8 |
| ![QPS Sweep](plots/05_qps_sweep_latency.png) | Steady-state latency across 9 QPS levels |
| ![Breakdown](plots/06_latency_breakdown.png) | TTFT vs TPOT breakdown by concurrency |
| ![E2E Stack](plots/08_e2e_breakdown.png) | E2E breakdown: prefill + decode |

### 3-Way Comparison (GPU vs TPU vs MaaS)

| Plot | Description |
|------|-------------|
| ![Comparison](plots/09_gpu_tpu_vertexai_comparison.png) | 4-panel comparison across all platforms |
| ![Table](plots/10_gpu_tpu_vertexai_table.png) | Full comparison table with winners |

### Vertex AI MaaS Deep Dive

| Plot | Description |
|------|-------------|
| ![MaaS Dashboard](plots/14_maas_benchmark.png) | MaaS: TTFT, latency, throughput |

---

## Reproducing the Benchmarks

### Prerequisites

- GCP project with GPU/TPU quota
- `gcloud` CLI configured
- Python 3.10+, `vllm >= 0.19.0`
- HuggingFace access to `google/gemma-4-26B-A4B-it`

### GPU (RTX Pro 6000)

```bash
# Step 1: Create GPU VM (adjust machine type for your project/quota)
gcloud compute instances create gemma4-gpu-bench \
    --zone=us-central1-a \
    --accelerator=type=nvidia-rtx-pro-6000,count=1 \
    --boot-disk-size=200GB \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE

# Step 2: SSH in and install vLLM
gcloud compute ssh gemma4-gpu-bench --zone=us-central1-a
pip install vllm==0.19.0

# Step 3: Start vLLM server (see configs/vllm-gpu.yaml for full configuration)
vllm serve google/gemma-4-26B-A4B-it \
    --dtype bfloat16 --quantization fp8 \
    --gpu-memory-utilization 0.95 --max-model-len 128000 \
    --max-num-batched-tokens 65536 --max-num-seqs 8 \
    --enable-chunked-prefill --enable-prefix-caching

# Benchmark: single request
vllm bench serve --dataset-name random \
    --random-input-len 20000 --random-output-len 250 \
    --num-prompts 1 --request-rate inf

# Benchmark: QPS sweep
for QPS in 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.7 1.0; do
    vllm bench serve --dataset-name random \
        --random-input-len 20000 --random-output-len 250 \
        --num-prompts 10 --request-rate $QPS --seed 42
done

# Benchmark: burst sweep
for N in 1 2 5 8 10 15 20 30; do
    vllm bench serve --dataset-name random \
        --random-input-len 20000 --random-output-len 250 \
        --num-prompts $N --request-rate inf --seed 42
done
```

### TPU v6e-8 (Trillium)

```bash
# Step 1: Create TPU VM
gcloud compute tpus tpu-vm create gemma4-tpu-bench \
    --zone=us-east5-b \
    --accelerator-type=v6e-8 \
    --version=v2-alpha-tpuv6e

# Step 2: SSH in
gcloud compute tpus tpu-vm ssh gemma4-tpu-bench --zone=us-east5-b

# Step 3: Pull and run vLLM TPU Docker image
docker run -d --name vllm-gemma4 --privileged --net=host \
    -v /dev/shm:/dev/shm --shm-size 16g \
    -e "VLLM_ARGS=--model google/gemma-4-26B-A4B-it \
        --max-model-len 32768 --tensor-parallel-size 8 \
        --disable_chunked_mm_input" \
    vllm/vllm-tpu:gemma4

# Step 4: Run benchmarks (same commands as GPU, or use the custom script)
python3 scripts/tpu-benchmark.py
# Or use vllm bench serve (same commands as GPU)
```

### Vertex AI MaaS (Model-as-a-Service)

```bash
# No deployment needed — uses global endpoint directly:
python3 scripts/maas-benchmark.py
```

---

## Repository Structure

```
├── README.md                          # This file
├── BENCHMARK_REPORT.md                # Detailed report with all raw data
├── configs/
│   ├── vllm-gpu.yaml                 # GPU vLLM configuration
│   └── vllm-tpu.yaml                 # TPU vLLM configuration  
├── data/
│   ├── gpu-benchmark-results.txt     # Raw GPU benchmark output
│   ├── maas-benchmark-results.txt    # Raw MaaS benchmark output
│   └── tpu-benchmark-results.txt     # Raw TPU v6e-8 benchmark output
├── scripts/
│   ├── generate-plots.py             # Generate all 10 plots (50 data points)
│   ├── maas-benchmark.py             # Vertex AI MaaS benchmark
│   ├── tpu-benchmark.py              # TPU vLLM benchmark
│   └── run-benchmarks.sh             # Automated benchmark runner
└── plots/
    ├── 01-06,08_*.png                # GPU vs TPU plots (7 plots)
    ├── 09-10_*.png                   # 3-way comparison
    └── 14_maas_benchmark.png         # MaaS deep dive
```

---

## License

Benchmark data and scripts are provided as-is for reference. Gemma 4 model is subject to [Google's Gemma license](https://ai.google.dev/gemma/terms).
