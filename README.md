# Gemma 4 26B-A4B Inference Benchmark

> **GPU vs TPU vs Vertex AI MaaS** — 3-way performance comparison for serving Gemma 4 26B-A4B-it (51 measured data points)

![3-Way Comparison](plots/09_gpu_tpu_vertexai_comparison.png)

## Overview

This repository benchmarks **Google's Gemma 4 26B-A4B-it** (26B params, MoE with ~4B active) across three deployment options:

| Platform | Hardware | Serving Stack | Data Points | Pricing |
|----------|----------|---------------|-------------|---------|
| **GPU** | 4× NVIDIA RTX Pro 6000 (Blackwell), 384GB GDDR7 | vLLM 0.19.0 (FP8, TP=4) | 18 | $18.00/hr on-demand |
| **TPU** | Cloud TPU v6e-4 (Trillium), 4 chips, ~128GB HBM | vLLM (BF16, TP=4) | 16 | $10.80/hr on-demand |
| **Vertex AI MaaS** | Global endpoint (aiplatform.googleapis.com) | Model-as-a-Service | 17 | $0.15/M in, $0.60/M out |

### Workload

- **Input**: ~20,000 tokens (unique random per request, no prefix caching)
- **Output**: 250 tokens
- **Target**: ~3.5s P90 E2E latency

---

## Quick Results (P90 E2E — Customer SLA Metric)

All values below are **P90 E2E latency** (10 runs per data point) — the metric that matters for customer SLAs.

### Single Request (Baseline, 10 runs)

| Metric | GPU (4×RTX TP=4) | TPU v6e-4 | MaaS | Winner |
|--------|-------------------|-----------|------|--------|
| **Mean TTFT** | 1,108ms | **706ms** | 1,330ms | **TPU** |
| **Mean E2E** | 3.31s | **2.09s** | 2.94s | **TPU** |
| **P90 E2E** | 3.31s | **2.52s** ✅ | 3.09s | **TPU** |

### Under Load (0.3 QPS Steady State)

| Metric | GPU (4×RTX TP=4) | TPU v6e-4 | MaaS | Winner |
|--------|-------------------|-----------|------|--------|
| **Mean TTFT** | 1,106ms | **706ms** | 1,292ms | **TPU** |
| **Mean E2E** | 4.51s | **2.14s** | 2.91s | **TPU** |
| **P90 E2E** | 5.04s | **2.52s** ✅ | 3.08s | **TPU** |

### Burst (20 concurrent requests)

| Metric | GPU (4×RTX TP=4) | TPU v6e-4 | MaaS | Winner |
|--------|-------------------|-----------|------|--------|
| **Mean TTFT** | 10,851ms | 7,831ms | **4,201ms** | **MaaS** |
| **Mean E2E** | 21.31s | 13.55s | **7.42s** | **MaaS** |
| **P90 E2E** | 21.65s | 16.74s | **8.22s** | **MaaS** |

> **Fair methodology**: All TPU benchmarks use unique random prompts per request (no prefix caching bias). Each request generates a fresh ~20K token prompt to ensure TTFT reflects real prefill computation.

---

## Key Findings

1. **🏆 TPU wins single-request & low-QPS**: P90 E2E 2.52s — best of all platforms for single requests and steady QPS ≤ 0.4
2. **TPU has lowest TTFT**: 706ms (unique random prompts) vs GPU 1,108ms vs MaaS 1,330ms
3. **✅ TPU meets 3.5s target for single & QPS ≤ 0.4**: P90 E2E 2.52s at both single and 0.3 QPS
4. **✅ GPU meets 3.5s single-request target**: P90 E2E = 3.31s with 4×RTX TP=4
5. **MaaS wins burst scenarios**: P90 E2E 8.22s at N=20 vs TPU 16.74s vs GPU 21.65s — auto-scaling advantage
6. **MaaS stays flat under QPS sweep**: P90 E2E ~3.0-3.1s from 0.1-0.5 QPS
7. **⚠️ No platform meets 3.5s at burst N=20**: GPU 21.65s, TPU 16.74s, MaaS 8.22s — all exceed target
8. **TPU is cheapest per hour**: $10.80/hr vs GPU $18.00/hr (40% cheaper)
9. **All P90 E2E values are measured** from real per-request distributions (not estimated from mean TTFT+TPOT)

---

## Plots

### GPU vs TPU (vLLM direct)

| Plot | Description |
|------|-------------|
| ![TPOT](plots/01_tpot_vs_concurrency.png) | Decode latency degrades with concurrency |
| ![TTFT](plots/02_ttft_vs_concurrency.png) | Prefill latency vs concurrency |
| ![E2E](plots/03_e2e_vs_concurrency.png) | End-to-end latency vs concurrency |
| ![Throughput](plots/04_throughput_vs_concurrency.png) | Throughput vs concurrency |
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

### GPU (4× RTX Pro 6000, TP=4)

```bash
# Step 1: Create GPU VM
gcloud compute instances create gemma4-gpu-bench \
    --zone=us-central1-a \
    --machine-type=g4-standard-192 \
    --accelerator=type=nvidia-rtx-pro-6000,count=4 \
    --boot-disk-size=200GB \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE

# Step 2: SSH in and install vLLM
gcloud compute ssh gemma4-gpu-bench --zone=us-central1-a
pip install vllm==0.19.0

# Step 3: Start vLLM server (see configs/vllm-gpu.yaml)
vllm serve google/gemma-4-26B-A4B-it \
    --dtype bfloat16 --quantization fp8 \
    --gpu-memory-utilization 0.95 --max-model-len 128000 \
    --max-num-batched-tokens 65536 --max-num-seqs 32 \
    --enable-chunked-prefill --enable-prefix-caching \
    --tensor-parallel-size 4 --trust-remote-code

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

### TPU v6e-4 (Trillium)

```bash
# Step 1: Create TPU VM
gcloud compute tpus tpu-vm create gemma4-tpu-bench \
    --zone=asia-northeast1-b \
    --accelerator-type=v6e-4 \
    --version=v2-alpha-tpuv6e

# Step 2: SSH in
gcloud compute tpus tpu-vm ssh gemma4-tpu-bench --zone=asia-northeast1-b

# Step 3: Pull and run vLLM TPU Docker image
docker run -d --name vllm-gemma4 --privileged --net=host \
    -v /dev/shm:/dev/shm --shm-size 16g \
    -e VLLM_FLASH_ATTN=1 \
    -e "VLLM_ARGS=--model google/gemma-4-26B-A4B-it \
        --max-model-len 128000 --tensor-parallel-size 4 \
        --disable_chunked_mm_input" \
    vllm/vllm-tpu:gemma4

# Step 4: Run benchmarks (unique random prompts, no prefix caching)
python3 scripts/tpu-benchmark-128k.py
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
│   ├── vllm-gpu.yaml                 # GPU 4×RTX TP=4 vLLM configuration
│   └── vllm-tpu.yaml                 # TPU v6e-4 vLLM configuration  
├── data/
│   ├── gpu-benchmark-results.txt     # Raw GPU benchmark output
│   ├── gpu-p90-results.json          # GPU P90 data (JSON)
│   ├── maas-benchmark-results.txt    # Raw MaaS benchmark output
│   ├── maas-p90-results.json         # MaaS P90 data (JSON)
│   ├── tpu-benchmark-p90-results.txt # TPU benchmark output (unique random prompts)
│   ├── tpu-p90-results.json          # TPU P90 data (JSON)
│   └── tpu-p90-results-128k.json     # TPU P90 data — 128K context (JSON)
├── scripts/
│   ├── generate-plots.py             # Generate all 10 plots (51 data points)
│   ├── gpu-benchmark.py              # GPU benchmark script
│   ├── maas-benchmark.py             # Vertex AI MaaS benchmark
│   ├── tpu-benchmark-128k.py         # TPU benchmark (unique random prompts)
│   ├── tpu-deploy-and-bench.sh       # TPU deployment + benchmark automation
│   └── run-benchmarks.sh             # Automated benchmark runner
└── plots/
    ├── 01-06,08_*.png                # GPU vs TPU plots (7 plots)
    ├── 09-10_*.png                   # 3-way comparison
    └── 14_maas_benchmark.png         # MaaS deep dive
```

---

## License

Benchmark data and scripts are provided as-is for reference. Gemma 4 model is subject to [Google's Gemma license](https://ai.google.dev/gemma/terms).
