#!/usr/bin/env python3
"""
Generate benchmark plots for Gemma 4 26B-A4B inference comparison.

All data points are real measurements:
- RTX Pro 6000 (1×GPU): 17 data points (9 QPS sweep + 8 burst sweep) via vllm bench serve
- RTX Pro 6000 (4×GPU TP=4): 18 data points (9 QPS sweep + 1 baseline + 8 burst sweep)
- TPU v6e-8: 16 data points (9 QPS sweep + 7 burst sweep) via vllm bench serve
- Vertex AI MaaS: 17 data points (9 QPS sweep + 8 burst sweep) via aiplatform.googleapis.com

Usage:
    pip install matplotlib numpy
    python3 scripts/generate-plots.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('plots', exist_ok=True)

OUTPUT_TOKENS = 250

def calc_e2e(ttft_ms, tpot_ms, n=OUTPUT_TOKENS):
    """E2E = TTFT + (n-1) * TPOT"""
    return ttft_ms + (n - 1) * tpot_ms

# =============================================================================
# MEASURED DATA — RTX Pro 6000 Blackwell (1x GPU, FP8, prefix caching ON)
# vLLM v0.19.0 on g4-standard-48
# =============================================================================

# QPS sweep: 10 prompts per rate, fresh random prompts per QPS level
# Customer flags: --kv-sharing-fast-prefill, --max-model-len 128000, --max-num-seqs 8
rtx_qps = [
    # (target_qps, achieved_qps, out_tok_s, ttft_ms, tpot_ms, peak_conc)
    (0.10, 0.10,  24.41,  825.6,  9.82, 3),
    (0.15, 0.14,  36.19,  830.0,  9.96, 4),
    (0.20, 0.19,  47.69,  895.7, 10.01, 4),
    (0.25, 0.24,  58.96,  967.9, 10.11, 4),
    (0.30, 0.28,  69.98, 1056.2, 10.40, 4),
    (0.40, 0.37,  91.29, 1205.4, 10.50, 4),
    (0.50, 0.45, 111.71, 1388.4, 11.03, 5),
    (0.70, 0.60, 148.77, 1932.9, 11.91, 5),
    (1.00, 0.78, 195.87, 2372.5, 11.94, 6),
]

# Single request baseline (10 cold runs, fresh prompts, no prefix cache hits)
rtx_baseline = {
    'ttft_ms': 5730.5,
    'tpot_ms': 9.81,
    'itl_median_ms': 9.82,
    'output_tok_s': 75.4,
    'e2e_s': 8.164,
    'e2e_p90_s': 8.333,
    'ttft_p90_ms': 5898.7,
}

# P90 E2E from gpu-p90-results.json (10 runs per data point)
rtx_p90_qps = [3.279, 3.508, 4.133, 4.970, 5.897, 6.609, 7.146, 8.213, 8.127]
rtx_p90_burst = [0.0, 8.538, 9.146, 8.659, 11.699, 11.735, 14.877, 18.074]

# Burst sweep: all requests at once (--request-rate inf), fresh prompts
rtx_burst = [
    # (N, achieved_qps, out_tok_s, ttft_ms, tpot_ms, peak_conc)
    ( 1, 0.12, 30.0,   5964.8,  9.82,  1),
    ( 2, 0.23, 58.6,   5854.2, 10.81,  2),
    ( 5, 0.55, 136.9,  6261.1, 11.86,  5),
    ( 8, 0.92, 231.0,  6018.9, 11.08,  8),
    (10, 1.07, 266.5,  6719.1, 11.10, 10),
    (15, 1.50, 374.2,  7268.4, 11.49, 15),
    (20, 1.78, 443.4,  8600.8, 11.13, 20),
    (30, 2.29, 572.6, 10368.5, 11.30, 30),
]

# =============================================================================
# MEASURED DATA — GPU 4x RTX Pro 6000 (TP=4, FP8, prefix caching ON)
# vLLM v0.19.0 on g4-standard-192 (4x GPUs)
# =============================================================================

# QPS sweep: 10 prompts per rate, fresh random prompts per QPS level
tp4_qps = [
    # (target_qps, achieved_qps, out_tok_s, ttft_ms, tpot_ms, peak_conc)
    (0.10, 0.10,  24.23,   967.1,  9.90, 3),
    (0.15, 0.14,  35.79,  1032.7, 10.64, 4),
    (0.20, 0.19,  47.00,  1046.8, 11.79, 4),
    (0.25, 0.24,  57.85,  1067.3, 12.65, 4),
    (0.30, 0.28,  68.35,  1105.8, 13.66, 4),
    (0.40, 0.37,  88.36,  1209.5, 14.87, 4),
    (0.50, 0.45, 105.84,  1339.8, 17.69, 5),
    (0.70, 0.60, 138.67,  1564.6, 20.68, 5),
    (1.00, 0.78, 170.64,  2067.3, 25.14, 6),
]

# Single request baseline (10 cold runs, fresh prompts)
tp4_baseline = {
    'ttft_ms': 1108.3,
    'tpot_ms': 8.83,
    'output_tok_s': 76.67,
    'e2e_s': 3.306,
    'e2e_p90_s': 3.313,
    'ttft_p90_ms': 1115.2,
}

# P90 E2E from gpu-tp4-p90-results.json (10 runs per data point)
tp4_p90_qps = [3.791, 4.112, 4.541, 4.668, 5.035, 5.451, 6.444, 7.669, 8.787]
tp4_p90_burst = [3.158, 3.480, 5.568, 5.684, 5.106, 16.647, 21.651, 31.277]

# Burst sweep: all requests at once (--request-rate inf), fresh prompts
tp4_burst = [
    # (N, achieved_qps, out_tok_s, ttft_ms, tpot_ms, peak_conc)
    ( 1, 0.37,  95.4,    535.7,  8.69,  1),
    ( 2, 0.66, 169.7,    592.3,  9.73,  2),
    ( 5, 1.05, 283.3,   1797.3, 11.91,  5),
    ( 8, 1.42, 351.0,   1971.8, 14.75,  8),
    (10, 1.99, 489.7,   1588.9, 13.84, 10),
    (15, 1.51, 405.0,   3762.7, 24.89, 15),
    (20, 0.93, 230.9,  10851.3, 43.23, 20),
    (30, 0.96, 239.9,  15541.9, 62.91, 30),
]

# =============================================================================
# MEASURED DATA — TPU v6e-8 Trillium (8 chips, 256GB HBM, BF16)
# vLLM Docker (vllm/vllm-tpu:gemma4), TP=8, max-model-len=128000
# Full 16 data points (9 QPS sweep + 7 burst sweep)
# =============================================================================

# QPS sweep: 10 prompts per rate, fresh random prompts per QPS level
tpu_qps = [
    # (target_qps, mean_ttft_ms, median_ttft_ms, p99_ttft_ms, mean_tpot_ms, mean_itl_ms)
    (0.10,  386.4,  213.3, 1788.8,  7.08,  7.08),
    (0.15,  381.8,  209.4, 1784.9,  6.98,  6.98),
    (0.20,  384.0,  210.8, 1786.3,  7.00,  7.00),
    (0.25,  385.4,  212.7, 1787.4,  7.11,  7.11),
    (0.30,  385.6,  213.1, 1783.7,  7.03,  7.03),
    (0.40,  383.0,  211.0, 1785.0,  7.01,  7.01),
    (0.50,  385.6,  212.5, 1787.0,  7.09,  7.09),
    (0.70,  420.6,  212.1, 1842.9,  7.08,  7.08),
    (1.00,  467.6,  216.3, 1885.3,  7.20,  7.20),
]

# Burst sweep: all requests at once, fresh prompts
tpu_burst = [
    # (N, mean_ttft_ms, median_ttft_ms, p99_ttft_ms, mean_tpot_ms, mean_itl_ms)
    ( 1, 1939.5, 1939.5, 1939.5,  6.91,  6.91),
    ( 2, 2020.8, 2020.8, 2021.2,  7.01,  7.01),
    ( 5, 2263.4, 2263.6, 2264.3,  8.16,  8.16),
    ( 8, 2175.5, 2177.8, 2180.0,  9.54,  9.54),
    (10, 2348.5, 2347.9, 2355.0, 10.83, 10.83),
    (15, 2433.8, 2435.6, 2438.9, 14.49, 14.49),
    (20, 2682.7, 2679.2, 2694.1, 19.23, 19.23),
]

# Single request baseline (10 cold runs, fresh prompts)
tpu_baseline = {
    'ttft_ms': 1947.9,  # 128K context, cold prompts
    'tpot_ms': 6.93,
    'itl_median_ms': 6.94,
    'output_tok_s': 145.0,
    'total_tok_s': 5530.0,
    'e2e_p90_s': 3.685,
    'ttft_p90_ms': 1954.8,
}

# P90 E2E from tpu-p90-results.json (10 runs per data point)
tpu_p90_qps = [2.150, 2.122, 2.140, 2.170, 2.134, 2.134, 2.150, 2.427, 2.878]
tpu_p90_burst = [0.0, 3.725, 4.182, 4.400, 4.820, 5.758, 7.033]  # N=1..20

# =============================================================================
# MEASURED DATA — Vertex AI (Managed Model Garden endpoint, TPU-backed)
# Custom benchmark script, streaming mode
# Workload: ~20k input (repeated text), 250 output tokens
# Vertex AI Endpoint data retained for reference but not plotted
# =============================================================================

# QPS sweep: 10 prompts per rate
vai_qps = [
    # (target_qps, mean_ttft_ms, median_ttft_ms, p99_ttft_ms, mean_latency_s, peak_conc, req_throughput)
    (0.10, 2918.3, 2788.9, 3973.8, 2.92, 1, 0.108),
    (0.15, 2760.8, 2759.4, 2800.1, 2.76, 1, 0.159),
    (0.20, 2746.4, 2746.2, 2822.3, 2.75, 1, 0.209),
    (0.25, 2730.1, 2727.4, 2798.1, 2.73, 1, 0.258),
    (0.30, 2733.2, 2737.0, 2763.9, 2.73, 1, 0.305),
    (0.40, 2743.9, 2741.4, 2777.9, 2.74, 2, 0.396),
    (0.50, 3029.2, 3063.1, 3109.4, 3.03, 2, 0.478),
    (0.70, 3498.1, 3577.2, 3738.0, 3.50, 3, 0.622),
    (1.00, 3785.5, 3842.3, 4053.5, 3.79, 5, 0.800),
]

# Burst sweep: all requests simultaneous
vai_burst = [
    # (N, mean_ttft_ms, median_ttft_ms, p99_ttft_ms, mean_latency_s, peak_conc, req_throughput)
    ( 1, 2731.4, 2731.4, 2731.4, 2.73,  1, 0.366),
    ( 2, 3316.4, 3316.5, 3339.2, 3.32,  2, 0.599),
    ( 5, 4553.0, 4593.3, 4604.5, 4.55,  5, 1.086),
    ( 8, 4914.7, 4940.4, 5035.4, 4.91,  8, 1.588),
    (10, 6275.1, 6320.0, 6388.5, 6.28, 10, 1.565),
    (15, 7430.5, 7546.7, 7672.6, 7.43, 15, 1.954),
    (20, 9882.3, 9983.5, 10185.2, 9.88, 20, 1.962),
    (30, 11932.5, 12078.7, 12483.8, 11.93, 30, 2.402),
]

# =============================================================================
# MEASURED DATA — Vertex AI MaaS (Model-as-a-Service, global endpoint)
# google/gemma-4-26b-a4b-it-maas via aiplatform.googleapis.com
# Streaming mode, ~20k input tokens (FRESH RANDOM per request), 250 output
# Fair comparison: unique prompts prevent prefix caching (like customer workload)
# =============================================================================

# Single request baseline (10 cold runs, fresh prompts)
maas_baseline = {
    'ttft_mean_ms': 1330.0,
    'ttft_p90_ms': 1525.0,
    'e2e_mean_s': 2.935,
    'e2e_p90_s': 3.090,
    'tpot_mean_ms': 6.34,
    'tpot_p90_ms': 6.63,
}

# QPS sweep: 10 fresh random prompts per rate
maas_qps = [
    # (target_qps, mean_ttft_ms, median_ttft_ms, p99_ttft_ms, mean_latency_s, req_throughput, p90_ttft_ms, p90_e2e_s)
    (0.10, 1274.5, 1264.7, 1371.3, 2.903, 0.109, 1347.6, 2.935),
    (0.15, 1396.7, 1418.4, 1577.9, 3.024, 0.150, 1527.3, 3.142),
    (0.20, 1311.1, 1292.2, 1401.4, 2.941, 0.200, 1380.1, 3.052),
    (0.25, 1284.1, 1289.9, 1369.9, 2.945, 0.250, 1344.7, 3.056),
    (0.30, 1292.0, 1265.9, 1451.4, 2.912, 0.300, 1446.5, 3.079),
    (0.40, 1335.4, 1369.2, 1486.1, 3.005, 0.400, 1440.3, 3.094),
    (0.50, 1283.6, 1242.8, 1471.5, 2.932, 0.500, 1399.2, 3.054),
    (0.70, 1276.5, 1260.4, 1435.3, 3.273, 0.700, 1378.9, 3.798),
    (1.00, 1303.3, 1229.7, 1570.0, 3.371, 1.000, 1568.3, 3.610),
]

# Burst sweep: all requests simultaneous, fresh random prompts
maas_burst = [
    # (N, mean_ttft_ms, median_ttft_ms, p99_ttft_ms, mean_latency_s, req_throughput, p90_ttft_ms, p90_e2e_s)
    ( 1, 1484.4, 1484.4, 1484.4, 3.805, 0.263, 0.0, 0.000),
    ( 2, 1533.1, 1533.1, 1533.3, 3.635, 0.550, 1533.2, 3.793),
    ( 5, 1655.4, 1656.5, 2237.5, 3.925, 1.274, 2028.9, 4.204),
    ( 8, 1836.4, 1742.9, 2305.0, 4.450, 1.798, 2116.2, 4.951),
    (10, 2391.5, 2178.3, 3294.7, 5.120, 1.953, 2824.0, 6.106),
    (15, 3273.6, 3187.9, 3767.6, 6.169, 2.431, 3703.2, 7.302),
    (20, 4201.1, 4195.0, 5376.2, 7.423, 2.694, 4964.0, 8.221),
    (30, 5594.4, 5535.1, 6918.3, 9.058, 3.312, 6376.5, 9.903),
]

# =============================================================================
# STYLING
# =============================================================================
RTX_COLOR = '#2ecc71'
TP4_COLOR = '#e67e22'
TPU_COLOR = '#3498db'
VAI_COLOR = '#e74c3c'
MAAS_COLOR = '#9b59b6'
RTX_LABEL = 'GPU 1×RTX (17 pts)'
TP4_LABEL = 'GPU 4×RTX TP=4 (18 pts)'
TPU_LABEL = 'TPU v6e-8 (16 pts)'
VAI_LABEL = 'Vertex AI Endpoint (17 pts)'
MAAS_LABEL = 'Vertex AI MaaS (17 pts)'
plt.rcParams.update({'font.size': 11})


# =============================================================================
# Plot 1: TPOT vs Concurrency (GPU + TPU)
# =============================================================================
def plot_01():
    fig, ax = plt.subplots(figsize=(12, 7))
    conc = [d[0] for d in rtx_burst]
    tpot = [d[4] for d in rtx_burst]
    ax.plot(conc, tpot, 's-', color=RTX_COLOR, linewidth=2.5, markersize=10,
            label=RTX_LABEL, zorder=3)
    for c, t in zip(conc, tpot):
        ax.annotate(f'{t:.1f}ms', xy=(c, t), fontsize=8,
                    xytext=(8, 8), textcoords='offset points')
    # TPU v6e-8 full burst data (7 points)
    tpu_conc = [d[0] for d in tpu_burst]
    tpu_tpot_vals = [d[4] for d in tpu_burst]
    ax.plot(tpu_conc, tpu_tpot_vals, 'D-', color=TPU_COLOR, linewidth=2.5, markersize=10,
            label=TPU_LABEL, zorder=4)
    for c, t in zip(tpu_conc, tpu_tpot_vals):
        ax.annotate(f'{t:.1f}ms', xy=(c, t), fontsize=8, fontweight='bold',
                    color=TPU_COLOR, xytext=(8, -12), textcoords='offset points')
    # GPU 4×RTX TP=4 burst TPOT (8 points)
    tp4_conc = [d[0] for d in tp4_burst]
    tp4_tpot_vals = [d[4] for d in tp4_burst]
    ax.plot(tp4_conc, tp4_tpot_vals, 'o-', color=TP4_COLOR, linewidth=2.5, markersize=10,
            label=TP4_LABEL, zorder=5)
    ax.axhspan(0, 10, color='green', alpha=0.06)
    ax.axhspan(10, 14, color='yellow', alpha=0.06)
    ax.axhspan(14, 30, color='red', alpha=0.06)
    ax.axvline(x=8, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
               label='RTX max_num_seqs=8')
    ax.set_xlabel('Concurrent Requests', fontsize=13)
    ax.set_ylabel('Mean TPOT (ms)', fontsize=13)
    ax.set_title('Decode Latency vs Concurrency\nGPU (1×, 4×TP=4) vs TPU v6e-8',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 32)
    ax.set_ylim(6, 65)
    plt.tight_layout()
    plt.savefig('plots/01_tpot_vs_concurrency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot 1: plots/01_tpot_vs_concurrency.png")


# =============================================================================
# Plot 2: TTFT vs Concurrency (GPU + TPU)
# =============================================================================
def plot_02():
    fig, ax = plt.subplots(figsize=(12, 7))
    conc = [d[0] for d in rtx_burst]
    ttft = [d[3] for d in rtx_burst]
    ax.plot(conc, ttft, 's-', color=RTX_COLOR, linewidth=2.5, markersize=10,
            label=RTX_LABEL, zorder=3)
    for c, t in zip(conc, ttft):
        ax.annotate(f'{t:.0f}ms', xy=(c, t), fontsize=8,
                    xytext=(8, 8), textcoords='offset points')
    # TPU v6e-8 full burst data (7 points)
    tpu_conc = [d[0] for d in tpu_burst]
    tpu_ttft_vals = [d[1] for d in tpu_burst]
    ax.plot(tpu_conc, tpu_ttft_vals, 'D-', color=TPU_COLOR, linewidth=2.5, markersize=10,
            label=TPU_LABEL, zorder=4)
    for c, t in zip(tpu_conc, tpu_ttft_vals):
        ax.annotate(f'{t:.0f}ms', xy=(c, t), fontsize=8, fontweight='bold',
                    color=TPU_COLOR, xytext=(8, -12), textcoords='offset points')
    # GPU 4×RTX TP=4 burst TTFT (8 points)
    tp4_conc = [d[0] for d in tp4_burst]
    tp4_ttft_vals = [d[3] for d in tp4_burst]
    ax.plot(tp4_conc, tp4_ttft_vals, 'o-', color=TP4_COLOR, linewidth=2.5, markersize=10,
            label=TP4_LABEL, zorder=5)
    ax.axvline(x=8, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
               label='RTX max_num_seqs=8')
    ax.set_xlabel('Concurrent Requests', fontsize=13)
    ax.set_ylabel('Mean TTFT (ms)', fontsize=13)
    ax.set_title('Prefill Latency vs Concurrency\nGPU (1×, 4×TP=4) vs TPU v6e-8',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 32)
    plt.tight_layout()
    plt.savefig('plots/02_ttft_vs_concurrency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot 2: plots/02_ttft_vs_concurrency.png")


# =============================================================================
# Plot 3: E2E Latency vs Concurrency (GPU + TPU)
# =============================================================================
def plot_03():
    fig, ax = plt.subplots(figsize=(12, 7))
    conc = [d[0] for d in rtx_burst]
    e2e = [calc_e2e(d[3], d[4]) / 1000 for d in rtx_burst]
    ax.plot(conc, e2e, 's-', color=RTX_COLOR, linewidth=2.5, markersize=10,
            label=RTX_LABEL, zorder=3)
    for c, e in zip(conc, e2e):
        ax.annotate(f'{e:.1f}s', xy=(c, e), fontsize=9,
                    xytext=(10, 5), textcoords='offset points')
    # TPU v6e-8 full burst E2E (7 points)
    tpu_conc = [d[0] for d in tpu_burst]
    tpu_e2e = [calc_e2e(d[1], d[4]) / 1000 for d in tpu_burst]
    ax.plot(tpu_conc, tpu_e2e, 'D-', color=TPU_COLOR, linewidth=2.5, markersize=10,
            label=TPU_LABEL, zorder=4)
    # GPU 4×RTX TP=4 burst E2E
    tp4_conc = [d[0] for d in tp4_burst]
    tp4_e2e = [calc_e2e(d[3], d[4]) / 1000 for d in tp4_burst]
    ax.plot(tp4_conc, tp4_e2e, 'o-', color=TP4_COLOR, linewidth=2.5, markersize=10,
            label=TP4_LABEL, zorder=5)
    ax.axhline(y=3.5, color='red', linestyle=':', linewidth=2.5, label='Target: 3.5s E2E')
    ax.axhspan(0, 3.5, color='green', alpha=0.04)
    ax.set_xlabel('Concurrent Requests', fontsize=13)
    ax.set_ylabel('Mean E2E Latency (seconds)', fontsize=13)
    ax.set_title('End-to-End Latency vs Concurrency\nGPU (1×, 4×TP=4) vs TPU — 20K input, 250 output',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, max(e2e + tpu_e2e) * 1.1)
    plt.tight_layout()
    plt.savefig('plots/03_e2e_vs_concurrency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot 3: plots/03_e2e_vs_concurrency.png")


# =============================================================================
# Plot 4: Throughput vs Concurrency (GPU + TPU)
# =============================================================================
def plot_04():
    fig, ax = plt.subplots(figsize=(12, 7))
    conc = [d[0] for d in rtx_burst]
    mean_thr = [d[2] for d in rtx_burst]
    ax.plot(conc, mean_thr, 's-', color=RTX_COLOR, linewidth=2.5, markersize=10,
            label=RTX_LABEL, zorder=3)
    # TPU v6e-8: estimate throughput from burst data (N * 250 / duration_approx)
    tpu_conc = [d[0] for d in tpu_burst]
    tpu_thr_est = [tpu_baseline['output_tok_s']] + [d[0] * 250 / (calc_e2e(d[1], d[4]) / 1000) for d in tpu_burst[1:]]
    ax.plot(tpu_conc, tpu_thr_est, 'D-', color=TPU_COLOR, linewidth=2.5, markersize=10,
            label=TPU_LABEL, zorder=4)
    # GPU 4×RTX TP=4 throughput
    tp4_conc = [d[0] for d in tp4_burst]
    tp4_thr = [d[2] for d in tp4_burst]
    ax.plot(tp4_conc, tp4_thr, 'o-', color=TP4_COLOR, linewidth=2.5, markersize=10,
            label=TP4_LABEL, zorder=5)
    ax.axvline(x=8, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
               label='RTX max_num_seqs=8')
    ax.set_xlabel('Concurrent Requests', fontsize=13)
    ax.set_ylabel('Output Token Throughput (tok/s)', fontsize=13)
    ax.set_title('Throughput vs Concurrency', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 600)
    plt.tight_layout()
    plt.savefig('plots/04_throughput_vs_concurrency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot 4: plots/04_throughput_vs_concurrency.png")


# =============================================================================
# Plot 5: QPS Sweep (RTX + TPU points)
# =============================================================================
def plot_05():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    qps = [d[1] for d in rtx_qps]
    ttft = [d[3] for d in rtx_qps]
    tpot = [d[4] for d in rtx_qps]
    e2e = [calc_e2e(d[3], d[4]) / 1000 for d in rtx_qps]
    ax1.plot(qps, ttft, 's-', color=RTX_COLOR, linewidth=2.5, markersize=9, label='RTX TTFT', zorder=3)
    ax1.plot(qps, tpot, 's--', color=RTX_COLOR, linewidth=2, markersize=8, label='RTX TPOT', alpha=0.7, zorder=3)
    # TPU v6e-8 full QPS sweep (9 points)
    tpu_qps_rates = [d[0] for d in tpu_qps]
    tpu_ttft_vals = [d[1] for d in tpu_qps]
    tpu_tpot_vals = [d[4] for d in tpu_qps]
    ax1.plot(tpu_qps_rates, tpu_ttft_vals, 'D-', color=TPU_COLOR, linewidth=2.5, markersize=9, label='TPU TTFT', zorder=4)
    ax1.plot(tpu_qps_rates, tpu_tpot_vals, 'D--', color=TPU_COLOR, linewidth=2, markersize=8, label='TPU TPOT', alpha=0.7, zorder=4)
    ax1.set_xlabel('Request Rate (QPS)', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('TTFT & TPOT vs QPS', fontsize=13, fontweight='bold')
    # GPU 4×RTX TP=4 QPS sweep
    tp4_qps_rates = [d[0] for d in tp4_qps]
    tp4_ttft_vals = [d[3] for d in tp4_qps]
    tp4_tpot_vals = [d[4] for d in tp4_qps]
    ax1.plot(tp4_qps_rates, tp4_ttft_vals, 'o-', color=TP4_COLOR, linewidth=2.5, markersize=9, label='TP=4 TTFT', zorder=5)
    ax1.plot(tp4_qps_rates, tp4_tpot_vals, 'o--', color=TP4_COLOR, linewidth=2, markersize=8, label='TP=4 TPOT', alpha=0.7, zorder=5)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax2.plot(qps, e2e, 's-', color=RTX_COLOR, linewidth=2.5, markersize=9, label='GPU 1×', zorder=3)
    tpu_e2e_qps = [calc_e2e(d[1], d[4]) / 1000 for d in tpu_qps]
    ax2.plot(tpu_qps_rates, tpu_e2e_qps, 'D-', color=TPU_COLOR, linewidth=2.5, markersize=9, label='TPU', zorder=4)
    tp4_e2e_qps = [calc_e2e(d[3], d[4]) / 1000 for d in tp4_qps]
    ax2.plot(tp4_qps_rates, tp4_e2e_qps, 'o-', color=TP4_COLOR, linewidth=2.5, markersize=9, label='GPU 4× TP=4', zorder=5)
    ax2.axhline(y=3.5, color='red', linestyle=':', linewidth=2, label='Target: 3.5s')
    ax2.set_xlabel('Achieved Request Rate (QPS)', fontsize=12)
    ax2.set_ylabel('E2E Latency (seconds)', fontsize=12)
    ax2.set_title('E2E Latency vs QPS', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.suptitle('Steady-State Performance', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('plots/05_qps_sweep_latency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot 5: plots/05_qps_sweep_latency.png")


# =============================================================================
# Plot 6: Latency Breakdown Bar Chart (GPU + TPU)
# =============================================================================
def plot_06():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    scenarios = ['N=1', 'N=2', 'N=5', 'N=8', 'N=10', 'N=15', 'N=20', 'N=30']
    x = np.arange(len(scenarios))
    width = 0.35
    rtx_ttft = [d[3] for d in rtx_burst]
    rtx_tpot = [d[4] for d in rtx_burst]
    bars1 = ax1.bar(x - width/2, rtx_ttft, width, label='RTX', color=RTX_COLOR, alpha=0.85, zorder=3)
    # TPU v6e-8 full burst data (7 points matching N=1,2,5,8,10,15,20)
    tpu_ttft_map = {d[0]: d[1] for d in tpu_burst}
    tpu_tpot_map = {d[0]: d[4] for d in tpu_burst}
    burst_ns = [1, 2, 5, 8, 10, 15, 20, 30]
    tpu_bar_ttft = [tpu_ttft_map.get(n, 0) for n in burst_ns]
    tpu_bar_tpot = [tpu_tpot_map.get(n, 0) for n in burst_ns]
    tpu_valid = [i for i, n in enumerate(burst_ns) if n in tpu_ttft_map]
    ax1.bar([x[i] + width/2 for i in tpu_valid], [tpu_bar_ttft[i] for i in tpu_valid], width,
            label='TPU', color=TPU_COLOR, alpha=0.85, zorder=3)
    ax1.set_ylabel('Mean TTFT (ms)', fontsize=12)
    ax1.set_title('Time to First Token', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    bars2 = ax2.bar(x - width/2, rtx_tpot, width, label='RTX', color=RTX_COLOR, alpha=0.85, zorder=3)
    ax2.bar([x[i] + width/2 for i in tpu_valid], [tpu_bar_tpot[i] for i in tpu_valid], width,
            label='TPU', color=TPU_COLOR, alpha=0.85, zorder=3)
    ax2.set_ylabel('Mean TPOT (ms)', fontsize=12)
    ax2.set_title('Time per Output Token', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.suptitle('Latency Breakdown: TTFT & TPOT', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/06_latency_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot 6: plots/06_latency_breakdown.png")


# =============================================================================
# Plot 8: E2E Breakdown Stacked (GPU + TPU)
# =============================================================================
def plot_08():
    fig, ax = plt.subplots(figsize=(14, 7))
    scenarios = ['N=1', 'N=5', 'N=8', 'N=10', 'N=20', 'N=30']
    burst_idx = [0, 2, 3, 4, 6, 7]
    x = np.arange(len(scenarios))
    width = 0.35
    rtx_ttft_s = [rtx_burst[i][3] / 1000 for i in burst_idx]
    rtx_decode_s = [rtx_burst[i][4] * (OUTPUT_TOKENS - 1) / 1000 for i in burst_idx]
    ax.bar(x - width/2, rtx_ttft_s, width, label='RTX TTFT (prefill)',
           color=RTX_COLOR, alpha=0.9, zorder=3)
    ax.bar(x - width/2, rtx_decode_s, width, bottom=rtx_ttft_s,
           label='RTX Decode', color=RTX_COLOR, alpha=0.4, zorder=3, hatch='//')
    # TPU v6e-8 full burst data for stacked bars
    tpu_ttft_map2 = {d[0]: d[1] / 1000 for d in tpu_burst}
    tpu_decode_map2 = {d[0]: d[4] * 249 / 1000 for d in tpu_burst}
    sel_ns = [1, 5, 8, 10, 20]
    tpu_bar_idx = [i for i, n in enumerate([1, 5, 8, 10, 20, 30]) if n in tpu_ttft_map2]
    tpu_ttft_s = [tpu_ttft_map2.get(n, 0) for n in [1, 5, 8, 10, 20, 30] if n in tpu_ttft_map2]
    tpu_decode_s = [tpu_decode_map2.get(n, 0) for n in [1, 5, 8, 10, 20, 30] if n in tpu_decode_map2]
    ax.bar([x[i] + width/2 for i in tpu_bar_idx], tpu_ttft_s, width,
           label='TPU TTFT (prefill)', color=TPU_COLOR, alpha=0.9, zorder=3)
    ax.bar([x[i] + width/2 for i in tpu_bar_idx], tpu_decode_s, width,
           bottom=tpu_ttft_s, label='TPU Decode', color=TPU_COLOR, alpha=0.4, zorder=3, hatch='//')
    ax.axhline(y=3.5, color='red', linestyle=':', linewidth=2.5, label='Target: 3.5s E2E')
    ax.set_xlabel('Burst Size', fontsize=13)
    ax.set_ylabel('Time (seconds)', fontsize=13)
    ax.set_title('E2E Latency Breakdown: TTFT + Decode', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('plots/08_e2e_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot 8: plots/08_e2e_breakdown.png")


# =============================================================================
# Plot 9: GPU vs TPU vs MaaS Comparison (3-Way, 4-panel)
# =============================================================================
def plot_09():
    plt.rcParams.update({
        'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#16213e',
        'axes.edgecolor': '#444', 'axes.labelcolor': '#eee',
        'text.color': '#eee', 'xtick.color': '#ccc', 'ytick.color': '#ccc',
        'grid.color': '#333', 'grid.alpha': 0.4,
    })
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Gemma 4 26B-A4B -- 3-Way Comparison\n(~20k input, 250 output tokens)',
                 fontsize=18, fontweight='bold', color='white', y=0.98)
    labels = ['GPU\n(RTX 6000)', 'TPU v6e-8', 'Vertex AI\n(MaaS)']
    colors = ['#00ff88', '#ff6b6b', '#bb86fc']

    # Panel 1: Single TTFT (from baseline dicts and burst[0] data)
    ax = axes[0,0]
    vals = [rtx_baseline['ttft_ms'], tpu_baseline['ttft_ms'], maas_burst[0][1]]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+50, f'{v:,}ms', ha='center', va='bottom', fontweight='bold', color='white', fontsize=12)
    ax.set_ylabel('TTFT (ms)')
    ax.set_title('Single Request: Mean TTFT', fontweight='bold', color='#ffd700')
    ax.grid(True, axis='y')

    # Panel 2: Single E2E — P90 (customer SLA metric, all chips)
    ax = axes[0,1]
    vals = [rtx_baseline['e2e_p90_s'], tpu_baseline['e2e_p90_s'], maas_baseline['e2e_p90_s']]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f'{v:.2f}s', ha='center', va='bottom', fontweight='bold', color='white', fontsize=12)
    ax.set_ylabel('P90 E2E Latency (s)')
    ax.set_title('Single Request: P90 E2E Latency', fontweight='bold', color='#ffd700')
    ax.grid(True, axis='y')
    ax.axhline(y=3.5, color='#ff6b6b', linestyle='--', alpha=0.3, label='3.5s target')
    ax.legend(facecolor='#16213e', edgecolor='#444')

    # Panel 3: Burst 20 E2E — P90 for all chips
    ax = axes[1,0]
    vals = [rtx_p90_burst[6], tpu_p90_burst[6], maas_burst[6][7]]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.15, f'{v:.1f}s', ha='center', va='bottom', fontweight='bold', color='white', fontsize=12)
    ax.set_ylabel('P90 E2E Latency (s)')
    ax.set_title('Burst 20: P90 E2E Latency', fontweight='bold', color='#ffd700')
    ax.axhline(y=3.5, color='#ff6b6b', linestyle='--', alpha=0.3, label='3.5s target')
    ax.legend(facecolor='#16213e', edgecolor='#444')
    ax.grid(True, axis='y')

    # Panel 4: Burst E2E sweep — P90 for all chips
    ax = axes[1,1]
    rtx_burst_n = [d[0] for d in rtx_burst]
    rtx_bp90_valid = [(n, p) for n, p in zip(rtx_burst_n, rtx_p90_burst) if p > 0]
    ax.plot([x[0] for x in rtx_bp90_valid], [x[1] for x in rtx_bp90_valid], 's-', color='#00ff88', linewidth=2, markersize=8, label='GPU P90 E2E')
    tpu_bn = [d[0] for d in tpu_burst]
    tpu_bp90_valid = [(n, p) for n, p in zip(tpu_bn, tpu_p90_burst) if p > 0]
    ax.plot([x[0] for x in tpu_bp90_valid], [x[1] for x in tpu_bp90_valid], 'D-', color='#ff6b6b', linewidth=2, markersize=8, label='TPU P90 E2E', zorder=4)
    maas_burst_n_p90 = [d[0] for d in maas_burst if d[7] > 0]
    maas_burst_p90 = [d[7] for d in maas_burst if d[7] > 0]
    ax.plot(maas_burst_n_p90, maas_burst_p90, '^-', color='#bb86fc', linewidth=2, markersize=8, label='MaaS P90 E2E', zorder=5)
    ax.axhline(y=3.5, color='#ff6b6b', linestyle='--', alpha=0.3, label='3.5s target')
    ax.set_xlabel('Burst Size (N concurrent)', color='#ccc')
    ax.set_ylabel('P90 E2E Latency (s)')
    ax.set_title('Burst Sweep: P90 E2E Latency', fontweight='bold', color='#ffd700')
    ax.legend(facecolor='#16213e', edgecolor='#444', fontsize=8)
    ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('plots/09_gpu_tpu_vertexai_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.edgecolor': 'black', 'axes.labelcolor': 'black', 'text.color': 'black',
        'xtick.color': 'black', 'ytick.color': 'black', 'grid.color': '#ccc', 'grid.alpha': 0.3})
    print("  Plot 9: plots/09_gpu_tpu_vertexai_comparison.png")


# =============================================================================
# Plot 10: Comparison Table (GPU, TPU, MaaS)
# =============================================================================
def plot_10():
    plt.rcParams.update({
        'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#16213e',
        'axes.edgecolor': '#444', 'axes.labelcolor': '#eee', 'text.color': '#eee',
    })
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis('off')
    col_labels = ['Metric', 'GPU (RTX Pro 6000)', 'TPU v6e-8 (vLLM)', 'Vertex AI MaaS', 'Winner']
    table_data = [
        ['-- Single Request --', '', '', '', ''],
        ['Mean TTFT (ms)', f'{rtx_baseline["ttft_ms"]:,.0f}', f'{tpu_baseline["ttft_ms"]:,.0f}', f'{maas_baseline["ttft_mean_ms"]:,.0f}', 'MaaS'],
        ['Mean E2E (s)', f'{rtx_baseline["e2e_s"]:.2f}', f'{calc_e2e(tpu_baseline["ttft_ms"], tpu_baseline["tpot_ms"])/1000:.2f}', f'{maas_baseline["e2e_mean_s"]:.2f}', 'MaaS'],
        ['P90 E2E (s)', f'{rtx_baseline["e2e_p90_s"]:.2f}', f'{tpu_baseline["e2e_p90_s"]:.2f}', f'{maas_baseline["e2e_p90_s"]:.2f}', 'MaaS'],
        ['-- 0.3 QPS Steady --', '', '', '', ''],
        ['Mean TTFT (ms)', f'{rtx_qps[4][3]:,.0f}', f'{tpu_qps[4][1]:,.0f}', f'{maas_qps[4][1]:,.0f}', 'TPU'],
        ['Mean E2E (s)', f'{calc_e2e(rtx_qps[4][3], rtx_qps[4][4])/1000:.2f}', f'{calc_e2e(tpu_qps[4][1], tpu_qps[4][4])/1000:.2f}', f'{maas_qps[4][4]:.2f}', 'TPU'],
        ['P90 E2E (s)', f'{rtx_p90_qps[4]:.2f}', f'{tpu_p90_qps[4]:.2f}', f'{maas_qps[4][7]:.2f}', 'TPU'],
        ['-- Burst 20 --', '', '', '', ''],
        ['Mean TTFT (ms)', f'{rtx_burst[6][3]:,.0f}', f'{tpu_burst[6][1]:,.0f}', f'{maas_burst[6][1]:,.0f}', 'TPU'],
        ['Mean E2E (s)', f'{calc_e2e(rtx_burst[6][3], rtx_burst[6][4])/1000:.2f}', f'{calc_e2e(tpu_burst[6][1], tpu_burst[6][4])/1000:.2f}', f'{maas_burst[6][4]:.2f}', 'TPU'],
        ['P90 E2E (s)', f'{rtx_p90_burst[6]:.2f}', f'{tpu_p90_burst[6]:.2f}', f'{maas_burst[6][7]:.2f}', 'TPU'],
        ['-- Cost --', '', '', '', ''],
        ['On-demand ($/hr)', '$4.50', '$21.60', 'Pay-per-token', 'GPU'],
        ['Cost/M out tokens', '$16.58', '$71.07', '$12.60**', 'MaaS'],
    ]
    cell_colors = []
    for row in table_data:
        if row[0].startswith('--'):
            cell_colors.append(['#0f3460']*5)
        else:
            cell_colors.append(['#16213e']*5)
    table = ax.table(cellText=table_data, colLabels=col_labels, cellColours=cell_colors,
                     colColours=['#0f3460']*5, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#444')
        cell.set_text_props(color='white')
        if key[0] == 0:
            cell.set_text_props(fontweight='bold', color='white')
    ax.set_title('GPU vs TPU vs MaaS — 3-Way Comparison\n(~20k input, 250 output tokens)',
                 fontsize=16, fontweight='bold', color='white', pad=20)
    fig.text(0.5, 0.02, '* GPU cold TTFT used (customer has low cache hit rate)  ** MaaS: $0.15/M in + $0.60/M out (incl. 20K input cost)',
             ha='center', fontsize=9, color='#888', style='italic')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('plots/10_gpu_tpu_vertexai_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.edgecolor': 'black', 'axes.labelcolor': 'black', 'text.color': 'black'})
    print("  Plot 10: plots/10_gpu_tpu_vertexai_table.png")


# =============================================================================
# Plot 14: MaaS Dashboard (4-panel, MaaS data only)
# =============================================================================
def plot_14():
    plt.rcParams.update({
        'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#16213e',
        'axes.edgecolor': '#444', 'axes.labelcolor': '#eee', 'text.color': '#eee',
        'xtick.color': '#ccc', 'ytick.color': '#ccc', 'grid.color': '#333', 'grid.alpha': 0.4,
    })
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Vertex AI MaaS -- Gemma 4 26B-A4B Benchmark\n(~20k input, 250 output tokens)',
                 fontsize=18, fontweight='bold', color='white', y=0.98)

    # QPS sweep TTFT
    ax = axes[0,0]
    qps_rates = [d[0] for d in maas_qps]
    ttfts = [d[1] for d in maas_qps]
    ax.plot(qps_rates, ttfts, 'o-', color='#bb86fc', linewidth=2.5, markersize=8, label='MaaS')
    ax.fill_between(qps_rates, [d[2] for d in maas_qps], [d[3] for d in maas_qps], alpha=0.2, color='#bb86fc')
    ax.set_xlabel('Request Rate (QPS)')
    ax.set_ylabel('Mean TTFT (ms)')
    ax.set_title('QPS Sweep: TTFT vs Request Rate', fontweight='bold', color='#ffd700')
    ax.legend(facecolor='#16213e', edgecolor='#444')
    ax.grid(True)

    # QPS sweep E2E latency (mean + P90)
    ax = axes[0,1]
    maas_lat = [d[4] for d in maas_qps]
    maas_lat_p90 = [d[7] for d in maas_qps]
    ax.plot(qps_rates, maas_lat, 'o--', color='#bb86fc', linewidth=1.5, markersize=7, alpha=0.5, label='Mean E2E')
    ax.plot(qps_rates, maas_lat_p90, 'o-', color='#bb86fc', linewidth=2.5, markersize=8, label='P90 E2E')
    ax.axhline(y=3.5, color='#ff6b6b', linestyle='--', alpha=0.3, label='3.5s target')
    ax.set_xlabel('Request Rate (QPS)')
    ax.set_ylabel('E2E Latency (s)')
    ax.set_title('QPS Sweep: E2E Latency (Mean & P90)', fontweight='bold', color='#ffd700')
    ax.legend(facecolor='#16213e', edgecolor='#444')
    ax.grid(True)

    # Burst E2E (mean + P90)
    ax = axes[1,0]
    burst_n = [d[0] for d in maas_burst]
    burst_mean = [d[4] for d in maas_burst]
    burst_p90 = [d[7] for d in maas_burst if d[7] > 0]
    burst_n_p90 = [d[0] for d in maas_burst if d[7] > 0]
    ax.plot(burst_n, burst_mean, 'o--', color='#bb86fc', linewidth=1.5, markersize=7, alpha=0.5, label='Mean E2E')
    ax.plot(burst_n_p90, burst_p90, 'o-', color='#bb86fc', linewidth=2.5, markersize=8, label='P90 E2E')
    ax.axhline(y=3.5, color='#ff6b6b', linestyle='--', alpha=0.3, label='3.5s target')
    ax.set_xlabel('Burst Size (N concurrent)')
    ax.set_ylabel('E2E Latency (s)')
    ax.set_title('Burst Sweep: E2E Latency (Mean & P90)', fontweight='bold', color='#ffd700')
    ax.legend(facecolor='#16213e', edgecolor='#444')
    ax.grid(True)

    # Burst throughput
    ax = axes[1,1]
    burst_tput = [d[5] for d in maas_burst]
    ax.plot(burst_n, burst_tput, 'D-', color='#bb86fc', linewidth=2.5, markersize=8, label='MaaS')
    ax.set_xlabel('Burst Size (N concurrent)')
    ax.set_ylabel('Throughput (req/s)')
    ax.set_title('Burst Sweep: Throughput vs Concurrency', fontweight='bold', color='#ffd700')
    ax.legend(facecolor='#16213e', edgecolor='#444')
    ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('plots/14_maas_benchmark.png', dpi=150, bbox_inches='tight')
    plt.close()
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.edgecolor': 'black', 'axes.labelcolor': 'black', 'text.color': 'black',
        'xtick.color': 'black', 'ytick.color': 'black', 'grid.color': '#ccc', 'grid.alpha': 0.3})
    print("  Plot 14: plots/14_maas_benchmark.png")


# =============================================================================
if __name__ == '__main__':
    total = len(rtx_qps)+len(rtx_burst)+len(tp4_qps)+len(tp4_burst)+len(tpu_qps)+len(tpu_burst)+len(maas_qps)+len(maas_burst)
    print(f"\nGenerating 10 benchmark plots...")
    print(f"  RTX 1×GPU: {len(rtx_qps)} QPS + {len(rtx_burst)} burst = {len(rtx_qps)+len(rtx_burst)} data points")
    print(f"  RTX 4×GPU TP=4: {len(tp4_qps)} QPS + {len(tp4_burst)} burst = {len(tp4_qps)+len(tp4_burst)} data points")
    print(f"  TPU: {len(tpu_qps)} QPS + {len(tpu_burst)} burst = {len(tpu_qps)+len(tpu_burst)} data points")
    print(f"  Vertex AI MaaS: {len(maas_qps)} QPS + {len(maas_burst)} burst = {len(maas_qps)+len(maas_burst)} data points")
    print(f"  Total: {total} data points")
    print("=" * 60)

    plot_01()
    plot_02()
    plot_03()
    plot_04()
    plot_05()
    plot_06()
    plot_08()
    plot_09()
    plot_10()
    plot_14()

    print("=" * 60)
    print("All 10 plots saved to plots/")
    print()
