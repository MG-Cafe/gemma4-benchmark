#!/usr/bin/env python3
"""
GPU benchmark for Gemma 4 26B-A4B with TP=4 (4x RTX Pro 6000).
Matches TPU v6e-8 (TP=8) benchmark methodology.
Uses vllm bench serve with fresh random prompts per run.
Collects P90 E2E, TTFT, TPOT for all scenarios.
"""

import subprocess
import json
import re
import time
import sys
import numpy as np

BASE_URL = "http://localhost:8000"
MODEL = "google/gemma-4-26B-A4B-it"
INPUT_LEN = 20000
OUTPUT_LEN = 250
NUM_PROMPTS = 10  # per QPS level
SEED_BASE = 1000  # different seed per run for fresh prompts

QPS_RATES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
BURST_SIZES = [1, 2, 5, 8, 10, 15, 20, 30]

results = {}

def run_bench(num_prompts, request_rate, seed, label):
    """Run vllm bench serve and parse results."""
    cmd = [
        "vllm", "bench", "serve",
        "--base-url", BASE_URL,
        "--model", MODEL,
        "--dataset-name", "random",
        "--random-input-len", str(INPUT_LEN),
        "--random-output-len", str(OUTPUT_LEN),
        "--num-prompts", str(num_prompts),
        "--request-rate", str(request_rate),
        "--seed", str(seed),
    ]
    print(f"\n  Running: {label} (seed={seed}, n={num_prompts}, rate={request_rate})")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr
        print(f"    Output length: {len(output)} chars")
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT for {label}")
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

    # Parse the output
    data = {}
    for line in output.split('\n'):
        line = line.strip()
        # Match patterns like "Mean TTFT (ms):    825.62"
        if 'Mean TTFT' in line:
            m = re.search(r'[\d.]+$', line)
            if m: data['ttft_mean'] = float(m.group())
        elif 'Median TTFT' in line:
            m = re.search(r'[\d.]+$', line)
            if m: data['ttft_median'] = float(m.group())
        elif 'P99 TTFT' in line:
            m = re.search(r'[\d.]+$', line)
            if m: data['ttft_p99'] = float(m.group())
        elif 'Mean TPOT' in line:
            m = re.search(r'[\d.]+$', line)
            if m: data['tpot_mean'] = float(m.group())
        elif 'Median TPOT' in line:
            m = re.search(r'[\d.]+$', line)
            if m: data['tpot_median'] = float(m.group())
        elif 'P99 TPOT' in line:
            m = re.search(r'[\d.]+$', line)
            if m: data['tpot_p99'] = float(m.group())
        elif 'Mean ITL' in line:
            m = re.search(r'[\d.]+$', line)
            if m: data['itl_mean'] = float(m.group())
        elif 'Output token throughput' in line:
            m = re.search(r'([\d.]+)', line)
            if m: data['output_tok_s'] = float(m.group())
        elif 'Request throughput' in line:
            m = re.search(r'([\d.]+)', line)
            if m: data['req_throughput'] = float(m.group())

    if 'ttft_mean' not in data:
        print(f"    WARNING: Could not parse output for {label}")
        print(f"    Last 20 lines: {output[-2000:]}")
        return None

    # Calculate E2E from TTFT + (N-1)*TPOT
    if 'tpot_mean' in data:
        data['e2e_mean'] = data['ttft_mean']/1000 + (OUTPUT_LEN-1) * data['tpot_mean']/1000
    return data


def run_multiple(num_prompts, request_rate, runs, label_prefix):
    """Run benchmark multiple times and compute P90."""
    all_runs = []
    for i in range(runs):
        seed = SEED_BASE + i * 100
        data = run_bench(num_prompts, request_rate, seed, f"{label_prefix} run {i+1}/{runs}")
        if data:
            all_runs.append(data)
        time.sleep(2)

    if not all_runs:
        return None

    # Compute statistics across runs
    result = {'n_runs': len(all_runs), 'n_prompts': num_prompts}
    for key in ['ttft_mean', 'tpot_mean', 'e2e_mean']:
        vals = [r[key] for r in all_runs if key in r]
        if vals:
            result[f'{key}'] = np.mean(vals)
            result[f'{key.replace("mean","p90")}'] = np.percentile(vals, 90)
            result[f'{key.replace("mean","p99")}'] = np.percentile(vals, 99)

    # Also get per-request E2E P90 (across all requests in all runs)
    all_e2e = []
    for r in all_runs:
        if 'e2e_mean' in r:
            all_e2e.append(r['e2e_mean'])
    if all_e2e:
        result['e2e_mean'] = np.mean(all_e2e)
        result['e2e_p90'] = np.percentile(all_e2e, 90)
        result['e2e_p99'] = np.percentile(all_e2e, 99)

    if 'output_tok_s' in all_runs[0]:
        result['output_tok_s'] = np.mean([r.get('output_tok_s', 0) for r in all_runs])

    return result


def main():
    print("=" * 60)
    print("GPU 4x RTX Pro 6000 (TP=4) Benchmark")
    print("=" * 60)

    # 1. Single request baseline (10 runs)
    print("\n===== SINGLE REQUEST BASELINE (10 runs) =====")
    single = run_multiple(1, float('inf'), 10, "single")
    if single:
        results['single_10runs'] = single
        print(f"  E2E: mean={single['e2e_mean']:.3f}s  P90={single.get('e2e_p90',0):.3f}s")
        print(f"  TTFT: mean={single['ttft_mean']:.1f}ms")

    # 2. QPS sweep (10 runs per rate)
    for qps in QPS_RATES:
        print(f"\n===== QPS={qps} =====")
        data = run_multiple(NUM_PROMPTS, qps, 10, f"qps_{qps}")
        if data:
            results[f'qps_{qps}'] = data
            print(f"  E2E: mean={data['e2e_mean']:.3f}s  P90={data.get('e2e_p90',0):.3f}s")
            print(f"  TTFT: mean={data['ttft_mean']:.1f}ms")

    # 3. Burst sweep (10 runs per size)
    for n in BURST_SIZES:
        print(f"\n===== BURST N={n} =====")
        data = run_multiple(n, float('inf'), 10, f"burst_{n}")
        if data:
            results[f'burst_{n}'] = data
            print(f"  E2E: mean={data['e2e_mean']:.3f}s  P90={data.get('e2e_p90',0):.3f}s")
            print(f"  TTFT: mean={data['ttft_mean']:.1f}ms")

    # Save results
    out_file = "/tmp/gpu-tp4-p90-results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nJSON saved to {out_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY — GPU 4x RTX Pro 6000 (TP=4)")
    print("=" * 60)
    for key, val in results.items():
        e2e_mean = val.get('e2e_mean', 0)
        e2e_p90 = val.get('e2e_p90', 0)
        ttft = val.get('ttft_mean', 0)
        print(f"  {key:20s}: E2E mean={e2e_mean:.3f}s  P90={e2e_p90:.3f}s  TTFT={ttft:.0f}ms")


if __name__ == '__main__':
    main()
