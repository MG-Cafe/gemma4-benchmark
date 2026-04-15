#!/usr/bin/env python3
"""
Vertex AI MaaS Gemma 4 Benchmark - Fair Comparison (P90 Edition).

Uses FRESH RANDOM prompts per request to match GPU/TPU benchmark methodology.
This ensures no prefix caching benefits, matching customer's low-cache-hit workload.

~20K input tokens, 250 output tokens, streaming mode.

Usage:
    export PROJECT_ID="gpu-launchpad-playground"
    python3 maas-benchmark-p90.py
"""

import google.auth
import google.auth.transport.requests
import requests
import json
import time
import sys
import os
import random
import string
import concurrent.futures
import numpy as np

PROJECT_ID = os.environ.get("PROJECT_ID", "gpu-launchpad-playground")
MAAS_URL = f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/global/endpoints/openapi/chat/completions"
MODEL = "google/gemma-4-26b-a4b-it-maas"
MAX_TOKENS = 250
NUM_PROMPTS = 10

# Word bank for generating unique random text
WORDS = [
    "algorithm", "benchmark", "compute", "distributed", "efficiency",
    "framework", "gradient", "hardware", "inference", "kernel",
    "latency", "memory", "network", "optimize", "pipeline",
    "quantize", "runtime", "scalable", "throughput", "utilization",
    "vector", "workload", "accelerator", "bandwidth", "compiler",
    "datacenter", "embedding", "function", "generation", "hypothesis",
    "iteration", "jitter", "knowledge", "learning", "model",
    "neural", "operation", "parameter", "query", "reduction",
    "sampling", "tensor", "unrolling", "variable", "weight",
    "architecture", "bottleneck", "capacity", "deployment", "execution",
]

def generate_random_prompt(target_tokens=20000):
    """Generate a unique random prompt of ~target_tokens tokens."""
    # Roughly 1.3 tokens per word, so we need ~15000 words for 20K tokens
    num_words = int(target_tokens / 1.3)
    # Create a unique seed phrase so every prompt is different
    seed = ''.join(random.choices(string.ascii_lowercase, k=20))
    words = [random.choice(WORDS) for _ in range(num_words)]
    # Add unique identifier every 100 words to prevent any prefix matching
    result_parts = []
    for i in range(0, len(words), 100):
        chunk = ' '.join(words[i:i+100])
        uid = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
        result_parts.append(f"[{uid}] {chunk}")
    text = f"Analyze the following technical document (ref:{seed}):\n\n" + '\n'.join(result_parts)
    text += f"\n\nProvide a detailed analysis of the key themes and patterns in this document."
    return text


creds, _ = google.auth.default()

def get_headers():
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return {"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"}


def make_request(prompt, headers, stream=True):
    """Send a single request with the given unique prompt."""
    payload = {
        "model": MODEL,
        "stream": stream,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,  # Non-zero to get varied outputs
        "messages": [{"role": "user", "content": prompt}],
    }
    t_start = time.time()
    try:
        if stream:
            resp = requests.post(MAAS_URL, json=payload, headers=headers, stream=True, timeout=600)
            ttft = None
            total_tokens = 0
            token_times = []
            for line in resp.iter_lines():
                t_now = time.time()
                if ttft is None:
                    ttft = t_now - t_start
                if line:
                    line_str = line.decode('utf-8', errors='ignore')
                    if line_str.startswith('data: ') and line_str != 'data: [DONE]':
                        total_tokens += 1
                        token_times.append(t_now)
            t_end = time.time()
            
            # Calculate TPOT from token times
            tpot = None
            if len(token_times) >= 3:
                inter_token = [token_times[i+1] - token_times[i] for i in range(1, len(token_times)-1)]
                if inter_token:
                    tpot = np.mean(inter_token) * 1000  # ms
            
            return {
                "ok": resp.status_code == 200,
                "status": resp.status_code,
                "ttft": ttft or (t_end - t_start),
                "total": t_end - t_start,
                "tpot": tpot,
                "tokens": total_tokens,
                "t0": t_start,
                "t1": t_end,
            }
        else:
            resp = requests.post(MAAS_URL, json=payload, headers=headers, timeout=600)
            t_end = time.time()
            return {
                "ok": resp.status_code == 200,
                "status": resp.status_code,
                "ttft": t_end - t_start,
                "total": t_end - t_start,
                "tpot": None,
                "tokens": 0,
                "t0": t_start,
                "t1": t_end,
            }
    except Exception as e:
        t_end = time.time()
        return {"ok": False, "status": -1, "ttft": t_end - t_start,
                "total": t_end - t_start, "tpot": None, "tokens": 0, "t0": t_start, "t1": t_end}


def percentile(data, p):
    if len(data) < 2:
        return 0.0
    return float(np.percentile(data, p))


def run_batch(prompts, headers, delay=0):
    """Run batch with optional delay between requests."""
    results = []
    if delay > 0:
        # Sequential with delay (QPS mode)
        for p in prompts:
            r = make_request(p, headers)
            results.append(r)
            time.sleep(delay)
    else:
        # All at once (burst mode)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
            futures = [ex.submit(make_request, p, headers) for p in prompts]
            results = [f.result() for f in futures]
    return results


def print_stats(label, results):
    ok = [r for r in results if r["ok"]]
    if not ok:
        print(f"  {label}: ALL FAILED")
        return
    e2e = [r["total"] for r in ok]
    ttfts = [r["ttft"] * 1000 for r in ok]  # convert to ms
    tpots = [r["tpot"] for r in ok if r.get("tpot")]
    
    print(f"  ===== {label} =====")
    print(f"  Requests: {len(ok)}/{len(results)}")
    print(f"  E2E:  mean={np.mean(e2e):.3f}s  med={np.median(e2e):.3f}s  "
          f"P90={percentile(e2e, 90):.3f}s  P99={percentile(e2e, 99):.3f}s  "
          f"min={min(e2e):.3f}s  max={max(e2e):.3f}s")
    print(f"  TTFT: mean={np.mean(ttfts):.1f}ms  med={np.median(ttfts):.1f}ms  "
          f"P90={percentile(ttfts, 90):.1f}ms  P99={percentile(ttfts, 99):.1f}ms")
    if tpots:
        print(f"  TPOT: mean={np.mean(tpots):.2f}ms  med={np.median(tpots):.2f}ms  "
              f"P90={percentile(tpots, 90):.2f}ms")
    
    return {
        "label": label,
        "n": len(ok),
        "e2e_mean": float(np.mean(e2e)),
        "e2e_p90": float(percentile(e2e, 90)),
        "e2e_p99": float(percentile(e2e, 99)),
        "ttft_mean": float(np.mean(ttfts)),
        "ttft_p90": float(percentile(ttfts, 90)),
        "tpot_mean": float(np.mean(tpots)) if tpots else None,
        "tpot_p90": float(percentile(tpots, 90)) if tpots else None,
    }


def main():
    print("=" * 70)
    print("Vertex AI MaaS Benchmark - Fair Comparison (Fresh Random Prompts)")
    print(f"Model: {MODEL}")
    print(f"Input: ~20K tokens (unique random per request)")
    print(f"Output: {MAX_TOKENS} tokens")
    print(f"Requests per test: {NUM_PROMPTS}")
    print("=" * 70)
    sys.stdout.flush()
    
    headers = get_headers()
    all_results = []
    
    # Phase 1: Single request baseline (10 cold runs)
    print("\n>>> PHASE 1: Single Request Baseline (10 cold runs)")
    sys.stdout.flush()
    single_results = []
    for i in range(10):
        prompt = generate_random_prompt()
        r = make_request(prompt, headers)
        single_results.append(r)
        print(f"  Run {i+1}/10: E2E={r['total']:.3f}s  TTFT={r['ttft']*1000:.1f}ms  "
              f"TPOT={r.get('tpot', 'N/A')}{'ms' if r.get('tpot') else ''}")
        sys.stdout.flush()
        time.sleep(2)
    s = print_stats("SINGLE REQUEST (10 runs)", single_results)
    if s: all_results.append(s)
    
    # Phase 2: QPS sweep
    print("\n>>> PHASE 2: QPS Sweep")
    sys.stdout.flush()
    qps_levels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
    
    for qps in qps_levels:
        delay = 1.0 / qps
        # Generate fresh prompts for each QPS level
        prompts = [generate_random_prompt() for _ in range(NUM_PROMPTS)]
        headers = get_headers()  # refresh token
        
        results = []
        for i, p in enumerate(prompts):
            r = make_request(p, headers)
            results.append(r)
            if i < len(prompts) - 1:
                time.sleep(delay)
        
        s = print_stats(f"QPS={qps}", results)
        if s: all_results.append(s)
        sys.stdout.flush()
        time.sleep(5)
    
    # Phase 3: Burst sweep
    print("\n>>> PHASE 3: Burst Sweep")
    sys.stdout.flush()
    burst_sizes = [1, 2, 5, 8, 10, 15, 20, 30]
    
    for n in burst_sizes:
        # Generate fresh prompts for each burst
        prompts = [generate_random_prompt() for _ in range(n)]
        headers = get_headers()  # refresh token
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
            futures = [ex.submit(make_request, p, headers) for p in prompts]
            results = [f.result() for f in futures]
        
        s = print_stats(f"BURST N={n}", results)
        if s: all_results.append(s)
        sys.stdout.flush()
        time.sleep(10)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - P90 E2E")
    print("=" * 70)
    for r in all_results:
        line = f"{r['label']:20s}  mean={r['e2e_mean']:.3f}s  P90={r['e2e_p90']:.3f}s"
        line += f"  TTFT_mean={r['ttft_mean']:.1f}ms"
        if r.get('tpot_mean'):
            line += f"  TPOT_mean={r['tpot_mean']:.2f}ms"
        print(line)
    
    # Save JSON
    with open("/tmp/maas-p90-results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON saved to /tmp/maas-p90-results.json")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
