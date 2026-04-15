#!/usr/bin/env python3
"""
Vertex AI Gemma 4 Benchmark
Measures TTFT, latency, throughput at various QPS and burst levels.
Uses ~20k input tokens and 250 output tokens.

Prerequisites:
    pip install google-auth requests numpy
    gcloud auth application-default login

Usage:
    # Set your endpoint URL (from Model Garden deployment)
    export VERTEX_ENDPOINT_URL="https://YOUR-ENDPOINT.prediction.vertexai.goog"
    export VERTEX_PROJECT_ID="your-project-id"
    export VERTEX_ENDPOINT_ID="your-endpoint-id"
    python3 vertex-ai-benchmark.py
"""

import google.auth
import google.auth.transport.requests
import requests
import json
import time
import sys
import os
import concurrent.futures
import numpy as np

# Config — set via environment variables
ENDPOINT_URL = os.environ.get("VERTEX_ENDPOINT_URL", "")
PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID", "")
ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID", "")

if not all([ENDPOINT_URL, PROJECT_ID, ENDPOINT_ID]):
    print("ERROR: Set VERTEX_ENDPOINT_URL, VERTEX_PROJECT_ID, VERTEX_ENDPOINT_ID env vars")
    print("Example:")
    print('  export VERTEX_ENDPOINT_URL="https://mg-endpoint-xxx.us-central1-xxx.prediction.vertexai.goog"')
    print('  export VERTEX_PROJECT_ID="your-project-id"')
    print('  export VERTEX_ENDPOINT_ID="your-endpoint-id"')
    sys.exit(1)

PREDICT_URL = f"{ENDPOINT_URL}/v1/projects/{PROJECT_ID}/locations/us-central1/endpoints/{ENDPOINT_ID}:rawPredict"
LONG_INPUT = "The quick brown fox jumps over the lazy dog. " * 2000  # ~20k tokens
SHORT_INPUT = "Hello, explain quantum computing briefly."
MAX_OUTPUT_TOKENS = 250
NUM_PROMPTS_QPS = 10


def get_auth_headers():
    creds, _ = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return {"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"}


def make_request(headers, input_text, max_tokens, stream=False):
    """Send a single request, return timing dict."""
    payload = {
        "prompt": input_text,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }

    t_start = time.time()
    try:
        if stream:
            resp = requests.post(PREDICT_URL, json=payload, headers=headers, stream=True, timeout=600)
            ttft = None
            for line in resp.iter_lines():
                if ttft is None:
                    ttft = time.time() - t_start
            t_end = time.time()
            return {
                "success": resp.status_code == 200,
                "status": resp.status_code,
                "ttft": ttft if ttft else t_end - t_start,
                "total_time": t_end - t_start,
                "t_start": t_start,
                "t_end": t_end,
            }
        else:
            resp = requests.post(PREDICT_URL, json=payload, headers=headers, timeout=600)
            t_end = time.time()
            return {
                "success": resp.status_code == 200,
                "status": resp.status_code,
                "ttft": t_end - t_start,
                "total_time": t_end - t_start,
                "t_start": t_start,
                "t_end": t_end,
                "response": resp.text[:200] if resp.status_code != 200 else "",
            }
    except Exception as e:
        t_end = time.time()
        return {"success": False, "status": 0, "ttft": t_end - t_start,
                "total_time": t_end - t_start, "t_start": t_start, "t_end": t_end, "error": str(e)}


def run_qps_benchmark(headers, qps, num_prompts, input_text, max_tokens):
    results = []
    interval = 1.0 / qps if qps > 0 else 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_prompts) as executor:
        futures = []
        for i in range(num_prompts):
            if i > 0:
                time.sleep(interval)
            futures.append(executor.submit(make_request, headers, input_text, max_tokens, stream=True))
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    return results


def run_burst_benchmark(headers, num_prompts, input_text, max_tokens):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_prompts) as executor:
        futures = [executor.submit(make_request, headers, input_text, max_tokens, stream=True)
                   for _ in range(num_prompts)]
        return [f.result() for f in concurrent.futures.as_completed(futures)]


def print_results(results, label):
    successful = [r for r in results if r["success"]]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Successful: {len(successful)}/{len(results)}")
    if not successful:
        for r in results[:3]:
            print(f"    Error: {r.get('error', r.get('response', ''))[:100]}")
        return

    ttfts = [r["ttft"] for r in successful]
    total_times = [r["total_time"] for r in successful]
    wall_start = min(r["t_start"] for r in successful)
    wall_end = max(r["t_end"] for r in successful)
    wall_time = wall_end - wall_start

    print(f"  Mean TTFT (ms):     {np.mean(ttfts)*1000:.2f}")
    print(f"  Median TTFT (ms):   {np.median(ttfts)*1000:.2f}")
    print(f"  P99 TTFT (ms):      {np.percentile(ttfts, 99)*1000:.2f}")
    print(f"  Mean latency (s):   {np.mean(total_times):.2f}")
    print(f"  Req throughput:     {len(successful)/wall_time:.4f} req/s")
    print(f"  Wall time (s):      {wall_time:.2f}")


def wait_for_endpoint(headers, max_wait=600):
    print("Waiting for endpoint to be ready...")
    start = time.time()
    while time.time() - start < max_wait:
        payload = {"prompt": "Hi", "max_tokens": 1}
        try:
            resp = requests.post(PREDICT_URL, json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                print(f"  Ready! ({time.time()-start:.0f}s)")
                return True
            elif resp.status_code == 429 and "scale-up" in resp.text:
                print(f"  {time.time()-start:.0f}s - Scaling up...")
            else:
                print(f"  Status {resp.status_code}: {resp.text[:80]}")
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(15)
    return False


def main():
    print("=" * 60)
    print("  Vertex AI Gemma 4 Benchmark")
    print(f"  Input: ~20k tokens, Output: {MAX_OUTPUT_TOKENS} tokens")
    print("=" * 60)

    headers = get_auth_headers()
    if not wait_for_endpoint(headers):
        sys.exit(1)

    headers = get_auth_headers()
    print("\nWarmup...")
    r = make_request(headers, SHORT_INPUT, 10, stream=True)
    print(f"  Status={r['status']}, time={r['total_time']:.2f}s")

    # QPS sweep
    for qps in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]:
        headers = get_auth_headers()
        print(f"\nQPS={qps}...")
        results = run_qps_benchmark(headers, qps, NUM_PROMPTS_QPS, LONG_INPUT, MAX_OUTPUT_TOKENS)
        print_results(results, f"QPS={qps}, N=10")

    # Burst sweep
    for n in [1, 2, 5, 8, 10, 15, 20, 30]:
        headers = get_auth_headers()
        print(f"\nBurst N={n}...")
        results = run_burst_benchmark(headers, n, LONG_INPUT, MAX_OUTPUT_TOKENS)
        print_results(results, f"Burst N={n}")

    print("\nDone!")


if __name__ == "__main__":
    main()
