#!/usr/bin/env python3
"""
Vertex AI MaaS (Model-as-a-Service) Gemma 4 Benchmark.
Uses the global aiplatform.googleapis.com endpoint (no deployment needed).

Measures TTFT, latency, throughput at various QPS and burst levels.
Uses ~20k input tokens and 250 output tokens for apple-to-apple comparison.

Prerequisites:
    pip install google-auth requests numpy
    gcloud auth application-default login

Usage:
    export PROJECT_ID="your-project-id"
    python3 maas-benchmark.py
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

PROJECT_ID = os.environ.get("PROJECT_ID", "")
if not PROJECT_ID:
    print("ERROR: Set PROJECT_ID env var")
    print("  export PROJECT_ID=your-project-id")
    sys.exit(1)

MAAS_URL = f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/global/endpoints/openapi/chat/completions"
MODEL = "google/gemma-4-26b-a4b-it-maas"
LONG_INPUT = "The quick brown fox jumps over the lazy dog. " * 2000  # ~20k tokens
MAX_TOKENS = 250
NUM_PROMPTS = 10

creds, _ = google.auth.default()


def get_headers():
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return {"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"}


def make_request(headers, stream=True):
    payload = {
        "model": MODEL,
        "stream": stream,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "messages": [{"role": "user", "content": LONG_INPUT}],
    }
    t_start = time.time()
    try:
        if stream:
            resp = requests.post(MAAS_URL, json=payload, headers=headers, stream=True, timeout=600)
            ttft = None
            for line in resp.iter_lines():
                if ttft is None:
                    ttft = time.time() - t_start
            t_end = time.time()
            return {
                "ok": resp.status_code == 200,
                "status": resp.status_code,
                "ttft": ttft or (t_end - t_start),
                "total": t_end - t_start,
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
                "t0": t_start,
                "t1": t_end,
                "err": resp.text[:100] if resp.status_code != 200 else "",
            }
    except Exception as e:
        t_end = time.time()
        return {
            "ok": False, "status": 0,
            "ttft": t_end - t_start, "total": t_end - t_start,
            "t0": t_start, "t1": t_end, "err": str(e)[:80],
        }


def run_qps(qps, n=NUM_PROMPTS):
    headers = get_headers()
    interval = 1.0 / qps
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        futures = []
        for i in range(n):
            if i > 0:
                time.sleep(interval)
            futures.append(ex.submit(make_request, headers))
        return [f.result() for f in concurrent.futures.as_completed(futures)]


def run_burst(n):
    headers = get_headers()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(make_request, headers) for _ in range(n)]
        return [f.result() for f in concurrent.futures.as_completed(futures)]


def report(results, label):
    ok = [r for r in results if r["ok"]]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Successful: {len(ok)}/{len(results)}")
    if not ok:
        for r in results[:3]:
            print(f"    Error: {r.get('err', '')[:100]}")
        return

    ttfts = [r["ttft"] for r in ok]
    tots = [r["total"] for r in ok]
    wall = max(r["t1"] for r in ok) - min(r["t0"] for r in ok)

    print(f"  Mean TTFT (ms):     {np.mean(ttfts)*1000:.2f}")
    print(f"  Median TTFT (ms):   {np.median(ttfts)*1000:.2f}")
    print(f"  P99 TTFT (ms):      {np.percentile(ttfts, 99)*1000:.2f}")
    print(f"  Mean latency (s):   {np.mean(tots):.2f}")
    print(f"  Req throughput:     {len(ok)/wall:.4f} req/s")
    print(f"  Wall time (s):      {wall:.2f}")


def main():
    print("=" * 60)
    print("  Vertex AI MaaS Gemma 4 26B-A4B Benchmark")
    print(f"  Model: {MODEL}")
    print(f"  Input: ~20k tokens, Output: {MAX_TOKENS} tokens")
    print("=" * 60)

    # Warmup
    print("\nWarmup...")
    h = get_headers()
    r = make_request(h, stream=False)
    print(f"  Status={r['status']}, time={r['total']:.2f}s")

    # QPS sweep
    for qps in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]:
        print(f"\nQPS={qps}...")
        results = run_qps(qps)
        report(results, f"QPS={qps}, N={NUM_PROMPTS}")

    # Burst sweep
    for n in [1, 2, 5, 8, 10, 15, 20, 30]:
        print(f"\nBurst N={n}...")
        results = run_burst(n)
        report(results, f"Burst N={n}")

    print("\nDone!")


if __name__ == "__main__":
    main()
