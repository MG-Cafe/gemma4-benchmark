#!/usr/bin/env python3
"""
GPU Benchmark with P90 E2E Capture
Sends requests to vLLM server and captures per-request E2E, TTFT, TPOT.
Reports mean, median, P90, P99 for all metrics.

Usage:
    python3 gpu-benchmark-p90.py [--host localhost --port 8000]
"""
import argparse
import asyncio
import json
import random
import string
import time
import statistics
import aiohttp
from dataclasses import dataclass, field
from typing import List

@dataclass
class RequestResult:
    e2e_s: float = 0.0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    output_tokens: int = 0
    success: bool = False

def generate_random_text(num_tokens: int) -> str:
    """Generate ~num_tokens of random text (rough approximation: 4 chars per token)."""
    chars_needed = num_tokens * 4
    words = []
    for _ in range(num_tokens):
        word_len = random.randint(3, 8)
        words.append(''.join(random.choices(string.ascii_lowercase, k=word_len)))
    return ' '.join(words)

async def send_request(session, url, prompt, max_tokens, seed=None):
    """Send a single streaming request and measure TTFT, TPOT, E2E."""
    payload = {
        "model": "google/gemma-4-26B-A4B-it",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    if seed is not None:
        payload["seed"] = seed

    result = RequestResult()
    start_time = time.perf_counter()
    first_token_time = None
    token_times = []
    token_count = 0

    try:
        async with session.post(url, json=payload) as resp:
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: ') and line != 'data: [DONE]':
                    now = time.perf_counter()
                    try:
                        data = json.loads(line[6:])
                        choices = data.get('choices', [])
                        if choices and choices[0].get('text', ''):
                            token_count += 1
                            if first_token_time is None:
                                first_token_time = now
                            else:
                                token_times.append(now)
                    except json.JSONDecodeError:
                        pass

        end_time = time.perf_counter()
        result.e2e_s = end_time - start_time
        result.output_tokens = token_count

        if first_token_time is not None:
            result.ttft_ms = (first_token_time - start_time) * 1000

        if len(token_times) >= 2:
            inter_token_latencies = []
            prev = first_token_time
            for t in token_times:
                inter_token_latencies.append((t - prev) * 1000)
                prev = t
            result.tpot_ms = statistics.mean(inter_token_latencies)

        result.success = True
    except Exception as e:
        print(f"  Request failed: {e}")
        result.e2e_s = time.perf_counter() - start_time

    return result

async def run_scenario(url, prompt, max_tokens, num_requests, request_rate, seed=None, label=""):
    """Run a benchmark scenario and return results."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Requests: {num_requests}, Rate: {request_rate} QPS, Seed: {seed}")
    print(f"{'='*70}")

    connector = aiohttp.TCPConnector(limit=100)
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []

        if request_rate == float('inf') or request_rate <= 0:
            # Burst: send all at once
            for i in range(num_requests):
                tasks.append(send_request(session, url, prompt, max_tokens, seed))
            results = await asyncio.gather(*tasks)
        else:
            # QPS: send at specified rate
            results = []
            interval = 1.0 / request_rate
            for i in range(num_requests):
                task = asyncio.create_task(send_request(session, url, prompt, max_tokens, seed))
                tasks.append(task)
                if i < num_requests - 1:
                    await asyncio.sleep(interval)
            results = await asyncio.gather(*tasks)

    results = [r for r in results if r.success]
    return results

def compute_percentile(values, p):
    """Compute p-th percentile."""
    if not values:
        return 0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1

def print_results(results: List[RequestResult], label: str):
    """Print statistics for a set of results."""
    if not results:
        print("  No successful results!")
        return

    e2e_vals = [r.e2e_s for r in results]
    ttft_vals = [r.ttft_ms for r in results]
    tpot_vals = [r.tpot_ms for r in results if r.tpot_ms > 0]
    output_toks = [r.output_tokens for r in results]

    print(f"\n  ============ {label} ============")
    print(f"  Successful requests:        {len(results)}")
    print(f"  Mean output tokens:         {statistics.mean(output_toks):.0f}")

    print(f"\n  --------------- E2E Latency (seconds) ----------------")
    print(f"  Mean E2E:                   {statistics.mean(e2e_vals):.3f}s")
    print(f"  Median E2E:                 {statistics.median(e2e_vals):.3f}s")
    print(f"  P90 E2E:                    {compute_percentile(e2e_vals, 90):.3f}s")
    print(f"  P99 E2E:                    {compute_percentile(e2e_vals, 99):.3f}s")
    print(f"  Min E2E:                    {min(e2e_vals):.3f}s")
    print(f"  Max E2E:                    {max(e2e_vals):.3f}s")

    print(f"\n  --------------- Time to First Token (ms) ----------------")
    print(f"  Mean TTFT:                  {statistics.mean(ttft_vals):.2f}ms")
    print(f"  Median TTFT:                {statistics.median(ttft_vals):.2f}ms")
    print(f"  P90 TTFT:                   {compute_percentile(ttft_vals, 90):.2f}ms")
    print(f"  P99 TTFT:                   {compute_percentile(ttft_vals, 99):.2f}ms")

    if tpot_vals:
        print(f"\n  --------------- Time per Output Token (ms) ----------------")
        print(f"  Mean TPOT:                  {statistics.mean(tpot_vals):.2f}ms")
        print(f"  Median TPOT:                {statistics.median(tpot_vals):.2f}ms")
        print(f"  P90 TPOT:                   {compute_percentile(tpot_vals, 90):.2f}ms")
        print(f"  P99 TPOT:                   {compute_percentile(tpot_vals, 99):.2f}ms")

    print(f"  ============================================\n")

    return {
        'e2e_mean': statistics.mean(e2e_vals),
        'e2e_median': statistics.median(e2e_vals),
        'e2e_p90': compute_percentile(e2e_vals, 90),
        'e2e_p99': compute_percentile(e2e_vals, 99),
        'ttft_mean': statistics.mean(ttft_vals),
        'ttft_median': statistics.median(ttft_vals),
        'ttft_p90': compute_percentile(ttft_vals, 90),
        'ttft_p99': compute_percentile(ttft_vals, 99),
        'tpot_mean': statistics.mean(tpot_vals) if tpot_vals else 0,
        'tpot_p90': compute_percentile(tpot_vals, 90) if tpot_vals else 0,
        'num_requests': len(results),
    }

async def main():
    parser = argparse.ArgumentParser(description='GPU Benchmark with P90 E2E')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--input-tokens', type=int, default=20000)
    parser.add_argument('--output-tokens', type=int, default=250)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/completions"
    prompt = generate_random_text(args.input_tokens)
    max_tokens = args.output_tokens

    print(f"=" * 70)
    print(f"GPU Benchmark — P90 E2E Capture")
    print(f"Server: {args.host}:{args.port}")
    print(f"Input: ~{args.input_tokens} tokens, Output: {max_tokens} tokens")
    print(f"=" * 70)

    all_results = {}

    # 1. Single request (cold, no seed — 3 runs for P90)
    print("\n\n>>> PHASE 1: Single Request Baseline (5 runs, no seed, cold)")
    results = await run_scenario(url, prompt, max_tokens, 5, float('inf'),
                                  seed=None, label="Single Request Baseline (5 runs)")
    # Wait between each single request to avoid caching
    single_results = []
    for i in range(5):
        fresh_prompt = generate_random_text(args.input_tokens)
        r = await run_scenario(url, fresh_prompt, max_tokens, 1, float('inf'),
                                seed=None, label=f"Single Request #{i+1} (fresh prompt)")
        single_results.extend(r)
        await asyncio.sleep(2)
    all_results['single'] = print_results(single_results, "SINGLE REQUEST BASELINE (5 runs, cold)")

    # 2. QPS Sweep (10 requests per rate, seed=42 for prefix caching comparison)
    print("\n\n>>> PHASE 2: QPS Sweep (10 requests per rate)")
    qps_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
    for qps in qps_rates:
        # Use fresh prompt each time to avoid prefix caching
        fresh_prompt = generate_random_text(args.input_tokens)
        results = await run_scenario(url, fresh_prompt, max_tokens, 10, qps,
                                      seed=None, label=f"QPS = {qps}")
        all_results[f'qps_{qps}'] = print_results(results, f"QPS = {qps}")
        await asyncio.sleep(5)

    # 3. Burst Sweep
    print("\n\n>>> PHASE 3: Burst Sweep (all requests at once)")
    burst_sizes = [1, 2, 5, 8, 10, 15, 20, 30]
    for n in burst_sizes:
        fresh_prompt = generate_random_text(args.input_tokens)
        results = await run_scenario(url, fresh_prompt, max_tokens, n, float('inf'),
                                      seed=None, label=f"Burst N={n}")
        all_results[f'burst_{n}'] = print_results(results, f"BURST N={n}")
        await asyncio.sleep(10)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — P90 E2E Results")
    print("=" * 70)
    print(f"{'Scenario':<25} {'Mean E2E':>10} {'P90 E2E':>10} {'P99 E2E':>10} {'Mean TTFT':>12} {'P90 TTFT':>12} {'Mean TPOT':>12}")
    print("-" * 95)
    for key, stats in all_results.items():
        if stats:
            print(f"{key:<25} {stats['e2e_mean']:>9.3f}s {stats['e2e_p90']:>9.3f}s {stats['e2e_p99']:>9.3f}s {stats['ttft_mean']:>10.1f}ms {stats['ttft_p90']:>10.1f}ms {stats['tpot_mean']:>10.2f}ms")

    # Save raw JSON
    with open('/tmp/gpu-benchmark-p90-results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to /tmp/gpu-benchmark-p90-results.json")

if __name__ == '__main__':
    asyncio.run(main())
