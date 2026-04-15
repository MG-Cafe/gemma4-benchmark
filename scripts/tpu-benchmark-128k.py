#!/usr/bin/env python3
"""
TPU v6e-8 Benchmark for Gemma 4 26B-A4B using vLLM + OpenAI API.
128K context length benchmark (max-model-len=128000).
Runs single-request baseline, QPS sweep, and burst sweep.
Outputs structured P90 results matching the existing report format.
"""

import requests, json, time, sys, os
import concurrent.futures
import numpy as np

BASE_URL = "http://localhost:8000/v1"
MODEL = "google/gemma-4-26B-A4B-it"
LONG_INPUT = "The quick brown fox jumps over the lazy dog. " * 2000  # ~20k tokens
MAX_TOKENS = 250

def make_request(stream=True):
    """Send one chat completion request, measure TTFT and total time."""
    payload = {
        "model": MODEL,
        "stream": stream,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "messages": [{"role": "user", "content": LONG_INPUT}]
    }
    t_start = time.time()
    try:
        if stream:
            resp = requests.post(f"{BASE_URL}/chat/completions",
                                 json=payload, stream=True, timeout=600)
            ttft = None
            tokens = 0
            for line in resp.iter_lines():
                if line:
                    if ttft is None:
                        ttft = time.time() - t_start
                    decoded = line.decode('utf-8', errors='ignore')
                    if decoded.startswith('data: ') and '[DONE]' not in decoded:
                        tokens += 1
            t_end = time.time()
            total = t_end - t_start
            tpot = ((total - (ttft or total)) / max(tokens - 1, 1)) * 1000 if tokens > 1 else 0
            return {
                'ok': resp.status_code == 200,
                'ttft': (ttft or total) * 1000,
                'total': total,
                'tpot': tpot,
                'tokens': tokens,
                't0': t_start, 't1': t_end,
            }
        else:
            resp = requests.post(f"{BASE_URL}/chat/completions",
                                 json=payload, timeout=600)
            t_end = time.time()
            total = t_end - t_start
            data = resp.json()
            tokens = data.get('usage', {}).get('completion_tokens', 0)
            return {
                'ok': resp.status_code == 200,
                'ttft': total * 1000,
                'total': total,
                'tpot': 0,
                'tokens': tokens,
                't0': t_start, 't1': t_end,
            }
    except Exception as e:
        t_end = time.time()
        return {'ok': False, 'ttft': (t_end - t_start) * 1000,
                'total': t_end - t_start, 'tpot': 0, 'tokens': 0,
                't0': t_start, 't1': t_end, 'err': str(e)[:80]}


def run_qps(qps, n=10):
    results = []
    interval = 1.0 / qps if qps > 0 else 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        futures = []
        for i in range(n):
            if i > 0:
                time.sleep(interval)
            futures.append(ex.submit(make_request))
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    return results


def run_burst(n):
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(make_request) for _ in range(n)]
        return [f.result() for f in concurrent.futures.as_completed(futures)]


def stats_line(results, label):
    ok = [r for r in results if r['ok']]
    if not ok:
        return f"  ===== {label} =====\n  ERROR: All requests failed\n"

    e2e = [r['total'] for r in ok]
    ttfts = [r['ttft'] for r in ok]
    tpots = [r['tpot'] for r in ok if r['tpot'] > 0]

    lines = []
    lines.append(f"  ===== {label} =====")
    lines.append(f"  Requests: {len(ok)}")
    lines.append(f"  E2E:  mean={np.mean(e2e):.3f}s  med={np.median(e2e):.3f}s  P90={np.percentile(e2e,90):.3f}s  P99={np.percentile(e2e,99):.3f}s  min={np.min(e2e):.3f}s  max={np.max(e2e):.3f}s")
    lines.append(f"  TTFT: mean={np.mean(ttfts):.1f}ms  med={np.median(ttfts):.1f}ms  P90={np.percentile(ttfts,90):.1f}ms  P99={np.percentile(ttfts,99):.1f}ms")
    if tpots:
        lines.append(f"  TPOT: mean={np.mean(tpots):.2f}ms  med={np.median(tpots):.2f}ms  P90={np.percentile(tpots,90):.2f}ms")
    return "\n".join(lines)


def summary_line(results, label):
    ok = [r for r in results if r['ok']]
    if not ok:
        return f"{label:20s}  ERROR"
    e2e = [r['total'] for r in ok]
    ttfts = [r['ttft'] for r in ok]
    tpots = [r['tpot'] for r in ok if r['tpot'] > 0]
    tpot_mean = np.mean(tpots) if tpots else 0
    return f"{label:20s}  mean={np.mean(e2e):.3f}s  P90={np.percentile(e2e,90):.3f}s  TTFT_mean={np.mean(ttfts):.1f}ms  TPOT_mean={tpot_mean:.2f}ms"


def main():
    output = []

    def p(s=""):
        print(s)
        output.append(s)

    p("=" * 70)
    p("TPU v6e-8 Benchmark - P90 E2E (128K context, BF16)")
    p(f"Input: ~20000 tokens, Output: {MAX_TOKENS} tokens")
    p("=" * 70)

    # Check server
    try:
        r = requests.get(f"{BASE_URL}/models", timeout=10)
        models = [m['id'] for m in r.json().get('data', [])]
        p(f"\nServer OK, models: {models}")
    except Exception as e:
        p(f"\nERROR: Cannot reach vLLM at {BASE_URL}: {e}")
        sys.exit(1)

    # Warmup
    p("\nWarmup...")
    r = make_request(stream=False)
    p(f"  status={r['ok']} time={r['total']:.2f}s tokens={r['tokens']}")

    all_results = {}

    # Phase 1: Single request baseline
    p("\n>>> PHASE 1: Single Request (10 runs, fresh prompts)")
    single_results = []
    for i in range(10):
        r = make_request()
        single_results.append(r)
        if r['ok']:
            p(f"  Run {i+1}/10: E2E={r['total']:.3f}s  TTFT={r['ttft']:.1f}ms  TPOT={r['tpot']:.2f}ms")
        else:
            p(f"  Run {i+1}/10: FAILED - {r.get('err','?')[:60]}")
    p("")
    p(stats_line(single_results, "SINGLE REQUEST (10 runs)"))
    all_results['single_10runs'] = single_results

    # Phase 2: QPS sweep
    p("\n>>> PHASE 2: QPS Sweep")
    for qps in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]:
        p("")
        results = run_qps(qps)
        p(stats_line(results, f"QPS={qps}"))
        all_results[f'qps_{qps}'] = results

    # Phase 3: Burst sweep
    p("\n>>> PHASE 3: Burst Sweep")
    for n in [1, 2, 5, 8, 10, 15, 20]:
        p("")
        results = run_burst(n)
        p(stats_line(results, f"BURST N={n}"))
        all_results[f'burst_{n}'] = results

    # Summary
    p("")
    p("=" * 70)
    p("SUMMARY - P90 E2E")
    p("=" * 70)
    for key in ['single_10runs'] + [f'qps_{q}' for q in [0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.7,1.0]] + [f'burst_{n}' for n in [1,2,5,8,10,15,20]]:
        p(summary_line(all_results[key], key))

    # Save results
    outfile = "/tmp/tpu-p90-results-128k.txt"
    with open(outfile, 'w') as f:
        f.write("\n".join(output) + "\n")
    p(f"\nResults saved to {outfile}")

    # Also save JSON
    json_out = {}
    for key, results in all_results.items():
        ok = [r for r in results if r['ok']]
        if ok:
            e2e = [r['total'] for r in ok]
            ttfts = [r['ttft'] for r in ok]
            tpots = [r['tpot'] for r in ok if r['tpot'] > 0]
            json_out[key] = {
                'e2e_mean': float(np.mean(e2e)),
                'e2e_p90': float(np.percentile(e2e, 90)),
                'e2e_p99': float(np.percentile(e2e, 99)),
                'ttft_mean': float(np.mean(ttfts)),
                'ttft_p90': float(np.percentile(ttfts, 90)),
                'tpot_mean': float(np.mean(tpots)) if tpots else 0,
                'tpot_p90': float(np.percentile(tpots, 90)) if tpots else 0,
                'n_ok': len(ok),
                'n_total': len(results),
            }
    json_file = "/tmp/tpu-p90-results-128k.json"
    with open(json_file, 'w') as f:
        json.dump(json_out, f, indent=2)
    p(f"JSON saved to {json_file}")


if __name__ == '__main__':
    main()
