#!/usr/bin/env python3
"""
TPU v6e-8 Benchmark for Gemma 4 26B-A4B using vLLM + OpenAI API.
Runs QPS sweep and burst sweep matching GPU/MaaS benchmarks.

Deploy on TPU VM:
  1. Create TPU v6e-8: gcloud compute tpus tpu-vm create gemma4-tpu-bench \
       --zone=us-east7-ai1b --accelerator-type=v6e-8 --version=v2-alpha-tpuv6e
  2. Pull Docker image: docker pull vllm/vllm-tpu:gemma4
  3. Run vLLM container:
       docker run -d --name vllm-gemma4 --privileged --net=host \
         -v /dev/shm:/dev/shm --shm-size 16g \
         -e "VLLM_ARGS=--model google/gemma-4-26b-a4b-it \
             --max-model-len 32768 --tensor-parallel-size 8 \
             --disable_chunked_mm_input" \
         vllm/vllm-tpu:gemma4
  4. python3 tpu-benchmark.py
"""

import requests, json, time, sys
import concurrent.futures
import numpy as np

BASE_URL = "http://localhost:8000/v1"
MODEL = "google/gemma-4-26b-a4b-it"
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
                'ttft': (ttft or total) * 1000,  # ms
                'total': total,
                'tpot': tpot,  # ms
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
    """Send n requests at the given rate."""
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
    """Send n requests all at once."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(make_request) for _ in range(n)]
        return [f.result() for f in concurrent.futures.as_completed(futures)]


def report(results, label):
    ok = [r for r in results if r['ok']]
    print(f'{label}: {len(ok)}/{len(results)} ok', end='')
    if not ok:
        errs = [r.get('err', '?')[:50] for r in results[:2]]
        print(f' ERR: {errs}')
        return
    ttfts = [r['ttft'] for r in ok]
    tots = [r['total'] for r in ok]
    tpots = [r['tpot'] for r in ok if r['tpot'] > 0]
    wall = max(r['t1'] for r in ok) - min(r['t0'] for r in ok)
    tput = len(ok) / wall if wall > 0 else 0
    tokens = sum(r['tokens'] for r in ok)
    tok_s = tokens / wall if wall > 0 else 0

    print(f' | TTFT mean={np.mean(ttfts):.1f} med={np.median(ttfts):.1f} p99={np.percentile(ttfts,99):.1f}ms', end='')
    if tpots:
        print(f' | TPOT mean={np.mean(tpots):.2f}ms', end='')
    print(f' | lat={np.mean(tots):.2f}s | tput={tput:.3f}req/s tok/s={tok_s:.1f}')


def main():
    print('=== TPU v6e-8 Gemma 4 26B-A4B Benchmark ===')
    print(f'Input: ~20k tokens, Output: {MAX_TOKENS} tokens')
    print()

    # Check server
    try:
        r = requests.get(f"{BASE_URL}/models", timeout=5)
        models = [m['id'] for m in r.json().get('data', [])]
        print(f'Server OK, models: {models}')
    except Exception as e:
        print(f'ERROR: Cannot reach vLLM at {BASE_URL}: {e}')
        sys.exit(1)

    # Warmup
    print('\nWarmup...')
    r = make_request(stream=False)
    print(f'  status={r["ok"]} time={r["total"]:.2f}s tokens={r["tokens"]}')

    # QPS sweep
    print('\n=== QPS Sweep (N=10 each) ===')
    for qps in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]:
        results = run_qps(qps)
        report(results, f'QPS={qps}')

    # Burst sweep
    print('\n=== Burst Sweep ===')
    for n in [1, 2, 5, 8, 10, 15, 20, 30]:
        results = run_burst(n)
        report(results, f'Burst N={n}')

    print('\nDone!')


if __name__ == '__main__':
    main()
