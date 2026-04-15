#!/bin/bash
# Run all vLLM benchmarks for Gemma 4 26B-A4B
# Usage: ./run-benchmarks.sh [gpu|tpu]
#
# Prerequisites:
#   - vLLM server running on localhost:8000
#   - vllm bench serve available

set -euo pipefail

MODEL="google/gemma-4-26B-A4B-it"
INPUT_LEN=20000
OUTPUT_LEN=250
SEED=42
BASE_URL="http://localhost:8000"

echo "============================================================"
echo "  Gemma 4 26B-A4B Benchmark Suite"
echo "  Input: ${INPUT_LEN} tokens, Output: ${OUTPUT_LEN} tokens"
echo "============================================================"

# Check server is up
echo "Checking vLLM server..."
curl -sf "${BASE_URL}/health" > /dev/null || { echo "ERROR: vLLM not running on ${BASE_URL}"; exit 1; }
echo "Server is healthy!"

# Single request baseline
echo ""
echo "=== Single Request Baseline ==="
vllm bench serve \
    --dataset-name random \
    --random-input-len $INPUT_LEN \
    --random-output-len $OUTPUT_LEN \
    --num-prompts 1 \
    --request-rate inf \
    --seed $SEED

# QPS sweep
echo ""
echo "=== QPS Sweep (10 prompts each) ==="
for QPS in 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.7 1.0; do
    echo ""
    echo "--- QPS=$QPS ---"
    vllm bench serve \
        --dataset-name random \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --num-prompts 10 \
        --request-rate $QPS \
        --seed $SEED
done

# Burst sweep
echo ""
echo "=== Burst Sweep (all at once) ==="
for N in 1 2 5 8 10 15 20 30; do
    echo ""
    echo "--- Burst N=$N ---"
    vllm bench serve \
        --dataset-name random \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --num-prompts $N \
        --request-rate inf \
        --seed $SEED
done

echo ""
echo "============================================================"
echo "  All benchmarks complete!"
echo "============================================================"
