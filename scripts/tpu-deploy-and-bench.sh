#!/bin/bash
# Full TPU deployment + benchmark automation script
# Run this ON the TPU VM via: gcloud compute tpus tpu-vm ssh ... --command='bash /tmp/tpu-deploy-and-bench.sh'

set -e
echo "=== $(date) - Starting TPU deployment + benchmark ==="

# Step 1: Pull vLLM TPU docker image
echo ">>> Step 1: Pulling vLLM TPU docker image..."
sudo docker pull vllm/vllm-tpu:gemma4 2>&1 | tail -3

# Step 2: Start vLLM server
echo ">>> Step 2: Starting vLLM server with max-model-len=128000..."
sudo docker rm -f vllm-gemma4 2>/dev/null || true
sudo docker run -d \
  --name vllm-gemma4 \
  --privileged \
  --network host \
  --shm-size 16G \
  -e HF_TOKEN=${HF_TOKEN} \
  -e VLLM_XLA_CACHE_PATH=/tmp/xla_cache \
  vllm/vllm-tpu:gemma4 \
  --model google/gemma-4-26B-A4B-it \
  --max-model-len 128000 \
  --tensor-parallel-size 4 \
  --disable_chunked_mm_input \
  --port 8000

# Step 3: Wait for server readiness
echo ">>> Step 3: Waiting for vLLM server to be ready..."
MAX_WAIT=1800  # 30 min max
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
  if curl -s http://localhost:8000/v1/models | grep -q "gemma"; then
    echo "  Server READY after ${WAITED}s!"
    break
  fi
  sleep 30
  WAITED=$((WAITED + 30))
  echo "  Waiting... ${WAITED}s elapsed"
  # Check if container is still running
  if ! sudo docker ps | grep -q vllm-gemma4; then
    echo "ERROR: Container stopped! Logs:"
    sudo docker logs vllm-gemma4 2>&1 | tail -20
    exit 1
  fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
  echo "ERROR: Server did not start within ${MAX_WAIT}s"
  sudo docker logs vllm-gemma4 2>&1 | tail -30
  exit 1
fi

# Step 4: Install Python deps
echo ">>> Step 4: Installing Python dependencies..."
pip3 install numpy requests 2>&1 | tail -2

# Step 5: Run benchmark
echo ">>> Step 5: Running benchmark with UNIQUE RANDOM prompts..."
python3 /tmp/tpu-benchmark-128k.py 2>&1

echo "=== $(date) - Benchmark complete! ==="
echo "Results saved to /tmp/tpu-p90-results-128k.txt and /tmp/tpu-p90-results-128k.json"
