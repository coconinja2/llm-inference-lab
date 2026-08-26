#!/usr/bin/env bash
# Serves a model with the local vLLM CPU image built by build_vllm_cpu_image.sh.
# Usage: ./scripts/run_vllm_cpu.sh <model_id> [extra vllm serve args...]
set -euo pipefail

MODEL="${1:?usage: run_vllm_cpu.sh <model_id> [extra args...]}"
shift || true

docker rm -f vllm-cpu-server >/dev/null 2>&1 || true

docker run -d --name vllm-cpu-server \
  --shm-size=1g \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 8000:8000 \
  vllm-cpu:arm64 \
  --model "$MODEL" \
  --gpu-memory-utilization 0.4 \
  --enforce-eager \
  "$@"

echo "vLLM CPU server starting for model '$MODEL' -> http://localhost:8000"
echo "Follow logs: docker logs -f vllm-cpu-server"
