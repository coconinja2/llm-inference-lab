#!/usr/bin/env bash
# Builds vLLM's official CPU backend image for local (Apple Silicon) use.
# vLLM has no native macOS support, so we build its Linux/arm64 image via
# Docker Desktop's Linux VM instead. See README.md for why, and for the
# OOM / shm-size / gpu-memory-utilization issues this build ran into.
set -euo pipefail

VLLM_SRC="${VLLM_SRC:-vendor/vllm-src}"
VLLM_REF="${VLLM_REF:-v0.27.1}"

if [ ! -d "$VLLM_SRC" ]; then
  git clone --depth 1 --branch "$VLLM_REF" https://github.com/vllm-project/vllm.git "$VLLM_SRC"
fi

cd "$VLLM_SRC"

# max_jobs=4: the Dockerfile's default of 32 parallel compile jobs OOMs
# against Docker Desktop's default ~7.65GB VM memory allocation.
docker buildx build \
  --platform linux/arm64 \
  -f docker/Dockerfile.cpu \
  --target vllm-openai \
  --build-arg max_jobs=4 \
  -t vllm-cpu:arm64 \
  --load .
