# LLM Inference Lab

A hands-on, incremental project for learning LLM inference and serving
internals — vLLM, KV cache, continuous batching, GPU kernels (Triton/CUDA),
distributed inference, and Kubernetes GPU workloads — by building small,
measured experiments rather than one large system.

Every optimization in this project has a baseline and a benchmark. Negative
results (an idea that made things worse) are recorded, not hidden.

## Roadmap

1. **vLLM inference benchmarking** — reproducible baseline measurements of
   TTFT, inter-token latency, throughput, and memory across concurrency,
   prompt length, and output length. *(in progress)*
2. **Understand vLLM** — trace a request through vLLM's actual source: API →
   scheduler → batching → prefill → KV cache → decode → streaming.
3. **Tiny inference server** — a small, readable educational server
   implementing the same request lifecycle from scratch.
4. **GPU kernel experiments** — PyTorch → Triton → CUDA implementations of a
   single inference op, benchmarked at each step.
5. **Distributed inference** — multi-worker routing, tensor parallelism,
   NCCL, failure handling.
6. **Kubernetes** — GPU workload deployment, metrics, queue-based
   autoscaling.
7. **Original inference optimization research** — hypothesis-driven
   experiments (computation reuse, alternative attention, etc.), each with a
   baseline, benchmark, and written analysis of why it worked or didn't.

## Environment

Developed on a MacBook Air (Apple M3, 8-core CPU, 16GB RAM) with **no NVIDIA
GPU**. This shapes Phase 1-3 significantly:

- **vLLM has no native macOS support.** Its CPU backend is built against
  Linux toolchains (oneDNN, specific compiler flags) and fails to build via
  pip directly on Darwin.
- vLLM's `docker/Dockerfile.cpu` *does* support `linux/arm64`, so we build
  vLLM's real CPU backend inside a Linux container via Docker Desktop's
  Linux VM. This is genuine vLLM — real scheduler, real continuous batching,
  real (CPU-backed) KV cache — just without GPU-side metrics, and slow.
- GPU-dependent phases (4: CUDA/Triton, 6: Kubernetes GPU scaling) require a
  cloud GPU instance and are deferred until later.
- The benchmark harness built in Phase 1 is designed to run unchanged
  against a real vLLM GPU deployment later, so CPU and GPU numbers stay
  directly comparable.

### Building vLLM's CPU image locally

```bash
./scripts/build_vllm_cpu_image.sh
./scripts/run_vllm_cpu.sh gpt2 --dtype float32 --max-model-len 512
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2", "prompt": "Hello", "max_tokens": 20}'
```

Two non-obvious issues came up getting this running and are worth knowing
about if you hit them too:

- **OOM during the C++/Rust compile** (`cannot allocate memory` from
  `cmake --build`): the Dockerfile defaults to 32 parallel compile jobs,
  which exceeds Docker Desktop's default ~7.65GB VM memory allocation.
  Fixed via `--build-arg max_jobs=4`.
- **`Insufficient space in /dev/shm`**: vLLM's API server and engine core
  processes communicate over shared memory; Docker's default 64MB `/dev/shm`
  is too small. Fixed via `--shm-size=1g`.
- **`Available memory ... is less than desired CPU memory utilization`**:
  on the CPU backend, `--gpu-memory-utilization` (despite its name) controls
  the fraction of *CPU* memory vLLM reserves. The default (0.9) overshoots
  a memory-constrained container; lowered to `0.4`.

## Status

- [x] Environment inspected (no CUDA, no NVIDIA GPU, Apple Silicon)
- [x] vLLM CPU backend built and verified serving real completions locally
- [ ] Benchmark harness (TTFT, inter-token latency, throughput, memory)
- [ ] Baseline results across concurrency / prompt length / output length
