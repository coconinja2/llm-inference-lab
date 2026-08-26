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
./scripts/run_vllm_cpu.sh Qwen/Qwen3.5-0.8B --max-model-len 2048
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3.5-0.8B", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 20}'
```

Several non-obvious issues came up getting this running and are worth
knowing about if you hit them too:

- **OOM during the C++/Rust compile** (`cannot allocate memory` from
  `cmake --build`): the Dockerfile defaults to 32 parallel compile jobs,
  which exceeds Docker Desktop's default ~7.65GB VM memory allocation.
  Fixed via `--build-arg max_jobs=4`.
- **`Insufficient space in /dev/shm`**: vLLM's API server and engine core
  processes communicate over shared memory; Docker's default 64MB `/dev/shm`
  is too small. Fixed via `--shm-size=1g`.
- **`Available memory ... is less than desired CPU memory utilization`**:
  on the CPU backend, `--gpu-memory-utilization` (despite its name) controls
  the fraction of *CPU* memory vLLM reserves for the KV cache — a tight,
  model-size-dependent budget on a memory-constrained container. Too high
  and the weights + KV cache don't fit; too low and the margin computed
  during vLLM's internal memory-profiling pass goes negative and it refuses
  to start. We bumped Docker Desktop's VM memory from ~7.65GB to ~11.67GB
  (Settings → Resources → Memory) and tuned `--gpu-memory-utilization`
  empirically (0.4 works for a 0.8B model here).
- **`torch.compile` warmup can hang for a very long time on CPU** for
  hybrid/multimodal architectures — observed 10+ minutes with no
  completion on a 0.8B hybrid-attention model. `--enforce-eager` skips that
  compilation step entirely; it trades some runtime performance for a
  server that actually finishes starting, and is now the script's default.

### Model choice: a real negative result

We first tried **nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16**, a hybrid
Mamba-2 + MoE architecture. vLLM's CPU backend does have native kernels for
it (`vllm/model_executor/layers/mamba/ops/cpu/`), so this wasn't an
architecture-support failure — it was a hard memory ceiling: the 7.4GB
bf16 checkpoint alone consumed nearly all of an 11.67GB Docker VM, leaving
no room for KV cache no matter how far `--gpu-memory-utilization`,
`--max-num-seqs`, or `--enforce-eager` were pushed. Recorded here rather
than hidden, per the project's ground rules on negative results.

We're using **Qwen/Qwen3.5-0.8B** for Phase 1 instead — small enough
(1.6GB bf16) to fit comfortably, while still being architecturally
interesting: a hybrid linear-attention (GDN) + full-attention model, not a
plain transformer, which vLLM's CPU backend also has native kernels for.

## Benchmark harness

`benchmark/` is a small, dependency-light harness, not a wrapper around an
existing load-testing tool — the point of Phase 1 is to understand exactly
what's being measured and why.

- **`client.py`** — issues one streaming completion request and times it.
  It timestamps each raw SSE line the instant it's read off the socket,
  *before* any JSON parsing, so parsing cost never leaks into the latency
  numbers. From that it derives:
  - **TTFT (time to first token)** — request sent → first generated token
    received. This is dominated by *prefill*: the model runs one forward
    pass over the *entire* prompt to build the initial KV cache before it
    can produce anything. Prefill is compute-bound and scales with prompt
    length.
  - **Inter-token latency (ITL)** — gaps between subsequent tokens. This is
    dominated by *decode*: one forward pass per output token, reading the
    growing KV cache. Decode is normally memory-bandwidth-bound, not
    compute-bound — a very different bottleneck than prefill.
- **`workload.py`** — builds prompts with a precise, controlled token count
  (tokenize a source passage with the model's real tokenizer, tile/slice to
  the target length) so "prompt length" is an actual controlled variable,
  not an approximation.
- **`run.py`** — sweeps `(concurrency, prompt_tokens, output_tokens)`
  configs. For each config it fires `concurrency` requests at once (one
  "wave", via `asyncio.gather`) so the scheduler actually has multiple
  requests to batch, repeats for a few waves, and writes both a raw
  per-request JSONL and an aggregated summary CSV. Aggregate throughput is
  computed as *total completion tokens across every request in a wave,
  divided by the wave's wall-clock duration* — this is what actually
  reflects continuous batching, since it credits the system for serving
  requests in parallel rather than timing them back-to-back.
- **`plot.py`** — renders the summary CSV into PNGs.

```bash
python3 -m venv .venv && .venv/bin/pip install -r benchmark/requirements.txt
./scripts/run_vllm_cpu.sh Qwen/Qwen3.5-0.8B --max-model-len 2048
.venv/bin/python benchmark/run.py \
  --concurrency 1 2 4 --prompt-tokens 32 128 --output-tokens 32 --waves 2 --tag baseline
.venv/bin/python benchmark/plot.py results/summary_baseline_*.csv
```

## Phase 1 baseline results

Qwen3.5-0.8B, vLLM 0.27.1 CPU backend (`--enforce-eager`,
`--gpu-memory-utilization 0.4`), Docker VM: 8 CPUs / 11.67GB, 2 waves per
config.

| concurrency | prompt_tokens | TTFT mean (ms) | ITL mean (ms) | aggregate throughput (tok/s) |
|---|---|---|---|---|
| 1 | 32  | 1266  | 121 | 6.4  |
| 1 | 128 | 5431  | 123 | 3.5  |
| 2 | 32  | 1978  | 149 | 9.6  |
| 2 | 128 | 8843  | 227 | 4.0  |
| 4 | 32  | 3799  | 173 | 13.9 |
| 4 | 128 | 17979 | 283 | 4.8  |

Full CSV, raw per-request JSONL, and plots are in
[`results/`](results/).

**What this shows:**

- **TTFT scales with both prompt length and concurrency, and the
  concurrency scaling is steep.** Going from 32→128 prompt tokens is a 4x
  increase in prefill work; TTFT at concurrency=1 goes up ~4.3x (1266ms →
  5431ms), tracking prefill's compute-bound nature almost linearly. But
  concurrency compounds this badly here: at 128 prompt tokens, TTFT goes
  from 5.4s (concurrency=1) to 18s (concurrency=4) — over 3x worse for 4x
  the requests. This container only has ~7 usable CPU cores (one reserved),
  so concurrent prefills are mostly competing for the same compute rather
  than running in parallel.
- **ITL degrades more at higher prompt lengths under concurrency**
  (121→283ms going from 1x/32-tok to 4x/128-tok) rather than staying flat.
  vLLM's default chunked prefill interleaves prefill chunks from
  long-prompt requests with decode steps from other in-flight requests on
  the same scheduler tick — so a big prefill in the batch measurably slows
  down everyone else's token-by-token generation, not just its own TTFT.
- **Aggregate throughput still rises with concurrency** (6.4→13.9 tok/s at
  32 prompt tokens) — continuous batching is doing real work even on CPU.
  But the gain shrinks a lot at 128 prompt tokens (3.5→4.8 tok/s, a much
  flatter curve) because prefill's compute cost is eating the CPU budget
  that would otherwise go to parallel decode steps.
- **KV cache**, concretely: the reason TTFT and ITL are governed by
  different costs at all. Prefill computes and *writes* the KV cache
  entries for every prompt token in one pass (why it's compute-bound and
  scales with prompt length); decode *reads* the whole accumulated cache
  once per new token to attend over everything generated so far (why it's
  normally bandwidth-bound, not compute-bound, and why ITL is far more
  stable than TTFT until the scheduler starts contending resources across
  requests, as seen above).

These are CPU numbers, without GPU-side memory/utilization metrics — the
benchmark script is written to run unchanged against a real GPU vLLM
deployment later, so this baseline becomes a direct CPU-vs-GPU comparison
rather than being thrown away.

## Status

- [x] Environment inspected (no CUDA, no NVIDIA GPU, Apple Silicon)
- [x] vLLM CPU backend built and verified serving real completions locally
  (Qwen/Qwen3.5-0.8B; Nemotron-3-Nano-4B ruled out on memory grounds)
- [x] Benchmark harness (TTFT, inter-token latency, throughput)
- [x] Baseline results across concurrency / prompt length (see above)
- [ ] Output-length sweep and GPU memory / KV-cache-usage metrics (blocked
  on cloud GPU access — deferred per plan)
