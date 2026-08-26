"""Streaming client that measures per-token timing for a single request.

Measurement approach: we timestamp the arrival of each raw SSE line as it's
read off the socket, *before* doing any JSON parsing. Parsing happens after
the timestamp is recorded, so parsing/processing cost never leaks into the
latency numbers. TTFT is the time from just-before-send to the first chunk
that carries actual generated text (not any empty/role-only chunk some
servers send first). Inter-token latencies are gaps between subsequent
text-bearing chunks.
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class RequestResult:
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    token_times_s: list[float] = field(default_factory=list)  # arrival time of each token, relative to request start
    total_latency_s: float = 0.0

    @property
    def inter_token_latencies_s(self) -> list[float]:
        return [b - a for a, b in zip(self.token_times_s, self.token_times_s[1:])]


async def stream_completion(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    token_times: list[float] = []
    ttft: float | None = None
    prompt_tokens = 0
    completion_tokens = 0

    t_start = time.perf_counter()
    async with client.stream("POST", f"{base_url}/v1/completions", json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            t_now = time.perf_counter()
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                if ttft is None:
                    ttft = t_now - t_start
                token_times.append(t_now - t_start)
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
    t_end = time.perf_counter()

    if ttft is None:
        raise RuntimeError("No tokens received in stream")

    return RequestResult(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens or len(token_times),
        ttft_s=ttft,
        token_times_s=token_times,
        total_latency_s=t_end - t_start,
    )


async def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--prompt", default="Explain what a KV cache is in one paragraph.")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    async with httpx.AsyncClient(timeout=120.0) as client:
        result = await stream_completion(client, args.base_url, args.model, args.prompt, args.max_tokens)

    itl = result.inter_token_latencies_s
    print(f"prompt_tokens:      {result.prompt_tokens}")
    print(f"completion_tokens:  {result.completion_tokens}")
    print(f"TTFT:               {result.ttft_s * 1000:.1f} ms")
    if itl:
        print(f"mean ITL:           {sum(itl) / len(itl) * 1000:.1f} ms")
        print(f"min/max ITL:        {min(itl) * 1000:.1f} / {max(itl) * 1000:.1f} ms")
    print(f"total latency:      {result.total_latency_s * 1000:.1f} ms")
    if result.total_latency_s > 0:
        print(f"throughput:         {result.completion_tokens / result.total_latency_s:.2f} tok/s")


if __name__ == "__main__":
    asyncio.run(_main())
