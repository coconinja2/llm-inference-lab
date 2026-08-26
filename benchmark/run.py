"""Sweeps (concurrency, prompt_tokens, output_tokens) configs against a
running vLLM server and records per-request + aggregate metrics.

For each config, `concurrency` requests are fired at once (one "wave") via
asyncio.gather, repeated for `waves` rounds. This is what actually exercises
continuous batching: within a wave, the server's scheduler is packing
decode steps from multiple in-flight requests into the same batch, which is
why aggregate throughput and per-request inter-token latency tell different
stories as concurrency rises.

Aggregate throughput per config = (sum of completion tokens across every
request in every wave) / (sum of each wave's wall-clock duration). This
credits the system for serving requests in parallel, not just back to back.
"""

import argparse
import asyncio
import csv
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path

import httpx

from client import stream_completion
from workload import build_prompt, load_tokenizer

RESULTS_DIR = Path(__file__).parent.parent / "results"


async def run_wave(client, base_url, model, prompt, max_tokens, concurrency):
    t_wave_start = time.perf_counter()
    results = await asyncio.gather(
        *[stream_completion(client, base_url, model, prompt, max_tokens) for _ in range(concurrency)]
    )
    wave_duration = time.perf_counter() - t_wave_start
    return results, wave_duration


async def sweep(base_url, model, concurrencies, prompt_lens, output_lens, waves):
    tokenizer = load_tokenizer(model)
    prompts = {n: build_prompt(tokenizer, n) for n in prompt_lens}

    raw_records = []
    summary_records = []

    limits = httpx.Limits(max_connections=max(concurrencies) + 1)
    async with httpx.AsyncClient(timeout=300.0, limits=limits) as client:
        for concurrency, prompt_len, output_len in itertools.product(concurrencies, prompt_lens, output_lens):
            config = {"concurrency": concurrency, "prompt_tokens": prompt_len, "output_tokens": output_len}
            print(f"running {config} x{waves} waves...")

            wave_results = []
            wave_durations = []
            for _ in range(waves):
                results, duration = await run_wave(
                    client, base_url, model, prompts[prompt_len], output_len, concurrency
                )
                wave_results.extend(results)
                wave_durations.append(duration)
                for r in results:
                    raw_records.append({**config, **asdict(r), "token_times_s": None})  # drop raw series from jsonl row-per-request view

            ttfts = [r.ttft_s for r in wave_results]
            all_itls = [itl for r in wave_results for itl in r.inter_token_latencies_s]
            total_completion_tokens = sum(r.completion_tokens for r in wave_results)
            total_wave_time = sum(wave_durations)

            summary_records.append({
                **config,
                "waves": waves,
                "n_requests": len(wave_results),
                "ttft_mean_ms": _mean(ttfts) * 1000,
                "ttft_p50_ms": _percentile(ttfts, 50) * 1000,
                "ttft_p95_ms": _percentile(ttfts, 95) * 1000,
                "itl_mean_ms": _mean(all_itls) * 1000 if all_itls else None,
                "itl_p50_ms": _percentile(all_itls, 50) * 1000 if all_itls else None,
                "itl_p95_ms": _percentile(all_itls, 95) * 1000 if all_itls else None,
                "aggregate_throughput_tok_s": total_completion_tokens / total_wave_time if total_wave_time else None,
            })

    return raw_records, summary_records


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _percentile(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--output-tokens", type=int, nargs="+", default=[32])
    parser.add_argument("--waves", type=int, default=2)
    parser.add_argument("--tag", default="baseline")
    args = parser.parse_args()

    raw_records, summary_records = asyncio.run(sweep(
        args.base_url, args.model, args.concurrency, args.prompt_tokens, args.output_tokens, args.waves
    ))

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    raw_path = RESULTS_DIR / f"raw_{args.tag}_{ts}.jsonl"
    with open(raw_path, "w") as f:
        for rec in raw_records:
            f.write(json.dumps(rec) + "\n")

    summary_path = RESULTS_DIR / f"summary_{args.tag}_{ts}.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_records[0].keys()))
        writer.writeheader()
        writer.writerows(summary_records)

    print(f"\nwrote {raw_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
