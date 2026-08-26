"""Plots TTFT, inter-token latency, and aggregate throughput vs. concurrency
from a summary CSV produced by run.py, one line per prompt-length value.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_summary(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path)
    out_dir.mkdir(exist_ok=True)

    metrics = [
        ("ttft_mean_ms", "Mean TTFT (ms)"),
        ("itl_mean_ms", "Mean inter-token latency (ms)"),
        ("aggregate_throughput_tok_s", "Aggregate throughput (tok/s)"),
    ]

    for metric, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        for prompt_len, group in df.groupby("prompt_tokens"):
            group = group.sort_values("concurrency")
            ax.plot(group["concurrency"], group[metric], marker="o", label=f"prompt_tokens={prompt_len}")
        ax.set_xlabel("Concurrency")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. concurrency")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / f"{metric}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent.parent / "results" / "plots")
    args = parser.parse_args()
    plot_summary(args.csv_path, args.out_dir)
