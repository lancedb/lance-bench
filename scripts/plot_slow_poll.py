#!/usr/bin/env python3
"""Plot mean_slow_poll and mean_long_delay from FTS index benchmark CSV."""

import argparse
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

STAGE_COLORS = {
    "load_data": "#888888",
    "tokenize_docs": "#2196F3",
    "merge_partitions": "#F44336",
    "write_metadata": "#FF9800",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", default="/tmp/run.csv", help="Path to progress CSV (default: /tmp/run.csv)")
    parser.add_argument(
        "-o", "--output", default="/tmp/slow_poll_plot.png", help="Output image path (default: /tmp/slow_poll_plot.png)"
    )
    args = parser.parse_args()

    rows = []
    with open(args.csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print("No data rows found", file=sys.stderr)
        sys.exit(1)

    elapsed = [float(r["elapsed_s"]) for r in rows]
    slow_poll_ms = [float(r["mean_slow_poll_ns"]) / 1e6 for r in rows]
    long_delay_ms = [float(r["mean_long_delay_ns"]) / 1e6 for r in rows]
    stages = [r["stage"] for r in rows]
    colors = [STAGE_COLORS.get(s, "#000000") for s in stages]

    # Use the first benchmark name as title
    title = rows[0]["benchmark_name"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.scatter(elapsed, slow_poll_ms, c=colors, s=14, alpha=0.7, edgecolors="none")
    ax1.set_ylabel("Mean Slow Poll (ms)")
    ax1.set_title(f"{title} — Tokio Task Metrics")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(elapsed, long_delay_ms, c=colors, s=14, alpha=0.7, edgecolors="none")
    ax2.set_ylabel("Mean Long Delay (ms)")
    ax2.set_xlabel("Elapsed Time (s)")
    ax2.grid(True, alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=s) for s, c in STAGE_COLORS.items()
    ]
    ax1.legend(handles=legend_elements, loc="upper left")
    ax2.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
