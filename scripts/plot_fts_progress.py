"""Plot RSS, disk, and CPU usage over time from FTS index benchmark progress CSV."""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path: Path) -> dict[str, list[dict]]:
    """Load CSV and group rows by benchmark_name."""
    groups: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["elapsed_s"] = float(row["elapsed_s"])
            row["rss_mb"] = float(row["rss_mb"])
            row["cpu_active"] = float(row["cpu_active"])
            row["rss_bytes"] = int(row["rss_bytes"])
            row["disk_delta_mb"] = float(row["disk_delta_mb"])
            row["progress_value"] = int(row["progress_value"])
            groups[row["benchmark_name"]].append(row)
    return dict(groups)


def plot_benchmark(name: str, rows: list[dict], output: Path | None, *, show_disk: bool = True) -> None:
    """Create a single figure with dual y-axes for one benchmark run."""
    elapsed = [r["elapsed_s"] for r in rows]
    rss = [r["rss_mb"] for r in rows]
    disk = [r["disk_delta_mb"] for r in rows]
    cpu_active = [r["cpu_active"] for r in rows]

    stage_starts = [(r["elapsed_s"], r["stage"]) for r in rows if r["event"] == "start"]
    stage_completes = [r["elapsed_s"] for r in rows if r["event"] == "complete"]

    fig, ax_mb = plt.subplots(figsize=(12, 5))
    ax_cpu = ax_mb.twinx()

    ax_mb.step(elapsed, rss, where="post", color="#2563eb", linewidth=1.5, label="RSS (MB)")
    if show_disk:
        ax_mb.step(elapsed, disk, where="post", color="#16a34a", linewidth=1.5, label="Index Disk (MB)")
    ax_cpu.step(elapsed, cpu_active, where="post", color="#dc2626", linewidth=1.5, label="Active CPUs")

    # Stage boundary markers
    for t, stage in stage_starts:
        ax_mb.axvline(t, color="#9ca3af", linestyle="--", linewidth=0.8)
        ax_mb.text(
            t,
            ax_mb.get_ylim()[1],
            f" {stage}",
            fontsize=7,
            rotation=90,
            va="top",
            color="#6b7280",
        )
    for t in stage_completes:
        ax_mb.axvline(t, color="#d1d5db", linestyle=":", linewidth=0.7)

    ax_mb.set_xlabel("Elapsed Time (s)")
    ax_mb.set_ylabel("MB", color="#1e3a5f")
    ax_cpu.set_ylabel("Active CPUs", color="#dc2626")
    ax_mb.tick_params(axis="y", labelcolor="#1e3a5f")
    ax_cpu.tick_params(axis="y", labelcolor="#dc2626")

    # Combined legend
    lines_mb, labels_mb = ax_mb.get_legend_handles_labels()
    lines_cpu, labels_cpu = ax_cpu.get_legend_handles_labels()
    ax_mb.legend(lines_mb + lines_cpu, labels_mb + labels_cpu, loc="upper left")

    ax_mb.set_title(name)
    ax_mb.grid(True, alpha=0.3)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"  Saved {output}")
    else:
        plt.show()

    plt.close(fig)


def make_filename(name: str, base: Path) -> Path:
    """Derive an output filename from the benchmark name and base path."""
    safe = name.replace("/", "_").replace("=", "")
    return base.parent / f"{base.stem}_{safe}{base.suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FTS index benchmark progress CSV")
    parser.add_argument("csv_path", type=Path, help="Path to the progress CSV file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save plots to files (one per benchmark). Benchmark name is appended to the stem.",
    )
    parser.add_argument(
        "--no-disk",
        action="store_true",
        help="Skip the disk usage trace",
    )
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"File not found: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    groups = load_csv(args.csv_path)
    if not groups:
        print("No data found in CSV", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {sum(len(v) for v in groups.values())} samples across {len(groups)} benchmark(s)")

    for name, rows in groups.items():
        out = make_filename(name, args.output) if args.output else None
        plot_benchmark(name, rows, out, show_disk=not args.no_disk)


if __name__ == "__main__":
    main()
