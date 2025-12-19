#!/usr/bin/env python3
"""Compare PR benchmark results against historical baseline using z-scores."""

import argparse
import sys
from pathlib import Path

import lance
import numpy as np

# Add packages to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from lance_bench_db.dataset import connect
from lance_bench_db.models import Result


def fetch_pr_results(pr_sha: str, local_results_path: Path) -> dict[str, dict]:
    """Fetch benchmark results for PR commit.

    Args:
        pr_sha: Full or short commit SHA for the PR
        local_results_path: Path to local results table

    Returns:
        Dictionary mapping benchmark_name to result dict
    """
    print(f"Reading PR results from local table: {local_results_path}")
    pr_results_df = lance.dataset(local_results_path).to_table().to_pandas()

    if pr_results_df.empty:
        short_sha = pr_sha[:7]
        print(f"Warning: No results found for commit {short_sha}")
        return {}

    # Convert to dict format
    pr_results_dict = pr_results_df.to_dict("records")

    # Group by benchmark_name (should be unique for a single commit)
    pr_by_benchmark = {}
    for result in pr_results_dict:
        benchmark_name = result["benchmark_name"]
        pr_by_benchmark[benchmark_name] = result

    print(f"Found {len(pr_by_benchmark)} PR benchmark results")
    return pr_by_benchmark


def fetch_historical_baseline(benchmark_name: str, limit: int = 20) -> list[dict]:
    """Fetch N most recent historical results for a benchmark.

    Args:
        benchmark_name: Name of the benchmark
        exclude_sha: Commit SHA to exclude (the PR commit)
        limit: Maximum number of historical results to fetch

    Returns:
        List of result dicts sorted by timestamp (newest first)
    """
    db = connect()
    results_table = Result.open_table(db)

    # Get all results for this benchmark
    all_results_df = results_table.search().where(f"benchmark_name = '{benchmark_name}'").to_pandas()

    if all_results_df.empty:
        return []

    # Sort by timestamp descending and take top N
    historical_df = all_results_df.sort_values(by="timestamp", ascending=False)
    historical_df = historical_df.head(limit)

    return historical_df.to_dict("records")


def calculate_z_score(pr_value: float, historical_values: list[float]) -> float | None:
    """Calculate z-score for PR value against historical distribution.

    Args:
        pr_value: PR benchmark result value
        historical_values: List of historical result values

    Returns:
        Z-score, or None if calculation not possible
    """
    if len(historical_values) < 2:
        # Need at least 2 historical values
        return None

    mean = np.mean(historical_values)
    std_dev = np.std(historical_values, ddof=1)  # Sample standard deviation

    if std_dev == 0:
        # Cannot calculate z-score with zero variance
        return None

    z_score = (pr_value - mean) / std_dev
    return float(z_score)


def determine_status(z_score: float | None, threshold: float = 2.0) -> tuple[str, str]:
    """Determine status based on z-score.

    For benchmarks, lower values are better (faster execution time).
    - Negative z-score: PR value < baseline mean → improvement (faster)
    - Positive z-score: PR value > baseline mean → regression (slower)

    Args:
        z_score: Calculated z-score (None if not calculable)
        threshold: Threshold for flagging (default 2.0)

    Returns:
        Tuple of (emoji, status_text)
    """
    if z_score is None:
        return "❓", "Insufficient Data"

    if z_score < -threshold:
        return "🚀", "Likely Improved"
    elif z_score > threshold:
        return "⚠️", "Likely Regressed"
    else:
        return "✅", "Within Normal Range"


def format_value(value: float, units: str) -> str:
    """Format a value with appropriate units.

    Args:
        value: Numeric value
        units: Unit string (e.g., "nanoseconds", "recall")

    Returns:
        Formatted string
    """
    if units == "nanoseconds":
        # Convert to appropriate time unit
        if value < 1_000:
            return f"{value:.2f} ns"
        elif value < 1_000_000:
            return f"{value / 1_000:.2f} µs"
        elif value < 1_000_000_000:
            return f"{value / 1_000_000:.2f} ms"
        else:
            return f"{value / 1_000_000_000:.2f} s"
    elif units == "recall":
        return f"{value:.4f}"
    else:
        return f"{value:.2f} {units}"


def generate_comparison_report(pr_sha: str, pr_number: int, comparisons: list[dict], limit: int, threshold: float = 2.0) -> str:
    """Generate markdown report from comparison results.

    Args:
        pr_sha: PR commit SHA
        pr_number: PR number
        comparisons: List of comparison dicts
        threshold: Z-score threshold for flagging

    Returns:
        Markdown-formatted report string
    """
    short_sha = pr_sha[:7]

    # Count status categories
    # Lower values are better, so negative z-score = improvement, positive = regression
    improvements = [c for c in comparisons if c["z_score"] is not None and c["z_score"] < -threshold]
    regressions = [c for c in comparisons if c["z_score"] is not None and c["z_score"] > threshold]
    stable = [c for c in comparisons if c["z_score"] is not None and abs(c["z_score"]) <= threshold]
    insufficient_data = [c for c in comparisons if c["z_score"] is None]

    # Build report
    lines = []
    lines.append(f"## Benchmark Results for PR #{pr_number}")
    lines.append("")
    lines.append(f"**Commit:** `{short_sha}`")
    lines.append(f"**Baseline:** Up to {limit} most recent historical results per benchmark")
    lines.append("")

    lines.append("### Summary")
    lines.append(f"- **Total benchmarks:** {len(comparisons)}")
    lines.append(f"- 🚀 **Improvements:** {len(improvements)}")
    lines.append(f"- ⚠️ **Regressions:** {len(regressions)}")
    lines.append(f"- ✅ **Stable:** {len(stable)}")
    lines.append(f"- ❓ **Insufficient data:** {len(insufficient_data)}")
    lines.append("")

    # Flagged benchmarks section
    flagged = improvements + regressions
    if flagged:
        lines.append(f"### Flagged Benchmarks (|z-score| > {threshold})")
        lines.append("")
        lines.append("| Benchmark | PR Result | Baseline Mean | Baseline Std Dev | Z-Score | Status |")
        lines.append("|-----------|-----------|---------------|------------------|---------|--------|")

        # Sort by absolute z-score descending
        flagged.sort(key=lambda c: abs(c["z_score"]), reverse=True)

        for comp in flagged:
            emoji, status = determine_status(comp["z_score"], threshold)
            pr_val = format_value(comp["pr_value"], comp["units"])
            baseline_mean = format_value(comp["baseline_mean"], comp["units"])
            baseline_std = format_value(comp["baseline_std"], comp["units"])
            z_score_str = f"{comp['z_score']:+.2f}"

            lines.append(
                f"| {comp['benchmark_name']} | {pr_val} | {baseline_mean} | {baseline_std} | {z_score_str} | {emoji} {status} |"
            )
        lines.append("")
    else:
        lines.append("### ✅ All Benchmarks Within Normal Range")
        lines.append("")
        lines.append(f"No benchmarks had |z-score| > {threshold}")
        lines.append("")

    # All results in collapsible section
    lines.append("### All Results")
    lines.append("<details>")
    lines.append(f"<summary>View all {len(comparisons)} benchmark results</summary>")
    lines.append("")
    lines.append("| Benchmark | PR Result | Baseline Mean | Baseline N | Z-Score | Status |")
    lines.append("|-----------|-----------|---------------|------------|---------|--------|")

    # Sort alphabetically
    comparisons_sorted = sorted(comparisons, key=lambda c: c["benchmark_name"])

    for comp in comparisons_sorted:
        emoji, status = determine_status(comp["z_score"], threshold)
        pr_val = format_value(comp["pr_value"], comp["units"])
        baseline_mean = format_value(comp["baseline_mean"], comp["units"]) if comp["baseline_mean"] is not None else "N/A"
        z_score_str = f"{comp['z_score']:+.2f}" if comp["z_score"] is not None else "N/A"
        baseline_n = comp["baseline_count"]

        lines.append(f"| {comp['benchmark_name']} | {pr_val} | {baseline_mean} | {baseline_n} | {z_score_str} | {emoji} |")

    lines.append("")
    lines.append("</details>")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("*Generated by bench-bot* 🤖")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PR benchmark results against historical baseline")
    parser.add_argument(
        "pr_sha",
        type=str,
        help="PR commit SHA to compare",
    )
    parser.add_argument(
        "--pr-number",
        type=int,
        required=True,
        help="PR number for the report",
    )
    parser.add_argument(
        "--baseline-limit",
        type=int,
        default=20,
        help="Number of historical results to use for baseline (default: 20)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Z-score threshold for flagging improvements/regressions (default: 2.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--min-baseline-count",
        type=int,
        default=5,
        help="Minimum number of baseline results required for z-score calculation (default: 5)",
    )
    parser.add_argument(
        "--local-results",
        type=Path,
        default=None,
        help="Path to local results table (PR mode)",
        required=True,
    )

    args = parser.parse_args()

    # Fetch PR results
    pr_results = fetch_pr_results(args.pr_sha, args.local_results)

    if not pr_results:
        print("Error: No PR results found. Benchmarks may not have completed yet.", file=sys.stderr)
        sys.exit(1)

    # Compare each benchmark
    comparisons = []

    for benchmark_name, pr_result in pr_results.items():
        print(f"Comparing {benchmark_name}...")

        # Fetch historical baseline
        historical_results = fetch_historical_baseline(benchmark_name, args.baseline_limit)

        # Extract values
        pr_value = pr_result["summary"]["mean"]
        units = pr_result["units"]

        if len(historical_results) < args.min_baseline_count:
            print(f"  Warning: Only {len(historical_results)} historical results (need {args.min_baseline_count})")
            # Not enough data for comparison
            comparisons.append(
                {
                    "benchmark_name": benchmark_name,
                    "pr_value": pr_value,
                    "units": units,
                    "baseline_mean": None,
                    "baseline_std": None,
                    "baseline_count": len(historical_results),
                    "z_score": None,
                }
            )
            continue

        # Calculate baseline statistics
        historical_means = [r["summary"]["mean"] for r in historical_results]
        baseline_mean = np.mean(historical_means)
        baseline_std = np.std(historical_means, ddof=1)

        # Calculate z-score
        z_score = calculate_z_score(pr_value, historical_means)

        if z_score is not None:
            print(f"  Z-score: {z_score:+.2f}")

        comparisons.append(
            {
                "benchmark_name": benchmark_name,
                "pr_value": pr_value,
                "units": units,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "baseline_count": len(historical_results),
                "z_score": z_score,
            }
        )

    # Generate report
    print("\nGenerating comparison report...")
    report = generate_comparison_report(args.pr_sha, args.pr_number, comparisons, args.baseline_limit, args.threshold)

    # Output report
    if args.output:
        args.output.write_text(report)
        print(f"\n✅ Report saved to {args.output}")
    else:
        print("\n" + "=" * 80)
        print(report)
        print("=" * 80)

    # Exit with non-zero if there are regressions
    # Lower values are better, so positive z-score = regression
    regressions = [c for c in comparisons if c["z_score"] is not None and c["z_score"] > args.threshold]
    if regressions:
        print(f"\n⚠️ Found {len(regressions)} potential regression(s)", file=sys.stderr)
        sys.exit(0)  # Don't fail the workflow, just flag it
    else:
        print("\n✅ No regressions detected")
        sys.exit(0)


if __name__ == "__main__":
    main()
