#!/usr/bin/env python3
"""Generate regression analysis report for benchmark results.

This script analyzes benchmark results to detect potential performance regressions
using statistical testing (t-test) and visualizes trends over time.
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Add packages to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from lance_bench_db.dataset import connect, get_database_uri
from lance_bench_db.models import Result


def fetch_all_results() -> list[dict]:
    """Fetch all benchmark results from the database.

    Returns:
        List of result dictionaries
    """
    print(f"Connecting to database at <{get_database_uri()}>...")
    db = connect()
    results_table = Result.open_table(db)

    print("Fetching all results...")
    # Fetch all results
    results = results_table.to_pandas()

    print(f"Fetched {len(results)} total results")
    return results.to_dict("records")


def group_and_sort_results(results: list[dict]) -> dict[str, list[dict]]:
    """Group results by benchmark name and sort by timestamp.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary mapping benchmark_name to sorted list of results
    """
    grouped = defaultdict(list)

    for result in results:
        benchmark_name = result["benchmark_name"]
        grouped[benchmark_name].append(result)

    # Sort each group by dut.timestamp
    for benchmark_name in grouped:
        grouped[benchmark_name].sort(key=lambda r: r["dut"]["timestamp"])

    print(f"Grouped into {len(grouped)} unique benchmarks")
    return dict(grouped)


def calculate_regression_pvalue(results: list[dict], recent_count: int = 4) -> float | None:
    """Calculate p-value for potential regression using t-test.

    Compares the most recent N results against all older results.

    Args:
        results: List of results sorted by timestamp (oldest to newest)
        recent_count: Number of recent results to compare

    Returns:
        P-value from t-test, or None if insufficient data
    """
    if len(results) < recent_count + 2:
        # Need at least recent_count + 2 results for meaningful comparison
        return None

    # Extract mean values from summary
    all_means = [r["summary"]["mean"] for r in results]

    # Split into recent and older results
    older_means = all_means[:-recent_count]
    recent_means = all_means[-recent_count:]

    # Perform two-sample t-test (two-tailed)
    # Null hypothesis: recent and older results have same mean
    # Lower p-value suggests they're different (potential regression)
    result = stats.ttest_ind(recent_means, older_means)
    # Result is tuple-like with (statistic, pvalue)
    pvalue: float = result[1]  # type: ignore[index]

    return pvalue


def analyze_benchmarks(grouped_results: dict[str, list[dict]], recent_count: int = 4) -> list[tuple[str, float, list[dict]]]:
    """Analyze all benchmarks and calculate p-values.

    Args:
        grouped_results: Dictionary of benchmark_name -> sorted results
        recent_count: Number of recent results to compare

    Returns:
        List of (benchmark_name, p_value, results) tuples sorted by p-value (descending)
    """
    analyzed = []

    for benchmark_name, results in grouped_results.items():
        pvalue = calculate_regression_pvalue(results, recent_count)

        if pvalue is not None:
            analyzed.append((benchmark_name, pvalue, results))
        else:
            print(f"Skipping {benchmark_name}: insufficient data ({len(results)} results)")

    # Sort by p-value ascending (lowest p-value = most likely regression)
    analyzed.sort(key=lambda x: x[1], reverse=False)

    print(f"\nAnalyzed {len(analyzed)} benchmarks with sufficient data")
    return analyzed


def determine_time_unit(values: list[float]) -> tuple[str, float, str]:
    """Determine the best time unit for displaying values.

    Args:
        values: List of values in nanoseconds

    Returns:
        Tuple of (unit_name, divisor, unit_label)
    """
    max_val = max(values) if values else 0

    if max_val < 1_000:  # Less than 1 microsecond
        return ("nanoseconds", 1, "ns")
    elif max_val < 1_000_000:  # Less than 1 millisecond
        return ("microseconds", 1_000, "µs")
    elif max_val < 1_000_000_000:  # Less than 1 second
        return ("milliseconds", 1_000_000, "ms")
    else:
        return ("seconds", 1_000_000_000, "s")


def create_regression_chart(analyzed_benchmarks: list[tuple[str, float, list[dict]]], output_path: Path) -> None:
    """Create an interactive HTML chart showing benchmark trends with tooltips.

    Args:
        analyzed_benchmarks: List of (benchmark_name, p_value, results) tuples
        output_path: Path to save the HTML chart
    """
    n_benchmarks = len(analyzed_benchmarks)
    if n_benchmarks == 0:
        print("No benchmarks to plot")
        return

    print(f"\nCreating interactive chart with {n_benchmarks} subplots...")

    # Create subplots with minimal spacing
    # Maximum vertical spacing is 1 / (rows - 1), but we want much less
    max_allowed_spacing = 1.0 / (n_benchmarks - 1) if n_benchmarks > 1 else 0.1
    # Use minimal spacing - just enough to see separation
    vertical_spacing = min(0.005, max_allowed_spacing * 0.3)

    fig = make_subplots(
        rows=n_benchmarks,
        cols=1,
        subplot_titles=[f"{name} (p={pval:.4f})" for name, pval, _ in analyzed_benchmarks],
        vertical_spacing=vertical_spacing,
    )

    for idx, (benchmark_name, pvalue, results) in enumerate(analyzed_benchmarks):
        row = idx + 1

        # Extract data for plotting
        timestamps = [r["dut"]["timestamp"] for r in results]
        means = [r["summary"]["mean"] for r in results]

        # Determine best unit for this benchmark
        unit_name, divisor, unit_label = determine_time_unit(means)

        # Convert values to appropriate unit
        means_scaled = [m / divisor for m in means]

        # Convert timestamps to datetime for better x-axis labels
        dates = [datetime.fromtimestamp(ts) for ts in timestamps]

        # Format hover text with detailed information
        hover_texts = []
        for r in results:
            ts = datetime.fromtimestamp(r["dut"]["timestamp"])
            hover_text = (
                f"<b>{benchmark_name}</b><br>"
                f"Version: {r['dut']['version']}<br>"
                f"Timestamp: {ts.strftime('%Y-%m-%d %H:%M:%S')}<br>"
                f"Mean: {r['summary']['mean'] / divisor:.2f} {unit_label}<br>"
                f"Min: {r['summary']['min'] / divisor:.2f} {unit_label}<br>"
                f"Max: {r['summary']['max'] / divisor:.2f} {unit_label}<br>"
                f"Std Dev: {r['summary']['standard_deviation'] / divisor:.2f} {unit_label}"
            )
            hover_texts.append(hover_text)

        # Add trace for the benchmark data
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=means_scaled,
                mode="lines+markers",
                name=benchmark_name,
                hovertext=hover_texts,
                hoverinfo="text",
                marker={"size": 6},
                line={"width": 2},
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        # Add vertical line separating recent results
        if len(results) >= 4:
            split_date = dates[-4]
            # Add as a shape - need to specify both xref and yref for correct subplot
            # xref format: "x" for first plot, "x2" for second, etc.
            # yref format: "y domain" for first plot, "y2 domain" for second, etc.
            xref = "x" if idx == 0 else f"x{idx + 1}"
            yref = "y domain" if idx == 0 else f"y{idx + 1} domain"
            fig.add_shape(
                type="line",
                x0=split_date,
                x1=split_date,
                y0=0,
                y1=1,
                xref=xref,
                yref=yref,
                line={"color": "red", "width": 1, "dash": "dash"},
                opacity=0.5,
            )

        # Color code subplot title based on p-value
        # Use brighter colors for dark mode
        color = "#4ade80" if pvalue > 0.05 else "#fb923c" if pvalue > 0.01 else "#f87171"
        if fig.layout.annotations:  # type: ignore[attr-defined]
            fig.layout.annotations[idx].update(font={"color": color, "size": 10})  # type: ignore[attr-defined,index]

        # Update y-axis label with appropriate unit
        fig.update_yaxes(title_text=f"Time ({unit_label})", row=row, col=1, title_font={"size": 10})

    # Update x-axis for bottom plot
    fig.update_xaxes(title_text="Date", row=n_benchmarks, col=1, title_font={"size": 10})

    # Update layout with dark mode theme
    fig.update_layout(
        title={
            "text": "Benchmark Regression Analysis<br><sub>(Sorted by p-value: Low→High)</sub>",
            "font": {"size": 16},
        },
        height=max(n_benchmarks * 300, 600),
        hovermode="closest",
        template="plotly_dark",
        paper_bgcolor="#0f172a",  # Slate-900
        plot_bgcolor="#1e293b",  # Slate-800
    )

    # Save as HTML
    print(f"Saving interactive chart to {output_path}...")
    fig.write_html(output_path)
    print("✓ Interactive HTML chart saved successfully")


def print_summary(analyzed_benchmarks: list[tuple[str, float, list[dict]]], threshold: float = 0.05) -> None:
    """Print summary of regression analysis.

    Args:
        analyzed_benchmarks: List of (benchmark_name, p_value, results) tuples
        threshold: P-value threshold for flagging regressions
    """
    print("\n" + "=" * 80)
    print("REGRESSION ANALYSIS SUMMARY")
    print("=" * 80)

    # Count potential regressions
    regressions = [b for b in analyzed_benchmarks if b[1] < threshold]
    warnings = [b for b in analyzed_benchmarks if 0.01 <= b[1] < threshold]
    likely_regressions = [b for b in analyzed_benchmarks if b[1] < 0.01]

    print(f"\nTotal benchmarks analyzed: {len(analyzed_benchmarks)}")
    print(f"Potential regressions (p < {threshold}): {len(regressions)}")
    print(f"  - High concern (p < 0.01): {len(likely_regressions)}")
    print(f"  - Medium concern (0.01 ≤ p < 0.05): {len(warnings)}")
    print(f"Likely stable (p ≥ {threshold}): {len(analyzed_benchmarks) - len(regressions)}")

    if likely_regressions:
        print("\n⚠️  HIGH CONCERN BENCHMARKS (p < 0.01):")
        for benchmark_name, pvalue, results in likely_regressions:
            recent_mean = np.mean([r["summary"]["mean"] for r in results[-4:]])
            older_mean = np.mean([r["summary"]["mean"] for r in results[:-4]])
            change_pct = ((recent_mean - older_mean) / older_mean) * 100
            print(f"  - {benchmark_name}")
            print(f"    p-value: {pvalue:.6f}")
            print(f"    Change: {change_pct:+.2f}% ({'slower' if change_pct > 0 else 'faster'})")

    if warnings:
        print("\n⚠️  MEDIUM CONCERN BENCHMARKS (0.01 ≤ p < 0.05):")
        for benchmark_name, pvalue, _ in warnings:
            print(f"  - {benchmark_name} (p={pvalue:.4f})")

    print("\n" + "=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate regression analysis report for benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default report
  python regression_report.py

  # Specify output location and recent count
  python regression_report.py -o report.html --recent-count 5

  # Use custom p-value threshold
  python regression_report.py --threshold 0.01
        """,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("regression_report.html"),
        help="Output path for the interactive chart (default: regression_report.html)",
    )
    parser.add_argument(
        "--recent-count",
        type=int,
        default=4,
        help="Number of recent results to compare against older results (default: 4)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="P-value threshold for flagging regressions (default: 0.05)",
    )

    args = parser.parse_args()

    try:
        # Fetch all results
        results = fetch_all_results()

        if not results:
            print("No results found in database")
            sys.exit(1)

        # Group and sort by benchmark
        grouped_results = group_and_sort_results(results)

        # Analyze for regressions
        analyzed_benchmarks = analyze_benchmarks(grouped_results, args.recent_count)

        if not analyzed_benchmarks:
            print("No benchmarks with sufficient data for analysis")
            sys.exit(1)

        # Print summary
        print_summary(analyzed_benchmarks, args.threshold)

        # Create chart
        create_regression_chart(analyzed_benchmarks, args.output)

        print(f"\n✅ Report generated successfully: {args.output}")

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
