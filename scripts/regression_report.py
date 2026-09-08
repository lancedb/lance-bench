#!/usr/bin/env python3
"""Generate regression analysis report for benchmark results.

This script analyzes benchmark results to detect potential performance regressions
using statistical testing (t-test) and visualizes trends over time.
"""

import argparse
import json
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


def create_regression_chart(
    analyzed_benchmarks: list[tuple[str, float, list[dict]]],
    output_path: Path,
    page_size: int = 100,
) -> None:
    """Create an interactive HTML chart showing benchmark trends with pagination.

    Renders the most recent `page_size` data points per benchmark by default.
    Older data is embedded as JSON and loaded via JavaScript pagination controls.

    Args:
        analyzed_benchmarks: List of (benchmark_name, p_value, results) tuples
        output_path: Path to save the HTML chart
        page_size: Number of data points to show per page (default: 100)
    """
    n_benchmarks = len(analyzed_benchmarks)
    if n_benchmarks == 0:
        print("No benchmarks to plot")
        return

    print(f"\nCreating interactive chart with {n_benchmarks} subplots (page size: {page_size})...")

    # Serialize all data for JavaScript pagination — keyed by benchmark name.
    # Only the fields needed for rendering are included (not full result dicts).
    all_benchmark_data: dict[str, dict] = {}
    for benchmark_name, _pvalue, results in analyzed_benchmarks:
        means = [r["summary"]["mean"] for r in results]
        _unit_name, divisor, unit_label = determine_time_unit(means)
        data_points = []
        for r in results:
            ts = datetime.fromtimestamp(r["dut"]["timestamp"])
            # Store raw numbers only; hover text is built in JavaScript on demand
            # to avoid embedding ~250 chars of HTML per data point for all history.
            data_points.append(
                {
                    "d": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                    "m": round(r["summary"]["mean"] / divisor, 4),
                    "n": round(r["summary"]["min"] / divisor, 4),
                    "x": round(r["summary"]["max"] / divisor, 4),
                    "s": round(r["summary"]["standard_deviation"] / divisor, 4),
                    "v": r["dut"]["version"],
                }
            )
        all_benchmark_data[benchmark_name] = {
            "unit_label": unit_label,
            "total_points": len(results),
            "data": data_points,
        }

    # Build the initial Plotly figure showing page 0 (most recent page_size points).
    max_allowed_spacing = 1.0 / (n_benchmarks - 1) if n_benchmarks > 1 else 0.1
    vertical_spacing = min(0.005, max_allowed_spacing * 0.3)

    fig = make_subplots(
        rows=n_benchmarks,
        cols=1,
        subplot_titles=[f"{name} (p={pval:.4f})" for name, pval, _ in analyzed_benchmarks],
        vertical_spacing=vertical_spacing,
    )

    for idx, (benchmark_name, pvalue, results) in enumerate(analyzed_benchmarks):
        row = idx + 1
        bm_data = all_benchmark_data[benchmark_name]
        unit_label = bm_data["unit_label"]

        # Page 0: most recent page_size data points
        initial_data = bm_data["data"][-page_size:]

        hover_texts = [
            f"<b>{benchmark_name}</b><br>"
            f"Version: {d['v']}<br>"
            f"Date: {d['d'][:10]}<br>"
            f"Mean: {d['m']:.4g} {unit_label}<br>"
            f"Min: {d['n']:.4g} {unit_label}<br>"
            f"Max: {d['x']:.4g} {unit_label}<br>"
            f"Std Dev: {d['s']:.4g} {unit_label}"
            for d in initial_data
        ]

        fig.add_trace(
            go.Scatter(
                x=[d["d"] for d in initial_data],
                y=[d["m"] for d in initial_data],
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

        # Add split line between "recent" and "older" results.
        # Only show it if the split point falls within the initial visible range.
        if len(results) >= 4 and initial_data:
            split_date = datetime.fromtimestamp(results[-4]["dut"]["timestamp"]).strftime("%Y-%m-%dT%H:%M:%S")
            if split_date >= initial_data[0]["d"]:
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

        color = "#4ade80" if pvalue > 0.05 else "#fb923c" if pvalue > 0.01 else "#f87171"
        if fig.layout.annotations:  # type: ignore[attr-defined]
            fig.layout.annotations[idx].update(font={"color": color, "size": 10})  # type: ignore[attr-defined,index]

        fig.update_yaxes(title_text=f"Time ({unit_label})", row=row, col=1, title_font={"size": 10})

    fig.update_xaxes(title_text="Date", row=n_benchmarks, col=1, title_font={"size": 10})
    fig.update_layout(
        title={
            "text": "Benchmark Regression Analysis<br><sub>(Sorted by p-value: Low→High)</sub>",
            "font": {"size": 16},
        },
        height=max(n_benchmarks * 300, 600),
        hovermode="closest",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
    )

    # Max page index (0-based).  Each benchmark may have a different depth;
    # use the deepest one so the "Older" button is available as long as any
    # benchmark still has unseen data.
    max_page = max((bm["total_points"] + page_size - 1) // page_size - 1 for bm in all_benchmark_data.values())

    # Embed all data as JSON; the JS pagination reads from this.
    data_json = json.dumps(all_benchmark_data)
    names_json = json.dumps([name for name, _, _ in analyzed_benchmarks])

    # Use CDN so Plotly.js (~3.5MB) is browser-cached rather than re-parsed
    # on every load.  Requires an internet connection to view the report.
    plotly_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id="regression-chart",
    )

    css = (
        "body{background:#0f172a;color:#e2e8f0;font-family:sans-serif;margin:0;padding:0}"
        "#pagination-controls{display:flex;align-items:center;gap:16px;padding:12px 20px;"
        "background:#1e293b;border-bottom:1px solid #334155;position:sticky;top:0;z-index:1000}"
        "button{background:#334155;color:#e2e8f0;border:1px solid #475569;padding:8px 16px;"
        "border-radius:6px;cursor:pointer;font-size:14px}"
        "button:hover:not(:disabled){background:#475569}"
        "button:disabled{opacity:.4;cursor:not-allowed}"
        "#page-info{font-size:14px;color:#94a3b8}"
    )

    # JavaScript — note: literal JS braces must be doubled in the f-string.
    js = f"""
const BENCHMARK_DATA = {data_json};
const PAGE_SIZE = {page_size};
const MAX_PAGE = {max_page};
const BENCHMARK_NAMES = {names_json};
const N_BENCHMARKS = {n_benchmarks};
let currentPage = 0;

function getPageSlice(bmData, page) {{
  const total = bmData.data.length;
  const end = total - page * PAGE_SIZE;
  const start = Math.max(0, end - PAGE_SIZE);
  return bmData.data.slice(start, end);
}}

function makeHover(name, unit, d) {{
  return '<b>' + name + '</b><br>' +
    'Version: ' + d.v + '<br>' +
    'Date: ' + d.d.substring(0, 10) + '<br>' +
    'Mean: ' + d.m + ' ' + unit + '<br>' +
    'Min: ' + d.n + ' ' + unit + '<br>' +
    'Max: ' + d.x + ' ' + unit + '<br>' +
    'Std Dev: ' + d.s + ' ' + unit;
}}

function updateChart() {{
  const chartDiv = document.getElementById('regression-chart');
  const xs = [], ys = [], texts = [];
  BENCHMARK_NAMES.forEach(name => {{
    const bm = BENCHMARK_DATA[name];
    const slice = getPageSlice(bm, currentPage);
    xs.push(slice.map(d => d.d));
    ys.push(slice.map(d => d.m));
    texts.push(slice.map(d => makeHover(name, bm.unit_label, d)));
  }});
  Plotly.restyle(chartDiv, {{x: xs, y: ys, hovertext: texts}});
  // Reset all subplot axis ranges so they re-fit the new data window.
  const layoutUpdate = {{}};
  for (let i = 0; i < N_BENCHMARKS; i++) {{
    const xi = i === 0 ? 'xaxis' : 'xaxis' + (i + 1);
    const yi = i === 0 ? 'yaxis' : 'yaxis' + (i + 1);
    layoutUpdate[xi + '.autorange'] = true;
    layoutUpdate[yi + '.autorange'] = true;
  }}
  Plotly.relayout(chartDiv, layoutUpdate);
  updateControls();
}}

function updateControls() {{
  document.getElementById('prev-btn').disabled = currentPage >= MAX_PAGE;
  document.getElementById('next-btn').disabled = currentPage <= 0;
  let label = 'Showing most recent ' + PAGE_SIZE + ' data points';
  if (currentPage > 0) {{
    for (const name of BENCHMARK_NAMES) {{
      const slice = getPageSlice(BENCHMARK_DATA[name], currentPage);
      if (slice.length > 0) {{
        const start = slice[0].d.substring(0, 10);
        const end = slice[slice.length - 1].d.substring(0, 10);
        label = start + ' – ' + end + ' (page ' + (currentPage + 1) + ' of ' + (MAX_PAGE + 1) + ')';
        break;
      }}
    }}
  }}
  document.getElementById('page-info').textContent = label;
}}

function changePage(delta) {{
  const p = currentPage + delta;
  if (p < 0 || p > MAX_PAGE) return;
  currentPage = p;
  updateChart();
}}

updateControls();
"""

    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '  <meta charset="utf-8">\n'
        "  <title>Benchmark Regression Analysis</title>\n"
        f"  <style>{css}</style>\n"
        "</head>\n<body>\n"
        '  <div id="pagination-controls">\n'
        '    <button id="prev-btn" onclick="changePage(1)">← Older</button>\n'
        '    <span id="page-info">Loading…</span>\n'
        '    <button id="next-btn" onclick="changePage(-1)" disabled>Newer →</button>\n'
        "  </div>\n" + plotly_html + f"\n  <script>{js}</script>\n"
        "</body>\n</html>\n"
    )

    print(f"Saving interactive chart to {output_path}...")
    output_path.write_text(html, encoding="utf-8")
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
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Number of data points to display per page in the chart (default: 100)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Maximum number of benchmarks to chart, taken from the top of the p-value ranking (default: 50)",
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

        # Print summary across all benchmarks before truncating for the chart
        print_summary(analyzed_benchmarks, args.threshold)

        # Limit to top-n benchmarks for charting; the rest are summarised above
        # but excluded from the HTML to keep file size and render time manageable.
        if len(analyzed_benchmarks) > args.top_n:
            print(f"\nℹ️  Charting top {args.top_n} of {len(analyzed_benchmarks)} benchmarks (use --top-n to change).")
            analyzed_benchmarks = analyzed_benchmarks[: args.top_n]

        # Create chart
        create_regression_chart(analyzed_benchmarks, args.output, args.page_size)

        print(f"\n✅ Report generated successfully: {args.output}")

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
