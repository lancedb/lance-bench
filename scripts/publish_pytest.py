#!/usr/bin/env python3
"""Publish pytest-benchmark results to Lance dataset."""

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add packages to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from lance_bench_db.dataset import connect, get_database_uri
from lance_bench_db.models import DutBuild, Result, SummaryValues

# Import shared utilities
from publish_util import get_test_bed


def get_dut_from_commit_info(
    commit_info: dict | None,
    dut_name: str = "lance",
    dut_version: str | None = None,
    dut_timestamp: int | None = None,
) -> DutBuild:
    """Create a DutBuild instance from pytest commit_info or provided args.

    Args:
        commit_info: Optional commit info dictionary from pytest-benchmark output
        dut_name: Name of the device under test
        dut_version: Version of the device under test (overrides commit_info)
        dut_timestamp: Build timestamp of the device under test (overrides commit_info)

    Returns:
        DutBuild instance

    Raises:
        ValueError: If version or timestamp cannot be determined from either source
    """
    # Use provided version/timestamp, or extract from commit_info if available
    version = dut_version if dut_version is not None else ""
    timestamp = dut_timestamp if dut_timestamp is not None else 0

    if commit_info and dut_version is None:
        # Extract version from commit ID (short SHA)
        commit_id = commit_info.get("id", "")
        if commit_id:
            version = commit_id[:7]

    if commit_info and dut_timestamp is None:
        # Parse timestamp from commit time
        commit_time = commit_info.get("time", "")
        if commit_time:
            try:
                # Parse ISO format timestamp
                dt = datetime.fromisoformat(commit_time.replace("Z", "+00:00"))
                timestamp = int(dt.timestamp())
            except Exception:
                pass

    # Validate that we have both version and timestamp
    if not version:
        raise ValueError(
            "DUT version could not be determined. Provide --dut-version argument or ensure commit_info contains 'id' field."
        )

    if timestamp == 0:
        raise ValueError(
            "DUT timestamp could not be determined. Provide --dut-timestamp argument or ensure commit_info contains 'time' field."
        )

    return DutBuild(name=dut_name, version=version, timestamp=timestamp)


def parse_pytest_output(
    json_path: Path,
    testbed_name: str | None = None,
    dut_name: str = "lance",
    dut_version: str | None = None,
    dut_timestamp: int | None = None,
) -> list[Result]:
    """Parse pytest-benchmark JSON output and convert to Result instances.

    Args:
        json_path: Path to the pytest-benchmark JSON output file
        testbed_name: Optional testbed name. If not provided, uses hostname
        dut_name: Name of the device under test
        dut_version: Version of the device under test (overrides commit_info)
        dut_timestamp: Build timestamp of the device under test (overrides commit_info)

    Returns:
        List of Result instances
    """
    results = []

    with open(json_path) as f:
        data = json.load(f)

    # Create TestBed from current system
    test_bed = get_test_bed(testbed_name)

    # Extract commit info and create DutBuild
    commit_info = data.get("commit_info")
    dut = get_dut_from_commit_info(commit_info, dut_name, dut_version, dut_timestamp)

    # Parse each benchmark
    benchmarks = data.get("benchmarks", [])
    for benchmark in benchmarks:
        # Get benchmark metadata
        benchmark_name = benchmark.get("fullname", benchmark.get("name", "unknown"))

        # Get stats
        stats = benchmark.get("stats", {})
        if not stats:
            print(f"Warning: No stats found for benchmark {benchmark_name}, skipping")
            continue

        # Get raw data values (in seconds)
        raw_data = stats.get("data", [])
        if not raw_data:
            print(f"Warning: No data found for benchmark {benchmark_name}, skipping")
            continue

        # Convert seconds to nanoseconds for consistency with Criterion
        values = [v * 1_000_000_000 for v in raw_data]

        # Extract summary statistics (also convert to nanoseconds)
        summary = SummaryValues(
            min=float(stats.get("min", 0.0) * 1_000_000_000),
            q1=float(stats.get("q1", 0.0) * 1_000_000_000),
            median=float(stats.get("median", 0.0) * 1_000_000_000),
            q3=float(stats.get("q3", 0.0) * 1_000_000_000),
            max=float(stats.get("max", 0.0) * 1_000_000_000),
            mean=float(stats.get("mean", 0.0) * 1_000_000_000),
            standard_deviation=float(stats.get("stddev", 0.0) * 1_000_000_000),
        )

        # Create result
        result = Result(
            id=str(uuid.uuid4()),
            dut=dut,
            test_bed=test_bed,
            benchmark_name=benchmark_name,
            values=values,
            summary=summary,
            units="nanoseconds",  # Converted from seconds to nanoseconds
            throughput=None,  # pytest-benchmark doesn't typically track throughput
            metadata=json.dumps(benchmark),  # Store the full benchmark data as metadata
            timestamp=int(datetime.now().timestamp()),
        )

        results.append(result)

    return results


def upload_results(results: list[Result]) -> None:
    """Upload results to the LanceDB table.

    Args:
        results: List of Result instances to upload
    """
    # Convert results to Arrow table using the Result class method
    table = Result.to_arrow_table(results)

    # Connect to database and open results table
    db = connect()
    results_table = Result.open_table(db)

    # Add data to table
    results_table.add(table)

    print(f"Successfully uploaded {len(results)} benchmark results to <{get_database_uri()}>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish pytest-benchmark results to Lance dataset")
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to pytest-benchmark JSON output file",
    )
    parser.add_argument(
        "--testbed-name",
        type=str,
        default=None,
        help="Name of the testbed (defaults to hostname)",
    )
    parser.add_argument(
        "--dut-name",
        type=str,
        default="lance",
        help="Name of the device under test (defaults to 'lance')",
    )
    parser.add_argument(
        "--dut-version",
        type=str,
        default=None,
        help="Version of the device under test (defaults to commit SHA from JSON if available)",
    )
    parser.add_argument(
        "--dut-timestamp",
        type=int,
        default=None,
        help="Build timestamp of the device under test (defaults to commit time from JSON if available, Unix timestamp)",
    )

    args = parser.parse_args()

    if not args.json_path.exists():
        print(f"Error: File not found: {args.json_path}", file=sys.stderr)
        sys.exit(1)

    # Parse results
    print(f"Parsing pytest-benchmark output from {args.json_path}...")
    results = parse_pytest_output(
        args.json_path,
        args.testbed_name,
        args.dut_name,
        args.dut_version,
        args.dut_timestamp,
    )
    print(f"Found {len(results)} benchmark results")

    if not results:
        print("No benchmark results found in the input file", file=sys.stderr)
        sys.exit(1)

    # Upload results
    print(f"Uploading results to LanceDB at <{get_database_uri()}>...")
    upload_results(results)


if __name__ == "__main__":
    main()
