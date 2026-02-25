#!/usr/bin/env python3
"""Publish FTS index training benchmark results to Lance dataset."""

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


def parse_fts_index_output(
    output_path: Path,
    testbed_name: str | None = None,
    dut_name: str = "lance",
    dut_version: str | None = None,
    dut_timestamp: int | None = None,
) -> list[Result]:
    """Parse FTS index benchmark JSON output and convert to Result instances.

    Args:
        output_path: Path to the benchmark JSON output file
        testbed_name: Optional testbed name. If not provided, uses hostname
        dut_name: Name of the device under test
        dut_version: Version of the device under test
        dut_timestamp: Build timestamp of the device under test

    Returns:
        List of Result instances
    """
    # Create TestBed from current system
    test_bed = get_test_bed(testbed_name)

    # Validate DUT info
    if not dut_version:
        raise ValueError("DUT version is required. Provide --dut-version argument.")
    if not dut_timestamp:
        raise ValueError("DUT timestamp is required. Provide --dut-timestamp argument.")

    # Create DutBuild
    dut = DutBuild(name=dut_name, version=dut_version, timestamp=dut_timestamp)

    # Parse JSON output
    with open(output_path) as f:
        data = json.load(f)

    results = []
    for entry in data["results"]:
        values_ns = entry["values_ns"]
        duration_ns = entry["duration_ns"]

        # Create summary values from the single measurement
        summary = SummaryValues(
            min=float(duration_ns),
            q1=float(duration_ns),
            median=float(duration_ns),
            q3=float(duration_ns),
            max=float(duration_ns),
            mean=float(duration_ns),
            standard_deviation=0.0,
        )

        # Store RAM metrics and dataset info in metadata
        metadata = {
            "dataset_name": entry["dataset_name"],
            "dataset_description": entry["dataset_description"],
            "num_rows": entry["num_rows"],
            "total_text_bytes": entry["total_text_bytes"],
            "peak_rss_bytes": entry["peak_rss_bytes"],
            "delta_rss_bytes": entry["delta_rss_bytes"],
            "with_position": entry["with_position"],
        }

        result = Result(
            id=str(uuid.uuid4()),
            dut=dut,
            test_bed=test_bed,
            benchmark_name=entry["benchmark_name"],
            values=[float(v) for v in values_ns],
            summary=summary,
            units="ns",
            throughput=None,
            metadata=json.dumps(metadata),
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

    print(f"✓ Uploaded {len(results)} benchmark results to <{get_database_uri()}>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish FTS index training benchmark results to Lance dataset")
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to FTS index benchmark JSON output file",
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
        required=True,
        help="Version of the device under test (required)",
    )
    parser.add_argument(
        "--dut-timestamp",
        type=int,
        required=True,
        help="Build timestamp of the device under test (required, Unix timestamp)",
    )

    args = parser.parse_args()

    if not args.output_path.exists():
        print(f"❌ File not found: {args.output_path}", file=sys.stderr)
        sys.exit(1)

    # Parse results
    print(f"ℹ️ Parsing FTS index benchmark output from {args.output_path}...")
    results = parse_fts_index_output(
        args.output_path,
        args.testbed_name,
        args.dut_name,
        args.dut_version,
        args.dut_timestamp,
    )
    print(f"ℹ️ Found {len(results)} benchmark results")

    if not results:
        print("❌ No benchmark results found in the input file", file=sys.stderr)
        sys.exit(1)

    # Upload results
    print(f"ℹ️ Uploading results to LanceDB at <{get_database_uri()}>...")
    upload_results(results)


if __name__ == "__main__":
    main()
