#!/usr/bin/env python3
"""Publish DBpedia benchmark results to Lance dataset."""

import argparse
import json
import re
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


def parse_dbpedia_output(
    output_path: Path,
    testbed_name: str | None = None,
    dut_name: str = "lance",
    dut_version: str | None = None,
    dut_timestamp: int | None = None,
) -> list[Result]:
    """Parse DBpedia benchmark output and convert to Result instances.

    Args:
        output_path: Path to the DBpedia benchmark output file
        testbed_name: Optional testbed name. If not provided, uses hostname
        dut_name: Name of the device under test
        dut_version: Version of the device under test
        dut_timestamp: Build timestamp of the device under test

    Returns:
        List of Result instances
    """
    results = []

    # Create TestBed from current system
    test_bed = get_test_bed(testbed_name)

    # Validate DUT info
    if not dut_version:
        raise ValueError("DUT version is required. Provide --dut-version argument.")
    if not dut_timestamp:
        raise ValueError("DUT timestamp is required. Provide --dut-timestamp argument.")

    # Create DutBuild
    dut = DutBuild(name=dut_name, version=dut_version, timestamp=dut_timestamp)

    # Regex pattern for query results
    # Example: "IVF256,PQ32: refine=None, recall@100=0.85"
    query_pattern = re.compile(r"IVF(\d+),PQ(\d+):\s+refine=(\w+),\s+recall@(\d+)=([\d.]+)")

    # Parse output file
    with open(output_path) as f:
        for line in f:
            line = line.strip()

            # Match query result lines
            match = query_pattern.match(line)
            if match:
                ivf = int(match.group(1))
                pq = int(match.group(2))
                refine_str = match.group(3)
                k = int(match.group(4))
                recall = float(match.group(5))

                # Convert refine "None" string to actual None
                refine = None if refine_str == "None" else int(refine_str)

                # Create benchmark name
                refine_suffix = "None" if refine is None else str(refine)
                benchmark_name = f"dbpedia_query_IVF{ivf}_PQ{pq}_refine_{refine_suffix}"

                # Create metadata
                metadata = {
                    "ivf": ivf,
                    "pq": pq,
                    "refine": refine,
                    "k": k,
                    "metric": "cosine",  # DBpedia benchmark uses cosine by default
                }

                # Create summary values (single measurement)
                summary = SummaryValues(
                    min=recall,
                    q1=recall,
                    median=recall,
                    q3=recall,
                    max=recall,
                    mean=recall,
                    standard_deviation=0.0,
                )

                # Create result with recall as the measurement value
                result = Result(
                    id=str(uuid.uuid4()),
                    dut=dut,
                    test_bed=test_bed,
                    benchmark_name=benchmark_name,
                    values=[recall],  # Single recall measurement
                    summary=summary,
                    units="recall",
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

    print(f"Successfully uploaded {len(results)} benchmark results to <{get_database_uri()}>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish DBpedia benchmark results to Lance dataset")
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to DBpedia benchmark output file",
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
        print(f"Error: File not found: {args.output_path}", file=sys.stderr)
        sys.exit(1)

    # Parse results
    print(f"Parsing DBpedia benchmark output from {args.output_path}...")
    results = parse_dbpedia_output(
        args.output_path,
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
