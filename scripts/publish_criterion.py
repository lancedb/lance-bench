#!/usr/bin/env python3
"""Publish Criterion benchmark results to Lance dataset."""

import argparse
import json
import math
import platform
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add packages to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from lance_bench_db.dataset import connect, get_database_uri
from lance_bench_db.models import DutBuild, Result, SummaryValues, TestBed, Throughput, UnitSystem


def get_test_bed(name: str | None = None) -> TestBed:
    """Create a TestBed instance from the current system information.

    Args:
        name: Optional testbed name. If not provided, uses platform.node()
    """
    import psutil

    return TestBed(
        name=name or platform.node(),
        cpu=platform.processor() or platform.machine(),
        memory_gb=int(psutil.virtual_memory().total / (1024**3)),
        os=f"{platform.system()} {platform.release()}",
        created_at=int(datetime.now().timestamp()),
    )


def parse_criterion_output(
    json_path: Path,
    testbed_name: str | None = None,
    dut_name: str = "lance",
    dut_version: str = "",
    dut_timestamp: int = 0,
) -> list[Result]:
    """Parse Criterion JSON output and convert to Result instances.

    Args:
        json_path: Path to the Criterion JSON output file
        testbed_name: Optional testbed name. If not provided, uses platform.node()
        dut_name: Name of the device under test
        dut_version: Version of the device under test
        dut_timestamp: Build timestamp of the device under test

    Returns:
        List of Result instances
    """
    results = []
    test_bed = get_test_bed(testbed_name)
    dut = DutBuild(name=dut_name, version=dut_version, timestamp=dut_timestamp)

    with open(json_path) as f:
        for line in f:
            message = json.loads(line)

            # We only care about benchmark complete messages
            if message.get("reason") != "benchmark-complete":
                continue

            benchmark_id = message.get("id", "unknown")

            # Get all measured values if available
            measured_values = message.get("measured_values", [])
            if not measured_values:
                # If no measured values, use the point estimate
                raise ValueError("No measured values found")

            iteration_counts = message.get("iteration_count", [])
            if not iteration_counts:
                raise ValueError("No iteration counts found")
            if len(iteration_counts) != len(measured_values):
                raise ValueError(
                    f"Mismatch between measured values and iteration counts: {len(measured_values)} != {len(iteration_counts)}"
                )

            # Note: This reduces the precision of the values somewhat but keeps things simple.
            values = [
                measured_value / iteration_count for measured_value, iteration_count in zip(measured_values, iteration_counts)
            ]

            # Note: Criterion does its own measure of mean (or typically slope) which is more accurate
            # but again, to keep things simple, we re-calculate them purely from the measured values.
            # Otherwise there may be confusion understanding the results.

            # Calculate summary statistics
            sorted_values = sorted(measured_values)
            mean = sum(values) / len(values)
            std_dev = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
            n = len(sorted_values)

            summary = SummaryValues(
                min=float(sorted_values[0]),
                q1=float(sorted_values[n // 4]),
                median=float(sorted_values[n // 2]),
                q3=float(sorted_values[3 * n // 4]),
                max=float(sorted_values[-1]),
                mean=float(mean),
                standard_deviation=float(std_dev),
            )

            # Extract throughput if available
            throughput = None
            if "throughput" in message:
                tp_data = message["throughput"]
                if len(tp_data) != 0:
                    if len(tp_data) > 1:
                        raise ValueError(f"Multiple throughput values found: {tp_data}")
                    throughput = Throughput(
                        multiplier=float(tp_data[0].get("per_iteration", 0.0)),
                        name=tp_data[0].get("unit", "bytes"),
                        unit_system=UnitSystem.BINARY,  # Criterion typically uses binary
                    )

            result = Result(
                id=str(uuid.uuid4()),
                dut=dut,
                test_bed=test_bed,
                benchmark_name=benchmark_id,
                values=measured_values,
                summary=summary,
                units="nanoseconds",  # Criterion uses nanoseconds by default
                throughput=throughput,
                metadata=json.dumps(message),  # Store the full message as metadata
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
    parser = argparse.ArgumentParser(description="Publish Criterion benchmark results to Lance dataset")
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to Criterion JSON output file",
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

    if not args.json_path.exists():
        print(f"Error: File not found: {args.json_path}", file=sys.stderr)
        sys.exit(1)

    # Parse results
    print(f"Parsing Criterion output from {args.json_path}...")
    results = parse_criterion_output(
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
