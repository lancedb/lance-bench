"""Data models for benchmark results."""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import lancedb
import pyarrow as pa


class UnitSystem(Enum):
    """Unit system for throughput measurements."""

    DECIMAL = "decimal"  # KB, MB, GB - powers of 1000
    BINARY = "binary"  # KiB, MiB, GiB - powers of 1024


@dataclass
class Throughput:
    """Represents throughput measurement for a benchmark."""

    multiplier: float
    name: str
    unit_system: UnitSystem


@dataclass
class SummaryValues:
    """Statistical summary of benchmark values."""

    min: float
    q1: float
    median: float
    q3: float
    max: float
    mean: float
    standard_deviation: float


@dataclass
class TestBed:
    """Represents a test bed configuration where benchmarks are executed."""

    name: str
    cpu: str
    memory_gb: int
    os: str
    created_at: int


@dataclass
class DutBuild:
    """The device under test build (e.g. typically lance but may be parquet, vortex, etc.)"""

    name: str
    version: str
    timestamp: int


@dataclass
class Result:
    """Represents a benchmark result."""

    id: str
    dut: DutBuild
    test_bed: TestBed
    benchmark_name: str
    values: list[float]
    summary: SummaryValues
    units: str
    throughput: Optional[Throughput]
    metadata: str
    timestamp: int

    @staticmethod
    def to_arrow_table(results: Iterable["Result"]) -> pa.Table:
        """Convert an iterable of Result instances to a PyArrow table.

        Args:
            results: Iterable of Result instances

        Returns:
            PyArrow Table containing all results with nested struct fields
        """
        results_list = list(results)

        # Define struct types for nested fields
        dut_type = pa.struct(
            [
                ("name", pa.string()),
                ("version", pa.string()),
                ("timestamp", pa.int64()),
            ]
        )

        testbed_type = pa.struct(
            [
                ("name", pa.string()),
                ("cpu", pa.string()),
                ("memory_gb", pa.int32()),
                ("os", pa.string()),
                ("created_at", pa.int64()),
            ]
        )

        summary_type = pa.struct(
            [
                ("min", pa.float64()),
                ("q1", pa.float64()),
                ("median", pa.float64()),
                ("q3", pa.float64()),
                ("max", pa.float64()),
                ("mean", pa.float64()),
                ("standard_deviation", pa.float64()),
            ]
        )

        throughput_type = pa.struct(
            [
                ("multiplier", pa.float64()),
                ("name", pa.string()),
                ("unit_system", pa.string()),
            ]
        )

        schema = pa.schema(
            [
                ("id", pa.string()),
                ("dut", dut_type),
                ("test_bed", testbed_type),
                ("benchmark_name", pa.string()),
                ("values", pa.list_(pa.float64())),
                ("summary", summary_type),
                ("units", pa.string()),
                ("throughput", throughput_type),
                ("metadata", pa.string()),
                ("timestamp", pa.int64()),
            ]
        )

        if not results_list:
            return schema.empty_table()

        # Extract data from results
        data = {
            "id": [r.id for r in results_list],
            "dut": [
                {
                    "name": r.dut.name,
                    "version": r.dut.version,
                    "timestamp": r.dut.timestamp,
                }
                for r in results_list
            ],
            "test_bed": [
                {
                    "name": r.test_bed.name,
                    "cpu": r.test_bed.cpu,
                    "memory_gb": r.test_bed.memory_gb,
                    "os": r.test_bed.os,
                    "created_at": r.test_bed.created_at,
                }
                for r in results_list
            ],
            "benchmark_name": [r.benchmark_name for r in results_list],
            "values": [r.values for r in results_list],
            "summary": [
                {
                    "min": r.summary.min,
                    "q1": r.summary.q1,
                    "median": r.summary.median,
                    "q3": r.summary.q3,
                    "max": r.summary.max,
                    "mean": r.summary.mean,
                    "standard_deviation": r.summary.standard_deviation,
                }
                for r in results_list
            ],
            "units": [r.units for r in results_list],
            "throughput": [
                {
                    "multiplier": r.throughput.multiplier,
                    "name": r.throughput.name,
                    "unit_system": r.throughput.unit_system.value,
                }
                if r.throughput
                else None
                for r in results_list
            ],
            "metadata": [r.metadata for r in results_list],
            "timestamp": [r.timestamp for r in results_list],
        }

        return pa.table(data, schema=schema)

    @staticmethod
    def open_table(db: lancedb.DBConnection) -> lancedb.table.Table:
        """Open or create the results table in the database.

        If the table doesn't exist, it will be created with the appropriate schema.

        Args:
            db: LanceDB database connection

        Returns:
            lancedb.table.Table: The results table
        """
        try:
            return db.open_table("results")
        except Exception:
            # Table doesn't exist, create it with the schema
            empty_table = Result.to_arrow_table([])
            return db.create_table("results", empty_table)
