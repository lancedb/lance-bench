#!/usr/bin/env python3
"""Merge multiple Lance tables into a single table"""

import argparse
from pathlib import Path

import lance


def merge_lance_tables(table_paths: list[Path], output: Path) -> None:
    """Merge multiple Lance tables into a single table"""
    rows_merged = 0
    for table_path in table_paths:
        dataset = lance.dataset(table_path)
        rows_merged += dataset.count_rows()
        lance.write_dataset(dataset, output, mode="append")
    print(f"Merged {rows_merged} rows into {output}")


def find_lance_tables(results: list[Path]) -> list[Path]:
    """Find all Lance tables in the given directory"""
    return [p for p in results if p.is_dir()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple Lance tables into a single table")
    parser.add_argument(
        "results",
        type=Path,
        nargs="+",
        help="Directory containing Lance tables to merge",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Lance table",
    )

    args = parser.parse_args()

    tables = find_lance_tables(args.results)

    print(f"Found {len(tables)} Lance tables to merge")

    # Merge Lance tables
    merge_lance_tables(tables, args.output)


if __name__ == "__main__":
    main()
