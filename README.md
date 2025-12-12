# Lance Benchmark Infrastructure

Automated benchmark tracking and performance monitoring for [Lance](https://github.com/lance-format/lance)

## Overview

This repository provides continuous performance monitoring for Lance by:

- Running benchmarks automatically on new commits
- Tracking performance metrics over time in LanceDB
- Supporting both Rust (Criterion) and Python (pytest-benchmark) benchmarks
- Providing historical backfill capabilities

## Architecture

### Benchmark Workflows

#### Automated Scheduling

- **Schedule Benchmarks** (`schedule-benchmarks.yml`) - Runs every 6 hours
  - Fetches latest commit from lance-format/lance
  - Checks if results exist in the database
  - Triggers benchmark runs for new commits

#### Benchmark Execution

- **Run Rust Benchmarks** (`run-rust-benchmarks.yml`) - Reusable workflow

  - Runs Criterion benchmarks for a specific Rust crate
  - Currently benchmarks: lance-io, lance-linalg, lance-encoding
  - Publishes results using `publish_criterion.py`

- **Run Python Benchmarks** (`run-python-benchmarks.yml`) - Reusable workflow
  - Builds Lance Python package with maturin
  - Generates test datasets
  - Runs pytest benchmarks
  - Publishes results using `publish_pytest.py`

### Data Storage

Results are stored in a LanceDB database with the following schema:

- **TestBed**: System information (CPU, memory, OS)
- **DutBuild**: Device Under Test (name, version, commit timestamp)
- **Result**: Benchmark results with statistics and raw values
- **SummaryValues**: Min, max, mean, median, quartiles, std dev
- **Throughput**: Optional throughput metrics

Database location: `s3://lance-bench-results` (or `~/.lance-bench` locally)

### Scripts

#### Publishing Scripts

- **`publish_criterion.py`** - Parse and publish Rust Criterion benchmark results
- **`publish_pytest.py`** - Parse and publish Python pytest-benchmark results
- **`publish_util.py`** - Shared utilities (TestBed creation)

#### Automation Scripts

- **`schedule_benchmarks.py`** - Check for new commits and trigger benchmarks
- **`backfill_benchmarks.py`** - Backfill results for historical commits

## Setup

### Prerequisites

- Python 3.12+
- uv (Python package manager)
- AWS credentials (for S3 access to results database)

### Installation

```bash
# Install dependencies
uv sync

# Set environment variables
export LANCE_BENCH_URI="s3://lance-bench-results"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

### GitHub Actions Secrets

Required secrets for CI workflows:

- `LANCE_BENCH_DB_URI` - Database URI (S3 or local path)
- `BENCH_S3_USER_ACCESS_KEY` - AWS access key
- `BENCH_S3_USER_SECRET_KEY` - AWS secret key
- `SCHEDULER_GITHUB_TOKEN` - GitHub PAT with `actions:write` and `contents:read` permissions

> **Note**: The default `GITHUB_TOKEN` cannot trigger other workflows, so `SCHEDULER_GITHUB_TOKEN` is required for the scheduler.

## Usage

### Publishing Benchmark Results

#### Criterion (Rust) Results

```bash
# Run Criterion benchmarks with JSON output
cd lance/rust/lance-io
cargo criterion --benches --message-format=json > criterion-output.json

# Publish results
uv run python scripts/publish_criterion.py \
  criterion-output.json \
  --testbed-name "my-machine" \
  --dut-name "lance" \
  --dut-version "0.15.0+abc1234" \
  --dut-timestamp 1702345678
```

#### pytest-benchmark (Python) Results

```bash
# Run pytest benchmarks with JSON output
pytest benchmarks/ --benchmark-json=pytest-output.json --benchmark-only

# Publish results
uv run python scripts/publish_pytest.py \
  pytest-output.json \
  --testbed-name "my-machine" \
  --dut-name "lance" \
  --dut-version "0.15.0+abc1234" \
  --dut-timestamp 1702345678
```

> **Note**: Both `--dut-version` and `--dut-timestamp` are required. For pytest, these can be auto-extracted from `commit_info` in the JSON if available.

### Backfilling Historical Results

```bash
# Set required environment variables
export GITHUB_TOKEN="your-github-token"
export LANCE_BENCH_URI="s3://lance-bench-results"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

# Run backfill script
uv run python scripts/backfill_benchmarks.py
```

Configuration in `backfill_benchmarks.py`:

- `MAX_COMMITS` - Number of commits to process (default: 10)
- `COMMIT_INTERVAL` - Process every Nth commit (default: 1)

### Manual Workflow Triggers

Trigger workflows manually via GitHub Actions UI:

1. Go to Actions tab
2. Select workflow (e.g., "Schedule Benchmarks for New Lance Commits")
3. Click "Run workflow"

## Development

### Project Structure

```
lance-bench/
├── .github/workflows/          # GitHub Actions workflows
│   ├── schedule-benchmarks.yml # Automated scheduler (runs 4x daily)
│   ├── run-benchmarks.yml      # Orchestrator for Rust benchmarks
│   ├── run-rust-benchmarks.yml # Reusable Rust benchmark workflow
│   └── run-python-benchmarks.yml # Reusable Python benchmark workflow
├── scripts/                    # Python scripts
│   ├── publish_criterion.py    # Publish Rust benchmark results
│   ├── publish_pytest.py       # Publish Python benchmark results
│   ├── publish_util.py         # Shared publishing utilities
│   ├── schedule_benchmarks.py  # Scheduler script
│   └── backfill_benchmarks.py  # Backfill historical results
├── packages/
│   └── lance_bench_db/         # Database package
│       ├── models.py           # Data models (Result, TestBed, etc.)
│       └── dataset.py          # Database connection utilities
├── pyproject.toml              # Python dependencies
└── uv.lock                     # Locked dependencies
```

### Adding New Benchmarks

#### Rust Benchmarks

1. Add Criterion benchmarks to the Lance repository
2. Update `run-benchmarks.yml` to include the new crate path

#### Python Benchmarks

1. Add pytest benchmarks to `lance/python/python/ci_benchmarks/benchmarks/`
2. Python benchmarks are automatically discovered by pytest

### Database Schema

Results are stored with this structure:

```python
Result(
    id: str,                    # UUID
    dut: DutBuild,              # Device info (name, version, timestamp)
    test_bed: TestBed,          # System info (CPU, memory, OS)
    benchmark_name: str,        # Full benchmark name
    values: list[float],        # Raw measurements (nanoseconds)
    summary: SummaryValues,     # Statistical summary
    units: str,                 # "nanoseconds"
    throughput: Throughput?,    # Optional throughput info
    metadata: str,              # JSON string of full benchmark data
    timestamp: int              # Unix timestamp when result was created
)
```

## Troubleshooting

### Common Issues

**"DUT version could not be determined"**

- Ensure `--dut-version` is provided or `commit_info.id` exists in pytest JSON

**"Database connection failed"**

- Check AWS credentials are set correctly
- Verify `LANCE_BENCH_URI` is accessible
- For S3: Ensure IAM permissions include `s3:GetObject`, `s3:PutObject`

**Scheduler workflow not triggering benchmarks**

- Verify `SCHEDULER_GITHUB_TOKEN` secret is set with correct permissions
- Check workflow logs for API rate limits or authentication errors

**Benchmarks failing to build**

- Rust benchmarks: Ensure protobuf-compiler is installed
- Python benchmarks: Verify maturin build succeeds with Rust toolchain

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with local database (`~/.lance-bench`)
5. Submit a pull request

## License

Apache License 2.0 - See LICENSE file for details

## Related Projects

- [Lance](https://github.com/lancedb/lance) - The Lance columnar data format
- [LanceDB](https://github.com/lancedb/lancedb) - Vector database built on Lance
