# Claude Code Guide for lance-bench

This document provides guidance for AI assistants (Claude, GitHub Copilot, etc.) working on the lance-bench repository.

## Project Overview

**Purpose**: Automated benchmark infrastructure for tracking Lance performance over time
**Tech Stack**: Python, GitHub Actions, LanceDB, AWS S3
**Key Dependencies**: PyGithub, LanceDB, PyArrow, pytest-benchmark

## Architecture

### Core Components

1. **GitHub Actions Workflows** (`.github/workflows/`)
   - Automated scheduling (every 6 hours)
   - Reusable benchmark runners for Rust and Python
   - Orchestration workflows

2. **Publishing Scripts** (`scripts/`)
   - Parse benchmark output (Criterion/pytest-benchmark)
   - Transform to common data model
   - Upload to LanceDB

3. **Database Package** (`packages/lance_bench_db/`)
   - Data models (Result, TestBed, DutBuild, SummaryValues)
   - Connection utilities
   - PyArrow schema definitions

### Data Flow

```
New Lance Commit
    ↓
Scheduler (schedule_benchmarks.py)
    ↓
Check Database (has_results_for_commit)
    ↓
Trigger Workflow (run-benchmarks.yml)
    ↓
Run Benchmarks (Rust: Criterion, Python: pytest-benchmark)
    ↓
Publish Results (publish_criterion.py / publish_pytest.py)
    ↓
Store in LanceDB (S3: s3://lance-bench-results)
```

## Code Conventions

### Python Style
- Use type hints for all function signatures
- Prefer `pathlib.Path` over string paths
- Use `None` as default for optional parameters (not `""` or `0`)
- Raise exceptions for invalid states (fail fast)
- Use f-strings for formatting
- Use emoji prefixes in logs: ✓ (success), ℹ️ (info), ⚠️ (warning), ❌ (error)

### Error Handling
```python
# Good: Explicit None default with validation
def func(version: str | None = None) -> str:
    if version is None:
        raise ValueError("Version is required")
    return version

# Bad: Empty string default that silently fails
def func(version: str = "") -> str:
    return version  # Could be empty!
```

### Shared Code
- Extract common utilities to `publish_util.py`
- Both `publish_criterion.py` and `publish_pytest.py` should use shared functions
- Example: `get_test_bed()` is shared between both publishers

## Important Patterns

### 1. Version and Timestamp Format
All results use this format:
- **Version**: `{VERSION}+{SHORT_SHA}` (e.g., `"0.15.0+abc1234"`)
- **Timestamp**: Unix timestamp of the commit
- Extract from: `Cargo.toml` (version) + `git show -s --format=%ct` (timestamp)

### 2. Database Queries
Check for existing results by short SHA:
```python
short_sha = commit_sha[:7]
query = results_table.search().where(f"dut.version LIKE '%{short_sha}%'").limit(1)
results = query.to_list()
has_results = len(results) > 0
```

### 3. Unit Conversion
- **Criterion**: Outputs nanoseconds (keep as-is)
- **pytest-benchmark**: Outputs seconds (convert to nanoseconds)
```python
values = [v * 1_000_000_000 for v in raw_data]  # seconds -> nanoseconds
```

### 4. Workflow Triggering
Cannot use default `GITHUB_TOKEN` to trigger workflows:
```python
# Requires SCHEDULER_GITHUB_TOKEN secret
workflow.create_dispatch(ref="main", inputs={"git_sha": commit_sha})
```

### 5. Retry Logic
Database connections should retry with exponential backoff:
```python
for attempt in range(3):
    try:
        db = connect()
        break
    except Exception as e:
        if attempt == 2:
            raise
        time.sleep((attempt + 1) * 2)
```

## Key Files Reference

### GitHub Actions Workflows
- `schedule-benchmarks.yml` - Cron scheduler (4x daily)
- `run-benchmarks.yml` - Rust benchmark orchestrator
- `run-rust-benchmarks.yml` - Reusable Rust workflow (called per crate)
- `run-python-benchmarks.yml` - Reusable Python workflow
- `lint.yml` - Code quality checks

### Scripts
- `schedule_benchmarks.py` - Check latest commit, trigger if new
- `backfill_benchmarks.py` - Process historical commits
- `publish_criterion.py` - Parse Criterion JSON, publish to DB
- `publish_pytest.py` - Parse pytest-benchmark JSON, publish to DB
- `publish_util.py` - Shared utilities (get_test_bed)

### Database Package
- `packages/lance_bench_db/models.py` - Data models and schema
- `packages/lance_bench_db/dataset.py` - Connection and URI resolution

## Common Tasks

### Adding a New Rust Benchmark
1. Add benchmark to Lance repository using Criterion
2. Edit `run-benchmarks.yml` to add new job:
   ```yaml
   bench-new-crate:
     uses: ./.github/workflows/run-rust-benchmarks.yml
     with:
       git_sha: ${{ inputs.git_sha }}
       crate_path: "rust/new-crate"
     secrets:
       LANCE_BENCH_DB_URI: ${{ secrets.LANCE_BENCH_DB_URI }}
       # ... other secrets
   ```

### Adding a New Python Benchmark
1. Add pytest benchmark to `lance/python/python/ci_benchmarks/benchmarks/`
2. No workflow changes needed (pytest auto-discovers)

### Modifying the Data Model
1. Update `packages/lance_bench_db/models.py`
2. Update PyArrow schema in `Result.to_arrow_table()`
3. Consider migration strategy for existing data

### Debugging Workflow Issues
1. Check GitHub Actions logs in the Actions tab
2. For scheduler: Look for rate limits, auth errors
3. For benchmarks: Check build logs, benchmark output
4. For publishing: Verify database connection, AWS credentials

## Testing Guidelines

### Local Testing
```bash
# Set local database
export LANCE_BENCH_URI="$HOME/.lance-bench"

# Test publishing (without AWS)
uv run python scripts/publish_criterion.py \
  /path/to/criterion-output.json \
  --testbed-name "local-test" \
  --dut-version "test+1234567" \
  --dut-timestamp $(date +%s)

# Test scheduler (requires GitHub token)
export GITHUB_TOKEN="your-token"
uv run python scripts/schedule_benchmarks.py
```

### Integration Testing
- Use manual workflow dispatch in GitHub Actions
- Monitor first few scheduled runs for issues
- Verify results appear in database

## Gotchas and Edge Cases

### 1. GitHub Actions `GITHUB_TOKEN` Limitation
The default `GITHUB_TOKEN` cannot trigger other workflows. Always use `SCHEDULER_GITHUB_TOKEN` for workflow dispatch.

### 2. Maturin Build Requirements
Python benchmarks need Rust toolchain even though they're Python tests (Lance is Rust-backed).

### 3. Dataset Generation
Python benchmarks require datasets to be generated first (`gen_all.py`). This is not idempotent - it creates new datasets each time.

### 4. Benchmark Output Formats
- Criterion: Line-delimited JSON (JSONL)
- pytest-benchmark: Single JSON object
- Both are parsed differently in publish scripts

### 5. Version Validation
Both `dut_version` and `dut_timestamp` are required for publishing. The scripts will raise `ValueError` if either cannot be determined.

### 6. S3 Permissions
LanceDB S3 access requires:
- `s3:GetObject`
- `s3:PutObject`
- `s3:ListBucket`

### 7. Commit SHA Matching
- Database stores short SHA (7 chars) in version string
- Queries use `LIKE '%{short_sha}%'` to match
- Collisions are possible but unlikely in practice

## Best Practices

### When Adding Features
1. Follow existing patterns (especially for publish scripts)
2. Add error handling with clear messages
3. Use type hints and docstrings
4. Test locally before pushing
5. Consider backwards compatibility with existing data

### When Debugging
1. Check environment variables are set
2. Verify AWS credentials have correct permissions
3. Look for rate limiting (GitHub API: 5000 req/hour)
4. Confirm LanceDB connection works locally first

### When Refactoring
1. Keep publish scripts similar in structure
2. Extract common code to `publish_util.py`
3. Maintain backwards compatibility with stored data
4. Update both README.md and this file

## Environment Variables

### Required for Publishing
- `LANCE_BENCH_URI` - Database location (S3 or local)
- `AWS_ACCESS_KEY_ID` - AWS credentials
- `AWS_SECRET_ACCESS_KEY` - AWS credentials

### Required for Scheduler
- `GITHUB_TOKEN` - For GitHub API access and workflow triggering
- `LANCE_BENCH_REPO` - Repository name (usually from `github.repository`)

### Optional
- `MAX_COMMITS` - For backfill script (default: 10)
- `COMMIT_INTERVAL` - For backfill script (default: 1)

## Database Schema Notes

### Result Table
- Primary key: `id` (UUID)
- Indexed by: `dut.version` (contains commit SHA)
- Time-series: `timestamp` field
- Raw data: `values` array (all measurements)
- Aggregates: `summary` struct (min, max, mean, etc.)

### Version Format
Always: `{VERSION}+{SHORT_SHA}`
- VERSION from Cargo.toml (e.g., "0.15.0")
- SHORT_SHA is first 7 chars of commit
- Example: "0.15.0+abc1234"

### Units
All measurements stored in **nanoseconds** for consistency:
- Criterion native: nanoseconds
- pytest-benchmark: convert seconds → nanoseconds

## Questions to Ask When Uncertain

1. **Adding new dependency**: Is it already available in Lance repo?
2. **Changing data model**: How to handle existing data?
3. **New workflow**: Should it be reusable (`workflow_call`) or standalone?
4. **Error handling**: Should this fail fast or retry?
5. **Path references**: Is this relative to lance-bench or lance repo?

## Useful Commands

```bash
# Install dependencies
uv sync

# Run linter
ruff check scripts/

# Format code
ruff format scripts/

# Type check
mypy scripts/

# Test database connection
python -c "from lance_bench_db.dataset import connect; print(connect())"

# Query results for a commit
python -c "
from lance_bench_db.dataset import connect
from lance_bench_db.models import Result
db = connect()
table = Result.open_table(db)
results = table.search().where('dut.version LIKE \"%abc1234%\"').to_list()
print(len(results))
"
```

## Related Documentation

- [Lance GitHub](https://github.com/lancedb/lance)
- [LanceDB Docs](https://lancedb.github.io/lancedb/)
- [Criterion Docs](https://bheisler.github.io/criterion.rs/book/)
- [pytest-benchmark Docs](https://pytest-benchmark.readthedocs.io/)
- [GitHub Actions Docs](https://docs.github.com/en/actions)
