# Bench-Bot: PR Comment Benchmark Trigger

## Overview

Bench-bot is a GitHub Actions-based bot that allows users to trigger benchmarks on PRs in the lance-format/lance repository by posting comments. The bot validates authorization, parses command options, posts acknowledgment comments, and triggers the appropriate benchmarks.

## Architecture

The system uses a two-stage workflow architecture:

1. **Stage 1 (lance repository)**: [benchmark-comment-trigger.yml](https://github.com/lance-format/lance/.github/workflows/benchmark-comment-trigger.yml)

   - Listens for `issue_comment` events on PRs
   - Filters for comments containing both `@bench-bot` AND `benchmark`
   - Fetches PR head SHA via GitHub API
   - Forwards to lance-bench via `repository_dispatch`

2. **Stage 2 (lance-bench repository)**: [comment-monitor.yml](/.github/workflows/comment-monitor.yml)
   - Receives `repository_dispatch` events
   - Validates user authorization (collaborator or PR author)
   - Parses command options and configuration
   - Posts acknowledgment comment to PR
   - Triggers [run-benchmarks.yml](/.github/workflows/run-benchmarks.yml) with parsed configuration

## Workflow Sequence

```
PR Comment in lance-format/lance
    ↓
benchmark-comment-trigger.yml (if contains @bench-bot + benchmark)
    ↓
Forward via repository_dispatch
    ↓
comment-monitor.yml in lance-bench
    ↓
scripts/process_pr_comment.py (parse + authorize)
    ↓
scripts/post_pr_comment.py (acknowledgment)
    ↓
run-benchmarks.yml (execute benchmarks)
    ↓
scripts/compare_and_comment.py (post results)
```

## Supported Commands

| Command                                                      | Effect                                 |
| ------------------------------------------------------------ | -------------------------------------- |
| `@bench-bot benchmark`                                       | Run all benchmarks (Rust + Python)     |
| `@bench-bot benchmark --rust-only`                           | Run all Rust benchmarks only           |
| `@bench-bot benchmark --python-only`                         | Run Python benchmarks only             |
| `@bench-bot benchmark --crate lance-io`                      | Run benchmarks for lance-io crate only |
| `@bench-bot benchmark --crate lance-io --crate lance-linalg` | Run benchmarks for specific crates     |

## Authorization

Users are authorized to trigger benchmarks if they are:

- Repository collaborators with `write` or `admin` access, OR
- The author of the PR

Authorization is validated by [scripts/process_pr_comment.py](/scripts/process_pr_comment.py) using the GitHub API.

## Required Secrets

### lance-format/lance repository

- `LANCE_BENCH_DISPATCH_TOKEN`: PAT with `contents:write` and `actions:write` scopes to trigger `repository_dispatch` events in lancedb/lance-bench

### lance-bench repository

- `PR_COMMENT_TOKEN`: PAT with `issues:write` and `pull-requests:write` scopes for lance-format/lance to read PRs, check collaborators, and post comments
- All existing benchmark secrets (LANCE_BENCH_DB_URI, AWS credentials, etc.)
