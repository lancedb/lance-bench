#!/usr/bin/env python3
"""Scheduled benchmark trigger for latest Lance commits.

This script runs on a schedule (4x daily) and:
1. Fetches the latest commit from lance-format/lance repository
2. Checks if benchmark results already exist for that commit
3. Triggers benchmark workflow if no results exist (fire-and-forget)
4. Exits with a non-zero status on any error so the run is visible.
"""

import os
import sys
import time
from pathlib import Path

from github import Auth, Github, RateLimitExceededException

# Add packages to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from lance_bench_db.dataset import connect
from lance_bench_db.models import Result

# Configuration
LANCE_REPO = "lance-format/lance"
WORKFLOW_NAME = "run-benchmarks.yml"


def get_github_client() -> Github:
    """Create and return a GitHub client.

    Returns:
        Github client instance

    Raises:
        ValueError: If GITHUB_TOKEN environment variable is not set
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("❌ GITHUB_TOKEN environment variable must be set")
        sys.exit(1)

    auth = Auth.Token(token)
    return Github(auth=auth)


def get_short_sha(commit_sha: str) -> str:
    """Get the short SHA for a commit.

    Args:
        commit_sha: Full commit SHA

    Returns:
        Short commit SHA (first 7 characters)
    """
    return commit_sha[:7]


def fetch_latest_commit(github_client: Github) -> tuple[str, str, str]:
    """Fetch the latest commit from the lance repository.

    Args:
        github_client: GitHub client instance

    Returns:
        Tuple of (commit_sha, author, message_preview). Exits with non-zero on error.
    """
    try:
        repo = github_client.get_repo(LANCE_REPO)
        latest_commit = repo.get_commits()[0]

        # Get commit details for logging
        commit_sha = latest_commit.sha
        author = latest_commit.commit.author.name
        message_preview = latest_commit.commit.message.split("\n")[0][:60]

        return commit_sha, author, message_preview

    except RateLimitExceededException:
        print("❌ GitHub API rate limit exceeded")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Error fetching latest commit: {e}")
        sys.exit(1)


def has_results_for_commit(commit_sha: str) -> bool:
    """Check if benchmark results already exist for a commit.

    Args:
        commit_sha: Full commit SHA

    Returns:
        True if results exist, False otherwise

    Raises:
        Exception: If unable to connect to database (fails after retries)
    """
    short_sha = get_short_sha(commit_sha)
    last_error = None

    # Retry database connection up to 3 times
    for attempt in range(3):
        try:
            db = connect()
            results_table = Result.open_table(db)

            # Query for any results where the dut.version contains the short SHA
            # The version format is "{VERSION}+{SHORT_SHA}"
            query = results_table.search().where(f"dut.version LIKE '%{short_sha}%'").limit(1)
            results = query.to_list()

            return len(results) > 0

        except Exception as e:
            last_error = e
            if attempt == 2:  # Last attempt
                print(f"❌ Cannot connect to database after 3 attempts: {e}")
                print("   Database connection is required to avoid duplicate benchmark runs")
                print("   Please check LANCE_BENCH_URI and AWS credentials")
                raise

            # Retry with backoff
            wait_time = (attempt + 1) * 2  # 2s, 4s
            print(f"⚠️  Database connection attempt {attempt + 1} failed, retrying in {wait_time}s...")
            time.sleep(wait_time)

    # Should never reach here, but raise exception if all retries failed
    raise Exception(f"Database connection failed after 3 attempts: {last_error}")


def trigger_workflow(github_client: Github, commit_sha: str) -> bool:
    """Trigger the benchmark workflow for a commit.

    Args:
        github_client: GitHub client instance
        commit_sha: Commit SHA to benchmark

    Returns:
        True if triggered successfully, False otherwise
    """
    try:
        lance_bench_repo = os.environ.get("LANCE_BENCH_REPO")
        if not lance_bench_repo:
            print("❌ LANCE_BENCH_REPO environment variable must be set")
            return False

        repo = github_client.get_repo(lance_bench_repo)
        workflow = repo.get_workflow(WORKFLOW_NAME)

        # Trigger the workflow with the commit SHA as input
        success = workflow.create_dispatch(ref="main", inputs={"git_sha": commit_sha})

        if not success:
            print("⚠️  Failed to trigger workflow (API returned False)")
            return False

        # Get workflow URL for logging
        workflow_url = f"https://github.com/{lance_bench_repo}/actions/workflows/{WORKFLOW_NAME}"
        print(f"✓ Triggered benchmarks for {get_short_sha(commit_sha)}")
        print(f"  View runs: {workflow_url}")
        return True

    except Exception as e:
        print(f"⚠️  Failed to trigger workflow: {e}")
        print("   Will retry in next scheduled run")
        return False


def main() -> None:
    """Main entry point for the scheduler script."""
    print("=" * 60)
    print("Lance Benchmark Scheduler")
    print("=" * 60)
    print()

    # Create GitHub client
    github_client = get_github_client()

    # Fetch latest commit
    print(f"Fetching latest commit from {LANCE_REPO}...")
    commit_sha, author, message_preview = fetch_latest_commit(github_client)
    short_sha = get_short_sha(commit_sha)

    print(f"Latest commit: {short_sha}")
    print(f"  Author: {author}")
    print(f"  Message: {message_preview}")
    print()

    # Check if results already exist
    print("Checking for existing results...")
    try:
        has_results = has_results_for_commit(commit_sha)
    except Exception:
        # Error already logged by has_results_for_commit
        sys.exit(1)

    if has_results:
        print(f"ℹ️  Results already exist for {short_sha}, skipping")
        print()
        return

    print(f"No results found for {short_sha}")
    print()

    # Trigger workflow
    print("Triggering benchmark workflow...")
    success = trigger_workflow(github_client, commit_sha)

    if success:
        print()
        print("✅ Benchmark workflow triggered successfully")
    else:
        print()
        print("❌ Workflow trigger failed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
