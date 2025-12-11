#!/usr/bin/env python3
"""Backfill benchmark results for historical commits.

This script:
1. Fetches historical commits from the lance-format/lance repository
2. Checks if results already exist for each commit
3. Triggers benchmark workflows for commits without results
4. Waits for each workflow to complete before proceeding
"""

import os
import sys
import time
from pathlib import Path

from github import Auth, Github

# Add packages to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from lance_bench_db.dataset import connect
from lance_bench_db.models import Result

# Configuration
LANCE_REPO = "lance-format/lance"
WORKFLOW_NAME = "run-benchmarks.yml"
MAX_COMMITS = 50
COMMIT_INTERVAL = 10


def get_github_client() -> Github:
    """Create and return a GitHub client.

    Returns:
        Github client instance

    Raises:
        ValueError: If GITHUB_TOKEN environment variable is not set
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN environment variable must be set")

    auth = Auth.Token(token)
    return Github(auth=auth)


def fetch_commits(github_client: Github) -> list[str]:
    """Fetch commits from the lance repository.

    Args:
        github_client: GitHub client instance

    Returns:
        List of commit SHAs
    """
    print(f"Fetching commits from {LANCE_REPO}...")

    total_commits = MAX_COMMITS * COMMIT_INTERVAL
    print(f"Fetching {total_commits} commits...")

    repo = github_client.get_repo(LANCE_REPO)
    commits = []

    # Get commits from the default branch
    for commit in repo.get_commits():
        commits.append(commit.sha)
        if len(commits) >= total_commits:
            break

    print(f"Fetched {len(commits)} commits")
    return commits


def select_commits(commits: list[str]) -> list[str]:
    """Select every Nth commit from the list.

    Args:
        commits: List of all commit SHAs

    Returns:
        List of selected commit SHAs
    """
    selected = []
    for i in range(0, len(commits), COMMIT_INTERVAL):
        if len(selected) >= MAX_COMMITS:
            break
        selected.append(commits[i])

    print(f"Selected {len(selected)} commits to benchmark (every {COMMIT_INTERVAL}th commit)")
    print()
    return selected


def get_short_sha(commit_sha: str) -> str:
    """Get the short SHA for a commit.

    Args:
        commit_sha: Full commit SHA

    Returns:
        Short commit SHA (first 7 characters)
    """
    return commit_sha[:7]


def has_results_for_commit(commit_sha: str) -> bool:
    """Check if benchmark results already exist for a commit.

    Args:
        commit_sha: Full commit SHA

    Returns:
        True if results exist, False otherwise
    """
    short_sha = get_short_sha(commit_sha)

    try:
        db = connect()
        results_table = Result.open_table(db)

        # Query for any results where the dut.version contains the short SHA
        # The version format is "{VERSION}+{SHORT_SHA}"
        query = results_table.search().where(f"dut.version LIKE '%{short_sha}%'").limit(1)
        results = query.to_list()

        return len(results) > 0
    except Exception as e:
        print(f"  Warning: Error checking for existing results: {e}")
        return False


def trigger_workflow(github_client: Github, commit_sha: str) -> int | None:
    """Trigger the benchmark workflow for a commit.

    Args:
        github_client: GitHub client instance
        commit_sha: Commit SHA to benchmark

    Returns:
        Workflow run ID if successful, None otherwise
    """
    print("  Triggering workflow...")

    try:
        repo = github_client.get_repo(LANCE_REPO)
        workflow = repo.get_workflow(WORKFLOW_NAME)

        # Trigger the workflow with the commit SHA as input
        success = workflow.create_dispatch(ref="main", inputs={"git_sha": commit_sha})

        if not success:
            print("  ERROR: Failed to trigger workflow")
            return None

    except Exception as e:
        print(f"  ERROR: Failed to trigger workflow: {e}")
        return None

    # Wait a moment for the workflow to be created
    time.sleep(5)

    # Get the most recent workflow run ID
    print("  Waiting for workflow to start...")
    for attempt in range(1, 11):
        try:
            # Get the most recent workflow run for this workflow
            runs = workflow.get_runs(event="workflow_dispatch")
            if runs.totalCount > 0:
                latest_run = runs[0]
                return latest_run.id

        except Exception as e:
            print(f"    Attempt {attempt}: Error getting workflow run: {e}")

        print(f"    Attempt {attempt}: Workflow not found yet, retrying...")
        time.sleep(3)

    print(f"  ERROR: Could not find workflow run for commit {commit_sha}")
    return None


def wait_for_workflow(github_client: Github, run_id: int) -> bool:
    """Wait for a workflow to complete.

    Args:
        github_client: GitHub client instance
        run_id: Workflow run ID

    Returns:
        True if workflow succeeded, False otherwise
    """
    print(f"  Watching workflow run ID: {run_id}")

    repo = github_client.get_repo(LANCE_REPO)

    try:
        # Poll the workflow run status
        while True:
            run = repo.get_workflow_run(run_id)

            if run.status == "completed":
                if run.conclusion == "success":
                    print("  ✅ Workflow completed successfully")
                    return True
                else:
                    print(f"  ❌ Workflow failed with conclusion: {run.conclusion}")
                    print("  Continuing with next commit...")
                    return False

            # Wait before checking again
            time.sleep(10)

    except KeyboardInterrupt:
        print("  Interrupted while waiting for workflow")
        raise
    except Exception as e:
        print(f"  ERROR: Failed to watch workflow: {e}")
        return False


def process_commit(github_client: Github, commit_sha: str, commit_num: int, total_commits: int) -> None:
    """Process a single commit: check for results and run workflow if needed.

    Args:
        github_client: GitHub client instance
        commit_sha: Commit SHA to process
        commit_num: Current commit number (1-indexed)
        total_commits: Total number of commits to process
    """
    print(f"[{commit_num}/{total_commits}] Processing commit: {commit_sha}")

    # Check if results already exist
    if has_results_for_commit(commit_sha):
        short_sha = get_short_sha(commit_sha)
        print(f"  ℹ️  Results already exist for commit {short_sha}, skipping...")
        print()
        return

    # Trigger the workflow
    run_id = trigger_workflow(github_client, commit_sha)

    if run_id is None:
        print("  Skipping to next commit...")
        print()
        return

    # Wait for the workflow to complete
    wait_for_workflow(github_client, run_id)
    print()

    # Small delay between commits
    time.sleep(2)


def main() -> None:
    """Main entry point for the backfill script."""
    try:
        # Create GitHub client
        github_client = get_github_client()

        # Fetch and select commits
        commits = fetch_commits(github_client)
        selected_commits = select_commits(commits)

        # Process each commit
        for i, commit_sha in enumerate(selected_commits, start=1):
            process_commit(github_client, commit_sha, i, len(selected_commits))

        print(f"Backfill complete! Processed {len(selected_commits)} commits.")

    except KeyboardInterrupt:
        print("\n\nBackfill interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
