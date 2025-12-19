#!/usr/bin/env python3
"""Process PR comments and determine if benchmarks should run."""

import argparse
import json
import os
import re
import sys

from github import Auth, Github


def parse_benchmark_command(comment_body: str) -> dict | None:
    """Parse benchmark command from comment.

    Returns dict with run configuration or None if not a benchmark command.
    """
    # Check for @bench-bot mention
    if "@bench-bot" not in comment_body.lower():
        return None

    # Check for benchmark command
    if "benchmark" not in comment_body.lower():
        return None

    # Default: run everything
    result = {"run_rust": True, "run_python": True, "crates": []}

    # Parse flags
    if "--rust-only" in comment_body:
        result["run_python"] = False

    if "--python-only" in comment_body:
        result["run_rust"] = False

    # Parse --crate flags
    crate_pattern = r"--crate\s+([\w-]+)"
    crates = re.findall(crate_pattern, comment_body)
    if crates:
        result["crates"] = crates
        result["run_python"] = False  # Specific crates = Rust only

    return result


def is_authorized(github_client: Github, repo_name: str, pr_number: int, username: str) -> bool:
    """Check if user is authorized to trigger benchmarks."""
    repo = github_client.get_repo(repo_name)

    # Check 1: Is user a repository collaborator?
    try:
        permission = repo.get_collaborator_permission(username)
        if permission in ["admin", "write"]:
            print(f"✓ User {username} is a collaborator with {permission} access")
            return True
    except Exception as e:
        print(f"Could not check collaborator status: {e}")

    # Check 2: Is user the PR author?
    try:
        pr = repo.get_pull(pr_number)
        if pr.user.login == username:
            print(f"✓ User {username} is the PR author")
            return True
    except Exception as e:
        print(f"Could not check PR author: {e}")

    print(f"✗ User {username} is not authorized")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Process PR comment for benchmark triggers")
    parser.add_argument("--payload-json", type=str, required=True, help="JSON payload from repository_dispatch event")
    parser.add_argument("--output", type=str, default="command.json", help="Output file for parsed command")

    args = parser.parse_args()

    # Parse payload
    payload = json.loads(args.payload_json)

    # Extract event data from simplified payload
    comment_body = payload.get("comment_body", "")
    username = payload.get("comment_user", "")
    pr_number = payload.get("pr_number")
    commit_sha = payload.get("pr_head_sha", "")
    repo_name = payload.get("repository", "lance-format/lance")

    print(f"Processing comment from {username} on PR #{pr_number}")
    print(f"Comment: {comment_body[:100]}...")

    # Parse command
    command = parse_benchmark_command(comment_body)

    if command is None:
        print("Not a benchmark command, skipping")
        # Set GitHub Actions output
        with open(os.environ.get("GITHUB_OUTPUT", "/dev/stdout"), "a") as f:
            f.write("should_run=false\n")
        sys.exit(0)

    print(f"Parsed command: {command}")

    # Check authorization
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    auth = Auth.Token(token)
    github_client = Github(auth=auth)

    if not is_authorized(github_client, repo_name, pr_number, username):
        print(f"User {username} is not authorized to trigger benchmarks")
        with open(os.environ.get("GITHUB_OUTPUT", "/dev/stdout"), "a") as f:
            f.write("should_run=false\n")
        sys.exit(0)

    # Build configuration
    config = {
        "commit_sha": commit_sha,
        "pr_number": pr_number,
        "pr_repo": repo_name,
        "run_rust": command["run_rust"],
        "run_python": command["run_python"],
        "crates": command["crates"],
        "user": username,
    }

    # Generate summary for acknowledgment
    parts = []
    if command["run_rust"]:
        if command["crates"]:
            parts.append(f"Rust crates: {', '.join(command['crates'])}")
        else:
            parts.append("All Rust benchmarks")
    if command["run_python"]:
        parts.append("Python benchmarks")

    config_summary = " + ".join(parts) if parts else "No benchmarks (invalid config)"

    # Write output file
    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)

    # Set GitHub Actions outputs
    with open(os.environ.get("GITHUB_OUTPUT", "/dev/stdout"), "a") as f:
        f.write("should_run=true\n")
        f.write(f"pr_number={pr_number}\n")
        f.write(f"pr_repo={repo_name}\n")
        f.write(f"commit_sha={commit_sha}\n")
        f.write(f"user={username}\n")
        f.write(f"config_summary={config_summary}\n")
        f.write(f"config={json.dumps(config)}\n")

    print("✓ Command processed successfully")
    sys.exit(0)


if __name__ == "__main__":
    main()
