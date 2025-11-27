#!/usr/bin/env bash
set -euo pipefail

# Configuration
LANCE_REPO="lance-format/lance"
WORKFLOW_NAME="run-benchmarks.yml"
MAX_COMMITS=50
COMMIT_INTERVAL=10

echo "Fetching commits from ${LANCE_REPO}..."

# Get the list of commit SHAs from the lance repository
# We need MAX_COMMITS * COMMIT_INTERVAL commits to get every 10th commit
TOTAL_COMMITS=$((MAX_COMMITS * COMMIT_INTERVAL))

# Fetch commits using GitHub API
COMMITS=$(gh api \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/repos/${LANCE_REPO}/commits?per_page=${TOTAL_COMMITS}" \
  --jq '.[].sha')

# Convert to array and take every 10th commit
COMMIT_ARRAY=($COMMITS)
SELECTED_COMMITS=()

for ((i=0; i<${#COMMIT_ARRAY[@]}; i+=COMMIT_INTERVAL)); do
  if [ ${#SELECTED_COMMITS[@]} -ge $MAX_COMMITS ]; then
    break
  fi
  SELECTED_COMMITS+=("${COMMIT_ARRAY[$i]}")
done

echo "Selected ${#SELECTED_COMMITS[@]} commits to benchmark (every ${COMMIT_INTERVAL}th commit)"
echo ""

# Process each commit
for i in "${!SELECTED_COMMITS[@]}"; do
  COMMIT_SHA="${SELECTED_COMMITS[$i]}"
  COMMIT_NUM=$((i + 1))

  echo "[$COMMIT_NUM/${#SELECTED_COMMITS[@]}] Processing commit: $COMMIT_SHA"

  # Trigger the workflow
  echo "  Triggering workflow..."
  gh workflow run "$WORKFLOW_NAME" \
    -f git_sha="$COMMIT_SHA"

  # Wait a moment for the workflow to be created
  sleep 5

  # Get the most recent workflow run ID
  echo "  Waiting for workflow to start..."
  RUN_ID=""
  for attempt in {1..10}; do
    RUN_ID=$(gh run list \
      --workflow="$WORKFLOW_NAME" \
      --limit=1 \
      --json databaseId \
      --jq '.[0].databaseId' 2>/dev/null || echo "")

    if [ -n "$RUN_ID" ]; then
      break
    fi
    echo "    Attempt $attempt: Workflow not found yet, retrying..."
    sleep 3
  done

  if [ -z "$RUN_ID" ]; then
    echo "  ERROR: Could not find workflow run for commit $COMMIT_SHA"
    echo "  Skipping to next commit..."
    continue
  fi

  echo "  Watching workflow run ID: $RUN_ID"

  # Wait for the workflow to complete
  if gh run watch "$RUN_ID" --exit-status; then
    echo "  ✅ Workflow completed successfully"
  else
    echo "  ❌ Workflow failed"
    echo "  Continuing with next commit..."
  fi

  echo ""

  # Small delay between commits
  sleep 2
done

echo "Backfill complete! Processed ${#SELECTED_COMMITS[@]} commits."
