#!/usr/bin/env bash
#
# Copyright 2026 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Load trusted contributors as JSON array
TRUSTED_JSON=$(yq e '.trusted_contributors' .github/config/trusted_contributors.yaml -o=json)

# Fetch open PRs
PRS=$(gh pr list --state open --limit 500 --json number,labels)

echo "$PRS" | jq -c '.[]' | while read -r pr; do
  PR_NUMBER=$(echo "$pr" | jq -r '.number')
  HAS_LABEL=$(echo "$pr" | jq -r '.labels[]?.name' | grep -c "^community-reviewed$" || true)

  # Fetch reviews for this PR
  REVIEWS=$(gh api "repos/bazelbuild/bazel/pulls/$PR_NUMBER/reviews" --jq '.[] | {user: .user.login, state: .state, submitted_at: .submitted_at}' || echo "[]")

  if [[ -z "$REVIEWS" || "$REVIEWS" == "[]" ]]; then
    continue
  fi

  # Filter trusted reviews and get the latest one
  LATEST_TRUSTED_REVIEW=$(echo "$REVIEWS" | jq -c --argjson trusted "$TRUSTED_JSON" '
    select(.user as $u | $trusted | index($u) != null)
  ' | jq -s 'sort_by(.submitted_at) | last' || echo "null")

  if [[ -z "$LATEST_TRUSTED_REVIEW" || "$LATEST_TRUSTED_REVIEW" == "null" ]]; then
    continue
  fi

  STATE=$(echo "$LATEST_TRUSTED_REVIEW" | jq -r '.state')

  if [[ "$STATE" == "APPROVED" ]]; then
    if [[ "$HAS_LABEL" -eq 0 ]]; then
      echo "Adding label to PR #$PR_NUMBER"
      gh pr edit "$PR_NUMBER" --add-label "community-reviewed"
    fi
  elif [[ "$STATE" == "CHANGES_REQUESTED" || "$STATE" == "DISMISSED" ]]; then
    if [[ "$HAS_LABEL" -gt 0 ]]; then
      echo "Removing label from PR #$PR_NUMBER"
      gh pr edit "$PR_NUMBER" --remove-label "community-reviewed"
    fi
  fi
done
