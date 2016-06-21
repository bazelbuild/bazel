#!/bin/bash

# Copyright 2015 The Bazel Authors. All rights reserved.
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

set -eu

# Generate the release notes from the git history.

# It uses the RELNOTES tag in the history to knows the important changes to
# report:
#   RELNOTES: indicates a change important the user.
#   RELNOTES[NEW]: introduces a new feature.
#   RELNOTES[INC]: indicates an incompatible change.
# The previous releases base is detected using the CHANGELOG file from the
# repository.
RELNOTES_TYPES=("INC" "NEW" "")
RELNOTES_DESC=("Incompatible changes" "New features" "Important changes")

# Get the baseline version and cherry-picks of the previous release
#  Parameter: $1 is the path to the changelog file
#  Output: "${BASELINE} ${CHERRYPICKS}"
#    BASELINE is the hash of the baseline commit of the latest release
#    CHERRYPICKS is the list of hash of cherry-picked commits of the latest release
#  return 1 if there is no initial release
function get_last_release() {
  local changelog=$1
  [ -f "$changelog" ] || return 1  # No changelog = initial release
  local BASELINE_LINE=$(grep -m 1 -n '^Baseline: ' "$changelog") || return 1
  [ -n "${BASELINE_LINE}" ] || return 1  # No baseline = initial release
  local BASELINE_LINENB=$(echo "${BASELINE_LINE}" | cut -d ":" -f 1)
  BASELINE=$(echo "${BASELINE_LINE}" | cut -d " " -f 2)
  local CHERRYPICK_LINE=$(($BASELINE_LINENB + 3))
  # grep -B999 looks for all lines before the empty line and after that we
  # restrict to only lines with the cherry picked hash then finally we cut
  # the hash.
  local CHERRY_PICKS=$(tail -n +${CHERRYPICK_LINE} "$changelog" \
      | grep -m 1 "^$" -B999 \
      | grep -E '^   \+ [a-z0-9]+:' \
      | cut -d ":" -f 1 | cut -d "+" -f 2)
  echo $BASELINE $CHERRY_PICKS
  return 0
}

# Now get the list of commit with a RELNOTES since latest release baseline ($1)
# discarding cherry_picks ($2..) and rollbacks. The returned list of commits is
# from the oldest to the newest
function get_release_notes_commits() {
  local baseline=$1
  shift
  local cherry_picks="$@"
  local rollback_commits=$(git log --oneline -E --grep='^Rollback of commit [a-z0-9]+.$' ${baseline}.. \
      | grep -E '^[a-z0-9]+ Rollback of commit [a-z0-9]+.$')
  local rollback_hashes=$(echo "$rollback_commits" | cut -d " " -f 1)
  local rolledback_hashes=$(echo "$rollback_commits" | cut -d " " -f 5 | sed -E 's/^(.......).*$/\1/')
  local exclude_hashes=$(echo $cherry_picks $rollback_hashes $rolledback_hashes | xargs echo | sed 's/ /|/g')
  git log --reverse --pretty=format:%h ${baseline}.. -E --grep='^RELNOTES(\[[^\]+\])?:' \
      | grep -Ev "^(${exclude_hashes})"
}

# Extract the release note from a commit hash ($1). It extracts
# the RELNOTES([??]): lines. A new empty line ends the relnotes tag.
# It adds the relnotes, if not "None" ("None.") or "n/a" ("n/a.") to
# the correct array:
#   RELNOTES_INC for incompatible changes
#   RELNOTES_NEW for new features changes
#   RELNOTES for other changes
function extract_release_note() {
  local relnote="$(git show -s $1 --pretty=format:%B | awk '/^RELNOTES(\[[^\]]+\])?:/,/^$/')"
  local regex="^RELNOTES(\[([a-zA-Z]*)\])?:[[:space:]]*([^[:space:]].*[^[:space:]])[[:space:]]*$"
  if [[ "$relnote" =~ $regex ]]; then
      local relnote_kind=${BASH_REMATCH[2]}
      local relnote_text="${BASH_REMATCH[3]}"
      if [[ ! "$(echo $relnote_text | awk '{print tolower($0)}')" =~ ^(none|n/a)?.?$ ]]; then
        eval "RELNOTES_${relnote_kind}+=(\"\${relnote_text}\")"
      fi
  fi
}

# Build release notes arrays from a list of commits ($@) and return the release
# note in an array of array.
function get_release_notes() {
  for i in "${RELNOTES_TYPES[@]}"; do
    eval "RELNOTES_${i}=()"
  done
  for i in $@; do
    extract_release_note $i
  done
}

# fmt behaves differently on *BSD and on GNU/Linux, use fold.
function wrap_text() {
  fold -s -w $1 | sed 's/ *$//'
}

# Returns the list of release notes in arguments into a list of points in
# a markdown list. The release notes are wrapped to 70 characters so it
# displays nicely in a git history.
function format_release_notes() {
  local i
  for (( i=1; $i <= $#; i=$i+1 )); do
    local relnote="${!i}"
    local lines=$(echo "$relnote" | wrap_text 66)  # wrap to 70 counting the 4 leading spaces.
    echo "  - $lines" | head -1
    echo "$lines" | tail -n +2 | sed 's/^/    /'
  done
}

# Create the release notes since commit $1 ($2...${[#]} are the cherry-picks,
# so the commits to ignore.
function release_notes() {
  local i
  local commits=$(get_release_notes_commits $@)
  local length="${#RELNOTES_TYPES[@]}"
  get_release_notes "$commits"
  for (( i=0; $i < $length; i=$i+1 )); do
    local relnotes_title="${RELNOTES_DESC[$i]}"
    local relnotes_type=${RELNOTES_TYPES[$i]}
    local relnotes="RELNOTES_${relnotes_type}[@]"
    local nb_relnotes=$(eval "echo \${#$relnotes}")
    if (( "${nb_relnotes}" > 0 )); then
      echo "${relnotes_title}:"
      echo
      format_release_notes "${!relnotes}"
      echo
    fi
  done
}

# A wrapper around all the previous function, using the CHANGELOG.md
# file in $1 to compute the last release commit hash.
function create_release_notes() {
  local last_release=$(get_last_release "$1") || \
      { echo "Initial release."; return 0; }
  [ -n "${last_release}" ] || { echo "Initial release."; return 0; }
  release_notes ${last_release}
}

# Create the revision information given a list of commits. The first
# commit should be the baseline, and the other one are the cherry-picks.
# The result is of the form:
# Baseline: BASELINE_COMMIT
#
# Cherry picks:
#    + CHERRY_PICK1: commit message summary of the CHERRY_PICK1. This
#                    message will be wrapped into 70 columns.
#    + CHERRY_PICK2: commit message summary of the CHERRY_PICK2.
function create_revision_information() {
  echo "Baseline: $(git rev-parse --short "${1}")"
  first=1
  shift
  while [ -n "${1-}" ]; do
    if [[ "$first" -eq 1 ]]; then
      echo -e "\nCherry picks:"
      first=0
    fi

    local hash="$(git rev-parse --short "${1}")"
    local subject=$(git show -s --pretty=format:%s $hash)
    local lines=$(echo "$subject" | wrap_text 56)  # 14 leading spaces.
    echo "   + $hash: $lines" | head -1
    echo "$lines" | tail -n +2 | sed 's/^/              /'
    shift
  done
}
