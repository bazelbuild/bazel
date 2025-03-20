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

RELNOTES_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source ${RELNOTES_SCRIPT_DIR}/common.sh

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
function __get_last_release() {
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
function __get_release_notes_commits() {
  local baseline=$1
  shift
  local cherry_picks="$@"
  local rollback_commits=$(git log --oneline -E --grep='^Rollback of commit [a-z0-9]+.$' ${baseline}.. \
      | grep -E '^[a-z0-9]+ Rollback of commit [a-z0-9]+.$' || true)
  local rollback_hashes=$(echo "$rollback_commits" | cut -d " " -f 1)
  local rolledback_hashes=$(echo "$rollback_commits" | cut -d " " -f 5 | sed -E 's/^(.......).*$/\1/')
  local exclude_hashes=$(echo DUMMY $cherry_picks $rollback_hashes $rolledback_hashes | xargs echo | sed 's/ /|/g')
  git log --reverse --pretty=format:%H ${baseline}.. -E --grep='^RELNOTES(\[[^\]+\])?:' \
      | grep -Ev "^(${exclude_hashes})" || true
}

# Extract the release note from a commit hash ($1). It extracts
# the RELNOTES([??]): lines. A new empty line ends the relnotes tag.
# It adds the relnotes, if not "None" ("None.") or "n/a" ("n/a.") to
# the correct array:
#   RELNOTES_INC for incompatible changes
#   RELNOTES_NEW for new features changes
#   RELNOTES for other changes
function __extract_release_note() {
  local find_relnote_awk_script="
    BEGIN { in_relnote = 0 }
    /^$/ { in_relnote = 0 }
    /^PiperOrigin-RevId:.*$/ { in_relnote = 0 }
    /^RELNOTES(\[[^\]]+\])?:/ { in_relnote = 1 }
    { if (in_relnote) { print } }"
  local relnote="$(git show -s $1 --pretty=format:%B | awk "${find_relnote_awk_script}")"
  local regex="^RELNOTES(\[([a-zA-Z]*)\])?:[[:space:]]*([^[:space:]].*[^[:space:]])[[:space:]]*$"
  if [[ "$relnote" =~ $regex ]]; then
      local relnote_kind=${BASH_REMATCH[2]}
      local relnote_text="${BASH_REMATCH[3]}"
      if [[ ! "$(echo $relnote_text | awk '{print tolower($0)}')" =~ ^((none|n/a|no)(\.( .*)?)?|\.)$ ]]; then
        eval "RELNOTES_${relnote_kind}+=(\"\${relnote_text}\")"
      fi
  fi
}

# Build release notes arrays from a list of commits ($@) and return the release
# note in an array of array.
function __generate_release_notes() {
  for i in "${RELNOTES_TYPES[@]}"; do
    eval "RELNOTES_${i}=()"
  done
  for commit in $@; do
    __extract_release_note "${commit}"
  done
}

# Returns the list of release notes in arguments into a list of points in
# a markdown list. The release notes are wrapped to 70 characters so it
# displays nicely in a git history.
function __format_release_notes() {
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
function __release_notes() {
  local last_release=$1
  local i
  local commits=$(__get_release_notes_commits $last_release)
  local length="${#RELNOTES_TYPES[@]}"
  __generate_release_notes "$commits"
  for (( i=0; $i < $length; i=$i+1 )); do
    local relnotes_title="${RELNOTES_DESC[$i]}"
    local relnotes_type=${RELNOTES_TYPES[$i]}
    local relnotes="RELNOTES_${relnotes_type}[@]"
    local nb_relnotes=$(eval "echo \${#$relnotes}")
    if (( "${nb_relnotes}" > 0 )); then
      echo "${relnotes_title}:"
      echo
      __format_release_notes "${!relnotes}"
      echo
    fi
  done

  # Add a list of contributors to thank.
  # Stages:
  #   1. Get the list of authors from the last release til now, both name and
  #     email.
  #   2. Sort and uniqify.
  #   3. Remove googlers. (This is why the email is needed)
  #   4. Cut the email address, leaving only the name.
  #   5-n. Remove trailing spaces and newlines, substituting with a comman and a
  #     space, removing any trailing spaces again.
  local external_authors=$(git log $last_release..HEAD --format="%aN <%aE>" \
    | sort \
    | uniq \
    | grep -v "google.com" \
    | cut -d'<' -f 1 \
    | sed -e 's/[[:space:]]$//' \
    | tr '\n' ',' \
    | sed -e 's/,$/\n/' \
    | sed -e 's/,/, /g')
  echo "This release contains contributions from many people at Google, as well as ${external_authors}."
}

# A wrapper around all the previous function, using the CHANGELOG.md
# file in $1 to compute the last release commit hash.
function create_release_notes() {
  local last_release=$(__get_last_release "$1") || \
      { echo "Initial release."; return 0; }
  [ -n "${last_release}" ] || { echo "Initial release."; return 0; }
  __release_notes ${last_release}
}

# Trim empty lines at the beginning and the end of the buffer.
function __trim_empty_lines() {
  # Replace all new line by a linefeed, then using sed, remove the leading
  # and trailing linefeeds and convert them back to newline
  tr '\n' '\f' | sed -e "s/^\f*//" -e "s/\f*$//" | tr '\f' '\n'
}

# Launch the editor and return the edited release notes.
function __release_note_processor() {
  local tmpfile="$1"

  # Strip the release notes.
  local relnotes="$(cat ${tmpfile} | grep -v '^#' | __trim_empty_lines)"
  if [ -z "${relnotes}" ]; then
    echo "Release notes are empty, cancelling release creation..." >&2
    return 1
  fi

  echo "${relnotes}" > "${tmpfile}"
}

# Create the revision information given a list of commits. The first
# commit should be the baseline, and the other ones are the cherry-picks.
# The result is of the form:
# Baseline: BASELINE_COMMIT
#
# Cherry picks:
#
#    + CHERRY_PICK1: commit message summary of the CHERRY_PICK1. This
#                    message will be wrapped into 70 columns.
#    + CHERRY_PICK2: commit message summary of the CHERRY_PICK2.
function __create_revision_information() {
  echo "Baseline: $(__git_commit_hash "${1}")"
  local first=1
  shift
  while [ -n "${1-}" ]; do
    if [[ "$first" -eq 1 ]]; then
      echo -e "\nCherry picks:"
      echo
      first=0
    fi
    local hash="$(__git_commit_hash "${1}")"
    local subject="$(__git_commit_subject $hash)"
    local lines=$(echo "$subject" | wrap_text 65)  # 5 leading spaces.
    echo "   + $hash:"
    echo "$lines" | sed 's/^/     /'
    shift
  done
}

# Get the baseline of master.
# Args: $1: release branch (or HEAD)
# TODO(philwo) this gives the wrong baseline when HEAD == release == master.
function get_release_baseline() {
  git merge-base master "$1"
}

# Get the list of cherry-picks since master
# Args:
#   $1: branch, default to HEAD
#   $2: baseline change, default to $(get_release_baseline $1)
function get_cherrypicks() {
  local branch="${1:-HEAD}"
  local baseline="${2:-$(get_release_baseline "${branch}")}"
  # List of changes since the baseline on the release branch
  local changes="$(git_log_hash "${baseline}" "${branch}" --reverse)"
  # List of changes since the baseline on the master branch, and their patch-id
  local master_changes="$(git_log_hash "${baseline}" master | xargs git show | git patch-id)"
  # Now for each changes on the release branch
  for i in ${changes}; do
    # Find the change with the same patch-id on the master branch if the note is not present
    hash=$(echo "${master_changes}" \
        | grep "^$(git show "$i" | git patch-id | cut -d " " -f 1)" \
        | cut -d " " -f 2)
    if [ -z "${hash}" ]; then
     # We don't know which cherry-pick it is coming from, fall back to the new commit hash.
     echo "$i"
    else
     echo "${hash}"
    fi
  done
}

# Generate the title of the release with the date from the release name ($1).
function get_release_title() {
  echo "Release ${1} ($(date +%Y-%m-%d))"
}

# Generate the release message to be added to the changelog
# from the release notes for release $1
# Args:
#   $1: release name
#   $2: release ref (default HEAD)
#   $3: delimiter around the revision information (default none)
function generate_release_message() {
  local release_name="$1"
  local branch="${2:-HEAD}"
  local delimiter="${3-}"

  get_release_title "$release_name"
  echo

  if [[ "$(is_rolling_release)" -eq 0 ]]; then
    if [ -n "${delimiter}" ]; then
      echo "${delimiter}"
    fi
    python3 ${RELNOTES_SCRIPT_DIR}/relnotes.py
    if [ -n "${delimiter}" ]; then
      echo "${delimiter}"
    fi
  else
    local baseline="$(get_release_baseline "${branch}")"
    local cherrypicks="$(get_cherrypicks "${branch}" "${baseline}")"

    if [ -n "${delimiter}" ]; then
      echo "${delimiter}"
    fi
    __create_revision_information $baseline $cherrypicks
    if [ -n "${delimiter}" ]; then
      echo "${delimiter}"
    fi

    echo

    # Generate the release notes
    local tmpfile=$(mktemp --tmpdir relnotes-XXXXXXXX)
    trap "rm -f ${tmpfile}" EXIT

    # Save the changelog so we compute the relnotes against HEAD.
    git show master:CHANGELOG.md > "${tmpfile}"

    local relnotes="$(create_release_notes "${tmpfile}" "${baseline}" ${cherrypicks})"
    echo "${relnotes}" > "${tmpfile}"

    __release_note_processor "${tmpfile}" || return 1
    relnotes="$(cat ${tmpfile})"

    cat "${tmpfile}"
  fi
}

# Returns the release notes for the CHANGELOG.md for all releases -
# release candidate, full release, and rolling release.
function get_full_release_notes() {
  local release_name="$(get_full_release_name "$@")"
  generate_release_message "${release_name}" "$@"
}