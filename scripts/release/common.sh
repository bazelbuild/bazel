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

# Some common method for release scripts

# A release candidate is created from a branch named "release-%name%"
# where %name% is the name of the release. Once promoted to a release,
# A tag %name% will be created from this branch and the corresponding
# branch removed.
# The last commit of the release branch is always a commit containing
# the release notes in the commit message and updating the CHANGELOG.md.
# This last commit will be cherry-picked back in the master branch
# when the release candidate is promoted to a release.
# To follow tracks and to support how CI systems fetch the refs, we
# store two commit notes: the release name and the candidate number.

# Get the short hash of a commit
function __git_commit_hash() {
  git rev-parse "${1}"
}

# Get the subject (first line of the commit message) of a commit
function __git_commit_subject() {
  git show -s --pretty=format:%s "$@"
}

# Returns the branch name of the current git repository
function git_get_branch() {
  git symbolic-ref --short HEAD
}

# Show the commit message of the ref specified in argument
function git_commit_msg() {
  git show -s --pretty=format:%B "$@"
}

# Extract the release candidate number from the git notes
function get_release_candidate() {
  git notes --ref=release-candidate show "$@" 2>/dev/null || true
}

# Extract the release name from the git notes
function get_release_name() {
  git notes --ref=release show "$@" 2>/dev/null || true
}

# Get the list of commit hashes between two revisions
function git_log_hash() {
  local baseline="$1"
  local head="$2"
  shift 2
  git log --pretty=format:%H "${baseline}".."${head}" "$@"
}

# Extract the full release name from the git notes
function get_full_release_name() {
  local name="$(get_release_name "$@")"
  local rc="$(get_release_candidate "$@")"
  if [ -n "${rc}" ]; then
    echo "${name}rc${rc}"
  else
    echo "${name}"
  fi
}

# Extract the release notes from the git notes
function get_release_notes() {
  git notes --ref=release-notes show "$@" 2>/dev/null || true
}

# Returns the info from the branch of the release. It is the current branch
# but it errors out if the current branch is not a release branch. This
# method returns the tag of the release and the number of the current
# candidate in this release.
function get_release_branch() {
  local branch_name=$(git_get_branch)
  if [ -z "$(get_release_name)" ] || [ -z "$(get_release_candidate)" ]; then
    echo "Not a release branch: ${branch_name}." >&2
    exit 1
  fi
  echo "${branch_name}"
}

# fmt behaves differently on *BSD and on GNU/Linux, use fold.
function wrap_text() {
  fold -s -w $1 | sed 's/ *$//'
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
    local hash=$(git notes --ref=cherrypick show "$i" 2>/dev/null || true)
    if [ -z "${hash}" ]; then
      # Find the change with the same patch-id on the master branch if the note is not present
      hash=$(echo "${master_changes}" \
          | grep "^$(git show "$i" | git patch-id | cut -d " " -f 1)" \
          | cut -d " " -f 2)
    fi
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
  local baseline="$(get_release_baseline "${branch}")"
  local cherrypicks="$(get_cherrypicks "${branch}" "${baseline}")"

  get_release_title "$release_name"
  echo

  if [ -n "${delimiter}" ]; then
    echo "${delimiter}"
  fi
  __create_revision_information $baseline $cherrypicks
  if [ -n "${delimiter}" ]; then
    echo "${delimiter}"
  fi

  echo
  get_release_notes "${branch}"
}

# Returns the release notes for the CHANGELOG.md taken from either from
# the notes for a release candidate or from the commit message for a
# full release.
function get_full_release_notes() {
  local release_name="$(get_full_release_name "$@")"

  if [[ "${release_name}" =~ rc[0-9]+$ ]]; then
    # Release candidate, we need to generate from the notes
    generate_release_message "${release_name}" "$@"
  else
    # Full release, returns the commit message
    git_commit_msg "$@"
  fi
}
