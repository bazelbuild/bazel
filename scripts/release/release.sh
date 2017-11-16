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

# Generate the release branches and handle the release tags.

# Repositories to push the release branch and the release tag.
: ${RELEASE_REPOSITORIES:="git@github.com:bazelbuild/bazel"}

# Repositories to push the master branch
: ${MASTER_REPOSITORIES:="https://bazel.googlesource.com/bazel"}

# Name of the default editor
: ${EDITOR=vi}

# Author of the release commits
: ${RELEASE_AUTHOR="Bazel Release System <noreply@google.com>"}

# Load relnotes.sh
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source ${SCRIPT_DIR}/relnotes.sh

# Load common.sh
source ${SCRIPT_DIR}/common.sh

# Editing release notes info for the user
RELEASE_NOTE_MESSAGE='# Editing release notes
# Modify the release notes to make them suitable for the release.
# Every line starting with a # will be removed as well as every
# empty line at the start and at the end.
'

# Fetch everything from remote repositories to avoid conflicts
function fetch() {
  for i in ${RELEASE_REPOSITORIES}; do
    git fetch -f $i &>/dev/null || true
    git fetch -f $i refs/notes/*:refs/notes/* &>/dev/null || true
  done
}

# Set the release name $1 (and eventually the candidate number $2).
function set_release_name() {
  git notes --ref=release remove 2>/dev/null || true
  git notes --ref=release-candidate remove 2>/dev/null || true
  git notes --ref=release append -m "$1"
  if [[ ! -z "${2-}" ]]; then
    git notes --ref=release-candidate append -m "$2"
  fi
}

# Trim empty lines at the beginning and the end of the buffer.
function trim_empty_lines() {
  local f="$(echo $'\f')"  # linefeed because OSX sed does not support \f
  # Replace all new line by a linefeed, then using sed, remove the leading
  # and trailing linefeeds and convert them back to newline
  tr '\n' '\f' | sed -e "s/^$f*//" -e "s/$f*$//" | tr '\f' '\n'
}

# Launch the editor and return the edited release notes
function release_note_editor() {
  local tmpfile="$1"
  local branch_name="${2-}"
  $EDITOR ${tmpfile} || {
    echo "Editor failed, cancelling release creation..." >&2
    return 1
  }
  # Stripping the release notes
  local relnotes="$(cat ${tmpfile} | grep -v '^#' | trim_empty_lines)"
  if [ -z "${relnotes}" ]; then
    echo "Release notes are empty, cancelling release creation..." >&2
    return 1
  fi
  echo "${relnotes}" >${tmpfile}
}

# Create the release commit by changing the CHANGELOG file
function create_release_commit() {
  local infos=$(generate_release_message "${1}" HEAD '```')
  local changelog_path="$PWD/CHANGELOG.md"
  local master=$(get_master_ref)

  # Get the changelog from master to avoid missing release notes
  # from release that were in-between
  git checkout -q ${master} CHANGELOG.md || true

  # CHANGELOG.md
  local tmpfile="$(mktemp ${TMPDIR:-/tmp}/relnotes-XXXXXXXX)"
  trap "rm -f ${tmpfile}" EXIT
  echo -n "## ${infos}" >${tmpfile}
  if [ -f "${changelog_path}" ]; then
    {
      echo
      echo
      cat "${changelog_path}"
      echo
    } >> ${tmpfile}
  fi
  cat "${tmpfile}" > ${changelog_path}
  git add ${changelog_path}
  rm -f "${tmpfile}"
  trap - EXIT

  # Commit
  infos="$(echo "${infos}" | grep -Ev '^```$')"
  git commit --no-verify -m "${infos}" --no-edit --author "${RELEASE_AUTHOR}"
}

function apply_cherry_picks() {
  echo "Applying cherry-picks"
  # Apply cherry-picks
  for commit in "$@"; do
    local previous_head="$(git rev-parse HEAD)"
    echo "  Cherry-picking ${commit}"
    git cherry-pick ${commit} >/dev/null || {
      echo "Failed to cherry-pick ${commit}. please resolve the conflict and exit." >&2
      echo "  Use 'git cherry-pick --abort; exit' to abort the cherry-picks." >&2
      echo "  Use 'git cherry-pick --continue; exit' to resolve the conflict." >&2
      bash
      if [ "$(git rev-parse HEAD)" == "${previous_head}" ]; then
        echo "Cherry-pick aborted, aborting the whole command..." >&2
        return 1
      fi
    }
    # Add the origin of the cherry-pick in case the patch-id diverge and we cannot
    # find the original commit.
    git notes --ref=cherrypick add -f -m "${commit}"
  done
  return 0
}

# Find out the last release since the fork between "${1:-HEAD}" from master.
function find_last_release() {
  local branch="${1:-HEAD}"
  local baseline="${2:-$(get_release_baseline "${branch}")}"
  local changes="$(git log --pretty=format:%H "${baseline}~".."${branch}")"
  for change in ${changes}; do
    if git notes --ref=release show ${change} &>/dev/null; then
      echo ${change}
      return 0
    fi
  done
}

# Execute the create command:
#   Create a new release named "$1" with "$2" as the baseline commit.
function create_release() {
  local force_rc=
  if [[ "$1" =~ ^--force_rc=([0-9]*)$ ]]; then
    force_rc=${BASH_REMATCH[1]}
    shift 1
  fi
  local release_name="$1"
  local baseline="$2"
  shift 2
  local origin_branch=$(git_get_branch)
  local branch_name="release-${release_name}"

  fetch

  local last_release="$(git rev-parse --verify "${branch_name}" 2>/dev/null || true)"
  echo "Creating new release branch ${branch_name} for release ${release_name}"
  git checkout -B ${branch_name} ${baseline}

  apply_cherry_picks $@ || {
    git checkout ${origin_branch}
    git branch -D ${branch_name}
    exit 1
  }

  setup_git_notes "${force_rc}" "${release_name}" "${last_release}" || {
    git checkout ${origin_branch}
    git branch -D ${branch_name}
    exit 1
  }

  echo "Created $(get_full_release_name) on branch ${branch_name}."
}

# Setup the git notes for a release.
# It looks in the history since the merge base with master to find
# the latest release candidate and create the git notes for:
#   the release name (ref=release)
#   the release candidate number (ref=release-candidate)
#   the release notes (ref=release-notes)
# Args:
#   $1: set to a number to force the release candidate number (default
#       is taken by increasing the previous release candidate number).
#   $2: (optional) Specify the release name (default is taken by the
#       last release candidate or the branch name, e.g. if the branch
#       name is release-v1, then the name will be 'v1').
#   $3: (optional) Specify the commit for last release.
function setup_git_notes() {
  local force_rc="$1"
  local branch_name="$(git_get_branch)"

  # Figure out where we are in release: find the rc, the baseline, cherrypicks
  # and release name.
  local rc=${force_rc:-1}
  local baseline="$(get_release_baseline)"
  local cherrypicks="$(get_cherrypicks "HEAD" "${baseline}")"
  local last_release="${3-$(find_last_release "HEAD" "${baseline}")}"
  local release_name="${2-}"
  if [ -n "${last_release}" ]; then
    if [ -z "${force_rc}" ]; then
      rc=$(($(get_release_candidate "${last_release}")+1))
    fi
    if [ -z "${release_name}" ]; then
      release_name="$(get_release_name "${last_release}")"
    fi
  elif [ -z "${release_name}" ]; then
    if [[ "${branch_name}" =~ ^release-(.*)$ ]]; then
      release_name="${BASH_REMATCH[1]}"
    else
      echo "Cannot determine the release name." >&2
      echo "Please create a release branch with the name release-<name>." >&2
      return 1
    fi
  fi

  # Edit the release notes
  local tmpfile=$(mktemp ${TMPDIR:-/tmp}/relnotes-XXXXXXXX)
  trap "rm -f ${tmpfile}" EXIT

  echo "Creating release notes"

  # Save the changelog so we compute the relnotes against HEAD.
  git show master:CHANGELOG.md >${tmpfile} 2>/dev/null || echo >${tmpfile}
  # Compute the new release notes
  local relnotes="$(create_release_notes "${tmpfile}" "${baseline}" ${cherrypicks})"

  # Try to merge the release notes if there was a previous release
  if [ -n "${last_release}" ]; then
    # Compute the previous release notes
    local last_baseline="$(get_release_baseline "${last_release}")"
    git checkout -q "${last_release}"
    local last_relnotes="$(create_release_notes "${tmpfile}")"
    git checkout -q "${branch_name}"
    local last_savedrelnotes="$(get_release_notes "${last_release}")"
    relnotes="$(merge_release_notes "${branch_name}" "${relnotes}" \
      "${last_relnotes}" "${last_savedrelnotes}")"
  fi
  echo "${RELEASE_NOTE_MESSAGE}" > ${tmpfile}
  echo "# $(get_release_title "${release_name}rc${rc}")" >> ${tmpfile}
  echo >> ${tmpfile}
  echo "${relnotes}" >>"${tmpfile}"
  release_note_editor ${tmpfile} "${branch_name}" || return 1
  relnotes="$(cat ${tmpfile})"

  # Add the git notes
  set_release_name "${release_name}" "${rc}"
  git notes --ref=release-notes add -f -m "${relnotes}"

  # Clean-up
  rm -f ${tmpfile}
  trap - EXIT
}

# Force push a ref $2 to repo $1 if exists
function push_if_exists() {
  if git show-ref -q "${2}"; then
    git push -f "${1}" "+${2}"
  fi
}

# Push release notes refs but also a given ref
function push_notes_and_ref() {
  local ref="$1"
  for repo in ${RELEASE_REPOSITORIES}; do
    push_if_exists "${repo}" "${ref}"
    push_if_exists "${repo}" "refs/notes/release"
    push_if_exists "${repo}" "refs/notes/release-candidate"
    push_if_exists "${repo}" "refs/notes/release-notes"
    push_if_exists "${repo}" "refs/notes/cherrypick"
  done
}

# Push the release branch to the release repositories so a release
# candidate can be created.
function push_release_candidate() {
  push_notes_and_ref "$(get_release_branch)"
}

# Deletes the release branch after a release or abandoning the release
function cleanup_branches() {
  local tag_name=$1
  local i
  echo "Destroying the release branches for release ${tag_name}"
  # Destroy branch, ignoring if it doesn't exist.
  git branch -D release-${tag_name} &>/dev/null || true
  for i in $RELEASE_REPOSITORIES; do
    git push -f $i :release-${tag_name} &>/dev/null || true
  done
}

# Releases the current release branch, creating the necessary tag,
# destroying the release branch, updating the master's CHANGELOG.md
# and pushing everything to GitHub.
function do_release() {
  local branch=$(get_release_branch)
  local tag_name=$(get_release_name)

  echo -n "You are about to release branch ${branch} in tag ${tag_name}, confirm? [y/N] "
  read answer
  if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    echo "Creating the release commit"
    create_release_commit "${tag_name}"
    set_release_name "${tag_name}"
    echo "Creating the tag"
    git tag ${tag_name}

    echo "Cherry-picking CHANGELOG.md modification into master"
    git checkout master
    # Ensuring we are up to date for master
    git pull --rebase $(echo "$MASTER_REPOSITORIES" | cut -d " " -f 1) master
    # We do not cherry-pick because we might have conflict if the baseline
    # does not contains the latest CHANGELOG.md file, so trick it.
    local changelog_path="$PWD/CHANGELOG.md"
    git show ${branch}:CHANGELOG.md >${changelog_path}
    local tmpfile=$(mktemp ${TMPDIR:-/tmp}/relnotes-XXXXXXXX)
    trap 'rm -f ${tmpfile}' EXIT
    git_commit_msg ${branch} >${tmpfile}
    git add ${changelog_path}
    git commit --no-verify -F ${tmpfile} --no-edit --author "${RELEASE_AUTHOR}"
    rm -f ${tmpfile}
    trap - EXIT

    echo "Pushing the change to remote repositories"
    for i in $MASTER_REPOSITORIES; do
      git push $i +master
    done
    push_notes_and_ref "refs/tags/${tag_name}"
    cleanup_branches ${tag_name}
  fi
}

# Abandon the current release, deleting the branch on the local
# repository and on GitHub, discarding all changes
function abandon_release() {
  local branch_info=$(get_release_branch)
  local tag_name=$(get_release_name)
  echo -n "You are about to abandon release ${tag_name}, confirm? [y/N] "
  read answer
  if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    git notes --ref=release remove 2>/dev/null || true
    git notes --ref=release-candidate remove 2>/dev/null || true
    git checkout -q master >/dev/null
    cleanup_branches ${tag_name}
  fi
}

function usage() {
  cat >&2 <<EOF
Usage: $1 command [arguments]
Available commands are:
  - create [--force_rc=RC] RELEASE_NAME BASELINE [COMMIT1 ... COMMITN]:
      creates a new release branch for release named RELEASE_NAME,
      cutting it at the commit BASELINE and cherry-picking
      COMMIT1 ... COMMITN. The release candidate number will be
      computed from existing release branch unless --force_rc is
      specified.
  - generate-rc [--force_rc=RC]: generate a release candidate out of
      the current branch, the branch should be named "release-XXX"
      where "XXX" is the name of the release. This branch should be
      a fork of master in which some cherry-picks where taken from
      master. --force_rc can be used to override the RC number
      (by default it tries to look for latest release and increment
      the rc number).
  - push: push the current release branch to release repositories.
  - release: do the actual release of the current release branch.
  - abandon: abandon the current release branch.

The typical workflow for the release manager is:
1. Create the release branch using the decided name for the release
   (usually a version number). The BASELINE is generally a baseline
   that has been tested extensively including inside Google.
2. Push to the repository and wait for the continuous integration
   to rebuild and deploy the various artifacts and send the annoucement
   mails about a new release candidate being available.
3. If necessary, creates a new release branch with the same name to
   address return from the users and go back to 2.
4. Either abandon or release the branch depending on if too much
   problems are reported or if the release is considered viable. The
   CI system should then rebuild from the tag and deploy the artifact
   to GitHub and sends the announcement mails about the new release.
EOF
   exit 1
}

git diff-index --quiet HEAD -- || {
  echo "There are pending changes in this git repository." >&2
  echo "Please commit or stash them before using that script." >&2
  exit 1
}

[ "$(git rev-parse --show-toplevel)" == "$PWD" ] || {
  echo "You should run this script from the root of the git repository." >&2
  exit 1
}

progname=$0
cmd=${1-}
shift || usage $progname

case $cmd in
  create)
    (( $# >= 2 )) || usage $progname
    create_release "$@"
    ;;
  push)
    push_release_candidate
    ;;
  release)
    do_release
    ;;
  generate-rc)
    force_rc=
    if [[ "${1-}" =~ ^--force_rc=([0-9]*)$ ]]; then
      force_rc=${BASH_REMATCH[1]}
      shift 1
    fi
    setup_git_notes "${force_rc}"
    ;;
  abandon)
    abandon_release
    ;;
  *)
    usage $progname
    ;;
esac
