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

# Set the release name $1 (and eventually the candidate number $2).
function set_release_name() {
  git notes --ref=release remove 2>/dev/null || true
  git notes --ref=release-candidate remove 2>/dev/null || true
  git notes --ref=release append -m "$1"
  local relname="$1"
  if [[ ! -z "${2-}" ]]; then
    git notes --ref=release-candidate append -m "$2"
    relname="${relname}RC${2}"
  fi
  echo "$relname"
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
  local origin_branch="$2"
  local branch_name="${3-}"
  $EDITOR ${tmpfile} || {
    echo "Editor failed, cancelling release creation..." >&2
    git checkout -q ${origin_branch} >/dev/null
    [ -z "${branch_name}" ] || git branch -D ${branch_name}
    exit 1
  }
  # Stripping the release notes
  local relnotes="$(cat ${tmpfile} | grep -v '^#' | trim_empty_lines)"
  if [ -z "${relnotes}" ]; then
    echo "Release notes are empty, cancelling release creation..." >&2
    git checkout -q ${origin_branch} >/dev/null
    [ -z "${branch_name}" ] || git branch -D ${branch_name}
    exit 1
  fi
  echo "${relnotes}" >${tmpfile}
}

# Create the release commit by changing the CHANGELOG file
function create_release_commit() {
  local release_title="$1"
  local release_name="$2"
  local relnotes="$3"
  local tmpfile="$4"
  local baseline="$5"
  shift 5
  local cherrypicks=$@
  local changelog_path="$PWD/CHANGELOG.md"

  version_info=$(create_revision_information $baseline $cherrypicks)
  # CHANGELOG.md
  cat >${tmpfile} <<EOF
## ${release_title}

EOF
  if [ -n "${version_info}" ]; then
    cat >>${tmpfile} <<EOF
\`\`\`
${version_info}
\`\`\`
EOF
  fi
  cat >>${tmpfile} <<EOF

${relnotes}
EOF

  if [ -f "${changelog_path}" ]; then
    echo >>${tmpfile}
    cat "${changelog_path}" >>${tmpfile}
  fi
  cat ${tmpfile} > ${changelog_path}
  git add ${changelog_path}
  # Commit message
  cat >${tmpfile} <<EOF
${release_title}

${version_info}

${relnotes}
EOF
  git commit --no-verify -F ${tmpfile} --no-edit --author "${RELEASE_AUTHOR}"
}

function apply_cherry_picks() {
  echo "Applying cherry-picks"
  # Apply cherry-picks
  for i in $@; do
    local previous_head="$(git rev-parse HEAD)"
    echo "  Cherry-picking $i"
    git cherry-pick $i >/dev/null || {
      echo "Failed to cherry-pick $i. please resolve the conflict and exit." >&2
      echo "  Use 'git cherry-pick --abort; exit' to abort the cherry-picks." >&2
      echo "  Use 'git cherry-pick --continue; exit' to resolve the conflict." >&2
      bash
      if [ "$(git rev-parse HEAD)" == "${previous_head}" ]; then
        echo "Cherry-pick aborted, aborting the whole command..." >&2
        return 1
      fi
    }
  done
  return 0
}

# Execute the create command:
#   Create a new release named "$1" with "$2" as the baseline commit.
function create_release() {
  local release_name="$1"
  local baseline="$2"
  shift 2
  local origin_branch=$(git_get_branch)
  local branch_name="release-${release_name}"
  local release_title="Release ${release_name} ($(date +%Y-%m-%d))"
  local tmpfile=$(mktemp ${TMPDIR:-/tmp}/relnotes-XXXXXXXX)
  local tmpfile2=$(mktemp ${TMPDIR:-/tmp}/relnotes-XXXXXXXX)
  trap 'rm -f ${tmpfile} ${tmpfile2}' EXIT

  # Get the rc number (1 by default)
  local rc=1
  if [ -n "$(git branch --list --column ${branch_name})" ]; then
    rc=$(($(get_release_candidate "${branch_name}")+1))
  fi

  # Save the changelog so we compute the relnotes against HEAD.
  git show master:CHANGELOG.md >${tmpfile2} 2>/dev/null || echo >${tmpfile2}

  echo "Creating new release branch ${branch_name} for release ${release_name}"
  git checkout -B ${branch_name} ${baseline}

  apply_cherry_picks $@ || {
    git checkout ${origin_branch}
    git branch -D ${branch_name}
    exit 1
  }

  echo "Creating release notes"
  echo "${RELEASE_NOTE_MESSAGE}" > ${tmpfile}
  echo "# ${release_title}" >> ${tmpfile}
  echo >> ${tmpfile}
  create_release_notes "${tmpfile2}" >> ${tmpfile}
  release_note_editor ${tmpfile} "${origin_branch}" "${branch_name}"
  local relnotes="$(cat ${tmpfile})"

  create_release_commit "${release_title}" "${release_name}" \
      "${relnotes}" "${tmpfile}" "${baseline}" $@
  release_name=$(set_release_name "${release_name}" "${rc}")
  git checkout ${origin_branch} &> /dev/null
  echo "Created ${release_name} on branch ${branch_name}."

  rm -f ${tmpfile} ${tmpfile2}
  trap - EXIT
}

# Push the release branch to the release repositories so a release
# candidate can be created.
function push_release_candidate() {
  local branch="$(get_release_branch)"
  for repo in ${RELEASE_REPOSITORIES}; do
    git push -f ${repo} +${branch}
    git push -f ${repo} +refs/notes/release
    git push -f ${repo} +refs/notes/release-candidate
  done
}

# Deletes the release branch after a release or abandoning the release
function cleanup_branches() {
  local tag_name=$1
  local i
  echo "Destroying the release branches for release ${tag_name}"
  # Destroy branch, ignoring if it doesn't exists.
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
  if [ "$answer" = "y" -o "$answer" = "Y" ]; then
    # Remove release "candidate"
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
    for i in $RELEASE_REPOSITORIES; do
      git push $i +refs/tags/${tag_name}
      git push $i +refs/notes/release-candidate
      git push $i +refs/notes/release
    done
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
  if [ "$answer" = "y" -o "$answer" = "Y" ]; then
    git checkout -q master >/dev/null
    cleanup_branches ${tag_name}
  fi
}

function usage() {
  cat >&2 <<EOF
Usage: $1 command [arguments]
Available commands are:
  - create RELEASE_NAME BASELINE [COMMIT1 ... COMMITN]: creates a new
      release branch for release named RELEASE_NAME, cutting it at
      the commit BASELINE and cherry-picking COMMIT1 ... COMMITN.
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
  abandon)
    abandon_release
    ;;
  *)
    usage $progname
    ;;
esac
