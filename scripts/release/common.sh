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
  git notes --ref=release-candidate show "$@" 2>/dev/null | xargs echo || true
}

# Extract the release name from the git notes
function get_release_name() {
  git notes --ref=release show "$@" 2>/dev/null | xargs echo || true
}

# Returns the info from the branch of the release. It is the current branch
# but it errors out if the current branch is not a release branch. This
# method returns the tag of the release and the number of the current
# candidate in this release.
function get_release_branch() {
  local branch_name=$(git_get_branch)
  if [ -z "$(get_release_name)" -o -z "$(get_release_candidate)" ]; then
    echo "Not a release branch: ${branch_name}." >&2
    exit 1
  fi
  echo "${branch_name}"
}
