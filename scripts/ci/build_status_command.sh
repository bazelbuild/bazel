#!/usr/bin/env bash

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

# A build status command to provide the package info generator with
# the information about the commit being built

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname ${SCRIPT_DIR})/release/common.sh"

git_hash=$(git rev-parse --short HEAD)
echo "RELEASE_GIT_HASH ${git_hash}"
url="${GIT_REPOSITORY_URL:-https://github.com/bazelbuild/bazel}"
echo "RELEASE_COMMIT_URL ${url}/commit/${git_hash}"
if [ -n "${BUILT_BY-}" ]; then
  echo "RELEASE_BUILT_BY ${BUILT_BY}"
fi
if [ -n "${BUILD_LOG-}" ]; then
  echo "RELEASE_BUILD_LOG ${BUILD_LOG}"
fi
release_name=$(get_full_release_name)
if [ -n "${release_name}" ]; then
  echo "RELEASE_NAME ${release_name}"
  echo "RELEASE_NOTES $(get_full_release_notes | tr '\n' '\f')"
fi
