#!/bin/bash

# Copyright 2019 The Bazel Authors. All rights reserved.
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

# This script fetches
# `https://github.com/bazelbuild/<rules_xyz>/archive/<version>.tar.gz`
# and uploads it to the cannonical location on `https://mirror.bazel.build`.

set -euo pipefail

function validate_input {
  local n="$1"
  local input="$2"
  # Valid inputs contain only alphanumeric letters or [_-.]
  # and must be between 3 (e.g. numbered releases like 0.1 or 1.0) and 40
  # (e.g. git commit hashes) characters long.
  if [[ ! "${input}" =~ ^[a-zA-Z0-9_\\-\\.]{3,40}$ ]]; then
    echo "Argument ${n} with value '${input}' contains invalid characters," \
         "or is not between 3 and 40 characters long"
    exit 1
  fi
}

if [ "$#" -ne 2 ]; then
  echo "Usage: bazel run //tools:upload_bazel_mirror -- <rules_xyz> <version>"
  exit 1
fi

REPO="$1"
validate_input 1 "${REPO}"

# TODO(yannic): Add option to get latest commit or release from GitHub API.
VERSION="$2"
validate_input 2 "${VERSION}"

# Create a temp directory to hold the versioned tarball,
# and clean it up when the script exits.
tmpdir="$(mktemp -d)"
function cleanup {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

url="https://github.com/bazelbuild/${REPO}/archive/${VERSION}.tar.gz"
versioned_filename="${REPO}-${VERSION}.tar.gz"
versioned_archive="${tmpdir}/${versioned_filename}"

# Download tarball into temporary folder.
# -L to follow redirects.
curl -L --fail --output "${versioned_archive}" "${url}"

# Upload the tarball to GCS.
# -n for no-clobber, so we don't overwrite existing files
gsutil cp -n "${versioned_archive}" \
  "gs://bazel-mirror/github.com/bazelbuild/${REPO}/archive/${VERSION}.tar.gz"
