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

set -eo pipefail

function print_usage {
  echo "Usage: bazel run //tools:upload_bazel_rules -- <rules_xyz> <version>"
}

REPO="$1"
if [ -z "$REPO" ]; then
  print_usage
  exit 1
fi

# TODO(yannic): Add option to get latest commit or release from GitHub API.
VERSION="$2"
if [ -z "$VERSION" ]; then
  print_usage
  exit 1
fi

# From now on, fail if there are any unbound variables.
set -u

# Create a temp directory to hold the versioned tarball,
# and clean it up when the script exits.
tmpdir=$(mktemp -d)
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
