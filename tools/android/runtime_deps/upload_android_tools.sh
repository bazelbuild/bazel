#!/bin/bash

# Copyright 2014 The Bazel Authors. All rights reserved.
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

# This script builds a new version of android_tools.tar.gz and uploads
# it to Google Cloud Storage. This script is non-destructive and idempotent.
#
# To run this script, call `bazel run //tools/android/runtime_deps:upload_android_tools
#
# android_tools.tar.gz contains runtime jars required by the Android rules. We unbundled
# these jars out from Bazel to keep its binary size small.
#
# If you make changes any tool bundled in android_tools.tar.gz and want them to be used for
# the next Bazel release, increment the version number and run this script. Then, update the
# following files with new version, URL, and sha256 of the new android_tools tarball:
#
# - WORKSPACE
# - src/main/java/com/google/devtools/build/lib/bazel/rules/android/android_remote_tools.WORKSPACE
#
# More context: https://github.com/bazelbuild/bazel/issues/1055

set -xeuo pipefail

# The version of android_tools.tar.gz
VERSION="0.1"
VERSIONED_FILENAME="android_tools_pkg-$VERSION.tar.gz"

# Create a temp directory to hold the versioned tarball, and clean it up when the script exits.
tmpdir=$(mktemp -d)
function cleanup {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

# Add the version to the tarball.
android_tools_archive="tools/android/runtime_deps/android_tools.tar.gz"
versioned_android_tools_archive="$tmpdir/$VERSIONED_FILENAME"
cp $android_tools_archive $versioned_android_tools_archive

# Upload the tarball to GCS.
# -n for no-clobber, so we don't overwrite existing files
# -a public-read to make this tarball public
gsutil cp -n -a public-read $versioned_android_tools_archive \
  gs://bazel-mirror/bazel_android_tools/$VERSIONED_FILENAME
