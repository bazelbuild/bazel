#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
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

set -eux

# Create a zip with:
#   * The binary.
#   * A sha256 sum of the binary.
#   * A README with info about the build.
function create_zip() {
  local git_hash
  git_hash=$(git rev-parse --short HEAD)
  local bazel_dir
  bazel_dir=bazel-$git_hash
  mkdir $bazel_dir
  cp bazel-bin/src/bazel $bazel_dir
  sha256sum $bazel_dir/bazel > $bazel_dir/sha256.txt
  cat > $bazel_dir/README.md <<EOF
Bazel binary built by Travis CI
-------------------------------

* [Build log](https://travis-ci.org/google/bazel/builds/$TRAVIS_BUILD_ID
* [Commit](https://github.com/$TRAVIS_REPO_SLUG/commit/$git_hash)
EOF
  bazel_zip=bazel-${git_hash}.zip
  echo "Creating $bazel_zip"
  zip -r -qq $bazel_zip $bazel_dir
  rm -r $bazel_dir
}

# Put the bazel zip in an uploaded dir.
function copy_to_upload_dir() {
  local date_dir=$(date +%F)
  local upload_dir=ci/$date_dir/
  mkdir -p $upload_dir
  mv $bazel_zip $upload_dir
  # Create a symlink to "latest" in the dir.
  (cd $upload_dir; ln -s $bazel_zip bazel.zip)
  echo "$bazel_zip moved to $upload_dir"
}

create_zip
copy_to_upload_dir
