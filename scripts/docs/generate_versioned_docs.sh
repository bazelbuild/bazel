#!/bin/bash
# Copyright 2016 The Bazel Authors. All rights reserved.
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
# Usage: 
# $BAZEL_ROOT/scripts/docs/generate_versioned_docs.sh 0.8.0 0.9.0 ..

set -eu

ORIGINAL_GIT_REF=$(git symbolic-ref --quiet --short HEAD || git rev-parse --short HEAD)

readonly RELEASES=$@; shift

main() {
  for release in $RELEASES; do
    git checkout $release
    bazel build //site:jekyll-tree.tar --action_env=BAZEL_RELEASE=$release
    gsutil cp -n -a public-read bazel-genfiles/site/jekyll-tree.tar gs://bazel-mirror/bazel_versioned_docs/jekyll-tree-$release.tar
  done
}

cleanup() {
  git checkout $ORIGINAL_GIT_REF
}

trap cleanup EXIT

if [ -e WORKSPACE ]
then
  main
else
  echo "Please run this from the root of the Bazel workspace."
fi
