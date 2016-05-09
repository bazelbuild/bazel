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

set -eu

readonly PORT=${1-12345}

readonly WORKING_DIR=$(mktemp -d)
trap "rm -rf $WORKING_DIR" EXIT

function check {
  which $1 > /dev/null || (echo "$1 not installed. Please install $1."; exit 1)
}

function main {
  check jekyll

  bazel build //site:jekyll-tree.tar
  tar -xf bazel-genfiles/site/jekyll-tree.tar -C $WORKING_DIR

  cd $WORKING_DIR
  jekyll serve --port $PORT
}
main
