#!/bin/bash
#
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
#
# Setting up the environment for our legacy integration tests.
#
source $(cd "$(dirname $(dirname "${BASH_SOURCE[0]}"))" && pwd)/bazel/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

PRODUCT_NAME=bazel
WORKSPACE_NAME=main
bazelrc=$TEST_TMPDIR/bazelrc

function put_bazel_on_path() {
  put_blaze_on_path "$@"
}

function write_default_bazelrc() {
  write_default_blazerc "$@"
}

function put_bazel_on_path() {
  # do nothing as test-setup already does that
  true
}

function write_default_bazelrc() {
  setup_bazelrc
}

function create_and_cd_client() {
  setup_clean_workspace
  echo "workspace(name = '$WORKSPACE_NAME')" >WORKSPACE
  touch .bazelrc
}
