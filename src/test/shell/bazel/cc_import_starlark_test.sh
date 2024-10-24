#!/bin/bash -eu
#
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

# This integration test exists so that we can run our Starlark tests
# for cc_import with Bazel built from head. Once the Stararlark
# implementation can rely on release Bazel, we can add the tests directly.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_all_starlark_written_tests() {
  local workspace_name="${FUNCNAME[0]}"
  mkdir -p "${workspace_name}"

  workspace_dir=$(pwd)/"$workspace_name"

  TEST_FILES_DIR="$RUNFILES_DIR/io_bazel"
  cd $TEST_FILES_DIR
  # Using --parents breaks on Mac.
  mkdir "$workspace_dir/tools"
  mkdir "$workspace_dir/tools/build_defs"
  mkdir "$workspace_dir/tools/build_defs/cc"
  mkdir "$workspace_dir/tools/build_defs/cc/tests"
  cp "tools/build_defs/cc/BUILD" "$workspace_dir/tools/build_defs/cc/BUILD"
  cp "tools/build_defs/cc/cc_import.bzl" "$workspace_dir/tools/build_defs/cc/cc_import.bzl"
  cp -r "tools/build_defs/cc/tests" "$workspace_dir/tools/build_defs/cc/"

  cd "$workspace_dir"

  # TODO(gnish): Re-enable tests once bazel picks up changes.
  # bazel test --experimental_starlark_cc_import tools/build_defs/cc/tests:cc_import_tests
}

# run_suite "cc_import_starlark_test"
