#!/usr/bin/env bash
#
# Copyright 2018 The Bazel Authors. All rights reserved.
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
# bazel_strategy_test.sh: integration tests for Bazel strategy flags.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  create_new_workspace
  mkdir -p foo

  cat > foo/BUILD <<EOF
genrule(
    name = "genrule",
    srcs = [],
    outs = ["my_file"],
    cmd = "echo bar > \$@",
)
EOF
}

function test_regexp_strategy() {
  assert_build --strategy_regexp=foo=standalone //foo:my_file
  assert_contains "bar" bazel-genfiles/foo/my_file
}

function test_regexp_strategy_regexp() {
  assert_build --strategy_regexp=my.*file=standalone //foo:my_file
  assert_contains "bar" bazel-genfiles/foo/my_file
}

run_suite "Bazel strategy integration tests"