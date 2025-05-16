#!/usr/bin/env bash
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
# Test the execution of test logs
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_execute_testlog() {
  add_rules_shell "MODULE.bazel"
  mkdir dir
  cat <<EOF > dir/test.sh
#!/bin/sh
echo hello there
exit 0
EOF

  chmod +x dir/test.sh

  cat <<EOF > dir/BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(
  name = "test",
  srcs = [ "test.sh" ],
  size = "small",
)
EOF

  bazel build //dir:test
  logdir=$(bazel info bazel-testlogs)
  bazel test //dir:test

  # If we don't close the file explicitly, this typically generates
  # "text file busy."
  ${logdir}/dir/test/test.log
}

run_suite "execute testlogs"
