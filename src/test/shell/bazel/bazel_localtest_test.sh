#!/usr/bin/env bash
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

set -eu

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_run_local() {
  add_rules_shell "MODULE.bazel"
  mkdir -p dir
  cat > emptyfile

  # In standalone mode,
  # we have access to /var which is not mounted in sandboxed mode
  cat <<EOF > dir/test_local.sh
#!/bin/sh
test -e "$(pwd)/emptyfile" && exit 0 || true
echo "no $(pwd)/emptyfile in standalone mode"
exit 1
EOF

  chmod +x dir/test_local.sh

  cat <<EOF > dir/BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(
  name = "localtest",
  srcs = [ "test_local.sh" ],
  size = "small",
  local = 1
)
EOF

  bazel test //dir:all &> $TEST_log || fail "expected success"
}

run_suite "test tests"
