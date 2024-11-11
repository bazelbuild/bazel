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
#
# Integration tests for Bazel client.
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_product_name_with_bazel_info() {
  cat > WORKSPACE <<EOF
workspace(name = 'blerp')
EOF
  bazel info >& "$TEST_log" || fail "Expected zero exit"

  expect_log "^bazel-bin:.*_bazel.*bazel-out.*bin\$"
  expect_log "^bazel-genfiles:.*_bazel.*bazel-out.*genfiles\$"
  expect_log "^bazel-testlogs:.*_bazel.*bazel-out.*testlogs\$"
  expect_log "^output_path:.*/execroot/blerp/bazel-out\$"
  expect_log "^execution_root:.*/execroot/blerp\$"
  expect_log "^server_log:.*/java\.log.*\$"
}

# This test is for Bazel only and not for Google's internal version (Blaze),
# because Bazel uses a different way to compute the workspace name.
function test_server_process_name_has_workspace_name() {
  mkdir foobarspace
  cd foobarspace
  ps -o cmd= "$(bazel info server_pid)" &>"$TEST_log"
  expect_log "^bazel(foobarspace)"
  bazel shutdown
}
