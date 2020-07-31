#!/bin/bash
#
# Copyright 2020 The Bazel Authors. All rights reserved.
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
# Tests the persistent test runner
#

# Load test environment
# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

cat >>$TEST_TMPDIR/bazelrc <<'EOF'
build --strategy=TestRunner=worker,local
build --experimental_persistent_test_runner
EOF

function test_simple_sh_test() {
  mkdir -p examples/tests/
  cat << 'EOF' > examples/tests/BUILD
sh_test(
  name = "shell",
  srcs = [ "shell.sh" ],
)
EOF
  cat << 'EOF' > examples/tests/shell.sh
EOF
  chmod +x examples/tests/shell.sh
  bazel test examples/tests:shell &> $TEST_log || fail "Shell test failed"
}

run_suite "persistent_test_runner"
