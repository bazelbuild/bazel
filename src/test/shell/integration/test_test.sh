#!/bin/bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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


function test_passing_test_is_reported_correctly() {
  mkdir -p tests
  cat >tests/BUILD <<'EOF'
sh_test(
    name = "success",
    size = "small",
    srcs = ["success.sh"],
)
EOF
  cat >tests/success.sh <<'EOF'
#!/bin/sh

echo "success.sh is successful"
exit 0
EOF
  chmod +x tests/success.sh

  bazel test --nocache_test_results //tests:success &>$TEST_log \
      || fail "expected success"
  expect_not_log "success.sh is successful"
  expect_log "^Executed 1 out of 1 test: 1 test passes."
}

function test_failing_test_is_reported_correctly() {
  mkdir -p tests
  cat >tests/BUILD <<'EOF'
sh_test(
    name = "fail",
    size = "small",
    srcs = ["fail.sh"],
)
EOF
  cat >tests/fail.sh <<'EOF'
#!/bin/sh

echo "fail.sh is failing"
exit 42
EOF
  chmod +x tests/fail.sh

  bazel test --nocache_test_results //tests:fail &>$TEST_log \
      && fail "expected failure" || true
  expect_log "^//tests:fail[[:space:]]\+FAILED in [[:digit:]]\+\.[[:digit:]]\+s"
  expect_log "^Executed 1 out of 1 test: 1 fails"
}

run_suite "test tests"
