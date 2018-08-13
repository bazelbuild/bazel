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
#
# An end-to-end test that Bazel's experimental UI produces reasonable output.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

set -euo pipefail

function set_up() {
  mkdir -p pkg
  cat > pkg/BUILD <<'EOF'
[genrule(
  name = "%sstamped" % ("un" if i == 0 else ""),
  outs = ["%sstamped.txt" % ("un" if i == 0 else "")],
  cmd = """
if [[ -f bazel-out/volatile-status.txt ]]; then
  grep BUILD_TIMESTAMP bazel-out/volatile-status.txt | cut -f 2 -d ' ' >$@
else
  echo "foo bar baz" > $@
fi
""",
  stamp = bool(i),
) for i in [0, 1]]
EOF
}

function test_source_date_epoch() {
  bazel clean --expunge &> $TEST_log
  bazel build --nostamp //pkg:* &> $TEST_log || fail "failed to build //pkg:*"
  assert_contains "^[0-9]\{9,\}\s*$" bazel-genfiles/pkg/stamped.txt
  assert_contains "^foo bar baz\s*$" bazel-genfiles/pkg/unstamped.txt

  bazel clean --expunge &> $TEST_log
  SOURCE_DATE_EPOCH=0 bazel build --stamp //pkg:* &> $TEST_log || fail "failed to build //pkg:*"
  assert_contains "^0\s*$" bazel-genfiles/pkg/stamped.txt
  assert_contains "^foo bar baz\s*$" bazel-genfiles/pkg/unstamped.txt

  bazel clean --expunge &> $TEST_log
  SOURCE_DATE_EPOCH=10 bazel build --stamp //pkg:* &> $TEST_log || fail "failed to build //pkg:*"
  assert_contains "^10\s*$" bazel-genfiles/pkg/stamped.txt
  assert_contains "^foo bar baz\s*$" bazel-genfiles/pkg/unstamped.txt
}

run_suite "${PRODUCT_NAME} stamping test"
