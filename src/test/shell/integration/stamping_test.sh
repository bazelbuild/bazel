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

set -e

function set_up() {
  mkdir -p pkg
  cat > pkg/BUILD <<EOF
genrule(
  name = "stamped",
  outs = ["stamped.txt"],
  cmd = "grep BUILD_TIMESTAMP bazel-out/volatile-status.txt | cut -d ' ' -f 2 >\$@",
  stamp = True,
)

genrule(
  name = "unstamped",
  outs = ["unstamped.txt"],
  cmd = "(grep BUILD_TIMESTAMP bazel-out/volatile-status.txt || echo 'x 0') | cut -d ' ' -f 2 >\$@",
  stamp = False,
)

genrule(
  name = "unspecified",
  outs = ["unspecified.txt"],
  cmd = "(grep BUILD_TIMESTAMP bazel-out/volatile-status.txt || echo 'x 0') | cut -d ' ' -f 2 >\$@",
)
EOF
}

function expect_equals() {
  [ "${1}" = "${2}" ]
}

function expect_not_equals() {
  [ "${1}" != "${2}" ]
}

function test_source_date_epoch() {
  # test --nostamp
  bazel clean --expunge &> $TEST_log
  bazel build --nostamp //pkg:* &> $TEST_log || fail "failed to build //pkg:*"
  expect_not_equals 0 $(cat bazel-genfiles/pkg/stamped.txt)
  expect_equals 0 $(cat bazel-genfiles/pkg/unstamped.txt)
  expect_equals 0 $(cat bazel-genfiles/pkg/unspecified.txt)

  # test --nostamp, explicit epoch=0
  bazel clean --expunge &> $TEST_log
  SOURCE_DATE_EPOCH=0 bazel build --stamp //pkg:* &> $TEST_log || fail "failed to build //pkg:*"
  expect_equals 0 $(cat bazel-genfiles/pkg/stamped.txt)
  expect_equals 0 $(cat bazel-genfiles/pkg/unstamped.txt)
  expect_equals 0 $(cat bazel-genfiles/pkg/unspecified.txt)

  # test --stamp, explicit epoch=0
  bazel clean --expunge &> $TEST_log
  SOURCE_DATE_EPOCH=10 bazel build --stamp //pkg:* &> $TEST_log || fail "failed to build //pkg:*"
  expect_equals 10 $(cat bazel-genfiles/pkg/stamped.txt)
  expect_equals 0 $(cat bazel-genfiles/pkg/unstamped.txt)
  expect_equals 0 $(cat bazel-genfiles/pkg/unspecified.txt)

  # test no stamp flag, explicit epoch=10
  bazel clean --expunge &> $TEST_log
  SOURCE_DATE_EPOCH=10 bazel build //pkg:* &> $TEST_log || fail "failed to build //pkg:*"
  expect_equals 10 $(cat bazel-genfiles/pkg/stamped.txt)
  expect_equals 0 $(cat bazel-genfiles/pkg/unstamped.txt)
  expect_equals 0 $(cat bazel-genfiles/pkg/unspecified.txt)
}

run_suite "Tests for genrule stamping"
