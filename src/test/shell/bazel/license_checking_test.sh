#!/bin/bash
#
# Copyright 2019 The Bazel Authors. All rights reserved.
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
# Tests Bazel's license checking semantics.
#
# This is a temporary test supporting the migration away from any license
# checking at all. When https://github.com/bazelbuild/bazel/issues/7444 is
# fixed this test can be deleted.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function write_rules {
  mkdir -p third_party/restrictive
  cat <<EOF > third_party/restrictive/BUILD
licenses(['restricted'])
genrule(
    name = 'lib',
    srcs = [],
    outs = ['lib.out'],
    cmd = 'echo hi > $@',
    licenses = ['restricted'],
    visibility = ['//bad:__pkg__']
)
exports_files(['file'])
EOF

  mkdir -p bad
  cat <<EOF > bad/BUILD
distribs(['client'])
genrule(
    name = 'main',
    srcs = [],
    outs = ['bad.out'],
    tools = ['//third_party/restrictive:lib'],
    cmd = 'echo hi > $@'
)
EOF

  mkdir -p third_party/missing_license
  cat <<EOF > third_party/missing_license/BUILD
genrule(
    name = 'lib',
    srcs = [],
    outs = ['lib.out'],
    cmd = 'echo hi > $@'
)
exports_files(['file'])
EOF
}

function test_default_mode {
  create_new_workspace
  write_rules
  bazel build --nobuild //bad:main >& $TEST_log || fail "build should succeed"
}

function test_license_checking {
  create_new_workspace
  write_rules
  bazel build --nobuild //bad:main --check_licenses >& $TEST_log && fail "build shouldn't succeed"
  expect_log "Build target '//bad:main' is not compatible with license '\[restricted\]' from target '//third_party/restrictive:lib'"
}

function test_disable_license_checking_override {
  create_new_workspace
  write_rules
  bazel build --nobuild //bad:main --check_licenses --incompatible_disable_third_party_license_checking \
    >& $TEST_log || fail "build should succeed"
}

function test_third_party_no_license_is_checked {
  create_new_workspace
  write_rules
  bazel build --nobuild //third_party/missing_license:lib >& $TEST_log && fail "build shouldn't succeed"
  expect_log "third-party rule '//third_party/missing_license:lib' lacks a license declaration"
}

function test_third_party_no_license_no_check {
  create_new_workspace
  write_rules
  bazel build --nobuild //third_party/missing_license:lib --nocheck_third_party_targets_have_licenses \
    || fail "build should succeed"
}

function test_third_party_no_license_disable_license_checking_override {
  create_new_workspace
  write_rules
  bazel build --nobuild //third_party/missing_license:lib --incompatible_disable_third_party_license_checking \
    || fail "build should succeed"
}

run_suite "license checking tests"

