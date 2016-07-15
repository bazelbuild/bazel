#!/bin/bash -x
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
#
# Tests the examples provided in Bazel with MSVC toolchain
#

if ! type rlocation &> /dev/null; then
  # We do not care about this test on old Bazel releases.
  exit 0
fi

# Load test environment
source $(rlocation io_bazel/src/test/shell/bazel/test-setup.sh) \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

if ! is_windows; then
  echo "This test suite requires running on Windows. But now is ${PLATFORM}" >&2
  exit 0
fi

function set_up() {
  copy_examples
  export PATH=$PATH:/c/python_27_amd64/files
}

common_args="-s --verbose_failures --cpu=x64_windows_msvc"

function assert_build_windows() {
  bazel build ${common_args} $* || fail "Failed to build $*"
}

function assert_test_ok_windows() {
  bazel test ${common_args} --test_output=errors $* \
    || fail "Test $1 failed while expecting success"
}

function assert_test_fails_windows() {
  bazel test ${common_args} --test_output=errors $* >& $TEST_log \
    && fail "Test $* succeed while expecting failure" \
    || true
  expect_log "$1.*FAILED"
}

#
# Native rules
#
function test_cpp() {
  local cpp_pkg=examples/cpp
  assert_build_windows "//examples/cpp:hello-world"
  test -f "./bazel-bin/${cpp_pkg}/libhello-lib.a" || fail "libhello-lib.a should be generated"
  assert_binary_run "./bazel-bin/${cpp_pkg}/hello-world foo" "Hello foo"
  assert_test_ok_windows "//examples/cpp:hello-success_test"
  assert_test_fails_windows "//examples/cpp:hello-fail_test"
}

run_suite "cpp examples on Windows"

