#!/bin/bash
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

# Load the test setup defined in the parent directory
source $(rlocation io_bazel/src/test/shell/integration_test_setup.sh) \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

if ! is_windows; then
  echo "This test suite requires running on Windows. But now is ${PLATFORM}" >&2
  exit 0
fi

function set_up() {
  copy_examples
  export PATH=/c/python_27_amd64/files:$PATH
  EXTRA_BAZELRC="build --cpu=x64_windows_msvc"
  setup_bazelrc
}

# An assertion that execute a binary from a sub directory (to test runfiles)
function assert_binary_run_from_subdir() {
    ( # Needed to make execution from a different path work.
    export PATH=${bazel_javabase}/bin:"$PATH" &&
    mkdir -p x &&
    cd x &&
    unset JAVA_RUNFILES &&
    unset TEST_SRCDIR &&
    assert_binary_run "../$1" "$2" )
}

#
# Native rules
#
function test_cpp() {
  local cpp_pkg=examples/cpp
  assert_build_output ./bazel-bin/${cpp_pkg}/libhello-lib.a ${cpp_pkg}:hello-world
  assert_bazel_run "//examples/cpp:hello-world foo" "Hello foo"
  assert_test_ok "//examples/cpp:hello-success_test"
  assert_test_fails "//examples/cpp:hello-fail_test"
}

function test_cpp_alwayslink() {
  mkdir -p cpp/main
  cat >cpp/main/BUILD <<EOF
cc_library(
    name = "lib",
    srcs = ["lib.cc"],
    alwayslink = 1,
)
cc_library(
    name = "main",
    srcs = ["main.cc"],
)
cc_binary(
    name = "bin",
    deps = [":main", ":lib"],
)
EOF

  cat >cpp/main/lib.cc <<EOF
extern int global_variable;
int init() {
    ++global_variable;
    return global_variable;
}
int x = init();
int y = init();
EOF

  cat >cpp/main/main.cc <<EOF
#include<stdio.h>
int global_variable = 0;
int main(void) {
    printf("global : %d\n", global_variable);
    return 0;
}
EOF
  assert_build //cpp/main:bin
  ./bazel-bin/cpp/main/bin >& $TEST_log \
    || fail "//cpp/main:bin execution failed"
  expect_log "global : 2"
}

function test_java() {
  local java_pkg=examples/java-native/src/main/java/com/example/myproject

  assert_build_output ./bazel-bin/${java_pkg}/libhello-lib.jar ${java_pkg}:hello-lib
  assert_build_output ./bazel-bin/${java_pkg}/libcustom-greeting.jar ${java_pkg}:custom-greeting
  assert_build_output ./bazel-bin/${java_pkg}/hello-world ${java_pkg}:hello-world
  assert_build_output ./bazel-bin/${java_pkg}/hello-resources ${java_pkg}:hello-resources
  assert_binary_run_from_subdir "bazel-bin/${java_pkg}/hello-world foo" "Hello foo"
}

function test_java_test() {
  setup_javatest_support
  local java_native_tests=//examples/java-native/src/test/java/com/example/myproject
  local java_native_main=//examples/java-native/src/main/java/com/example/myproject

  assert_build "-- //examples/java-native/... -${java_native_main}:hello-error-prone"
  assert_build_fails "${java_native_main}:hello-error-prone" \
      "Did you mean 'result = b == -1;'?"
  assert_test_ok "${java_native_tests}:hello"
  assert_test_ok "${java_native_tests}:custom"
  assert_test_fails "${java_native_tests}:fail"
  assert_test_fails "${java_native_tests}:resource-fail"
}

function test_native_python() {
  # On windows, we build a python executable zip as the python binary
  assert_build //examples/py_native:bin
  # run the python package directly
  ./bazel-bin/examples/py_native/bin >& $TEST_log \
    || fail "//examples/py_native:bin execution failed"
  expect_log "Fib(5) == 8"
  # Using python <zipfile> to run the python package
  python ./bazel-bin/examples/py_native/bin >& $TEST_log \
    || fail "//examples/py_native:bin execution failed"
  expect_log "Fib(5) == 8"
  assert_test_ok //examples/py_native:test
  assert_test_fails //examples/py_native:fail
}

run_suite "examples on Windows"

