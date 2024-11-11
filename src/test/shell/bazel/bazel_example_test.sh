#!/bin/bash
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
# Tests the examples provided in Bazel
#
# --- begin runfiles.bash initialization ---
set -euo pipefail
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
    if [[ -f "$0.runfiles_manifest" ]]; then
      export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
    elif [[ -f "$0.runfiles/MANIFEST" ]]; then
      export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
    elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
      export RUNFILES_DIR="$0.runfiles"
    fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# $1 is equal to the $(JAVABASE) make variable
javabase="$1"
if [[ $javabase = external/* ]]; then
  javabase=${javabase#external/}
fi
javabase="$(rlocation "${javabase}/bin/java")"
javabase=${javabase%/bin/java}

function set_up() {
  copy_examples
  cat > MODULE.bazel <<EOF
module(name="io_bazel")
EOF
  add_rules_java "MODULE.bazel"
}

#
# Native rules
#
function test_cpp() {
  assert_build "//examples/cpp:hello-world"
  assert_bazel_run "//examples/cpp:hello-world foo" "Hello foo"
  assert_test_ok "//examples/cpp:hello-success_test"
  assert_test_fails "//examples/cpp:hello-fail_test"
}

# An assertion that execute a binary from a sub directory (to test runfiles)
function assert_binary_run_from_subdir() {
    ( # Needed to make execution from a different path work.
    export PATH=${javabase}/bin:"$PATH" &&
    mkdir -p x &&
    cd x &&
    unset JAVA_RUNFILES &&
    unset TEST_SRCDIR &&
    assert_binary_run "../$1" "$2" )
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
  assert_test_ok "${java_native_tests}:hello"
  assert_test_ok "${java_native_tests}:custom"
  assert_test_fails "${java_native_tests}:fail"
  assert_test_fails "${java_native_tests}:resource-fail"
}

function test_java_test_with_junitrunner() {
  # Test with junitrunner.
  setup_javatest_support
  local java_native_tests=//examples/java-native/src/test/java/com/example/myproject
  assert_test_ok "${java_native_tests}:custom_with_test_class"
}

function test_genrule_and_genquery() {
  # With toolchain resolution java runtime only appears in cquery results.
  assert_build_output ./bazel-bin/examples/gen/genquery examples/gen:genquery
  local want=./bazel-genfiles/examples/gen/genrule.txt
  assert_build_output $want examples/gen:genrule

  diff $want ./bazel-bin/examples/gen/genquery \
    || fail "genrule and genquery output differs"

  grep -vqE "^@local_jdk//:jdk$" $want || {
    cat $want
    fail "@local_jdk//:jdk found in genquery output"
  }
}

function test_native_python() {
  assert_build //examples/py_native:bin
  assert_test_ok //examples/py_native:test
  assert_test_fails //examples/py_native:fail
}

function test_native_python_with_zip() {
  assert_build //examples/py_native:bin --build_python_zip
  # run the python package directly
  ./bazel-bin/examples/py_native/bin >& $TEST_log \
    || fail "//examples/py_native:bin execution failed"
  expect_log "Fib(5) == 8"
  # Using python <zipfile> to run the python package
  python ./bazel-bin/examples/py_native/bin >& $TEST_log \
    || fail "//examples/py_native:bin execution failed"
  expect_log "Fib(5) == 8"
  assert_test_ok //examples/py_native:test --build_python_zip
  assert_test_fails //examples/py_native:fail --build_python_zip
}

function test_shell() {
  assert_build "//examples/shell:bin"
  assert_bazel_run "//examples/shell:bin" "Hello Bazel!"
  assert_test_ok "//examples/shell:test"
}

#
# Starlark rules
#
function test_python() {
  assert_build "//examples/py:bin"

  # Don't invoke the Python binary with RUNFILES_* set, as that causes
  # it to look in the runfiles directory of this test, instead of the
  # one belonging to the Python binary.
  env -u RUNFILES_DIR -u RUNFILES_MANIFEST_FILE \
    ./bazel-bin/examples/py/bin >& $TEST_log \
    || fail "//examples/py:bin execution failed"
  expect_log "Fib(5)=8"

  # Mutate //examples/py:bin so that it needs to build again.
  echo "print('Hello')" > ./examples/py/bin.py
  # Ensure that we can rebuild //examples/py::bin without error.
  assert_build "//examples/py:bin"
  env -u RUNFILES_DIR -u RUNFILES_MANIFEST_FILE \
    ./bazel-bin/examples/py/bin >& $TEST_log \
    || fail "//examples/py:bin 2nd build execution failed"
  expect_log "Hello"
}

function test_java_starlark() {
  local java_pkg=examples/java-starlark/src/main/java/com/example/myproject
  assert_build_output ./bazel-bin/${java_pkg}/libhello-lib.jar ${java_pkg}:hello-lib
  assert_build_output ./bazel-bin/${java_pkg}/hello-data ${java_pkg}:hello-data
  assert_build_output ./bazel-bin/${java_pkg}/hello-world ${java_pkg}:hello-world
  # we built hello-world but hello-data is still there.
  want=./bazel-bin/${java_pkg}/hello-data
  test -x $want || fail "executable $want not found"
  assert_binary_run_from_subdir "bazel-bin/${java_pkg}/hello-data foo" "Heyo foo"
}

function test_java_test_starlark() {
  setup_starlark_javatest_support
  javatests=examples/java-starlark/src/test/java/com/example/myproject
  assert_build //${javatests}:pass
  assert_test_ok //${javatests}:pass
  assert_test_fails //${javatests}:fail
}

run_suite "examples"
