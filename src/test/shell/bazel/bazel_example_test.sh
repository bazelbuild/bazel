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

set -euo pipefail
# --- begin runfiles.bash initialization ---
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

function set_up() {
  copy_examples
  cat > WORKSPACE <<EOF
workspace(name = "io_bazel")
EOF
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
    export PATH=${bazel_javabase}/bin:"$PATH" &&
    mkdir -p x &&
    cd x &&
    unset JAVA_RUNFILES &&
    unset TEST_SRCDIR &&
    unset RUNFILES_MANIFEST_FILE &&
    unset RUNFILES_MANIFEST_ONLY &&
    unset RUNFILES_DIR &&
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
  JAVA_VERSION="$(bazel query --output=build 'kind(java_toolchain, deps(@bazel_tools//tools/jdk:toolchain))' 2>/dev/null | grep source_version | cut -d '"' -f 2)"
  if [[ -n "${JAVA_VERSION:-}" ]]; then
    JAVA_VERSION="1.${JAVA_VERSION}"
  else
    fail "Could not determine Java version."
  fi
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
  # The --javabase flag is to force the tools/jdk:jdk label to be used
  # so it appears in the dependency list.
  assert_build_output ./bazel-bin/examples/gen/genquery examples/gen:genquery --javabase=//tools/jdk
  local want=./bazel-genfiles/examples/gen/genrule.txt
  assert_build_output $want examples/gen:genrule --javabase=//tools/jdk

  diff $want ./bazel-bin/examples/gen/genquery \
    || fail "genrule and genquery output differs"

  grep -qE "^@bazel_tools//tools/jdk:jdk$" $want || {
    cat $want
    fail "@bazel_tools//tools/jdk:jdk not found in genquery output"
  }
}

function test_native_python() {
  assert_build //examples/py_native:bin --python2_path=python
  assert_test_ok //examples/py_native:test --python2_path=python
  assert_test_fails //examples/py_native:fail --python2_path=python
}

function test_native_python_with_zip() {
  assert_build //examples/py_native:bin --python2_path=python --build_python_zip
  # run the python package directly
  ./bazel-bin/examples/py_native/bin >& $TEST_log \
    || fail "//examples/py_native:bin execution failed"
  expect_log "Fib(5) == 8"
  local zipfile=./bazel-bin/examples/py_native/bin
  if is_windows; then
    zipfile="${zipfile}.zip"
  fi
  # Using python <zipfile> to run the python package
  python "$zipfile" >& $TEST_log \
    || fail "//examples/py_native:bin execution failed"
  expect_log "Fib(5) == 8"
  assert_test_ok //examples/py_native:test --python2_path=python --build_python_zip
  assert_test_fails //examples/py_native:fail --python2_path=python --build_python_zip
}

function test_shell() {
  assert_build "//examples/shell:bin"
  unset RUNFILES_DIR
  unset RUNFILES_MANIFEST_FILE
  assert_bazel_run "//examples/shell:bin" "Hello Bazel!"
  assert_test_ok "//examples/shell:test"
}

#
# Skylark rules
#
function test_python() {
  assert_build "//examples/py:bin"

  ./bazel-bin/examples/py/bin >& $TEST_log \
    || fail "//examples/py:bin execution failed"
  expect_log "Fib(5)=8"

  # Mutate //examples/py:bin so that it needs to build again.
  echo "print('Hello')" > ./examples/py/bin.py
  # Ensure that we can rebuild //examples/py::bin without error.
  assert_build "//examples/py:bin"
  ./bazel-bin/examples/py/bin >& $TEST_log \
    || fail "//examples/py:bin 2nd build execution failed"
  expect_log "Hello"
}

run_suite "examples"
