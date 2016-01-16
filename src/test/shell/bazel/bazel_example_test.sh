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

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

function set_up() {
  copy_examples
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
  assert_build_fails "${java_native_main}:hello-error-prone" \
      "Did you mean 'result = b == -1;'?"
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

function test_java_test_with_workspace_name() {
  local java_pkg=examples/java-native/src/main/java/com/example/myproject
  # Use named workspace and test if we can still execute hello-world
  bazel clean

  rm -f WORKSPACE
  cat >WORKSPACE <<'EOF'
workspace(name = "toto")
EOF

  assert_build_output ./bazel-bin/${java_pkg}/hello-world ${java_pkg}:hello-world
  assert_binary_run_from_subdir "bazel-bin/${java_pkg}/hello-world foo" "Hello foo"
}

function test_genrule_and_genquery() {
  # The --javabase flag is to force the tools/jdk:jdk label to be used
  # so it appears in the dependency list.
  assert_build_output ./bazel-bin/examples/gen/genquery examples/gen:genquery --javabase=//tools/jdk
  local want=./bazel-genfiles/examples/gen/genrule.txt
  assert_build_output $want examples/gen:genrule --javabase=//tools/jdk

  diff $want ./bazel-bin/examples/gen/genquery \
    || fail "genrule and genquery output differs"

  grep -qE "^//tools/jdk:jdk$" $want || {
    cat $want
    fail "//tools/jdk:jdk not found in genquery output"
  }
}

if [ "${PLATFORM}" = "darwin" ]; then
  function test_objc() {
    setup_objc_test_support
    # https://github.com/bazelbuild/bazel/issues/162
    # prevents us from running iOS tests.
    # TODO(bazel-team): Execute iOStests here when this issue is resolved.
    assert_build_output ./bazel-bin/examples/objc/PrenotCalculator.ipa \
        --ios_sdk_version=$IOS_SDK_VERSION //examples/objc:PrenotCalculator
  }
fi

function test_native_python() {
  assert_build //examples/py_native:bin --python2_path=python
  assert_test_ok //examples/py_native:test --python2_path=python
  assert_test_fails //examples/py_native:fail --python2_path=python
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

function test_java_skylark() {
  local java_pkg=examples/java-skylark/src/main/java/com/example/myproject
  assert_build_output ./bazel-bin/${java_pkg}/libhello-lib.jar ${java_pkg}:hello-lib
  assert_build_output ./bazel-bin/${java_pkg}/hello-data ${java_pkg}:hello-data
  assert_build_output ./bazel-bin/${java_pkg}/hello-world ${java_pkg}:hello-world
  # we built hello-world but hello-data is still there.
  want=./bazel-bin/${java_pkg}/hello-data
  test -x $want || fail "executable $want not found"
  assert_binary_run_from_subdir "bazel-bin/${java_pkg}/hello-data foo" "Heyo foo"
}

function test_java_test_skylark() {
  setup_skylark_javatest_support
  javatests=examples/java-skylark/src/test/java/com/example/myproject
  assert_build //${javatests}:pass
  assert_test_ok //${javatests}:pass
  assert_test_fails //${javatests}:fail
}

function test_protobuf() {
  setup_protoc_support
  local jar=bazel-bin/examples/proto/libtest_proto.jar
  assert_build_output $jar //examples/proto:test_proto
  unzip -v $jar | grep -q 'KeyVal\.class' \
    || fail "Did not find KeyVal class in proto jar."
}

run_suite "examples"
