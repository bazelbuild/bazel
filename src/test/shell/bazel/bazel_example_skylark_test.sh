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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  copy_examples
  cat > WORKSPACE <<EOF
workspace(name = "io_bazel")
EOF
}

function test_java_skylark() {
  # Must be ported to Windows. Currently fails because:
  # - Greeter.java uses runfiles, so it must depend on
  #   @bazel_tools//tools/runfiles:java-runfiles and use Runfiles::rlocation
  # - The Skylark implementation of the java_* rules does not use JavaInfo and
  #   therefore cannot depend on native java rules.
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
  # Must be ported to Windows. Currently fails because:
  # - The Skylark implementation of the java_test creates a stub script without
  #   Windows path support
  setup_skylark_javatest_support
  javatests=examples/java-skylark/src/test/java/com/example/myproject
  assert_build //${javatests}:pass
  assert_test_ok //${javatests}:pass
  assert_test_fails //${javatests}:fail
}

run_suite "examples"
