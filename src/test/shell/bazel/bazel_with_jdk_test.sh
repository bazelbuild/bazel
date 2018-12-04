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
# Tests that the version of Bazel with a bundled JDK works.
#

# Load the test setup defined in the parent directory
source $(rlocation io_bazel/src/test/shell/integration_test_setup.sh) \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function bazel() {
  $(rlocation io_bazel/src/bazel) --bazelrc=$TEST_TMPDIR/bazelrc "$@"
  return $?
}

function set_up() {
  # TODO(philwo) remove this when the testenv improvement change is in
  if is_windows; then
    export PATH=/c/python_27_amd64/files:$PATH
    EXTRA_BAZELRC="build --cpu=x64_windows_msvc"
    setup_bazelrc
  fi

  # The default test setup adds a --host_javabase flag, which prevents us from
  # actually using the bundled one. Remove it.
  fgrep -v -- "--host_javabase" "$TEST_TMPDIR/bazelrc" > "$TEST_TMPDIR/bazelrc.new"
  mv "$TEST_TMPDIR/bazelrc.new" "$TEST_TMPDIR/bazelrc"
  # ... but ensure JAVA_HOME is set, so we can find a default --javabase
  export JAVA_HOME="${bazel_javabase}"
}

function test_bazel_uses_bundled_jdk() {
  bazel --batch info &> "$TEST_log" || fail "bazel info failed"
  install_base="$(bazel --batch info install_base)"

  # Case-insensitive match, because Windows paths are case-insensitive.
  grep -sqi -- "^java-home: ${install_base}/_embedded_binaries/embedded_tools/jdk" $TEST_log || \
      fail "bazel's java-home is not inside the install base"
}

# Tests that "bazel license" prints the license of the bundled JDK by grepping for
# representative strings from those files. If this test breaks after upgrading the version of the
# bundled JDK, the strings may have to be updated.
function test_bazel_license_prints_jdk_license() {
  bazel --batch license \
      &> "$TEST_log" || fail "running bazel license failed"

  expect_log "OPENJDK ASSEMBLY EXCEPTION" || \
      fail "'bazel license' did not print an expected string from ASSEMBLY_EXCEPTION"

  expect_log "Provided you have not received the software directly from Azul and have already" || \
      fail "'bazel license' did not print an expected string from DISCLAIMER"

  expect_log '"CLASSPATH" EXCEPTION TO THE GPL' || \
      fail "'bazel license' did not print an expected string from LICENSE"

  expect_log "which may be included with JRE [0-9]\+, JDK [0-9]\+, and OpenJDK [0-9]\+" || \
      fail "'bazel license' did not print an expected string from THIRD_PARTY_README"
}

run_suite "bazel test suite"
