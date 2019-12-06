#!/bin/bash
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

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

declare -a PTR_BAZEL_ARGS=("--test_strategy=standalone" \
    "--strategy=TestRunner=worker" \
    "--experimental_persistent_test_runner" \
    "--test_output=all" \
    "--worker_verbose")

function set_up() {
  setup_javatest_support

  if "$is_windows"; then
      java_tools_url="file:///$(rlocation io_bazel/src/java_tools_java11.zip)"
  else
      java_tools_url="file://$(rlocation io_bazel/src/java_tools_java11.zip)"
  fi

  cat >>WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "remote_java_tools_linux",
    urls = ["${java_tools_url}"]
)
http_archive(
    name = "remote_java_tools_darwin",
    urls = ["${java_tools_url}"]
)
http_archive(
    name = "remote_java_tools_windows",
    urls = ["${java_tools_url}"]
)
EOF
}

function test_java_test_persistent_test_runner() {
  mkdir -p javatests/com/google/ptr

  cat > javatests/com/google/ptr/DummyTest.java <<EOF
package com.google.ptr;

import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.Test;

@RunWith(JUnit4.class)
public class DummyTest {

  @Test
  public void dummyTest() {
    System.out.println("dummyTest was run");
  }
}
EOF

  cat > javatests/com/google/ptr/BUILD <<EOF
java_test(
    name = "DummyTest",
    srcs = ["DummyTest.java"],
    deps = [
        "//third_party:junit4",
    ],
)
EOF

  # Make sure we start clean with no workers already in use.
  bazel clean
  # The first test run creates the TestRunner worker.
  bazel test javatests/com/google/ptr:DummyTest "${PTR_BAZEL_ARGS[@]}" \
      &> "${TEST_log}" || fail "Expected success"
  expect_log "dummyTest was run"
  expect_log "Created new non-sandboxed TestRunner worker (id [0-9]\+)"

  # Change the test to fail in order to make sure the persistent test runner
  # picks up the latest test changes.
  cat > javatests/com/google/ptr/DummyTest.java <<EOF
package com.google.ptr;

import static org.junit.Assert.fail;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.Test;

@RunWith(JUnit4.class)
public class DummyTest {

  @Test
  public void dummyTest() {
    System.out.println("dummyTest will fail");
    fail();
  }
}
EOF

  # The second run uses the previously created worker. Does not create
  # any new workers.
  bazel test javatests/com/google/ptr:DummyTest "${PTR_BAZEL_ARGS[@]}" \
      &> "${TEST_log}" && fail "Expected failure" || true

  expect_log "dummyTest will fail"
  expect_log "There was 1 failure"
  expect_log "at com.google.ptr.DummyTest.dummyTest(DummyTest.java:14)"
  expect_not_log "Created new non-sandboxed TestRunner worker"
  expect_not_log "Destroying TestRunner worker (id [0-9]\+)"

  # Change the test to pass again but with a different content to avoid
  # cached test results.
  cat > javatests/com/google/ptr/DummyTest.java <<EOF
package com.google.ptr;

import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.Test;

@RunWith(JUnit4.class)
public class DummyTest {

  @Test
  public void dummyTest() {
    System.out.println("Re-running dummyTest without failure.");
  }
}
EOF

  # The third run uses the previously created worker. Does not create
  # any new workers.
  bazel test javatests/com/google/ptr:DummyTest "${PTR_BAZEL_ARGS[@]}" \
      &> "${TEST_log}" || fail "Expected success"
  expect_log "Re-running dummyTest without failure."
  expect_not_log "Created new non-sandboxed TestRunner worker"
  expect_not_log "Destroying TestRunner worker (id [0-9]\+)"
}

function test_java_test_persistent_test_runner_with_dep() {
  mkdir -p javatests/com/google/ptr
  mkdir -p java/com/google/ptr

  cat > java/com/google/ptr/Dummy.java <<EOF
package com.google.ptr;

public class Dummy {
  public Dummy() { }

  public void printValue() {
    System.out.println("dummyTest was run");
  }
}
EOF

  cat > java/com/google/ptr/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
java_library(
    name = "dummy",
    srcs = ["Dummy.java"],
)
EOF

  cat > javatests/com/google/ptr/DummyTest.java <<EOF
package com.google.ptr;

import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.Test;

@RunWith(JUnit4.class)
public class DummyTest {

  @Test
  public void dummyTest() {
    new Dummy().printValue();
  }
}
EOF

  cat > javatests/com/google/ptr/BUILD <<EOF
java_test(
    name = "DummyTest",
    srcs = ["DummyTest.java"],
    deps = [
        "//java/com/google/ptr:dummy",
        "//third_party:junit4",
    ],
)
EOF

  # Make sure we start clean with no workers already in use.
  bazel clean
  # The first test run creates the TestRunner worker.
  bazel test javatests/com/google/ptr:DummyTest "${PTR_BAZEL_ARGS[@]}" \
      &> "${TEST_log}" || fail "Expected success"
 expect_log "dummyTest was run"
 expect_log "Created new non-sandboxed TestRunner worker (id [0-9]\+)"

  # Change the library content to test if the same worker is re-used
  # and if the library jar was reloaded in the PTR's classloader.
  cat > java/com/google/ptr/Dummy.java <<EOF
package com.google.ptr;

public class Dummy {
  public Dummy() { }

  public void printValue() {
    System.out.println("Re-printing dummy message.");
  }
}
EOF
  # The second run uses the previously created worker. Does not create
  # any new workers.
  bazel test javatests/com/google/ptr:DummyTest "${PTR_BAZEL_ARGS[@]}" \
      &> "${TEST_log}" || fail "Expected success"
  expect_log "Re-printing dummy message."
  expect_not_log "Created new non-sandboxed TestRunner worker"
  expect_not_log "Destroying TestRunner worker (id [0-9]\+)"
}

run_suite "Persistent Java Test Runner integration tests"
