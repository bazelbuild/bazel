#!/bin/bash
# Copyright 2018 The Bazel Authors. All rights reserved.
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

# Tests that --server_javabase/--host_javabase and --javabase work as expected
# for Bazel with the embedded JDK.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_server_javabase() {
  mkdir -p test_server_javabase/bin
  MAGIC="the cake is a lie"

  cat << EOF > test_server_javabase/bin/java
#!/bin/bash
echo "$MAGIC"
EOF
  chmod +x test_server_javabase/bin/java

  # Check that we're able to change the server_javabase to a user specified
  # version.
  bazel --batch --server_javabase=test_server_javabase version >& $TEST_log
  expect_log "$MAGIC"

  bazel --batch version >& $TEST_log
  expect_not_log "$MAGIC"

  # Check that we're using the embedded JDK by default as server_javabase.
  bazel --batch info >& $TEST_log
  expect_log "java-home: .*/_embedded_binaries/embedded_tools/jdk"
}

function test_rhs_host_javabase() {
  mkdir -p foobar/bin
  cat << EOF > BUILD
java_runtime(
    name = "rhs_host_javabase",
    java_home = "$PWD/foobar",
    visibility = ["//visibility:public"],
)
EOF

  mkdir java
  cat << EOF > java/BUILD
java_library(
    name = "javalib",
    srcs = ["HelloWorld.java"],
)
EOF
  touch java/HelloWorld.java

  # We expect the given host_javabase to appear in the command line of
  # java_library actions.
  bazel aquery --output=text --host_javabase=//:rhs_host_javabase //java:javalib >& $TEST_log
  expect_log "exec .*foobar/bin/java"

  # If we don't specify anything, we expect the embedded JDK to be used.
  # Note that this will change in the future but is the current state.
  bazel aquery --output=text //java:javalib >& $TEST_log
  expect_log "exec external/embedded_jdk/bin/java"
}

function test_rhs_javabase() {
  mkdir -p zoo/bin
  cat << EOF > BUILD
java_runtime(
    name = "rhs_javabase",
    java_home = "$PWD/zoo",
    visibility = ["//visibility:public"],
)
EOF

  mkdir java
  cat << EOF > java/BUILD
java_binary(
    name = "javabin",
    srcs = ["HelloWorld.java"],
)
EOF
  cat << EOF > java/HelloWorld.java
public class HelloWorld {}
EOF

  # Check that the RHS javabase appears in the launcher.
  bazel build --javabase=//:rhs_javabase //java:javabin
  cat bazel-bin/java/javabin >& $TEST_log
  expect_log "JAVABIN=.*/zoo/bin/java"

  # Check that we use local_jdk when it's not specified.
  bazel build //java:javabin
  cat bazel-bin/java/javabin >& $TEST_log
  expect_log "JAVABIN=.*/local_jdk/bin/java"
}


run_suite "Tests of specifying custom server_javabase/host_javabase and javabase."
