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

function test_host_javabase() {
  mkdir -p foobar/bin
  cat << EOF > BUILD
java_runtime(
    name = "host_javabase",
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
  bazel aquery --output=text --host_javabase=//:host_javabase //java:javalib >& $TEST_log
  expect_log "exec .*foobar/bin/java"

  # If we don't specify anything, we expect the embedded JDK to be used.
  # Note that this will change in the future but is the current state.
  bazel aquery --output=text //java:javalib >& $TEST_log
  expect_not_log "exec external/embedded_jdk/bin/java"
  expect_log "exec external/remotejdk11_.*/bin/java"

  bazel aquery --output=text --host_javabase=//:host_javabase \
    //java:javalib >& $TEST_log
  expect_log "exec .*foobar/bin/java"
  expect_not_log "exec external/remotejdk_.*/bin/java"

  bazel aquery --output=text --incompatible_use_jdk11_as_host_javabase \
    //java:javalib >& $TEST_log
  expect_log "exec external/remotejdk11_.*/bin/java"
}

function test_javabase() {
  mkdir -p zoo/bin
  cat << EOF > BUILD
load("@bazel_tools//tools/jdk:default_java_toolchain.bzl", "default_java_toolchain")
default_java_toolchain(
    name = "toolchain",
    # Implicitly use the host_javabase bootclasspath, since the target doesn't
    # exist in this test.
    bootclasspath = [],
    javabuilder = ["@bazel_tools//tools/jdk:vanillajavabuilder"],
    jvm_opts = [],
    visibility = ["//visibility:public"],
)
java_runtime(
    name = "javabase",
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
  bazel build --java_toolchain=//:toolchain --javabase=//:javabase //java:javabin
  cat bazel-bin/java/javabin >& $TEST_log
  expect_log "JAVABIN=.*/zoo/bin/java"

  # Check that we use local_jdk when it's not specified.
  bazel build //java:javabin
  cat bazel-bin/java/javabin >& $TEST_log
  expect_log "JAVABIN=.*/local_jdk/bin/java"
}

function write_javabase_files() {
  mkdir -p javabase_test
  cat << EOF > javabase_test/BUILD
java_binary(
    name = "a",
    srcs = ["A.java"],
    main_class = "A",
)
EOF

  cat << EOF > javabase_test/A.java
class A {
  public static void main(String[] args) {
    System.err.println("hello");
  }
}
EOF
}

function test_no_javabase() {
  # Only run this test when there's no locally installed JDK.
  which javac && return

  write_javabase_files

  bazel build //javabase_test:a

  ($(bazel-bin/javabase_test/a --print_javabin) -version || true) >& $TEST_log
  expect_log "bazel-bin/javabase_test/a.runfiles/local_jdk/bin/java: No such file or directory"
}

function test_no_javabase() {
  # Only run this test when there's no locally installed JDK.
  which javac && return

  write_javabase_files

  bazel --batch build //javabase_test:a

  ($(bazel-bin/javabase_test/a --print_javabin) -version || true) >& $TEST_log
  expect_log "bazel-bin/javabase_test/a.runfiles/local_jdk/bin/java: No such file or directory"
}

function test_genrule() {
  mkdir -p foo/bin bar/bin
  cat << EOF > BUILD
java_runtime(
    name = "foo_javabase",
    java_home = "$PWD/foo",
    visibility = ["//visibility:public"],
)

java_runtime(
    name = "bar_runtime",
    visibility = ["//visibility:public"],
    srcs = ["bar/bin/java"],
)

genrule(
    name = "without_java",
    srcs = ["in"],
    outs = ["out_without"],
    cmd = "cat \$(SRCS) > \$(OUTS)",
)

genrule(
    name = "with_java",
    srcs = ["in"],
    outs = ["out_with"],
    cmd = "echo \$(JAVA) > \$(OUTS)",
    toolchains = [":bar_runtime"],
)
EOF

  # Use --max_config_changes_to_show=0, as changed option names may otherwise
  # erroneously match the expected regexes.

  # Test the genrule with no java dependencies.
  bazel cquery --max_config_changes_to_show=0 --implicit_deps \
    'deps(//:without_java)' >& $TEST_log
  expect_not_log "foo"
  expect_not_log "bar"
  expect_not_log "embedded_jdk"
  expect_not_log "remotejdk_"
  expect_not_log "remotejdk11_"

  # Test the genrule that specifically depends on :bar_runtime.
  bazel cquery --max_config_changes_to_show=0 --implicit_deps \
    'deps(//:with_java)' >& $TEST_log
  expect_not_log "foo"
  expect_log "bar"
  expect_not_log "embedded_jdk"
  expect_not_log "remotejdk_"
  expect_not_log "remotejdk11_"

  # Setting the javabase should not change the use of :bar_runtime from the
  # roolchains attribute.
  bazel cquery --max_config_changes_to_show=0 --implicit_deps \
    'deps(//:with_java)' --host_javabase=:foo_javabase >& $TEST_log
  expect_not_log "foo"
  expect_log "bar"
  expect_not_log "embedded_jdk"
  expect_not_log "remotejdk_"
  expect_not_log "remotejdk11_"
}

run_suite "Tests of specifying custom server_javabase/host_javabase and javabase."
