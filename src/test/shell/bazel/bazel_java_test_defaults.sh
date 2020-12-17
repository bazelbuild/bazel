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
# Tests the java rules with the default values provided by Bazel.
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

# Java source files version shall match --java_language_version_flag version.
# Output class files shall be created in corresponding version (JDK 8, class version is 52).
function test_default_java_toolchain_target_version() {
  mkdir -p java/main
  cat >java/main/BUILD <<EOF
java_binary(
    name = 'JavaBinary',
    srcs = ['JavaBinary.java'],
    main_class = 'JavaBinary',
)
load(
    "@bazel_tools//tools/jdk:default_java_toolchain.bzl",
    "default_java_toolchain",
)
default_java_toolchain(
  name = "default_toolchain",
  source_version = "8",
  target_version = "8",
  visibility = ["//visibility:public"],
)
EOF

   cat >java/main/JavaBinary.java <<EOF
public class JavaBinary {
   public static void main(String[] args) {
    System.out.println("Successfully executed JavaBinary!");
  }
}
EOF
  bazel run java/main:JavaBinary \
      --java_language_version=8 \
      --java_runtime_version=11 \
      --extra_toolchains=//java/main:default_toolchain_definition \
      --verbose_failures -s &>"${TEST_log}" \
      || fail "Building with //java/main:default_toolchain failed"
  expect_log "Successfully executed JavaBinary!"
  javap -verbose -cp bazel-bin/java/main/JavaBinary.jar JavaBinary | grep major &>"${TEST_log}"
  expect_log "major version: 52"
}

# Java source files version shall match --java_language_version_flag version.
# Output class files shall be created in corresponding version (JDK 11, class version is 55).
function test_java_language_version_output_classes() {
  mkdir -p java/main
  cat >java/main/BUILD <<EOF
java_binary(
    name = 'JavaBinary',
    srcs = ['JavaBinary.java'],
    main_class = 'JavaBinary',
)
EOF

   cat >java/main/JavaBinary.java <<EOF
public class JavaBinary {
   public static void main(String[] args) {
    // Java 11 new String methods.
    String myString = "   strip_trailing_java11   ";
    System.out.println(myString.stripLeading().stripTrailing());
  }
}
EOF
  bazel run java/main:JavaBinary --java_language_version=11 --java_runtime_version=11 \
      --verbose_failures -s &>"${TEST_log}" \
      || fail "Building with --java_language_version=11 failed"
  expect_log "strip_trailing_java11"
  javap -verbose -cp bazel-bin/java/main/JavaBinary.jar JavaBinary | grep major &>"${TEST_log}"
  expect_log "major version: 55"

  bazel run java/main:JavaBinary --java_language_version=15 --java_runtime_version=15 \
      --verbose_failures -s &>"${TEST_log}" \
      || fail "Building with --java_language_version=15 failed"
  expect_log "strip_trailing_java11"
  javap -verbose -cp bazel-bin/java/main/JavaBinary.jar JavaBinary | grep major &>"${TEST_log}"
  expect_log "major version: 59"
}

function test_tools_jdk_toolchain_nojacocorunner() {
  mkdir -p java/main
  cat >java/main/BUILD <<EOF
java_binary(
    name = 'JavaBinary',
    srcs = ['JavaBinary.java'],
    main_class = 'JavaBinary',
)
load(
    "@bazel_tools//tools/jdk:default_java_toolchain.bzl",
    "default_java_toolchain",
)
default_java_toolchain(
  name = "default_toolchain",
  jacocorunner = None,
  visibility = ["//visibility:public"],
)
EOF

   cat >java/main/JavaBinary.java <<EOF
public class JavaBinary {
   public static void main(String[] args) {
    System.out.println("Successfully executed JavaBinary!");
  }
}
EOF
  bazel coverage java/main:JavaBinary \
      --java_runtime_version=11 \
      --extra_toolchains=//java/main:default_toolchain_definition \
      --verbose_failures -s &>"${TEST_log}" \
      && fail "Coverage succeeded even when jacocorunner not set"
  expect_log "jacocorunner not set in java_toolchain:"
}


function test_default_java_toolchain_manualConfiguration() {
  cat > BUILD <<EOF
load("@bazel_tools//tools/jdk:default_java_toolchain.bzl", "default_java_toolchain")
default_java_toolchain(
  name = "vanilla",
  javabuilder = ["//:VanillaJavaBuilder"],
  jvm_opts = [],
)
EOF
  bazel build //:vanilla || fail "default_java_toolchain target failed to build"
}

function test_default_java_toolchain_manualConfigurationWithLocation() {
  cat > BUILD <<EOF
load("@bazel_tools//tools/jdk:default_java_toolchain.bzl", "default_java_toolchain", "JDK9_JVM_OPTS")
default_java_toolchain(
  name = "toolchain",
  jvm_opts = [
      # In JDK9 we have seen a ~30% slow down in JavaBuilder performance when using
      # G1 collector and having compact strings enabled.
      "-XX:+UseParallelOldGC",
      "-XX:-CompactStrings",
      # override the javac in the JDK.
      "--patch-module=java.compiler=\$(location @remote_java_tools//:java_compiler_jar)",
      "--patch-module=jdk.compiler=\$(location @remote_java_tools//:jdk_compiler_jar)",
  ] + JDK9_JVM_OPTS,
  tools = [
      "@remote_java_tools//:java_compiler_jar",
      "@remote_java_tools//:jdk_compiler_jar",
    ],
)
EOF
  bazel build //:toolchain || fail "default_java_toolchain target failed to build"
}

function test_default_java_toolchain_jvm8Toolchain() {
  cat > BUILD <<EOF
load("@bazel_tools//tools/jdk:default_java_toolchain.bzl", "default_java_toolchain", "JVM8_TOOLCHAIN_CONFIGURATION")
default_java_toolchain(
  name = "jvm8_toolchain",
  configuration = JVM8_TOOLCHAIN_CONFIGURATION,
  java_runtime = "@local_jdk//:jdk",
)
EOF
  bazel query //:jvm8_toolchain || fail "default_java_toolchain target failed to build"
}

function test_default_java_toolchain_javabuilderToolchain() {
  cat > BUILD <<EOF
load("@bazel_tools//tools/jdk:default_java_toolchain.bzl", "default_java_toolchain", "DEFAULT_TOOLCHAIN_CONFIGURATION")
default_java_toolchain(
  name = "javabuilder_toolchain",
  configuration = DEFAULT_TOOLCHAIN_CONFIGURATION,
)
EOF
  bazel build //:javabuilder_toolchain || fail "default_java_toolchain target failed to build"
}

function test_default_java_toolchain_vanillaToolchain() {
  cat > BUILD <<EOF
load("@bazel_tools//tools/jdk:default_java_toolchain.bzl", "default_java_toolchain", "VANILLA_TOOLCHAIN_CONFIGURATION")
default_java_toolchain(
  name = "vanilla_toolchain",
  configuration = VANILLA_TOOLCHAIN_CONFIGURATION,
  java_runtime = "@local_jdk//:jdk",
)
EOF
  bazel build //:vanilla_toolchain || fail "default_java_toolchain target failed to build"
}

function test_default_java_toolchain_prebuiltToolchain() {
  cat > BUILD <<EOF
load("@bazel_tools//tools/jdk:default_java_toolchain.bzl", "default_java_toolchain", "PREBUILT_TOOLCHAIN_CONFIGURATION")
default_java_toolchain(
  name = "prebuilt_toolchain",
  configuration = PREBUILT_TOOLCHAIN_CONFIGURATION,
)
EOF
  bazel build //:prebuilt_toolchain || fail "default_java_toolchain target failed to build"
}

function test_default_java_toolchain_nonprebuiltToolchain() {
  cat > BUILD <<EOF
load("@bazel_tools//tools/jdk:default_java_toolchain.bzl", "default_java_toolchain", "NONPREBUILT_TOOLCHAIN_CONFIGURATION")
default_java_toolchain(
  name = "nonprebuilt_toolchain",
  configuration = NONPREBUILT_TOOLCHAIN_CONFIGURATION,
)
EOF
  bazel build //:nonprebuilt_toolchain || fail "default_java_toolchain target failed to build"
}


run_suite "Java integration tests with default Bazel values"
