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
      --java_toolchain=//java/main:default_toolchain \
      --javabase=@bazel_tools//tools/jdk:remote_jdk11 \
      --verbose_failures -s &>"${TEST_log}" \
      || fail "Building with //java/main:default_toolchain failed"
  expect_log "Successfully executed JavaBinary!"
  javap -verbose -cp bazel-bin/java/main/JavaBinary.jar JavaBinary | grep major &>"${TEST_log}"
  expect_log "major version: 52"
}

function test_tools_jdk_toolchain_java10() {
  mkdir -p java/main
  cat >java/main/BUILD <<EOF
java_binary(
    name = 'JavaBinary',
    srcs = ['JavaBinary.java'],
    main_class = 'JavaBinary',
)
EOF

   cat >java/main/JavaBinary.java <<EOF
import java.util.ArrayList;
public class JavaBinary {
   public static void main(String[] args) {
    var myList = new ArrayList<String>();
    for (int i = 0; i < 3; i++) {
      myList.add("myString" + i);
    }

    for (String string : myList) {
      System.out.println(string);
    }
  }
}
EOF
  bazel run java/main:JavaBinary \
      --java_toolchain=@bazel_tools//tools/jdk:toolchain_java10 \
      --javabase=@bazel_tools//tools/jdk:remote_jdk10 \
      --verbose_failures -s &>"${TEST_log}" \
      || fail "Building with @bazel_tools//tools/jdk:toolchain_java10 failed"
  expect_log "myString0"
  expect_log "myString1"
  expect_log "myString2"
  javap -verbose -cp bazel-bin/java/main/JavaBinary.jar JavaBinary | grep major &>"${TEST_log}"
  expect_log "major version: 54"
}

function test_tools_jdk_toolchain_java11() {
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
  bazel run java/main:JavaBinary \
      --java_toolchain=@bazel_tools//tools/jdk:toolchain_java11 \
      --javabase=@bazel_tools//tools/jdk:remote_jdk11 \
      --verbose_failures -s &>"${TEST_log}" \
      || fail "Building with @bazel_tools//tools/jdk:toolchain_java11 failed"
  expect_log "strip_trailing_java11"
  javap -verbose -cp bazel-bin/java/main/JavaBinary.jar JavaBinary | grep major &>"${TEST_log}"
  expect_log "major version: 55"
}

run_suite "Java integration tests with default Bazel values"