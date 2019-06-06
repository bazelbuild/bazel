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
# Tests that bazel runs projects with Java 12 features.

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

JAVA_TOOLCHAIN="$1"; shift
JAVA_TOOLS_ZIP="$1"; shift
JAVA_RUNTIME="$1"; shift

echo "JAVA_TOOLS_ZIP=$JAVA_TOOLS_ZIP"


JAVA_TOOLS_RLOCATION=$(rlocation io_bazel/$JAVA_TOOLS_ZIP)

if "$is_windows"; then
    JAVA_TOOLS_ZIP_FILE_URL="file:///${JAVA_TOOLS_RLOCATION}"
else
    JAVA_TOOLS_ZIP_FILE_URL="file://${JAVA_TOOLS_RLOCATION}"
fi
JAVA_TOOLS_ZIP_FILE_URL=${JAVA_TOOLS_ZIP_FILE_URL:-}

add_to_bazelrc "build --java_toolchain=${JAVA_TOOLCHAIN}"
add_to_bazelrc "build --host_java_toolchain=${JAVA_TOOLCHAIN}"
add_to_bazelrc "build --javabase=${JAVA_RUNTIME}"
add_to_bazelrc "build --host_javabase=${JAVA_RUNTIME}"

function set_up() {
    cat >>WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
# java_tools versions only used to test Bazel with various JDK toolchains.

http_archive(
    name = "local_java_tools",
    urls = ["${JAVA_TOOLS_ZIP_FILE_URL}"]
)
EOF
    cat $(rlocation io_bazel/src/test/shell/bazel/testdata/jdk_http_archives) >> WORKSPACE
}

function test_java12_switch_statement() {
  mkdir -p java/main
  cat >java/main/BUILD <<EOF
java_binary(
    name = 'Javac12Example',
    srcs = ['Javac12Example.java'],
    main_class = 'Javac12Example',
    javacopts = ["--enable-preview"],
    jvm_flags = ["--enable-preview"],
)
EOF

  cat >java/main/Javac12Example.java <<EOF
public class Javac12Example {
  enum Day {
    MONDAY,
    TUESDAY,
    WEDNESDAY,
    THURSDAY,
    FRIDAY,
    SATURDAY,
    SUNDAY
  };

  public static void main(String[] args) {
    Day day = Day.MONDAY;
    int numLetters = switch (day) {
      case MONDAY, FRIDAY, SUNDAY -> 6;
      case TUESDAY -> 7;
      case THURSDAY, SATURDAY -> 8;
      case WEDNESDAY -> 9;
      default -> throw new IllegalStateException("Huh? " + day);
    };
    System.out.println("Associated day number is " + numLetters);
  }
}
EOF
  bazel run java/main:Javac12Example --test_output=all --verbose_failures &>"${TEST_log}"
  expect_log "Associated day number is 6"
}

run_suite "Tests new Java 12 language features"
