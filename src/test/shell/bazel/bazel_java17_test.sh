#!/bin/bash
#
# Copyright 2021 The Bazel Authors. All rights reserved.
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
# Tests that bazel runs projects with Java 17 features.

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

JAVA_TOOLS_ZIP="$1"; shift
JAVA_TOOLS_PREBUILT_ZIP="$1"; shift

echo "JAVA_TOOLS_ZIP=$JAVA_TOOLS_ZIP"


JAVA_TOOLS_RLOCATION=$(rlocation io_bazel/$JAVA_TOOLS_ZIP)

if "$is_windows"; then
    JAVA_TOOLS_ZIP_FILE_URL="file:///${JAVA_TOOLS_RLOCATION}"
    JAVA_TOOLS_PREBUILT_ZIP_FILE_URL="file:///$(rlocation io_bazel/$JAVA_TOOLS_PREBUILT_ZIP)"
else
    JAVA_TOOLS_ZIP_FILE_URL="file://${JAVA_TOOLS_RLOCATION}"
    JAVA_TOOLS_PREBUILT_ZIP_FILE_URL="file://$(rlocation io_bazel/$JAVA_TOOLS_PREBUILT_ZIP)"
fi
JAVA_TOOLS_ZIP_FILE_URL=${JAVA_TOOLS_ZIP_FILE_URL:-}
JAVA_TOOLS_PREBUILT_ZIP_FILE_URL=${JAVA_TOOLS_PREBUILT_ZIP_FILE_URL:-}


function set_up() {
    cat >>WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
# java_tools versions only used to test Bazel with various JDK toolchains.

http_archive(
    name = "remote_java_tools",
    urls = ["${JAVA_TOOLS_ZIP_FILE_URL}"]
)
http_archive(
    name = "remote_java_tools_linux",
    urls = ["${JAVA_TOOLS_PREBUILT_ZIP_FILE_URL}"]
)
http_archive(
    name = "remote_java_tools_windows",
    urls = ["${JAVA_TOOLS_PREBUILT_ZIP_FILE_URL}"]
)
http_archive(
    name = "remote_java_tools_darwin_x86_64",
    urls = ["${JAVA_TOOLS_PREBUILT_ZIP_FILE_URL}"]
)
http_archive(
    name = "remote_java_tools_darwin_arm64",
    urls = ["${JAVA_TOOLS_PREBUILT_ZIP_FILE_URL}"]
)
EOF
}

# Java source files version shall match --java_language_version_flag version.
function test_java17_text_block() {
  mkdir -p java/main
  cat >java/main/BUILD <<EOF
java_binary(
    name = 'Javac17Example',
    srcs = ['Javac17Example.java'],
    main_class = 'Javac17Example',
)
EOF

  cat >java/main/Javac17Example.java <<EOF
public class Javac17Example {
  static Object textBlock = """
              Hello,
              World
              """;

  static sealed class Foo permits Bar {}

  static final class Bar extends Foo {}

  public static void main(String[] args) {
    System.out.println(textBlock);
  }
}
EOF

  bazel run java/main:Javac17Example --java_language_version=11 --java_runtime_version=11 \
     --test_output=all --verbose_failures &>"${TEST_log}" \
     && fail "Running with --java_language_version=11 unexpectedly succeeded."

  bazel run java/main:Javac17Example --java_language_version=17 --java_runtime_version=17 \
     --test_output=all --verbose_failures &>"${TEST_log}" \
     || fail "Running with --java_language_version=17 failed"
  expect_log "^Hello,\$"
  expect_log "^World\$"
}


run_suite "Tests Java 17 language features"
