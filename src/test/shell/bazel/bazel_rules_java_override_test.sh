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

RULES_JAVA_REPO_NAME=$(cat "$(rlocation io_bazel/src/test/shell/bazel/RULES_JAVA_REPO_NAME)")

function setup_java_library_target() {
    cat > BUILD <<'EOF'
java_library(
    name = "math",
    srcs = ["src/main/com/example/Math.java"],
    visibility = ["//visibility:public"],
)
EOF

    mkdir -p src/main/com/example
    cat > src/main/com/example/Math.java <<'EOF'
package com.example;

public class Math {

  public static boolean isEven(int n) {
    return n % 2 == 0;
  }
}
EOF
}

function test_override_with_empty_java_tools_fails() {

  touch emptyfile
  zip -q "${RUNFILES_DIR}/empty.zip" emptyfile

  override_java_tools "${RULES_JAVA_REPO_NAME}" "empty.zip" "empty.zip"

  setup_java_library_target

  bazel build //:math &>$TEST_log && fail "expected build failure" || true

  expect_log "ERROR: no such package '@@rules_java"
}

function test_build_without_override_succeeds() {
  override_java_tools "${RULES_JAVA_REPO_NAME}" "released" "released"

  setup_java_library_target

  bazel build //:math &>$TEST_log || fail "expected build success"
}

run_suite "rules_java override tests"
