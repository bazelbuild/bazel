#!/usr/bin/env bash
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

# Load the test setup defined in the parent directory
source $(rlocation io_bazel/src/test/shell/integration_test_setup.sh) \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

if ! is_windows; then
  echo "This test suite must be run on Windows." >&2
  exit 0
fi

export JAVA_VERSION="$1"

setup_javabase
export JAVA_HOME=${bazel_javabase}

function set_up() {
  setup_bazelrc
  add_to_bazelrc "build --java_language_version=$JAVA_VERSION"
  add_to_bazelrc "build --java_runtime_version=local_jdk"
}

# An assertion that execute a binary from a sub directory (to test runfiles)
function assert_binary_run_from_subdir() {
    ( # Needed to make execution from a different path work.
    export PATH=${bazel_javabase}/bin:"$PATH" &&
    mkdir -p x &&
    cd x &&
    unset JAVA_RUNFILES &&
    unset TEST_SRCDIR &&
    unset RUNFILES_MANIFEST_FILE &&
    unset RUNFILES_MANIFEST_ONLY &&
    unset RUNFILES_DIR &&
    assert_binary_run "../$1" "$2" )
}

function create_tmp_drive() {
  mkdir "$TEST_TMPDIR/tmp_drive"

  TMP_DRIVE_PATH=$(cygpath -w "$TEST_TMPDIR\\tmp_drive")
  for X in {A..Z}
  do
    TMP_DRIVE=${X}
    subst ${TMP_DRIVE}: ${TMP_DRIVE_PATH} >NUL || TMP_DRIVE=""
    if [ -n "${TMP_DRIVE}" ]; then
      break
    fi
  done

  if [ -z "${TMP_DRIVE}" ]; then
    fail "Cannot create temporary drive."
  fi

  export TMP_DRIVE
}

function delete_tmp_drive() {
  if [ -n "${TMP_DRIVE}" ]; then
    subst ${TMP_DRIVE}: /D
  fi
}

function test_java_with_jar_under_different_drive() {
  create_tmp_drive

  trap delete_tmp_drive EXIT

  add_rules_java "MODULE.bazel"

  cat > BUILD <<'EOF'
load("@rules_java//java:defs.bzl", "java_binary")

java_binary(
  name = "hello",
  srcs = ["Hello.java"],
  main_class = "pkg.Hello",
)
EOF
  cat > Hello.java <<'EOF'
package pkg;

public class Hello {
  public static void main(String[] args) {
    String[] version = System.getProperty("java.version").split("\\.");
    String major = version[0].equals("1") ? version[1] : version[0];
    System.out.printf("Java %s\n", major);
  }
}
EOF

  bazel --output_user_root=${TMP_DRIVE}:/tmp build :hello &> "$TEST_log" \
      || fail "build failed"

  # TODO(tjgq): Remove explicit `--java_version` once rules_java sets it.
  assert_binary_run_from_subdir "bazel-bin/hello --classpath_limit=0 --java_version=$JAVA_VERSION" \
    "Java $JAVA_VERSION"
}

run_suite "Tests that the Java Windows launcher works with cross-drive jars."
