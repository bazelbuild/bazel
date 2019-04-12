#!/bin/bash
#
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

# Load the test setup defined in the parent directory
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

# `uname` returns the current platform, e.g "MSYS_NT-10.0" or "Linux".
# `tr` converts all upper case letters to lower case.
# `case` matches the result if the `uname | tr` expression to string prefixes
# that use the same wildcards as names do in Bash, i.e. "msys*" matches strings
# starting with "msys", and "*" matches everything (it's the default case).
case "$(uname -s | tr [:upper:] [:lower:])" in
msys*)
  # As of 2019-01-15, Bazel on Windows only supports MSYS Bash.
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  # Disable MSYS path conversion that converts path-looking command arguments to
  # Windows paths (even if they arguments are not in fact paths).
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

function expect_path_in_java_tools() {
  path="$1"; shift
  java_version="$1"; shift

  count=$(zipinfo -1 $(rlocation io_bazel/src/java_tools_${java_version}.zip) | grep -c "$path")
  [[ "$count" -gt 0 ]] || fail "Path $path not found in java_tools_${java_version}.zip"
}

function test_java_tools_has_ijar() {
  expect_path_in_java_tools "java_tools/ijar" java9
  expect_path_in_java_tools "java_tools/ijar" java10
}

function test_java_tools_has_ijar_binary() {
  expect_path_in_java_tools "java_tools/ijar/ijar" java9
  expect_path_in_java_tools "java_tools/ijar/ijar" java10
}

function test_java_tools_has_zlib() {
  expect_path_in_java_tools "java_tools/zlib" java9
  expect_path_in_java_tools "java_tools/zlib" java10
}

function test_java_tools_has_native_windows() {
  expect_path_in_java_tools "java_tools/src/main/native/windows" java9
  expect_path_in_java_tools "java_tools/src/main/native/windows" java10
}

function test_java_tools_has_cpp_util() {
  expect_path_in_java_tools "java_tools/src/main/cpp/util" java9
  expect_path_in_java_tools "java_tools/src/main/cpp/util" java10
}

function test_java_tools_has_desugar_deps() {
  expect_path_in_java_tools \
      "java_tools/src/main/protobuf/desugar_deps.proto" java9
      expect_path_in_java_tools \
      "java_tools/src/main/protobuf/desugar_deps.proto" java10
}

function test_java_tools_has_singlejar() {
  expect_path_in_java_tools "java_tools/src/tools/singlejar" java9
  expect_path_in_java_tools "java_tools/src/tools/singlejar" java10
}

function test_java_tools_has_singlejar_local() {
  expect_path_in_java_tools \
      "java_tools/src/tools/singlejar/singlejar_local" java9
      expect_path_in_java_tools \
      "java_tools/src/tools/singlejar/singlejar_local" java10
}

function test_java_tools_has_VanillaJavaBuilder() {
  expect_path_in_java_tools "java_tools/VanillaJavaBuilder_deploy.jar" java9
  expect_path_in_java_tools "java_tools/VanillaJavaBuilder_deploy.jar" java10
}

function test_java_tools_has_JavaBuilder() {
  expect_path_in_java_tools "java_tools/JavaBuilder_deploy.jar" java9
  expect_path_in_java_tools "java_tools/JavaBuilder_deploy.jar" java10
}

function test_java_tools_has_turbine_direct() {
  expect_path_in_java_tools "java_tools/turbine_direct_binary_deploy.jar" java9
  expect_path_in_java_tools "java_tools/turbine_direct_binary_deploy.jar" java10
}

function test_java_tools_has_turbine_deploy() {
  expect_path_in_java_tools "java_tools/turbine_deploy.jar" java9
  expect_path_in_java_tools "java_tools/turbine_deploy.jar" java10
}

function test_java_tools_has_Runner() {
  expect_path_in_java_tools "java_tools/Runner_deploy.jar" java9
  expect_path_in_java_tools "java_tools/Runner_deploy.jar" java10
}

function test_java_tools_has_jdk_compiler() {
  expect_path_in_java_tools "java_tools/jdk_compiler.jar" java9
  expect_path_in_java_tools "java_tools/jdk_compiler.jar" java10
}

function test_java_tools_has_java_compiler() {
  expect_path_in_java_tools "java_tools/java_compiler.jar" java9
  expect_path_in_java_tools "java_tools/java_compiler.jar" java10
}

function test_java_tools_has_javac() {
  expect_path_in_java_tools "java_tools/javac-9+181-r4173-1.jar" java9
}

function test_java_tools_has_jarjar() {
  expect_path_in_java_tools "java_tools/jarjar_command_deploy.jar" java9
  expect_path_in_java_tools "java_tools/jarjar_command_deploy.jar" java10
}

function test_java_tools_has_Jacoco() {
  expect_path_in_java_tools "java_tools/JacocoCoverage_jarjar_deploy.jar" java9
  expect_path_in_java_tools "java_tools/JacocoCoverage_jarjar_deploy.jar" java10
}

function test_java_tools_has_GenClass() {
  expect_path_in_java_tools "java_tools/GenClass_deploy.jar" java9
  expect_path_in_java_tools "java_tools/GenClass_deploy.jar" java10
}

function test_java_tools_has_ExperimentalRunner() {
  expect_path_in_java_tools "java_tools/ExperimentalRunner_deploy.jar" java9
  expect_path_in_java_tools "java_tools/ExperimentalRunner_deploy.jar" java10
}

run_suite "Java tools archive tests"
