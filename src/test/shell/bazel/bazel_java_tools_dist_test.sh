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

# A test that checks the content of the java_tools sources zip that is distributed
# along with the java_tools release.

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

function expect_path_in_java_tools() {
  path="$1"; shift

  (zipinfo -1 $(rlocation io_bazel/src/java_tools_dist.zip) \
    | grep -c "$path") >& ${TEST_log} || fail "Path $path not found in java_tools_dist.zip"
}

function test_java_tools_has_ijar() {
  expect_path_in_java_tools "third_party/ijar"
}

function test_java_tools_has_zlib() {
  expect_path_in_java_tools "third_party/zlib"
}

function test_java_tools_has_native_windows() {
  expect_path_in_java_tools "src/main/native/windows"
}

function test_java_tools_has_cpp_util() {
  expect_path_in_java_tools "src/main/cpp/util"
}

function test_java_tools_has_desugar_deps() {
  expect_path_in_java_tools "src/main/protobuf/desugar_deps.proto"
}

function test_java_tools_has_singlejar() {
  expect_path_in_java_tools "src/java_tools/singlejar"
  expect_path_in_java_tools "src/java_tools/singlejar/java/com/google/devtools/build/singlejar"
  expect_path_in_java_tools "src/java_tools/singlejar/java/com/google/devtools/build/zip"
}

function test_java_tools_has_native_singlejar() {
  expect_path_in_java_tools "src/tools/singlejar"
}

function test_java_tools_has_buildjar() {
  expect_path_in_java_tools "src/java_tools/buildjar/java/com/google/devtools/build/buildjar"
  expect_path_in_java_tools "src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac"
  expect_path_in_java_tools "src/java_tools/buildjar/java/com/google/devtools/build/buildjar/genclass"
  expect_path_in_java_tools "src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper"
}

function test_java_tools_has_turbine() {
  expect_path_in_java_tools "src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/BUILD"
}

function test_java_tools_has_junitrunner() {
  expect_path_in_java_tools "src/java_tools/junitrunner/BUILD"
  expect_path_in_java_tools "src/java_tools/junitrunner/java/com/google/testing/coverage"
  expect_path_in_java_tools "src/java_tools/junitrunner/java/com/google/testing/junit/junit4"
  expect_path_in_java_tools "src/java_tools/junitrunner/java/com/google/testing/junit/runner"
  expect_path_in_java_tools "src/java_tools/junitrunner/java/com/google/testing/junit/runner/junit4"
  expect_path_in_java_tools "src/java_tools/junitrunner/java/com/google/testing/junit/runner/internal"
  expect_path_in_java_tools "src/java_tools/junitrunner/java/com/google/testing/junit/runner/model"
  expect_path_in_java_tools "src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding"
  expect_path_in_java_tools "src/java_tools/junitrunner/java/com/google/testing/junit/runner/util"
}

function test_java_tools_has_jacocoagent() {
  expect_path_in_java_tools "third_party/java/jacoco/org.jacoco.agent-.*-sources.jar"
  expect_path_in_java_tools "third_party/java/jacoco/org.jacoco.core-.*-sources.jar"
  expect_path_in_java_tools "third_party/java/jacoco/org.jacoco.report-.*-sources.jar"
  expect_path_in_java_tools "third_party/asm/asm-analysis-.*-sources.jar"
  expect_path_in_java_tools "third_party/asm/asm-commons-.*-sources.jar"
  expect_path_in_java_tools "third_party/asm/asm-.*-sources.jar"
}

function test_java_tools_has_proguard() {
  expect_path_in_java_tools "third_party/java/proguard"
  expect_path_in_java_tools "third_party/java/proguard/proguard6.2.2"
  expect_path_in_java_tools "third_party/java/proguard/proguard6.2.2/bin"
  expect_path_in_java_tools "third_party/java/proguard/proguard6.2.2/buildscripts"
  expect_path_in_java_tools "third_party/java/proguard/proguard6.2.2/src"
  expect_path_in_java_tools "third_party/java/proguard/proguard6.2.2/src/proguard"
}

run_suite "Java tools archive tests"
