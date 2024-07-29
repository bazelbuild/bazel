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

function set_up() {
  local java_tools_rlocation=$(rlocation io_bazel/src/java_tools.zip)
  local java_tools_zip_file_url="file://${java_tools_rlocation}"
  if "$is_windows"; then
        java_tools_zip_file_url="file:///${java_tools_rlocation}"
  fi
  local java_tools_prebuilt_rlocation=$(rlocation io_bazel/src/java_tools_prebuilt.zip)
  local java_tools_prebuilt_zip_file_url="file://${java_tools_prebuilt_rlocation}"
  if "$is_windows"; then
        java_tools_prebuilt_zip_file_url="file:///${java_tools_prebuilt_rlocation}"
  fi
  cat >> MODULE.bazel <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "local_java_tools",
    urls = ["${java_tools_zip_file_url}"]
)
http_archive(
    name = "local_java_tools_prebuilt",
    urls = ["${java_tools_prebuilt_zip_file_url}"]
)
EOF
  # Dependencies of java_tools
  add_platforms "MODULE.bazel"
  add_rules_cc "MODULE.bazel"
  add_rules_proto "MODULE.bazel"
  add_rules_license "MODULE.bazel"
  add_abseil_cpp "MODULE.bazel"
}

function expect_path_in_java_tools() {
  path="$1"; shift

  count=$(zipinfo -1 $(rlocation io_bazel/src/java_tools.zip) | grep -c "$path")
  [[ "$count" -gt 0 ]] || fail "Path $path not found in java_tools.zip"
}

function expect_path_in_java_tools_prebuilt() {
  path="$1"; shift

  count=$(zipinfo -1 $(rlocation io_bazel/src/java_tools_prebuilt.zip) | grep -c "$path")
  [[ "$count" -gt 0 ]] || fail "Path $path not found in java_tools_prebuilt.zip"
}


function test_java_tools_has_ijar() {
  expect_path_in_java_tools "java_tools/ijar"
  expect_path_in_java_tools_prebuilt "java_tools/ijar"
}

function test_java_tools_has_ijar_binary() {
  expect_path_in_java_tools_prebuilt "java_tools/ijar/ijar"
}

function test_java_tools_has_zlib() {
  expect_path_in_java_tools "java_tools/zlib"
}

function test_java_tools_has_native_windows() {
  expect_path_in_java_tools "java_tools/src/main/native/windows"
}

function test_java_tools_has_cpp_util() {
  expect_path_in_java_tools "java_tools/src/main/cpp/util"
}

function test_java_tools_has_desugar_deps() {
  expect_path_in_java_tools "java_tools/src/main/protobuf/desugar_deps.proto"
}

function test_java_tools_has_singlejar() {
  expect_path_in_java_tools "java_tools/src/tools/singlejar"
  expect_path_in_java_tools_prebuilt "java_tools/src/tools/singlejar"
}

function test_java_tools_has_singlejar_local() {
  expect_path_in_java_tools_prebuilt "java_tools/src/tools/singlejar/singlejar_local"
}

function test_java_tools_has_VanillaJavaBuilder() {
  expect_path_in_java_tools "java_tools/VanillaJavaBuilder_deploy.jar"
}

function test_java_tools_has_JavaBuilder() {
  expect_path_in_java_tools "java_tools/JavaBuilder_deploy.jar"
}

function test_java_tools_has_turbine_direct() {
  expect_path_in_java_tools "java_tools/turbine_direct_binary_deploy.jar"
  expect_path_in_java_tools_prebuilt "java_tools/turbine_direct_graal"
}

function test_java_tools_has_one_version() {
  expect_path_in_java_tools "java_tools/src/tools/one_version"
  expect_path_in_java_tools_prebuilt "java_tools/src/tools/one_version"
}

function test_java_tools_has_Runner() {
  expect_path_in_java_tools "java_tools/Runner_deploy.jar"
}

function test_java_tools_has_Jacoco() {
  expect_path_in_java_tools "java_tools/JacocoCoverage_jarjar_deploy.jar"
}

function test_java_tools_has_GenClass() {
  expect_path_in_java_tools "java_tools/GenClass_deploy.jar"
}

function test_java_tools_has_BUILD() {
  expect_path_in_java_tools "BUILD"
}

function test_java_tools_has_jacocoagent() {
  expect_path_in_java_tools "java_tools/third_party/java/jacoco/jacocoagent-.*.jar"
  expect_path_in_java_tools "java_tools/third_party/java/jacoco/org.jacoco.agent-.*.jar"
  expect_path_in_java_tools "java_tools/third_party/java/jacoco/org.jacoco.core-.*.jar"
  expect_path_in_java_tools "java_tools/third_party/java/jacoco/org.jacoco.report-.*.jar"
  expect_path_in_java_tools "java_tools/third_party/java/jacoco/asm-tree-.*.jar"
  expect_path_in_java_tools "java_tools/third_party/java/jacoco/asm-commons-.*.jar"
  expect_path_in_java_tools "java_tools/third_party/java/jacoco/asm-.*.jar"
  expect_path_in_java_tools "java_tools/third_party/java/jacoco/LICENSE"
}

function test_java_tools_has_proguard() {
  expect_path_in_java_tools "java_tools/third_party/java/proguard/proguard.jar"
  expect_path_in_java_tools "java_tools/third_party/java/proguard/GPL.md"
}

function test_java_tools_toolchain_builds() {
  bazel build @bazel_tools//tools/jdk:toolchain || fail "toolchain failed to build"
}

function test_java_tools_singlejar_builds() {
  bazel build @local_java_tools//:singlejar_cc_bin || fail "singlejar failed to build"
}

function test_java_tools_singlejar_builds_with_layering_check() {
  if [[ ! $(type -P clang) ]]; then
    return
  fi

  bazel build --repo_env=CC=clang --features=layering_check \
    @local_java_tools//:singlejar_cc_bin || fail "singlejar failed to build with layering check"
}

function test_java_tools_ijar_builds() {
  bazel build @local_java_tools//:ijar_cc_binary || fail "ijar failed to build"
}

function test_java_tools_ijar_builds_with_layering_check() {
  if [[ ! $(type -P clang) ]]; then
    return
  fi

  bazel build --repo_env=CC=clang --features=layering_check \
    @local_java_tools//:ijar_cc_binary || fail "ijar failed to build with layering check"
}

function test_java_tools_ijar_builds() {
  bazel build @local_java_tools//:one_version_cc_bin || fail "one_version failed to build"
}

function test_java_tools_one_version_builds_with_layering_check() {
  if [[ ! $(type -P clang) ]]; then
    return
  fi

  bazel build --repo_env=CC=clang --features=layering_check \
    @local_java_tools//:one_version_cc_bin || fail "one_version failed to build with layering check"
}

run_suite "Java tools archive tests"
