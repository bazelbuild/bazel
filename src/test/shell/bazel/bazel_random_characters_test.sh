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
#
# Tests the examples provided in Bazel
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

# Disable MSYS path conversions, so that
# "bazel build //foo" won't become "bazel build /foo", nor will
# "bazel build foo/bar" become "bazel build foo\bar".
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"

function basic_glob_scenario_test_template() {
  add_rules_java MODULE.bazel
  local chars="$1"
  local pkg="pkg${chars}"
  echo "chars = ${chars}, pkg = ${pkg}"
  mkdir -p "${pkg}/resources"
  cat >"${pkg}/BUILD" <<EOF
load("@rules_java//java:java_library.bzl", "java_library")

java_library(name = 'main',
    resources = glob(["resources/**"]),
    srcs = ['Main.java'])
EOF

  for i in $(seq 1 10); do
    cat >"${pkg}/resources/file${chars}$i" <<EOF
file${chars}$i
EOF
  done

  cat >"$pkg/Main.java" <<'EOF'
public class Main {
  public static void main(String[] args) {
    System.out.println("Hello, World!");
  }
}
EOF

  bazel build "//${pkg}:main" &>"${TEST_log}" \
      || fail "Failed to build java target"

  nb_files="$(unzip -l "bazel-bin/${pkg}/libmain.jar" \
      | grep -F "file${chars}" | tee "${TEST_log}" | wc -l | xargs echo)"
  [ "10" = "${nb_files}" ] || fail "Expected 10 files, got ${nb_files}"
}

function test_space_dollar_and_parentheses() {
  basic_glob_scenario_test_template '$( )'
}

run_suite "Integration tests for handling of special characters"

