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
# execution_phase_tests.sh: miscellaneous integration tests of Bazel for
# behaviors that affect the execution phase.
#

# --- begin runfiles.bash initialization ---
set -euo pipefail
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

function set_up() {
  cd ${WORKSPACE_DIR}
  mkdir -p "foo"
}

#### TESTS #############################################################
function test_flag_enabled() {
  cat > foo/BUILD <<EOF
cc_binary(
    name = "foo",
    srcs = [
        "foo.cc",
    ],
)
cc_binary(
    name = "bar",
    srcs = [
        "bar.cc",
    ],
)
EOF
  cat > foo/foo.cc <<EOF
int main(void) {
  return 0;
}
EOF
  cp foo/foo.cc foo/bar.cc

  bazel build --experimental_merged_skyframe_analysis_execution //foo:all &> "$TEST_log" || fail "Expected success"
}

function test_failed_builds() {
  cat > foo/BUILD <<EOF
cc_binary(
    name = "execution_failure",
    srcs = [
        "foo.cc"
    ],
    deps = [
        ":bar",
    ]
)

cc_library(
    name = "bar",
    srcs = [
        "bar.cc",
        "missing.a",
    ],
)

cc_binary(
    name = "analysis_failure",
    srcs = [
        "foo.cc"
    ],
    deps = [
        ":bar1",
    ]
)
EOF
  touch foo/foo.cc
  touch foo/bar.cc

  bazel build --experimental_merged_skyframe_analysis_execution //foo:execution_failure &> "$TEST_log" && fail "Expected failure"
  exit_code="$?"
  [[ "$exit_code" -eq 1 ]] || fail "Unexpected exit code: $exit_code"
  expect_log "missing input file '//foo:missing.a'"

  bazel build --experimental_merged_skyframe_analysis_execution //foo:analysis_failure &> "$TEST_log" && fail "Expected failure"
  exit_code="$?"
  [[ "$exit_code" -eq 1 ]] || fail "Unexpected exit code: $exit_code"
  expect_log "Analysis of target '//foo:analysis_failure' failed"

  bazel build --nokeep_going --experimental_merged_skyframe_analysis_execution //foo:analysis_failure //foo:execution_failure &> "$TEST_log" && fail "Expected failure"
  exit_code="$?"
  [[ "$exit_code" -eq 1 ]] || fail "Unexpected exit code: $exit_code"
  # With --nokeep_going, technically nothing can be said about the message: whichever target fails first would abort the build.


  bazel build --keep_going --experimental_merged_skyframe_analysis_execution //foo:analysis_failure //foo:execution_failure &> "$TEST_log" && fail "Expected failure"
  exit_code="$?"
  [[ "$exit_code" -eq 1 ]] || fail "Unexpected exit code: $exit_code"
  expect_log "missing input file '//foo:missing.a'"
  expect_log "Analysis of target '//foo:analysis_failure' failed"
}

run_suite "Integration tests of ${PRODUCT_NAME} with merged Analysis & Execution phases of Skyframe."
