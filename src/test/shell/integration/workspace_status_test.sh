#!/usr/bin/env bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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
# workspace_status_test.sh: integration tests for bazel workspace status.
# This tests shared functionality between Bazel/Blaze workspace status actions,
# as opposed to ../bazel/bazel_workspace_status_test.sh, which is Bazel-only
# functionality.

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
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

add_to_bazelrc "build --package_path=%workspace%"

#### TESTS #############################################################

function test_workspace_status_with_stderr() {
  local ok="$TEST_TMPDIR/ok.sh"
  cat > "$ok" <<EOF
#!/usr/bin/env bash
echo "This is stderr" >&2
exit 0
EOF
  chmod +x "$ok"

  bazel build --stamp --workspace_status_command="$ok" >& "$TEST_log" \
      || fail "Build failed"
  # Tolerate Bazel/Blaze differences in action description.
  expect_log "INFO: From .*.txt:"
  expect_log "This is stderr"
}

function test_workspace_status_invalidation() {
  bazel build --stamp --workspace_status_command=/bin/false >& "$TEST_log" \
    && fail "build succeeded"
  # Tolerate Bazel/Blaze differences in failure message.
  expect_log "Failed to determine"
  expect_not_log IllegalStateException # regtest for #806095
  bazel build --stamp --workspace_status_command=/bin/true >& "$TEST_log" \
    || fail "build failed"
}

# Regression test for bug 4095015.
function test_false_and_verbose_failures() {
    mkdir -p x || fail "mkdir x failed"
    echo "cc_library(name='x')" >x/BUILD

    blaze build --workspace_status_command=/bin/false --verbose_failures //x \
        >& "$TEST_log" && fail "Expected build to fail".
    # Tolerate Bazel/Blaze differences in failure message.
    expect_log "Failed to determine"
    expect_not_log NullPointerException
    expect_not_log Crash
}

function test_that_script_is_run_from_workspace_directory() {
    mkdir -p x || fail "mkdir x failed"
    echo "cc_library(name='x')" >x/BUILD

    local pwdfile="$TEST_TMPDIR/pwdfile"
    echo "pwd >$pwdfile" >myscript && chmod +x myscript
    # We'll even run Bazel from a subdirectory.
    (cd x && bazel build --workspace_status_command=myscript //x >& "$TEST_log")
    local exit_code="$?"
    [[ "$exit_code" -eq 0 ]] || fail "Build failed"
    cat "$pwdfile" > "$TEST_log"
    expect_log "^$(pwd)\$"
}

function test_errmsg() {
  bazel build --workspace_status_command="$TEST_TMPDIR/wscmissing.sh" --stamp \
       &> $TEST_log && fail "build succeeded"
  expect_log "wscmissing.sh: No such file or directory\|wscmissing.sh: not found"
}

function test_embed_label_must_be_single_line() {
  bazel build --embed_label="$(echo -e 'abc\nxyz')" >& "$TEST_log" \
    && fail "Expected failure"
  expect_log "Value must not contain multiple lines"
}

run_suite "${PRODUCT_NAME} workspace status command tests"
