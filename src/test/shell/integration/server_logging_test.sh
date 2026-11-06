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
# Test of Bazel's java logging.

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

#### TESTS #############################################################

function test_log_file_uses_single_line_formatter() {
  local client_log
  client_log="$(bazel info server_log)" || fail "bazel info failed"

  # Construct a regular expression to match a sample message in the log using
  # the single-line format.  More specifically, we expect the log entry's
  # context (timestamp, severity and class) to appear in the same line as the
  # actual log message.
  local timestamp_re='[0-9]{6} [0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{3}'
  local sample_re="^${timestamp_re}:I .*BlazeVersionInfo"

  if ! grep -E "${sample_re}" "${client_log}"; then
    # Dump some client log lines on stdout for debugging of this test failure.
    head -n 10 "${client_log}"
    fail "invalid format in java.log; see output for sample lines"
  fi
}

function test_server_log_redacts_credentials_and_tokens() {
  local bes_token="mysupersecretbestoken12345"
  local creds_token="MY_SECRET_CREDENTIALS_FILE.json"
  local policy_secret="secret_token_from_policy"

  # Shut down any previous server so server startup logging in BlazeRuntime is triggered.
  bazel shutdown >& "$TEST_log" || fail "bazel shutdown failed"

  bazel \
    --invocation_policy="flag_policies:{flag_name:'bes_header' set_value:{behavior:FINAL_VALUE_IGNORE_OVERRIDES flag_value:'Authorization=Bearer ${policy_secret}'}}" \
    info \
    --bes_header="Authorization=Bearer ${bes_token}" \
    --google_credentials="${creds_token}" \
    >& "$TEST_log" || fail "bazel info failed"

  local server_log
  server_log="$(bazel info server_log)" || fail "bazel info server_log failed"

  assert_not_contains "${bes_token}" "${server_log}"
  assert_not_contains "${creds_token}" "${server_log}"
  assert_not_contains "${policy_secret}" "${server_log}"

  assert_contains "Setting value for option '--bes_header' from invocation policy to '<REDACTED>'" "${server_log}"
  assert_contains "--google_credentials=<REDACTED>" "${server_log}"
}

function test_batch_mode_server_log_redacts_invocation_policy() {
  local policy_secret="secret_token_from_batch_policy"

  bazel shutdown >& "$TEST_log" || fail "bazel shutdown failed"

  local server_log
  server_log="$(bazel \
    --batch \
    --invocation_policy="flag_policies:{flag_name:'bes_header' set_value:{behavior:FINAL_VALUE_IGNORE_OVERRIDES flag_value:'Authorization=Bearer ${policy_secret}'}}" \
    info server_log)" || fail "bazel info server_log failed"

  assert_not_contains "${policy_secret}" "${server_log}"
  assert_contains "Running Bazel in batch mode" "${server_log}"
  assert_contains "--invocation_policy=<REDACTED>" "${server_log}"
}

run_suite "${PRODUCT_NAME} logging test"
