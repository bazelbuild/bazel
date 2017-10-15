#!/bin/bash
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

set -e

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_log_file_uses_single_line_formatter() {
  local client_log="$(bazel info output_base)/java.log"

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

run_suite "${PRODUCT_NAME} logging test"
