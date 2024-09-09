#!/bin/bash
#
# Copyright 2024 The Bazel Authors. All rights reserved.
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
# Test profiles.

# --- begin runfiles.bash initialization v2 ---
# Copy-pasted from the Bazel Bash runfiles library v2.
set -uo pipefail; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v2 ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_profile_management_build() {
  local uuid=7c859c8f-87fd-46d0-9e5c-669410812722
  local output_base="$(bazel info output_base)"
  bazel build --invocation_id="$uuid"
  cmp "${output_base}/command-${uuid}.profile.gz" "${output_base}/command.profile.gz" || fail "profile not linked"
}

function test_profile_management_info() {
  local uuid=e9d9df11-04a9-4f78-9f71-5a0153cb6f0c
  local output_base="$(bazel info output_base --invocation_id=${uuid} --generate_json_trace_profile)"
  ls -lh $output_base
  cmp "${output_base}/command-${uuid}.profile.gz" "${output_base}/command.profile.gz" || fail "profile not linked"
}

run_suite "integration tests for the builtin profiler"
