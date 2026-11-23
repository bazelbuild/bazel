#!/usr/bin/env bash
#
# Copyright 2015 The Bazel Authors. All rights reserved.
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

# --- begin runfiles.bash initialization v3 ---
# Copy-pasted from the Bazel Bash runfiles library v3.
set -uo pipefail; set +e; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v3 ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

FUTIMENS="$(rlocation "io_bazel/src/test/shell/integration/futimens")"

# Writes a script file that, when executed, prints the given output.
function write_input_file() {
  # Relative to the current working directory.
  local file_path=$1
  local desired_output=$2
  echo -e "#!/bin/bash\necho \"${desired_output}\"" > "${file_path}"
}

# Tests that bazel actually has millisecond resolution on unix file mtimes.
function test_millisecond_resolution() {
  mkdir a || fail "Failed to make directory"
  cat > a/BUILD << 'EOF'
genrule(
    name = "foo",
    srcs = ["foo.in"],
    outs = ["foo.sh"],
    executable = 1,
    output_to_bindir = 1,
    cmd = "cat $< > $@ && chmod +x $@",
)
EOF
  write_input_file "a/foo.in" "version1"
  "${FUTIMENS}" --file=a/foo.in --seconds=1 --nanoseconds=1000000 \
    || fail "Expected success"
  bazel build "a:foo" || fail "Expected success"
  "${PRODUCT_NAME}-bin/a/foo.sh" >& "$TEST_log"
  expect_log "version1"
  write_input_file "a/foo.in" "version2"
  "${FUTIMENS}" --file=a/foo.in --seconds=1 --nanoseconds=2000000 \
    || fail "Expected success"
  bazel build "a:foo" || fail "Expected success"
  "${PRODUCT_NAME}-bin/a/foo.sh" >& "$TEST_log"
  expect_log "version2"
}

run_suite "Tests for file timestamp granularity"
