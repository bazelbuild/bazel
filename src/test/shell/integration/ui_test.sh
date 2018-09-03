#!/bin/bash
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
# An end-to-end test that Bazel's experimental UI produces reasonable output.

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

if "$is_windows"; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

#### SETUP #############################################################

function create_pkg() {
  local -r pkg=$1
  mkdir -p $pkg
  cat > $pkg/true.sh <<EOF
#!/bin/sh
exit 0
EOF
  chmod 755 $pkg/true.sh
  cat > $pkg/BUILD <<EOF
sh_test(
  name = "true",
  srcs = ["true.sh"],
)
EOF
}

#### TESTS #############################################################

function test_basic_progress() {
  local -r pkg=$FUNCNAME
  create_pkg $pkg
  bazel test --curses=yes --color=yes $pkg:true 2>$TEST_log || fail "bazel test failed"
  # some progress indicator is shown
  expect_log '\[[0-9,]* / [0-9,]*\]'
  # something is written in green
  expect_log $'\x1b\[32m'
  # curses are used to delete at least one line
  expect_log $'\x1b\[1A\x1b\[K'
}

function test_line_wrapping() {
  local -r pkg=$FUNCNAME
  create_pkg $pkg
  bazel test --curses=yes --color=yes --terminal_columns=5 $pkg:true 2>$TEST_log || fail "bazel test failed"
  # curses are used to delete at least one line
  expect_log $'\x1b\[1A\x1b\[K'
  # something is written in green
  expect_log $'\x1b\[32m'
  # lines are wrapped, hence at least one line should end with backslash
  expect_log '\\'$'\r''$\|\\$'
}

function test_noline_wrapping_color_nocurses() {
  local -r pkg=$FUNCNAME
  create_pkg $pkg
  bazel test --curses=no --color=yes --terminal_columns=5 $pkg:true 2>$TEST_log || fail "bazel test failed"
  # something is written in green
  expect_log $'\x1b\[32m'
  # no lines are deleted
  expect_not_log $'\x1b\[K'
  # as no line wrapping occurs, no backlsash should be before a carriage return or at a line ending
  expect_not_log '\\'$'\r'
  expect_not_log '\\$'
}

run_suite "Basic integration tests for the standard UI"
