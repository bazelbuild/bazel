#!/bin/bash
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
#
# loading_phase_tests.sh: miscellaneous integration tests of Bazel,
# that use only the loading or analysis phases.
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

output_base=$TEST_TMPDIR/out
TEST_stderr=$(dirname $TEST_log)/stderr

#### HELPER FUNCTIONS ##################################################

function set_up() {
    cd ${WORKSPACE_DIR}
}

function tear_down() {
    bazel shutdown
}

#### TESTS #############################################################

function test_glob_with_io_error() {
  local -r pkg="${FUNCNAME}"
  mkdir -p "$pkg" && cd "$pkg" || fail "could not create and cd \"$pkg\""
  touch WORKSPACE

  mkdir -p t/u
  touch t/u/v

  echo "filegroup(name='t', srcs=glob(['u/*']))" > t/BUILD
  chmod 000 t/u

  bazel query '//t:*' >& $TEST_log && fail "Expected failure"
  expect_log 'error globbing.*Permission denied'

  chmod 755 t/u
  bazel query '//t:*' >& $TEST_log || fail "Expected success"
  expect_not_log 'error globbing.*Permission denied'
  expect_log '//t:u'
  expect_log '//t:u/v'
}

function test_build_file_symlinks() {
  local -r pkg="${FUNCNAME}"
  mkdir -p "$pkg" && cd "$pkg" || fail "could not create and cd \"$pkg\""
  touch WORKSPACE

  mkdir b || fail "couldn't make b"
  ln -s b a || fail "couldn't link a to b"

  bazel query a:all >& $TEST_log && fail "Expected failure"
  expect_log "no such package 'a'"

  touch b/BUILD
  bazel query a:all >& $TEST_log || fail "Expected success"
  expect_log "Empty results"

  unlink a || fail "couldn't unlink a"
  ln -s c a
  bazel query a:all >& $TEST_log && fail "Expected failure"
  expect_log "no such package 'a'"

  mkdir c || fail "couldn't make c"
  ln -s foo c/BUILD || "couldn't link c/BUILD to c/foo"
  bazel query a:all >& $TEST_log && fail "Expected failure"
  expect_log "no such package 'a'"

  touch c/foo
  bazel query a:all >& $TEST_log || fail "Expected success"
  expect_log "Empty results"
}

run_suite "Integration tests of ${PRODUCT_NAME} using loading/analysis phases."

