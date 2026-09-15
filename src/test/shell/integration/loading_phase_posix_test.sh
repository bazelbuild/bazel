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

function test_glob_control_chars() {
  local char escape raw
  for char in {1..31} 127; do
    local pkg="$FUNCNAME/char$char"
    mkdir -p $pkg
    echo "filegroup(name='t', srcs=glob(['*']))" > $pkg/BUILD
    printf -v escape \\%03o $char
    printf -v raw %b "$escape"
    touch "$pkg/$raw"
    bazel query "//$pkg:*" >& $TEST_log && fail "Expected failure"
    expect_log 'invalid label'
  done
}

function test_glob_utf8() {
  local -r pkg="$FUNCNAME"
  mkdir $pkg
  echo "filegroup(name='t', srcs=glob(['*']))" > $pkg/BUILD
  cd $pkg
  # This might print error messages for individual file names on systems like
  # macOS that use a file system that only permits correct UTF-8 strings as file
  # names. The errors can be ignored - we just test with whatever files the OS
  # allowed us to create.
  perl -CS -e 'for $i (160..0xd7ff) {print chr $i, $i%20?"":"\n"}' | xargs touch || true
  cd ..
  bazel query "//$pkg:*" >& $TEST_log || fail "Expected success"
}

function run_test_glob_with_io_error() {
  local option=$1
  local -r pkg="${FUNCNAME}"
  mkdir -p "$pkg" || fail "could not create \"$pkg\""

  mkdir -p $pkg/t/u
  touch $pkg/t/u/v

  echo "filegroup(name='t', srcs=glob(['u/*']))" > $pkg/t/BUILD
  chmod 000 $pkg/t/u

  bazel query "$option" "//$pkg/t:*" >& $TEST_log && fail "Expected failure"
  # TODO(katre): when error message from ErrorDeterminingRepositoryException is
  #  improved, add it here too.
  expect_log 'error globbing'

  chmod 755 $pkg/t/u
  bazel query "$option" "//$pkg/t:*" >& $TEST_log || fail "Expected success"
  expect_not_log 'error globbing'
  expect_log "//$pkg/t:u"
  expect_log "//$pkg/t:u/v"
}

function test_glob_with_io_error_nokeep_going() {
  run_test_glob_with_io_error "--nokeep_going"
}

function test_glob_with_io_error_keep_going() {
  run_test_glob_with_io_error "--keep_going"
}

function test_build_file_symlinks() {
  local -r pkg="${FUNCNAME}"
  mkdir -p "$pkg" || fail "could not create \"$pkg\""

  mkdir $pkg/b || fail "couldn't make $pkg/b"
  ln -s b $pkg/a || fail "couldn't link $pkg/a to b"

  bazel query $pkg/a:all >& $TEST_log && fail "Expected failure"
  expect_log "no such package '$pkg/a'"

  touch $pkg/b/BUILD
  bazel query $pkg/a:all >& $TEST_log || fail "Expected success"
  expect_log "Empty results"

  unlink $pkg/a || fail "couldn't unlink a"
  ln -s c $pkg/a
  bazel query $pkg/a:all >& $TEST_log && fail "Expected failure"
  expect_log "no such package '$pkg/a'"

  mkdir $pkg/c || fail "couldn't make c"
  ln -s foo $pkg/c/BUILD || fail "couldn't link $pkg/c/BUILD to foo"
  bazel query $pkg/a:all >& $TEST_log && fail "Expected failure"
  expect_log "no such package '$pkg/a'"

  touch $pkg/c/foo
  bazel query $pkg/a:all >& $TEST_log || fail "Expected success"
  expect_log "Empty results"
}

run_suite "Integration tests of ${PRODUCT_NAME} using loading/analysis phases."

