#!/bin/bash
#
# Copyright 2019 The Bazel Authors. All rights reserved.
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

# `uname` returns the current platform, e.g "MSYS_NT-10.0" or "Linux".
# `tr` converts all upper case letters to lower case.
# `case` matches the result if the `uname | tr` expression to string prefixes
# that use the same wildcards as names do in Bash, i.e. "msys*" matches strings
# starting with "msys", and "*" matches everything (it's the default case).
case "$(uname -s | tr [:upper:] [:lower:])" in
msys*)
  # As of 2019-04-08, Bazel on Windows only supports MSYS Bash.
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  # Disable MSYS path conversion that converts path-looking command arguments to
  # Windows paths (even if they arguments are not in fact paths).
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
  declare -r EXE_EXT=".exe"
else
  declare -r EXE_EXT=""
fi

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

# Writes a python file that prints all arguments (except argv[0]).
#
# Args:
# $1: directory (package path) where the file will be written
function create_py_file_that_prints_args() {
  local -r pkg="$1"; shift
  mkdir -p "$pkg" || fail "mkdir -p $pkg"
  cat >"$pkg/a.py" <<'eof'
from __future__ import print_function
import sys
for i in range(1, len(sys.argv)):
    print("arg%d=(%s)" % (i, sys.argv[i]))
eof
}

# Writes a BUILD file for a py_binary with an untokenizable "args" entry.
#
# Args:
# $1: directory (package path) where the file will be written
function create_build_file_for_untokenizable_args() {
  local -r pkg="$1"; shift
  mkdir -p "$pkg" || fail "mkdir -p $pkg"
  cat >"$pkg/BUILD" <<'eof'
py_binary(
    name = "cannot_tokenize",
    srcs = ["a.py"],
    main = "a.py",
    args = ["'abc"],
)
eof
}

# Writes a BUILD file for a py_binary with many different "args" entries.
#
# Use this together with assert_*_output_of_the_program_with_many_args().
#
# Args:
# $1: directory (package path) where the file will be written
function create_build_file_with_many_args() {
  local -r pkg="$1"; shift
  mkdir -p "$pkg" || fail "mkdir -p $pkg"
  cat >"$pkg/BUILD" <<'eof'
py_binary(
    name = "x",
    srcs = ["a.py"],
    main = "a.py",
    args = [
        "''",
        "' '",
        "'\"'",
        "'\"\\'",
        "'\\'",
        "'\\\"'",
        "'with space'",
        "'with^caret'",
        "'space ^caret'",
        "'caret^ space'",
        "'with\"quote'",
        "'with\\backslash'",
        "'one\\ backslash and \\space'",
        "'two\\\\backslashes'",
        "'two\\\\ backslashes \\\\and space'",
        "'one\\\"x'",
        "'two\\\\\"x'",
        "'a \\ b'",
        "'a \\\" b'",
        "'A'",
        "'\"a\"'",
        "'B C'",
        "'\"b c\"'",
        "'D\"E'",
        "'\"d\"e\"'",
        "'C:\\F G'",
        "'\"C:\\f g\"'",
        "'C:\\H\"I'",
        "'\"C:\\h\"i\"'",
        "'C:\\J\\\"K'",
        "'\"C:\\j\\\"k\"'",
        "'C:\\L M '",
        "'\"C:\\l m \"'",
        "'C:\\N O\\'",
        "'\"C:\\n o\\\"'",
        "'C:\\P Q\\ '",
        "'\"C:\\p q\\ \"'",
        "'C:\\R\\S\\'",
        "'C:\\R x\\S\\'",
        "'\"C:\\r\\s\\\"'",
        "'\"C:\\r x\\s\\\"'",
        "'C:\\T U\\W\\'",
        "'\"C:\\t u\\w\\\"'",
        "a",
        r"b\\\"",
        "c",
    ],
)
eof
}

# Asserts that the $TEST_log contains bad py_binary args.
#
# This assertion guards (and demonstrates) the status quo.
#
# See create_build_file_with_many_args() and create_py_file_that_prints_args().
function assert_bad_output_of_the_program_with_many_args() {
  expect_log 'arg1=()'
  expect_log 'arg2=( )'
  expect_log 'arg3=(")'
  expect_log 'arg4=("\\)'
  # arg5 and arg6 already stumble. No need to assert more.
  expect_log 'arg5=(\\)'
  expect_log 'arg6=(\\ with)'
  # To illustrate the bug again, these args match those in the bug report:
  # https://github.com/bazelbuild/bazel/issues/7958
  expect_log 'arg40=(a)'
  expect_log 'arg41=(b\\ c)'
}

# Asserts that the $TEST_log contains all py_binary args as argN=(VALUE).
#
# See create_build_file_with_many_args() and create_py_file_that_prints_args().
function assert_good_output_of_the_program_with_many_args() {
  expect_log 'arg1=()'
  expect_log 'arg2=( )'
  expect_log 'arg3=(")'
  expect_log 'arg4=("\\)'
  expect_log 'arg5=(\\)'
  expect_log 'arg6=(\\")'
  expect_log 'arg7=(with space)'
  expect_log 'arg8=(with^caret)'
  expect_log 'arg9=(space ^caret)'
  expect_log 'arg10=(caret^ space)'
  expect_log 'arg11=(with"quote)'
  expect_log 'arg12=(with\\backslash)'
  expect_log 'arg13=(one\\ backslash and \\space)'
  expect_log 'arg14=(two\\\\backslashes)'
  expect_log 'arg15=(two\\\\ backslashes \\\\and space)'
  expect_log 'arg16=(one\\"x)'
  expect_log 'arg17=(two\\\\"x)'
  expect_log 'arg18=(a \\ b)'
  expect_log 'arg19=(a \\" b)'
  expect_log 'arg20=(A)'
  expect_log 'arg21=("a")'
  expect_log 'arg22=(B C)'
  expect_log 'arg23=("b c")'
  expect_log 'arg24=(D"E)'
  expect_log 'arg25=("d"e")'
  expect_log 'arg26=(C:\\F G)'
  expect_log 'arg27=("C:\\f g")'
  expect_log 'arg28=(C:\\H"I)'
  expect_log 'arg29=("C:\\h"i")'
  expect_log 'arg30=(C:\\J\\"K)'
  expect_log 'arg31=("C:\\j\\"k")'
  expect_log 'arg32=(C:\\L M )'
  expect_log 'arg33=("C:\\l m ")'
  expect_log 'arg34=(C:\\N O\\)'
  expect_log 'arg35=("C:\\n o\\")'
  expect_log 'arg36=(C:\\P Q\\ )'
  expect_log 'arg37=("C:\\p q\\ ")'
  expect_log 'arg38=(C:\\R\\S\\)'
  expect_log 'arg39=(C:\\R x\\S\\)'
  expect_log 'arg40=("C:\\r\\s\\")'
  expect_log 'arg41=("C:\\r x\\s\\")'
  expect_log 'arg42=(C:\\T U\\W\\)'
  expect_log 'arg43=("C:\\t u\\w\\")'
  expect_log 'arg44=(a)'
  expect_log 'arg45=(b\\")'
  expect_log 'arg46=(c)'
}

# ----------------------------------------------------------------------
# TESTS
# ----------------------------------------------------------------------

function test_args_escaping_disabled_on_windows() {
  local -r pkg="${FUNCNAME[0]}"  # unique package name for this test

  create_py_file_that_prints_args "$pkg"
  create_build_file_with_many_args "$pkg"

  bazel run --verbose_failures --noincompatible_windows_escape_python_args \
    "${pkg}:x" &>"$TEST_log" || fail "expected success"
  if "$is_windows"; then
    # On Windows, the target runs but prints bad output.
    assert_bad_output_of_the_program_with_many_args
  else
    # On other platforms, the program runs fine and prints correct output.
    assert_good_output_of_the_program_with_many_args
  fi
}

function test_args_escaping() {
  local -r pkg="${FUNCNAME[0]}"  # unique package name for this test

  create_py_file_that_prints_args "$pkg"
  create_build_file_with_many_args "$pkg"

  # On all platforms, the target prints good output.
  bazel run --verbose_failures --incompatible_windows_escape_python_args \
    "${pkg}:x" &>"$TEST_log" || fail "expected success"
  assert_good_output_of_the_program_with_many_args
}

function test_untokenizable_args_when_escaping_is_disabled() {
  local -r pkg="${FUNCNAME[0]}"  # unique package name for this test

  create_py_file_that_prints_args "$pkg"
  create_build_file_for_untokenizable_args "$pkg"

  # On all platforms, Bazel can build the target.
  if bazel build --verbose_failures --noincompatible_windows_escape_python_args \
      "${pkg}:cannot_tokenize" 2>"$TEST_log"; then
    fail "expected failure"
  fi
  expect_log "unterminated quotation"
}

function test_untokenizable_args_when_escaping_is_enabled() {
  local -r pkg="${FUNCNAME[0]}"  # unique package name for this test

  create_py_file_that_prints_args "$pkg"
  create_build_file_for_untokenizable_args "$pkg"

  local -r flag="--incompatible_windows_escape_python_args"
  bazel run --verbose_failures "$flag" "${pkg}:cannot_tokenize" \
    2>"$TEST_log" && fail "expected failure" || true
  expect_log "ERROR:.*in args attribute of py_binary rule.*unterminated quotation"
}

run_suite "Tests about how Bazel passes py_binary.args to the binary"
