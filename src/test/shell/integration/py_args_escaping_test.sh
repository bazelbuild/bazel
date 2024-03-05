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
# $1: directory (workspace and package path) where the file will be written
function create_py_file_that_prints_args() {
  local -r ws="$1"; shift
  mkdir -p "$ws" || fail "mkdir -p $ws"
  cat >"$ws/a.py" <<'eof'
from __future__ import print_function
import sys
for i in range(1, len(sys.argv)):
    print("arg%d=(%s)" % (i, sys.argv[i]))
eof
}

# Writes a BUILD file for a py_binary with an untokenizable "args" entry.
#
# Args:
# $1: directory (workspace and package path) where the file will be written
function create_build_file_for_untokenizable_args() {
  local -r ws="$1"; shift
  mkdir -p "$ws" || fail "mkdir -p $ws"
  cat >"$ws/BUILD" <<'eof'
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
  local -r ws="$1"; shift
  mkdir -p "$ws" || fail "mkdir -p $ws"
  cat >"$ws/BUILD" <<'eof'
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

function test_args_escaping() {
  local -r ws="$TEST_TMPDIR/${FUNCNAME[0]}"  # unique workspace for this test
  mkdir -p "$ws"
  create_workspace_with_default_repos "$ws/WORKSPACE"

  create_py_file_that_prints_args "$ws"
  create_build_file_with_many_args "$ws"

  ( cd "$ws"
    bazel run --verbose_failures :x &>"$TEST_log" \
      || fail "expected success"
  )
  assert_good_output_of_the_program_with_many_args
  rm "$TEST_log"

  ( cd "$ws"
    bazel run --verbose_failures :x &>"$TEST_log" || fail "expected success"
  )
  assert_good_output_of_the_program_with_many_args
  rm "$TEST_log"
}

function test_untokenizable_args() {
  local -r ws="$TEST_TMPDIR/${FUNCNAME[0]}"  # unique workspace for this test
  mkdir -p "$ws"
  create_workspace_with_default_repos "$ws/WORKSPACE"

  create_py_file_that_prints_args "$ws"
  create_build_file_for_untokenizable_args "$ws"

  ( cd "$ws"
    bazel run --verbose_failures :cannot_tokenize \
      2>"$TEST_log" && fail "expected failure" || true
  )
  expect_log "ERROR:.*in args attribute of py_binary rule.*unterminated quotation"
}

function test_host_config() {
  local -r ws="$TEST_TMPDIR/${FUNCNAME[0]}"  # unique workspace for this test
  mkdir -p "$ws"
  create_workspace_with_default_repos "$ws/WORKSPACE"

  cat >"$ws/BUILD" <<'eof'
load("//:rule.bzl", "run_host_configured")

run_host_configured(
    name = "x",
    tool = ":print_args",
    out = "x.out",
)

py_binary(
    name = "print_args",
    srcs = ["print_args.py"],
)
eof

  cat >"$ws/print_args.py" <<'eof'
import sys
with open(sys.argv[1], "wt") as f:
    for i in range(2, len(sys.argv)):
        f.write("arg%d=(%s)" % (i, sys.argv[i]))
eof

  cat >"$ws/rule.bzl" <<'eof'
def _impl(ctx):
    ctx.actions.run(
        outputs = [ctx.outputs.out],
        executable = ctx.executable.tool,
        arguments = [ctx.outputs.out.path, "a", "", "\"b \\\"c", "z"],
        use_default_shell_env = True,
    )
    return DefaultInfo(files = depset(direct = [ctx.outputs.out]))

run_host_configured = rule(
    implementation = _impl,
    attrs = {
        "tool": attr.label(executable = True, cfg = "exec"),
        "out": attr.output(),
    },
)
eof

  ( cd "$ws"
    bazel build --verbose_failures :x &>"$TEST_log" || fail "expected success"
    cat bazel-bin/x.out >> "$TEST_log"
  )
  # This output is right.
  expect_log 'arg2=(a)'
  expect_log 'arg3=()'
  expect_log 'arg4=("b \\"c)'
  expect_log 'arg5=(z)'
}

run_suite "Tests about how Bazel passes py_binary.args to the binary"
