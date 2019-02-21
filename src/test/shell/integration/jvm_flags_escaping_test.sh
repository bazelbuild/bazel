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
  # As of 2019-02-20, Bazel on Windows only supports MSYS Bash.
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

function create_java_file_that_prints_jvm_args() {
  local -r pkg="$1"
  mkdir -p "$pkg"
  cat >"$pkg/A.java" <<'eof'
public class A {
  public static void main(String[] args) {
    for (int i = 0; ; ++i) {
      String value = System.getProperty("arg" + i);
      if (value == null) {
        break;
      } else {
        System.out.printf("arg%d=(%s)%n", i, value);
      }
    }
  }
}
eof
}

function create_build_file_for_untokenizable_flag() {
  local -r pkg="$1"
  mkdir -p "$pkg"
  cat >"$pkg/BUILD" <<'eof'
java_binary(
    name = "cannot_tokenize",
    srcs = ["A.java"],
    main_class = "A",
    jvm_flags = ["-Darg0='abc"],
)
eof
}

function create_build_file_with_many_jvm_flags() {
  local -r pkg="$1"
  mkdir -p "$pkg"
  cat >"$pkg/BUILD" <<'eof'
java_binary(
    name = "x",
    srcs = ["A.java"],
    main_class = "A",
    jvm_flags = [
        "-Darg0=''",
        "-Darg1=' '",
        "-Darg2='\"'",
        "-Darg3='\"\\'",
        "-Darg4='\\'",
        "-Darg5='\\\"'",
        "-Darg6='with space'",
        "-Darg7='with^caret'",
        "-Darg8='space ^caret'",
        "-Darg9='caret^ space'",
        "-Darg10='with\"quote'",
        "-Darg11='with\\backslash'",
        "-Darg12='one\\ backslash and \\space'",
        "-Darg13='two\\\\backslashes'",
        "-Darg14='two\\\\ backslashes \\\\and space'",
        "-Darg15='one\\\"x'",
        "-Darg16='two\\\\\"x'",
        "-Darg17='a \\ b'",
        "-Darg18='a \\\" b'",
        "-Darg19='A'",
        "-Darg20='\"a\"'",
        "-Darg21='B C'",
        "-Darg22='\"b c\"'",
        "-Darg23='D\"E'",
        "-Darg24='\"d\"e\"'",
        "-Darg25='C:\\F G'",
        "-Darg26='\"C:\\f g\"'",
        "-Darg27='C:\\H\"I'",
        "-Darg28='\"C:\\h\"i\"'",
        "-Darg29='C:\\J\\\"K'",
        "-Darg30='\"C:\\j\\\"k\"'",
        "-Darg31='C:\\L M '",
        "-Darg32='\"C:\\l m \"'",
        "-Darg33='C:\\N O\\'",
        "-Darg34='\"C:\\n o\\\"'",
        "-Darg35='C:\\P Q\\ '",
        "-Darg36='\"C:\\p q\\ \"'",
        "-Darg37='C:\\R\\S\\'",
        "-Darg38='C:\\R x\\S\\'",
        "-Darg39='\"C:\\r\\s\\\"'",
        "-Darg40='\"C:\\r x\\s\\\"'",
        "-Darg41='C:\\T U\\W\\'",
        "-Darg42='\"C:\\t u\\w\\\"'",
    ],
)
eof
}

function assert_output_of_the_program_with_many_jvm_flags() {
  expect_log 'arg0=()'
  expect_log 'arg1=( )'
  expect_log 'arg2=(")'
  expect_log 'arg3=("\\)'
  expect_log 'arg4=(\\)'
  expect_log 'arg5=(\\")'
  expect_log 'arg6=(with space)'
  expect_log 'arg7=(with^caret)'
  expect_log 'arg8=(space ^caret)'
  expect_log 'arg9=(caret^ space)'
  expect_log 'arg10=(with"quote)'
  expect_log 'arg11=(with\\backslash)'
  expect_log 'arg12=(one\\ backslash and \\space)'
  expect_log 'arg13=(two\\\\backslashes)'
  expect_log 'arg14=(two\\\\ backslashes \\\\and space)'
  expect_log 'arg15=(one\\"x)'
  expect_log 'arg16=(two\\\\"x)'
  expect_log 'arg17=(a \\ b)'
  expect_log 'arg18=(a \\" b)'
  expect_log 'arg19=(A)'
  expect_log 'arg20=("a")'
  expect_log 'arg21=(B C)'
  expect_log 'arg22=("b c")'
  expect_log 'arg23=(D"E)'
  expect_log 'arg24=("d"e")'
  expect_log 'arg25=(C:\\F G)'
  expect_log 'arg26=("C:\\f g")'
  expect_log 'arg27=(C:\\H"I)'
  expect_log 'arg28=("C:\\h"i")'
  expect_log 'arg29=(C:\\J\\"K)'
  expect_log 'arg30=("C:\\j\\"k")'
  expect_log 'arg31=(C:\\L M )'
  expect_log 'arg32=("C:\\l m ")'
  expect_log 'arg33=(C:\\N O\\)'
  expect_log 'arg34=("C:\\n o\\")'
  expect_log 'arg35=(C:\\P Q\\ )'
  expect_log 'arg36=("C:\\p q\\ ")'
  expect_log 'arg37=(C:\\R\\S\\)'
  expect_log 'arg38=(C:\\R x\\S\\)'
  expect_log 'arg39=("C:\\r\\s\\")'
  expect_log 'arg40=("C:\\r x\\s\\")'
  expect_log 'arg41=(C:\\T U\\W\\)'
  expect_log 'arg42=("C:\\t u\\w\\")'
}

function expect_program_runs() {
  local -r path="$1"
  (RUNFILES_DIR= \
   RUNFILES_MANIFEST_FILE= \
   RUNFILES_MANIFEST_ONLY= \
   "$path" >&"$TEST_log" ; ) || fail "expected success"
}

function expect_program_cannot_run() {
  local -r path="$1"
  (RUNFILES_DIR= \
   RUNFILES_MANIFEST_FILE= \
   RUNFILES_MANIFEST_ONLY= \
   "$path" >&"$TEST_log" ; ) && fail "expected failure" || true
}

# ----------------------------------------------------------------------
# TESTS
# ----------------------------------------------------------------------

function test_no_windows_jvm_flags_escaping() {
  local -r pkg="${FUNCNAME[0]}"
  local -r flag="--noincompatible_windows_escape_jvm_flags"

  create_java_file_that_prints_jvm_args "$pkg"
  create_build_file_with_many_jvm_flags "$pkg"

  # On all platforms, Bazel can build the target.
  bazel build --verbose_failures "$flag" "${pkg}:x" &>"$TEST_log" \
    || fail "expected success"
  if "$is_windows"; then
    # On Windows, the target cannot run.
    expect_program_cannot_run "bazel-bin/$pkg/x${EXE_EXT}"
    expect_log "Could not find or load main class"
  else
    # On other platforms, the program can run fine.
    expect_program_runs "bazel-bin/$pkg/x${EXE_EXT}"
    assert_output_of_the_program_with_many_jvm_flags
  fi
}

function test_windows_jvm_flags_escaping() {
  local -r pkg="${FUNCNAME[0]}"
  local -r flag="--incompatible_windows_escape_jvm_flags"

  create_java_file_that_prints_jvm_args "$pkg"
  create_build_file_with_many_jvm_flags "$pkg"

  # On all platforms, Bazel can build and run the target.
  bazel build --verbose_failures "$flag" "${pkg}:x" &>"$TEST_log" \
    || fail "expected success"
  expect_program_runs "bazel-bin/$pkg/x${EXE_EXT}"
  assert_output_of_the_program_with_many_jvm_flags
}

function test_untokenizable_jvm_flag_when_escaping_is_disabled() {
  local -r pkg="${FUNCNAME[0]}"
  local -r flag="--noincompatible_windows_escape_jvm_flags"

  create_java_file_that_prints_jvm_args "$pkg"
  create_build_file_for_untokenizable_flag "$pkg"

  # On all platforms, Bazel can build the target.
  bazel build "$flag" --verbose_failures "${pkg}:cannot_tokenize" \
    2>"$TEST_log" || fail "expected success"
  if "$is_windows"; then
    # On Windows, Bazel will not check the flag. It will just propagate the flag
    # to the launcher, which also just passes it to the JVM.  This is bad, the
    # flag should have been rejected because it cannot be Bash-tokenized.
    expect_program_runs "bazel-bin/$pkg/cannot_tokenize${EXE_EXT}"
    expect_log "arg0=('abc)"
  else
    # On other platforms, Bazel will build the target but it fails to run.
    expect_program_cannot_run "bazel-bin/$pkg/cannot_tokenize${EXE_EXT}"
    expect_log "syntax error near unexpected token"
  fi
}

function test_untokenizable_jvm_flag_when_escaping_is_enabled() {
  local -r pkg="${FUNCNAME[0]}"
  local -r flag="--incompatible_windows_escape_jvm_flags"
  create_java_file_that_prints_jvm_args "$pkg"
  create_build_file_for_untokenizable_flag "$pkg"

  if "$is_windows"; then
    # On Windows, Bazel will check the flag. It will just propagate the flag
    # to the launcher, which also just passes it to the JVM.  This is bad, the
    # flag should have been rejected because it cannot be Bash-tokenized.
    bazel build "$flag" --verbose_failures "${pkg}:cannot_tokenize" \
      2>"$TEST_log" && fail "expected failure" || true
    expect_log "ERROR:.*in jvm_flags attribute of java_binary rule"
  else
    # On other platforms, Bazel will build the target but it fails to run.
    bazel build "$flag" --verbose_failures "${pkg}:cannot_tokenize" \
      2>"$TEST_log" || fail "expected success"
    expect_program_cannot_run "bazel-bin/$pkg/cannot_tokenize${EXE_EXT}"
    expect_log "syntax error near unexpected token"
  fi
}

run_suite "Tests about how Bazel passes java_binary.jvm_flags to the binary"
