# Copyright 2020 The Bazel Authors. All rights reserved.
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

# Support for unittest.bash

#### Set up the test environment.

set -euo pipefail

cat_jvm_log () {
  if [[ "$log_content" =~ \
      "(error code:".*", error message: '".*"', log file: '"(.*)"')" ]]; then
    echo >&2
    echo "Content of ${BASH_REMATCH[1]}:" >&2
    cat "${BASH_REMATCH[1]}" >&2
  fi
}

# Print message in "$1" then exit with status "$2"
die () {
  # second argument is optional, defaulting to 1
  local status_code=${2:-1}
  # Stop capturing stdout/stderr, and dump captured output
  if [[ "$CAPTURED_STD_ERR" -ne 0 || "$CAPTURED_STD_OUT" -ne 0 ]]; then
    restore_outputs
    if [[ "$CAPTURED_STD_OUT" -ne 0 ]]; then
        cat "${TEST_TMPDIR}/captured.out"
        CAPTURED_STD_OUT=0
    fi
    if [[ "$CAPTURED_STD_ERR" -ne 0 ]]; then
        cat "${TEST_TMPDIR}/captured.err" 1>&2
        cat_jvm_log "$(cat "${TEST_TMPDIR}/captured.err")"
        CAPTURED_STD_ERR=0
    fi
  fi

  if [[ -n "${1-}" ]] ; then
      echo "$1" 1>&2
  fi
  if [[ -n "${BASH-}" ]]; then
    local caller_n=0
    while [[ $caller_n -lt 4 ]] && \
        caller_out=$(caller $caller_n 2>/dev/null); do
      test $caller_n -eq 0 && echo "CALLER stack (max 4):"
      echo "  $caller_out"
      let caller_n=caller_n+1
    done 1>&2
  fi
  if [[ -n "${status_code}" && "${status_code}" -ne 0 ]]; then
      exit "$status_code"
  else
      exit 1
  fi
}

# Print message in "$1" then record that a non-fatal error occurred in
# ERROR_COUNT
ERROR_COUNT="${ERROR_COUNT:-0}"
error () {
  if [[ -n "$1" ]] ; then
      echo "$1" 1>&2
  fi
  ERROR_COUNT=$(($ERROR_COUNT + 1))
}

# Die if "$1" != "$2", print $3 as death reason
check_eq () {
  [[ "$1" = "$2" ]] || die "Check failed: '$1' == '$2' ${3:+ ($3)}"
}

# Die if "$1" == "$2", print $3 as death reason
check_ne () {
  [[ "$1" != "$2" ]] || die "Check failed: '$1' != '$2' ${3:+ ($3)}"
}

# The structure of the following if statements is such that if '[[' fails
# (e.g., a non-number was passed in) then the check will fail.

# Die if "$1" > "$2", print $3 as death reason
check_le () {
  [[ "$1" -gt "$2" ]] || die "Check failed: '$1' <= '$2' ${3:+ ($3)}"
}

# Die if "$1" >= "$2", print $3 as death reason
check_lt () {
  [[ "$1" -lt "$2" ]] || die "Check failed: '$1' < '$2' ${3:+ ($3)}"
}

# Die if "$1" < "$2", print $3 as death reason
check_ge () {
  [[ "$1" -ge "$2" ]] || die "Check failed: '$1' >= '$2' ${3:+ ($3)}"
}

# Die if "$1" <= "$2", print $3 as death reason
check_gt () {
  [[ "$1" -gt "$2" ]] || die "Check failed: '$1' > '$2' ${3:+ ($3)}"
}

# Die if $2 !~ $1; print $3 as death reason
check_match ()
{
  expr match "$2" "$1" >/dev/null || \
    die "Check failed: '$2' does not match regex '$1' ${3:+ ($3)}"
}

# Run command "$1" at exit. Like "trap" but multiple atexits don't
# overwrite each other. Will break if someone does call trap
# directly. So, don't do that.
ATEXIT="${ATEXIT-}"
atexit () {
  if [[ -z "$ATEXIT" ]]; then
      ATEXIT="$1"
  else
      ATEXIT="$1 ; $ATEXIT"
  fi
  trap "$ATEXIT" EXIT
}

## TEST_TMPDIR
if [[ -z "${TEST_TMPDIR:-}" ]]; then
  export TEST_TMPDIR="$(mktemp -d ${TMPDIR:-/tmp}/bazel-test.XXXXXXXX)"
fi
if [[ ! -e "${TEST_TMPDIR}" ]]; then
  mkdir -p -m 0700 "${TEST_TMPDIR}"
  # Clean TEST_TMPDIR on exit
  atexit "rm -fr ${TEST_TMPDIR}"
fi

# Functions to compare the actual output of a test to the expected
# (golden) output.
#
# Usage:
#   capture_test_stdout
#   ... do something ...
#   diff_test_stdout "$TEST_SRCDIR/path/to/golden.out"

# Redirect a file descriptor to a file.
CAPTURED_STD_OUT="${CAPTURED_STD_OUT:-0}"
CAPTURED_STD_ERR="${CAPTURED_STD_ERR:-0}"

capture_test_stdout () {
  exec 3>&1 # Save stdout as fd 3
  exec 4>"${TEST_TMPDIR}/captured.out"
  exec 1>&4
  CAPTURED_STD_OUT=1
}

capture_test_stderr () {
  exec 6>&2 # Save stderr as fd 6
  exec 7>"${TEST_TMPDIR}/captured.err"
  exec 2>&7
  CAPTURED_STD_ERR=1
}

# Force XML_OUTPUT_FILE to an existing path
if [[ -z "${XML_OUTPUT_FILE:-}" ]]; then
  XML_OUTPUT_FILE=${TEST_TMPDIR}/output.xml
fi

# Functions to provide easy access to external repository outputs in the sibling
# repository layout.
#
# Usage:
#   bin_dir <repository name>
#   genfiles_dir <repository name>
#   testlogs_dir <repository name>

testlogs_dir() {
  echo $(bazel info bazel-testlogs | sed "s|bazel-out|bazel-out/$1|")
}
