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

set -eu

if ! stat "$0" >&/dev/null; then
  echo >&2 "ERROR[runfiles_windows_test.sh] cannot locate GNU coreutils"
  exit 1
fi

function _log_base() {
  prefix=$1
  shift
  echo >&2 "${prefix}[$(basename "$0") $(date "+%H:%M:%S.%N (%z)")] $@"
}

function log_fatal() {
  _log_base "ERROR" "$@"
  exit 1
}

function fail() {
  _log_base "FAILED" "$@"
  exit 1
}

# Look up runfiles.sh manually, do not rely on rlocation being already defined.
# If all is working well, then rlocation should already be defined, because the
# native launcher of the sh_test already sourced runfiles.sh from @bazel_tools,
# but this test exercises runfiles.sh itself (from HEAD).
[ -n "${RUNFILES_MANIFEST_FILE:-}" ] \
  || log_fatal "RUNFILES_MANIFEST_FILE is undefined or empty"
runfiles_sh="$(cat "$RUNFILES_MANIFEST_FILE" \
  | fgrep "io_bazel/src/tools/runfiles/runfiles.sh" \
  | cut -d' ' -f2-)"
[ -n "$runfiles_sh" ] || fail "cannot find runfiles.sh"

# Unset existing definitions of the functions we want to test.
if type rlocation >&/dev/null; then
  unset is_absolute
  unset is_windows
  unset rlocation
fi
if type rlocation >&/dev/null; then
  fail "rlocation is still defined"
fi

# Assert that runfiles.sh needs $RUNFILES_MANIFEST_FILE.
unset RUNFILES_MANIFEST_FILE
if (source "$runfiles_sh" >&/dev/null) then
  fail "should fail to source '$runfiles_sh'"
fi

# Set a mock $RUNFILES_MANIFEST_FILE.
export RUNFILES_MANIFEST_FILE="$TEST_TMPDIR/mock-runfiles.txt"
cat >"$RUNFILES_MANIFEST_FILE" <<'end_of_manifest'
runfile/without/absolute/path
runfile/spaceless c:\path\to\runfile1
runfile/spaceful c:\path\to\runfile with spaces
end_of_manifest

# Source runfiles.sh and exercise its functions.
source "$runfiles_sh" || fail "cannot source '${runfiles_sh}'"

# Exercise the functions in runfiles.sh.
is_windows || fail "expected is_windows() to be true"

is_absolute "d:/foo" || fail "expected d:/foo to be absolute"
is_absolute "D:\\foo" || fail "expected D:\\foo to be absolute"
if is_absolute "/foo"; then
  fail "expected /foo not to be absolute"
fi

[[ -z "$(rlocation runfile/without/absolute/path)" ]] \
  || fail "rlocation 1 failed"
[[ "$(rlocation runfile/spaceless)" = "c:\\path\\to\\runfile1" ]] \
  || fail "rlocation 2 failed"
[[ "$(rlocation runfile/spaceful)" = "c:\\path\\to\\runfile with spaces" ]] \
  || fail "rlocation 3 failed"
[[ "$(rlocation "c:\\some absolute/path")" = "c:\\some absolute/path" ]] \
  || fail "rlocation 4 failed"
