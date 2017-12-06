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

stat "$0" >&/dev/null || log_fatal "cannot locate GNU coreutils"

# Unset existing definitions of the functions we want to test.
if type rlocation >&/dev/null; then
  unset is_absolute
  unset is_windows
  unset rlocation
fi
if rlocation >&/dev/null; then
  fail "rlocation is still defined"
fi

# Find runfiles.sh
runfiles_sh=$(dirname $0)/runfiles.sh
[[ -e "$runfiles_sh" ]] || fail "cannot find '$runfiles_sh'"

# Assert that runfiles.sh attempts to look up the runfiles directory.
# It will find the actual runfiles directory of this test.
unset RUNFILES_DIR
source "$runfiles_sh" || fail "cannot source '$runfiles_sh'"
[[ "$RUNFILES_DIR" = *.runfiles ]] \
  || fail "'$runfiles_sh' cannot find the runfiles directory"

# Set a mock $RUNFILES_DIR.
# Unset `rlocation` so runfiles.sh will define it again.
export RUNFILES_DIR="/path/to runfiles"
unset is_absolute
source "$runfiles_sh" || fail "cannot source '$runfiles_sh'"

# Exercise the functions in runfiles.sh.
if is_windows; then
  fail "expected is_windows() to be false"
fi

if is_absolute "d:/foo"; then
  fail "expected d:/foo not to be absolute"
fi
if is_absolute "D:\\foo"; then
  fail "expected D:\\foo not to be absolute"
fi
is_absolute "/foo" || fail "expected /foo to be absolute"

[[ "$(rlocation "some/runfile")" = "/path/to runfiles/some/runfile" ]] \
  || fail "rlocation 1 failed"
[[ "$(rlocation "/some absolute/runfile")" = "/some absolute/runfile" ]] \
  || fail "rlocation 2 failed"
