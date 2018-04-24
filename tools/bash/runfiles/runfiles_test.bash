#!/bin/bash
#
# Copyright 2018 The Bazel Authors. All rights reserved.
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

function _log_base() {
  prefix=$1
  shift
  echo >&2 "${prefix}[$(basename "${BASH_SOURCE[0]}"):${BASH_LINENO[1]} ($(date "+%H:%M:%S %z"))] $*"
}

function fail() {
  _log_base "FAILED" "$@"
  exit 1
}

function log_fail() {
  # non-fatal version of fail()
  _log_base "FAILED" $*
}

function log_info() {
  _log_base "INFO" $*
}

which uname >&/dev/null || fail "cannot locate GNU coreutils"

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  function is_windows() { true; }
  ;;
*)
  function is_windows() { false; }
  ;;
esac

function find_runfiles_lib() {
  # Unset existing definitions of the functions we want to test.
  if type rlocation >&/dev/null; then
    unset rlocation
    unset runfiles_export_envvars
  fi

  if [[ "${RUNFILES_MANIFEST_ONLY:-}" == "1" ]]; then
    grep -m1 "^io_bazel/tools/bash/runfiles/runfiles.bash" \
             "${RUNFILES_MANIFEST_FILE}" | cut -d ' ' -f 2-
  elif [[ -n "${RUNFILES_DIR:-}" && -d "$RUNFILES_DIR" ]]; then
    echo "${RUNFILES_DIR}/io_bazel/tools/bash/runfiles/runfiles.bash"
  else
    echo >&2 "ERROR: cannot find //tools/bash/runfiles:runfiles.bash"
    return 1
  fi
}

function test_rlocation_call_requires_no_envvars() {
  export RUNFILES_DIR=mock/runfiles
  export RUNFILES_MANIFEST_FILE=
  export RUNFILES_MANIFEST_ONLY=
  source "$runfiles_lib_path" || fail
}

function test_rlocation_argument_validation() {
  export RUNFILES_DIR=
  export RUNFILES_MANIFEST_FILE=
  export RUNFILES_MANIFEST_ONLY=
  source "$runfiles_lib_path"

  # Test invalid inputs to make sure rlocation catches these.
  if rlocation "../foo" >&/dev/null; then
    fail
  fi
  if rlocation "foo/.." >&/dev/null; then
    fail
  fi
  if rlocation "foo/../bar" >&/dev/null; then
    fail
  fi
  if rlocation "./foo" >&/dev/null; then
    fail
  fi
  if rlocation "foo/." >&/dev/null; then
    fail
  fi
  if rlocation "foo/./bar" >&/dev/null; then
    fail
  fi
  if rlocation "//foo" >&/dev/null; then
    fail
  fi
  if rlocation "foo//" >&/dev/null; then
    fail
  fi
  if rlocation "foo//bar" >&/dev/null; then
    fail
  fi
  if rlocation "\\foo" >&/dev/null; then
    fail
  fi
}

function test_rlocation_abs_path() {
  export RUNFILES_DIR=
  export RUNFILES_MANIFEST_FILE=
  export RUNFILES_MANIFEST_ONLY=
  source "$runfiles_lib_path"

  if is_windows; then
    [[ "$(rlocation "c:/Foo")" == "c:/Foo" ]] || fail
    [[ "$(rlocation "c:\\Foo")" == "c:\\Foo" ]] || fail
  else
    [[ "$(rlocation "/Foo")" == "/Foo" ]] || fail
  fi
}

function test_init_manifest_based_runfiles() {
  local tmpdir="$(mktemp -d $TEST_TMPDIR/tmp.XXXXXXXX)"
  cat > $tmpdir/foo.runfiles_manifest << 'EOF'
a/b c/d
e/f g h
EOF

  export RUNFILES_DIR=
  export RUNFILES_MANIFEST_FILE=$tmpdir/foo.runfiles_manifest
  export RUNFILES_MANIFEST_ONLY=1
  source "$runfiles_lib_path"

  [[ -z "$(rlocation a)" ]] || fail
  [[ "$(rlocation a/b)" == "c/d" ]] || fail
  [[ "$(rlocation e/f)" == "g h" ]] || fail
  [[ -z "$(rlocation c/d)" ]] || fail
}

function test_manifest_based_envvars() {
  local tmpdir="$(mktemp -d $TEST_TMPDIR/tmp.XXXXXXXX)"
  echo "a b" > $tmpdir/foo.runfiles_manifest

  export RUNFILES_DIR=
  export RUNFILES_MANIFEST_FILE=$tmpdir/foo.runfiles_manifest
  export RUNFILES_MANIFEST_ONLY=1
  mkdir -p $tmpdir/foo.runfiles
  source "$runfiles_lib_path"

  runfiles_export_envvars
  [[ "${RUNFILES_DIR:-}" == "$tmpdir/foo.runfiles" ]] || fail
  [[ "${RUNFILES_MANIFEST_FILE:-}" == "$tmpdir/foo.runfiles_manifest" ]] || fail
  [[ "${RUNFILES_MANIFEST_ONLY:-}" == 1 ]] || fail
}

function test_init_directory_based_runfiles() {
  local tmpdir="$(mktemp -d $TEST_TMPDIR/tmp.XXXXXXXX)"

  export RUNFILES_DIR=${tmpdir}/mock/runfiles
  export RUNFILES_MANIFEST_FILE=
  export RUNFILES_MANIFEST_ONLY=
  source "$runfiles_lib_path"

  mkdir -p "$RUNFILES_DIR"
  [[ "$(rlocation a)" == */mock/runfiles/a ]] || fail
  [[ "$(rlocation a/b)" == *mock/runfiles/a/b ]] || fail
}

function test_directory_based_envvars() {
  export RUNFILES_DIR=mock/runfiles
  export RUNFILES_MANIFEST_FILE=
  export RUNFILES_MANIFEST_ONLY=
  source "$runfiles_lib_path"

  runfiles_export_envvars
  [[ "${RUNFILES_DIR:-}" == "mock/runfiles" ]] || fail
  [[ -z "${RUNFILES_MANIFEST_FILE:-}" ]] || fail
  [[ -z "${RUNFILES_MANIFEST_ONLY:-}" ]] || fail
}

function main() {
  local -r manifest_only="${RUNFILES_MANIFEST_ONLY:-}"
  local -r manifest_file="${RUNFILES_MANIFEST_FILE:-}"
  local -r dir="${RUNFILES_DIR:-}"
  local -r runfiles_lib_path=$(find_runfiles_lib)

  local -r tests=$(declare -F | grep " -f test" | awk '{print $3}')
  local failure=0
  for t in $tests; do
    export RUNFILES_MANIFEST_ONLY="$manifest_only"
    export RUNFILES_MANIFEST_FILE="$manifest_file"
    export RUNFILES_DIR="$dir"
    if ! ($t); then
      failure=1
    fi
  done
  return $failure
}

main
