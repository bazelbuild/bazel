#!/bin/bash
#
# Copyright 2022 The Bazel Authors. All rights reserved.
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
# Tests remote execution and caching.

set -euo pipefail

# --- begin runfiles.bash initialization ---
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
source "$(rlocation "io_bazel/src/test/shell/bazel/remote/remote_utils.sh")" \
  || { echo "remote_utils.sh not found!" >&2; exit 1; }

function set_up() {
  CACHEDIR="${TEST_TMPDIR}/disk_cache"
}

function tear_down() {
  bazel clean >& $TEST_log
  rm -rf $CACHEDIR
}

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
  declare -r EXE_EXT=".exe"
else
  declare -r EXE_EXT=""
fi

function test_missing_blobs_from_ac() {
  check_missing_blobs_from_ac
}

function test_missing_blobs_from_ac_bwob() {
  check_missing_blobs_from_ac --remote_download_minimal
}

function check_missing_blobs_from_ac() {
  # Test that if files referenced by AC are missing from CAS, Bazel should
  # ignore the AC and rerun the generating action.
  cat > BUILD <<'EOF'
genrule(
  name = 'foo',
  srcs = ['foo.in'],
  outs = ['foo.out'],
  cmd = 'cat $(SRCS) > $@',
)

genrule(
  name = 'foobar',
  srcs = [':foo.out', 'bar.in'],
  outs = ['foobar.out'],
  cmd = 'cat $(SRCS) > $@',
)
EOF
  echo foo > foo.in
  echo bar > bar.in

  # Populate disk cache
  bazel build --disk_cache=$CACHEDIR //:foobar >& $TEST_log \
    || fail "Failed to build //:foobar"

  # Hit disk cache
  bazel clean
  bazel build --disk_cache=$CACHEDIR "$@" //:foobar >& $TEST_log \
    || fail "Failed to build //:foobar"

  expect_log "2 disk cache hit"

  # Remove CAS from disk cache
  rm -rf $CACHEDIR/cas

  # Since blobs referenced by the AC are missing, the AC should be ignored
  bazel clean
  bazel build --disk_cache=$CACHEDIR "$@" //:foobar >& $TEST_log \
    || fail "Failed to build //:foobar"

  expect_not_log "disk cache hit"
}

run_suite "Disk cache tests"
