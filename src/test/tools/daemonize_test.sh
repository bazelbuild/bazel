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

set -euo pipefail

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
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

source "$(rlocation "io_bazel/src/test/shell/unittest.bash")"

readonly DAEMONIZE=$(rlocation "io_bazel/src/main/tools/daemonize")

set_up() {
  cd "${TEST_TMPDIR}"
}

tear_down() {
  rm -rf "${TEST_TMPDIR}"/{full,pidfile}
}

test_pid_file_open_permission_error_fails() {
  touch pidfile
  chmod -w pidfile

  "${DAEMONIZE}" -l /dev/null -p pidfile /bin/true /bin/true \
      >&"${TEST_log}" && fail "Expected failure"

  expect_log "daemonize: Failed to create pidfile: Permission denied"
}

test_pid_file_write_disk_full_fails() {
  # unshare is not available on macOS.
  [[ "$(uname -s)" == "Darwin" ]] && return 0
  unshare -Urm bash -ec "
trap 'exit 255' ERR
mkdir full
mount -t tmpfs -o size=1K none full

/bin/dd if=/dev/zero of=full/fill bs=1024 count=1 status=none

exec '${DAEMONIZE}' -l /dev/null -p full/pidfile /bin/true a >&'${TEST_log}'
" && fail 'Expected failure'

  assert_equals 1 $?
  expect_log "$(_concat \
      "daemonize: Failed to write pid [[:digit:]]\+ to full/pidfile: " \
      "No space left on device")"
}

_concat() {
  printf "%s" "$@"
}

run_suite "daemonize tests"
