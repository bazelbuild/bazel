#!/bin/bash
# Copyright 2026 The Bazel Authors. All rights reserved.
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

source "${TEST_SRCDIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
source "$(rlocation "io_bazel/src/test/shell/unittest.bash")"

readonly LINUX_SANDBOX=$(rlocation "io_bazel/src/main/tools/linux-sandbox")

set_up() {
  cd "${TEST_TMPDIR}"
  mkdir -p root/work
}

readonly VERIFICATION_SCRIPT='
  FAILED=0
  check_type() {
    local path=$1
    local expected=$2
    local actual=$(stat -f -c %T "$path" 2>/dev/null)
    if [ "$actual" != "$expected" ]; then
      echo "Failure: $path type is not $expected (actual: $actual)"
      FAILED=1
    fi
  }
  check_symlink() {
    local path=$1
    if ! [ -L "$path" ]; then
      echo "Failure: $path is not a symlink"
      FAILED=1
    fi
  }
  check_char_dev() {
    local path=$1
    if ! [ -c "$path" ]; then
      echo "Failure: $path is not a character device"
      FAILED=1
    fi
  }
  check_dir() {
    local path=$1
    if ! [ -d "$path" ]; then
      echo "Failure: $path is not a directory"
      FAILED=1
    fi
  }

  check_type /sys sysfs
  check_type /proc proc
  check_type /dev/shm tmpfs

  check_symlink /dev/stdin
  check_symlink /dev/stdout
  check_symlink /dev/stderr

  check_char_dev /dev/null
  check_char_dev /dev/zero
  check_char_dev /dev/full
  check_char_dev /dev/random
  check_char_dev /dev/urandom

  check_dir /proc/self/fd

  exit $FAILED
'

test_mounts_non_hermetic_no_netns() {
  "${LINUX_SANDBOX}" -W "${TEST_TMPDIR}" -- /bin/sh -c "${VERIFICATION_SCRIPT}" \
    &> "$TEST_log" || fail "sandbox not set up correctly"
}

test_mounts_non_hermetic_netns() {
  "${LINUX_SANDBOX}" -n -W "${TEST_TMPDIR}" -- /bin/sh -c "${VERIFICATION_SCRIPT}" \
    &> "$TEST_log" || fail "sandbox not set up correctly"
}

test_mounts_hermetic_no_netns() {
  "${LINUX_SANDBOX}" -W "${TEST_TMPDIR}/root/work" -h "${TEST_TMPDIR}/root" \
    -M /bin -m /bin -M /lib -m /lib -M /lib64 -m /lib64 -M /usr -m /usr \
    -- /bin/sh -c "${VERIFICATION_SCRIPT}" \
    &> "$TEST_log" || fail "sandbox not set up correctly"
}

test_mounts_hermetic_netns() {
  "${LINUX_SANDBOX}" -n -W "${TEST_TMPDIR}/root/work" -h "${TEST_TMPDIR}/root" \
    -M /bin -m /bin -M /lib -m /lib -M /lib64 -m /lib64 -M /usr -m /usr \
    -- /bin/sh -c "${VERIFICATION_SCRIPT}" \
    &> "$TEST_log" || fail "sandbox not set up correctly"
}

run_suite "linux-sandbox mounts tests"
