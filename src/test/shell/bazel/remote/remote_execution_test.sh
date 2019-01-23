#!/bin/bash
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  work_path=$(mktemp -d "${TEST_TMPDIR}/remote.XXXXXXXX")
  cas_path=$(mktemp -d "${TEST_TMPDIR}/remote.XXXXXXXX")
  pid_file=$(mktemp -u "${TEST_TMPDIR}/remote.XXXXXXXX")
  attempts=1
  while [ $attempts -le 5 ]; do
    (( attempts++ ))
    worker_port=$(pick_random_unused_tcp_port) || fail "no port found"
    "${BAZEL_RUNFILES}/src/tools/remote/worker" \
        --work_path="${work_path}" \
        --listen_port=${worker_port} \
        --cas_path=${cas_path} \
        --incompatible_remote_symlinks \
        --pid_file="${pid_file}" >& /tmp/remote-worker-log &
    local wait_seconds=0
    until [ -s "${pid_file}" ] || [ "$wait_seconds" -eq 15 ]; do
      sleep 1
      ((wait_seconds++)) || true
    done
    if [ -s "${pid_file}" ]; then
      break
    fi
  done
  if [ ! -s "${pid_file}" ]; then
    fail "Timed out waiting for remote worker to start."
  fi
}

function tear_down() {
  bazel clean --expunge >& $TEST_log
  if [ -s "${pid_file}" ]; then
    local pid=$(cat "${pid_file}")
    kill "${pid}" || true
  fi
  rm -rf "${pid_file}"
  rm -rf "${work_path}"
  rm -rf "${cas_path}"
}

function is_file_uploaded() {
  h=$(shasum -a256 < "$1")
  if [ -e "$cas_path/${h:0:64}" ]; then return 0; else return 1; fi
}



function test_failing_cc_test_remote_spawn_cache() {
    rm "$TEST_log"
  ls -lh "${cas_path}"
  mkdir -p a
  cat > a/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
cc_test(
name = 'test',
srcs = [ 'test.cc' ],
)
EOF
  cat > a/test.cc <<EOF
#include <iostream>
int main() { std::cout << "Fail me!" << std::endl; return 1; }
EOF
  bazel test \
      --remote_cache=localhost:${worker_port} \
      --test_output=errors \
      //a:test \
      && fail "Expected test failure" || true
    ls -lh "${cas_path}"
   ls -lh bazel-testlogs/a/test
   $(is_file_uploaded bazel-testlogs/a/test/test.log) \
     || fail "Expected test log to be uploaded to remote execution"
   $(is_file_uploaded bazel-testlogs/a/test/test.xml) \
     || fail "Expected test xml to be uploaded to remote execution"
  # Check that logs are uploaded regardless of the spawn being cacheable.
  # Re-running a changed test that failed once renders the test spawn uncacheable.
  rm -f a/test.cc
  cat > a/test.cc <<EOF
#include <iostream>
int main() { std::cout << "Fail me again!" << std::endl; return 1; }
EOF
  bazel test \
      --remote_cache=localhost:${worker_port} \
      --test_output=errors \
      //a:test \
      && fail "Expected test failure" || true
  ls -lh bazel-testlogs/a/test
  shasum -a 256 < bazel-testlogs/a/test/test.log
  set -x
  cat /tmp/remote-worker-log
  ls -lh "${cas_path}"
   ($(is_file_uploaded bazel-testlogs/a/test/test.log) && fail "Expected test log to not be uploaded to remote execution") || true
   ($(is_file_uploaded bazel-testlogs/a/test/test.xml) && fail "Expected test xml to not be uploaded to remote execution") || true
  echo "all done"
}

run_suite "Remote execution and remote cache tests"
