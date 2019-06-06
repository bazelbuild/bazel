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
#
# Tests remote execution and caching.
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/../../sandboxing_test_utils.sh" \
  || { echo "sandboxing_test_utils.sh not found!" >&2; exit 1; }

function set_up() {
  work_path=$(mktemp -d "${TEST_TMPDIR}/remote.XXXXXXXX")
  writable_path=$(mktemp -d "${TEST_TMPDIR}/remote.XXXXXXXX")
  readonly_path=$(mktemp -d "${TEST_TMPDIR}/remote.XXXXXXXX")
  pid_file=$(mktemp -u "${TEST_TMPDIR}/remote.XXXXXXXX")
  worker_port=$(pick_random_unused_tcp_port) || fail "no port found"
  "${BAZEL_RUNFILES}/src/tools/remote/worker" \
      --work_path="${work_path}" \
      --listen_port=${worker_port} \
      --sandboxing \
      --sandboxing_writable_path="${writable_path}" \
      --pid_file="${pid_file}" >& $TEST_log &
  local wait_seconds=0
  until [ -s "${pid_file}" ] || [ "$wait_seconds" -eq 30 ]; do
    sleep 1
    ((wait_seconds++)) || true
  done
  if [ ! -s "${pid_file}" ]; then
    fail "Timed out waiting for remote worker to start."
  fi

  mkdir -p examples/genrule
  cat > examples/genrule/BUILD <<'EOF'
genrule(
  name = "simple",
  srcs = ["a.txt"],
  outs = ["simple.txt"],
  cmd = "wc $(location :a.txt) > $@",
)

genrule(
  name = "writes_to_writable_path",
  srcs = ["writable_path.txt"],
  outs = ["writes_to_writable_path.txt"],
  cmd = "touch $@; touch \"`cat $(location :writable_path.txt)`/out.txt\"",
)

genrule(
  name = "writes_to_readonly_path",
  srcs = ["readonly_path.txt"],
  outs = ["writes_to_readonly_path.txt"],
  cmd = "touch $@; touch \"`cat $(location :readonly_path.txt)`/out.txt\"",
)
EOF
  echo -n "12345" > examples/genrule/a.txt
  echo -n "$writable_path" > examples/genrule/writable_path.txt
  echo -n "$readonly_path" > examples/genrule/readonly_path.txt
}

function tear_down() {
  if [ -s "${pid_file}" ]; then
    local pid=$(cat "${pid_file}")
    kill "${pid}" || true
  fi
  rm -rf "${pid_file}"
  rm -rf "${work_path}"
}

function test_genrule() {
  bazel build \
      --spawn_strategy=remote \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_cache=grpc://localhost:${worker_port} \
      examples/genrule:simple &> $TEST_log \
    || fail "Hermetic genrule failed: examples/genrule:simple"
}

function test_genrule_can_write_to_path() {
  bazel build \
      --spawn_strategy=remote \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_cache=grpc://localhost:${worker_port} \
      examples/genrule:writes_to_writable_path &> $TEST_log \
    || fail "Hermetic genrule failed: examples/genrule:writes_to_writable_path"
  [ -f "$(cat examples/genrule/writable_path.txt)/out.txt" ] \
    || fail "Genrule did not write to expected path: $(cat examples/genrule/writable_path.txt)/out.txt"
}

function test_genrule_cannot_write_to_other_path() {
  bazel build \
      --spawn_strategy=remote \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_cache=grpc://localhost:${worker_port} \
      examples/genrule:writes_to_readonly_path &> $TEST_log \
    && fail "Non-hermetic genrule succeeded: examples/genrule:writes_to_readonly_path" || true
  [ -f "$(cat examples/genrule/readonly_path.txt)/out.txt" ] \
    && fail "Genrule was able to write to readonly path: $(cat examples/genrule/readonly_path.txt)/out.txt" || true
}

# The test shouldn't fail if the environment doesn't support running it.
check_supported_platform || exit 0
check_sandbox_allowed || exit 0

run_suite "Remote execution with sandboxing tests"
