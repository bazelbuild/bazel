#!/usr/bin/env bash
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
source "$(rlocation "io_bazel/src/test/shell/sandboxing_test_utils.sh")" \
  || { echo "sandboxing_test_utils.sh not found!" >&2; exit 1; }
source "$(rlocation "io_bazel/src/test/shell/bazel/remote/remote_utils.sh")" \
  || { echo "remote_utils.sh not found!" >&2; exit 1; }

function set_up() {
  writable_path=$(mktemp -d "${TEST_TMPDIR}/remote.XXXXXXXX")
  readonly_path=$(mktemp -d "${TEST_TMPDIR}/remote.XXXXXXXX")
  start_worker \
      --sandboxing \
      --sandboxing_writable_path="${writable_path}"

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
  bazel clean >& $TEST_log
  stop_worker
  if [ -d "${readonly_path}" ]; then
    rm -rf "${readonly_path}"
  fi
  if [ -d "${writable_path}" ]; then
    rm -rf "${writable_path}"
  fi
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
if [[ "$(uname -s)" != Linux ]]; then
  echo "RemoteWorker claims to only support Linux at the moment" 1>&2
  exit 0
fi
check_sandbox_allowed || exit 0

run_suite "Remote execution with sandboxing tests"
