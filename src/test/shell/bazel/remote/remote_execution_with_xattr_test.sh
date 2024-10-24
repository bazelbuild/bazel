#!/bin/bash
#
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
  start_worker
}

function tear_down() {
  bazel clean >& $TEST_log
  stop_worker
}

function test_remote_grpc_cache_with_xattr() {
  # Create a simple build action that depends on two input files.
  echo hello > file1
  echo world > file2
  cat > BUILD << 'EOF'
genrule(
  name = "nothing",
  srcs = ["file1", "file2"],
  outs = ["nothing.txt"],
  cmd = "echo foo > $@",
)
EOF

  # Place an extended attribute on one of the input files that contains
  # the file's hash. This will allow Bazel to compute the file's digest
  # without reading the actual file contents.
  build_file_hash="$(openssl sha256 file1 | cut -d ' ' -f 2)"
  if type xattr 2> /dev/null; then
    xattr -wx user.checksum.sha256 "${build_file_hash}" file1
  elif type setfattr 2> /dev/null; then
    setfattr -n user.checksum.sha256 -v "0x${build_file_hash}" file1
  else
    # Skip this test, as this platform doesn't provide any known utility
    # for setting extended attributes.
    return 0
  fi

  # Run the build action with remote caching. We should not see 'VFS md5'
  # profiling events for file1, as getxattr() should be called instead.
  # We should see an entry for file2, though.
  bazel \
      --unix_digest_hash_attribute_name=user.checksum.sha256 \
      build \
      --incompatible_autoload_externally= \
      --remote_cache=grpc://localhost:${worker_port} \
      --profile=profile_log \
      --record_full_profiler_data \
      //:nothing || fail "Build failed"
  grep -q "VFS md5.*file1" profile_log && \
      fail "Bazel should not have computed a digest for file1"
  grep -q "VFS md5.*file2" profile_log || \
      fail "Bazel should have computed a digest for file2"
}

run_suite "Remote execution with extended attributes enabled"
