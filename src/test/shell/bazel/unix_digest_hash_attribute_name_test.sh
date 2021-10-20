#!/usr/bin/env bash
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

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
set -euo pipefail
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

function test_xattr_operations_in_profile_log {
  touch WORKSPACE
  cat > BUILD << 'EOF'
genrule(
    name = "foo",
    outs = ["foo.out"],
    cmd = "touch $@",
)
EOF
  # Place an extended attribute with the checksum on the BUILD file.
  # macOS ships with the xattr command line tool, while Linux generally
  # ships with setfattr. Both have different styles of invocation.
  build_file_hash="$(openssl sha256 BUILD | cut -d ' ' -f 2)"
  if type xattr 2> /dev/null; then
    xattr -wx user.checksum.sha256 "${build_file_hash}" BUILD
  elif type setfattr 2> /dev/null; then
    setfattr -n user.checksum.sha256 -v "0x${build_file_hash}" BUILD
  fi

  bazel \
      --unix_digest_hash_attribute_name=user.checksum.sha256 \
      build \
      --profile=profile_log \
      --record_full_profiler_data \
      //:foo || fail "Build failed"
  grep -q "VFS xattr.*BUILD" profile_log || \
      fail "Bazel did not perform getxattr() calls"
}

run_suite "Integration tests for --unix_digest_hash_attribute_name"
