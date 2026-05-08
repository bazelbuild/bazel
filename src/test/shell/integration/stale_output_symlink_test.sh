#!/usr/bin/env bash
#
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
#
# Regression test for https://github.com/bazelbuild/bazel/issues/29480.
#
# When an output path was previously created as a symlink (e.g. the executable
# of an sh_binary), and a later build needs to use that same path as a
# directory (because a same-named subpackage appeared at that location),
# Bazel must clean up the stale symlink before placing files underneath, and
# the per-action filesystem's cached symlink resolution must be invalidated.

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function tear_down() {
  bazel clean
  bazel shutdown
  rm -rf tools
}

function test_symlink_output_replaced_by_subpackage() {
  add_rules_shell "MODULE.bazel"
  # The bug only manifests when an action filesystem (RemoteActionFileSystem) is in use, which
  # happens whenever a disk or remote cache is configured.
  add_to_bazelrc "build --disk_cache=$TEST_TMPDIR/disk_cache"
  mkdir -p tools || fail "Can't create tools"
  cat > tools/wrapper_script.sh << 'EOF'
#!/bin/bash
echo hello
EOF
  chmod +x tools/wrapper_script.sh

  cat > tools/BUILD.bazel << 'EOF'
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

sh_binary(
    name = "bazel_wrapper",
    srcs = ["wrapper_script.sh"],
)
EOF

  bazel build //tools:bazel_wrapper >&$TEST_log \
    || fail "Failed first build of //tools:bazel_wrapper"

  # rules_shell creates the executable as a symlink to the source script,
  # which is what triggers the stale-state path in subsequent builds.
  local exec_path
  exec_path=$(bazel info bazel-bin)/tools/bazel_wrapper
  [[ -L "$exec_path" ]] \
    || fail "Expected $exec_path to be a symlink, got: $(ls -la "$exec_path" 2>&1)"

  # Move the sh_binary into a same-named subpackage. The output that was
  # previously a symlinked file at .../tools/bazel_wrapper must now become a
  # directory at .../tools/bazel_wrapper/ which contains the new outputs.
  rm tools/BUILD.bazel
  mkdir -p tools/bazel_wrapper
  cat > tools/bazel_wrapper/BUILD.bazel << 'EOF'
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

sh_binary(
    name = "bazel_wrapper",
    srcs = ["//tools:wrapper_script.sh"],
)
EOF
  cat > tools/BUILD.bazel << 'EOF'
exports_files(["wrapper_script.sh"])
EOF

  bazel build //tools/bazel_wrapper:bazel_wrapper >&$TEST_log \
    || fail "Failed second build after moving sh_binary to a subpackage"
}

run_suite "Stale output symlink tests"
