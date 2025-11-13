#!/usr/bin/env bash
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

function test_output_readonly() {
  # Test that permission of output files are 0555 if --experimental_writable_outputs is not set.
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = "foo",
  srcs = [],
  outs = ["foo"],
  cmd = "echo 'foo' > \$@",
)
EOF

  bazel build \
      //a:foo >& $TEST_log || fail "Failed to build"

  ls -l bazel-bin/a/foo >& $TEST_log
  expect_log "-r-xr-xr-x"

  bazel clean >& $TEST_log || fail "Failed to clean"

  # Verify that changing the value of --experimental_writable_outputs results
  # in an update of the output permissions (invalidation of the action cache, etc)
  bazel build \
      --experimental_writable_outputs \
      //a:foo >& $TEST_log || fail "Failed to build"

  ls -l bazel-bin/a/foo >& $TEST_log
  expect_log "-rwxr-xr-x"

  bazel clean >& $TEST_log || fail "Failed to clean"
}

function test_output_writable() {
  # Test that permission of output files are 0755 if --experimental_writable_outputs is set.
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = "foo",
  srcs = [],
  outs = ["foo"],
  cmd = "echo 'foo' > \$@",
)
EOF

  bazel build \
      --experimental_writable_outputs \
      //a:foo >& $TEST_log || fail "Failed to build"

  ls -l bazel-bin/a/foo >& $TEST_log
  expect_log "-rwxr-xr-x"

  bazel clean >& $TEST_log || fail "Failed to clean"

  # Verify that changing the value of --experimental_writable_outputs results
  # in an update of the output permissions (invalidation of the action cache, etc)
  bazel build \
      //a:foo >& $TEST_log || fail "Failed to build"

  ls -l bazel-bin/a/foo >& $TEST_log
  expect_log "-r-xr-xr-x"

  bazel clean >& $TEST_log || fail "Failed to clean"
}

function test_create_tree_artifact_outputs_permissions() {
  mkdir -p pkg
  cat > pkg/def.bzl <<'EOF'
def _r(ctx):
    d = ctx.actions.declare_directory("%s_dir" % ctx.label.name)
    ctx.actions.run_shell(
        outputs = [d],
        command = "cd %s && touch x && touch y" % d.path,
    )
    return [DefaultInfo(files = depset([d]))]

r = rule(implementation = _r)
EOF

cat > pkg/BUILD <<'EOF'
load(":def.bzl", "r")

r(name = "a")
EOF

  bazel build pkg:a &>$TEST_log || fail "expected build to succeed"

  ls -l bazel-bin/pkg >& $TEST_log
  expect_log "dr-xr-xr-x"
  ls -l bazel-bin/pkg/a_dir/x >& $TEST_log
  expect_log "-r-xr-xr-x"
  ls -l bazel-bin/pkg/a_dir/y >& $TEST_log
  expect_log "-r-xr-xr-x"

  bazel build pkg:a --experimental_writable_outputs &>$TEST_log || fail "expected build to succeed"

  ls -l bazel-bin/pkg >& $TEST_log
  expect_log "drwxr-xr-x"
  ls -l bazel-bin/pkg/a_dir/x >& $TEST_log
  expect_log "-rwxr-xr-x"
  ls -l bazel-bin/pkg/a_dir/y >& $TEST_log
  expect_log "-rwxr-xr-x"
}

run_suite "bazel file permissions tests"
