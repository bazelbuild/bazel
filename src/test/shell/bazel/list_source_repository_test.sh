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

# The repository output must be invalidated when a tracked source disappears.
# .git/logs/HEAD does not change for ordinary workspace edits, so without the
# source-tree watches the second build can reuse a stale list containing the
# deleted file. Build twice in one Bazel server session to exercise that
# invalidation.
function test_source_deletion_invalidates_repository() {
  cp "$(rlocation "io_bazel/src/test/shell/bazel/list_source_repository.bzl")" .
  cat > "$(setup_module_dot_bazel)" <<'EOF'
list_source_repository = use_repo_rule(
    "//:list_source_repository.bzl",
    "list_source_repository",
)

list_source_repository(name = "local_source_list")
EOF
  cat > BUILD <<'EOF'
genrule(
    name = "capture_sources",
    srcs = ["@local_source_list//:sources"],
    outs = ["captured_sources.txt"],
    cmd = "cp $< $@",
)
EOF
  mkdir -p source
  touch source/removed.txt

  bazel build //:capture_sources >& "$TEST_log" || fail "initial build failed"
  grep -qx "source/removed.txt" bazel-bin/captured_sources.txt \
    || fail "source missing from initial repository output"

  rm source/removed.txt
  bazel build //:capture_sources >& "$TEST_log" || fail "rebuild after deletion failed"
  if grep -qx "source/removed.txt" bazel-bin/captured_sources.txt; then
    fail "deleted source remained in repository output"
  fi
}

run_suite "list_source_repository tests"
