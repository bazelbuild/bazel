#!/usr/bin/env bash
#
# Copyright 2018 The Bazel Authors. All rights reserved.
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
# Test that calling 'bazel build' then 'bazel query' does not invalidate the
# Skyframe graph.

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

add_to_bazelrc "build --package_path=%workspace%"
add_to_bazelrc "query --incompatible_unambiguous_label_stringification"

#### TESTS #############################################################
# In some circumstances calling 'bazel build' then 'bazel query' forces
# invalidation of skyframe graph. This test helps to spot behavior like this.
#
# Note: this test scripts against bazel dump --skyframe=detailed output format.
# If format changes, test should be changed.
function test_query_doesnt_invalidate_skyframe() {
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
    name = "genrule",
    outs = ["genrule.txt"],
    cmd = "echo TEST > \$@",
)
EOF

  # Settle MODULE.bazel.lock before the build+query check below: a lockfile
  # updated at the end of the first build would be detected as a workspace file
  # change on the subsequent query, dirtying extension-backed repos (such as
  # @@platforms//host) and //a:genrule in Skyframe.
  bazel build --nobuild //a:genrule || fail "Warm-up build failed"

  bazel build //a:genrule || fail "Build failed"

  bazel query 'deps(//a:genrule)' || fail "Query failed"

  bazel dump --skyframe=deps >& "$TEST_log"

  # Occurs twice. Once with the top-level configuration and once with the
  # rule-transitioned configuration.
  expect_log_n "^CONFIGURED_TARGET:ConfiguredTargetKey{label=//a:genrule" 2

  rm a/BUILD
}

run_suite "Bazel blaze_query_invalidate tests"
