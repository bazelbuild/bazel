#!/bin/bash
#
# Copyright 2021 The Bazel Authors. All rights reserved.
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
# execution_phase_tests.sh: miscellaneous integration tests of Bazel for
# behaviors that affect the execution phase.
#

# --- begin runfiles.bash initialization ---
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

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

#### HELPER FUNCTIONS ##################################################

if ! type try_with_timeout >&/dev/null; then
  # Bazel's testenv.sh defines try_with_timeout but the Google-internal version
  # uses a different testenv.sh.
  function try_with_timeout() { $* ; }
fi

function set_up() {
  cd ${WORKSPACE_DIR}
  mkdir -p "foo"
}

#### TESTS #############################################################
function test_oom_sensitive_skyfunctions_semaphore_size_values_changed() {
  cat > foo/BUILD <<EOF
genrule(
  name = "foo",
  srcs = ["foo.in"],
  outs = ["foo.out"],
  cmd = "cat \$(location foo.in) >\$@",
)
EOF
  touch foo/foo.in

  # Semaphore disabled.
  bazel build --experimental_oom_sensitive_skyfunctions_semaphore_size=0 //foo:foo \
    &> "$TEST_log" || fail "build failed"

  # Default semaphore size.
  bazel build --experimental_oom_sensitive_skyfunctions_semaphore_size="HOST_CPUS" //foo:foo \
    &> "$TEST_log" || fail "build failed"

  # Non-standard semaphore size.
  bazel build --experimental_oom_sensitive_skyfunctions_semaphore_size="HOST_CPUS*1.5" //foo:foo \
    &> "$TEST_log" || fail "build failed"
}

function test_skyframe_cpu_heavy_skykeys_thread_pool_size_values_changed() {
  cat > foo/BUILD <<EOF
genrule(
  name = "foo",
  srcs = ["foo.in"],
  outs = ["foo.out"],
  cmd = "cat \$(location foo.in) >\$@",
)
EOF
  touch foo/foo.in

  bazel build --experimental_skyframe_cpu_heavy_skykeys_thread_pool_size=0 //foo:foo \
    &> "$TEST_log" || fail "build failed"

  bazel build --experimental_skyframe_cpu_heavy_skykeys_thread_pool_size="HOST_CPUS" //foo:foo \
    &> "$TEST_log" || fail "build failed"
}

run_suite "Integration tests for the options for Skyframe's analysis phase."
