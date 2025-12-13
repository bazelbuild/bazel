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
# execution_phase_tests.sh: miscellaneous integration tests of Bazel for
# behaviors that affect the execution phase.
#

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

function testenv_set_up() {
  echo 'startup --quiet' > .bazelrc
}

tear_down() {
  # Rollover to a new server log by shutting down the Bazel server.
  bazel shutdown
}

# Helper function to assert that one pattern appears before another in a file.
# Usage: assert_line_order <pattern1> <pattern2> <file>
# Verifies that pattern1 appears on an earlier line than pattern2.
function assert_line_order() {
  local pattern1=$1
  local pattern2=$2
  local file=$3
  local message="Expected '$pattern1' to appear before '$pattern2' in '$file'"
  
  local line1=$(grep -n "$pattern1" "$file" | head -1 | cut -d: -f1)
  local line2=$(grep -n "$pattern2" "$file" | head -1 | cut -d: -f1)
  
  if [[ -z "$line1" ]]; then
    fail "Pattern '$pattern1' not found in '$file'" $(__copy_to_undeclared_outputs "$file")
    return 1
  fi
  
  if [[ -z "$line2" ]]; then
    fail "Pattern '$pattern2' not found in '$file'" $(__copy_to_undeclared_outputs "$file")
    return 1
  fi
  
  if [[ "$line1" -lt "$line2" ]]; then
    return 0
  else
    fail "$message (line $line1 >= line $line2)" $(__copy_to_undeclared_outputs "$file")
    return 1
  fi
}

# Tests that you can set the spawn strategy flags to a list of strategies.
function test_multiple_strategies() {
  SERVER_LOG=$(bazel info server_log)
  bazel build --spawn_strategy=worker,local || fail
  # Can't test for exact strategy names here, because they differ between platforms and products.
  assert_contains "DefaultStrategyImplementations: \[.*, .*\]" "$SERVER_LOG"
}

# Tests that the hardcoded Worker strategies are not introduced with the new
# strategy selection
function test_no_worker_defaults() {
  SERVER_LOG=$(bazel info server_log)
  bazel build || fail
  # Can't test for exact strategy names here, because they differ between platforms and products.
  assert_not_contains "\"Closure\"" "$SERVER_LOG"
  assert_not_contains "\"DexBuilder\"" "$SERVER_LOG"
  assert_not_contains "\"Javac\"" "$SERVER_LOG"
}

# Tests that spawn strategy data structures are populated in the expected order and that it is
# reflected in the server log.
function test_spawn_strategy_order() {
  SERVER_LOG=$(bazel info server_log)
  bazel build \
    --spawn_strategy=worker,local \
    --genrule_strategy=local \
    --strategy=LOREM=local,worker \
    --strategy=IPSUM=worker,local \
    --strategy_regexp='//bar=local,worker' \
    --strategy_regexp='//foo.*\.cc,-//foo/bar=worker,local' \
    --strategy_regexp='//buzz=local' \
    --dynamic_local_strategy==local \
    --dynamic_local_strategy=FOO=worker,local \
    --dynamic_local_strategy=BAR=local,worker \
    --dynamic_remote_strategy==remote \
    --dynamic_remote_strategy=BETA=local,remote \
    --dynamic_remote_strategy=ALPHA=remote,local \
    --remote_local_fallback_strategy=local \
    --allowed_strategies_by_exec_platform=@platforms//host:host=local,worker \
    --allowed_strategies_by_exec_platform=//:foo_platform=worker,local \
    || fail

  # Values in specified order
  assert_contains 'DefaultStrategyImplementations: \[WorkerSpawnStrategy, StandaloneSpawnStrategy\]' "$SERVER_LOG"

  # Keys (regex filters) in reverse specified order, values in specified order
  assert_line_order \
    'FilterDescriptionToStrategyImplementations: "(?:(?>//foo.*\.cc)),-(?:(?>//foo/bar))" = \[WorkerSpawnStrategy, StandaloneSpawnStrategy\]' \
    'FilterDescriptionToStrategyImplementations: "(?:(?>//bar))" = \[StandaloneSpawnStrategy, WorkerSpawnStrategy\]' \
    "$SERVER_LOG"
  assert_line_order \
    'FilterDescriptionToStrategyImplementations: "(?:(?>//buzz))" = \[StandaloneSpawnStrategy\]' \
    'FilterDescriptionToStrategyImplementations: "(?:(?>//foo.*\.cc)),-(?:(?>//foo/bar))" = \[WorkerSpawnStrategy, StandaloneSpawnStrategy\]' \
    "$SERVER_LOG"

  # Keys (mnemonics) and values in specified order
  assert_contains 'MnemonicToStrategyImplementations: "Genrule" = \[StandaloneSpawnStrategy\]' "$SERVER_LOG"
  assert_line_order \
    'MnemonicToStrategyImplementations: "LOREM" = \[StandaloneSpawnStrategy, WorkerSpawnStrategy\]' \
    'MnemonicToStrategyImplementations: "IPSUM" = \[WorkerSpawnStrategy, StandaloneSpawnStrategy\]' \
    "$SERVER_LOG"

  # Keys in last-specified order and values in specified order
  assert_contains 'MnemonicToLocalDynamicStrategyImplementations: "" = \[StandaloneSpawnStrategy\]' "$SERVER_LOG"
  assert_line_order \
    'MnemonicToLocalDynamicStrategyImplementations: "FOO" = \[WorkerSpawnStrategy, StandaloneSpawnStrategy\]' \
    'MnemonicToLocalDynamicStrategyImplementations: "BAR" = \[StandaloneSpawnStrategy, WorkerSpawnStrategy\]' \
    "$SERVER_LOG"

  # Keys in last-specified order and values in specified order
  assert_contains 'MnemonicToRemoteDynamicStrategyImplementations: "" = \[RemoteSpawnStrategy\]' "$SERVER_LOG"
  assert_line_order \
    'MnemonicToRemoteDynamicStrategyImplementations: "BETA" = \[StandaloneSpawnStrategy, RemoteSpawnStrategy\]' \
    'MnemonicToRemoteDynamicStrategyImplementations: "ALPHA" = \[RemoteSpawnStrategy, StandaloneSpawnStrategy\]' \
    "$SERVER_LOG"

  # Keys in last-specified order and values in specified order
  assert_line_order \
    'FilterPlatformToStrategyImplementations: "@@platforms//host:host" = \[StandaloneSpawnStrategy, WorkerSpawnStrategy\]' \
    'FilterPlatformToStrategyImplementations: "//:foo_platform" = \[WorkerSpawnStrategy, StandaloneSpawnStrategy\]' \
    "$SERVER_LOG"

  assert_contains 'RemoteLocalFallbackImplementation: \[StandaloneSpawnStrategy\]' "$SERVER_LOG"
}

# Tests that Bazel catches an invalid strategy list that has an empty string as an element.
function test_empty_strategy_in_list_is_forbidden() {
  bazel build --spawn_strategy=worker,,local &> $TEST_log || true
  expect_log "--spawn_strategy=worker,,local: Empty values are not allowed as part of this comma-separated list of options"
}

# Test that when you set a strategy to the empty string, it gets removed from the map of strategies
# and thus results in the default strategy being used (the one set via --spawn_strategy=).
function test_empty_strategy_means_default() {
  SERVER_LOG=$(bazel info server_log)
  bazel build --spawn_strategy=worker,local --strategy=FooBar=local || fail
  assert_contains "\"FooBar\" = " "$SERVER_LOG"

  bazel build --spawn_strategy=worker,local --strategy=FooBar=local --strategy=FooBar= || fail
  assert_contains "\"FooBar\" = " "$SERVER_LOG"
}

# Tests that spawn filters for execution platform propagate into the spawn registry as expected.
# Resolution tested in unit tests.
function test_allowed_strategies_by_exec_platform() {
  SERVER_LOG=$(bazel info server_log)
  bazel build --spawn_strategy=worker,local --allowed_strategies_by_exec_platform="${default_host_platform}"=local || fail
  assert_contains '"'"${default_host_platform}"'" = \[\(Standalone\|Local\).*SpawnStrategy\]' "$SERVER_LOG"
}

# Tests that canonical labels can be used to target execution platform strategy filters.
function test_allowed_strategies_by_exec_platform_canonicalized() {
  SERVER_LOG=$(bazel info server_log)
  bazel build --spawn_strategy=worker,local --allowed_strategies_by_exec_platform="${default_host_platform}"=local || fail
  assert_contains '"'"${default_host_platform}"'" = \[\(Standalone\|Local\).*SpawnStrategy\]' "$SERVER_LOG"
}

# Tests that expected message is printed when no spawn strategy can be resolved.
function test_allowed_strategies_by_exec_platform_exhausted() {
  cat >BUILD <<'EOF'
genrule(
  name = "foo",
  srcs = ["input.txt"],
  outs = ["out"],
  cmd = "",
)
EOF
  touch input.txt
  bazel build --spawn_strategy=dynamic --allowed_strategies_by_exec_platform="${default_host_platform}"=worker,local //:foo 2> $TEST_log || true
  assert_contains "Your .* --allowed_strategies_by_exec_platform flags are probably too strict" "$TEST_log"
}

# Runs a build, waits for the given dir and file to appear, and then kills
# Bazel to check what happens with said files.
function build_and_interrupt() {
  local dir="${1}"; shift
  local file="${1}"; shift

  bazel clean
  bazel build --genrule_strategy=local \
    "${@}" //pkg &> $TEST_log &
  local pid=$!
  while [[ ! -e "${dir}" && ! -e "${file}" ]]; do
    echo "Still waiting for action to create outputs" >>$TEST_log
    sleep 1
  done
  kill "${pid}"
  wait || true
}

function test_local_deletes_plain_outputs_on_interrupt() {
  if is_windows; then
    cat 1>&2 <<EOF
This test is known to be broken on Windows because the kill+wait sequence
in build_and_interrupt doesn't seem to do the right thing.
Skipping...
EOF
    return 0
  fi

  mkdir -p pkg
  cat >pkg/BUILD <<'EOF'
genrule(
  name = "pkg",
  srcs = ["pkg.txt"],
  outs = ["dir", "file"],
  cmd = ("d=$(location :dir) f=$(location :file); "
         + "mkdir -p $$d; touch $$d/subfile $$f; sleep 60"),
)
EOF
  touch pkg/pkg.txt
  local genfiles_dir="$(bazel info $PRODUCT_NAME-genfiles)"
  local dir="${genfiles_dir}/pkg/dir"
  local file="${genfiles_dir}/pkg/file"

  build_and_interrupt "${dir}" "${file}" --noexperimental_local_lockfree_output
  [[ -d "${dir}" ]] || fail "Expected directory output to not exist"
  [[ -f "${file}" ]] || fail "Expected regular output to exist"

  build_and_interrupt "${dir}" "${file}" --experimental_local_lockfree_output
  if [[ -d "${dir}" ]]; then
    fail "Expected directory output to not exist"
  fi
  if [[ -f "${file}" ]]; then
     fail "Expected regular output to not exist"
  fi
}

function test_local_deletes_tree_artifacts_on_interrupt() {
  if is_windows; then
    cat 1>&2 <<EOF
This test is known to be broken on Windows because the kill+wait sequence
in build_and_interrupt doesn't seem to do the right thing.
Skipping...
EOF
    return 0
  fi

  mkdir -p pkg
  cat >pkg/rules.bzl <<'EOF'
def _test_tree_artifact_impl(ctx):
  tree = ctx.actions.declare_directory(ctx.attr.name + ".dir")
  cmd = """
mkdir -p {path} && touch {path}/file && sleep 60
""".format(path = tree.path)
  ctx.actions.run_shell(outputs = [tree], command = cmd)
  return DefaultInfo(files = depset([tree]))

test_tree_artifact = rule(
  implementation = _test_tree_artifact_impl,
)
EOF
  cat >pkg/BUILD <<'EOF'
load(":rules.bzl", "test_tree_artifact")
test_tree_artifact(name = "pkg")
EOF
  local bin_dir="$(bazel info $PRODUCT_NAME-bin)"
  local dir="${bin_dir}/pkg/pkg.dir"
  local file="${bin_dir}/pkg/pkg.dir/file"

  build_and_interrupt "${dir}" "${file}" --noexperimental_local_lockfree_output
  [[ -d "${dir}" ]] || fail "Expected tree artifact root to exist"

  build_and_interrupt "${dir}" "${file}" --experimental_local_lockfree_output
  [[ -d "${dir}" ]] || fail "Expected tree artifact root to exist"
  if [[ -f "${file}" ]]; then
     fail "Expected tree artifact contents to not exist"
  fi
}

function test_ignore_local_failures() {
  if is_windows; then
    cat 1>&2 <<EOF
This test is known to be broken on Windows because it does not have Posix signals
Skipping...
EOF
    return 0
  fi

  cat > BUILD <<'EOF'
genrule(
  name = "test",
  outs = ["test.txt"],
  cmd = (
      "if pwd | grep sandbox ; then "
      + "  echo Remote branch running; "
      + "  sleep 3; "
      + "  echo remote | tee $(location test.txt); "
      + "else "
      + "  echo Local branch killing itself; "
      + "  kill -9 $$$$; "
      + "  echo local | tee $(location test.txt); "
      + "fi; "
  ),
)

EOF

  bazel --noquiet build --internal_spawn_scheduler --genrule_strategy=dynamic \
    --dynamic_remote_strategy=sandboxed \
    --dynamic_local_strategy=standalone \
    --sandbox_add_mount_pair=/tmp \
    --experimental_dynamic_ignore_local_signals=8,9,10 \
    --experimental_local_lockfree_output \
    --experimental_local_execution_delay=0 \
    --verbose_failures \
    :all &>"$TEST_log" || fail "build failed"

  expect_not_log '^local$'
  expect_log '^remote$'
}

function test_internal_spawn_scheduler() {
  # This is just a basic test to see whether the dynamic scheduler is setting
  # up the correct local and remote strategies on all platforms.
  bazel build --internal_spawn_scheduler &>"$TEST_log" || fail "build failed"
}

run_suite "Tests for the execution strategy selection."
