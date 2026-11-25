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
#
# blaze_nouse_action_cache_test.sh: Test(s) of Bazel with the action cache
# disabled by --nouse_action_cache.
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

#### HELPER FUNCTIONS ##################################################

function create_hello_world() {
  mkdir -p hello/ || fail "Expected success"
  cat > hello/BUILD <<EOF
load("@rules_shell//shell:sh_test.bzl", "sh_test")
genrule(name='hello', outs=['hello.out'], cmd='echo "Hello" > \$@', message='Running hello genrule')
sh_test(name = 'passing', srcs = ['pass.sh'])
EOF

  cat >hello/pass.sh <<EOF
#!/bin/sh
echo "PASS"
EOF

  chmod +x hello/pass.sh
}

function clear_action_cache() {
  action_cache_dir="$(bazel info output_base 2>/dev/null)/action_cache"
  rm -rf "$action_cache_dir"/*
}

function set_up() {
  cd "${WORKSPACE_DIR}"
  add_rules_shell "MODULE.bazel"
  create_hello_world
  clear_action_cache
}

function assert_no_action_cache() {
  local output_base
  output_base="$(bazel info output_base 2>/dev/null)" || fail "Cannot get output_base"
  local action_cache_dir="${output_base}/action_cache"
  if [[ -d "${action_cache_dir}" ]]; then
    local action_cache_dir_contents
    action_cache_dir_contents="$(ls -A "${action_cache_dir}")"
    [[ -z "${action_cache_dir_contents}" ]] \
      || fail "Expected there to be no action cache, found the following in ${action_cache_dir}: ${action_cache_dir_contents}"
  fi
}

#### TESTS #############################################################

# Tests that running a build, running bazel shutdown, then running the same build
# executes the actions again, and that the size of the on-disk action cache is
# zero.
function test_no_action_cache() {
  bazel build --experimental_ui_debug_all_events --nouse_action_cache //hello:hello &> "$TEST_log" \
      || fail "Cannot build"
  expect_log "Running hello genrule"

  bazel shutdown &> "$TEST_log" || fail "Cannot shutdown"

  bazel build --experimental_ui_debug_all_events --nouse_action_cache //hello:hello &> "$TEST_log" \
      || fail "Cannot build"
  expect_log "Running hello genrule"

  assert_no_action_cache
}

# Tests that switching from --use_action_cache to --nouse_action_cache doesn't
# use the action cache.
# Use --enforce_constraints as irrelevant argument to force rebuild without changing output paths.
function test_does_not_use_existing_action_cache() {
  bazel build //hello:hello --enforce_constraints=true &> "$TEST_log" || fail "Cannot build"

  bazel build --experimental_ui_debug_all_events --nouse_action_cache //hello:hello \
      --enforce_constraints=false &> "$TEST_log" || fail "Cannot build"
  expect_log "Running hello genrule"
}

# Tests that switching from --use_action_cache to --nouse_action_cache to
# --use_action_cache does use the action cache again.
function test_reuse_existing_action_cache_after_disabled() {
  bazel build //hello:hello --enforce_constraints=true &> "$TEST_log" || fail "Cannot build"

  bazel build --experimental_ui_debug_all_events --nouse_action_cache //hello:hello \
      --enforce_constraints=false &> "$TEST_log" || fail "Cannot build"
  expect_log "Running hello genrule"

  bazel build --experimental_ui_debug_all_events --enforce_constraints=false //hello:hello \
      &> "$TEST_log" || fail "Cannot build"
  expect_not_log "Running hello genrule"
}

# Tests that --nouse_action_cache works with tests.
function test_no_action_cache_for_tests() {
  bazel test --nouse_action_cache //hello:passing &> "$TEST_log" || fail "Cannot build"

  assert_no_action_cache
}

# b/150153544: end-to-end demonstration of failure. Create two shared Starlark
# actions, one of which is a dependency of a genrule. Build the first (manually
# unblocking the timing gates), then build the genrule and the second. The
# genrule will discover that its input is missing.
function DISABLED_test_shared_action_bug_with_nouse_action_cache() {
  mkdir foo || fail "Couldn't mkdir"
  cat > foo/shared.bzl <<EOF || fail "Couldn't write bzl"
bash_command = """
# Notify genrule that shared2 has started.
echo -n a > foo/shared_start_fifo
# Wait for genrule to test for presence of input file.
read -n1 ignored < foo/shared_continue_fifo
touch %s
"""

def _impl(ctx):
  f = ctx.actions.declare_file("shared_out")
  ctx.actions.run_shell(outputs = [f], mnemonic="Shared",
                        command = bash_command % (f.path))
  return [DefaultInfo(files = depset([f]))]

shared = rule(implementation = _impl)
EOF

  cat > foo/BUILD <<'EOF' || fail "Couldn't write BUILD file"
load(":shared.bzl", "shared")
shared(name = "shared1")
shared(name = "shared2")
genrule(
    name = "top",
    srcs = [":shared1"],
    outs = ["top.out"],
    cmd = """
# Wait for shared2 to start: input file should now be deleted.
read -n1 ignored < foo/shared_start_fifo;
# Check for presence of input file. Don't fail yet, because we need to unblock
# shared2.
if [[ -e $(SRCS) ]]; then
  src_file_present=1
else
  src_file_present=0
fi
# Now that we've tested, unblock shared2 and let it finish.
echo -n a > foo/shared_continue_fifo
if [[ $$src_file_present == 0 ]]; then
echo 'Missing input file $(SRCS)!'
exit 1
else
  touch $@
fi
"""
)
EOF

  mkfifo foo/shared_start_fifo foo/shared_continue_fifo || fail "Couldn't fifo"
  bazel build --nouse_action_cache //foo:shared1 >& "$TEST_log" \
      || fail "Expected success" &
  readonly local pid="$!"
  # Because just the shared1 action is running, unblock it manually.
  IFS= read -r -n1 ignored < foo/shared_start_fifo || fail "Couldn't read"
  echo -n a > foo/shared_continue_fifo || fail "Couldn't write"
  wait $pid || fail "Expected success"
  bazel build --nouse_action_cache //foo:shared2 //foo:top \
      >& "$TEST_log" || fail "Expected success"
}

run_suite "Tests of Bazel with action cache disabled."
