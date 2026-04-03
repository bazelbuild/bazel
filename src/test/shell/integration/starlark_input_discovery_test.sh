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

if is_windows; then
  export LC_ALL=C.utf8
elif is_linux; then
  export LC_ALL=C.UTF-8
else
  export LC_ALL=en_US.UTF-8
fi

add_to_bazelrc "build --package_path=%workspace%"
add_to_bazelrc "build --spawn_strategy=local"

#### HELPER FUNCTIONS ##################################################

function set_up() {
  mkdir -p pkg

  add_rules_shell "MODULE.bazel"

  # Rule that produces an unused_inputs_list file (the "producer" action).
  # This writes out the list of inputs that should be considered unused.
  cat > pkg/produce_unused_list.bzl << 'EOF'
def _produce_unused_list_impl(ctx):
    unused_list = ctx.outputs.unused_list
    content = "\n".join([f.path for f in ctx.files.unused_inputs])
    ctx.actions.write(
        output = unused_list,
        content = content,
    )

produce_unused_list = rule(
    attrs = {
        "unused_inputs": attr.label_list(allow_files = True),
        "unused_list": attr.output(),
    },
    implementation = _produce_unused_list_impl,
)
EOF

  # Rule that consumes the unused_inputs_list as an input and uses it for
  # input discovery. Writes a nanosecond timestamp to the output so we can
  # detect whether the action re-ran (the output changes only if re-executed).
  cat > pkg/consume_with_discovery.bzl << 'EOF'
def _consume_with_discovery_impl(ctx):
    inputs = ctx.attr.inputs.files
    output = ctx.outputs.out
    unused_inputs_list = ctx.file.unused_inputs_list
    all_inputs = depset([unused_inputs_list], transitive = [inputs])
    ctx.actions.run(
        inputs = all_inputs,
        outputs = [output],
        arguments = [output.path],
        executable = ctx.executable.executable,
        unused_inputs_list = unused_inputs_list,
    )

consume_with_discovery = rule(
    attrs = {
        "inputs": attr.label(),
        "executable": attr.label(executable = True, cfg = "exec"),
        "out": attr.output(),
        "unused_inputs_list": attr.label(allow_single_file = True),
    },
    implementation = _consume_with_discovery_impl,
)
EOF

  cat > pkg/BUILD << 'EOF'
load(":produce_unused_list.bzl", "produce_unused_list")
load(":consume_with_discovery.bzl", "consume_with_discovery")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

filegroup(
    name = "all_inputs",
    srcs = glob(["*.input"]),
)

produce_unused_list(
    name = "unused_list",
    unused_inputs = ["b.input"],
    unused_list = "unused.list",
)

sh_binary(
    name = "write_stamp",
    srcs = ["write_stamp.sh"],
)

consume_with_discovery(
    name = "output",
    out = "output.out",
    executable = ":write_stamp",
    inputs = ":all_inputs",
    unused_inputs_list = ":unused.list",
)
EOF

  # write_stamp.sh: writes a nanosecond timestamp to the output.
  # If the action re-runs, the timestamp changes; if cached, it stays the same.
  # Usage: write_stamp.sh output_file
  cat > pkg/write_stamp.sh << 'EOF'
#!/bin/sh
set -eu
date +%s%N > "$1"
EOF
  chmod +x pkg/write_stamp.sh

  # check_sandbox.sh: verifies files are present or absent in the sandbox.
  # Arguments: output_file expect_present... -- expect_absent...
  cat > pkg/check_sandbox.sh << 'SANDBOXEOF'
#!/bin/sh
set -eu
output_file="$1"
shift
mode="present"
for arg in "$@"; do
  if [ "${arg}" = "--" ]; then
    mode="absent"
    continue
  fi
  if [ "${mode}" = "present" ]; then
    if [ ! -f "${arg}" ]; then
      echo "FAIL: ${arg} should be in sandbox but is missing" >&2
      exit 1
    fi
  else
    if [ -f "${arg}" ]; then
      resolved=$(readlink -f "${arg}" 2>/dev/null || echo "${arg}")
      echo "FAIL: ${arg} should not be in sandbox but exists at ${resolved}" >&2
      exit 1
    fi
  fi
done
echo "ok" > "${output_file}"
SANDBOXEOF
  chmod +x pkg/check_sandbox.sh

  # Rule that checks file presence/absence in the sandbox.
  cat > pkg/check_sandbox_rule.bzl << 'EOF'
def _check_sandbox_impl(ctx):
    inputs = ctx.attr.inputs.files
    output = ctx.outputs.out
    unused_inputs_list = ctx.file.unused_inputs_list
    args = ctx.actions.args()
    args.add(output)
    args.add_all(ctx.files.expect_present)
    args.add("--")
    args.add_all(ctx.files.expect_absent)
    all_inputs = depset([unused_inputs_list], transitive = [inputs])
    ctx.actions.run(
        inputs = all_inputs,
        outputs = [output],
        arguments = [args],
        executable = ctx.executable.executable,
        unused_inputs_list = unused_inputs_list,
        execution_requirements = {"supports-path-mapping": "1"},
    )

check_sandbox_rule = rule(
    attrs = {
        "inputs": attr.label(),
        "executable": attr.label(executable = True, cfg = "exec"),
        "out": attr.output(),
        "unused_inputs_list": attr.label(allow_single_file = True),
        "expect_present": attr.label_list(allow_files = True),
        "expect_absent": attr.label_list(allow_files = True),
    },
    implementation = _check_sandbox_impl,
)
EOF

  echo "contentA" > pkg/a.input
  echo "contentB" > pkg/b.input
  echo "contentC" > pkg/c.input
}

function tear_down() {
  bazel clean
  bazel shutdown
  rm -rf pkg
}

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

# Get the stamp value from the output file.
function get_output_stamp() {
  cat "${PRODUCT_NAME}-bin/pkg/output.out"
}

function do_build() {
  bazel build //pkg:output "$@"
}

# Assert the action ran (stamp changed).
function assert_action_ran() {
  local before="$1"
  local after
  after=$(get_output_stamp)
  if [ "${before}" = "${after}" ]; then
    fail "Expected action to re-run, but stamp is unchanged: ${before}"
  fi
}

# Assert the action did NOT run (stamp unchanged).
function assert_action_cached() {
  local before="$1"
  local after
  after=$(get_output_stamp)
  assert_equals "${before}" "${after}"
}

# ----------------------------------------------------------------------
# TESTS
# ----------------------------------------------------------------------

# Tests that when unused_inputs_list is produced by a prior action (which
# traces the import tree), it is read during discoverInputs to trim inputs.
function test_input_discovery_trims_unused_inputs() {
  do_build || fail "build failed"
  local stamp1
  stamp1=$(get_output_stamp)

  # Change b.input (unused — not reachable from a.input).
  # The action should NOT re-run.

  echo "newContentB" > pkg/b.input
  do_build || fail "rebuild failed"
  assert_action_cached "${stamp1}"
}

# Tests that changing a used input triggers a rebuild.
function test_input_discovery_used_change_triggers_rebuild() {
  do_build || fail "initial build failed"
  local stamp1
  stamp1=$(get_output_stamp)

  # Change c.input (used — imported by a.input).

  echo "newContentC" > pkg/c.input
  do_build || fail "rebuild failed"
  assert_action_ran "${stamp1}"
}

# Tests that when the unused_inputs_list changes (different inputs become
# unused), the action correctly adjusts.
function test_input_discovery_list_changes() {
  do_build || fail "initial build failed"
  local stamp1
  stamp1=$(get_output_stamp)

  # Change the producer to mark "c" as unused instead of "b".
  cat > pkg/BUILD << 'EOF'
load(":produce_unused_list.bzl", "produce_unused_list")
load(":consume_with_discovery.bzl", "consume_with_discovery")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

filegroup(
    name = "all_inputs",
    srcs = glob(["*.input"]),
)

produce_unused_list(
    name = "unused_list",
    unused_inputs = ["c.input"],
    unused_list = "unused.list",
)

sh_binary(
    name = "write_stamp",
    srcs = ["write_stamp.sh"],
)

consume_with_discovery(
    name = "output",
    out = "output.out",
    executable = ":write_stamp",
    inputs = ":all_inputs",
    unused_inputs_list = ":unused.list",
)
EOF

  do_build || fail "rebuild failed"
  assert_action_ran "${stamp1}"
  local stamp2
  stamp2=$(get_output_stamp)

  # Now change c.input (newly unused). Should NOT re-run.

  echo "changedC" > pkg/c.input
  do_build || fail "rebuild failed"
  assert_action_cached "${stamp2}"

  # Change b.input (no longer unused). Should re-run.

  echo "changedB" > pkg/b.input
  do_build || fail "rebuild failed"
  assert_action_ran "${stamp2}"
}

# Tests that when no inputs are unused, changing any input triggers a rebuild.
function test_input_discovery_all_inputs_used() {
  cat > pkg/BUILD << 'EOF'
load(":produce_unused_list.bzl", "produce_unused_list")
load(":consume_with_discovery.bzl", "consume_with_discovery")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

filegroup(
    name = "all_inputs",
    srcs = glob(["*.input"]),
)

produce_unused_list(
    name = "unused_list",
    unused_inputs = [],
    unused_list = "unused.list",
)

sh_binary(
    name = "write_stamp",
    srcs = ["write_stamp.sh"],
)

consume_with_discovery(
    name = "output",
    out = "output.out",
    executable = ":write_stamp",
    inputs = ":all_inputs",
    unused_inputs_list = ":unused.list",
)
EOF

  do_build || fail "initial build failed"
  local stamp1
  stamp1=$(get_output_stamp)

  # Change b.input — should re-run since all inputs are used.

  echo "newContentB" > pkg/b.input
  do_build || fail "rebuild failed"
  assert_action_ran "${stamp1}"
}

# Tests that after server shutdown, the action cache preserves pruned inputs.
function test_input_discovery_cached_after_shutdown() {
  do_build || fail "initial build failed"
  local stamp1
  stamp1=$(get_output_stamp)

  bazel shutdown

  do_build || fail "rebuild after shutdown failed"
  assert_action_cached "${stamp1}"
}

# Tests that after server shutdown, changing an unused input does not cause
# the action to re-run. The action cache entry stores the pruned input set,
# so changes to pruned inputs are invisible to the cache check.
function test_input_discovery_unused_change_after_shutdown() {
  do_build || fail "initial build failed"
  local stamp1
  stamp1=$(get_output_stamp)

  bazel shutdown

  echo "newContentB" > pkg/b.input
  do_build || fail "rebuild after shutdown failed"
  assert_action_cached "${stamp1}"
}

# Tests that unused inputs are actually absent from the execution sandbox.
# The action script checks that b.input does NOT exist and fails if it does,
# proving that pre-execution input discovery removed it from the sandbox.
function test_input_discovery_removes_from_sandbox() {
  if is_windows; then
    # Sandboxing is not available on Windows.
    return 0
  fi
  cat > pkg/BUILD << 'EOF'
load(":produce_unused_list.bzl", "produce_unused_list")
load(":check_sandbox_rule.bzl", "check_sandbox_rule")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

filegroup(
    name = "all_inputs",
    srcs = glob(["*.input"]),
)

produce_unused_list(
    name = "unused_list",
    unused_inputs = ["b.input"],
    unused_list = "unused.list",
)

sh_binary(
    name = "check_sandbox",
    srcs = ["check_sandbox.sh"],
)

check_sandbox_rule(
    name = "output",
    out = "output.out",
    executable = ":check_sandbox",
    inputs = ":all_inputs",
    unused_inputs_list = ":unused.list",
    expect_present = ["a.input", "c.input"],
    expect_absent = ["b.input"],
)
EOF

  bazel build --spawn_strategy=sandboxed //pkg:output || fail "build failed — b.input was present in sandbox"
  local content
  content=$(cat "${PRODUCT_NAME}-bin/pkg/output.out")
  assert_equals "ok" "${content}"
}

# Tests that input discovery works correctly with path mapping
# (--experimental_output_paths=strip). With path mapping, derived artifact paths
# are stripped of their configuration segment. The unused_inputs_list producer
# writes mapped paths (since it sees mapped paths at execution time), and
# discoverInputs must match against those mapped paths.
function test_input_discovery_with_path_mapping() {
  if is_windows; then
    # Sandboxing is not available on Windows.
    return 0
  fi
  # A rule that generates a derived file by copying a source file.
  cat > pkg/generate.bzl << 'EOF'
def _generate_impl(ctx):
    out = ctx.outputs.out
    src = ctx.file.src
    ctx.actions.run_shell(
        inputs = [src],
        outputs = [out],
        command = "cp %s %s" % (src.path, out.path),
    )

generate = rule(
    attrs = {
        "src": attr.label(allow_single_file = True),
        "out": attr.output(),
    },
    implementation = _generate_impl,
)
EOF

  # A rule that produces an unused_inputs_list containing the mapped paths of
  # specified inputs. Uses a shell script to write the paths at execution time
  # (when path mapping is active), so the paths in the file are mapped.
  cat > pkg/produce_mapped_unused_list.bzl << 'EOF'
def _produce_mapped_unused_list_impl(ctx):
    unused_list = ctx.outputs.unused_list
    # Write the unused input paths. At execution time with path mapping,
    # file.path gives the analysis-time (unmapped) path. But the action's
    # command line sees mapped paths. We use a shell script that receives
    # the mapped paths as arguments.
    args = ctx.actions.args()
    args.add(unused_list)
    args.add_all(ctx.files.unused_inputs)
    ctx.actions.run_shell(
        outputs = [unused_list],
        inputs = ctx.files.unused_inputs,
        arguments = [args],
        command = 'out="$1"; shift; printf "%s\\n" "$@" > "$out"',
        execution_requirements = {"supports-path-mapping": "1"},
    )

produce_mapped_unused_list = rule(
    attrs = {
        "unused_inputs": attr.label_list(allow_files = True),
        "unused_list": attr.output(),
    },
    implementation = _produce_mapped_unused_list_impl,
)
EOF

  cat > pkg/BUILD << 'EOF'
load(":generate.bzl", "generate")
load(":produce_mapped_unused_list.bzl", "produce_mapped_unused_list")
load(":check_sandbox_rule.bzl", "check_sandbox_rule")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

generate(name = "gen_a", src = "a.input", out = "a.gen")
generate(name = "gen_b", src = "b.input", out = "b.gen")
generate(name = "gen_c", src = "c.input", out = "c.gen")

filegroup(
    name = "all_gen",
    srcs = ["a.gen", "b.gen", "c.gen"],
)

produce_mapped_unused_list(
    name = "unused_list",
    unused_inputs = ["b.gen"],
    unused_list = "unused.list",
)

sh_binary(
    name = "check_sandbox",
    srcs = ["check_sandbox.sh"],
)

check_sandbox_rule(
    name = "output",
    out = "output.out",
    executable = ":check_sandbox",
    inputs = ":all_gen",
    unused_inputs_list = ":unused.list",
    expect_present = ["a.gen", "c.gen"],
    expect_absent = ["b.gen"],
)
EOF

  bazel build \
    --experimental_output_paths=strip \
    --spawn_strategy=sandboxed \
    //pkg:output || fail "build failed with path mapping"
  local content
  content=$(cat "${PRODUCT_NAME}-bin/pkg/output.out")
  assert_equals "ok" "${content}"
}

run_suite "Tests Starlark input discovery with unused_inputs_list as input"
