#!/usr/bin/env bash
#
# Copyright 2019 The Bazel Authors. All rights reserved.
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
  cat > pkg/BUILD << 'EOF'
load(":build.bzl", "build_rule")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

filegroup(
    name = "all_inputs",
    srcs = glob(["*.input"]),
)

sh_binary(
    name = "cat_unused",
    srcs = ["cat_unused.sh"],
)

build_rule(
    name = "output",
    out = "output.out",
    executable = ":cat_unused",
    inputs = ":all_inputs",
)

sh_binary(
    name = "cat_unused2",
    srcs = ["cat_unused.sh"],
    data = [
      "output.out",
    ],
)

build_rule(
    name = "output2",
    out = "output2.out",
    executable = ":cat_unused2",
    inputs = ":all_inputs",
)
EOF

  cat > pkg/build.bzl << 'EOF'
def _impl(ctx):
    inputs = ctx.attr.inputs.files
    output = ctx.outputs.out
    unused_inputs_list = ctx.actions.declare_file(ctx.label.name + ".unused")
    arguments = []
    arguments += [output.path]
    arguments += [unused_inputs_list.path]
    for input in inputs.to_list():
        arguments += [input.path]
    ctx.actions.run(
        inputs = inputs,
        outputs = [output, unused_inputs_list],
        arguments = arguments,
        executable = ctx.executable.executable,
        unused_inputs_list = unused_inputs_list,
    )

build_rule = rule(
    attrs = {
        "inputs": attr.label(),
        "executable": attr.label(executable = True, cfg = "exec"),
        "out": attr.output(),
    },
    implementation = _impl,
)
EOF

  cat > pkg/cat_unused.sh << 'EOF'
#!/bin/sh
#
# Usage: cat_unused.sh output_file unused_file input...
# "Magic" input content values:
# - 'unused': mark the file unused, skip its content.
# - 'invalidUnused': produce an invalid unused file.
#
set -eu

output_file="$1"
shift
unused_file="$1"
shift

output=""
unused=""
for input in "$@"; do
  if grep -q "invalidUnused" "${input}"; then
    if [[ ! -z "${unused}" ]]; then
      unused="${unused}\n"
    fi
    unused="${unused}${input}_invalid"
  elif grep -q "unused" "${input}"; then
    if [[ ! -z "${unused}" ]]; then
      unused="${unused}\n"
    fi
    unused="${unused}${input}"
  else
    output="${output}$(cat "${input}") "
  fi
done

echo "${output}" > "${output_file}"
echo "${unused}" > "${unused_file}"
EOF

  chmod +x pkg/cat_unused.sh

  echo "contentA" > pkg/a.input
  echo "contentB" > pkg/b.input
  echo "contentC" > pkg/c.input
}

function tear_down() {
  rm -rf pkg
}

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

# Checks that the unused file contains exactly the list of files passed
# as parameters.
function check_unused_content() {
  unused_file="${PRODUCT_NAME}-bin/pkg/output.unused"
  expected=""
  for input in "$@"; do
    expected+="${input}"
    expected+=$'\n'
  done
  expected="$(echo "${expected}")" # Trimmed.
  actual="$(cat ${unused_file})"
  assert_equals "$expected" "$actual"
}

# Checks the content of the output.
function check_output_content() {
  output_file="${PRODUCT_NAME}-bin/pkg/output.out"
  actual="$(echo $(cat ${output_file}))" # Trimmed.
  assert_equals "$@" "$actual"
}

# ----------------------------------------------------------------------
# TESTS
# ----------------------------------------------------------------------

# Idea of the tests:
# - "cat_unused.sh" cats the lists of inputs.
# - if an input contains "unused", it is added to the "unused_list"
# - otherwise, its content is concatenated to the output.
# As a result, any input file that contains "unused" will be considered as
# unused by the build system..
#
# Note: this is not a valid use of "unused_inputs_list" as all input files do
# actually influence the build output, making this build rule
# non-deterministic.
# However, the goal of this test is to check the behavior of the build system
# with regard to the "unused_inputs_list" attribute.

# Typical "rebuild" scenario.
function test_dependency_pruning_scenario() {
  # Initial build.
  bazel build //pkg:output || fail "build failed"
  check_output_content "contentA contentB contentC"
  check_unused_content

  # Mark "b" as unused.
  echo "unused" > pkg/b.input
  bazel build //pkg:output || fail "build failed"
  check_output_content "contentA contentC"
  check_unused_content "pkg/b.input"

  # Change "b" again:
  # This time it should be used. But given that it was marked "unused"
  # the build should not trigger: "b" should still be considered unused.
  echo "newContentB" > pkg/b.input
  bazel build //pkg:output || fail "build failed"
  check_output_content "contentA contentC"
  check_unused_content "pkg/b.input"

  # Change c:
  # The build should be triggered, and the newer version of "b" should be used.
  echo "unused" > pkg/c.input
  bazel build //pkg:output || fail "build failed"
  check_output_content "contentA newContentB"
  check_unused_content "pkg/c.input"
}

function test_dependency_pruning_scenario_unicode() {
  local unicode="äöüÄÖÜß🌱"

  # Initial build.
  echo "contentD${unicode}" > "pkg/d${unicode}.input"
  bazel build //pkg:output || fail "build failed"
  check_output_content "contentA contentB contentC contentD${unicode}"
  check_unused_content

  # Mark "d" as unused.
  echo "unused" > "pkg/d${unicode}.input"
  bazel build //pkg:output || fail "build failed"
  check_output_content "contentA contentB contentC"
  check_unused_content "pkg/d${unicode}.input"

  # Change "d" again:
  # This time it should be used. But given that it was marked "unused"
  # the build should not trigger: "d" should still be considered unused.
  echo "newContentD${unicode}" > "pkg/d${unicode}.input"
  bazel build //pkg:output || fail "build failed"
  check_output_content "contentA contentB contentC"
  check_unused_content "pkg/d${unicode}.input"

  # Change c:
  # The build should be triggered, and the newer version of "d" should be used.
  echo "unused" > pkg/c.input
  bazel build //pkg:output || fail "build failed"
  check_output_content "contentA contentB newContentD${unicode}"
  check_unused_content "pkg/c.input"
}

# Verify that the state of the local action cache survives server shutdown.
function test_unused_shutdown() {
  # Mark "b" as unused + initial build
  echo "unused" > pkg/b.input
  bazel build //pkg:output || fail "build failed"
  check_output_content "contentA contentC"
  check_unused_content "pkg/b.input"

  # Shutdown.
  bazel shutdown

  # Change "b" again:
  # Check that the action is still cached, although b changed.
  echo "newContentB" > pkg/b.input
  bazel build //pkg:output || fail "build failed"
  check_output_content "contentA contentC"
  check_unused_content "pkg/b.input"

  # Change c:
  # The build should be triggered, and the newer version of "b" should be used.
  echo "unused" > pkg/c.input
  bazel build //pkg:output || fail "build failed"
  check_output_content "contentA newContentB"
  check_unused_content "pkg/c.input"
}

# Verify that actually used input files stay on the set ot inputs after a server
# shutdown.
function test_used_shutdown() {
  # Mark "b" as unused + initial build
  echo "unused" > pkg/b.input
  bazel build //pkg:output || fail "build failed"
  check_output_content "contentA contentC"
  check_unused_content "pkg/b.input"

  # Shutdown.
  bazel shutdown

  # Change "c", which is used.
  echo "newContentC" > pkg/c.input
  bazel build //pkg:output || fail "build failed"
  check_output_content "contentA newContentC"
  check_unused_content "pkg/b.input"
}

# Verify that file names that are not actually inputs in the unused file are
# ignored.
function test_invalid_unused() {
  # Mark "b" as producing an invalid unused file + initial build
  echo "invalidUnused" > pkg/b.input
  bazel build //pkg:output || fail "build failed"
  # Note: build should not fail: it is OK for unused file to contain
  # non-existing files.
  check_output_content "contentA contentC"
  check_unused_content "pkg/b.input_invalid"

  # Change "b" again:
  # It should just be picked-up, as it was not "unused".
  echo "newContentB" > pkg/b.input
  bazel build //pkg:output || fail "build failed"
  check_output_content "contentA newContentB contentC"
  check_unused_content
}

function test_missing_unused_inputs_list() {
  cat > pkg/cat_unused.sh << 'EOF'
#!/bin/sh
exit 0
EOF
  chmod +x pkg/cat_unused.sh
  bazel build //pkg:output >& $TEST_log && fail "Expected failure"
  exitcode=$?
  assert_equals 1 "$exitcode"
  expect_log "Action did not create expected output file listing unused inputs"
}

function setup_input_discovery() {
  mkdir -p pkg

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
        arguments = [output.path, unused_inputs_list.path],
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
  # Usage: write_stamp.sh output_file unused_inputs_list
  cat > pkg/write_stamp.sh << 'EOF'
#!/bin/sh
set -eu
# Every declared source input must be present exactly when the list does not name it.
for input in pkg/a.input pkg/b.input pkg/c.input; do
  if grep -Fxq "$input" "$2"; then
    test ! -e "$input" || { echo "Unused input present: $input" >&2; exit 1; }
  else
    test -f "$input" || { echo "Used input missing: $input" >&2; exit 1; }
  fi
done
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

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

# Get the stamp value from the output file.
function get_output_stamp() {
  cat "${PRODUCT_NAME}-bin/pkg/output.out"
}

function do_build() {
  bazel build --spawn_strategy=sandboxed //pkg:output "$@"
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

# Tests that a list produced by a prior action trims inputs before execution.
function test_input_discovery_trims_unused_inputs() {
  if is_windows; then
    # Sandboxing is not available on Windows.
    return 0
  fi
  setup_input_discovery
  do_build || fail "build failed"
  local stamp1
  stamp1=$(get_output_stamp)

  # Change b.input (listed as unused).
  # The action should NOT re-run.

  echo "newContentB" > pkg/b.input
  do_build || fail "rebuild failed"
  assert_action_cached "${stamp1}"
}

# Tests that changing a used input triggers a rebuild.
function test_input_discovery_used_change_triggers_rebuild() {
  if is_windows; then
    # Sandboxing is not available on Windows.
    return 0
  fi
  setup_input_discovery
  do_build || fail "initial build failed"
  local stamp1
  stamp1=$(get_output_stamp)

  # Change c.input (not listed as unused).

  echo "newContentC" > pkg/c.input
  do_build || fail "rebuild failed"
  assert_action_ran "${stamp1}"
}

# Tests that when the unused_inputs_list changes (different inputs become
# unused), the action correctly adjusts.
function test_input_discovery_list_changes() {
  if is_windows; then
    # Sandboxing is not available on Windows.
    return 0
  fi
  setup_input_discovery
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
  if is_windows; then
    # Sandboxing is not available on Windows.
    return 0
  fi
  setup_input_discovery
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
  if is_windows; then
    # Sandboxing is not available on Windows.
    return 0
  fi
  setup_input_discovery
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
  if is_windows; then
    # Sandboxing is not available on Windows.
    return 0
  fi
  setup_input_discovery
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
  setup_input_discovery
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
  setup_input_discovery
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

# A list that names itself must still invalidate the action when its contents change.
function test_input_discovery_list_names_itself() {
  if is_windows; then
    return 0
  fi
  setup_input_discovery
  cat > pkg/BUILD << 'EOF'
load(":consume_with_discovery.bzl", "consume_with_discovery")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

filegroup(name = "all_inputs", srcs = glob(["*.input"]))
sh_binary(name = "write_stamp", srcs = ["write_stamp.sh"])
consume_with_discovery(
    name = "output",
    out = "output.out",
    executable = ":write_stamp",
    inputs = ":all_inputs",
    unused_inputs_list = "self.list",
)
EOF
  printf 'pkg/b.input\npkg/self.list\n' > pkg/self.list
  do_build || fail "initial build failed"
  local stamp
  stamp=$(get_output_stamp)

  # Only the list changes: b must reappear, c must disappear, and the list must remain.
  printf 'pkg/c.input\npkg/self.list\n' > pkg/self.list
  do_build || fail "build after changing self-referencing list failed"
  assert_action_ran "${stamp}"
  stamp=$(get_output_stamp)

  echo "changedB" > pkg/b.input
  do_build || fail "build after changing restored input failed"
  assert_action_ran "${stamp}"
  stamp=$(get_output_stamp)

  # An empty list restores every input and resets the pruning state.
  : > pkg/self.list
  do_build || fail "build after clearing list failed"
  assert_action_ran "${stamp}"
  stamp=$(get_output_stamp)
  echo "changedC" > pkg/c.input
  do_build || fail "build after changing restored c.input failed"
  assert_action_ran "${stamp}"
}

# Both discovery and post-execution pruning must use a mapper that can verify
# identical contents when target- and exec-configuration inputs map to the same path.
function check_pruning_with_path_collisions() {
  local pre_execution="$1"
  cat > pkg/collisions.bzl << 'EOF'
def _generate_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".txt")
    ctx.actions.run_shell(
        inputs = [ctx.file.src],
        outputs = [out],
        arguments = [ctx.file.src.path, out.path],
        command = 'cp "$1" "$2"',
    )
    return [DefaultInfo(files = depset([out]))]

generate = rule(
    implementation = _generate_impl,
    attrs = {"src": attr.label(allow_single_file = True)},
)

def _consume_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".out")
    unused_list = ctx.actions.declare_file(ctx.label.name + ".unused")
    inputs = [ctx.file.same_target, ctx.file.same_exec, ctx.file.unused]
    outputs = [out]
    if ctx.attr.pre_execution:
        list_args = ctx.actions.args()
        list_args.add(unused_list)
        list_args.add(ctx.file.unused)
        ctx.actions.run_shell(
            inputs = [ctx.file.unused],
            outputs = [unused_list],
            arguments = [list_args],
            command = 'echo "$2" > "$1"',
            execution_requirements = {"supports-path-mapping": "1"},
        )
        inputs.append(unused_list)
    else:
        outputs.append(unused_list)
    args = ctx.actions.args()
    args.add(out)
    args.add(unused_list)
    args.add(ctx.file.unused)
    args.add("pre" if ctx.attr.pre_execution else "post")
    ctx.actions.run(
        executable = ctx.executable.tool,
        inputs = inputs,
        outputs = outputs,
        arguments = [args],
        unused_inputs_list = unused_list,
        execution_requirements = {"supports-path-mapping": "1"},
    )
    return [DefaultInfo(files = depset([out]))]

consume = rule(
    implementation = _consume_impl,
    attrs = {
        "same_target": attr.label(allow_single_file = True),
        "same_exec": attr.label(allow_single_file = True, cfg = "exec"),
        "unused": attr.label(allow_single_file = True),
        "tool": attr.label(allow_single_file = True, executable = True, cfg = "exec"),
        "pre_execution": attr.bool(),
    },
)
EOF
  cat > pkg/BUILD << EOF
load(":collisions.bzl", "consume", "generate")
generate(name = "same", src = "a.input")
generate(name = "unused", src = "b.input")
consume(
    name = "output",
    same_target = ":same",
    same_exec = ":same",
    unused = ":unused",
    tool = "consume.sh",
    pre_execution = ${pre_execution},
)
EOF
  cat > pkg/consume.sh << 'EOF'
#!/bin/sh
set -eu
if [ "$4" = pre ]; then
  test ! -e "$3" || { echo "Unused input present: $3" >&2; exit 1; }
  test -f "$2"
else
  echo "$3" > "$2"
fi
date +%s%N > "$1"
EOF
  chmod +x pkg/consume.sh
  do_build --experimental_output_paths=strip || fail "initial build with path collisions failed"
  local stamp
  stamp=$(get_output_stamp)
  echo "changed unused input" > pkg/b.input
  do_build --experimental_output_paths=strip || fail "rebuild with path collisions failed"
  assert_action_cached "${stamp}"
}

function test_input_discovery_with_path_collisions() {
  if is_windows; then
    return 0
  fi
  check_pruning_with_path_collisions True
}

function test_post_execution_pruning_with_path_collisions() {
  if is_windows; then
    return 0
  fi
  check_pruning_with_path_collisions False
}

run_suite "Tests Starlark dependency pruning"
