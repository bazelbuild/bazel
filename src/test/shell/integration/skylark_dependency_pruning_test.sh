#!/bin/bash
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

set -euo pipefail
# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }


#### HELPER FUNCTIONS ##################################################

function set_up() {
  mkdir -p pkg

  cat > pkg/BUILD << 'EOF'
load(":build.bzl", "build_rule")

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
        "executable": attr.label(executable = True, cfg = "host"),
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
    unused+="${input}_invalid\n"
  elif grep -q "unused" "${input}"; then
    unused+="${input}\n"
  else
    output+="$(cat "${input}") "
  fi
done

echo -n -e "${output}" > "${output_file}"
echo -n -e "${unused}" > "${unused_file}"
EOF

  chmod +x pkg/cat_unused.sh

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
  # The build should be trigerred, and the newer version of "b" should be used.
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

# Verify that the flag '--experimental_starlark_unused_inputs_list' is required
# for 'unused_inputs_list' usage. Note: defaults to true.
function test_experiment_flag_required() {
  # This should fail.
  bazel build --noexperimental_starlark_unused_inputs_list \
      //pkg:output >& $TEST_log && fail "Expected failure"
  exitcode=$?
  assert_equals 1 "$exitcode"
  expect_log "Use --experimental_starlark_unused_inputs_list"
}

run_suite "Tests Skylark dependency pruning"
