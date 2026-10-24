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
# Tests that no sandbox stages the output of ctx.actions.copy as a symlink.
#

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  mkdir -p pkg
  cat > pkg/rule.bzl <<'EOS'
def _impl(ctx):
    gen = ctx.actions.declare_file(ctx.label.name + "_gen.txt")
    ctx.actions.write(output = gen, content = "generated")
    cp = ctx.actions.declare_file(ctx.label.name + "_copy.txt")
    ctx.actions.copy(output = cp, src = gen)
    cp_cp = ctx.actions.declare_file(ctx.label.name + "_copy_copy.txt")
    ctx.actions.copy(output = cp_cp, src = cp)

    tree = ctx.actions.declare_directory(ctx.label.name + "_tree")
    ctx.actions.run_shell(
        outputs = [tree],
        command = "mkdir -p {t}/a/b && echo one > {t}/f1.txt && echo deep > {t}/a/b/f2.txt".format(t = tree.path),
    )
    tree_cp = ctx.actions.declare_directory(ctx.label.name + "_tree_copy")
    ctx.actions.copy(output = tree_cp, src = tree)

    # Consumers see only what is listed as inputs.
    report = ctx.actions.declare_file(ctx.label.name + "_report.txt")
    inputs = [cp, cp_cp, tree_cp]
    if ctx.attr.with_source:
        inputs.append(gen)
    ctx.actions.run_shell(
        inputs = inputs,
        outputs = [report],
        mnemonic = "Consume",
        command = """
: salt %s
here=$(pwd -P)
check() {
  if [ -L "$1" ]; then kind=LINK; else kind=FILE; fi
  real=$(perl -MCwd -e 'print Cwd::realpath(shift)' "$1")
  case "$real" in "$here"/*) where=INSIDE;; *) where=OUTSIDE;; esac
  echo "$1 $kind $where $(cat "$1" 2>/dev/null | head -1)"
}
{
  check %s
  check %s
  check %s/f1.txt
  check %s/a/b/f2.txt
  echo "tree_links=$(find %s -type l | wc -l | tr -d ' ')"
  if [ -e %s ]; then
    check %s
    # The inode of the staged symlink itself (ls does not follow it). It only stays the same
    # across runs if the sandbox directory was reused and the symlink was not re-created.
    echo "gen_inode=$(ls -i %s | awk '{print $1}')"
  fi
} > %s
""" % (ctx.attr.salt, cp.path, cp_cp.path, tree_cp.path, tree_cp.path, tree_cp.path, gen.path, gen.path, gen.path, report.path),
    )
    return [DefaultInfo(files = depset([report]))]

copy_consumer = rule(
    implementation = _impl,
    attrs = {
        "salt": attr.string(),
        "with_source": attr.bool(default = False),
    },
)
EOS
  cat > pkg/BUILD <<'EOS'
load(":rule.bzl", "copy_consumer")

copy_consumer(name = "plain")

copy_consumer(
    name = "with_source",
    with_source = True,
)
EOS
}

function set_salt() {
  # Changes only the consumer's command line.
  cat > pkg/BUILD <<EOS
load(":rule.bzl", "copy_consumer")

copy_consumer(name = "plain", salt = "$1")

copy_consumer(
    name = "with_source",
    with_source = True,
    salt = "$1",
)
EOS
}

# Asserts that every copy input the consumer saw was a regular file inside the sandbox.
function assert_copies_are_regular_files() {
  local report="$1"
  [[ -f "$report" ]] || fail "missing report $report"
  cp "$report" "$TEST_log"
  grep -q "_copy.txt FILE INSIDE generated" "$report" || fail "copy is not a regular file"
  grep -q "_copy_copy.txt FILE INSIDE generated" "$report" || fail "copy of copy is not a regular file"
  grep -q "f1.txt FILE INSIDE one" "$report" || fail "tree child f1 is not a regular file"
  grep -q "f2.txt FILE INSIDE deep" "$report" || fail "nested tree child is not a regular file"
  grep -q "tree_links=0" "$report" || fail "tree copy contains links"
}

function run_consumer_test() {
  local strategy="$1"
  bazel build --spawn_strategy="$strategy" //pkg:plain &> "$TEST_log" \
    || fail "build failed with $strategy"
  assert_copies_are_regular_files bazel-bin/pkg/plain_report.txt
}

function test_copy_is_regular_file_in_default_sandbox() {
  bazel build --spawn_strategy=sandboxed //pkg:plain &> "$TEST_log" || fail "build failed"
  assert_copies_are_regular_files bazel-bin/pkg/plain_report.txt
}

function test_copy_is_regular_file_in_processwrapper_sandbox() {
  run_consumer_test processwrapper-sandbox
}

function test_copy_is_regular_file_in_darwin_sandbox() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "SKIPPED: darwin-sandbox only exists on macOS" | tee -a "$TEST_log"
    return 0
  fi
  run_consumer_test darwin-sandbox
}

function test_copy_is_regular_file_in_linux_sandbox() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "SKIPPED: linux-sandbox only exists on Linux" | tee -a "$TEST_log"
    return 0
  fi
  run_consumer_test linux-sandbox
}

function test_source_input_is_still_staged_normally() {
  bazel build --spawn_strategy=sandboxed //pkg:with_source &> "$TEST_log" || fail "build failed"
  local report=bazel-bin/pkg/with_source_report.txt
  assert_copies_are_regular_files "$report"
  grep -q "with_source_gen.txt .* generated" "$report" || fail "source not visible"
}

function test_mark_survives_noop_rebuild_and_consumer_only_change() {
  bazel build --spawn_strategy=sandboxed //pkg:plain &> "$TEST_log" || fail "build failed"
  assert_copies_are_regular_files bazel-bin/pkg/plain_report.txt

  # No-op: all actions are up to date.
  bazel build --spawn_strategy=sandboxed //pkg:plain &> "$TEST_log" || fail "rebuild failed"
  expect_log "up-to-date"

  # Only the consumer re-runs; the copy actions are action cache or skyframe hits.
  set_salt one
  bazel build --spawn_strategy=sandboxed //pkg:plain &> "$TEST_log" || fail "build 2 failed"
  assert_copies_are_regular_files bazel-bin/pkg/plain_report.txt
}

function test_mark_survives_server_restart() {
  bazel build --spawn_strategy=sandboxed //pkg:plain &> "$TEST_log" || fail "build failed"
  bazel shutdown &> "$TEST_log" || fail "shutdown failed"
  set_salt two
  bazel build --spawn_strategy=sandboxed //pkg:plain &> "$TEST_log" || fail "build 2 failed"
  assert_copies_are_regular_files bazel-bin/pkg/plain_report.txt
}

# Returns the staged inode that the consumer of //pkg:with_source recorded.
function gen_inode() {
  grep "^gen_inode=" bazel-bin/pkg/with_source_report.txt || fail "no gen_inode in report"
}

function test_mark_survives_sandbox_reuse() {
  local flags="--spawn_strategy=sandboxed --reuse_sandbox_directories"
  local target=//pkg:with_source
  bazel build $flags $target &> "$TEST_log" || fail "build failed"
  assert_copies_are_regular_files bazel-bin/pkg/with_source_report.txt
  local first_inode
  first_inode="$(gen_inode)"

  # Only the consumer's command line changes, so it re-runs in a reused sandbox directory. The
  # plain (non-copy) input is a staged symlink that a reused sandbox keeps as is, so its inode
  # staying the same proves that a stash was reused, while the copy inputs must still be regular
  # files (they are always re-created).
  set_salt three
  bazel build $flags $target &> "$TEST_log" || fail "build 2 failed"
  assert_copies_are_regular_files bazel-bin/pkg/with_source_report.txt
  [[ "$(gen_inode)" == "$first_inode" ]] \
    || fail "sandbox was not reused (symlink inode changed from $first_inode to $(gen_inode))"

  set_salt four
  bazel build $flags --experimental_inmemory_sandbox_stashes $target &> "$TEST_log" \
    || fail "build 3 failed"
  assert_copies_are_regular_files bazel-bin/pkg/with_source_report.txt
  local third_inode
  third_inode="$(gen_inode)"
  set_salt five
  bazel build $flags --experimental_inmemory_sandbox_stashes $target &> "$TEST_log" \
    || fail "build 4 failed"
  assert_copies_are_regular_files bazel-bin/pkg/with_source_report.txt
  [[ "$(gen_inode)" == "$third_inode" ]] \
    || fail "in-memory stash was not reused (symlink inode changed from $third_inode)"
}

run_suite "Tests that ctx.actions.copy outputs are never staged as symlinks"
