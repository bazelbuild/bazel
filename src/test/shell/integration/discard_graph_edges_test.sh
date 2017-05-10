#!/bin/bash
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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
# discard_graph_edges_test.sh: basic tests for the --discard_graph_edges flag.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

#### SETUP #############################################################

set -e

function set_up() {
  mkdir -p testing || fail "Couldn't create directory"
  echo "cc_test(name='mytest', srcs=['mytest.cc'], malloc=':system_malloc')" > testing/BUILD || fail
  echo "cc_library(name='system_malloc', srcs=[])"                           >> testing/BUILD || fail
  echo "int main() {return 0;}"         > testing/mytest.cc || fail
}

STARTUP_FLAGS="--batch"
BUILD_FLAGS="--keep_going --discard_analysis_cache"

#### TESTS #############################################################

function test_build() {
  bazel $STARTUP_FLAGS build $BUILD_FLAGS //testing:mytest >& $TEST_log \
    || fail "Expected success"
}

function test_test() {
  bazel $STARTUP_FLAGS test $BUILD_FLAGS //testing:mytest >& $TEST_log \
    || fail "Expected success"
}

# bazel info inherits from bazel build, but it doesn't have much in common with it.
function test_info() {
  bazel $STARTUP_FLAGS info $BUILD_FLAGS >& $TEST_log || fail "Expected success"
}

function test_empty_build() {
  bazel $STARTUP_FLAGS build $BUILD_FLAGS >& $TEST_log || fail "Expected success"
}

function test_query() {
  bazel $STARTUP_FLAGS query 'somepath(//testing:mytest,//testing:system_malloc)' >& $TEST_log \
    || fail "Expected success"
  expect_log "//testing:mytest"
  expect_log "//testing:system_malloc"
}

function test_top_level_aspect() {
  mkdir -p "foo" || fail "Couldn't make directory"
  cat > foo/simpleaspect.bzl <<'EOF' || fail "Couldn't write bzl file"
def _simple_aspect_impl(target, ctx):
  result=depset()
  for orig_out in target.files:
    aspect_out = ctx.new_file(orig_out.basename + ".aspect")
    ctx.file_action(
        output=aspect_out,
        content = "Hello from aspect for %s" % orig_out.basename)
    result += [aspect_out]
  for src in ctx.rule.attr.srcs:
    result += src.aspectouts

  return struct(output_groups={
      "aspect-out" : result }, aspectouts = result)

simple_aspect = aspect(implementation=_simple_aspect_impl,
                       attr_aspects = ["srcs"])

def _rule_impl(ctx):
  output = ctx.outputs.out
  ctx.action(
      inputs=[],
      outputs=[output],
      progress_message="Touching output %s" % output,
      command="touch %s" % output.path)

simple_rule = rule(
    implementation =_rule_impl,
    attrs = {"srcs": attr.label_list(aspects=[simple_aspect])},
    outputs={"out": "%{name}.out"}
    )
EOF

cat > foo/BUILD <<'EOF' || fail "Couldn't write BUILD file"
load("//foo:simpleaspect.bzl", "simple_rule")

simple_rule(name = "foo", srcs = [":dep"])
simple_rule(name = "dep", srcs = [])
EOF
  bazel $STARTUP_FLAGS build $BUILD_FLAGS //foo:foo >& "$TEST_log" \
      || fail "Expected success"
  bazel --batch clean >& "$TEST_log" || fail "Expected success"
  bazel $STARTUP_FLAGS build $BUILD_FLAGS \
      --aspects foo/simpleaspect.bzl%simple_aspect \
      --output_groups=aspect-out //foo:foo >& "$TEST_log" \
      || fail "Expected success"
  [[ -e "bazel-bin/foo/foo.out.aspect" ]] || fail "Aspect foo not run"
  [[ -e "bazel-bin/foo/dep.out.aspect" ]] || fail "Aspect bar not run"
}

function extract_histogram_count() {
  local histofile="$1"
  local item="$2"
  # We can't use + here because Macs don't recognize it as a special character by default.
  grep "$item" "$histofile" | sed -e 's/^ *[0-9][0-9]*: *\([0-9][0-9]*\) .*$/\1/' \
      || fail "Couldn't get item from $histofile"
}

function prepare_histogram() {
  readonly local build_args="$1"
  rm -rf histodump
  mkdir -p histodump || fail "Couldn't create directory"
  readonly local server_pid_fifo="$TEST_TMPDIR/server_pid"
cat > histodump/foo.bzl <<'EOF' || fail "Couldn't create bzl file"
def foo():
  pass
EOF
cat > histodump/bar.bzl <<'EOF' || fail "Couldn't create bzl file"
def bar():
  pass
EOF
cat > histodump/baz.bzl <<'EOF' || fail "Couldn't create bzl file"
def baz():
  pass
EOF

  cat > histodump/BUILD <<EOF || fail "Couldn't create BUILD file"
load(":foo.bzl", "foo")
load(":bar.bzl", "bar")
load(":baz.bzl", "baz")
genrule(name = 'histodump',
        srcs = glob(["*.in"]),
        outs = ['histo.txt'],
        local = 1,
        cmd = '${bazel_javabase}/bin/jmap -histo:live \$\$(cat $server_pid_fifo) > \$(location histo.txt)'
       )
EOF
  rm -f "$server_pid_fifo"
  mkfifo "$server_pid_fifo"
  histo_file="$(bazel info "${PRODUCT_NAME}-genfiles" \
      2> /dev/null)/histodump/histo.txt"
  bazel clean >& "$TEST_log" || fail "Couldn't clean"
  bazel $STARTUP_FLAGS build $build_args //histodump:histodump >& "$TEST_log" &
  server_pid=$!
  echo "$server_pid" > "$server_pid_fifo"
  # Wait for previous command to finish.
  wait "$server_pid" || fail "Bazel command failed"
  cat "$histo_file" >> "$TEST_log"
  echo "$histo_file"
}

function test_packages_cleared() {
  local histo_file="$(prepare_histogram "--nodiscard_analysis_cache")"
  local package_count="$(extract_histogram_count "$histo_file" \
      'devtools\.build\.lib\..*\.Package$')"
  [[ "$package_count" -ge 9 ]] \
      || fail "package count $package_count too low: did you move/rename the class?"
  local glob_count="$(extract_histogram_count "$histo_file" "GlobValue")"
  [[ "$glob_count" -ge 8 ]] \
      || fail "glob count $glob_count too low: did you move/rename the class?"
  local env_count="$(extract_histogram_count "$histo_file" \
      'Environment\$Extension$')"
  [[ "$env_count" -ge 3 ]] \
      || fail "env extension count $env_count too low: did you move/rename the class?"
  local histo_file="$(prepare_histogram "$BUILD_FLAGS")"
  package_count="$(extract_histogram_count "$histo_file" \
      'devtools\.build\.lib\..*\.Package$')"
  # A few packages aren't cleared.
  [[ "$package_count" -le 8 ]] \
      || fail "package count $package_count too high"
  glob_count="$(extract_histogram_count "$histo_file" "GlobValue")"
  [[ "$glob_count" -le 1 ]] \
      || fail "glob count $glob_count too high"
  env_count="$(extract_histogram_count "$histo_file" \
      'Environment\$Extension$')"
  [[ "$env_count" -le 2 ]] \
      || fail "env extension count $env_count too high"
}

function test_actions_deleted_after_execution() {
  rm -rf histodump
  mkdir -p histodump || fail "Couldn't create directory"
  readonly local wait_fifo="$TEST_TMPDIR/wait_fifo"
  readonly local server_pid_file="$TEST_TMPDIR/server_pid.txt"
  cat > histodump/BUILD <<EOF || fail "Couldn't create BUILD file"
genrule(name = 'action0',
        outs = ['wait.out'],
        local = 1,
        cmd = 'cat $wait_fifo > /dev/null; touch \$@'
        )
EOF
  for i in $(seq 1 3); do
    iminus=$((i-1))
    cat >> histodump/BUILD <<EOF || fail "Couldn't append"
genrule(name = 'action${i}',
        srcs = [':action${iminus}'],
        outs = ['histo.${i}'],
        local = 1,
        cmd = '${bazel_javabase}/bin/jmap -histo:live '
              + '\$\$(cat ${server_pid_file}) > \$(location histo.${i})'
       )
EOF
  done
  mkfifo "$wait_fifo"
  local readonly histo_root="$(bazel info "${PRODUCT_NAME}-genfiles" \
      2> /dev/null)/histodump/histo."
  bazel clean >& "$TEST_log" || fail "Couldn't clean"
  bazel $STARTUP_FLAGS build $BUILD_FLAGS //histodump:action3 >& "$TEST_log" &
  server_pid=$!
  echo "$server_pid" > "$server_pid_file"
  echo "" > "$wait_fifo"
  # Wait for previous command to finish.
  wait "$server_pid" || fail "Bazel command failed"
  local genrule_action_count=100
  for i in $(seq 1 3); do
    local histo_file="$histo_root$i"
    local new_genrule_action_count="$(extract_histogram_count "$histo_file" \
        "GenRuleAction$")"
    if [[ "$new_genrule_action_count" -ge "$genrule_action_count" ]]; then
      cat "$histo_file" >> "$TEST_log"
      fail "Number of genrule actions did not decrease: $new_genrule_action_count vs. $genrule_action_count"
    fi
    if [[ -z "$new_genrule_action_count" ]]; then
      cat "$histo_file" >> "$TEST_log"
      fail "No genrule actions? Class may have been renamed"
    fi
    genrule_action_count="$new_genrule_action_count"
  done
}

# Action conflicts can cause deletion of nodes, and deletion is tricky with no edges.
function test_action_conflict() {
  mkdir -p conflict || fail "Couldn't create directory"

  cat > conflict/conflict_rule.bzl <<EOF || fail "Couldn't write bzl file"
def _create(ctx):
  files_to_build = set(ctx.outputs.outs)
  intemediate_outputs = [ctx.new_file("bar")]
  intermediate_cmd = "cat %s > %s" % (ctx.attr.name, intemediate_outputs[0].path)
  action_cmd = "touch " + list(files_to_build)[0].path
  ctx.action(outputs=list(intemediate_outputs),
             command=intermediate_cmd)
  ctx.action(inputs=list(intemediate_outputs),
             outputs=list(files_to_build),
             command=action_cmd)
  struct(files=files_to_build,
         data_runfiles=ctx.runfiles(transitive_files=files_to_build))

conflict = rule(
    implementation=_create,
    attrs={
        "outs": attr.output_list(mandatory=True),
        },
    )
EOF

  mkdir -p conflict || fail "Couldn't create directory"
  cat > conflict/BUILD <<EOF || fail "Couldn't create BUILD file"
load("//conflict:conflict_rule.bzl", "conflict")

conflict(name='hello', outs=['hello_out'])
conflict(name='goodbye', outs=['goodbye_out'])
genrule(name='foo',
        srcs = ['hello_out', 'goodbye_out'],
        outs = ['foo_out'],
        cmd = 'touch $@')

EOF

  # --nocache_test_results to make log-grepping easier.
  bazel $STARTUP_FLAGS test $BUILD_FLAGS \
      --nocache_test_results //conflict:foo //testing:mytest >& $TEST_log \
      && fail "Expected failure"
  exit_code=$?
  [ $exit_code -eq 1 ] || fail "Wrong exit code: $exit_code"
  expect_log "is generated by these conflicting actions"
  expect_not_log "Graph edges not stored"
  expect_log "mytest *PASSED"
}

function test_remove_actions() {
  bazel "$STARTUP_FLAGS" test $BUILD_FLAGS \
      --noexperimental_enable_critical_path_profiling //testing:mytest \
      >& $TEST_log || fail "Expected success"
}

function test_modules() {
  mkdir -p foo || fail "mkdir failed"
  cat > foo/BUILD <<EOF || fail "BUILD file creation failed"
package(features=['prune_header_modules','header_modules','use_header_modules'])
cc_library(name = 'a', hdrs = ['a.h'])
cc_library(name = 'b', hdrs = ['b.h'], deps = [':a'])
cc_library(name = 'c', deps = [':b'], srcs = ['c.cc'])
EOF
  touch foo/a.h || fail "touch a.h failed"
  touch foo/b.h || fail "touch b.h failed"
  echo '#include "foo/b.h"' > foo/c.cc || fail "c.cc creation failed"

  bazel "$STARTUP_FLAGS" build $BUILD_FLAGS \
      --noexperimental_enable_critical_path_profiling \
      //foo:c --experimental_skip_unused_modules \
    --experimental_prune_more_modules >& "$TEST_log" || fail "Build failed"
}

# The following tests are not expected to exercise codepath -- make sure nothing bad happens.

function test_no_batch() {
  bazel $STARTUP_FLAGS --nobatch test $BUILD_FLAGS //testing:mytest >& $TEST_log \
    || fail "Expected success"
}

function test_no_keep_going() {
  bazel $STARTUP_FLAGS test $BUILD_FLAGS --nokeep_going //testing:mytest >& $TEST_log \
    || fail "Expected success"
}

function test_no_discard_analysis_cache() {
  bazel $STARTUP_FLAGS test $BUILD_FLAGS --nodiscard_analysis_cache //testing:mytest >& $TEST_log \
    || fail "Expected success"
}

run_suite "test for --discard_graph_edges"
