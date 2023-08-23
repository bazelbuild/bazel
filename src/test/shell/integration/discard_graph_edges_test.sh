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
source "$(rlocation "io_bazel/src/test/shell/integration/discard_graph_edges_lib.sh")" \
  || { echo "discard_graph_edges_lib.sh not found!" >&2; exit 1; }

IS_WINDOWS=false
case "$(uname | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  IS_WINDOWS=true
esac

if "$IS_WINDOWS"; then
  EXE_EXT=".exe"
else
  EXE_EXT=""
fi

javabase="$1"
if [[ $javabase = external/* ]]; then
  javabase=${javabase#external/}
fi
jmaptool="$(rlocation "${javabase}/bin/jmap${EXE_EXT}")"

#### SETUP #############################################################

set -e

function set_up() {
  mkdir -p testing || fail "Couldn't create directory"
  echo "cc_test(name='mytest', srcs=['mytest.cc'], malloc=':system_malloc')" > testing/BUILD || fail
  echo "cc_library(name='system_malloc', srcs=[])"                           >> testing/BUILD || fail
  echo "int main() {return 0;}"         > testing/mytest.cc || fail
}

#### TESTS #############################################################

function test_build() {
  bazel $STARTUP_FLAGS build $BUILD_FLAGS //testing:mytest >& $TEST_log \
    || fail "Expected success"
}

function test_test() {
  bazel $STARTUP_FLAGS test $BUILD_FLAGS //testing:mytest >& $TEST_log \
    || fail "Expected success"
}

function test_failed_build() {
  mkdir -p foo || fail "Couldn't make directory"
  cat > foo/BUILD <<'EOF' || fail "Couldn't make BUILD file"
cc_library(name = 'foo', srcs = ['foo.cc'], deps = [':bar'])
cc_library(name = 'bar', srcs = ['bar.cc'])
EOF
  touch foo/foo.cc || fail "Couldn't make foo.cc"
  echo "#ERROR" > foo/bar.cc || fail "Couldn't make bar.cc"
  bazel $STARTUP_FLAGS build $BUILD_FLAGS //foo:foo >& "$TEST_log" \
      && fail "Expected failure"
  exit_code=$?
  [ $exit_code -eq 1 ] || fail "Wrong exit code: $exit_code"
  expect_log "#ERROR"
  expect_not_log "Graph edges not stored"
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

function test_configured_query() {
  bazel $STARTUP_FLAGS build $BUILD_FLAGS --nobuild \
      --experimental_post_build_query='deps(//testing:mytest, 1)' \
      //testing:mytest >& "$TEST_log" && fail "Expected failure"
  exit_code="$?"
  [[ "$exit_code" == 2 ]] || fail "Expected exit code 2 but was $exit_code"
}

function test_top_level_aspect() {
  mkdir -p "foo" || fail "Couldn't make directory"
  cat > foo/simpleaspect.bzl <<'EOF' || fail "Couldn't write bzl file"
def _simple_aspect_impl(target, ctx):
  result=[]
  for orig_out in target.files.to_list():
    aspect_out = ctx.actions.declare_file(orig_out.basename + ".aspect")
    ctx.actions.write(
        output=aspect_out,
        content = "Hello from aspect for %s" % orig_out.basename)
    result += [aspect_out]

  result = depset(result,
      transitive = [src.aspectouts for src in ctx.rule.attr.srcs])

  return struct(output_groups={
      "aspect-out" : result }, aspectouts = result)

simple_aspect = aspect(implementation=_simple_aspect_impl,
                       attr_aspects = ["srcs"])

def _rule_impl(ctx):
  output = ctx.outputs.out
  ctx.actions.run_shell(
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
cc_library(name = 'cclib', srcs = ['cclib.cc'])
genrule(name = 'histodump',
        outs = ['histo.txt'],
        local = 1,
        tools = [':cclib'],
        cmd = 'server_pid=\$\$(cat $server_pid_fifo) ; ' +
              '${jmaptool} -histo:live \$\$server_pid > ' +
              '\$(location histo.txt) ' +
              '|| echo "server_pid in genrule: \$\$server_pid"'
       )
EOF

  touch histodump/cclib.cc
  rm -f "$server_pid_fifo"
  mkfifo "$server_pid_fifo"
  histo_file="$(bazel info "${PRODUCT_NAME}-genfiles" \
      2> /dev/null)/histodump/histo.txt"
  bazel clean >& "$TEST_log" || fail "Couldn't clean"
  readonly local explicit_server_pid="$(bazel $STARTUP_FLAGS info server_pid)"
  bazel $STARTUP_FLAGS build --show_timestamps $build_args \
      //histodump:histodump >> "$TEST_log" 2>&1 &
  readonly local subshell_pid=$!
  # We plan to remove batch mode from the relevant flags for discarding
  # incrementality state. In the interim, tests that are not in batch mode
  # explicitly pass --nobatch, so we can use it as a signal.
  if [[ "$STARTUP_FLAGS" =~ "--nobatch" ]]; then
    server_pid="$explicit_server_pid"
  else
    server_pid="$subshell_pid"
  fi
  echo "server_pid in main thread is ${server_pid}" >> "$TEST_log"
  echo "$server_pid" > "$server_pid_fifo"
  echo "Finished writing pid to fifo at " >> "$TEST_log"
  date >> "$TEST_log"
  # Wait for previous command to finish.
  wait "$subshell_pid" || fail "Bazel command failed"
  cat "$histo_file" >> "$TEST_log"
  echo "$histo_file"
}

# TODO(b/62450749): This is flaky on CI.
function test_packages_cleared() {
  local histo_file="$(prepare_histogram "--nodiscard_analysis_cache")"
  local package_count="$(extract_histogram_count "$histo_file" \
      'devtools\.build\.lib\..*\.Package$')"
  [[ "$package_count" -ge 9 ]] \
      || fail "package count $package_count too low: did you move/rename the class?"
  local glob_count="$(extract_histogram_count "$histo_file" "GlobValue$")"
  [[ "$glob_count" -ge 2 ]] \
      || fail "glob count $glob_count too low: did you move/rename the class?"
  local module_count="$(extract_histogram_count "$histo_file" 'eval.Module$')"
  [[ "$module_count" -gt 25 ]] \
      || fail "Module count $module_count too low: was the class renamed/moved?" # was 74
  local ct_count="$(extract_histogram_count "$histo_file" \
       'RuleConfiguredTarget$')"
  [[ "$ct_count" -ge 18 ]] \
      || fail "RuleConfiguredTarget count $ct_count too low: did you move/rename the class?"
  local non_incremental_entry_count="$(extract_histogram_count "$histo_file" \
       '\.NonIncrementalInMemoryNodeEntry$')"
  [[ "$non_incremental_entry_count" -eq 0 ]] \
      || fail "$non_incremental_entry_count NonIncrementalInMemoryNodeEntry instances found in build keeping edges"
  local incremental_entry_count="$(extract_histogram_count "$histo_file" \
       '\.IncrementalInMemoryNodeEntry$')"
  [[ "$incremental_entry_count" -ge 100 ]] \
      || fail "Only $incremental_entry_count IncrementalInMemoryNodeEntry instances found in build keeping edges"
  local histo_file="$(prepare_histogram "$BUILD_FLAGS")"
  package_count="$(extract_histogram_count "$histo_file" \
      'devtools\.build\.lib\..*\.Package$')"
  # A few packages aren't cleared.
  [[ "$package_count" -le 22 ]] \
      || fail "package count $package_count too high"
  glob_count="$(extract_histogram_count "$histo_file" "GlobValue$")"
  [[ "$glob_count" -le 1 ]] \
      || fail "glob count $glob_count too high"
  module_count="$(extract_histogram_count "$histo_file" 'eval.Module$')"
  [[ "$module_count" -lt 190 ]] \
      || fail "Module count $module_count too high"
  ct_count="$(extract_histogram_count "$histo_file" \
       'RuleConfiguredTarget$')"
  [[ "$ct_count" -le 1 ]] \
      || fail "too many RuleConfiguredTarget: expected at most 1, got $ct_count"
  non_incremental_entry_count="$(extract_histogram_count "$histo_file" \
       '\.NonIncrementalInMemoryNodeEntry$')"
  [[ "$non_incremental_entry_count" -ge 100 ]] \
      || fail "Not enough ($non_incremental_entry_count) NonIncrementalInMemoryNodeEntry instances found in build discarding edges"
  incremental_entry_count="$(extract_histogram_count "$histo_file" \
       '\.IncrementalInMemoryNodeEntry$')"
  [[ "$incremental_entry_count" -le 10 ]] \
      || fail "Too many ($incremental_entry_count) IncrementalInMemoryNodeEntry instances found in build discarding edges"
}

# Action conflicts can cause deletion of nodes, and deletion is tricky with no edges.
function test_action_conflict() {
  mkdir -p conflict || fail "Couldn't create directory"

  cat > conflict/conflict_rule.bzl <<EOF || fail "Couldn't write bzl file"
def _create(ctx):
  files_to_build = depset(ctx.outputs.outs)
  intermediate_outputs = [ctx.actions.declare_file("bar")]
  intermediate_cmd = "cat %s > %s" % (ctx.attr.name, intermediate_outputs[0].path)
  action_cmd = "touch " + files_to_build.to_list()[0].path
  ctx.actions.run_shell(outputs=list(intermediate_outputs),
                        command=intermediate_cmd)
  ctx.actions.run_shell(inputs=list(intermediate_outputs),
                        outputs=files_to_build.to_list(),
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
  bazel $STARTUP_FLAGS test --keep_going $BUILD_FLAGS \
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
      //foo:c >& "$TEST_log" || fail "Build failed"
}

# The following tests are not expected to exercise codepath -- make sure nothing bad happens.

function test_no_batch() {
  bazel $STARTUP_FLAGS --nobatch test $BUILD_FLAGS --track_incremental_state \
      //testing:mytest >& "$TEST_log" || fail "Expected success"
}

function test_no_discard_analysis_cache() {
  bazel $STARTUP_FLAGS test $BUILD_FLAGS --nodiscard_analysis_cache \
      --track_incremental_state //testing:mytest >& "$TEST_log" \
      || fail "Expected success"
}

function test_packages_cleared_nobatch() {
  readonly local old_startup_flags="$STARTUP_FLAGS"
  STARTUP_FLAGS="--nobatch"
  readonly local old_build_flags="$BUILD_FLAGS"
  BUILD_FLAGS="--notrack_incremental_state --discard_analysis_cache"
  test_packages_cleared
  STARTUP_FLAGS="$old_startup_flags"
  BUILD_FLAGS="$old_build_flags"
}

function test_packages_cleared_implicit_noincrementality_data() {
  readonly local old_build_flags="$BUILD_FLAGS"
  BUILD_FLAGS="$BUILD_FLAGS --track_incremental_state"
  test_packages_cleared
  BUILD_FLAGS="$old_build_flags"
}

function test_actions_not_deleted_after_execution() {
  mkdir -p foo || fail "Couldn't mkdir"
  cat > foo/BUILD <<'EOF' || fail "Couldn't write file"
genrule(name = "foo", cmd = "touch $@", outs = ["foo.out"])
EOF

  readonly local server_pid="$(bazel info server_pid 2> /dev/null)"
  bazel build $BUILD_FLAGS //foo:foo \
      >& "$TEST_log" || fail "Expected success"
  "$jmaptool" -histo:live $server_pid > histo.txt
  genrule_action_count="$(extract_histogram_count histo.txt \
        'GenRuleAction$')"
  if [[ "$genrule_action_count" -lt 1 ]]; then
    cat histo.txt >> "$TEST_log"
    fail "GenRuleAction unexpectedly not found: $genrule_action_count"
  fi

}

function test_dump_after_discard_incrementality_data() {
  bazel build --notrack_incremental_state //testing:mytest >& "$TEST_log" \
       || fail "Expected success"
  bazel dump --skyframe=deps >& "$TEST_log" || fail "Expected success"
  expect_log "//testing:mytest"
}

function test_query_after_discard_incrementality_data() {
  bazel build --nobuild --notrack_incremental_state //testing:mytest \
       >& "$TEST_log" || fail "Expected success"
  bazel query --experimental_ui_debug_all_events --output=label_kind //testing:mytest \
       >& "$TEST_log" || fail "Expected success"
  expect_log "Loading package: testing"
  expect_log "cc_test rule //testing:mytest"
}

function test_shutdown_after_discard_incrementality_data() {
  readonly local server_pid="$(bazel info server_pid 2> /dev/null)"
  [[ -z "$server_pid" ]] && fail "Couldn't get server pid"
  bazel build --nobuild --notrack_incremental_state //testing:mytest \
       >& "$TEST_log" || fail "Expected success"
  bazel shutdown || fail "Expected success"
  readonly local new_server_pid="$(bazel info server_pid 2> /dev/null)"
  [[ "$server_pid" != "$new_server_pid" ]] \
      || fail "pids $server_pid and $new_server_pid equal"
}

function test_clean_after_discard_incrementality_data() {
  bazel build --nobuild --notrack_incremental_state //testing:mytest \
       >& "$TEST_log" || fail "Expected success"
  bazel clean >& "$TEST_log" || fail "Expected success"
}

function test_switch_back_and_forth() {
  readonly local server_pid="$(bazel info \
      --notrack_incremental_state server_pid 2> /dev/null)"
  [[ -z "$server_pid" ]] && fail "Couldn't get server pid"
  bazel test --experimental_ui_debug_all_events --notrack_incremental_state \
      //testing:mytest >& "$TEST_log" || fail "Expected success"
  expect_log "Loading package: testing"
  bazel test --experimental_ui_debug_all_events --notrack_incremental_state \
      //testing:mytest >& "$TEST_log" || fail "Expected success"
  expect_log "Loading package: testing"
  bazel test --experimental_ui_debug_all_events //testing:mytest >& "$TEST_log" \
      || fail "Expected success"
  expect_log "Loading package: testing"
  bazel test --experimental_ui_debug_all_events //testing:mytest >& "$TEST_log" \
      || fail "Expected success"
  expect_not_log "Loading package: testing"
  bazel test --experimental_ui_debug_all_events --notrack_incremental_state \
      //testing:mytest >& "$TEST_log" || fail "Expected success"
  expect_log "Loading package: testing"
  readonly local new_server_pid="$(bazel info server_pid 2> /dev/null)"
  [[ "$server_pid" == "$new_server_pid" ]] \
      || fail "pids $server_pid and $new_server_pid not equal"
}

function test_warns_on_unexpected_combos() {
  bazel --batch build --nobuild --discard_analysis_cache >& "$TEST_log" \
      || fail "Expected success"
  expect_log "--batch and --discard_analysis_cache specified, but --notrack_incremental_state not specified"
  bazel build --nobuild --discard_analysis_cache --notrack_incremental_state \
      >& "$TEST_log" || fail "Expected success"
  expect_log "--notrack_incremental_state was specified, but without --nokeep_state_after_build."
}

run_suite "test for --discard_graph_edges"
