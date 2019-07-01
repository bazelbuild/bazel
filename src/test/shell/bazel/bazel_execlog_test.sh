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
#
# Tests verbosity behavior on workspace initialization

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_dir_depends() {
  create_new_workspace
  cat > skylark.bzl <<'EOF'
def _dir_impl(ctx):
  output_dir = ctx.actions.declare_directory(ctx.attr.outdir)
  ctx.actions.run_shell(
    outputs = [output_dir],
    inputs = [],
    command = """
      mkdir -p $1/sub; \
      echo "first file!" > $1/foo; \
      echo "second file." > $1/sub/bar; \
    """,
    arguments = [output_dir.path],
  )
  return [ DefaultInfo(files = depset(direct = [output_dir])) ]

gen_dir = rule(
  implementation = _dir_impl,
  attrs = {
       "outdir": attr.string(mandatory = True),
  }
)
EOF
  cat > BUILD <<'EOF'
load(":skylark.bzl", "gen_dir")

gen_dir(
  name = "dir",
  outdir = "dir_name1",
)

genrule(
    name = "rule1",
    srcs = ["dir"],
    outs = ["combined1.txt"],
    cmd = "cat $(location dir)/foo $(location dir)/sub/bar > $(location combined1.txt)"
)

gen_dir(
  name = "dir2",
  outdir = "dir_name2",
)

genrule(
    name = "rule2",
    srcs = ["dir2"],
    outs = ["combined2.txt"],
    cmd = "cat $(location dir2)/foo $(location dir2)/sub/bar > $(location combined2.txt)"
)
EOF

  bazel build //:all --execution_log_json_file output.json 2>&1 >> $TEST_log || fail "could not build"

  # dir and dir2 are skylark functions that create a directory output
  # rule1 and rule2 are functions that consume the directory output
  #
  # the output files are named such that the rule's would be placed first in the
  # stable sorting order, followed by the dir's.
  # Since rule1 depends on dir1 and rule2 depends on dir2, we expect the
  # following order:
  # dir1, rule1, dir2, rule2
  #
  # If dependencies were not properly accounted for, the order would have been:
  # rule1, rule2, dir1, dir2

  dir1Num=`grep "Action dir_name1" -n output.json | grep -Eo '^[^:]+'`
  dir2Num=`grep "Action dir_name2" -n output.json | grep -Eo '^[^:]+'`
  rule1Num=`grep "Executing genrule //:rule1" -n output.json | grep -Eo '^[^:]+'`
  rule2Num=`grep "Executing genrule //:rule2" -n output.json | grep -Eo '^[^:]+'`

  if [ "$rule1Num" -lt "$dir1Num" ]
  then
    fail "rule1 dependency on dir1 is not recornized"
  fi

  if [ "$dir2Num" -lt "$rule1Num" ]
  then
    fail "rule1 comes after dir2"
  fi

  if [ "$rule2Num" -lt "$dir2Num" ]
  then
    fail "rule2 dependency on dir2 is not recornized"
  fi
}

function test_dir_relative() {
  cat > BUILD <<'EOF'
genrule(
      name = "rule",
      outs = ["out.txt"],
      cmd = "echo hello > $(location out.txt)"
)
EOF
  bazel build //:all --experimental_execution_log_file output 2>&1 >> $TEST_log || fail "could not build"
  wc output || fail "no output produced"
}

function test_negating_flags() {
  cat > BUILD <<'EOF'
genrule(
      name = "rule",
      outs = ["out.txt"],
      cmd = "echo hello > $(location out.txt)"
)
EOF
  bazel build //:all --experimental_execution_log_file=output --experimental_execution_log_file= 2>&1 >> $TEST_log || fail "could not build"
  if [[ -e output ]]; then
    fail "file shouldn't exist"
  fi

  bazel build //:all --execution_log_json_file=output --execution_log_json_file= 2>&1 >> $TEST_log || fail "could not build"
  if [[ -e output ]]; then
    fail "file shouldn't exist"
  fi

  bazel build //:all --execution_log_binary_file=output --execution_log_binary_file= 2>&1 >> $TEST_log || fail "could not build"
  if [[ -e output ]]; then
    fail "file shouldn't exist"
  fi
}

function test_no_output() {
  create_new_workspace
  cat > skylark.bzl <<'EOF'
def _impl(ctx):
  ctx.actions.write(ctx.outputs.executable, content="echo hello world", is_executable=True)
  return DefaultInfo()

my_test = rule(
  implementation = _impl,
  test = True,
)
EOF

  cat > BUILD <<'EOF'
load(":skylark.bzl", "my_test")

my_test(
  name = "little_test",
)
EOF

  bazel test //:little_test --execution_log_json_file output.json 2>&1 >> $TEST_log || fail "could not test"
  grep "listedOutputs" output.json || fail "log does not contain listed outputs"
}

run_suite "execlog_tests"
