#!/bin/bash
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
# Tests verbosity behavior on workspace initialization

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }


function test_execute() {
  create_new_workspace
  cat > BUILD <<EOF
genrule(
   name="test",
   srcs=["@repo//:t.txt"],
   outs=["out.txt"],
   cmd="echo Result > \$(location out.txt)"
)
EOF
  cat >> repos.bzl <<EOF
def _executeMe(repository_ctx):
  repository_ctx.execute(["echo", "testing!"])
  build_contents = "package(default_visibility = ['//visibility:public'])\n\n"
  build_contents += "exports_files([\"t.txt\"])\n"
  repository_ctx.file("BUILD", build_contents, False)
  repository_ctx.file("t.txt", "HELLO!\n", False)

ex_repo = repository_rule(
  implementation = _executeMe,
  local = True,
)
EOF
  cat >> WORKSPACE <<EOF
load("//:repos.bzl", "ex_repo")
ex_repo(name = "repo")
EOF

  bazel build //:test --experimental_workspace_rules_logging=yes &> $TEST_log || fail "could not build //:test"
  executes=`grep "repos.bzl:2:3: Executing a command." $TEST_log | wc -l`
  if [ "$executes" -ne "1" ]
  then
    fail "Expected exactly 1 occurrence of the given command, got $executes"
  fi

  # Cached executions are not replayed
  bazel build //:test --experimental_workspace_rules_logging=yes &> output || fail "could not build //:test"
  cat output &> $TEST_log
  executes=`grep "repos.bzl:2:3: Executing a command." output | wc -l`
  if [ "$executes" -ne "0" ]
  then
    fail "Expected exactly 0 occurrence of the given command, got $executes"
  fi
}

function test_reexecute() {
  create_new_workspace
  cat > BUILD <<EOF
genrule(
   name="test",
   srcs=["@repo//:t.txt"],
   outs=["out.txt"],
   cmd="echo Result > \$(location out.txt)"
)
EOF
  cat >> repos.bzl <<EOF
def _executeMe(repository_ctx):
  repository_ctx.execute(["echo", "testing!"])
  build_contents = "package(default_visibility = ['//visibility:public'])\n\n"
  build_contents += "exports_files([\"t.txt\"])\n"
  repository_ctx.file("BUILD", build_contents, False)
  repository_ctx.symlink(Label("@another//:dummy.txt"), "t.txt")

ex_repo = repository_rule(
  implementation = _executeMe,
  local = True,
)

def _another(repository_ctx):
  build_contents = "exports_files([\"dummy.txt\"])\n"
  repository_ctx.file("BUILD", build_contents, False)
  repository_ctx.file("dummy.txt", "dummy\n", False)

a_repo = repository_rule(
  implementation = _another,
  local = True,
)
EOF
  cat >> WORKSPACE <<EOF
load("//:repos.bzl", "ex_repo")
load("//:repos.bzl", "a_repo")
ex_repo(name = "repo")
a_repo(name = "another")
EOF

  bazel build //:test --experimental_workspace_rules_logging=yes &> $TEST_log || fail "could not build //:test"
  executes=`grep "repos.bzl:2:3: Executing a command." $TEST_log | wc -l`
  if [ "$executes" -le "2" ]
  then
    fail "Expected at least 2 occurrences of the given command, got $executes"
  fi
}

run_suite "workspaces_tests"
