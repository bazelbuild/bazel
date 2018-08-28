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

source "${CURRENT_DIR}/remote_helpers.sh" \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }

function test_execute() {
  create_new_workspace
  cat > BUILD <<'EOF'
genrule(
   name="test",
   srcs=["@repo//:t.txt"],
   outs=["out.txt"],
   cmd="echo Result > $(location out.txt)"
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
  executes=`grep "location: .*repos.bzl:2:3" $TEST_log | wc -l`
  if [ "$executes" -ne "1" ]
  then
    fail "Expected exactly 1 occurrence of the given command, got $executes"
  fi

  # Cached executions are not replayed
  bazel build //:test --experimental_workspace_rules_logging=yes &> output || fail "could not build //:test"
  cat output >> $TEST_log
  executes=`grep "location: .*repos.bzl:2:3" output | wc -l`
  if [ "$executes" -ne "0" ]
  then
    fail "Expected exactly 0 occurrence of the given command, got $executes"
  fi
}

function test_reexecute() {
  create_new_workspace
  cat > BUILD <<'EOF'
genrule(
   name="test",
   srcs=["@repo//:t.txt"],
   outs=["out.txt"],
   cmd="echo Result > $(location out.txt)"
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
  executes=`grep "location: .*repos.bzl:2:3" $TEST_log | wc -l`
  if [ "$executes" -le "2" ]
  then
    fail "Expected at least 2 occurrences of the given command, got $executes"
  fi
}

# Sets up a workspace with the given commands inserted into the repository rule
# that will be executed when doing bazel build //:test
function set_workspace_command() {
  create_new_workspace
  cat > BUILD <<'EOF'
genrule(
   name="test",
   srcs=["@repo//:t.txt"],
   outs=["out.txt"],
   cmd="echo Result > $(location out.txt)"
)
EOF
  cat >> repos.bzl <<EOF
def _executeMe(repository_ctx):
  $1
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
}

# Ensure details of the specific functions are present
function test_execute2() {
  set_workspace_command 'repository_ctx.execute(["echo", "test_contents"], 21, {"Arg1": "Val1"}, True)'

  bazel build //:test --experimental_workspace_rules_logging=yes &> ${TEST_log} || fail "could not build //:test\n"
  expect_log "location: .*repos.bzl:2:3"
  expect_log "arguments: \"echo\""
  expect_log "arguments: \"test_contents\""
  expect_log "timeout_seconds: 21"
  expect_log "quiet: true"
  expect_log "key: \"Arg1\""
  expect_log "value: \"Val1\""
  expect_log "rule: \"//external:repo\""
}


function test_download() {
  # Prepare HTTP server with Python
  local server_dir="${TEST_TMPDIR}/server_dir"
  mkdir -p "${server_dir}"
  local file="${server_dir}/file.txt"
  echo "file contents here" > "${file}"
  file_sha256="$(sha256sum "${file}" | head -c 64)"

  # Start HTTP server with Python
  startup_server "${server_dir}"

  set_workspace_command "repository_ctx.download(\"http://localhost:${fileserver_port}/file.txt\", \"file.txt\", \"${file_sha256}\")"

  bazel build //:test --experimental_workspace_rules_logging=yes &> ${TEST_log} && shutdown_server || fail "could not build //:test\n"
  expect_log "location: .*repos.bzl:2:3"
  expect_log "rule: \"//external:repo\""
  expect_log "download_event"
  expect_log "url: \"http://localhost:${fileserver_port}/file.txt\""
  expect_log "output: \"file.txt\""
  expect_log "sha256: \"${file_sha256}\""
}

function test_download_multiple() {
  # Prepare HTTP server with Python
  local server_dir="${TEST_TMPDIR}/server_dir"
  mkdir -p "${server_dir}"
  local file2="${server_dir}/file2.txt"
  echo "second contents here" > "${file2}"

  # Start HTTP server with Python
  startup_server "${server_dir}"

  set_workspace_command "repository_ctx.download([\"http://localhost:${fileserver_port}/file1.txt\",\"http://localhost:${fileserver_port}/file2.txt\"], \"out_for_list.txt\")"

  bazel build //:test --experimental_workspace_rules_logging=yes &> $TEST_log && shutdown_server || fail "could not build //:test\n"
  expect_log "location: .*repos.bzl:2:3"
  expect_log "rule: \"//external:repo\""
  expect_log "download_event"
  expect_log "url: \"http://localhost:${fileserver_port}/file1.txt\""
  expect_log "url: \"http://localhost:${fileserver_port}/file2.txt\""
  expect_log "output: \"out_for_list.txt\""
}

function test_download_and_extract() {
  # Prepare HTTP server with Python
  local server_dir="${TEST_TMPDIR}/server_dir"
  mkdir -p "${server_dir}"
  local file_prefix="${server_dir}/download_and_extract"

  pushd ${TEST_TMPDIR}
  echo "This is one file" > server_dir/download_and_extract.txt
  zip -r server_dir/download_and_extract.zip server_dir
  file_sha256="$(sha256sum server_dir/download_and_extract.zip | head -c 64)"
  popd

  # Start HTTP server with Python
  startup_server "${server_dir}"

  set_workspace_command "repository_ctx.download_and_extract(\"http://localhost:${fileserver_port}/download_and_extract.zip\", \"out_dir\", \"${file_sha256}\", \"zip\", \"server_dir/\")"

  bazel build //:test --experimental_workspace_rules_logging=yes &> ${TEST_log} && shutdown_server || fail "could not build //:test\n"

  expect_log "location: .*repos.bzl:2:3"
  expect_log "rule: \"//external:repo\""
  expect_log "download_and_extract_event"
  expect_log "url: \"http://localhost:${fileserver_port}/download_and_extract.zip\""
  expect_log "output: \"out_dir\""
  expect_log "sha256: \"${file_sha256}\""
  expect_log "type: \"zip\""
  expect_log "strip_prefix: \"server_dir/\""
}

function test_file() {
  set_workspace_command 'repository_ctx.file("filefile.sh", "echo filefile", True)'

  bazel build //:test --experimental_workspace_rules_logging=yes &> $TEST_log || fail "could not build //:test\n"
  expect_log 'location: .*repos.bzl:2:3'
  expect_log 'rule: "//external:repo"'
  expect_log 'file_event'
  expect_log 'path: ".*filefile.sh"'
  expect_log 'content: "echo filefile"'
  expect_log 'executable: true'
}

function test_os() {
  set_workspace_command 'print(repository_ctx.os.name)'

  bazel build //:test --experimental_workspace_rules_logging=yes &> $TEST_log || fail "could not build //:test\n"
  expect_log 'location: .*repos.bzl:2:9'
  expect_log 'rule: "//external:repo"'
  expect_log 'os_event'
}

function test_symlink() {
  set_workspace_command 'repository_ctx.file("symlink.txt", "something")
  repository_ctx.symlink("symlink.txt", "symlink_out.txt")'

  bazel build //:test --experimental_workspace_rules_logging=yes &> $TEST_log || fail "could not build //:test\n"
  expect_log 'location: .*repos.bzl:3:3'
  expect_log 'rule: "//external:repo"'
  expect_log 'symlink_event'
  expect_log 'from: ".*symlink.txt"'
  expect_log 'to: ".*symlink_out.txt"'
}

function test_template() {
  set_workspace_command 'repository_ctx.file("template_in.txt", "%{subKey}")
  repository_ctx.template("template_out.txt", "template_in.txt", {"subKey": "subVal"}, True)'

  bazel build //:test --experimental_workspace_rules_logging=yes &> $TEST_log || fail "could not build //:test\n"
  expect_log 'location: .*repos.bzl:3:3'
  expect_log 'rule: "//external:repo"'
  expect_log 'template_event'
  expect_log 'path: ".*template_out.txt"'
  expect_log 'template: ".*template_in.txt"'
  expect_log 'key: "subKey"'
  expect_log 'value: "subVal"'
  expect_log 'executable: true'
}

function test_which() {
  set_workspace_command 'print(repository_ctx.which("which_prog"))'

  bazel build //:test --experimental_workspace_rules_logging=yes &> $TEST_log || fail "could not build //:test\n"
  expect_log 'location: .*repos.bzl:2:9'
  expect_log 'rule: "//external:repo"'
  expect_log 'which_event'
  expect_log 'program: "which_prog"'
}

function tear_down() {
  shutdown_server
  if [ -d "${TEST_TMPDIR}/server_dir" ]; then
    rm -fr "${TEST_TMPDIR}/server_dir"
  fi
  true
}

run_suite "workspaces_tests"
