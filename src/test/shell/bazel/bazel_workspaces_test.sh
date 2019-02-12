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

function build_and_process_log() {
  bazel build //:test --experimental_workspace_rules_log_file=output 2>&1 >> $TEST_log || fail "could not build //:test"
  ${BAZEL_RUNFILES}/src/tools/workspacelog/parser --log_path=output > output.log.txt "$@" || fail "error parsing output"
}

function ensure_contains_exactly() {
  num=`grep "${1}" output.log.txt | wc -l`
  if [ "$num" -ne $2 ]
  then
    fail "Expected exactly $2 occurrences of $1, got $num: " `cat output.log.txt`
  fi
}

function ensure_contains_atleast() {
  num=`grep "${1}" output.log.txt | wc -l`
  if [ "$num" -lt $2 ]
  then
    fail "Expected at least $2 occurrences of $1, got $num: " `cat output.log.txt`
  fi
}

function ensure_output_contains_exactly() {
  file_path=$(bazel info output_base)/$1
  num=`grep "$2" $file_path | wc -l`
  if [ "$num" -ne 1 ]
  then
    fail "Expected to read \"$2\" in $1, but got $num occurrences: " `cat $file_path`
  fi
}

function test_execute() {
  set_workspace_command 'repository_ctx.execute(["echo", "testing!"])'
  build_and_process_log

  ensure_contains_exactly "location: .*repos.bzl:2:3" 1

  # Cached executions are not replayed
  build_and_process_log
  ensure_contains_exactly "location: .*repos.bzl:2:3" 0
}

# The workspace is set up so that the function is interrupted and re-executed.
# The log should contain both instances.
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

  build_and_process_log

  ensure_contains_atleast "location: .*repos.bzl:2:3" 2
}


# Ensure details of the specific functions are present
function test_execute2() {
  set_workspace_command 'repository_ctx.execute(["echo", "test_contents"], 21, {"Arg1": "Val1"}, True)'

  build_and_process_log --exclude_rule "//external:local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:3' 1
  ensure_contains_exactly 'arguments: "echo"' 1
  ensure_contains_exactly 'arguments: "test_contents"' 1
  ensure_contains_exactly 'timeout_seconds: 21' 1
  ensure_contains_exactly 'quiet: true' 1
  ensure_contains_exactly 'key: "Arg1"' 1
  ensure_contains_exactly 'value: "Val1"' 1
  # Workspace contains 2 file commands
  ensure_contains_atleast 'rule: "//external:repo"' 3
}

function test_execute_quiet2() {
  set_workspace_command 'repository_ctx.execute(["echo", "test2"], 32, {"A1": "V1"}, False)'

  build_and_process_log --exclude_rule "//external:local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:3' 1
  ensure_contains_exactly 'arguments: "echo"' 1
  ensure_contains_exactly 'arguments: "test2"' 1
  ensure_contains_exactly 'timeout_seconds: 32' 1
  # quiet: false does not show up when printing protos
  # since it's the default value
  ensure_contains_exactly 'quiet: ' 0
  ensure_contains_exactly 'key: "A1"' 1
  ensure_contains_exactly 'value: "V1"' 1
  # Workspace contains 2 file commands
  ensure_contains_atleast 'rule: "//external:repo"' 3
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

  build_and_process_log --exclude_rule "//external:local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:3' 1
  ensure_contains_atleast 'rule: "//external:repo"' 1
  ensure_contains_exactly 'download_event' 1
  ensure_contains_exactly "url: \"http://localhost:${fileserver_port}/file.txt\"" 1
  ensure_contains_exactly 'output: "file.txt"' 1
  ensure_contains_exactly "sha256: \"${file_sha256}\"" 1
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

  build_and_process_log --exclude_rule "//external:local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:3' 1
  ensure_contains_atleast 'rule: "//external:repo"' 1
  ensure_contains_exactly 'download_event' 1
  ensure_contains_exactly "url: \"http://localhost:${fileserver_port}/file1.txt\"" 1
  ensure_contains_exactly "url: \"http://localhost:${fileserver_port}/file2.txt\"" 1
  ensure_contains_exactly 'output: "out_for_list.txt"' 1
}

function test_download_then_extract() {
  # Prepare HTTP server with Python
  local server_dir="${TEST_TMPDIR}/server_dir"
  mkdir -p "${server_dir}"
  local file_prefix="${server_dir}/download_then_extract"

  pushd ${TEST_TMPDIR}
  echo "This is one file" > server_dir/download_then_extract.txt
  zip -r server_dir/download_then_extract.zip server_dir
  file_sha256="$(sha256sum server_dir/download_then_extract.zip | head -c 64)"
  popd

  # Start HTTP server with Python
  startup_server "${server_dir}"

  set_workspace_command "
  repository_ctx.download(\"http://localhost:${fileserver_port}/download_then_extract.zip\", \"downloaded_file.zip\", \"${file_sha256}\")
  repository_ctx.extract(\"downloaded_file.zip\", \"out_dir\", \"server_dir/\")"

  build_and_process_log --exclude_rule "//external:local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:3:3' 1
  ensure_contains_exactly 'location: .*repos.bzl:4:3' 1
  ensure_contains_atleast 'rule: "//external:repo"' 2
  ensure_contains_exactly 'download_event' 1
  ensure_contains_exactly "url: \"http://localhost:${fileserver_port}/download_then_extract.zip\"" 1
  ensure_contains_exactly 'output: "downloaded_file.zip"' 1
  ensure_contains_exactly "sha256: \"${file_sha256}\"" 1
  ensure_contains_exactly 'extract_event' 1
  ensure_contains_exactly 'archive: "downloaded_file.zip"' 1
  ensure_contains_exactly 'output: "out_dir"' 1
  ensure_contains_exactly 'strip_prefix: "server_dir/"' 1

  ensure_output_contains_exactly "external/repo/out_dir/download_and_extract.txt" "This is one file"
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

  build_and_process_log --exclude_rule "//external:local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:3' 1
  ensure_contains_atleast 'rule: "//external:repo"' 1
  ensure_contains_exactly 'download_and_extract_event' 1
  ensure_contains_exactly "url: \"http://localhost:${fileserver_port}/download_and_extract.zip\"" 1
  ensure_contains_exactly 'output: "out_dir"' 1
  ensure_contains_exactly "sha256: \"${file_sha256}\"" 1
  ensure_contains_exactly 'type: "zip"' 1
  ensure_contains_exactly 'strip_prefix: "server_dir/"' 1

  ensure_output_contains_exactly "external/repo/out_dir/download_and_extract.txt" "This is one file"
}

function test_file() {
  set_workspace_command 'repository_ctx.file("filefile.sh", "echo filefile", True)'

  build_and_process_log --exclude_rule "//external:local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:3' 1
  ensure_contains_atleast 'rule: "//external:repo"' 1

  # There are 3 file_event in external:repo as it is currently set up
  ensure_contains_exactly 'file_event' 3
  ensure_contains_exactly 'path: ".*filefile.sh"' 1
  ensure_contains_exactly 'content: "echo filefile"' 1
  ensure_contains_exactly 'executable: true' 1
}

function test_os() {
  set_workspace_command 'print(repository_ctx.os.name)'

  build_and_process_log --exclude_rule "//external:local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:9' 1
  ensure_contains_atleast 'rule: "//external:repo"' 1
  ensure_contains_exactly 'os_event' 1
}

function test_symlink() {
  set_workspace_command 'repository_ctx.file("symlink.txt", "something")
  repository_ctx.symlink("symlink.txt", "symlink_out.txt")'

  build_and_process_log --exclude_rule "//external:local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:3:3' 1
  ensure_contains_atleast 'rule: "//external:repo"' 1
  ensure_contains_exactly 'symlink_event' 1
  ensure_contains_exactly 'target: ".*symlink.txt"' 1
  ensure_contains_exactly 'path: ".*symlink_out.txt"' 1
}

function test_template() {
  set_workspace_command 'repository_ctx.file("template_in.txt", "%{subKey}", False)
  repository_ctx.template("template_out.txt", "template_in.txt", {"subKey": "subVal"}, True)'

  build_and_process_log --exclude_rule "//external:local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:3:3' 1
  ensure_contains_atleast 'rule: "//external:repo"' 1
  ensure_contains_exactly 'template_event' 1
  ensure_contains_exactly 'path: ".*template_out.txt"' 1
  ensure_contains_exactly 'template: ".*template_in.txt"' 1
  ensure_contains_exactly 'key: "subKey"' 1
  ensure_contains_exactly 'value: "subVal"' 1
  ensure_contains_exactly 'executable: true' 1
}

function test_which() {
  set_workspace_command 'print(repository_ctx.which("which_prog"))'

  build_and_process_log --exclude_rule "//external:local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:9' 1
  ensure_contains_atleast 'rule: "//external:repo"' 1
  ensure_contains_exactly 'which_event' 1
  ensure_contains_exactly 'program: "which_prog"' 1
}

function tear_down() {
  shutdown_server
  if [ -d "${TEST_TMPDIR}/server_dir" ]; then
    rm -fr "${TEST_TMPDIR}/server_dir"
  fi
  true
}

run_suite "workspaces_tests"

