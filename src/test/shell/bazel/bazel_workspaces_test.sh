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

mock_rules_java_to_avoid_downloading

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
  local -r num=$(grep -c "$1" output.log.txt || true)
  if (( num != $2 )); then
    fail "Expected exactly $2 occurrences of $1, got $num: " `cat output.log.txt`
  fi
}

function ensure_contains_atleast() {
  local -r num=$(grep -c "$1" output.log.txt || true)
  if (( num < $2 )); then
    fail "Expected at least $2 occurrences of $1, got $num: " `cat output.log.txt`
  fi
}

function ensure_output_contains_exactly_once() {
  local -r file_path=$(bazel info output_base)/$1
  local -r num=$(grep -c "$2" $file_path || true)
  if (( num != 1 )); then
    fail "Expected to read \"$2\" in $1, but got $num occurrences: " `cat $file_path`
  fi
}

function test_execute() {
  set_workspace_command 'repository_ctx.execute(["echo", "testing!"])'
  build_and_process_log

  ensure_contains_exactly "location: .*repos.bzl:2:25" 1

  # Cached executions are not replayed
  build_and_process_log
  ensure_contains_exactly "location: .*repos.bzl:2:25" 0
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

  ensure_contains_atleast "location: .*repos.bzl:2:" 2
}


# Ensure details of the specific functions are present
function test_execute2() {
  set_workspace_command 'repository_ctx.execute(["echo", "test_contents"], 21, {"Arg1": "Val1"}, True)'

  build_and_process_log --exclude_rule "repository @local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:25' 1
  ensure_contains_exactly 'arguments: "echo"' 1
  ensure_contains_exactly 'arguments: "test_contents"' 1
  ensure_contains_exactly 'timeout_seconds: 21' 1
  ensure_contains_exactly 'quiet: true' 1
  ensure_contains_exactly 'key: "Arg1"' 1
  ensure_contains_exactly 'value: "Val1"' 1
  # Workspace contains 2 file commands
  ensure_contains_atleast 'context: "repository @repo"' 3
}

function test_execute_quiet2() {
  set_workspace_command 'repository_ctx.execute(["echo", "test2"], 32, {"A1": "V1"}, False)'

  build_and_process_log --exclude_rule "repository @local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:25' 1
  ensure_contains_exactly 'arguments: "echo"' 1
  ensure_contains_exactly 'arguments: "test2"' 1
  ensure_contains_exactly 'timeout_seconds: 32' 1
  # quiet: false does not show up when printing protos
  # since it's the default value
  ensure_contains_exactly 'quiet: ' 0
  ensure_contains_exactly 'key: "A1"' 1
  ensure_contains_exactly 'value: "V1"' 1
  # Workspace contains 2 file commands
  ensure_contains_atleast 'context: "repository @repo"' 3
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

  build_and_process_log --exclude_rule "repository @local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:26' 1
  ensure_contains_atleast 'context: "repository @repo"' 1
  ensure_contains_exactly 'download_event' 1
  ensure_contains_exactly "url: \"http://localhost:${fileserver_port}/file.txt\"" 1
  ensure_contains_exactly 'output: "file.txt"' 1
  ensure_contains_exactly "sha256: \"${file_sha256}\"" 1
}

function test_download_multiple() {
  # Prepare HTTP server with Python
  local server_dir="${TEST_TMPDIR}/server_dir"
  mkdir -p "${server_dir}"
  local file1="${server_dir}/file1.txt"
  local file2="${server_dir}/file2.txt"
  echo "contents here" > "${file1}"
  echo "contents here" > "${file2}"
  sha256=$(sha256sum "${file2}" | head -c 64)

  # Start HTTP server with Python
  ls -al "${server_dir}"
  sha256sum "${file2}"

  startup_server "${server_dir}"

  set_workspace_command "repository_ctx.download([\"http://localhost:${fileserver_port}/file1.txt\",\"http://localhost:${fileserver_port}/file2.txt\"], \"out_for_list.txt\", sha256='${sha256}')"

  build_and_process_log --exclude_rule "repository @local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:26' 1
  ensure_contains_atleast 'context: "repository @repo"' 1
  ensure_contains_exactly 'download_event' 1
  ensure_contains_exactly "url: \"http://localhost:${fileserver_port}/file1.txt\"" 1
  ensure_contains_exactly "url: \"http://localhost:${fileserver_port}/file2.txt\"" 1
  ensure_contains_exactly 'output: "out_for_list.txt"' 1
}

function test_download_integrity_sha256() {
  # Prepare HTTP server with Python
  local server_dir="${TEST_TMPDIR}/server_dir"
  mkdir -p "${server_dir}"
  local file="${server_dir}/file.txt"
  echo "file contents here" > "${file}"

  # Use Python for hashing and encoding due to cross-platform differences in
  # presence + behavior of `shasum` and `base64`.
  sha256_py='import base64, hashlib, sys; print(base64.b64encode(hashlib.sha256(sys.stdin.buffer.read()).digest()).decode("ascii"))'
  file_integrity="sha256-$(cat "${file}" | python3 -c "${sha256_py}")"

  # Start HTTP server with Python
  startup_server "${server_dir}"

  set_workspace_command "repository_ctx.download(\"http://localhost:${fileserver_port}/file.txt\", \"file.txt\", integrity=\"${file_integrity}\")"

  build_and_process_log --exclude_rule "repository @local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:26' 1
  ensure_contains_atleast 'context: "repository @repo"' 1
  ensure_contains_exactly 'download_event' 1
  ensure_contains_exactly "url: \"http://localhost:${fileserver_port}/file.txt\"" 1
  ensure_contains_exactly 'output: "file.txt"' 1
  ensure_contains_exactly "sha256: " 0
  ensure_contains_exactly "integrity: \"${file_integrity}\"" 1
}

function test_download_integrity_sha512() {
  # Prepare HTTP server with Python
  local server_dir="${TEST_TMPDIR}/server_dir"
  mkdir -p "${server_dir}"
  local file="${server_dir}/file.txt"
  echo "file contents here" > "${file}"

  # Use Python for hashing and encoding due to cross-platform differences in
  # presence + behavior of `shasum` and `base64`.
  sha512_py='import base64, hashlib, sys; print(base64.b64encode(hashlib.sha512(sys.stdin.buffer.read()).digest()).decode("ascii"))'
  file_integrity="sha512-$(cat "${file}" | python3 -c "${sha512_py}")"


  # Start HTTP server with Python
  startup_server "${server_dir}"

  set_workspace_command "repository_ctx.download(\"http://localhost:${fileserver_port}/file.txt\", \"file.txt\", integrity=\"${file_integrity}\")"

  build_and_process_log --exclude_rule "repository @local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:26' 1
  ensure_contains_atleast 'context: "repository @repo"' 1
  ensure_contains_exactly 'download_event' 1
  ensure_contains_exactly "url: \"http://localhost:${fileserver_port}/file.txt\"" 1
  ensure_contains_exactly 'output: "file.txt"' 1
  ensure_contains_exactly "sha256: " 0
  ensure_contains_exactly "integrity: \"${file_integrity}\"" 1
}

function test_download_integrity_malformed() {
  # Verify that a malformed value for integrity leads to a failing build
  local server_dir="${TEST_TMPDIR}/server_dir"
  mkdir -p "${server_dir}"
  local file="${server_dir}/file.txt"
  startup_server "${server_dir}"
  echo "file contents here" > "${file}"

  # Unsupported checksum algorithm
  file_integrity="This is no a checksum algorithm"
  set_workspace_command "repository_ctx.download(\"http://localhost:${fileserver_port}/file.txt\", \"file.txt\", integrity=\"${file_integrity}\")"
  bazel build //:test > "${TEST_log}" 2>&1 && fail "Expected failure" || :
  expect_log "${file_integrity}"
  expect_log "[Uu]nsupported checksum algorithm"

  # Syntactically invalid checksum
  file_integrity="sha512-ThisIsNotASha512Hash"
  set_workspace_command "repository_ctx.download(\"http://localhost:${fileserver_port}/file.txt\", \"file.txt\", integrity=\"${file_integrity}\")"
  bazel build //:test > "${TEST_log}" 2>&1 && fail "Expected failure" || :
  expect_log "${file_integrity}"
  expect_log "[Ii]nvalid.*checksum"

  # Syntactically correct, but incorrect value
  file_integrity="sha512-cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
  set_workspace_command "repository_ctx.download(\"http://localhost:${fileserver_port}/file.txt\", \"file.txt\", integrity=\"${file_integrity}\")"
  bazel build //:test > "${TEST_log}" 2>&1 && fail "Expected failure" || :
  expect_log "${file_integrity}"
  expect_log "[Ii]nvalid.*checksum"
}

function test_download_then_extract() {
  # Prepare HTTP server with Python
  local server_dir="${TEST_TMPDIR}/server_dir"
  mkdir -p "${server_dir}"
  local file_prefix="${server_dir}/download_then_extract"

  pushd ${TEST_TMPDIR}
  echo "This is one file" > ${server_dir}/download_then_extract.txt
  zip -r ${server_dir}/download_then_extract.zip server_dir
  file_sha256="$(sha256sum $server_dir/download_then_extract.zip | head -c 64)"
  popd

  # Start HTTP server with Python
  startup_server "${server_dir}"

  set_workspace_command "
  repository_ctx.download(\"http://localhost:${fileserver_port}/download_then_extract.zip\", \"downloaded_file.zip\", \"${file_sha256}\")
  repository_ctx.extract(\"downloaded_file.zip\", \"out_dir\", \"server_dir/\")"

  build_and_process_log --exclude_rule "repository @local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:3:26' 1
  ensure_contains_exactly 'location: .*repos.bzl:4:25' 1
  ensure_contains_atleast 'context: "repository @repo"' 2
  ensure_contains_exactly 'download_event' 1
  ensure_contains_exactly "url: \"http://localhost:${fileserver_port}/download_then_extract.zip\"" 1
  ensure_contains_exactly 'output: "downloaded_file.zip"' 1
  ensure_contains_exactly "sha256: \"${file_sha256}\"" 1
  ensure_contains_exactly 'extract_event' 1
  ensure_contains_exactly 'archive: "downloaded_file.zip"' 1
  ensure_contains_exactly 'output: "out_dir"' 1
  ensure_contains_exactly 'strip_prefix: "server_dir/"' 1

  ensure_output_contains_exactly_once "external/repo/out_dir/download_then_extract.txt" "This is one file"
}

function test_download_then_extract_tar() {
  # Prepare HTTP server with Python
  local server_dir="${TEST_TMPDIR}/server_dir"
  local data_dir="${TEST_TMPDIR}/data_dir"
  mkdir -p "${server_dir}"
  mkdir -p "${data_dir}"

  pushd ${TEST_TMPDIR}
  echo "Experiment with tar" > ${data_dir}/download_then_extract_tar.txt
  tar -zcvf ${server_dir}/download_then_extract.tar.gz data_dir
  file_sha256="$(sha256sum $server_dir/download_then_extract.tar.gz | head -c 64)"
  popd

  # Start HTTP server with Python
  startup_server "${server_dir}"

  set_workspace_command "
  repository_ctx.download(\"http://localhost:${fileserver_port}/download_then_extract.tar.gz\", \"downloaded_file.tar.gz\", \"${file_sha256}\")
  repository_ctx.extract(\"downloaded_file.tar.gz\", \"out_dir\", \"data_dir/\")"

  build_and_process_log --exclude_rule "repository @local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:3:26' 1
  ensure_contains_exactly 'location: .*repos.bzl:4:25' 1
  ensure_contains_atleast 'context: "repository @repo"' 2
  ensure_contains_exactly 'download_event' 1
  ensure_contains_exactly "url: \"http://localhost:${fileserver_port}/download_then_extract.tar.gz\"" 1
  ensure_contains_exactly 'output: "downloaded_file.tar.gz"' 1
  ensure_contains_exactly "sha256: \"${file_sha256}\"" 1
  ensure_contains_exactly 'extract_event' 1
  ensure_contains_exactly 'archive: "downloaded_file.tar.gz"' 1
  ensure_contains_exactly 'output: "out_dir"' 1
  ensure_contains_exactly 'strip_prefix: "data_dir/"' 1

  ensure_output_contains_exactly_once "external/repo/out_dir/download_then_extract_tar.txt" "Experiment with tar"
}

function test_download_and_extract() {
  # Prepare HTTP server with Python
  local server_dir="${TEST_TMPDIR}/server_dir"
  mkdir -p "${server_dir}"
  local file_prefix="${server_dir}/download_and_extract"

  pushd ${TEST_TMPDIR}
  echo "This is one file" > ${server_dir}/download_and_extract.txt
  zip -r ${server_dir}/download_and_extract.zip server_dir
  file_sha256="$(sha256sum server_dir/download_and_extract.zip | head -c 64)"
  popd

  # Start HTTP server with Python
  startup_server "${server_dir}"

  set_workspace_command "repository_ctx.download_and_extract(\"http://localhost:${fileserver_port}/download_and_extract.zip\", \"out_dir\", \"${file_sha256}\", \"zip\", \"server_dir/\")"

  build_and_process_log --exclude_rule "repository @local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:38' 1
  ensure_contains_atleast 'context: "repository @repo"' 1
  ensure_contains_exactly 'download_and_extract_event' 1
  ensure_contains_exactly "url: \"http://localhost:${fileserver_port}/download_and_extract.zip\"" 1
  ensure_contains_exactly 'output: "out_dir"' 1
  ensure_contains_exactly "sha256: \"${file_sha256}\"" 1
  ensure_contains_exactly 'type: "zip"' 1
  ensure_contains_exactly 'strip_prefix: "server_dir/"' 1

  ensure_output_contains_exactly_once "external/repo/out_dir/download_and_extract.txt" "This is one file"
}

function test_extract_rename_files() {
  local archive_tar="${TEST_TMPDIR}/archive.tar"

  # Create a tar archive with two entries, which would have conflicting
  # paths if extracted to a case-insensitive filesystem.
  pushd "${TEST_TMPDIR}"
  mkdir prefix
  echo "First file: a" > prefix/a.txt
  tar -cvf archive.tar prefix
  rm prefix/a.txt
  echo "Second file: A" > prefix/A.txt
  tar -rvf archive.tar prefix
  popd

  set_workspace_command "
  repository_ctx.extract('${archive_tar}', 'out_dir', 'prefix/', rename_files={
    'prefix/A.txt': 'prefix/renamed-A.txt',
  })"

  build_and_process_log --exclude_rule "repository @local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:3:25' 1
  ensure_contains_atleast 'context: "repository @repo"' 2
  ensure_contains_exactly 'extract_event' 1
  ensure_contains_exactly 'rename_files' 1
  ensure_contains_exactly 'key: "prefix/A.txt"' 1
  ensure_contains_exactly 'value: "prefix/renamed.A.txt"' 1

  ensure_output_contains_exactly_once "external/repo/out_dir/a.txt" "First file: a"
  ensure_output_contains_exactly_once "external/repo/out_dir/renamed-A.txt" "Second file: A"
}

function test_file() {
  set_workspace_command 'repository_ctx.file("filefile.sh", "echo filefile", True)'

  build_and_process_log --exclude_rule "repository @local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:22' 1
  ensure_contains_atleast 'context: "repository @repo"' 1

  # There are 3 file_event in external:repo as it is currently set up
  ensure_contains_exactly 'file_event' 3
  ensure_contains_exactly 'path: ".*filefile.sh"' 1
  ensure_contains_exactly 'content: "echo filefile"' 1
  ensure_contains_exactly 'executable: true' 1
}

function test_file_nonascii() {
  set_workspace_command 'repository_ctx.file("filefile.sh", "echo fïlëfïlë", True)'

  build_and_process_log --exclude_rule "repository @local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:22' 1
  ensure_contains_atleast 'context: "repository @repo"' 1

  # There are 3 file_event in external:repo as it is currently set up
  ensure_contains_exactly 'file_event' 3
  ensure_contains_exactly 'path: ".*filefile.sh"' 1
  ensure_contains_exactly 'executable: true' 1

  # This test file is in UTF-8, so the string passed to file() is UTF-8.
  # Protobuf strings are Unicode encoded in UTF-8, so the logged text
  # is double-encoded relative to the workspace command.
  #
  # >>> content = "echo f\u00EFl\u00EBf\u00EFl\u00EB".encode("utf8")
  # >>> proto_content = content.decode("iso-8859-1").encode("utf8")
  # >>> print("".join(chr(c) if c <= 0x7F else ("\\" + oct(c)[2:]) for c in proto_content))
  # echo f\303\203\302\257l\303\203\302\253f\303\203\302\257l\303\203\302\253
  # >>>

  ensure_contains_exactly 'content: "echo f\\303\\203\\302\\257l\\303\\203\\302\\253f\\303\\203\\302\\257l\\303\\203\\302\\253"' 1
}

function test_read() {
  set_workspace_command '
  content = "echo filefile"
  repository_ctx.file("filefile.sh", content, True)
  read_result = repository_ctx.read("filefile.sh")
  if read_result != content:
    fail("read(): expected %r, got %r" % (content, read_result))'

  build_and_process_log --exclude_rule "repository @local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:4:22' 1
  ensure_contains_exactly 'location: .*repos.bzl:5:36' 1
  ensure_contains_atleast 'context: "repository @repo"' 1

  ensure_contains_exactly 'read_event' 1
  ensure_contains_exactly 'path: ".*filefile.sh"' 2
}

function test_read_roundtrip_legacy_utf8() {
  # See discussion on https://github.com/bazelbuild/bazel/pull/7309
  set_workspace_command '
  content = "echo fïlëfïlë"
  repository_ctx.file("filefile.sh", content, True, legacy_utf8=True)
  read_result = repository_ctx.read("filefile.sh")

  corrupted_content = "echo fÃ¯lÃ«fÃ¯lÃ«"
  if read_result != corrupted_content:
    fail("read(): expected %r, got %r" % (corrupted_content, read_result))'

  build_and_process_log --exclude_rule "repository @local_config_cc"
}

function test_read_roundtrip_nolegacy_utf8() {
  set_workspace_command '
  content = "echo fïlëfïlë"
  repository_ctx.file("filefile.sh", content, True, legacy_utf8=False)
  read_result = repository_ctx.read("filefile.sh")
  if read_result != content:
    fail("read(): expected %r, got %r" % (content, read_result))'

  build_and_process_log --exclude_rule "repository @local_config_cc"
}

function test_os() {
  set_workspace_command 'print(repository_ctx.os.name)'

  build_and_process_log --exclude_rule "repository @local_config_cc"

  ensure_contains_atleast 'context: "repository @repo"' 1
  ensure_contains_exactly 'os_event' 1
}

function test_symlink() {
  set_workspace_command 'repository_ctx.file("symlink.txt", "something")
  repository_ctx.symlink("symlink.txt", "symlink_out.txt")'

  build_and_process_log --exclude_rule "repository @local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:22' 1
  ensure_contains_atleast 'context: "repository @repo"' 1
  ensure_contains_exactly 'symlink_event' 1
  ensure_contains_exactly 'target: ".*symlink.txt"' 1
  ensure_contains_exactly 'path: ".*symlink_out.txt"' 1
}

function test_template() {
  set_workspace_command 'repository_ctx.file("template_in.txt", "%{subKey}", False)
  repository_ctx.template("template_out.txt", "template_in.txt", {"subKey": "subVal"}, True)'

  build_and_process_log --exclude_rule "repository @local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:22' 1
  ensure_contains_atleast 'context: "repository @repo"' 1
  ensure_contains_exactly 'template_event' 1
  ensure_contains_exactly 'path: ".*template_out.txt"' 1
  ensure_contains_exactly 'template: ".*template_in.txt"' 1
  ensure_contains_exactly 'key: "subKey"' 1
  ensure_contains_exactly 'value: "subVal"' 1
  ensure_contains_exactly 'executable: true' 1
}

function test_which() {
  set_workspace_command 'print(repository_ctx.which("which_prog"))'

  build_and_process_log --exclude_rule "repository @local_config_cc"

  ensure_contains_exactly 'location: .*repos.bzl:2:29' 1
  ensure_contains_atleast 'context: "repository @repo"' 1
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
