#!/bin/bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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
# Tests remote execution and caching.
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  work_path=$(mktemp -d "${TEST_TMPDIR}/remote.XXXXXXXX")
  pid_file=$(mktemp -u "${TEST_TMPDIR}/remote.XXXXXXXX")
  attempts=1
  while [ $attempts -le 5 ]; do
    (( attempts++ ))
    worker_port=$(pick_random_unused_tcp_port) || fail "no port found"
    http_port=$(pick_random_unused_tcp_port) || fail "no port found"
    "${BAZEL_RUNFILES}/src/tools/remote/worker" \
        --work_path="${work_path}" \
        --listen_port=${worker_port} \
        --http_listen_port=${http_port} \
        --pid_file="${pid_file}" &
    local wait_seconds=0
    until [ -s "${pid_file}" ] || [ "$wait_seconds" -eq 15 ]; do
      sleep 1
      ((wait_seconds++)) || true
    done
    if [ -s "${pid_file}" ]; then
      break
    fi
  done
  if [ ! -s "${pid_file}" ]; then
    fail "Timed out waiting for remote worker to start."
  fi
}

function tear_down() {
  bazel clean --expunge >& $TEST_log
  if [ -s "${pid_file}" ]; then
    local pid=$(cat "${pid_file}")
    kill "${pid}" || true
  fi
  rm -rf "${pid_file}"
  rm -rf "${work_path}"
}

function test_cc_binary_http_cache() {
  mkdir -p a
  cat > a/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
cc_binary(
name = 'test',
srcs = [ 'test.cc' ],
)
EOF
  cat > a/test.cc <<EOF
#include <iostream>
int main() { std::cout << "Hello world!" << std::endl; return 0; }
EOF
  bazel build //a:test \
    || fail "Failed to build //a:test without remote cache"
  cp -f bazel-bin/a/test ${TEST_TMPDIR}/test_expected

  bazel clean --expunge
  bazel build \
      --remote_http_cache=http://localhost:${http_port} \
      //a:test \
      || fail "Failed to build //a:test with remote REST cache service"
  diff bazel-bin/a/test ${TEST_TMPDIR}/test_expected \
      || fail "Remote cache generated different result"
  # Check that persistent connections are closed after the build. Is there a good cross-platform way
  # to check this?
  if [[ "$PLATFORM" = "linux" ]]; then
    if netstat -tn | grep -qE ":${http_port}\\s+ESTABLISHED$"; then
      fail "connections to to cache not closed"
    fi
  fi
}

function test_cc_binary_http_cache_bad_server() {
  mkdir -p a
  cat > a/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
cc_binary(
name = 'test',
srcs = [ 'test.cc' ],
)
EOF
  cat > a/test.cc <<EOF
#include <iostream>
int main() { std::cout << "Hello world!" << std::endl; return 0; }
EOF
  bazel build //a:test >& $TEST_log \
    || fail "Failed to build //a:test without remote cache"
  cp -f bazel-bin/a/test ${TEST_TMPDIR}/test_expected

  bazel clean --expunge >& $TEST_log
  bazel build \
      --remote_http_cache=http://bad.hostname/bad/cache \
      //a:test >& $TEST_log \
      || fail "Failed to build //a:test with remote REST cache service"
  diff bazel-bin/a/test ${TEST_TMPDIR}/test_expected \
      || fail "Remote cache generated different result"
  # Check that persistent connections are closed after the build. Is there a good cross-platform way
  # to check this?
  if [[ "$PLATFORM" = "linux" ]]; then
    if netstat -tn | grep -qE ":${http_port}\\s+ESTABLISHED$"; then
      fail "connections to to cache not closed"
    fi
  fi
}

function test_refuse_to_upload_symlink() {
    cat > BUILD <<'EOF'
genrule(
    name = 'make-link',
    outs = ['l', 't'],
    cmd = 'touch $(location t) && ln -s t $(location l)',
)
EOF
    bazel build \
          --noremote_allow_symlink_upload \
          --remote_http_cache=http://localhost:${http_port} \
          //:make-link &> $TEST_log \
          && fail "should have failed" || true
    expect_log "/l is a symbolic link"
}

function test_refuse_to_upload_symlink_in_directory() {
    cat > BUILD <<'EOF'
genrule(
    name = 'make-link',
    outs = ['dir'],
    cmd = 'mkdir $(location dir) && touch $(location dir)/t && ln -s t $(location dir)/l',
)
EOF
    bazel build \
          --noremote_allow_symlink_upload \
          --remote_http_cache=http://localhost:${http_port} \
          //:make-link &> $TEST_log \
          && fail "should have failed" || true
    expect_log "dir/l is a symbolic link"
}

function set_directory_artifact_skylark_testfixtures() {
  mkdir -p a
  cat > a/rule.bzl <<'EOF'
def _gen_output_dir_impl(ctx):
  output_dir = ctx.actions.declare_directory(ctx.attr.outdir)

  ctx.actions.run_shell(
      outputs = [output_dir],
      inputs = [],
      command = """
        mkdir -p $1/sub1; \
        echo "Hello, world!" > $1/foo.txt; \
        echo "Shuffle, duffle, muzzle, muff" > $1/sub1/bar.txt
      """,
      arguments = [output_dir.path],
  )
  return [
      DefaultInfo(
          files=depset(direct=[output_dir]),
          data_runfiles=ctx.runfiles(files=[output_dir]),
      ),
  ]

gen_output_dir = rule(
    implementation = _gen_output_dir_impl,
    attrs = {
        "outdir": attr.string(mandatory = True),
    },
)
EOF
  cat > a/BUILD <<'EOF'
package(default_visibility = ["//visibility:public"])
load("//a:rule.bzl", "gen_output_dir")

gen_output_dir(
    name = "output_dir",
    outdir = "dir",
)

genrule(
    name = "test",
    srcs = [":output_dir"],
    outs = ["qux"],
    cmd = "mkdir $@ && paste -d\"\n\" $(location :output_dir)/foo.txt $(location :output_dir)/sub1/bar.txt > $@/out.txt",
)

sh_binary(
    name = "a-tool",
    srcs = ["a-tool.sh"],
    data = [":output_dir"],
)

genrule(
    name = "test2",
    outs = ["test2-out.txt"],
    cmd = "$(location :a-tool) > $@",
    tools = [":a-tool"],
)
EOF

  cat > a/a-tool.sh <<'EOF'
#!/bin/sh -eu
cat "$0".runfiles/main/a/dir/foo.txt "$0".runfiles/main/a/dir/sub1/bar.txt
EOF
  chmod u+x a/a-tool.sh

  cat > a/test_expected <<EOF
Hello, world!
Shuffle, duffle, muzzle, muff
EOF
}

function test_directory_artifact_skylark_local() {
  set_directory_artifact_skylark_testfixtures

  bazel build //a:test >& $TEST_log \
    || fail "Failed to build //a:test without remote execution"
  diff bazel-genfiles/a/qux/out.txt a/test_expected \
      || fail "Local execution generated different result"
}

function test_directory_artifact_skylark() {
  set_directory_artifact_skylark_testfixtures

  bazel build \
      --spawn_strategy=remote \
      --remote_executor=localhost:${worker_port} \
      //a:test >& $TEST_log \
      || fail "Failed to build //a:test with remote execution"
  diff bazel-genfiles/a/qux/out.txt a/test_expected \
      || fail "Remote execution generated different result"
  bazel clean --expunge
  bazel build \
      --spawn_strategy=remote \
      --remote_executor=localhost:${worker_port} \
      //a:test >& $TEST_log \
      || fail "Failed to build //a:test with remote execution"
  expect_log "remote cache hit"
  diff bazel-genfiles/a/qux/out.txt a/test_expected \
      || fail "Remote cache hit generated different result"
}

function test_directory_artifact_skylark_grpc_cache() {
  set_directory_artifact_skylark_testfixtures

  bazel build \
      --remote_cache=localhost:${worker_port} \
      //a:test >& $TEST_log \
      || fail "Failed to build //a:test with remote gRPC cache"
  diff bazel-genfiles/a/qux/out.txt a/test_expected \
      || fail "Remote cache miss generated different result"
  bazel clean --expunge
  bazel build \
      --remote_cache=localhost:${worker_port} \
      //a:test >& $TEST_log \
      || fail "Failed to build //a:test with remote gRPC cache"
  expect_log "remote cache hit"
  diff bazel-genfiles/a/qux/out.txt a/test_expected \
      || fail "Remote cache hit generated different result"
}

function test_directory_artifact_skylark_rest_cache() {
  set_directory_artifact_skylark_testfixtures

  bazel build \
      --remote_rest_cache=http://localhost:${http_port} \
      //a:test >& $TEST_log \
      || fail "Failed to build //a:test with remote REST cache"
  diff bazel-genfiles/a/qux/out.txt a/test_expected \
      || fail "Remote cache miss generated different result"
  bazel clean --expunge
  bazel build \
      --remote_rest_cache=http://localhost:${http_port} \
      //a:test >& $TEST_log \
      || fail "Failed to build //a:test with remote REST cache"
  expect_log "remote cache hit"
  diff bazel-genfiles/a/qux/out.txt a/test_expected \
      || fail "Remote cache hit generated different result"
}

function test_directory_artifact_in_runfiles_skylark_rest_cache() {
  set_directory_artifact_skylark_testfixtures

  bazel build \
      --remote_rest_cache=http://localhost:${http_port} \
      //a:test2 >& $TEST_log \
      || fail "Failed to build //a:test2 with remote REST cache"
  diff bazel-genfiles/a/test2-out.txt a/test_expected \
      || fail "Remote cache miss generated different result"
  bazel clean --expunge
  bazel build \
      --remote_rest_cache=http://localhost:${http_port} \
      //a:test2 >& $TEST_log \
      || fail "Failed to build //a:test2 with remote REST cache"
  expect_log "remote cache hit"
  diff bazel-genfiles/a/test2-out.txt a/test_expected \
      || fail "Remote cache hit generated different result"
}

run_suite "Remote execution and remote cache tests"
