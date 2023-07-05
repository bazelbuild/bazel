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

set -euo pipefail

# --- begin runfiles.bash initialization ---
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
source "$(rlocation "io_bazel/src/test/shell/bazel/remote/remote_utils.sh")" \
  || { echo "remote_utils.sh not found!" >&2; exit 1; }

function set_up() {
  http_port=$(pick_random_unused_tcp_port) || fail "no port found"
  start_worker \
        --http_listen_port=${http_port}
}

function tear_down() {
  bazel clean >& $TEST_log
  stop_worker
}

function test_remote_http_cache_flag() {
  # Test that the deprecated --remote_http_cache flag still works.
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ["foo.txt"],
  cmd = "echo \"foo bar\" > \$@",
)
EOF

  bazel build \
      --remote_http_cache=http://localhost:${http_port} \
      //a:foo \
      || fail "Failed to build //a:foo with remote cache"
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

  bazel clean
  bazel build \
      --remote_cache=http://localhost:${http_port} \
      //a:test \
      || fail "Failed to build //a:test with remote HTTP cache service"
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

  bazel clean >& $TEST_log
  bazel build \
      --remote_cache=http://bad.hostname/bad/cache \
      //a:test >& $TEST_log \
      || fail "Failed to build //a:test with remote HTTP cache service"
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

function set_directory_artifact_starlark_testfixtures() {
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
    cmd = "paste -d\"\n\" $(location :output_dir)/foo.txt $(location :output_dir)/sub1/bar.txt > $@",
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

function test_directory_artifact_starlark_local() {
  set_directory_artifact_starlark_testfixtures

  bazel build //a:test >& $TEST_log \
    || fail "Failed to build //a:test without remote execution"
  diff bazel-genfiles/a/qux a/test_expected \
      || fail "Local execution generated different result"
}

function test_directory_artifact_starlark() {
  set_directory_artifact_starlark_testfixtures

  bazel build \
      --spawn_strategy=remote \
      --remote_executor=grpc://localhost:${worker_port} \
      //a:test >& $TEST_log \
      || fail "Failed to build //a:test with remote execution"
  diff bazel-genfiles/a/qux a/test_expected \
      || fail "Remote execution generated different result"
  bazel clean
  bazel build \
      --spawn_strategy=remote \
      --remote_executor=grpc://localhost:${worker_port} \
      //a:test >& $TEST_log \
      || fail "Failed to build //a:test with remote execution"
  expect_log "remote cache hit"
  diff bazel-genfiles/a/qux a/test_expected \
      || fail "Remote cache hit generated different result"
}

function test_directory_artifact_starlark_grpc_cache() {
  set_directory_artifact_starlark_testfixtures

  bazel build \
      --remote_cache=grpc://localhost:${worker_port} \
      //a:test >& $TEST_log \
      || fail "Failed to build //a:test with remote gRPC cache"
  diff bazel-genfiles/a/qux a/test_expected \
      || fail "Remote cache miss generated different result"
  bazel clean
  bazel build \
      --remote_cache=grpc://localhost:${worker_port} \
      //a:test >& $TEST_log \
      || fail "Failed to build //a:test with remote gRPC cache"
  expect_log "remote cache hit"
  diff bazel-genfiles/a/qux a/test_expected \
      || fail "Remote cache hit generated different result"
}

function test_directory_artifact_starlark_http_cache() {
  set_directory_artifact_starlark_testfixtures

  bazel build \
      --remote_cache=http://localhost:${http_port} \
      //a:test >& $TEST_log \
      || fail "Failed to build //a:test with remote HTTP cache"
  diff bazel-genfiles/a/qux a/test_expected \
      || fail "Remote cache miss generated different result"
  bazel clean
  bazel build \
      --remote_cache=http://localhost:${http_port} \
      //a:test >& $TEST_log \
      || fail "Failed to build //a:test with remote HTTP cache"
  expect_log "remote cache hit"
  diff bazel-genfiles/a/qux a/test_expected \
      || fail "Remote cache hit generated different result"
}

function test_directory_artifact_in_runfiles_starlark_http_cache() {
  set_directory_artifact_starlark_testfixtures

  bazel build \
      --remote_cache=http://localhost:${http_port} \
      //a:test2 >& $TEST_log \
      || fail "Failed to build //a:test2 with remote HTTP cache"
  diff bazel-genfiles/a/test2-out.txt a/test_expected \
      || fail "Remote cache miss generated different result"
  bazel clean
  bazel build \
      --remote_cache=http://localhost:${http_port} \
      //a:test2 >& $TEST_log \
      || fail "Failed to build //a:test2 with remote HTTP cache"
  expect_log "remote cache hit"
  diff bazel-genfiles/a/test2-out.txt a/test_expected \
      || fail "Remote cache hit generated different result"
}


function test_remote_state_cleared() {
  # Regression test for https://github.com/bazelbuild/bazel/issues/7555
  # Test that the remote cache state is properly reset, so that building without
  # a remote cache works after previously building with a remote cache.
  mkdir -p a
  cat > a/BUILD <<'EOF'
genrule(
  name = "gen1",
  outs = ["out1"],
  cmd = "touch $@",
)
EOF

  bazel build \
      --remote_cache=http://localhost:${http_port} \
      //a:gen1 \
    || fail "Failed to build //a:gen1 with remote cache"

  bazel clean

  bazel build //a:gen1 \
    || fail "Failed to build //a:gen1 without remote cache"
}

function test_genrule_combined_disk_http_cache() {
  # Test for the combined disk and http cache.
  # Built items should be pushed to both the disk and http cache.
  # If --noremote_upload_local_results flag is set,
  # built items should only be pushed to the disk cache.
  # If --noremote_accept_cached flag is set,
  # built items should only be checked from the disk cache.
  # If an item is missing on disk cache, but present on http cache,
  # then bazel should copy it from http cache to disk cache on fetch.

  local cache="${TEST_TMPDIR}/cache"
  local disk_flags="--disk_cache=$cache"
  local http_flags="--remote_cache=http://localhost:${http_port}"

  mkdir -p a
  cat > a/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
genrule(
name = 'test',
cmd = 'echo "Hello world" > \$@',
outs = [ 'test.txt' ],
)
EOF
  rm -rf $cache
  mkdir $cache

  # Build and push to disk cache but not http cache
  bazel build $disk_flags $http_flags --noremote_upload_local_results //a:test \
    || fail "Failed to build //a:test with combined disk http cache"
  cp -f bazel-genfiles/a/test.txt ${TEST_TMPDIR}/test_expected

  # Fetch from disk cache
  bazel clean
  bazel build $disk_flags //a:test --noremote_upload_local_results &> $TEST_log \
    || fail "Failed to fetch //a:test from disk cache"
  expect_log "1 disk cache hit" "Fetch from disk cache failed"
  diff bazel-genfiles/a/test.txt ${TEST_TMPDIR}/test_expected \
    || fail "Disk cache generated different result"

  # No cache result from http cache, rebuild target
  bazel clean
  bazel build $http_flags //a:test --noremote_upload_local_results &> $TEST_log \
    || fail "Failed to build //a:test"
  expect_not_log "1 remote cache hit" "Should not get cache hit from http cache"
  expect_log "1 .*-sandbox" "Rebuild target failed"
  diff bazel-genfiles/a/test.txt ${TEST_TMPDIR}/test_expected \
    || fail "Rebuilt target generated different result"

  rm -rf $cache
  mkdir $cache

  # No cache result from http cache, rebuild target, and upload result to http cache
  bazel clean
  bazel build $http_flags //a:test --noremote_accept_cached &> $TEST_log \
    || fail "Failed to build //a:test"
  expect_not_log "1 remote cache hit" "Should not get cache hit from http cache"
  expect_log "1 .*-sandbox" "Rebuild target failed"
  diff bazel-genfiles/a/test.txt ${TEST_TMPDIR}/test_expected \
    || fail "Rebuilt target generated different result"

  # No cache result from http cache, rebuild target, and upload result to disk cache
  bazel clean
  bazel build $disk_flags $http_flags //a:test --noremote_accept_cached &> $TEST_log \
    || fail "Failed to build //a:test"
  expect_not_log "1 remote cache hit" "Should not get cache hit from http cache"
  expect_log "1 .*-sandbox" "Rebuild target failed"
  diff bazel-genfiles/a/test.txt ${TEST_TMPDIR}/test_expected \
    || fail "Rebuilt target generated different result"

  # Fetch from disk cache
  bazel clean
  bazel build $disk_flags $http_flags //a:test --noremote_accept_cached &> $TEST_log \
    || fail "Failed to build //a:test"
  expect_log "1 disk cache hit" "Fetch from disk cache failed"
  diff bazel-genfiles/a/test.txt ${TEST_TMPDIR}/test_expected \
    || fail "Disk cache generated different result"

  rm -rf $cache
  mkdir $cache

  # Build and push to disk cache and http cache
  bazel clean
  bazel build $disk_flags $http_flags //a:test \
    || fail "Failed to build //a:test with combined disk http cache"
  diff bazel-genfiles/a/test.txt ${TEST_TMPDIR}/test_expected \
    || fail "Built target generated different result"

  # Fetch from disk cache
  bazel clean
  bazel build $disk_flags //a:test &> $TEST_log \
    || fail "Failed to fetch //a:test from disk cache"
  expect_log "1 disk cache hit" "Fetch from disk cache failed"
  diff bazel-genfiles/a/test.txt ${TEST_TMPDIR}/test_expected \
    || fail "Disk cache generated different result"

  # Fetch from http cache
  bazel clean
  bazel build $http_flags //a:test &> $TEST_log \
    || fail "Failed to fetch //a:test from http cache"
  expect_log "1 remote cache hit" "Fetch from http cache failed"
  diff bazel-genfiles/a/test.txt ${TEST_TMPDIR}/test_expected \
    || fail "HTTP cache generated different result"

  rm -rf $cache
  mkdir $cache

  # Copy from http cache to disk cache
  bazel clean
  bazel build $disk_flags $http_flags //a:test &> $TEST_log \
    || fail "Failed to copy //a:test from http cache to disk cache"
  expect_log "1 remote cache hit" "Copy from http cache to disk cache failed"
  diff bazel-genfiles/a/test.txt ${TEST_TMPDIR}/test_expected \
    || fail "HTTP cache generated different result"

  # Fetch from disk cache
  bazel clean
  bazel build $disk_flags //a:test &> $TEST_log \
    || fail "Failed to fetch //a:test from disk cache"
  expect_log "1 disk cache hit" "Fetch from disk cache after copy from http cache failed"
  diff bazel-genfiles/a/test.txt ${TEST_TMPDIR}/test_expected \
    || fail "Disk cache generated different result"

  rm -rf $cache
}

function test_tag_no_remote_cache() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
genrule(
  name = "foo",
  srcs = [],
  outs = ["foo.txt"],
  cmd = "echo \"foo\" > \"$@\"",
  tags = ["no-remote-cache"]
)
EOF

  bazel build \
    --spawn_strategy=local \
    --remote_cache=grpc://localhost:${worker_port} \
    //a:foo >& $TEST_log || fail "Failed to build //a:foo"

  expect_log "1 local"

  bazel clean

  bazel build \
    --spawn_strategy=local \
    --remote_cache=grpc://localhost:${worker_port} \
    //a:foo || fail "Failed to build //a:foo"

  expect_log "1 local"
  expect_not_log "remote cache hit"
}

function test_remote_http_cache_with_bad_netrc_content() {
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ["foo.txt"],
  cmd = "echo \"foo bar\" > \$@",
)
EOF
  cat > a/.netrc <<EOF
this is bad netrc content
EOF

  NETRC="${PWD}/a/.netrc" bazel build \
      --remote_cache=http://localhost:${http_port} \
      //a:foo \
      || fail "Failed to build //a:foo with bad netrc content"
}

function test_remote_http_cache_with_unused_netrc_no_warning() {
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ["foo.txt"],
  cmd = "echo \"foo bar\" > \$@",
)
EOF
  cat > a/.netrc <<EOF
machine foo.example.org login foouser password foopass
EOF

  NETRC="${PWD}/a/.netrc" bazel build \
      --remote_cache=http://localhost:${http_port} \
      //a:foo &> $TEST_log \
      || fail "Failed to build //a:foo"
  expect_not_log "WARNING: Credentials are transmitted in plaintext" "Should not print warning"
}

function test_remote_http_cache_with_netrc_warning() {
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ["foo.txt"],
  cmd = "echo \"foo bar\" > \$@",
)
EOF
  cat > a/.netrc <<EOF
machine localhost login foouser password foopass
EOF

  NETRC="${PWD}/a/.netrc" bazel build \
      --remote_cache=http://localhost:${http_port} \
      //a:foo &> $TEST_log \
      || fail "Failed to build //a:foo"
  expect_log "WARNING: Credentials are transmitted in plaintext"
}

run_suite "Remote execution and remote cache tests"
