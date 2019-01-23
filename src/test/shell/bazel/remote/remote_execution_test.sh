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
# Tests remote execution and caching.
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  work_path=$(mktemp -d "${TEST_TMPDIR}/remote.XXXXXXXX")
  cas_path=$(mktemp -d "${TEST_TMPDIR}/remote.XXXXXXXX")
  pid_file=$(mktemp -u "${TEST_TMPDIR}/remote.XXXXXXXX")
  attempts=1
  while [ $attempts -le 5 ]; do
    (( attempts++ ))
    worker_port=$(pick_random_unused_tcp_port) || fail "no port found"
    "${BAZEL_RUNFILES}/src/tools/remote/worker" \
        --work_path="${work_path}" \
        --listen_port=${worker_port} \
        --cas_path=${cas_path} \
        --incompatible_remote_symlinks \
        --pid_file="${pid_file}" >& $TEST_log &
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
  rm -rf "${cas_path}"
}

function test_cc_binary() {
  if [[ "$PLATFORM" == "darwin" ]]; then
    # TODO(b/37355380): This test is disabled due to RemoteWorker not supporting
    # setting SDKROOT and DEVELOPER_DIR appropriately, as is required of
    # action executors in order to select the appropriate Xcode toolchain.
    return 0
  fi

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
    || fail "Failed to build //a:test without remote execution"
  cp -f bazel-bin/a/test ${TEST_TMPDIR}/test_expected

  bazel clean --expunge >& $TEST_log
  bazel build \
      --spawn_strategy=remote \
      --remote_executor=localhost:${worker_port} \
      //a:test >& $TEST_log \
      || fail "Failed to build //a:test with remote execution"
  diff bazel-bin/a/test ${TEST_TMPDIR}/test_expected \
      || fail "Remote execution generated different result"
}

function test_cc_test() {
  if [[ "$PLATFORM" == "darwin" ]]; then
    # TODO(b/37355380): This test is disabled due to RemoteWorker not supporting
    # setting SDKROOT and DEVELOPER_DIR appropriately, as is required of
    # action executors in order to select the appropriate Xcode toolchain.
    return 0
  fi

  mkdir -p a
  cat > a/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
cc_test(
name = 'test',
srcs = [ 'test.cc' ],
)
EOF
  cat > a/test.cc <<EOF
#include <iostream>
int main() { std::cout << "Hello test!" << std::endl; return 0; }
EOF
  bazel test \
      --spawn_strategy=remote \
      --remote_executor=localhost:${worker_port} \
      --test_output=errors \
      //a:test >& $TEST_log \
      || fail "Failed to run //a:test with remote execution"
}

function test_cc_binary_grpc_cache() {
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
      --remote_cache=localhost:${worker_port} \
      //a:test >& $TEST_log \
      || fail "Failed to build //a:test with remote gRPC cache service"
  diff bazel-bin/a/test ${TEST_TMPDIR}/test_expected \
      || fail "Remote cache generated different result"
}

function test_cc_binary_grpc_cache_statsline() {
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
  bazel build \
      --remote_cache=localhost:${worker_port} \
      //a:test >& $TEST_log \
      || fail "Failed to build //a:test with remote gRPC cache service"
  bazel clean --expunge >& $TEST_log
  bazel build \
      --remote_cache=localhost:${worker_port} \
      //a:test 2>&1 | tee $TEST_log | grep "remote cache hit" \
      || fail "Output does not contain remote cache hits"
}

function test_failing_cc_test() {
  mkdir -p a
  cat > a/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
cc_test(
name = 'test',
srcs = [ 'test.cc' ],
)
EOF
  cat > a/test.cc <<EOF
#include <iostream>
int main() { std::cout << "Fail me!" << std::endl; return 1; }
EOF
  bazel test \
      --spawn_strategy=remote \
      --remote_executor=localhost:${worker_port} \
      --test_output=errors \
      //a:test >& $TEST_log \
      && fail "Expected test failure" || true
  # TODO(ulfjack): Check that the test failure gets reported correctly.
}

function test_local_fallback_works_with_local_strategy() {
  mkdir -p gen1
  cat > gen1/BUILD <<'EOF'
genrule(
name = "gen1",
srcs = [],
outs = ["out1"],
cmd = "touch \"$@\"",
tags = ["no-remote"],
)
EOF

  bazel build \
      --spawn_strategy=remote \
      --remote_executor=localhost:${worker_port} \
      --remote_local_fallback_strategy=local \
      --build_event_text_file=gen1.log \
      //gen1 >& $TEST_log \
      || fail "Expected success"

  mv gen1.log $TEST_log
  expect_log "1 process: 1 local"
}

function test_local_fallback_works_with_sandboxed_strategy() {
  mkdir -p gen2
  cat > gen2/BUILD <<'EOF'
genrule(
name = "gen2",
srcs = [],
outs = ["out2"],
cmd = "touch \"$@\"",
tags = ["no-remote"],
)
EOF

  bazel build \
      --spawn_strategy=remote \
      --remote_executor=localhost:${worker_port} \
      --remote_local_fallback_strategy=sandboxed \
      --build_event_text_file=gen2.log \
      //gen2 >& $TEST_log \
      || fail "Expected success"

  mv gen2.log $TEST_log
  expect_log "1 process: 1 .*-sandbox"
}

function is_file_uploaded() {
  h=$(shasum -a256 < $1)
  if [ -e "$cas_path/${h:0:64}" ]; then return 0; else return 1; fi
}

function test_failed_test_outputs_not_uploaded() {
  # Test that outputs of a failed test/action are not uploaded to the remote
  # cache. This is a regression test for https://github.com/bazelbuild/bazel/issues/7232
  mkdir -p a
  cat > a/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
cc_test(
  name = 'test',
  srcs = [ 'test.cc' ],
)
EOF
  cat > a/test.cc <<EOF
#include <iostream>
int main() { std::cout << "Fail me!" << std::endl; return 1; }
EOF
  bazel test \
      --remote_cache=localhost:${worker_port} \
      --test_output=errors \
      //a:test >& $TEST_log \
      && fail "Expected test failure" || true
   ($(is_file_uploaded bazel-testlogs/a/test/test.log) \
     && fail "Expected test log to not be uploaded to remote execution") || true
   ($(is_file_uploaded bazel-testlogs/a/test/test.xml) \
     && fail "Expected test xml to not be uploaded to remote execution") || true
}

# Tests that the remote worker can return a 200MB blob that requires chunking.
# Blob has to be that large in order to exceed the grpc default max message size.
function test_genrule_large_output_chunking() {
  mkdir -p a
  cat > a/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
genrule(
name = "large_output",
srcs = ["small_blob.txt"],
outs = ["large_blob.txt"],
cmd = "cp \$(location small_blob.txt) tmp.txt; " +
"(for i in {1..22} ; do cat tmp.txt >> \$@; cp \$@ tmp.txt; done)",
)
EOF
  cat > a/small_blob.txt <<EOF
0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
EOF
  bazel build //a:large_output >& $TEST_log \
    || fail "Failed to build //a:large_output without remote execution"
  cp -f bazel-genfiles/a/large_blob.txt ${TEST_TMPDIR}/large_blob_expected.txt

  bazel clean --expunge >& $TEST_log
  bazel build \
      --spawn_strategy=remote \
      --remote_executor=localhost:${worker_port} \
      //a:large_output >& $TEST_log \
      || fail "Failed to build //a:large_output with remote execution"
  diff bazel-genfiles/a/large_blob.txt ${TEST_TMPDIR}/large_blob_expected.txt \
      || fail "Remote execution generated different result"
}

function test_py_test() {
  mkdir -p a
  cat > a/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
py_test(
name = 'test',
srcs = [ 'test.py' ],
)
EOF
  cat > a/test.py <<'EOF'
import sys
if __name__ == "__main__":
    sys.exit(0)
EOF
  bazel test \
      --spawn_strategy=remote \
      --remote_executor=localhost:${worker_port} \
      --test_output=errors \
      //a:test >& $TEST_log \
      || fail "Failed to run //a:test with remote execution"
}

function test_py_test_with_xml_output() {
  mkdir -p a
  cat > a/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
py_test(
name = 'test',
srcs = [ 'test.py' ],
)
EOF
  cat > a/test.py <<'EOF'
import sys
import os
if __name__ == "__main__":
    f = open(os.environ['XML_OUTPUT_FILE'], "w")
    f.write('''
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="test" tests="1" failures="1" errors="1">
    <testcase name="first" status="run">
      <failure>That did not work!</failure>
    </testcase>
  </testsuite>
</testsuites>
''')
    sys.exit(0)
EOF
  bazel test \
      --spawn_strategy=remote \
      --remote_executor=localhost:${worker_port} \
      --test_output=errors \
      //a:test >& $TEST_log \
      || fail "Failed to run //a:test with remote execution"
  xml=bazel-testlogs/a/test/test.xml
  [ -e $xml ] || fail "Expected to find XML output"
  cat $xml > $TEST_log
  expect_log 'That did not work!'
}

function test_failing_py_test_with_xml_output() {
  mkdir -p a
  cat > a/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
py_test(
name = 'test',
srcs = [ 'test.py' ],
)
EOF
  cat > a/test.py <<'EOF'
import sys
import os
if __name__ == "__main__":
    f = open(os.environ['XML_OUTPUT_FILE'], "w")
    f.write('''
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="test" tests="1" failures="1" errors="1">
    <testcase name="first" status="run">
      <failure>That did not work!</failure>
    </testcase>
  </testsuite>
</testsuites>
''')
    sys.exit(1)
EOF
  bazel test \
      --spawn_strategy=remote \
      --remote_executor=localhost:${worker_port} \
      --test_output=errors \
      //a:test >& $TEST_log \
      && fail "Expected test failure" || true
  xml=bazel-testlogs/a/test/test.xml
  [ -e $xml ] || fail "Expected to find XML output"
  cat $xml > $TEST_log
  expect_log 'That did not work!'
}

function test_noinput_action() {
  mkdir -p a
  cat > a/rule.bzl <<'EOF'
def _impl(ctx):
  output = ctx.outputs.out
  ctx.actions.run_shell(
      outputs=[output],
      command="echo 'Hello World' > %s" % (output.path))

empty = rule(
    implementation=_impl,
    outputs={"out": "%{name}.txt"},
)
EOF
  cat > a/BUILD <<'EOF'
load("//a:rule.bzl", "empty")
package(default_visibility = ["//visibility:public"])
empty(name = 'test')
EOF
  bazel build \
      --remote_cache=localhost:${worker_port} \
      --test_output=errors \
      //a:test >& $TEST_log \
      || fail "Failed to run //a:test with remote execution"
}

function test_timeout() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
sh_test(
  name = "sleep",
  timeout = "short",
  srcs = ["sleep.sh"],
)
EOF

  cat > a/sleep.sh <<'EOF'
#!/bin/bash
for i in {1..3}
do
    echo "Sleeping $i..."
    sleep 1
done
EOF
  chmod +x a/sleep.sh
  bazel test \
      --spawn_strategy=remote \
      --remote_executor=localhost:${worker_port} \
      --test_output=errors \
      --test_timeout=1,1,1,1 \
      //a:sleep >& $TEST_log \
      && fail "Test failure (timeout) expected" || true
  expect_log "TIMEOUT"
  expect_log "Sleeping 1..."
  # The current implementation of the remote worker does not terminate actions
  # when they time out, therefore we cannot verify that:
  # expect_not_log "Sleeping 3..."
}

function test_passed_env_user() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
sh_test(
  name = "user_test",
  timeout = "short",
  srcs = ["user_test.sh"],
)
EOF

  cat > a/user_test.sh <<'EOF'
#!/bin/sh
echo "user=$USER"
EOF
  chmod +x a/user_test.sh
  bazel test \
      --spawn_strategy=remote \
      --remote_executor=localhost:${worker_port} \
      --test_output=all \
      --test_env=USER=boo \
      //a:user_test >& $TEST_log \
      || fail "Failed to run //a:user_test with remote execution"
  expect_log "user=boo"

  # Rely on the test-setup script to set USER value to whoami.
  export USER=
  bazel test \
      --spawn_strategy=remote \
      --remote_executor=localhost:${worker_port} \
      --test_output=all \
      //a:user_test >& $TEST_log \
      || fail "Failed to run //a:user_test with remote execution"
  expect_log "user=$(whoami)"
}

function test_exitcode() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
genrule(
  name = "foo",
  srcs = [],
  outs = ["foo.txt"],
  cmd = "echo \"hello world\" > \"$@\"",
)
EOF

  (set +e
    bazel build \
      --genrule_strategy=remote \
      --remote_executor=bazel-test-does-not-exist \
      //a:foo >& $TEST_log
    [ $? -eq 34 ]) || fail "Test failed due to wrong exit code"
}

# Bazel should display non-test errors to the user, instead of hiding them behind test failures.
# For example, if the network connection to the remote executor fails it shouldn't be displayed as
# a test error.
function test_display_non_testerrors() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
sh_test(
  name = "test",
  timeout = "short",
  srcs = ["test.sh"],
)
EOF
  cat > a/test.sh <<'EOF'
#!/bin/sh
#This will never run, because the remote side is not reachable.
EOF
  chmod +x a/test.sh
  bazel test \
      --spawn_strategy=remote \
      --remote_executor=bazel.does.not.exist:1234 \
      --noexperimental_remote_retry \
      --test_output=all \
      --test_env=USER=boo \
      //a:test >& $TEST_log \
      && fail "Test failure expected" || true
  expect_not_log "test.log"
  expect_log "Failed to query remote execution capabilities"
}

function set_symlinks_in_directory_testfixtures() {
    cat > BUILD <<'EOF'
genrule(
    name = 'make-links',
    outs = ['dir', 'r', 'a', 'rd', 'ad'],
    cmd = ('mkdir $(location dir) && ' +
        'cd $(location dir) && ' +
        'echo hello > foo && ' + # Regular file.
        'ln -s foo r && ' +  # Relative symlink, will be passed as symlink.
        'ln -s $$PWD/foo a && ' +  # Absolute symlink, will be copied.
        'mkdir bar && ' + # Regular directory.
        'echo bla > bar/baz && ' +
        'ln -s bar rd && ' +  # Relative symlink, will be passed as symlink.
        'ln -s $$PWD/bar ad && ' + # Absolute symlink, will be copied.
        'cd .. && ' +
        'ln -s dir/foo r && ' +  # Relative symlink, will be passed as symlink.
        'ln -s $$PWD/dir/foo a && ' +  # Absolute symlink, will be copied.
        'ln -s dir rd && ' +  # Relative symlink, will be passed as symlink.
        'ln -s $$PWD/dir ad' # Absolute symlink, will be copied.
    ),
)
EOF
    cat > "${TEST_TMPDIR}/expected_links" <<'EOF'
./ad/r
./ad/rd
./dir/r
./dir/rd
./r
./rd
EOF
}

function test_symlinks_in_directory() {
    set_symlinks_in_directory_testfixtures
    bazel build \
          --incompatible_remote_symlinks \
          --remote_executor=localhost:${worker_port} \
          --spawn_strategy=remote \
          //:make-links &> $TEST_log \
          || fail "Failed to build //:make-links with remote execution"
    expect_log "1 remote"
    find -L bazel-genfiles -type f -exec cat {} \; | sort | uniq -c &> $TEST_log
    expect_log "9 bla"
    expect_log "11 hello"
    CUR=$PWD && cd bazel-genfiles && \
      find . -type l | sort > "${TEST_TMPDIR}/links" && cd $CUR
    diff "${TEST_TMPDIR}/links" "${TEST_TMPDIR}/expected_links" \
      || fail "Remote execution created different symbolic links"
}

function test_symlinks_in_directory_cache_only() {
    # This test is the same as test_symlinks_in_directory, except it works
    # locally and uses the remote cache to query results.
    set_symlinks_in_directory_testfixtures
    bazel build \
          --incompatible_remote_symlinks \
          --remote_cache=localhost:${worker_port} \
          --spawn_strategy=local \
          //:make-links &> $TEST_log \
          || fail "Failed to build //:make-links with remote cache service"
    expect_log "1 local"
    bazel clean --expunge # Get rid of local results, rely on remote cache.
    bazel build \
          --incompatible_remote_symlinks \
          --remote_cache=localhost:${worker_port} \
          --spawn_strategy=local \
          //:make-links &> $TEST_log \
          || fail "Failed to build //:make-links with remote cache service"
    expect_log "1 remote cache hit"
    # Check that the results downloaded from remote cache are the same as local.
    find -L bazel-genfiles -type f -exec cat {} \; | sort | uniq -c &> $TEST_log
    expect_log "9 bla"
    expect_log "11 hello"
    CUR=$PWD && cd bazel-genfiles && \
      find . -type l | sort > "${TEST_TMPDIR}/links" && cd $CUR
    diff "${TEST_TMPDIR}/links" "${TEST_TMPDIR}/expected_links" \
      || fail "Cached result created different symbolic links"
}

function test_treeartifact_in_runfiles() {
     mkdir -p a
    cat > a/BUILD <<'EOF'
load(":output_directory.bzl", "gen_output_dir", "gen_output_dir_test")

gen_output_dir(
    name = "skylark_output_dir",
    outdir = "dir",
)

gen_output_dir_test(
    name = "skylark_output_dir_test",
    dir = ":skylark_output_dir",
)
EOF
     cat > a/output_directory.bzl <<'EOF'
def _gen_output_dir_impl(ctx):
  output_dir = ctx.actions.declare_directory(ctx.attr.outdir)
  ctx.actions.run_shell(
      outputs = [output_dir],
      inputs = [],
      command = """
        mkdir -p $1/sub; \
        echo "foo" > $1/foo; \
        echo "bar" > $1/sub/bar
      """,
      arguments = [output_dir.path],
  )
  return [
      DefaultInfo(files=depset(direct=[output_dir]),
                  runfiles = ctx.runfiles(files = [output_dir]))
  ]
gen_output_dir = rule(
    implementation = _gen_output_dir_impl,
    attrs = {
        "outdir": attr.string(mandatory = True),
    },
)
def _gen_output_dir_test_impl(ctx):
    test = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(test, "echo hello world")
    myrunfiles = ctx.runfiles(files=ctx.attr.dir.default_runfiles.files.to_list())
    return [
        DefaultInfo(
            executable = test,
            runfiles = myrunfiles,
        ),
    ]
gen_output_dir_test = rule(
    implementation = _gen_output_dir_test_impl,
    test = True,
    attrs = {
        "dir":  attr.label(mandatory = True),
    },
)
EOF
     # Also test this directory inputs with sandboxing. Ideally we would add such
     # a test into the sandboxing module.
     bazel test \
           --spawn_strategy=sandboxed \
           //a:skylark_output_dir_test \
           || fail "Failed to run //a:skylark_output_dir_test with sandboxing"

     bazel test \
           --spawn_strategy=remote \
           --remote_executor=localhost:${worker_port} \
           //a:skylark_output_dir_test \
           || fail "Failed to run //a:skylark_output_dir_test with remote execution"
}


# TODO(alpha): Add a test that fails remote execution when remote worker
# supports sandbox.

run_suite "Remote execution and remote cache tests"
