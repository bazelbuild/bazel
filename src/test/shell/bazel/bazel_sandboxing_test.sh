#!/bin/bash
#
# Copyright 2015 The Bazel Authors. All rights reserved.
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
# Test sandboxing spawn strategy
#

# Load test environment
# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source ${CURRENT_DIR}/../sandboxing_test_utils.sh \
  || { echo "sandboxing_test_utils.sh not found!" >&2; exit 1; }
source ${CURRENT_DIR}/remote_helpers.sh \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }

cat >>$TEST_TMPDIR/bazelrc <<'EOF'
# Testing the sandboxed strategy requires using the sandboxed strategy. While it is the default,
# we want to make sure that this explicitly fails when the strategy is not available on the system
# running the test.
build --spawn_strategy=sandboxed --genrule_strategy=sandboxed
EOF

function set_up {
  export BAZEL_GENFILES_DIR=$(bazel info bazel-genfiles 2>/dev/null)
  export BAZEL_BIN_DIR=$(bazel info bazel-bin 2>/dev/null)

  mkdir -p examples/genrule
  cat << 'EOF' > examples/genrule/a.txt
foo bar bz
EOF
  cat << 'EOF' > examples/genrule/b.txt
apples oranges bananas
EOF

  # Create cyclic symbolic links to check whether the strategy catches that.
  ln -sf cyclic2 examples/genrule/cyclic1
  ln -sf cyclic1 examples/genrule/cyclic2

  # Create relative symlinks.
  mkdir -p examples/genrule/symlinks/{a,ok/sub}
  echo OK > examples/genrule/symlinks/ok/x.txt
  ln -s $PWD/examples/genrule/symlinks/ok/sub examples/genrule/symlinks/a/b
  ln -s ../x.txt examples/genrule/symlinks/a/b/x.txt

  echo 'stuff to serve' > file_to_serve

  cat << 'EOF' > examples/genrule/BUILD
genrule(
  name = "works",
  srcs = [ "a.txt" ],
  outs = [ "works.txt" ],
  cmd = "wc $(location :a.txt) > $@",
)

sh_binary(
    name = "tool",
    srcs = ["tool.sh"],
    data = ["datafile"],
)

genrule(
    name = "tools_work",
    srcs = [],
    outs = ["tools.txt"],
    cmd = "$(location :tool) $@",
    tools = [":tool"],
)

genrule(
   name = "tooldir",
   srcs = [],
   outs = ["tooldir.txt"],
   cmd = "ls -l external/bazel_tools/tools/genrule | tee $@ >&2; " +
       "cat external/bazel_tools/tools/genrule/genrule-setup.sh >&2",
)

genrule(
  name = "relative_symlinks",
  srcs = [ "symlinks/a/b/x.txt" ],
  outs = [ "relative_symlinks.txt" ],
  cmd = "cat $(location :symlinks/a/b/x.txt) > $@",
)

genrule(
  name = "breaks1",
  srcs = [ "a.txt" ],
  outs = [ "breaks1.txt" ],
  cmd = "wc $(location :a.txt) `dirname $(location :a.txt)`/b.txt &> $@",
)

genrule(
  name = "breaks1_works_with_local",
  srcs = [ "a.txt" ],
  outs = [ "breaks1_works_with_local.txt" ],
  cmd = "wc $(location :a.txt) `dirname $(location :a.txt)`/b.txt > $@",
  local = 1,
)

genrule(
  name = "breaks1_works_with_local_tag",
  srcs = [ "a.txt" ],
  outs = [ "breaks1_works_with_local_tag.txt" ],
  cmd = "wc $(location :a.txt) `dirname $(location :a.txt)`/b.txt > $@",
  tags = [ "local" ],
)

load('//examples/genrule:skylark.bzl', 'skylark_breaks1')

skylark_breaks1(
  name = "skylark_breaks1",
  input = "a.txt",
  output = "skylark_breaks1.txt",
)

skylark_breaks1(
  name = "skylark_breaks1_works_with_local_tag",
  input = "a.txt",
  output = "skylark_breaks1_works_with_local_tag.txt",
  action_tags = [ "local" ],
)

genrule(
  name = "breaks2",
  srcs = [ "a.txt" ],
  outs = [ "breaks2.txt" ],
  # The point of this test is to attempt to read something from the filesystem
  # that is blocked via --sandbox_block_path= and thus should't be accessible.
  #
  # /var/log is an arbitrary choice of directory that should exist on all Linux
  # systems.
  #
  cmd = "ls /var/log &> $@",
)

genrule(
  name = "breaks3",
  srcs = [ "cyclic1", "cyclic2" ],
  outs = [ "breaks3.txt" ],
  cmd = "wc $(location :cyclic1) > $@",
)

genrule(
  name = "check_sandbox_contain_WORKSPACE",
  outs = [ "check_sandbox_contain_WORKSPACE.txt" ],
  cmd = "ls -l $$(dirname \"$$(pwd)\") &> $@",
)

genrule(
  name = "check_proc_works",
  outs = [ "check_proc_works.txt" ],
  cmd = "sh -c 'cd /proc/self && echo $$$$ && exec cat stat | sed \"s/\\([^ ]*\\) .*/\\1/g\"' > $@",
)
EOF
  cat << 'EOF' >> examples/genrule/datafile
this is a datafile
EOF
  # The workspace name is initialized in testenv.sh; use that var rather than
  # hardcoding it here. The extra sed pass is so we can selectively expand that
  # one var while keeping the rest of the heredoc literal.
  cat | sed "s/{{WORKSPACE_NAME}}/$WORKSPACE_NAME/" >> examples/genrule/tool.sh << 'EOF'
#!/bin/sh

set -e
cp $(dirname $0)/tool.runfiles/{{WORKSPACE_NAME}}/examples/genrule/datafile $1
echo "Tools work!"
EOF
  chmod +x examples/genrule/tool.sh
  cat << 'EOF' >> examples/genrule/skylark.bzl
def _skylark_breaks1_impl(ctx):
  print(ctx.outputs.output.path)
  ctx.actions.run_shell(
    inputs = [ ctx.file.input ],
    outputs = [ ctx.outputs.output ],
    command = "wc %s `dirname %s`/b.txt &> %s" % (ctx.file.input.path,
                                                 ctx.file.input.path,
                                                 ctx.outputs.output.path),
    execution_requirements = { tag: '' for tag in ctx.attr.action_tags },
  )

skylark_breaks1 = rule(
  _skylark_breaks1_impl,
  attrs = {
    "input": attr.label(mandatory=True, allow_files=True, single_file=True),
    "output": attr.output(mandatory=True),
    "action_tags": attr.string_list(),
  },
)
EOF
}

function test_sandboxed_genrule() {
  bazel build examples/genrule:works &> $TEST_log \
    || fail "Hermetic genrule failed: examples/genrule:works"
  [ -f "${BAZEL_GENFILES_DIR}/examples/genrule/works.txt" ] \
    || fail "Genrule did not produce output: examples/genrule:works"
}

function test_sandboxed_tooldir() {
  bazel build examples/genrule:tooldir &> $TEST_log \
    || fail "Hermetic genrule failed: examples/genrule:tooldir"
  [ -f "${BAZEL_GENFILES_DIR}/examples/genrule/tooldir.txt" ] \
    || fail "Genrule did not produce output: examples/genrule:works"
  cat "${BAZEL_GENFILES_DIR}/examples/genrule/tooldir.txt" > $TEST_log
  expect_log "genrule-setup.sh"
}

function test_sandboxed_genrule_with_tools() {
  bazel build examples/genrule:tools_work &> $TEST_log \
    || fail "Hermetic genrule failed: examples/genrule:tools_work"
  [ -f "${BAZEL_GENFILES_DIR}/examples/genrule/tools.txt" ] \
    || fail "Genrule did not produce output: examples/genrule:tools_work"
}

# Make sure that sandboxed execution doesn't accumulate files in the
# sandbox directory.
function test_sandbox_cleanup() {
  bazel --batch clean &> $TEST_log \
    || fail "bazel clean failed"
  bazel build examples/genrule:tools_work &> $TEST_log \
    || fail "Hermetic genrule failed: examples/genrule:tools_work"
  bazel shutdown &> $TEST_log || fail "bazel shutdown failed"
  if [[ -n "$(ls -A "$(bazel info output_base)/sandbox")" ]]; then
    fail "Build left files around afterwards"
  fi
}

# Test for #400: Linux sandboxing and relative symbolic links.
#
# let A = examples/genrule/symlinks/a/b/x.txt -> ../x.txt
# where   examples/genrule/symlinks/a/b -> examples/genrule/symlinks/ok/sub
# thus the realpath of A is example/genrule/symlinks/ok/x.txt
# but if the code doesn't correctly resolve intermediate symlinks and instead
# uses string operations to handle ".." parts, it will arrive at:
# examples/genrule/symlinks/a/x.txt, which is wrong.
#
function test_sandbox_relative_symlink_in_inputs() {
  bazel build examples/genrule:relative_symlinks &> $TEST_log \
    || fail "Hermetic genrule failed: examples/genrule:relative_symlinks"
  [ -f "${BAZEL_GENFILES_DIR}/examples/genrule/relative_symlinks.txt" ] \
    || fail "Genrule did not produce output: examples/genrule:relative_symlinks"
}

function test_sandbox_undeclared_deps() {
  output_file="${BAZEL_GENFILES_DIR}/examples/genrule/breaks1.txt"

  bazel build examples/genrule:breaks1 &> $TEST_log \
    && fail "Non-hermetic genrule succeeded: examples/genrule:breaks1" || true

  [ -f "$output_file" ] ||
    fail "Action did not produce output: $output_file"

  if [ $(wc -l $output_file) -gt 1 ]; then
    fail "Output contained more than one line: $output_file"
  fi

  fgrep "No such file or directory" $output_file ||
    fail "Output did not contain expected error message: $output_file"
}

function test_sandbox_undeclared_deps_with_local() {
  bazel build examples/genrule:breaks1_works_with_local &> $TEST_log \
    || fail "Non-hermetic genrule failed even though local=1: examples/genrule:breaks1_works_with_local"
  [ -f "${BAZEL_GENFILES_DIR}/examples/genrule/breaks1_works_with_local.txt" ] \
    || fail "Genrule did not produce output: examples/genrule:breaks1_works_with_local"
}

function test_sandbox_undeclared_deps_with_local_tag() {
  bazel build examples/genrule:breaks1_works_with_local_tag &> $TEST_log \
    || fail "Non-hermetic genrule failed even though tags=['local']: examples/genrule:breaks1_works_with_local_tag"
  [ -f "${BAZEL_GENFILES_DIR}/examples/genrule/breaks1_works_with_local_tag.txt" ] \
    || fail "Genrule did not produce output: examples/genrule:breaks1_works_with_local_tag"
}

function test_sandbox_undeclared_deps_skylark() {
  output_file="${BAZEL_BIN_DIR}/examples/genrule/skylark_breaks1.txt"
  bazel build examples/genrule:skylark_breaks1 &> $TEST_log \
    && fail "Non-hermetic genrule succeeded: examples/genrule:skylark_breaks1" || true

  [ -f "$output_file" ] ||
    fail "Action did not produce output: $output_file"

  if [ $(wc -l $output_file) -gt 1 ]; then
    fail "Output contained more than one line: $output_file"
  fi

  fgrep "No such file or directory" $output_file ||
    fail "Output did not contain expected error message: $output_file"
}

function test_sandbox_undeclared_deps_skylark_with_local_tag() {
  bazel build examples/genrule:skylark_breaks1_works_with_local_tag &> $TEST_log \
    || fail "Non-hermetic genrule failed even though tags=['local']: examples/genrule:skylark_breaks1_works_with_local_tag"
  [ -f "${BAZEL_BIN_DIR}/examples/genrule/skylark_breaks1_works_with_local_tag.txt" ] \
    || fail "Action did not produce output: examples/genrule:skylark_breaks1_works_with_local_tag"
}

function test_sandbox_block_filesystem() {
  output_file="${BAZEL_GENFILES_DIR}/examples/genrule/breaks2.txt"

  bazel build --sandbox_block_path=/var/log examples/genrule:breaks2 &> $TEST_log \
    && fail "Non-hermetic genrule succeeded: examples/genrule:breaks2" || true

  [ -f "$output_file" ] ||
    fail "Action did not produce output: $output_file"

  if [ $(wc -l $output_file) -gt 1 ]; then
    fail "Output contained more than one line: $output_file"
  fi

  fgrep "Permission denied" $output_file ||
    fail "Output did not contain expected error message: $output_file"
}

function test_sandbox_cyclic_symlink_in_inputs() {
  bazel build examples/genrule:breaks3 &> $TEST_log \
    && fail "Genrule with cyclic symlinks succeeded: examples/genrule:breaks3" || true
  [ ! -f "${BAZEL_GENFILES_DIR}/examples/genrule/breaks3.txt" ] || {
    output=$(cat "${BAZEL_GENFILES_DIR}/examples/genrule/breaks3.txt")
    fail "Genrule with cyclic symlinks breaks3 succeeded with following output: $output"
  }
}

function test_sandbox_network_access() {
  serve_file file_to_serve
  cat << EOF >> examples/genrule/BUILD

genrule(
  name = "sandbox_network_access",
  outs = [ "sandbox_network_access.txt" ],
  cmd = "curl -o \$@ localhost:${nc_port}",
)
EOF
  bazel build examples/genrule:sandbox_network_access &> $TEST_log \
    || fail "genrule 'sandbox_network_access' trying to use network failed, but should have succeeded"
  [ -f "${BAZEL_GENFILES_DIR}/examples/genrule/sandbox_network_access.txt" ] \
    || fail "genrule 'sandbox_network_access' did not produce output"
  kill_nc
}

function test_sandbox_block_network_access() {
  serve_file file_to_serve
  cat << EOF >> examples/genrule/BUILD

genrule(
  name = "breaks4",
  outs = [ "breaks4.txt" ],
  cmd = "curl -o \$@ localhost:${nc_port}",
)
EOF
  bazel build --experimental_sandbox_default_allow_network=false examples/genrule:breaks1 &> $TEST_log \
    && fail "Non-hermetic genrule succeeded: examples/genrule:breaks4" || true
  [ ! -f "${BAZEL_GENFILES_DIR}/examples/genrule/breaks4.txt" ] || {
    output=$(cat "${BAZEL_GENFILES_DIR}/examples/genrule/breaks4.txt")
    fail "Non-hermetic genrule breaks1 succeeded with following output: $output"
  }
  kill_nc
}

function test_sandbox_network_access_with_local() {
  serve_file file_to_serve
  cat << EOF >> examples/genrule/BUILD

genrule(
  name = "sandbox_network_access_with_local",
  outs = [ "sandbox_network_access_with_local.txt" ],
  cmd = "curl -o \$@ localhost:${nc_port}",
  tags = [ "local" ],
)
EOF
  bazel build examples/genrule:sandbox_network_access_with_local &> $TEST_log \
    || fail "genrule 'sandbox_network_access_with_local' trying to use network failed, but should have succeeded"
  [ -f "${BAZEL_GENFILES_DIR}/examples/genrule/sandbox_network_access_with_local.txt" ] \
    || fail "genrule 'sandbox_network_access_with_local' did not produce output"
  kill_nc
}

function test_sandbox_network_access_with_requires_network() {
  serve_file file_to_serve
  cat << EOF >> examples/genrule/BUILD

genrule(
  name = "sandbox_network_access_with_requires_network",
  outs = [ "sandbox_network_access_with_requires_network.txt" ],
  cmd = "curl -o \$@ localhost:${nc_port}",
  tags = [ "requires-network" ],
)
EOF
  bazel build --experimental_sandbox_default_allow_network=false \
    examples/genrule:sandbox_network_access_with_requires_network &> $TEST_log \
    || fail "genrule failed even though tags=['requires-network']: \
    examples/genrule:breaks4_works_with_requires_network"
  [ -f "${BAZEL_GENFILES_DIR}/examples/genrule/sandbox_network_access_with_requires_network.txt" ] \
    || fail "Genrule did not produce output: examples/genrule:sandbox_network_access_with_requires_network.txt"
  kill_nc
}

function test_sandbox_network_access_with_block_network() {
  serve_file file_to_serve
  cat << EOF >> examples/genrule/BUILD

genrule(
  name = "sandbox_network_access_with_block_network",
  outs = [ "sandbox_network_access_with_block_network.txt" ],
  cmd = "curl -o \$@ localhost:${nc_port}",
  tags = [ "block-network" ],
)
EOF
  bazel build --experimental_sandbox_default_allow_network=true examples/genrule:sandbox_network_access_with_block_network &> $TEST_log \
    && fail "genrule 'sandbox_network_access_with_block_network' trying to use network succeeded, but should have failed" || true
  [ ! -f "${BAZEL_GENFILES_DIR}/examples/genrule/sandbox_network_access_with_block_network.txt" ] \
    || fail "genrule 'sandbox_network_access_with_block_network' produced output, but was expected to fail"
  kill_nc
}

function test_sandbox_can_resolve_own_hostname() {
  setup_javatest_support
  mkdir -p src/test/java/com/example
  cat > src/test/java/com/example/HostNameTest.java <<'EOF'
package com.example;

import static org.junit.Assert.*;

import org.junit.Test;
import java.net.*;
import java.io.*;

public class HostNameTest {
  @Test
  public void testGetHostName() throws Exception {
    // This will throw an exception, if the local hostname cannot be resolved via DNS.
    assertNotNull(InetAddress.getLocalHost().getHostName());
  }
}
EOF
  cat > src/test/java/com/example/BUILD <<'EOF'
java_test(
  name = "HostNameTest",
  srcs = ["HostNameTest.java"],
  deps = ['//third_party:junit4'],
)
EOF

  bazel test --test_output=streamed src/test/java/com/example:HostNameTest &> $TEST_log \
    || fail "test should have passed"
}

function test_hostname_inside_sandbox_is_localhost_when_using_sandbox_fake_hostname_flag() {
  setup_javatest_support
  mkdir -p src/test/java/com/example
  cat > src/test/java/com/example/HostNameIsLocalhostTest.java <<'EOF'
package com.example;

import static org.junit.Assert.*;

import org.junit.Test;
import java.net.*;
import java.io.*;

public class HostNameIsLocalhostTest {
  @Test
  public void testHostNameIsLocalhost() throws Exception {
    // This will throw an exception, if the local hostname cannot be resolved via DNS.
    assertEquals(InetAddress.getLocalHost().getHostName(), "localhost");
  }
}
EOF
  cat > src/test/java/com/example/BUILD <<'EOF'
java_test(
  name = "HostNameIsLocalhostTest",
  srcs = ["HostNameIsLocalhostTest.java"],
  deps = ['//third_party:junit4'],
)
EOF

  bazel test --sandbox_fake_hostname --test_output=streamed src/test/java/com/example:HostNameIsLocalhostTest &> $TEST_log \
    || fail "test should have passed"
}

# TODO(philwo) - this doesn't work on Ubuntu 14.04 due to "unshare" being too
# old and not understanding the --user flag.
function DISABLED_test_sandbox_different_nobody_uid() {
  cat /etc/passwd | sed 's/\(^nobody:[^:]*:\)[0-9]*:[0-9]*/\15000:16000/g' > \
      "${TEST_TMPDIR}/passwd"
  unshare --user --mount --map-root-user -- bash - \
      << EOF || fail "Hermetic genrule with different UID for nobody failed" \
set -e
set -u

mount --bind ${TEST_TMPDIR}/passwd /etc/passwd
bazel build examples/genrule:works &> ${TEST_log}
EOF
}

function test_requires_root() {
  cat > test.sh <<'EOF'
#!/bin/sh
([ $(id -u) = "0" ] && [ $(id -g) = "0" ]) || exit 1
EOF
  chmod +x test.sh
  cat > BUILD <<'EOF'
sh_test(
  name = "test",
  srcs = ["test.sh"],
  tags = ["requires-fakeroot"],
)
EOF
  bazel test --test_output=errors :test || fail "test did not pass"
  bazel test --nocache_test_results --sandbox_fake_username --test_output=errors :test || fail "test did not pass"
}

# Tests that /proc/self == /proc/$$. This should always be true unless the PID namespace is active without /proc being remounted correctly.
function test_sandbox_proc_self() {
  bazel build examples/genrule:check_proc_works >& $TEST_log || fail "build should have succeeded"

  (
    # Catch the head and tail commands failing.
    set -e
    if [[ "$(head -n1 "${BAZEL_GENFILES_DIR}/examples/genrule/check_proc_works.txt")" \
          != "$(tail -n1 "${BAZEL_GENFILES_DIR}/examples/genrule/check_proc_works.txt")" ]] ; then
      fail "Reading PID from /proc/self/stat should have worked, instead have these: $(cat "${BAZEL_GENFILES_DIR}/examples/genrule/check_proc_works.txt")"
    fi
  )
}

function test_succeeding_action_with_ioexception_while_copying_outputs_throws_correct_exception() {
  cat > BUILD <<'EOF'
genrule(
  name = "test",
  outs = ["readonlydir/output.txt"],
  cmd = "touch $(location readonlydir/output.txt); chmod 0 $(location readonlydir/output.txt); chmod 0500 `dirname $(location readonlydir/output.txt)`",
)
EOF
  bazel build :test &> $TEST_log \
    && fail "build should have failed" || true

  # This is the generic "we caught an IOException" log message used by the
  # SandboxedStrategy. We don't want to see this in this case, because we have
  # special handling that prints a better error message and then lets the
  # sandbox code throw the actual ExecException.
  expect_not_log "I/O error during sandboxed execution"

  # There was no ExecException during sandboxed execution, because the action
  # returned an exit code of 0.
  expect_not_log "Executing genrule //:test failed: linux-sandbox failed: error executing command"

  # This is the error message telling us that some output artifacts couldn't be copied.
  expect_log "Could not move output artifacts from sandboxed execution"

  # The build fails, because the action didn't generate its output artifact.
  expect_log "ERROR:.*Executing genrule //:test failed"
}

function test_failing_action_with_ioexception_while_copying_outputs_throws_correct_exception() {
  cat > BUILD <<'EOF'
genrule(
  name = "test",
  outs = ["readonlydir/output.txt"],
  cmd = "touch $(location readonlydir/output.txt); chmod 0 $(location readonlydir/output.txt); chmod 0500 `dirname $(location readonlydir/output.txt)`; exit 1",
)
EOF
  bazel build :test &> $TEST_log \
    && fail "build should have failed" || true

  # This is the generic "we caught an IOException" log message used by the
  # SandboxedStrategy. We don't want to see this in this case, because we have
  # special handling that prints a better error message and then lets the
  # sandbox code throw the actual ExecException.
  expect_not_log "I/O error during sandboxed execution"

  # This is the error message printed by the EventHandler telling us that some
  # output artifacts couldn't be copied.
  expect_log "Could not move output artifacts from sandboxed execution"

  # This is the UserExecException telling us that the build failed.
  expect_log "Executing genrule //:test failed:"
}

function test_sandbox_debug() {
  cat > BUILD <<'EOF'
genrule(
  name = "broken",
  outs = ["bla.txt"],
  cmd = "exit 1",
)
EOF
  bazel build --verbose_failures :broken &> $TEST_log \
    && fail "build should have failed" || true
  expect_log "Use --sandbox_debug to see verbose messages from the sandbox"
  expect_log "Executing genrule //:broken failed"

  bazel build --verbose_failures --sandbox_debug :broken &> $TEST_log \
    && fail "build should have failed" || true
  expect_log "Executing genrule //:broken failed"
  expect_not_log "Use --sandbox_debug to see verbose messages from the sandbox"
  # This will appear a lot in the sandbox failure details.
  expect_log "/sandbox/"  # Part of the path to the sandbox location.
}

function test_sandbox_mount_customized_path () {
  if ! [ "${PLATFORM-}" = "linux" -a \
    "$(cat /dev/null /etc/*release | grep 'DISTRIB_CODENAME=' | sed 's/^.*=//')" = "trusty" ]; then
    echo "Skipping test: the toolchain used in this test is only supported on trusty."
    return 0
  fi

  # Create BUILD file
  cat > BUILD <<'EOF'
package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
)
EOF

  # Create cc file
  cat > hello-world.cc << 'EOF'
#include <iostream>
int main(int argc, char** argv) {
  std::cout << "Hello, world!" << std::endl;
  return 0;
}
EOF

  # Create WORKSPACE file
  cat > WORKSPACE <<'EOF'
local_repository(
  name = 'x86_64_unknown_linux_gnu',
  path = './downloaded_toolchain',
)
EOF

  # Define the mount source and target.
  source="${TEST_TMPDIR}/workspace/downloaded_toolchain/x86_64-unknown-linux-gnu/sysroot/lib64/ld-2.19.so"
  target_root="${TEST_SRCDIR}/mount_targets"
  target_folder="${target_root}/x86_64-unknown-linux-gnu/sysroot/lib64"
  target="${target_folder}/ld-2.19.so"

  # Unpack the toolchain.
  mkdir downloaded_toolchain
  tar -xf ${TEST_SRCDIR}/mount_path_toolchain/file/mount_path_toolchain.tar.gz -C ./downloaded_toolchain
  chmod -R 0755 downloaded_toolchain

  # Create an empty WORKSPACE file for the local repository.
  touch downloaded_toolchain/WORKSPACE

  # Replace the target_root_placeholder with the actual target_root
  sed -i "s|target_root_placeholder|$target_root|g" downloaded_toolchain/CROSSTOOL
  sed -i "s|name = 'cc-compiler-k8',|name = 'cc-compiler-k8', toolchain_identifier = 'linux_gnu_x86',|g" downloaded_toolchain/BUILD
  sed -i "s/k8|gcc/k8/g" downloaded_toolchain/BUILD

  # Prepare the bazel command flags
  flags="--crosstool_top=@x86_64_unknown_linux_gnu//:toolchain --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --verbose_failures --spawn_strategy=sandboxed"
  flags="${flags} --sandbox_add_mount_pair=${source}:${target}"

  # Execute the bazel build command without creating the target. Should fail.
  bazel clean --expunge &> $TEST_log
  bazel build $flags //:hello-world &> $TEST_log && fail "Should fail"
  expect_log "Bazel only supports bind mounting on top of existing files/directories."

  # Create the mount target manually as Bazel does not create target paths
  mkdir -p ${target_folder}
  touch ${target}

  # Execute bazel build command again. Should build.
  bazel clean --expunge &> $TEST_log
  bazel build $flags //:hello-world &> $TEST_log || fail "Should build"

  # Remove the mount target folder as Bazel does not do the cleanup
  rm -rf ${target_root}/x86_64-unknown-linux-gnu

  # Assert that output binary exists
  test -f bazel-bin/hello-world || fail "output not found"

  # Use linux_sandbox binary to run bazel-bin/hello-world binary in the sandbox environment
  # First, no path mounting. The execution should fail.
  echo "Run the binary bazel-bin/hello-world without mounting the path"
  $linux_sandbox -D -- bazel-bin/hello-world &> $TEST_log || code=$?
  expect_log "child exited normally with exitcode 1"

  # Second, with path mounting. The execution should succeed.
  echo "Run the binary bazel-bin/hello-world with mounting the path"
  # Create the mount target manually as sandbox binary does not create target paths
  mkdir -p ${target_folder}
  touch ${target}
  $linux_sandbox -D \
  -M ${source} \
  -m ${target} \
  -- bazel-bin/hello-world &> $TEST_log || code=$?
  expect_log "Hello, world!"
  expect_log "child exited normally with exitcode 0"

  # Remove the mount target folder as sandbox binary does not do the cleanup
  rm -rf ${target_root}/x86_64-unknown-linux-gnu
}

function test_experimental_symlinked_sandbox_uses_expanded_tree_artifacts_in_runfiles_tree() {
  touch WORKSPACE

  cat > def.bzl <<'EOF'
def _mkdata_impl(ctx):
    out = ctx.actions.declare_directory(ctx.label.name + ".d")
    script = "mkdir -p {out}; touch {out}/file; ln -s file {out}/link".format(out = out.path)
    ctx.actions.run_shell(
        outputs = [out],
        command = script,
    )
    runfiles = ctx.runfiles(files = [out])
    return [DefaultInfo(
        files = depset([out]),
        runfiles = runfiles,
    )]

mkdata = rule(
    _mkdata_impl,
)
EOF

  cat > mkdata_test.sh <<'EOF'
#!/bin/bash

set -euo pipefail

test_dir="$1"
cd "$test_dir"
ls -l | cut -f1,9 -d' ' >&2

if [ ! -f file -o -L file ]; then
  echo "'file' is not a regular file" >&2
  exit 1
fi
EOF
  chmod +x mkdata_test.sh

  cat > BUILD <<'EOF'
load("//:def.bzl", "mkdata")

mkdata(name = "mkdata")

sh_test(
    name = "mkdata_test",
    srcs = ["mkdata_test.sh"],
    args = ["$(location :mkdata)"],
    data = [":mkdata"],
)
EOF

  bazel test --incompatible_symlinked_sandbox_expands_tree_artifacts_in_runfiles_tree \
      --test_output=streamed :mkdata_test &>$TEST_log && fail "expected test to fail" || true

  bazel test --noincompatible_symlinked_sandbox_expands_tree_artifacts_in_runfiles_tree \
      --test_output=streamed :mkdata_test &>$TEST_log || fail "expected test to pass"
}

# regression test for https://github.com/bazelbuild/bazel/issues/6262
function test_create_tree_artifact_inputs() {
  touch WORKSPACE

  cat > def.bzl <<'EOF'
def _r(ctx):
    d = ctx.actions.declare_directory("%s_dir" % ctx.label.name)
    ctx.actions.run_shell(
        outputs = [d],
        command = "cd %s && pwd" % d.path,
    )
    return [DefaultInfo(files = depset([d]))]

r = rule(implementation = _r)
EOF

cat > BUILD <<'EOF'
load(":def.bzl", "r")

r(name = "a")
EOF

  bazel build --test_output=streamed :a &>$TEST_log || fail "expected build to succeed"
}

# The test shouldn't fail if the environment doesn't support running it.
check_supported_platform || exit 0
check_sandbox_allowed || exit 0

run_suite "sandbox"
