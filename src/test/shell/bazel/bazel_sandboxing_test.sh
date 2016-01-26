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
src_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source ${src_dir}/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }
source ${src_dir}/bazel_sandboxing_test_utils.sh \
  || { echo "bazel_sandboxing_test_utils.sh not found!" >&2; exit 1; }
source ${src_dir}/remote_helpers.sh \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }

cat >>$TEST_TMPDIR/bazelrc <<'EOF'
# Testing the sandboxed strategy requires using the sandboxed strategy. While it is the default,
# we want to make sure that this explicitly fails when the strategy is not available on the system
# running the test.
build --spawn_strategy=sandboxed --genrule_strategy=sandboxed
EOF

function set_up {
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

load('/examples/genrule/skylark', 'skylark_breaks1')

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
  # that resides outside the sandbox by using an absolute path to that file.
  #
  # /var/log is an arbitrary choice of directory (we don't mount it in the
  # sandbox and it should exist on every linux) which could be changed in
  # case it turns out it's necessary to put it in sandbox.
  #
  cmd = "ls /var/log &> $@",
)

genrule(
  name = "breaks3",
  srcs = [ "cyclic1", "cyclic2" ],
  outs = [ "breaks3.txt" ],
  cmd = "wc $(location :cyclic1) > $@",
)
EOF
  cat << 'EOF' >> examples/genrule/datafile
this is a datafile
EOF
  cat << 'EOF' >> examples/genrule/tool.sh
#!/bin/sh

set -e
cp $(dirname $0)/tool.runfiles/examples/genrule/datafile $1
echo "Tools work!"
EOF
  chmod +x examples/genrule/tool.sh
  cat << 'EOF' >> examples/genrule/skylark.bzl
def _skylark_breaks1_impl(ctx):
  print(ctx.outputs.output.path)
  ctx.action(
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
  ls -la "$(bazel info execution_root)/bazel-sandbox"
  if [[ "$(ls -A "$(bazel info execution_root)"/bazel-sandbox)" ]]; then
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

  bazel build examples/genrule:breaks2 &> $TEST_log \
    && fail "Non-hermetic genrule succeeded: examples/genrule:breaks2" || true

  [ -f "$output_file" ] ||
    fail "Action did not produce output: $output_file"

  if [ $(wc -l $output_file) -gt 1 ]; then
    fail "Output contained more than one line: $output_file"
  fi

  fgrep "No such file or directory" $output_file ||
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
  name = "breaks4",
  outs = [ "breaks4.txt" ],
  cmd = "curl -o \$@ localhost:${nc_port}",
)
EOF
  bazel build examples/genrule:breaks1 &> $TEST_log \
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
  name = "breaks4_works_with_local",
  outs = [ "breaks4_works_with_local.txt" ],
  cmd = "curl -o \$@ localhost:${nc_port}",
  tags = [ "local" ],
)
EOF
  bazel build examples/genrule:breaks4_works_with_local &> $TEST_log \
    || fail "Non-hermetic genrule failed even though tags=['local']: examples/genrule:breaks4_works_with_local"
  [ -f "${BAZEL_GENFILES_DIR}/examples/genrule/breaks4_works_with_local.txt" ] \
    || fail "Genrule did not produce output: examples/genrule:breaks4_works_with_local"
  kill_nc
}

function test_sandbox_network_access_with_requires_network() {
  serve_file file_to_serve
  cat << EOF >> examples/genrule/BUILD

genrule(
  name = "breaks4_works_with_requires_network",
  outs = [ "breaks4_works_with_requires_network.txt" ],
  cmd = "curl -o \$@ localhost:${nc_port}",
  tags = [ "requires-network" ],
)
EOF
  bazel build examples/genrule:breaks4_works_with_requires_network &> $TEST_log \
    || fail "Non-hermetic genrule failed even though tags=['requires-network']: examples/genrule:breaks4_works_with_requires_network"
  [ -f "${BAZEL_GENFILES_DIR}/examples/genrule/breaks4_works_with_requires_network.txt" ] \
    || fail "Genrule did not produce output: examples/genrule:breaks4_works_with_requires_network"
  kill_nc
}

# The test shouldn't fail if the environment doesn't support running it.
check_supported_platform || exit 0
check_sandbox_allowed || exit 0

run_suite "sandbox"
