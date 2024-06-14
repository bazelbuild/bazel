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
function set_up {
  add_to_bazelrc "build --spawn_strategy=sandboxed"
  add_to_bazelrc "build --genrule_strategy=sandboxed"

  # Enabled in testenv.sh.tmpl, but not in Bazel by default.
  sed -i.bak '/sandbox_tmpfs_path/d' "$bazelrc"
}

function assert_not_exists() {
  path="$1"
  [ ! -f "$path" ] && return 0

  fail "Expected file '$path' to not exist, but it did"
  return 1
}

function test_sandboxed_tooldir() {
  mkdir -p examples/genrule

  cat << 'EOF' > examples/genrule/BUILD
genrule(
   name = "tooldir",
   srcs = [],
   outs = ["tooldir.txt"],
   cmd = "ls -l external/bazel_tools/tools/genrule | tee $@ >&2; " +
       "cat external/bazel_tools/tools/genrule/genrule-setup.sh >&2",
)
EOF

  bazel build examples/genrule:tooldir &> $TEST_log \
    || fail "Hermetic genrule failed: examples/genrule:tooldir"
  [ -f "bazel-genfiles/examples/genrule/tooldir.txt" ] \
    || fail "Genrule did not produce output: examples/genrule:works"
  cat "bazel-genfiles/examples/genrule/tooldir.txt" > $TEST_log
  expect_log "genrule-setup.sh"
}

function test_sandbox_block_filesystem() {
  # The point of this test is to attempt to read something from the filesystem
  # that is blocked via --sandbox_block_path= and thus shouldn't be accessible.
  #
  # /var/log is an arbitrary choice of directory that should exist on all
  # Unix-like systems.
  local block_path
  case "$(uname -s)" in
    Darwin)
      # TODO(jmmv): sandbox-exec does not resolve symlinks, so attempting
      # to block /var/log does not work. Unsure if we should make this work
      # by resolving symlinks or documenting the expected behavior.
      block_path=/private/var/log
      ;;
    *)
      block_path=/var/log
      ;;
  esac

  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(
  name = "breaks",
  srcs = [ "a.txt" ],
  outs = [ "breaks.txt" ],
  cmd = "ls ${block_path} &> \$@",
)
EOF
  touch pkg/a.txt

  local output_file="bazel-genfiles/pkg/breaks.txt"

  bazel build --sandbox_block_path="${block_path}" \
    --sandbox_block_path=/doesnotexist pkg:breaks \
    &> $TEST_log \
    && fail "Non-hermetic genrule succeeded: examples/genrule:breaks" || true

  [ -f "$output_file" ] ||
    fail "Action did not produce output: $output_file"
  cat "${output_file}" >$TEST_log

  if [ "$(wc -l $output_file | awk '{print $1}')" -gt 1 ]; then
    fail "Output contained more than one line: $output_file"
  fi

  grep -E "(Operation not permitted|Permission denied)" $output_file ||
    fail "Output did not contain expected error message: $output_file"
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

# Tests that a pseudoterminal can be opened in linux when --sandbox_explicit_pseudoterminal is active
function test_can_enable_pseudoterminals() {
  if [[ "$(uname -s)" != Linux ]]; then
    echo "Skipping test: flag intended for linux systems"
    return 0
  fi

  cat > test.py <<'EOF'
import pty
pty.openpty()
EOF
  cat > BUILD <<'EOF'
py_test(
  name = "test",
  srcs = ["test.py"],
)
EOF
  bazel test --sandbox_explicit_pseudoterminal --verbose_failures --sandbox_debug :test || fail "test did not pass"
}

function test_sandbox_debug() {
  if [[ "$PLATFORM" == "darwin" ]]; then
    # The process wrapper sandbox used in MacOS doesn't emit debug output.
    return 0
  fi

  cat > BUILD <<'EOF'
genrule(
  name = "broken",
  outs = ["bla.txt"],
  cmd = "exit 1",
)
EOF
  bazel build --verbose_failures :broken &> $TEST_log \
    && fail "build should have failed" || true
  expect_log "Use --sandbox_debug to see verbose messages from the sandbox and retain the sandbox build root for debugging"
  expect_log "Executing genrule //:broken failed"

  bazel build --verbose_failures --sandbox_debug :broken &> $TEST_log \
    && fail "build should have failed" || true
  expect_log "Executing genrule //:broken failed"
  expect_not_log "Use --sandbox_debug to see verbose messages from the sandbox and retain the sandbox build root for debugging"
  expect_log "child exited normally with code 1"

  bazel build --verbose_failures --sandbox_debug --experimental_use_hermetic_linux_sandbox :broken &> $TEST_log \
    && fail "build should have failed with hermetic sandbox" || true
  expect_log "child exited normally with code 1"

  bazel build --verbose_failures --sandbox_debug --incompatible_sandbox_hermetic_tmp :broken &> $TEST_log \
    && fail "build should have failed with hermetic sandbox /tmp" || true
  expect_log "child exited normally with code 1"
}

function test_sandbox_expands_tree_artifacts_in_runfiles_tree() {
  create_workspace_with_default_repos WORKSPACE

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

  bazel test --test_output=streamed //:mkdata_test &>$TEST_log && fail "expected test to fail" || true
  expect_log "'file' is not a regular file"
}

# Regression test for https://github.com/bazelbuild/bazel/issues/6262.
function test_create_tree_artifact_outputs() {
  create_workspace_with_default_repos WORKSPACE

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

# Regression test for https://github.com/bazelbuild/bazel/issues/20032.
function test_read_only_tree_artifact() {
  create_workspace_with_default_repos WORKSPACE

  cat > def.bzl <<'EOF'
def _r(ctx):
  d = ctx.actions.declare_directory(ctx.label.name)
  ctx.actions.run_shell(
    outputs = [d],
    command = "touch $1/file.txt && chmod -w $1",
    arguments = [d.path],
  )
  return DefaultInfo(files = depset([d]))

r = rule(_r)
EOF

  cat > BUILD <<'EOF'
load(":def.bzl", "r")

r(name = "a")
EOF

  bazel build --test_output=streamed :a &>$TEST_log || fail "expected build to succeed"
}

function test_empty_tree_artifact_as_inputs() {
  # Test that when an empty tree artifact is the input, an empty directory is
  # created in the sandbox for action to read.
  create_workspace_with_default_repos WORKSPACE

  mkdir -p pkg

  cat > pkg/def.bzl <<'EOF'
def _r(ctx):
    empty_d = ctx.actions.declare_directory("%s/empty_dir" % ctx.label.name)
    ctx.actions.run_shell(
        outputs = [empty_d],
        command = "mkdir -p %s" % empty_d.path,
    )
    f = ctx.actions.declare_file("%s/file" % ctx.label.name)
    ctx.actions.run_shell(
        inputs = [empty_d],
        outputs = [f],
        command = "touch %s && cd %s && pwd" % (f.path, empty_d.path),
    )
    return [DefaultInfo(files = depset([f]))]

r = rule(implementation = _r)
EOF

cat > pkg/BUILD <<'EOF'
load(":def.bzl", "r")

r(name = "a")
EOF

  bazel build //pkg:a &>$TEST_log || fail "expected build to succeed"
}

# Sets up targets under //test that, when building //test:all, verify that the
# sandbox setup ensures that /tmp contents written by one action are not visible
# to another action.
#
# Arguments:
#   - The path to a unique temporary directory under /tmp that a
#     file named "bazel_was_here" is written to in actions.
function setup_tmp_hermeticity_check() {
  local -r tmpdir=$1

  mkdir -p test
  cat > test/BUILD <<'EOF'
cc_binary(
    name = "create_file",
    srcs = ["create_file.cc"],
)

[
    genrule(
        name = "gen" + str(i),
        outs = ["gen{}.txt".format(i)],
        tools = [":create_file"],
        cmd = """
        path=$$($(location :create_file))
        cp "$$path" $@
        """,
    )
    for i in range(1, 3)
]
EOF
  cat > test/create_file.cc <<EOF
// Create a file in a fixed location only if it doesn't exist.
// Then write its path to stdout.
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int main() {
  if (mkdir("$tmpdir", 0755) < 0) {
    perror("mkdir");
    return 1;
  }
  int fd = open("$tmpdir/bazel_was_here", O_CREAT | O_EXCL | O_WRONLY, 0600);
  if (fd < 0) {
    perror("open");
    return 1;
  }
  if (write(fd, "HERMETIC\n", 9) != 9) {
    perror("write");
    return 1;
  }
  close(fd);
  printf("$tmpdir/bazel_was_here\n");
  return 0;
}
EOF
}

function test_add_mount_pair_tmp_source() {
  if [[ "$PLATFORM" == "darwin" ]]; then
    # Tests Linux-specific functionality
    return 0
  fi

  create_workspace_with_default_repos WORKSPACE

  local mounted=$(mktemp -d "/tmp/bazel_mounted.XXXXXXXX")
  trap "rm -fr $mounted" EXIT
  echo GOOD > "$mounted/data.txt"

  local tmp_dir=$(mktemp -d "/tmp/bazel_mounted.XXXXXXXX")
  trap "rm -fr $tmp_dir" EXIT
  setup_tmp_hermeticity_check "$tmp_dir"

  mkdir -p pkg
  cat > pkg/BUILD <<'EOF'
genrule(
    name = "gen",
    outs = ["gen.txt"],
    cmd = "cp /etc/data.txt $@",
)
EOF

  # This assumes the existence of /etc on the host system
  bazel build --sandbox_add_mount_pair="$mounted:/etc" \
    //pkg:gen //test:all || fail "build failed"
  assert_equals GOOD "$(cat bazel-bin/pkg/gen.txt)"
  assert_equals HERMETIC "$(cat bazel-bin/test/gen1.txt)"
  assert_equals HERMETIC "$(cat bazel-bin/test/gen2.txt)"
  assert_not_exists "$tmp_dir/bazel_was_here"
}

function test_add_mount_pair_tmp_target() {
  if [[ "$PLATFORM" == "darwin" ]]; then
    # Tests Linux-specific functionality
    return 0
  fi

  create_workspace_with_default_repos WORKSPACE

  local source_dir=$(mktemp -d "/tmp/bazel_mounted.XXXXXXXX")
  trap "rm -fr $source_dir" EXIT
  echo BAD > "$source_dir/data.txt"

  local tmp_dir=$(mktemp -d "/tmp/bazel_mounted.XXXXXXXX")
  trap "rm -fr $tmp_dir" EXIT
  setup_tmp_hermeticity_check "$tmp_dir"

  mkdir -p pkg
  cat > pkg/BUILD <<EOF
genrule(
    name = "gen",
    outs = ["gen.txt"],
    cmd = """ls "$source_dir" > \$@""",
)
EOF


  # This assumes the existence of /etc on the host system
  bazel build --sandbox_add_mount_pair="/etc:$source_dir" \
    //pkg:gen //test:all || fail "build failed"
  assert_contains passwd bazel-bin/pkg/gen.txt
  assert_not_contains data.txt bazel-bin/pkg/gen.txt
  assert_equals HERMETIC "$(cat bazel-bin/test/gen1.txt)"
  assert_equals HERMETIC "$(cat bazel-bin/test/gen2.txt)"
  assert_not_exists "$tmp_dir/bazel_was_here"
}

function test_add_mount_pair_tmp_target_and_source() {
  if [[ "$PLATFORM" == "darwin" ]]; then
    # Tests Linux-specific functionality
    return 0
  fi

  create_workspace_with_default_repos WORKSPACE

  local mounted=$(mktemp -d "/tmp/bazel_mounted.XXXXXXXX")
  trap "rm -fr $mounted" EXIT
  echo GOOD > "$mounted/data.txt"

  local tmp_dir=$(mktemp -d "/tmp/bazel_mounted.XXXXXXXX")
  trap "rm -fr $tmp_dir" EXIT
  setup_tmp_hermeticity_check "$tmp_dir"

  mkdir -p pkg
  cat > pkg/BUILD <<EOF
genrule(
    name = "gen",
    outs = ["gen.txt"],
    cmd = """cp "$mounted/data.txt" \$@""",
)
EOF

  bazel build --sandbox_add_mount_pair="$mounted" \
    //pkg:gen //test:all || fail "build failed"
  assert_equals GOOD "$(cat bazel-bin/pkg/gen.txt)"
  assert_equals HERMETIC "$(cat bazel-bin/test/gen1.txt)"
  assert_equals HERMETIC "$(cat bazel-bin/test/gen2.txt)"
  assert_not_exists "$tmp_dir/bazel_was_here"
}

function test_symlink_with_output_base_under_tmp() {
  if [[ "$PLATFORM" == "darwin" ]]; then
    # Tests Linux-specific functionality
    return 0
  fi

  local repo=$(mktemp -d "/tmp/bazel_mounted.XXXXXXXX")
  trap "rm -fr $repo" EXIT

  touch WORKSPACE

  mkdir -p $repo/pkg
  touch $repo/WORKSPACE
  cat > $repo/pkg/es1 <<'EOF'
EXTERNAL_SOURCE_CONTENT
EOF
  cat > $repo/pkg/BUILD <<'EOF'
exports_files(["es1"])
genrule(
    name="er1",
    srcs=[],
    outs=[":er1"],
    cmd="echo EXTERNAL_GEN_CONTENT > $@",
    visibility=["//visibility:public"],
)
EOF

  mkdir -p $repo/examples
  cd $repo/examples || fail "cd $repo/examples failed"

  cat > WORKSPACE <<EOF
local_repository(
    name = "repo",
    path = "$repo",
)
EOF

  mkdir -p pkg
  cat > pkg/s1 <<'EOF'
SOURCE_CONTENT
EOF
  cat > pkg/BUILD <<'EOF'
load(":r.bzl", "symlink_rule")

genrule(name="r1", srcs=[], outs=[":r1"], cmd="echo GEN_CONTENT > $@")
symlink_rule(name="r2", input=":r1")
genrule(name="r3", srcs=[":r2"], outs=[":r3"], cmd="cp $< $@")
symlink_rule(name="s2", input=":s1")
genrule(name="s3", srcs=[":s2"], outs=[":s3"], cmd="cp $< $@")
symlink_rule(name="er2", input="@repo//pkg:er1")
genrule(name="er3", srcs=[":er2"], outs=[":er3"], cmd="cp $< $@")
symlink_rule(name="es2", input="@repo//pkg:es1")
genrule(name="es3", srcs=[":es2"], outs=[":es3"], cmd="cp $< $@")
EOF

  cat > pkg/r.bzl <<'EOF'
def _symlink_impl(ctx):
  output = ctx.actions.declare_file(ctx.label.name)
  ctx.actions.symlink(output = output, target_file = ctx.file.input)
  return [DefaultInfo(files = depset([output]))]

symlink_rule = rule(
  implementation = _symlink_impl,
  attrs = {"input": attr.label(allow_single_file=True)})
EOF

  local tmp_output_base=$(mktemp -d "/tmp/bazel_output_base.XXXXXXXX")
  trap "chmod -R u+w $tmp_output_base && rm -fr $tmp_output_base" EXIT

  bazel --output_base="$tmp_output_base" build //pkg:{er,es,r,s}3 --sandbox_debug
  assert_contains EXTERNAL_GEN_CONTENT bazel-bin/pkg/er3
  assert_contains EXTERNAL_SOURCE_CONTENT bazel-bin/pkg/es3
  assert_contains GEN_CONTENT bazel-bin/pkg/r3
  assert_contains SOURCE_CONTENT bazel-bin/pkg/s3
  bazel --output_base="$tmp_output_base" shutdown
}

function test_symlink_to_directory_with_output_base_under_tmp() {
  if [[ "$PLATFORM" == "darwin" ]]; then
    # Tests Linux-specific functionality
    return 0
  fi

  create_workspace_with_default_repos WORKSPACE

  mkdir -p pkg
  cat > pkg/BUILD <<'EOF'
load(":r.bzl", "symlink_rule", "tree_rule")

tree_rule(name="t1")
symlink_rule(name="t2", input=":t1")
genrule(name="t3", srcs=[":t2"], outs=[":t3"], cmd=";\n".join(
    ["cat $(location :t2)/{a/a,b/b} > $@"]))
EOF

  cat > pkg/r.bzl <<'EOF'
def _symlink_impl(ctx):
  output = ctx.actions.declare_directory(ctx.label.name)
  ctx.actions.symlink(output = output, target_file = ctx.file.input)
  return [DefaultInfo(files = depset([output]))]

symlink_rule = rule(
  implementation = _symlink_impl,
  attrs = {"input": attr.label(allow_single_file=True)})

def _tree_impl(ctx):
  output = ctx.actions.declare_directory(ctx.label.name)
  ctx.actions.run_shell(
    outputs = [output],
    command = "export TREE=%s && mkdir $TREE/a $TREE/b && echo -n A > $TREE/a/a && echo -n B > $TREE/b/b" % output.path)
  return [DefaultInfo(files = depset([output]))]

tree_rule = rule(
  implementation = _tree_impl,
  attrs = {})

EOF

  local tmp_output_base=$(mktemp -d "/tmp/bazel_output_base.XXXXXXXX")
  trap "chmod -R u+w $tmp_output_base && rm -fr $tmp_output_base" EXIT

  bazel --output_base="$tmp_output_base" build //pkg:t3
  assert_contains AB bazel-bin/pkg/t3
  bazel --output_base="$tmp_output_base" shutdown
}

function test_tmpfs_path_under_tmp() {
  if [[ "$PLATFORM" == "darwin" ]]; then
    # Tests Linux-specific functionality
    return 0
  fi

  create_workspace_with_default_repos WORKSPACE

  local tmpfs=$(mktemp -d "/tmp/bazel_tmpfs.XXXXXXXX")
  trap "rm -fr $tmpfs" EXIT
  echo BAD > "$tmpfs/data.txt"

  local tmp_dir=$(mktemp -d "/tmp/bazel_mounted.XXXXXXXX")
  trap "rm -fr $tmp_dir" EXIT
  setup_tmp_hermeticity_check "$tmp_dir"

  mkdir -p pkg
  cat > pkg/BUILD <<EOF
genrule(
    name = "gen",
    outs = ["gen.txt"],
    cmd = """
# Verify that the tmpfs under /tmp exists and is empty.
[[ -d "$tmpfs" ]]
[[ ! -e "$tmpfs/data.txt" ]]
# Verify that the tmpfs on /etc exists and is empty.
[[ -d /etc ]]
[[ -z "\$\$(ls -A /etc)" ]]
touch \$@
""",
)
EOF

  # This assumes the existence of /etc on the host system
  bazel build --sandbox_tmpfs_path="$tmpfs" --sandbox_tmpfs_path=/etc \
    //pkg:gen //test:all || fail "build failed"
  assert_equals HERMETIC "$(cat bazel-bin/test/gen1.txt)"
  assert_equals HERMETIC "$(cat bazel-bin/test/gen2.txt)"
  assert_not_exists "$tmp_dir/bazel_was_here"
}

function test_hermetic_tmp_under_tmp {
  if [[ "$(uname -s)" != Linux ]]; then
    echo "Skipping test: --incompatible_sandbox_hermetic_tmp is only supported in Linux" 1>&2
    return 0
  fi

  temp_dir=$(mktemp -d /tmp/test.XXXXXX)
  trap 'rm -rf ${temp_dir}' EXIT

  mkdir -p "${temp_dir}/workspace/a"
  mkdir -p "${temp_dir}/package-path/b"
  mkdir -p "${temp_dir}/repo/c"
  mkdir -p "${temp_dir}/output-base"

  cd "${temp_dir}/workspace"
  cat > WORKSPACE <<EOF
local_repository(name="repo", path="${temp_dir}/repo")
EOF

  cat > a/BUILD <<'EOF'
genrule(
  name = "g",
  outs = ["go"],
  srcs = [],
  cmd = "echo GO > $@",
)
sh_binary(
  name = "bin",
  srcs = ["bin.sh"],
  data = [":s", ":go", "//b:s", "//b:go", "@repo//c:s", "@repo//c:go"],
)
genrule(
  name = "t",
  tools = [":bin"],
  srcs = [":s", ":go", "//b:s", "//b:go", "@repo//c:s", "@repo//c:go"],
  outs = ["to"],
  cmd = "\n".join([
    "RUNFILES=$(location :bin).runfiles/_main",
    "S=$(location :s); GO=$(location :go)",
    "BS=$(location //b:s); BGO=$(location //b:go)",
    "RS=$(location @repo//c:s); RGO=$(location @repo//c:go)",
    "for i in $$S $$GO $$BS $$BGO $$RS $$RGO; do",
    "  echo reading $$i",
    "  cat $$i >> $@",
    "done",
    "for i in a/s a/go b/s b/go ../repo/c/s ../repo/c/go; do",
    "  echo reading $$RUNFILES/$$i",
    "  cat $$RUNFILES/$$i >> $@",
    "done",
  ]),
)
EOF

  touch a/bin.sh
  chmod +x a/bin.sh

  touch ../repo/WORKSPACE
  cat > ../repo/c/BUILD <<'EOF'
exports_files(["s"])
genrule(
  name = "g",
  outs = ["go"],
  srcs = [],
  cmd = "echo GO > $@",
  visibility = ["//visibility:public"],
)
EOF

  cat > ../package-path/b/BUILD <<'EOF'
exports_files(["s"])
genrule(
  name = "g",
  outs = ["go"],
  srcs = [],
  cmd = "echo GO > $@",
  visibility = ["//visibility:public"],
)
EOF

  touch "a/s" "../package-path/b/s" "../repo/c/s"

  cat WORKSPACE
  bazel \
    --output_base="${temp_dir}/output-base" \
    build \
    --incompatible_sandbox_hermetic_tmp \
    --package_path="%workspace%:${temp_dir}/package-path" \
    //a:t || fail "build failed"
}

# Regression test for https://github.com/bazelbuild/bazel/issues/21215
function test_copy_input_symlinks() {
  create_workspace_with_default_repos WORKSPACE

  cat > MODULE.bazel <<'EOF'
repo = use_repo_rule("//pkg:repo.bzl", "repo")
repo(name = "some_repo")
EOF

  mkdir -p pkg
  cat > pkg/BUILD <<'EOF'
genrule(
    name = "copy_files",
    srcs = [
        "@some_repo//:files",
    ],
    outs = [
        "some_file1.json",
        "some_file2.json",
    ],
    cmd = "cp -r $(locations @some_repo//:files) $(RULEDIR)",
)
EOF
  cat > pkg/repo.bzl <<'EOF'
def _impl(rctx):
  rctx.file("some_file1.json", "hello")
  rctx.file("some_file2.json", "world")
  rctx.file("BUILD", """filegroup(
    name = "files",
    srcs = ["some_file1.json", "some_file2.json"],
    visibility = ["//visibility:public"],
)""")

repo = repository_rule(_impl)
EOF

  bazel build //pkg:copy_files || fail "build failed"
  assert_equals hello "$(cat bazel-bin/pkg/some_file1.json)"
  assert_equals world "$(cat bazel-bin/pkg/some_file2.json)"
}

# The test shouldn't fail if the environment doesn't support running it.
check_sandbox_allowed || exit 0

run_suite "sandbox"
