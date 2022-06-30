#!/bin/bash
#
# Copyright 2019 The Bazel Authors. All rights reserved.
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
# Tests for symlinks in Bazel.

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

#### SETUP #############################################################

# `uname` returns the current platform, e.g "MSYS_NT-10.0" or "Linux".
# `tr` converts all upper case letters to lower case.
# `case` matches the result if the `uname | tr` expression to string prefixes
# that use the same wildcards as names do in Bash, i.e. "msys*" matches strings
# starting with "msys", and "*" matches everything (it's the default case).
case "$(uname -s | tr [:upper:] [:lower:])" in
msys*)
  # As of 2019-01-15, Bazel on Windows only supports MSYS Bash.
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  # Disable MSYS path conversion that converts path-looking command arguments to
  # Windows paths (even if they arguments are not in fact paths).
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

function expect_symlink() {
  local file=$1

  if "$is_windows"; then
    # By default, WindowsFileSystem#createSymbolicLink() copies instead of symlinking:
    # https://source.bazel.build/bazel/+/master:src/main/java/com/google/devtools/build/lib/windows/WindowsFileSystem.java;l=96;drc=e86a93b9fba865c3374a5a7ccdabc9863035ef78
    return 0
  fi
  if [[ ! -L "$file" ]]; then
    fail "expected '$file' to be a symlink"
  fi
}

function set_up() {
  mkdir -p symlink
  touch symlink/BUILD
  cat > symlink/symlink.bzl <<EOF
def _dangling_symlink_impl(ctx):
    symlink = ctx.actions.declare_symlink(ctx.label.name)
    ctx.actions.symlink(
        output = symlink,
        target_path = ctx.attr.link_target,
    )
    return DefaultInfo(files = depset([symlink]))

dangling_symlink = rule(
    implementation = _dangling_symlink_impl,
    attrs = {"link_target": attr.string()},
)

def _write_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(output, ctx.attr.contents)
    return DefaultInfo(files = depset([output]))

write = rule(implementation = _write_impl, attrs = {"contents": attr.string()})
EOF

}

function test_smoke() {
  mkdir -p a
  cat > a/BUILD <<EOF
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent")
EOF

  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:a || fail "build failed"
  ls -l bazel-bin/a
}

function test_inmemory_cache_symlinks() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF
  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_not_log "running genrule"
}

function test_on_disk_cache_symlinks() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF
  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
  bazel shutdown
  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_not_log "running genrule"
}

function test_no_inmemory_cache_symlinks() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"

  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent2")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
}

function test_no_on_disk_cache_symlinks() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"

  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent2")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel shutdown
  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
}

function test_replace_symlink_with_file() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"

  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "write")
write(name="a", contents="non/existent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
}

function test_replace_file_with_symlink() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "write")
write(name="a", contents="non/existent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"

  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
}

function test_file_instead_of_symlink() {
  mkdir -p a
  cat > a/MakeSymlink.java <<'EOF'
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
public class MakeSymlink {
  public static void main(String[] args) throws IOException {
    Files.createSymbolicLink(Paths.get(args[0]), Paths.get(args[1]));
  }
}
EOF

  cat > a/a.bzl <<'EOF'
def _bad_symlink_impl(ctx):
    symlink = ctx.actions.declare_symlink(ctx.label.name)
    # Oops, should be "symlink".
    ctx.actions.write(
        output = symlink,
        content = ctx.attr.link_target,
    )
    return DefaultInfo(files = depset([symlink]))

bad_symlink = rule(implementation = _bad_symlink_impl, attrs = {"link_target": attr.string()})

def _bad_write_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name)
    # Oops, should be "write"
    ctx.actions.symlink(
        output = output,
        target_path = ctx.attr.contents,
    )
    return DefaultInfo(files = depset([output]))

bad_write = rule(implementation = _bad_write_impl, attrs = {"contents": attr.string()})
EOF

  cat > a/BUILD <<'EOF'
load(":a.bzl", "bad_symlink", "bad_write")

java_binary(
    name = "MakeSymlink",
    srcs = ["MakeSymlink.java"],
    main_class = "MakeSymlink",
)

bad_symlink(name="bs", link_target="bad/symlink")
genrule(name="bsg", srcs=[":bs"], outs=["bsgo"], cmd="echo BSGO > $@")

bad_write(name="bw", contents="badcontents")
genrule(name="bwg", srcs=[":bw"], outs=["bwgo"], cmd="echo BWGO > $@")

genrule(
    name="bg",
    srcs=[],
    outs=["bgo"],
    exec_tools = [":MakeSymlink"],
    cmd = "$(execpath :MakeSymlink) $@ bad/symlink",
)
EOF

  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:bsg >& $TEST_log && fail "build succeeded"
  expect_log "declared output 'a/bs' is not a symlink"

  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:bwg >& $TEST_log && fail "build succeeded"
  expect_log "symlink() with \"target_path\" param requires that \"output\" be declared as a symlink, not a regular file"

  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:bg >& $TEST_log && fail "build succeeded"
  expect_log "declared output 'a/bgo' is a dangling symbolic link"
}

function test_symlink_instead_of_file() {
  mkdir -p a
  cat > a/a.bzl <<'EOF'
def _bad_symlink_impl(ctx):
    target = ctx.actions.declare_file(ctx.label.name + ".target")
    ctx.actions.write(
        output = target,
        content = "Hello, World!",
    )

    symlink = ctx.actions.declare_symlink(ctx.label.name + ".link")
    ctx.actions.symlink(
        output = symlink,
        target_file = target,
    )
    return DefaultInfo(files = depset([symlink]))

bad_symlink = rule(implementation = _bad_symlink_impl)
EOF

  cat > a/BUILD <<'EOF'
load(":a.bzl", "bad_symlink")

bad_symlink(name="bs")
EOF

  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:bs >& $TEST_log && fail "build succeeded"
  expect_log "symlink() with \"target_file\" param requires that \"output\" be declared as a regular file, not a symlink"
}

function test_symlink_created_from_spawn() {
  mkdir -p a
  cat > a/MakeSymlink.java <<'EOF'
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
public class MakeSymlink {
  public static void main(String[] args) throws IOException {
    Files.createSymbolicLink(Paths.get(args[0]), Paths.get(args[1]));
  }
}
EOF

  cat > a/a.bzl <<'EOF'
def _a_impl(ctx):
    symlink = ctx.actions.declare_symlink(ctx.label.name + ".link")
    output = ctx.actions.declare_file(ctx.label.name + ".file")
    ctx.actions.run(
        outputs = [symlink],
        executable = ctx.executable._make_symlink,
        arguments = [symlink.path, ctx.attr.link_target],
        inputs = depset([]),
    )
    ctx.actions.run_shell(
        outputs = [output],
        inputs = depset([symlink]),
        command = "echo input link is $(readlink " + symlink.path + ") > " + output.path,
    )
    return DefaultInfo(files = depset([output]))

a = rule(
    implementation = _a_impl, 
    attrs = {
        "link_target": attr.string(),
        "_make_symlink": attr.label(
            default = ":MakeSymlink",
            executable = True,
            cfg = "exec",
        )
    }
)
EOF

  cat > a/BUILD <<'EOF'
load(":a.bzl", "a")

java_binary(
    name = "MakeSymlink",
    srcs = ["MakeSymlink.java"],
    main_class = "MakeSymlink",
)
a(name="a", link_target="../somewhere/../over/the/rainbow/..")
EOF

  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:a || fail "build failed"
  assert_contains "input link is ../somewhere/../over/the/rainbow/.." bazel-bin/a/a.file
}

function test_dangling_symlink_created_from_symlink_action() {
  mkdir -p a
  cat > a/a.bzl <<'EOF'
def _a_impl(ctx):
    symlink = ctx.actions.declare_symlink(ctx.label.name + ".link")
    output = ctx.actions.declare_file(ctx.label.name + ".file")
    ctx.actions.symlink(
        output = symlink,
        target_path = ctx.attr.link_target,
    )
    ctx.actions.run_shell(
        outputs = [output],
        inputs = depset([symlink]),
        command = "echo input link is $(readlink " + symlink.path + ") > " + output.path,
    )
    return DefaultInfo(files = depset([output]))

a = rule(implementation = _a_impl, attrs = {"link_target": attr.string()})
EOF

  cat > a/BUILD <<'EOF'
load(":a.bzl", "a")

a(name="a", link_target="../somewhere/../in/my/heart/..")
EOF

  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:a || fail "build failed"
  assert_contains "input link is ../somewhere/../in/my/heart/.." bazel-bin/a/a.file
}

function test_symlink_created_from_symlink_action() {
  mkdir -p a
  cat > a/a.bzl <<'EOF'
def _a_impl(ctx):
    target = ctx.actions.declare_file(ctx.label.name + ".target")
    ctx.actions.write(
        output = target,
        content = "Hello, World!",
    )

    symlink = ctx.actions.declare_file(ctx.label.name + ".link")
    ctx.actions.symlink(
        output = symlink,
        target_file = target,
    )
    return DefaultInfo(files = depset([symlink]))

a = rule(implementation = _a_impl)
EOF

  cat > a/BUILD <<'EOF'
load(":a.bzl", "a")

a(name="a")
EOF

  bazel --windows_enable_symlinks build //a:a || fail "build failed"
  assert_contains "Hello, World!" bazel-bin/a/a.link
  expect_symlink bazel-bin/a/a.link
}

function test_executable_dangling_symlink() {
  mkdir -p a
  cat > a/a.bzl <<'EOF'
def _a_impl(ctx):
    symlink = ctx.actions.declare_symlink(ctx.label.name + ".link")
    ctx.actions.symlink(
        output = symlink,
        target_path = "/foo/bar",
        is_executable = True,
    )
    return DefaultInfo(files = depset([symlink]))

a = rule(implementation = _a_impl)
EOF

  cat > a/BUILD <<'EOF'
load(":a.bzl", "a")

a(name="a")
EOF

  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:a >& $TEST_log && fail "build succeeded"
  expect_log "\"is_executable\" cannot be True when using \"target_path\""
}

function test_executable_symlink() {
  mkdir -p a
  cat > a/a.bzl <<'EOF'
def _a_impl(ctx):
    target = ctx.actions.declare_file(ctx.label.name + ".target")
    ctx.actions.write(
        output = target,
        content = "Hello, World!",
        is_executable = True,
    )

    symlink = ctx.actions.declare_file(ctx.label.name + ".link")
    ctx.actions.symlink(
        output = symlink,
        target_file = target,
        is_executable = True,
    )
    return DefaultInfo(files = depset([symlink]))

a = rule(implementation = _a_impl)
EOF

  cat > a/BUILD <<'EOF'
load(":a.bzl", "a")

a(name="a")
EOF

  bazel --windows_enable_symlinks build //a:a || fail "build failed"
  assert_contains "Hello, World!" bazel-bin/a/a.link
  expect_symlink bazel-bin/a/a.link
}

function test_executable_symlink_to_nonexecutable_file() {
  if "$is_windows"; then
    warn "Skipping test on Windows: Bazel's FileSystem uses java.io.File#canExecute(), which \
          doesn't test for executability, it tests whether the current program is permitted \
          to execute it"
    return 0
  fi

  mkdir -p a
  cat > a/a.bzl <<'EOF'
def _a_impl(ctx):
    symlink = ctx.actions.declare_file(ctx.label.name + ".link")
    ctx.actions.symlink(
        output = symlink,
        target_file = ctx.file.file,
        is_executable = True,
    )
    return DefaultInfo(files = depset([symlink]))

a = rule(
    implementation = _a_impl,
    attrs = {
        "file": attr.label(allow_single_file = True)
    },
)
EOF

  cat > a/BUILD <<'EOF'
load(":a.bzl", "a")

a(
    name = "a",
    file = "foo.txt",
)
EOF

  cat > a/foo.txt <<'EOF'
Hello, World!
EOF

  bazel --windows_enable_symlinks build //a:a >& $TEST_log && fail "build succeeded"
  expect_log "failed to create symbolic link 'bazel-out/[^/]*/bin/a/a.link': file 'a/foo.txt' is not executable"
}

function test_symlink_cycle() {
  mkdir -p a
  cat > a/a.bzl <<'EOF'
def _a_impl(ctx):
    symlink1 = ctx.actions.declare_file(ctx.label.name + "_1")
    symlink2 = ctx.actions.declare_file(ctx.label.name + "_2")
    ctx.actions.symlink(
        output = symlink1,
        target_file = symlink2,
    )
    ctx.actions.symlink(
        output = symlink2,
        target_file = symlink1,
    )
    return DefaultInfo(files = depset([symlink1, symlink2]))

a = rule(
    implementation = _a_impl,
)
EOF

  cat > a/BUILD <<'EOF'
load(":a.bzl", "a")

a(
    name = "a",
)
EOF

  bazel --windows_enable_symlinks build //a:a >& $TEST_log && fail "build succeeded"
  expect_log "cycle in dependency graph"
}

function test_unresolved_symlink_in_exec_cfg() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent")
genrule(
    name = "exec",
    srcs = [],
    outs = ["out"],
    cmd = "touch $@",
    exec_tools = [":a"],
)
EOF

  bazel --windows_enable_symlinks build --experimental_allow_unresolved_symlinks //a:exec || fail "build failed"
}

run_suite "Tests for symlink artifacts"
