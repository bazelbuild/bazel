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

add_to_bazelrc "startup --windows_enable_symlinks"

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

  # We use python rather than a simple ln since the latter doesn't handle dangling symlinks on
  # Windows.
  mkdir -p symlink_helper
  cat > symlink_helper/BUILD <<EOF
py_binary(
    name = "symlink_helper",
    srcs = ["symlink_helper.py"],
    visibility = ["//visibility:public"],
)
EOF

  cat > symlink_helper/symlink_helper.py <<EOF
import os
import sys
os.symlink(*sys.argv[1:])
EOF
}

function test_smoke() {
  mkdir -p a
  cat > a/BUILD <<EOF
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent")
EOF

  bazel build //a:a || fail "build failed"
  ls -l bazel-bin/a
}

function test_no_unresolved_symlinks() {
  mkdir -p a
  cat > a/BUILD <<EOF
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent")
EOF

  bazel build --noallow_unresolved_symlinks //a:a >& $TEST_log \
    && fail "build succeeded, but should have failed"
  expect_log 'declare_symlink() is not allowed; use the --allow_unresolved_symlinks'
}

function test_inmemory_cache_symlinks() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF
  bazel build //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
  bazel build //a:g >& $TEST_log || fail "build failed"
  expect_not_log "running genrule"
}

function test_on_disk_cache_symlinks() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF
  bazel build //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
  bazel shutdown
  bazel build //a:g >& $TEST_log || fail "build failed"
  expect_not_log "running genrule"
}

function test_no_inmemory_cache_symlinks() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel build //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"

  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent2")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel build //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
}

function test_no_on_disk_cache_symlinks() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel build //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"

  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent2")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel shutdown
  bazel build //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
}

function test_replace_symlink_with_file() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel build //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"

  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "write")
write(name="a", contents="non/existent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel build //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
}

function test_replace_file_with_symlink() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "write")
write(name="a", contents="non/existent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel build //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"

  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "dangling_symlink")
dangling_symlink(name="a", link_target="non/existent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel build //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
}

function test_file_instead_of_symlink() {
  mkdir -p a
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

bad_symlink(name="bs", link_target="bad/symlink")
genrule(name="bsg", srcs=[":bs"], outs=["bsgo"], cmd="echo BSGO > $@")

bad_write(name="bw", contents="badcontents")
genrule(name="bwg", srcs=[":bw"], outs=["bwgo"], cmd="echo BWGO > $@")

genrule(
    name="bg",
    srcs=[],
    outs=["bgo"],
    cmd = "$(location //symlink_helper) bad/symlink $@",
    tools = ["//symlink_helper"],
)
EOF

  bazel build //a:bsg >& $TEST_log && fail "build succeeded"
  expect_log "declared output 'a/bs' is not a symlink"

  bazel build //a:bwg >& $TEST_log && fail "build succeeded"
  expect_log "symlink() with \"target_path\" param requires that \"output\" be declared as a symlink, not a file or directory"

  bazel build //a:bg >& $TEST_log && fail "build succeeded"
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

  bazel build //a:bs >& $TEST_log && fail "build succeeded"
  expect_log "symlink() with \"target_file\" param requires that \"output\" be declared as a file or directory, not a symlink"
}

function test_symlink_created_from_spawn() {
  mkdir -p a
  cat > a/a.bzl <<'EOF'
def _a_impl(ctx):
    symlink = ctx.actions.declare_symlink(ctx.label.name + ".link")
    output = ctx.actions.declare_file(ctx.label.name + ".file")
    ctx.actions.run(
        outputs = [symlink],
        executable = ctx.executable._link,
        arguments = [ctx.attr.link_target, symlink.path],
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
        "_link": attr.label(
            default = "//symlink_helper",
            executable = True,
            cfg = "exec",
        ),
    }
)
EOF

  cat > a/BUILD <<'EOF'
load(":a.bzl", "a")

a(name="a", link_target="somewhere/over/the/rainbow")
EOF

  bazel build //a:a || fail "build failed"
  assert_contains "input link is somewhere/over/the/rainbow" bazel-bin/a/a.file
}

function test_dangling_symlink_created_from_symlink_action() {
  if "$is_windows"; then
    warn "Skipping test on Windows: Bazel's FileSystem cannot yet create relative symlinks."
    return 0
  fi

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

a(name="a", link_target="../somewhere/in/my/heart")
EOF

  bazel build //a:a || fail "build failed"
  assert_contains "input link is ../somewhere/in/my/heart" bazel-bin/a/a.file
}

function test_symlink_file_to_file_created_from_symlink_action() {
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

  bazel build //a:a || fail "build failed"
  assert_contains "Hello, World!" bazel-bin/a/a.link
  expect_symlink bazel-bin/a/a.link
}

function test_symlink_directory_to_directory_created_from_symlink_action() {
  mkdir -p a
  cat > a/a.bzl <<'EOF'
def _a_impl(ctx):
    target = ctx.actions.declare_directory(ctx.label.name + ".target")
    ctx.actions.run_shell(
        outputs = [target],
        command = "echo 'Hello, World!' > $1/inside.txt",
        arguments = [target.path],
    )

    symlink = ctx.actions.declare_directory(ctx.label.name + ".link")
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

  bazel build //a:a || fail "build failed"
  assert_contains "Hello, World!" bazel-bin/a/a.link/inside.txt
  expect_symlink bazel-bin/a/a.link
}

function test_symlink_file_to_directory_created_from_symlink_action() {
  mkdir -p a
  cat > a/a.bzl <<'EOF'
def _a_impl(ctx):
    target = ctx.actions.declare_directory(ctx.label.name + ".target")
    ctx.actions.run_shell(
        outputs = [target],
        command = "echo 'Hello, World!' > $1/inside.txt",
        arguments = [target.path],
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

  bazel build //a:a >& $TEST_log && fail "build succeeded"
  expect_log "symlink() with \"target_file\" directory param requires that \"output\" be declared as a directory"
}

function test_symlink_directory_to_file_created_from_symlink_action() {
  mkdir -p a
  cat > a/a.bzl <<'EOF'
def _a_impl(ctx):
    target = ctx.actions.declare_file(ctx.label.name + ".target")
    ctx.actions.run_shell(
        outputs = [target],
        command = "true",
    )

    symlink = ctx.actions.declare_directory(ctx.label.name + ".link")
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

  bazel build //a:a >& $TEST_log && fail "build succeeded"
  expect_log "symlink() with \"target_file\" file param requires that \"output\" be declared as a file"
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

  bazel build //a:a >& $TEST_log && fail "build succeeded"
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

  bazel build //a:a || fail "build failed"
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

  bazel build //a:a >& $TEST_log && fail "build succeeded"
  expect_log "failed to create symbolic link 'bazel-out/[^/]*/bin/a/a.link': file 'a/foo.txt' is not executable"
}

function test_executable_symlink_to_directory() {
  mkdir -p a
  cat > a/a.bzl <<'EOF'
def _a_impl(ctx):
    target = ctx.actions.declare_directory(ctx.label.name + ".target")
    ctx.actions.run_shell(
        outputs = [target],
        command = "true",
    )

    symlink = ctx.actions.declare_directory(ctx.label.name + ".link")
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

a(name = "a")
EOF

  cat > a/foo.txt <<'EOF'
Hello, World!
EOF

  bazel build //a:a >& $TEST_log && fail "build succeeded"
  expect_log "symlink() with \"output\" directory param cannot be executable"
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

  bazel build //a:a >& $TEST_log && fail "build succeeded"
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
    tools = [":a"],
)
EOF

  bazel build //a:exec || fail "build failed"
}

function setup_unresolved_symlink_as_input() {
  mkdir -p pkg
  cat > pkg/def.bzl <<'EOF'
def _r(ctx):
  symlink = ctx.actions.declare_symlink(ctx.label.name + "_s")
  ctx.actions.symlink(output = symlink, target_path = ctx.file.file.basename)

  output = ctx.actions.declare_file(ctx.label.name)
  ctx.actions.run_shell(
    command = """[ -s {symlink} ] && [ $(readlink {symlink}) == "{target}" ] && touch {output}""".format(
      symlink = symlink.path,
      target = ctx.file.file.basename,
      output = output.path,
    ),
    inputs = [symlink] + ([ctx.file.file] if ctx.attr.stage_target else []),
    outputs = [output],
  )
  return DefaultInfo(files=depset([output]))

r = rule(
    implementation = _r,
    attrs = {
        "file": attr.label(allow_single_file=True),
        "stage_target": attr.bool(),
    }
)
EOF
  cat > pkg/BUILD <<'EOF'
load(":def.bzl", "r")

genrule(name="a", outs=["file"], cmd="echo hello >$@")
r(name="b", file="file", stage_target=False)
r(name="c", file="file", stage_target=True)
EOF

}

function test_unresolved_symlink_as_input_local() {
  if "$is_windows"; then
    # TODO(#10298): Support unresolved symlinks on Windows.
    return 0
  fi

  setup_unresolved_symlink_as_input
  add_to_bazelrc build --spawn_strategy=local

  bazel build //pkg:b && fail "symlink should not resolve"

  bazel clean
  bazel build //pkg:file
  # Since the build isn't sandboxed, the symlink to //:a resolves even though
  # the action does not declare it as an input.
  bazel build //pkg:b || fail "symlink expected to resolve non-hermetically"

  bazel clean
  bazel build //pkg:c || fail "symlink should resolve"
}

function test_unresolved_symlink_as_input_sandbox() {
  if "$is_windows"; then
    # TODO(#10298): Support unresolved symlinks on Windows.
    return 0
  fi

  setup_unresolved_symlink_as_input
  add_to_bazelrc build --spawn_strategy=sandboxed

  bazel build //pkg:b && fail "sandboxed build is not hermetic"

  bazel clean
  bazel build //pkg:a
  # Since the build isn't sandboxed, the symlink to //:a does not resolves even
  # though it would in the unsandboxed exec root due to //:a having been built
  # before.
  bazel build //pkg:b && fail "sandboxed build is not hermetic"

  bazel clean
  bazel build //pkg:c || fail "symlink should resolve"
}

function setup_unresolved_symlink_as_runfile() {
  mkdir -p pkg
  cat > pkg/script.sh.tpl <<'EOF'
#!/usr/bin/env bash
cd $0.runfiles/__WORKSPACE_NAME__
[ -L __SYMLINK__ ] || { echo "runfile is not a symlink"; exit 1; }
[ $(readlink __SYMLINK__) == "__TARGET__" ] || { echo "runfile symlink does not have the expected target, got: $(readlink __SYMLINK__)"; exit 1; }
[ -s __SYMLINK__ ] || { echo "runfile not resolved"; exit 1; }
EOF
  cat > pkg/def.bzl <<'EOF'
def _r(ctx):
  symlink = ctx.actions.declare_symlink(ctx.label.name + "_s")
  target = ctx.file.file.basename
  ctx.actions.symlink(output=symlink, target_path=target)

  script = ctx.actions.declare_file(ctx.label.name)
  ctx.actions.expand_template(
      template = ctx.file._script_tpl,
      output = script,
      is_executable = True,
      substitutions = {
          "__SYMLINK__": symlink.short_path,
          "__TARGET__": target,
          "__WORKSPACE_NAME__": ctx.workspace_name,
      },
  )

  runfiles = ctx.runfiles(
      files = [symlink] + ([ctx.file.file] if ctx.attr.stage_runfile else []),
  )
  return DefaultInfo(executable=script, runfiles=runfiles)

r = rule(
    implementation = _r,
    attrs = {
        "file": attr.label(allow_single_file = True),
        "stage_runfile": attr.bool(),
        "_script_tpl": attr.label(default = "script.sh.tpl", allow_single_file = True),
    },
    executable = True,
)
EOF
  cat > pkg/BUILD <<'EOF'
load(":def.bzl", "r")

genrule(name="a", outs=["file"], cmd="echo hello >$@")
r(name="tool", file="file", stage_runfile=True)
r(name="non_hermetic_tool", file="file", stage_runfile=False)
genrule(
    name = "use_tool",
    outs = ["out"],
    cmd = "$(location :tool) && touch $@",
    tools = [":tool"],
)
genrule(
    name = "use_tool_non_hermetically",
    outs = ["out_non_hermetic"],
    cmd = "$(location :non_hermetic_tool) && touch $@",
    # Stage file outside the runfiles tree.
    tools = [":non_hermetic_tool", "file"],
)
EOF
}

function test_unresolved_symlink_as_runfile_local() {
  if "$is_windows"; then
    # TODO(#10298): Support unresolved symlinks on Windows.
    return 0
  fi

  setup_unresolved_symlink_as_runfile
  add_to_bazelrc build --spawn_strategy=local

  bazel build //pkg:use_tool || fail "local build failed"
  # Keep the implicitly built //pkg:a around to make the symlink resolve
  # outside the runfiles tree. The build should still fail as the relative
  # symlink is staged as is and doesn't resolve outside the runfiles tree.
  bazel build //pkg:use_tool_non_hermetically && fail "symlink in runfiles resolved outside the runfiles tree" || true
}

function test_unresolved_symlink_as_runfile_symlink() {
  if "$is_windows"; then
    # TODO(#10298): Support unresolved symlinks on Windows.
    return 0
  fi

  setup_unresolved_symlink_as_runfile
  add_to_bazelrc build --spawn_strategy=sandboxed

  bazel build //pkg:use_tool || fail "sandboxed build failed"
  # Keep the implicitly built //pkg:a around to make the symlink resolve
  # outside the runfiles tree. The build should still fail as the relative
  # symlink is staged as is and doesn't resolve outside the runfiles tree.
  bazel build //pkg:use_tool_non_hermetically && fail "symlink in runfiles resolved outside the runfiles tree" || true
}

run_suite "Tests for symlink artifacts"
