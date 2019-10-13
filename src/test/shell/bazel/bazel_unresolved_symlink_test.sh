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
# Tests for unresolved symlinks in Bazel.

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

function set_up() {
  mkdir -p symlink
  touch symlink/BUILD
  cat > symlink/symlink.bzl <<EOF
def _symlink_impl(ctx):
    symlink = ctx.actions.declare_symlink(ctx.label.name)
    ctx.actions.symlink(symlink, ctx.attr.link_target)
    return DefaultInfo(files = depset([symlink]))

symlink = rule(implementation = _symlink_impl, attrs = {"link_target": attr.string()})

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
load("//symlink:symlink.bzl", "symlink")
symlink(name="a", link_target="/nonexistent")
EOF

  bazel build --experimental_allow_unresolved_symlinks //a:a || fail "build failed"
  ls -l bazel-bin/a
}

function test_inmemory_cache_symlinks() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "symlink")
symlink(name="a", link_target="/nonexistent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF
  bazel build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
  bazel build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_not_log "running genrule"
}

function test_on_disk_cache_symlinks() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "symlink")
symlink(name="a", link_target="/nonexistent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF
  bazel build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
  bazel shutdown
  bazel build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_not_log "running genrule"
}

function test_no_inmemory_cache_symlinks() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "symlink")
symlink(name="a", link_target="/nonexistent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"

  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "symlink")
symlink(name="a", link_target="/nonexistent2")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
}

function test_no_on_disk_cache_symlinks() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "symlink")
symlink(name="a", link_target="/nonexistent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"

  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "symlink")
symlink(name="a", link_target="/nonexistent2")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel shutdown
  bazel build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
}

function test_replace_symlink_with_file() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "symlink")
symlink(name="a", link_target="/nonexistent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"

  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "write")
write(name="a", contents="/nonexistent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
}

function test_replace_file_with_symlink() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "write")
write(name="a", contents="/nonexistent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"

  cat > a/BUILD <<'EOF'
load("//symlink:symlink.bzl", "symlink")
symlink(name="a", link_target="/nonexistent")
genrule(name="g", srcs=[":a"], outs=["go"], cmd="echo running genrule; echo GO > $@")
EOF

  bazel build --experimental_allow_unresolved_symlinks //a:g >& $TEST_log || fail "build failed"
  expect_log "running genrule"
}

function test_file_instead_of_symlink() {
  mkdir -p a
  cat > a/a.bzl <<'EOF'
def _bad_symlink_impl(ctx):
    symlink = ctx.actions.declare_symlink(ctx.label.name)
    ctx.actions.write(symlink, ctx.attr.link_target)  # Oops, should be "symlink"
    return DefaultInfo(files = depset([symlink]))

bad_symlink = rule(implementation = _bad_symlink_impl, attrs = {"link_target": attr.string()})

def _bad_write_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.symlink(output, ctx.attr.contents)  # Oops, should be "write"
    return DefaultInfo(files = depset([output]))

bad_write = rule(implementation = _bad_write_impl, attrs = {"contents": attr.string()})
EOF

  cat > a/BUILD <<'EOF'
load(":a.bzl", "bad_symlink", "bad_write")

bad_symlink(name="bs", link_target="/badsymlink")
genrule(name="bsg", srcs=[":bs"], outs=["bsgo"], cmd="echo BSGO > $@")

bad_write(name="bw", contents="badcontents")
genrule(name="bwg", srcs=[":bw"], outs=["bwgo"], cmd="echo BWGO > $@")

genrule(name="bg", srcs=[], outs=["bgo"], cmd = "ln -s /badsymlink $@")
EOF

  bazel build --experimental_allow_unresolved_symlinks //a:bsg && fail "build succeeded"
  [[ "$?" == 1 ]] || fail "unexpected exit code"

  bazel build --experimental_allow_unresolved_symlinks //a:bwg && fail "build succeeded"
  [[ "$?" == 1 ]] || fail "unexpected exit code"

  bazel build --experimental_allow_unresolved_symlinks //a:bg && fail "build succeeded"
  [[ "$?" == 1 ]] || fail "unexpected exit code"
}

function test_symlink_created_from_spawn() {
  mkdir -p a
  cat > a/a.bzl <<'EOF'
def _a_impl(ctx):
    symlink = ctx.actions.declare_symlink(ctx.label.name + ".link")
    output = ctx.actions.declare_file(ctx.label.name + ".file")
    ctx.actions.run_shell(
        outputs = [symlink],
        inputs = depset([]),
        command = "ln -s " + ctx.attr.link_target + " " + symlink.path,
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

a(name="a", link_target="/somewhere/over/the/rainbow")
EOF

  bazel build --experimental_allow_unresolved_symlinks //a:a || fail "build failed"
  assert_contains "input link is /somewhere/over/the/rainbow" bazel-bin/a/a.file
}

function test_symlink_created_from_symlink_action() {
  mkdir -p a
  cat > a/a.bzl <<'EOF'
def _a_impl(ctx):
    symlink = ctx.actions.declare_symlink(ctx.label.name + ".link")
    output = ctx.actions.declare_file(ctx.label.name + ".file")
    ctx.actions.symlink(symlink, ctx.attr.link_target)
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

a(name="a", link_target="/somewhere/in/my/heart")
EOF

  bazel build --experimental_allow_unresolved_symlinks //a:a || fail "build failed"
  assert_contains "input link is /somewhere/in/my/heart" bazel-bin/a/a.file
}

run_suite "Tests for unresolved symlink artifacts"
