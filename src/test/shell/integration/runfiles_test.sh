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
# An end-to-end test that Bazel produces runfiles trees as expected.

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
set -euo pipefail
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
source "$(rlocation "io_bazel/src/test/shell/integration/runfiles_test_utils.sh")" \
  || { echo "runfiles_test_utils.sh not found!" >&2; exit 1; }

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

# We disable Python toolchains in EXTRA_BUILD_FLAGS because it throws off the
# counts and manifest checks in test_foo_runfiles.
# TODO(#8169): Update this test and remove the toolchain opt-out.
if "$is_windows"; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
  export EXT=".exe"
  export EXTRA_BUILD_FLAGS="--incompatible_use_python_toolchains=false \
--enable_runfiles --build_python_zip=0"
else
  export EXT=""
  export EXTRA_BUILD_FLAGS="--incompatible_use_python_toolchains=false"
fi

#### SETUP #############################################################

set -e

function create_pkg() {
  local -r pkg=$1
  mkdir -p $pkg
  cd $pkg

  mkdir -p a/b c/d e/f/g x/y
  touch py.py a/b/no_module.py c/d/one_module.py c/__init__.py e/f/g/ignored.txt x/y/z.sh
  chmod +x x/y/z.sh

  cd ..
  touch __init__.py
}

# This is basically a cross-platform version of `find -printf '%n %y %Y'`.
# i.e. recursively print paths, their raw file type, and for symbolic links,
# the type of file the link points to.
# Macs don't support `find -printf`, and stat, readlink etc all have different
# args and format specifiers. Basic bash works fine, though.
function recursive_path_info() {
  for path in $(find "$1" | sort); do
    if [[ -L "$path" ]]; then
      actual_type=symlink
    else
      actual_type=regular
    fi
    if [[ -f "$path" ]]; then
      effective_type=file
    elif [[ -d "$path" ]]; then
      effective_type="$actual_type dir"
    else
      # The various special file types shouldn't occur in practice, so just
      # call them unknown
      effective_type=unknown
    fi
    echo "$path $effective_type"
  done
}

#### TESTS #############################################################

function test_hidden() {
  local -r pkg=$FUNCNAME

  mkdir -p "$pkg/e/f/g"
  touch "$pkg/e/f/g/hidden.txt"
  cat > "$pkg/defs.bzl" << EOF
def _obscured_impl(ctx):
    executable = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(executable, "# nop")
    return [DefaultInfo(
        executable = executable,
        runfiles = ctx.runfiles(files = ctx.files.data),
    )]

obscured = rule(
    implementation = _obscured_impl,
    attrs = {"data": attr.label_list(allow_files = True)},
    # Must be executable to trigger the obscured runfile check
    executable = True,
)
EOF

  cat > $pkg/BUILD << EOF
load(":defs.bzl", "obscured")
obscured(name="bin", data=["e/f", "e/f/g/hidden.txt"])
genrule(name = "hidden",
        outs = [ "e/f/g/hidden.txt" ],
        cmd = "touch \$@")
EOF
  bazel build $pkg:bin $EXTRA_BUILD_FLAGS >&$TEST_log 2>&1 || fail "build failed"

  # we get a warning that hidden.txt is inaccessible
  expect_log_once "${pkg}/e/f/g/hidden.txt obscured by ${pkg}/e/f "
}

function test_foo_runfiles() {
  local -r pkg=$FUNCNAME
  create_pkg $pkg
cat > BUILD << EOF
py_library(name = "root",
           srcs = ["__init__.py"],
           visibility = ["//visibility:public"])
EOF
cat > $pkg/BUILD << EOF
sh_binary(name = "foo",
          srcs = [ "x/y/z.sh" ],
          data = [ ":py",
                   "e/f" ])
py_binary(name = "py",
          srcs = [ "py.py",
                   "a/b/no_module.py",
                   "c/d/one_module.py",
                   "c/__init__.py",
                 ],
          data = ["e/f/g/ignored.txt"],
          deps = ["//:root"])
EOF
  bazel build $pkg:foo $EXTRA_BUILD_FLAGS >&$TEST_log || fail "build failed"
  workspace_root=$PWD

  cd ${PRODUCT_NAME}-bin/$pkg/foo${EXT}.runfiles

  # workaround until we use assert/fail macros in the tests below
  touch $TEST_TMPDIR/__fail

  # output manifest exists and is non-empty
  test    -f MANIFEST
  test    -s MANIFEST

  cd ${WORKSPACE_NAME}


  cd $pkg

  # these are real empty files
  test \! -s a/__init__.py
  test \! -s a/b/__init__.py
  test \! -s c/d/__init__.py
  test \! -s __init__.py
  cd ..

  # These are basically tuples of (path filetype)
  expected="
. regular dir
./__init__.py file
./test_foo_runfiles regular dir
./test_foo_runfiles/__init__.py file
./test_foo_runfiles/a regular dir
./test_foo_runfiles/a/__init__.py file
./test_foo_runfiles/a/b regular dir
./test_foo_runfiles/a/b/__init__.py file
./test_foo_runfiles/a/b/no_module.py file
./test_foo_runfiles/c regular dir
./test_foo_runfiles/c/__init__.py file
./test_foo_runfiles/c/d regular dir
./test_foo_runfiles/c/d/__init__.py file
./test_foo_runfiles/c/d/one_module.py file
./test_foo_runfiles/e regular dir
./test_foo_runfiles/e/f symlink dir
./test_foo_runfiles/foo file
./test_foo_runfiles/py file
./test_foo_runfiles/py.py file
./test_foo_runfiles/x regular dir
./test_foo_runfiles/x/y regular dir
./test_foo_runfiles/x/y/z.sh file
"
  expected="$expected$(get_python_runtime_runfiles)"

  # For shell binary and python binary, we build both `bin` and `bin.exe`,
  # but on Linux we only build `bin`.
  if "$is_windows"; then
    expected="${expected}
./test_foo_runfiles/py.exe file
./test_foo_runfiles/foo.exe file
"
  fi

  # Sort and delete empty lines. This makes it easier to append to the
  # expected string and not have to worry about stray newlines from shell
  # commands and quoting.
  expected=$(sort <<<"$expected" | sed '/^$/d')
  actual=$(recursive_path_info .)
  assert_equals "$expected" "$actual"

  # The manifest only records files and symlinks, not real directories
  expected="$expected$(get_repo_mapping_manifest_file)"
  expected_manifest_size=$(echo "$expected" | grep -v ' regular dir' | wc -l)
  actual_manifest_size=$(wc -l < ../MANIFEST)
  assert_equals $expected_manifest_size $actual_manifest_size

  # that accounts for everything
  cd ..

  for i in $(find ${WORKSPACE_NAME} \! -type d); do
    target="$(readlink "$i" || true)"
    if [[ -z "$target" ]]; then
      echo "$i " >> ${TEST_TMPDIR}/MANIFEST2
    else
      if "$is_windows"; then
        echo "$i $(cygpath -m $target)" >> ${TEST_TMPDIR}/MANIFEST2
      else
        echo "$i $target" >> ${TEST_TMPDIR}/MANIFEST2
      fi
    fi
  done

  # Add the repo mapping manifest entry for Bazel.
  if [[ "$PRODUCT_NAME" == "bazel" ]]; then
    repo_mapping="_repo_mapping"
    repo_mapping_target="$(readlink "$repo_mapping")"
    if "$is_windows"; then
      repo_mapping_target="$(cygpath -m $repo_mapping_target)"
    fi
    echo "$repo_mapping $repo_mapping_target" >> ${TEST_TMPDIR}/MANIFEST2
  fi

  sort MANIFEST > ${TEST_TMPDIR}/MANIFEST_sorted
  sort ${TEST_TMPDIR}/MANIFEST2 > ${TEST_TMPDIR}/MANIFEST2_sorted
  diff -u ${TEST_TMPDIR}/MANIFEST_sorted ${TEST_TMPDIR}/MANIFEST2_sorted

  # Rebuild the same target with a new dependency.
  cd "$workspace_root"
cat > $pkg/BUILD << EOF
sh_binary(name = "foo",
          srcs = [ "x/y/z.sh" ],
          data = [ "e/f" ])
EOF
  bazel build $pkg:foo $EXTRA_BUILD_FLAGS >&$TEST_log || fail "build failed"

  cd ${PRODUCT_NAME}-bin/$pkg/foo${EXT}.runfiles

  # workaround until we use assert/fail macros in the tests below
  touch $TEST_TMPDIR/__fail

  # output manifest exists and is non-empty
  test    -f MANIFEST
  test    -s MANIFEST

  cd ${WORKSPACE_NAME}

  # these are real directories
  test \! -L $pkg
  test    -d $pkg

  # these directory should not exist anymore
  test \! -e a
  test \! -e c

  cd $pkg
  test \! -L e
  test    -d e
  test \! -L x
  test    -d x
  test \! -L x/y
  test    -d x/y

  # these are symlinks to the source tree
  test    -L foo
  test    -L x/y/z.sh
  test    -L e/f
  test    -d e/f

  # that accounts for everything
  cd ../..
  # For shell binary, we build both `bin` and `bin.exe`, but on Linux we only build `bin`
  # That's why we have one more symlink on Windows.
  if "$is_windows"; then
    assert_equals  4 $(find ${WORKSPACE_NAME} -type l | wc -l)
    assert_equals  0 $(find ${WORKSPACE_NAME} -type f | wc -l)
    assert_equals  5 $(find ${WORKSPACE_NAME} -type d | wc -l)
    assert_equals  9 $(find ${WORKSPACE_NAME} | wc -l)
    if [[ "$PRODUCT_NAME" == "bazel" ]]; then
      assert_equals  5 $(wc -l < MANIFEST)
    else
      assert_equals  4 $(wc -l < MANIFEST)
    fi
  else
    assert_equals  3 $(find ${WORKSPACE_NAME} -type l | wc -l)
    assert_equals  0 $(find ${WORKSPACE_NAME} -type f | wc -l)
    assert_equals  5 $(find ${WORKSPACE_NAME} -type d | wc -l)
    assert_equals  8 $(find ${WORKSPACE_NAME} | wc -l)
    if [[ "$PRODUCT_NAME" == "bazel" ]]; then
      assert_equals  4 $(wc -l < MANIFEST)
    else
      assert_equals  3 $(wc -l < MANIFEST)
    fi
  fi

  rm -f ${TEST_TMPDIR}/MANIFEST
  rm -f ${TEST_TMPDIR}/MANIFEST2
  for i in $(find ${WORKSPACE_NAME} \! -type d); do
    target="$(readlink "$i" || true)"
    if [[ -z "$target" ]]; then
      echo "$i " >> ${TEST_TMPDIR}/MANIFEST2
    else
      if "$is_windows"; then
        echo "$i $(cygpath -m $target)" >> ${TEST_TMPDIR}/MANIFEST2
      else
        echo "$i $target" >> ${TEST_TMPDIR}/MANIFEST2
      fi
    fi
  done

  # Add the repo mapping manifest entry for Bazel.
  if [[ "$PRODUCT_NAME" == "bazel" ]]; then
    repo_mapping="_repo_mapping"
    repo_mapping_target="$(readlink "$repo_mapping")"
    if "$is_windows"; then
      repo_mapping_target="$(cygpath -m $repo_mapping_target)"
    fi
    echo "$repo_mapping $repo_mapping_target" >> ${TEST_TMPDIR}/MANIFEST2
  fi

  sort MANIFEST > ${TEST_TMPDIR}/MANIFEST_sorted
  sort ${TEST_TMPDIR}/MANIFEST2 > ${TEST_TMPDIR}/MANIFEST2_sorted
  diff -u ${TEST_TMPDIR}/MANIFEST_sorted ${TEST_TMPDIR}/MANIFEST2_sorted
}

function test_workspace_name_change() {
  # TODO(b/174761497): Re-enable the test outside of Bazel.
  [[ "${PRODUCT_NAME}" != bazel ]] && return 0

  # Rewrite the workspace name but leave the rest of WORKSPACE alone.
  sed -ie 's,workspace(.*,workspace(name = "foo"),' WORKSPACE

  cat > BUILD <<EOF
cc_binary(
    name = "thing",
    srcs = ["thing.cc"],
    data = ["BUILD"],
)
EOF
  cat > thing.cc <<EOF
int main() { return 0; }
EOF
  bazel build //:thing $EXTRA_BUILD_FLAGS &> $TEST_log || fail "Build failed"
  [[ -d ${PRODUCT_NAME}-bin/thing${EXT}.runfiles/foo ]] || fail "foo not found"

  # Change workspace name to bar.
  sed -ie 's,workspace(.*,workspace(name = "bar"),' WORKSPACE
  bazel build //:thing $EXTRA_BUILD_FLAGS &> $TEST_log || fail "Build failed"
  [[ -d ${PRODUCT_NAME}-bin/thing${EXT}.runfiles/bar ]] || fail "bar not found"
  [[ ! -d ${PRODUCT_NAME}-bin/thing${EXT}.runfiles/foo ]] \
    || fail "Old foo still found"
}

# regression test for b/237547165
function test_fail_on_middleman_in_transitive_runfiles_for_executable() {
  cat > rule.bzl <<EOF
def _impl(ctx):
    exe = ctx.actions.declare_file(ctx.label.name + '.out')
    ctx.actions.write(exe, "")
    internal_outputs = ctx.attr.bin[OutputGroupInfo]._hidden_top_level_INTERNAL_
    runfiles = ctx.runfiles(transitive_files = internal_outputs)
    return DefaultInfo(runfiles = runfiles, executable = exe)
bad_runfiles = rule(
  implementation = _impl,
  attrs = {"bin" : attr.label()},
  executable = True,
)
EOF
  cat > BUILD <<EOF
load(":rule.bzl", "bad_runfiles");
cc_binary(
    name = "thing",
    srcs = ["thing.cc"],
)
bad_runfiles(name = "test", bin = ":thing")
EOF
  cat > thing.cc <<EOF
int main() { return 0; }
EOF
  bazel build //:test &> $TEST_log && fail "Expected build to fail but it succeeded"
  expect_log_once "Runfiles must not contain middleman artifacts"
}


run_suite "runfiles"
