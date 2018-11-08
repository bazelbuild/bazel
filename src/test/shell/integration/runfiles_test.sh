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

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
  export EXT=".exe"
  export EXTRA_BUILD_FLAGS="--enable_runfiles --build_python_zip=0"
else
  export EXT=""
  export EXTRA_BUILD_FLAGS=""
fi

#### SETUP #############################################################

set -e

function create_pkg() {
  local -r pkg=$1
  mkdir -p $pkg
  cd $pkg

  mkdir -p a/b c/d e/f/g x/y
  touch py.py a/b/no_module.py c/d/one_module.py c/__init__.py e/f/g/ignored.py x/y/z.sh
  chmod +x x/y/z.sh

  cd ..
  touch __init__.py
}

#### TESTS #############################################################

function test_hidden() {
  local -r pkg=$FUNCNAME
  create_pkg $pkg
  cat > $pkg/BUILD << EOF
py_binary(name = "py",
          srcs = [ "py.py" ],
          data = [ "e/f",
                   "e/f/g/hidden.py" ])
genrule(name = "hidden",
        outs = [ "e/f/g/hidden.py" ],
        cmd = "touch \$@")
EOF
  bazel build $pkg:py $EXTRA_BUILD_FLAGS >&$TEST_log 2>&1 || fail "build failed"

  # we get a warning that hidden.py is inaccessible
  expect_log_once "${pkg}/e/f/g/hidden.py obscured by ${pkg}/e/f "
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
                   "e/f/g/ignored.py" ],
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

  # these are real directories
  test \! -L $pkg
  test    -d $pkg

  cd $pkg
  test \! -L a
  test    -d a
  test \! -L a/b
  test    -d a/b
  test \! -L c
  test    -d c
  test \! -L c/d
  test    -d c/d
  test \! -L e
  test    -d e
  test \! -L x
  test    -d x
  test \! -L x/y
  test    -d x/y

  # these are symlinks to the source tree
  test    -L foo
  test    -L x/y/z.sh
  test    -L a/b/no_module.py
  test    -L c/d/one_module.py
  test    -L c/__init__.py
  test    -L e/f
  test    -d e/f
  # TODO(bazel-team): an __init__.py should appear here

  # these are real empty files
  test \! -L a/__init__.py
  test    -f a/__init__.py
  test \! -s a/__init__.py
  test \! -L a/b/__init__.py
  test    -f a/b/__init__.py
  test \! -s a/b/__init__.py
  test \! -L c/d/__init__.py
  test    -f c/d/__init__.py
  test \! -s c/d/__init__.py
  test \! -L __init__.py
  test    -f __init__.py
  test \! -s __init__.py

  # that accounts for everything
  cd ../..
  # For shell binary and python binary, we build both `bin` and `bin.exe`,
  # but on Linux we only build `bin`.
  # That's why we have two more symlinks on Windows.
  if "$is_windows"; then
    assert_equals 11 $(find ${WORKSPACE_NAME} -type l | wc -l)
    assert_equals  4 $(find ${WORKSPACE_NAME} -type f | wc -l)
    assert_equals  9 $(find ${WORKSPACE_NAME} -type d | wc -l)
    assert_equals 24 $(find ${WORKSPACE_NAME} | wc -l)
    assert_equals 15 $(wc -l < MANIFEST)
  else
    assert_equals  9 $(find ${WORKSPACE_NAME} -type l | wc -l)
    assert_equals  4 $(find ${WORKSPACE_NAME} -type f | wc -l)
    assert_equals  9 $(find ${WORKSPACE_NAME} -type d | wc -l)
    assert_equals 22 $(find ${WORKSPACE_NAME} | wc -l)
    assert_equals 13 $(wc -l < MANIFEST)
  fi

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
    assert_equals  4 $(wc -l < MANIFEST)
  else
    assert_equals  3 $(find ${WORKSPACE_NAME} -type l | wc -l)
    assert_equals  0 $(find ${WORKSPACE_NAME} -type f | wc -l)
    assert_equals  5 $(find ${WORKSPACE_NAME} -type d | wc -l)
    assert_equals  8 $(find ${WORKSPACE_NAME} | wc -l)
    assert_equals  3 $(wc -l < MANIFEST)
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
  sort MANIFEST > ${TEST_TMPDIR}/MANIFEST_sorted
  sort ${TEST_TMPDIR}/MANIFEST2 > ${TEST_TMPDIR}/MANIFEST2_sorted
  diff -u ${TEST_TMPDIR}/MANIFEST_sorted ${TEST_TMPDIR}/MANIFEST2_sorted
}

function test_workspace_name_change() {
  cat > WORKSPACE <<EOF
workspace(name = "foo")
EOF

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

  cat > WORKSPACE <<EOF
workspace(name = "bar")
EOF
  bazel build //:thing $EXTRA_BUILD_FLAGS &> $TEST_log || fail "Build failed"
  [[ -d ${PRODUCT_NAME}-bin/thing${EXT}.runfiles/bar ]] || fail "bar not found"
  [[ ! -d ${PRODUCT_NAME}-bin/thing${EXT}.runfiles/foo ]] \
    || fail "Old foo still found"
}


run_suite "runfiles"
