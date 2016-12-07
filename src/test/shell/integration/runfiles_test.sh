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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

#### SETUP #############################################################

set -e

function set_up() {
  mkdir -p pkg
  cd pkg

  mkdir -p a/b c/d e/f/g x/y
  touch py.py a/b/no_module.py c/d/one_module.py c/__init__.py e/f/g/ignored.py x/y/z.sh
  chmod +x x/y/z.sh

  cd ..
  touch __init__.py
}

#### TESTS #############################################################

function test_hidden() {
  cat > pkg/BUILD << EOF
py_binary(name = "py",
          srcs = [ "py.py" ],
          data = [ "e/f",
                   "e/f/g/hidden.py" ])
genrule(name = "hidden",
        outs = [ "e/f/g/hidden.py" ],
        cmd = "touch \$@")
EOF
  bazel build pkg:py >&$TEST_log 2>&1 || fail "build failed"

  # we get a warning that hidden.py is inaccessible
  expect_log_once "/genfiles/pkg/e/f/g/hidden.py obscured by pkg/e/f "
}

function test_foo_runfiles() {
cat > BUILD << EOF
py_library(name = "root",
           srcs = ["__init__.py"],
           visibility = ["//visibility:public"])
EOF
cat > pkg/BUILD << EOF
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
  bazel build pkg:foo >&$TEST_log || fail "build failed"

  cd ${PRODUCT_NAME}-bin/pkg/foo.runfiles

  # workaround until we use assert/fail macros in the tests below
  touch $TEST_TMPDIR/__fail

  # output manifest exists and is non-empty
  test    -f MANIFEST
  test    -s MANIFEST

  cd ${WORKSPACE_NAME}

  # these are real directories
  test \! -L pkg
  test    -d pkg

  cd pkg
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
  assert_equals  9 $(find ${WORKSPACE_NAME} -type l | wc -l)
  assert_equals  4 $(find ${WORKSPACE_NAME} -type f | wc -l)
  assert_equals  9 $(find ${WORKSPACE_NAME} -type d | wc -l)
  assert_equals 22 $(find ${WORKSPACE_NAME} | wc -l)
  assert_equals 13 $(wc -l < MANIFEST)

  for i in $(find ${WORKSPACE_NAME} \! -type d); do
    echo "$i $(readlink "$i")"
  done >MANIFEST2
  diff -u <(sort MANIFEST) <(sort MANIFEST2)
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
  bazel build //:thing &> $TEST_log || fail "Build failed"
  [[ -d ${PRODUCT_NAME}-bin/thing.runfiles/foo ]] || fail "foo not found"

  cat > WORKSPACE <<EOF
workspace(name = "bar")
EOF
  bazel build //:thing &> $TEST_log || fail "Build failed"
  [[ -d ${PRODUCT_NAME}-bin/thing.runfiles/bar ]] || fail "bar not found"
  [[ ! -d ${PRODUCT_NAME}-bin/thing.runfiles/foo ]] \
    || fail "Old foo still found"
}


run_suite "runfiles"
