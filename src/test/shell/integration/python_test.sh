#!/bin/bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

set -eu

function test_python_binary_empty_files_in_runfiles_are_regular_files() {
  mkdir -p test/mypackage
  cat > test/BUILD <<'EOF'
py_test(
    name = "a",
    srcs = [
        "a.py",
        "mypackage/b.py",
    ],
    main = "a.py"
)
EOF
  cat >test/a.py <<'EOF'
import os.path
import sys

print "This is my name: %s" % __file__
print "This is my working directory: %s" % os.getcwd()
os.chdir(os.path.dirname(__file__))
print "This is my new working directory: %s" % os.getcwd()

file_to_check = "mypackage/__init__.py"

if not os.path.exists(file_to_check):
  print "mypackage/__init__.py does not exist"
  sys.exit(1)

if os.path.islink(file_to_check):
  print "mypackage/__init__.py is a symlink, expected a regular file"
  sys.exit(1)

if not os.path.isfile(file_to_check):
  print "mypackage/__init__.py is not a regular file"
  sys.exit(1)

print "OK"
EOF
  touch test/mypackage/b.py

  bazel test --test_output=streamed //test:a &> $TEST_log || fail "test failed"
}

function test_building_transitive_py_binary_runfiles_trees() {
    touch main.py script.sh
    chmod u+x script.sh
    cat > BUILD <<'EOF'
py_binary(
    name = 'py-tool',
    srcs = ['main.py'],
    main = 'main.py',
)

sh_binary(
    name = 'sh-tool',
    srcs = ['script.sh'],
    data = [':py-tool'],
)
EOF
    bazel build --experimental_build_transitive_python_runfiles :sh-tool
    [ -d 'bazel-bin/py-tool.runfiles' ] || fail "py_binary runfiles tree not built"
    bazel clean
    bazel build --noexperimental_build_transitive_python_runfiles :sh-tool
    [ ! -e 'bazel-bin/py-tool.runfiles' ] || fail "py_binary runfiles tree built"
}

run_suite "Tests for the Python rules"
