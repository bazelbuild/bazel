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

# --- begin runfiles.bash initialization ---
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
  declare -r EXE_EXT=".exe"
else
  declare -r EXE_EXT=""
fi

#### TESTS #############################################################

# Tests in this file cannot invoke a real Python 3 runtime. This is because this
# file is shared by both Bazel's public test suite and Google's internal tests,
# and the internal tests do not have a Python 3 environment.
#
#   - If you only need a real Python 2 environment and do not use Python 3 at
#     all, you can place your test in this file.
#
#   - If you need to check Bazel's behavior concerning the *selection* of a
#     Python 2 or 3 runtime, but do not actually need the runtime itself, then
#     you may put your test in this file and call `use_fake_python_runtimes`
#     before your test logic.
#
#   - Otherwise, put your test in //src/test/shell/bazel. That suite can invoke
#     actual Python 2 and 3 interpreters.

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
from __future__ import print_function
import os.path
import sys

print("This is my name: %s" % __file__)
print("This is my working directory: %s" % os.getcwd())
os.chdir(os.path.dirname(__file__))
print("This is my new working directory: %s" % os.getcwd())

file_to_check = "mypackage/__init__.py"

if not os.path.exists(file_to_check):
  print("mypackage/__init__.py does not exist")
  sys.exit(1)

if os.path.islink(file_to_check):
  print("mypackage/__init__.py is a symlink, expected a regular file")
  sys.exit(1)

if not os.path.isfile(file_to_check):
  print("mypackage/__init__.py is not a regular file")
  sys.exit(1)

print("OK")
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
  [ -d "bazel-bin/py-tool${EXE_EXT}.runfiles" ] || fail "py_binary runfiles tree not built"
  bazel clean
  bazel build --noexperimental_build_transitive_python_runfiles :sh-tool
  [ ! -e "bazel-bin/py-tool${EXE_EXT}.runfiles" ] || fail "py_binary runfiles tree built"
}

# Test that Python 2 or Python 3 is actually invoked, with and without flag
# overrides.
function test_python_version() {
  use_fake_python_runtimes

  mkdir -p test
  touch test/main2.py test/main3.py
  cat > test/BUILD << EOF
py_binary(name = "main2",
    default_python_version = "PY2",
    srcs = ['main2.py'],
)
py_binary(name = "main3",
    default_python_version = "PY3",
    srcs = ["main3.py"],
)
EOF

  # No flag, use the default from the rule.
  bazel run //test:main2 \
      &> $TEST_log || fail "bazel run failed"
  expect_log "I am Python 2"
  bazel run //test:main3 \
      &> $TEST_log || fail "bazel run failed"
  expect_log "I am Python 3"

  # Force to Python 2.
  bazel run //test:main2 --force_python=PY2 \
      &> $TEST_log || fail "bazel run failed"
  expect_log "I am Python 2"
  bazel run //test:main3 --force_python=PY2 \
      &> $TEST_log || fail "bazel run failed"
  expect_log "I am Python 2"

  # Force to Python 3.
  bazel run //test:main2 --force_python=PY3 \
      &> $TEST_log || fail "bazel run failed"
  expect_log "I am Python 3"
  bazel run //test:main3 --force_python=PY3 \
      &> $TEST_log || fail "bazel run failed"
  expect_log "I am Python 3"
}

run_suite "Tests for the Python rules"
