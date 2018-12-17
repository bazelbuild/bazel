#!/bin/bash
#
# Copyright 2018 The Bazel Authors. All rights reserved.
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

# Test Python 2/3 version behavior. These tests require that the target platform
# has both Python versions available.

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

# `uname` returns the current platform, e.g "MSYS_NT-10.0" or "Linux".
# `tr` converts all upper case letters to lower case.
# `case` matches the result if the `uname | tr` expression to string prefixes
# that use the same wildcards as names do in Bash, i.e. "msys*" matches strings
# starting with "msys", and "*" matches everything (it's the default case).
case "$(uname -s | tr [:upper:] [:lower:])" in
msys*)
  # As of 2018-08-14, Bazel on Windows only supports MSYS Bash.
  declare -r is_windows=true
  # As of 2018-12-17, this test is disabled on windows (via "no_windows" tag),
  # so this code shouldn't even have run. See the TODO at
  # use_system_python_2_3_runtimes.
  fail "This test does not run on Windows."
  ;;
darwin*)
  # As of 2018-12-17, this test is disabled on mac, but there's no "no_mac" tag
  # so we just have to trivially succeed. See the TODO at
  # use_system_python_2_3_runtimes.
  echo "This test does not run on Mac; exiting early." >&2
  exit 0
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

# Use a py_runtime that invokes either the system's Python 2 or Python 3
# interpreter based on the Python mode. On Unix this is a workaround for #4815.
#
# TODO(brandjon): Get this running on windows by creating .bat wrappers that
# invoke "py -2" and "py -3". Make sure our windows workers have both Python
# versions installed.
#
# TODO(brandjon): Get this running on mac -- our workers lack a Python 2
# installation.
#
function use_system_python_2_3_runtimes() {
  PYTHON2_BIN=$(which python2 || echo "")
  PYTHON3_BIN=$(which python3 || echo "")
  # Debug output.
  echo "Python 2 interpreter: ${PYTHON2_BIN:-"Not found"}"
  echo "Python 3 interpreter: ${PYTHON3_BIN:-"Not found"}"
  # Fail if either isn't present.
  if [[ -z "${PYTHON2_BIN:-}" || -z "${PYTHON3_BIN:-}" ]]; then
    fail "Can't use system interpreter: Could not find one or both of \
'python2', 'python3'"
  fi

  add_to_bazelrc "build --python_top=//tools/python:default_runtime"

  mkdir -p tools/python

  cat > tools/python/BUILD << EOF
package(default_visibility=["//visibility:public"])

sh_binary(
    name = '2to3',
    srcs = ['2to3.sh']
)

config_setting(
    name = "py3_mode",
    values = {"force_python": "PY3"},
)

# TODO(brandjon): Replace dependency on "force_python" with a 2-valued feature
# flag instead
py_runtime(
    name = "default_runtime",
    files = [],
    interpreter_path = select({
        "py3_mode": "${PYTHON3_BIN}",
        "//conditions:default": "${PYTHON2_BIN}",
    }),
)
EOF
}

#### TESTS #############################################################

# Sanity test that our environment setup above works.
function test_can_run_py_binaries() {
  use_system_python_2_3_runtimes

  mkdir -p test

  cat > test/BUILD << EOF
py_binary(
    name = "main2",
    default_python_version = "PY2",
    srcs = ['main2.py'],
)
py_binary(
    name = "main3",
    default_python_version = "PY3",
    srcs = ["main3.py"],
)
EOF

  cat > test/main2.py << EOF
import platform
print("I am Python " + platform.python_version_tuple()[0])
EOF
  cp test/main2.py test/main3.py
  chmod u+x test/main2.py test/main3.py

  bazel run //test:main2 \
      &> $TEST_log || fail "bazel run failed"
  expect_log "I am Python 2"

  bazel run //test:main3 \
      &> $TEST_log || fail "bazel run failed"
  expect_log "I am Python 3"
}

# Test that access to runfiles works (in general, and under our test environment
# specifically).
function test_can_access_runfiles() {
  use_system_python_2_3_runtimes

  mkdir -p test

  cat > test/BUILD << EOF
py_binary(
  name = "main",
  srcs = ["main.py"],
  deps = ["@bazel_tools//tools/python/runfiles"],
  data = ["data.txt"],
)
EOF

  cat > test/data.txt << EOF
abcdefg
EOF

  cat > test/main.py << EOF
from bazel_tools.tools.python.runfiles import runfiles

r = runfiles.Create()
path = r.Rlocation("$WORKSPACE_NAME/test/data.txt")
print("Rlocation returned: " + str(path))
if path is not None:
  with open(path, 'rt') as f:
    print("File contents: " + f.read())
EOF
  chmod u+x test/main.py

  bazel build //test:main || fail "bazel build failed"
  MAIN_BIN=$(bazel info bazel-bin)/test/main
  RUNFILES_MANIFEST_FILE= RUNFILES_DIR= $MAIN_BIN &> $TEST_log
  expect_log "File contents: abcdefg"
}

run_suite "Tests for how the Python rules handle Python 2 vs Python 3"
