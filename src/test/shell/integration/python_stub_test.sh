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
  declare -r EXE_EXT=".exe"
else
  declare -r EXE_EXT=""
fi

# Tests in this file do not actually start a Python interpreter, but plug in a
# fake stub executable to serve as the "interpreter".
#
# Note that this means this suite cannot be used for tests of the actual stub
# script under Windows, since the stub script never runs (the launcher uses the
# mock interpreter rather than a system interpreter, see discussion in #7947).

use_fake_python_runtimes_for_testsuite

#### TESTS #############################################################

# Tests that Python 2 or Python 3 is actually invoked.
function test_python_version() {
  add_rules_python "MODULE.bazel"
  mkdir -p test
  touch test/main3.py
  cat > test/BUILD << EOF
load("@rules_python//python:py_binary.bzl", "py_binary")

py_binary(name = "main3",
    python_version = "PY3",
    srcs = ["main3.py"],
)
EOF

  # Stamping is disabled so that the invocation doesn't time out. What
  # happens is Google has stamping enabled by default, which causes the
  # Starlark rule implementation to run an action, which then tries to run
  # remotely, but network access is disabled by default, so it times out.
  bazel run --nostamp //test:main3 \
      &> $TEST_log || fail "bazel run failed"
  expect_log "I am Python 3"
}

function test_can_build_py_library_at_top_level_regardless_of_version() {
  add_rules_python "MODULE.bazel"
  mkdir -p test
  cat > test/BUILD << EOF
load("@rules_python//python:py_library.bzl", "py_library")

py_library(
    name = "lib3",
    srcs = ["lib3.py"],
    srcs_version = "PY3ONLY",
)
EOF
  touch test/lib3.py

  bazel build --python_version=PY3 //test:* \
      &> $TEST_log || fail "bazel build failed"
}

# When invoking a Python binary using the runfiles manifest, the stub
# script's argv[0] will point to a location in the execroot; not the
# runfiles directory of the caller. The stub script should still be
# capable of finding its runfiles directory by considering RUNFILES_DIR
# and RUNFILES_MANIFEST_FILE set by the caller.
function test_python_through_bash_without_runfile_links() {
  add_rules_python "MODULE.bazel"
  mkdir -p python_through_bash

  cat > python_through_bash/BUILD << EOF
load("@rules_python//python:py_binary.bzl", "py_binary")

py_binary(
    name = "inner",
    srcs = ["inner.py"],
)

sh_binary(
    name = "outer",
    srcs = ["outer.sh"],
    data = [":inner"],
)
EOF

  cat > python_through_bash/outer.sh << EOF
#!/bin/bash
# * Bazel run guarantees that our CWD is the runfiles directory itself, so a
#   relative path will work.
# * We can't use the usual shell runfiles library because it doesn't work in the
#   Google environment nested within a generated shell test.
find . -name inner$EXE_EXT | xargs env
EOF
  chmod +x python_through_bash/outer.sh

  touch python_through_bash/inner.py

  # The inner Python script requires runfiles, so force them on Windows.
  bazel run --nobuild_runfile_links --enable_runfiles \
    //python_through_bash:outer &> $TEST_log || fail "bazel run failed"
  expect_log "I am Python"
}

run_suite "Tests for the Python rules without Python execution"
