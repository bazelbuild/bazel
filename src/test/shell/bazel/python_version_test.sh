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

  # Point Python builds at a py_runtime target defined in a //tools package of
  # the main repo. This is not related to @bazel_tools//tools/python.
  add_to_bazelrc "build --python_top=//tools/python:default_runtime"

  mkdir -p tools/python

  cat > tools/python/BUILD << EOF
package(default_visibility=["//visibility:public"])

py_runtime(
    name = "default_runtime",
    files = [],
    interpreter_path = select({
        "@bazel_tools//tools/python:PY2": "${PYTHON2_BIN}",
        "@bazel_tools//tools/python:PY3": "${PYTHON3_BIN}",
    }),
)
EOF
}

function set_up() {
  use_system_python_2_3_runtimes
}

#### TESTS #############################################################

# Sanity test that our environment setup above works.
function test_can_run_py_binaries() {
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

function test_pybin_can_have_different_version_pybin_as_data_dep() {
  mkdir -p test

  cat > test/BUILD << EOF
py_binary(
  name = "py2bin",
  srcs = ["py2bin.py"],
  default_python_version = "PY2",
)
py_binary(
  name = "py3bin",
  srcs = ["py3bin.py"],
  default_python_version = "PY3",
)
py_binary(
  name = "py2bin_calling_py3bin",
  srcs = ["py2bin_calling_py3bin.py"],
  deps = ["@bazel_tools//tools/python/runfiles"],
  data = [":py3bin"],
  default_python_version = "PY2",
)
py_binary(
  name = "py3bin_calling_py2bin",
  srcs = ["py3bin_calling_py2bin.py"],
  deps = ["@bazel_tools//tools/python/runfiles"],
  data = [":py2bin"],
  default_python_version = "PY3",
)
EOF

  cat > test/py2bin.py << EOF
import platform

print("Inner bin uses Python " + platform.python_version_tuple()[0])
EOF
  chmod u+x test/py2bin.py
  cp test/py2bin.py test/py3bin.py

  cat > test/py2bin_calling_py3bin.py << EOF
import platform
import subprocess
from bazel_tools.tools.python.runfiles import runfiles

r = runfiles.Create()
bin_path = r.Rlocation("$WORKSPACE_NAME/test/py3bin")

print("Outer bin uses Python " + platform.python_version_tuple()[0])
subprocess.call([bin_path])
EOF
  sed s/py3bin/py2bin/ test/py2bin_calling_py3bin.py > test/py3bin_calling_py2bin.py
  chmod u+x test/py2bin_calling_py3bin.py test/py3bin_calling_py2bin.py

  EXPFLAG="--experimental_better_python_version_mixing=true"

  bazel build $EXPFLAG //test:py2bin_calling_py3bin //test:py3bin_calling_py2bin \
      || fail "bazel build failed"
  PY2_OUTER_BIN=$(bazel info bazel-bin $EXPFLAG)/test/py2bin_calling_py3bin
  PY3_OUTER_BIN=$(bazel info bazel-bin $EXPFLAG --force_python=PY3)/test/py3bin_calling_py2bin

  RUNFILES_MANIFEST_FILE= RUNFILES_DIR= $PY2_OUTER_BIN &> $TEST_log
  expect_log "Outer bin uses Python 2"
  expect_log "Inner bin uses Python 3"

  RUNFILES_MANIFEST_FILE= RUNFILES_DIR= $PY3_OUTER_BIN &> $TEST_log
  expect_log "Outer bin uses Python 3"
  expect_log "Inner bin uses Python 2"
}

function test_shbin_can_have_different_version_pybins_as_data_deps() {
  mkdir -p test

  cat > test/BUILD << EOF
py_binary(
  name = "py2bin",
  srcs = ["py2bin.py"],
  default_python_version = "PY2",
)
py_binary(
  name = "py3bin",
  srcs = ["py3bin.py"],
  default_python_version = "PY3",
)
sh_binary(
  name = "shbin_calling_py23bins",
  srcs = ["shbin_calling_py23bins.sh"],
  deps = ["@bazel_tools//tools/bash/runfiles"],
  data = [":py2bin", ":py3bin"],
)
EOF

  cat > test/py2bin.py << EOF
import platform

print("py2bin uses Python " + platform.python_version_tuple()[0])
EOF
  sed s/py2bin/py3bin/ test/py2bin.py > test/py3bin.py
  chmod u+x test/py2bin.py test/py3bin.py

  # The workspace name is initialized in testenv.sh; use that var rather than
  # hardcoding it here. The extra sed pass is so we can selectively expand that
  # one var while keeping the rest of the heredoc literal.
  cat | sed "s/{{WORKSPACE_NAME}}/$WORKSPACE_NAME/" > test/shbin_calling_py23bins.sh << 'EOF'
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

$(rlocation {{WORKSPACE_NAME}}/test/py2bin)
$(rlocation {{WORKSPACE_NAME}}/test/py3bin)
EOF

  chmod u+x test/shbin_calling_py23bins.sh

  EXPFLAG="--experimental_better_python_version_mixing=true"

  bazel build $EXPFLAG //test:shbin_calling_py23bins \
      || fail "bazel build failed"
  SH_BIN=$(bazel info bazel-bin $EXPFLAG)/test/shbin_calling_py23bins

  RUNFILES_MANIFEST_FILE= RUNFILES_DIR= $SH_BIN &> $TEST_log
  expect_log "py2bin uses Python 2"
  expect_log "py3bin uses Python 3"
}

function test_genrule_can_have_different_version_pybins_as_tools() {
  # This test currently checks that we can use --host_force_python to get
  # PY2 and PY3 binaries in tools. In the future we'll support both modes in the
  # same build without a flag (#6443).

  mkdir -p test

  cat > test/BUILD << 'EOF'
py_binary(
  name = "pybin",
  srcs = ["pybin.py"],
)
genrule(
  name = "genrule_calling_pybin",
  outs = ["out.txt"],
  tools = [":pybin"],
  cmd = "$(location :pybin) > $(location out.txt)"
)
EOF

  cat > test/pybin.py << EOF
import platform

print("pybin uses Python " + platform.python_version_tuple()[0])
EOF
  chmod u+x test/pybin.py

  # Run under both old and new semantics.
  for EXPFLAG in \
      "--experimental_better_python_version_mixing=true" \
      "--experimental_better_python_version_mixing=false"; do
    echo "Using $EXPFLAG" > $TEST_log
    bazel build $EXPFLAG --host_force_python=PY2 //test:genrule_calling_pybin \
        || fail "bazel build failed"
    ARTIFACT=$(bazel info bazel-genfiles $EXPFLAG)/test/out.txt
    cat $ARTIFACT > $TEST_log
    expect_log "pybin uses Python 2"

    echo "Using $EXPFLAG" > $TEST_log
    bazel build $EXPFLAG --host_force_python=PY3 //test:genrule_calling_pybin \
          || fail "bazel build failed"
      ARTIFACT=$(bazel info bazel-genfiles $EXPFLAG)/test/out.txt
      cat $ARTIFACT > $TEST_log
      expect_log "pybin uses Python 3"
  done
}

function test_can_build_same_target_for_both_versions_in_one_build() {
  mkdir -p test

  cat > test/BUILD << EOF
py_library(
  name = "common",
  srcs = ["common.py"],
)
py_binary(
  name = "py2bin",
  srcs = ["py2bin.py"],
  deps = [":common"],
  default_python_version = "PY2",
)
py_binary(
  name = "py3bin",
  srcs = ["py3bin.py"],
  deps = [":common"],
  default_python_version = "PY3",
)
sh_binary(
  name = "shbin",
  srcs = ["shbin.sh"],
  deps = ["@bazel_tools//tools/bash/runfiles"],
  data = [":py2bin", ":py3bin"],
)
EOF

  cat > test/common.py << EOF
import platform

print("common using Python " + platform.python_version_tuple()[0])
EOF

  cat > test/py2bin.py << EOF
import common
EOF
  chmod u+x test/py2bin.py
  cp test/py2bin.py test/py3bin.py

  # The workspace name is initialized in testenv.sh; use that var rather than
  # hardcoding it here. The extra sed pass is so we can selectively expand that
  # one var while keeping the rest of the heredoc literal.
  cat | sed "s/{{WORKSPACE_NAME}}/$WORKSPACE_NAME/" > test/shbin.sh << 'EOF'
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

$(rlocation {{WORKSPACE_NAME}}/test/py2bin)
$(rlocation {{WORKSPACE_NAME}}/test/py3bin)
EOF
  chmod u+x test/shbin.sh

  EXPFLAG="--experimental_better_python_version_mixing=true"

  bazel build $EXPFLAG //test:shbin \
      || fail "bazel build failed"
  SH_BIN=$(bazel info bazel-bin)/test/shbin

  RUNFILES_MANIFEST_FILE= RUNFILES_DIR= $SH_BIN &> $TEST_log
  expect_log "common using Python 2"
  expect_log "common using Python 3"
}

# TODO(brandjon): Rename this file to python_test.sh or else move the below to
# a separate suite.

# Tests that a non-standard library module on the PYTHONPATH added by Bazel
# can override the standard library. This behavior is not necessarily ideal, but
# it is the current semantics; see #6532 about changing that.
function test_source_file_does_not_override_standard_library() {
  mkdir -p test

  cat > test/BUILD << EOF
py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [":lib"],
    # Pass the empty string, to include the path to this package (within
    # runfiles) on the PYTHONPATH.
    imports = [""],
)

py_library(
    name = "lib",
    # A src name that clashes with a standard library module, such that this
    # local file can take precedence over the standard one depending on its
    # order in PYTHONPATH. Not just any module name would work. For instance,
    # "import sys" gets the built-in module regardless of whether there's some
    # "sys.py" file on the PYTHONPATH. This is probably because built-in modules
    # (i.e., those implemented in C) use a different loader than
    # Python-implemented ones, even though they're both part of the standard
    # distribution of the interpreter.
    srcs = ["re.py"],
)
EOF
  cat > test/main.py << EOF
import re
EOF
  cat > test/re.py << EOF
print("I am lib!")
EOF

  bazel run //test:main \
      &> $TEST_log || fail "bazel run failed"
  # Indicates that the local module overrode the system one.
  expect_log "I am lib!"
}

run_suite "Tests for how the Python rules handle Python 2 vs Python 3"
