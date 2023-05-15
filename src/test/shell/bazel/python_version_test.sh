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

#### TESTS #############################################################

# Check that our environment setup works.
function test_can_run_py_binaries() {
  mkdir -p test

  cat > test/BUILD << EOF
py_binary(
    name = "main3",
    python_version = "PY3",
    srcs = ["main3.py"],
)
EOF

  cat > test/main3.py << EOF
import platform
print("I am Python " + platform.python_version_tuple()[0])
EOF

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

# Regression test for #5104. This test ensures that it's possible to use
# --build_python_zip in combination with an in-workspace runtime, as opposed to
# with a system runtime or not using a py_runtime at all (the legacy
# --python_path mechanism).
#
# The specific issue #5104 was caused by file permissions being lost when
# unzipping runfiles, which led to an unexecutable runtime.
function test_build_python_zip_works_with_workspace_runtime() {
  mkdir -p test

  # The runfiles interpreter is either a sh script or bat script depending on
  # the current platform.
  if "$is_windows"; then
    INTERPRETER_FILE="mockpy.bat"
  else
    INTERPRETER_FILE="mockpy.sh"
  fi

  cat > test/BUILD << EOF
load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")

py_binary(
    name = "pybin",
    srcs = ["pybin.py"],
)

py_runtime(
    name = "mock_runtime",
    interpreter = ":$INTERPRETER_FILE",
    python_version = "PY3",
)

py_runtime_pair(
    name = "mock_runtime_pair",
    py3_runtime = ":mock_runtime",
)

toolchain(
    name = "mock_toolchain",
    toolchain = ":mock_runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)
EOF
  cat > test/pybin.py << EOF
# This doesn't actually run because we use a mock Python runtime that never
# executes the Python code.
print("I am pybin!")
EOF
  if "$is_windows"; then
    cat > "test/$INTERPRETER_FILE" << EOF
@ECHO I am mockpy!
EOF
  else
    cat > "test/$INTERPRETER_FILE" << EOF
#!/bin/sh
echo "I am mockpy!"
EOF
    chmod u+x test/mockpy.sh
  fi

  bazel run //test:pybin \
      --extra_toolchains=//test:mock_toolchain --build_python_zip \
      &> $TEST_log || fail "bazel run failed"
  expect_log "I am mockpy!"
}

# Verify that looking up runfiles that require repo mapping works
function test_build_python_zip_bzlmod_repo_mapping_runfiles() {
  cat > WORKSPACE
  cat > MODULE.bazel << EOF
module(name="pyzip")
bazel_dep(name = "rules_python", version = "0.19.0")
EOF
  mkdir test
  cat > test/BUILD << EOF
py_binary(
  name = "pybin",
  srcs = ["pybin.py"],
  deps = ["@rules_python//python/runfiles"],
  data = ["data.txt"],
)
EOF
  echo "data" > test/data.txt
  cat > test/pybin.py << EOF
from python.runfiles import runfiles
rf = runfiles.Create()
path = rf.Rlocation("pyzip/test/data.txt")
with open(path, "r") as fp:
  fp.read()
EOF

  bazel run --enable_bzlmod --build_python_zip //test:pybin &> $TEST_log || fail "bazel run failed"

  unzip -p bazel-bin/test/pybin.zip runfiles/_repo_mapping > actual_repo_mapping
  assert_contains ",pyzip,_main" actual_repo_mapping
}

# Test that running a zip app without RUN_UNDER_RUNFILES=1 removes the
# temporary directory it creates
function test_build_python_zip_cleans_up_temporary_module_space() {

  mkdir test
  cat > test/BUILD << EOF
py_binary(
  name = "pybin",
  srcs = ["pybin.py"],
)
EOF
  cat > test/pybin.py << EOF
print(__file__)
EOF

  bazel build //test:pybin --build_python_zip &> $TEST_log || fail "bazel build failed"
  pybin_location=$(bazel-bin/test/pybin)

  # The pybin location is "<ms root>/runfiles/<workspace>/test/pybin.py",
  # so we have to go up 4 directories to get to the module space root
  module_space_dir=$(dirname $(dirname $(dirname $(dirname "$pybin_location"))))
  if [[ -d "$module_space_dir" ]]; then
    fail "expected module space directory to be deleted, but $module_space_dir still exists"
  fi
}

function test_get_python_zip_file_via_output_group() {
  touch foo.py
  cat > BUILD <<'EOF'
py_binary(
  name = 'foo',
  srcs = ['foo.py'],
)
EOF
  bazel build :foo --build_python_zip=false --output_groups=python_zip_file \
      &> $TEST_log || fail "bazel build failed"
  [[ -f "bazel-bin/foo.zip" ]] || fail "failed to build python zip file via output group"
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
    srcs = ["mailbox.py"],
)
EOF
  cat > test/main.py << EOF
import mailbox
EOF
  cat > test/mailbox.py << EOF
print("I am lib!")
EOF

  bazel run //test:main \
      &> $TEST_log || fail "bazel run failed"
  # Indicates that the local module overrode the system one.
  expect_log "I am lib!"
}

# Tests that targets appear under the expected roots.
function test_output_roots() {
  # It's hard to get build output paths reliably, so we'll just check the output
  # of bazel info.

  # Legacy behavior, PY3 case.
  bazel info bazel-bin \
      --incompatible_py2_outputs_are_suffixed=false --python_version=PY3 \
      &> $TEST_log || fail "bazel info failed"
  expect_log "bazel-out/.*-py3.*/bin"

  # New behavior, PY3 case.
  bazel info bazel-bin \
      --incompatible_py2_outputs_are_suffixed=true --python_version=PY3 \
      &> $TEST_log || fail "bazel info failed"
  expect_log "bazel-out/.*/bin"
  expect_not_log "bazel-out/.*-py3.*/bin"
}

# Tests that bazel-bin points to where targets get built by default (or at least
# not to a directory with a -py2 or -py3 suffix), provided that
# --incompatible_py3_is_default and --incompatible_py2_outputs_are_suffixed are
# flipped together.
function test_default_output_root_is_bazel_bin() {
  bazel info bazel-bin \
      --incompatible_py3_is_default=false \
      --incompatible_py2_outputs_are_suffixed=false \
      &> $TEST_log || fail "bazel info failed"
  expect_log "bazel-out/.*/bin"
  expect_not_log "bazel-out/.*-py2.*/bin"
  expect_not_log "bazel-out/.*-py3.*/bin"

  bazel info bazel-bin \
      --incompatible_py3_is_default=true \
      --incompatible_py2_outputs_are_suffixed=true \
      &> $TEST_log || fail "bazel info failed"
  expect_log "bazel-out/.*/bin"
  expect_not_log "bazel-out/.*-py2.*/bin"
  expect_not_log "bazel-out/.*-py3.*/bin"
}

# Regression test for (bazelbuild/continuous-integration#578): Ensure that a
# py_binary built with the autodetecting toolchain works when used as a tool
# from Starlark rule. In particular, the wrapper script that launches the real
# second-stage interpreter must be able to tolerate PATH not being set.
function test_py_binary_with_autodetecting_toolchain_usable_as_tool() {
  mkdir -p test

  cat > test/BUILD << 'EOF'
load(":tooluser.bzl", "tooluser_rule")

py_binary(
    name = "tool",
    srcs = ["tool.py"],
)

tooluser_rule(
    name = "tooluser",
    out = "out.txt",
)
EOF
  cat > test/tooluser.bzl << EOF
def _tooluser_rule_impl(ctx):
    ctx.actions.run(
        inputs = [],
        outputs = [ctx.outputs.out],
        executable = ctx.executable._tool,
        arguments = [ctx.outputs.out.path],
    )

tooluser_rule = rule(
    implementation = _tooluser_rule_impl,
    attrs = {
        "_tool": attr.label(
            executable = True,
            default = "//test:tool",
            # cfg param is required but its value doesn't matter
            cfg = "target"),
        "out": attr.output(),
    },
)
EOF
  cat > test/tool.py << EOF
import sys
with open(sys.argv[1], 'wt') as out:
    print("Tool output", file=out)
EOF

  bazel build //test:tooluser \
      --incompatible_use_python_toolchains=true \
      || fail "bazel build failed"
  cat bazel-bin/test/out.txt &> $TEST_log
  expect_log "Tool output"
}

function test_external_runfiles() {
  cat >> WORKSPACE <<EOF
local_repository(
  name = "repo2",
  path = "repo2"
)
EOF
  mkdir repo2
  touch repo2/WORKSPACE
  cat > repo2/BUILD <<EOF
package(default_visibility=["//visibility:public"])
filegroup(name="r2files", srcs=["r2.txt"])
EOF
  touch repo2/r2.txt

  mkdir py
  cat > py/BUILD <<EOF
py_binary(
  name = "foo", srcs=["foo.py"],
  data = ["@repo2//:r2files"],
)
EOF
  touch py/foo.py

  # We're testing for this flag's behavior, so force it to true.
  # TODO(https://github.com/bazelbuild/bazel/issues/12821): Remove this test
  # when this behavior is removed
  bazel build --legacy_external_runfiles=true //py:foo
  if "$is_windows"; then
    exe=".exe"
  else
    exe=""
  fi

  # NOTE: The "main" name isn't special. It's just the name the integration test
  # setup puts in WORKSPACE.
  cp bazel-bin/py/foo$exe.runfiles_manifest runfiles_manifest
  assert_contains main/external/repo2/r2.txt runfiles_manifest \
    "runfiles manifest didn't have external path mapping"

  # By default, Python binaries are put into zip files on Windows and don't
  # have a real runfiles tree.
  if ! "$is_windows"; then
    find bazel-bin/py/foo.runfiles > runfiles_listing
    assert_contains bazel-bin/py/foo.runfiles/main/external/repo2/r2.txt \
      runfiles_listing \
      "runfiles didn't have external links"
  fi
}

run_suite "Tests for how the Python rules handle Python 2 vs Python 3"
