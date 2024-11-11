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
  # As of 2019-02-18, Bazel on Windows only supports MSYS Bash.
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

function create_pkg() {
  if [[ -d foo ]]; then
    return
  fi

  mkdir foo || fail "mkdir foo"

  cat >foo/BUILD <<'EOF'
load(":rules.bzl", "args1_test", "args2_test")
args1_test(name = "x")
args2_test(name = "y")
EOF

  cat >foo/rules.bzl <<'EOF'
def _impl(ctx, cmd, out):
    output = ctx.actions.declare_file(out)
    ctx.actions.run_shell(
        outputs = [output],
        command = cmd,
        arguments = ['echo', 'a', 'b', 'c'],
    )
    return [DefaultInfo(executable = output)]

def _impl1(ctx):
  # See https://github.com/bazelbuild/bazel/issues/7122
  return _impl(ctx, '${1+"$@"}', "foo1")

def _impl2(ctx):
  # See https://github.com/bazelbuild/bazel/issues/7122
  return _impl(ctx, '${1+"$@"} ', "foo2")

args1_test = rule(implementation = _impl1, test = True)

args2_test = rule(implementation = _impl2, test = True)
EOF
}

function assert_command_succeeded() {
  local -r output=$1

  # The command succeeds, though the build fails because the command does not
  # create any outputs. This is expected.
  expect_log "From.* foo/${output}"
  expect_log "output .*foo/${output}.* was not created"
  expect_log "not all outputs were created"
}

function test_windows_style_arg_escaping() {
  create_pkg

  bazel build foo:x --verbose_failures 2>$TEST_log \
      && fail "expected failure" || true
  assert_command_succeeded foo1

  bazel build foo:y --verbose_failures 2>$TEST_log \
      && fail "expected failure" || true
  assert_command_succeeded foo2
}

run_suite "Windows argument escaping test"
