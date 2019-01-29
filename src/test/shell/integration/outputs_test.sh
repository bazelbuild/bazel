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
# Test rules with outputs definitions.
#

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

function test_plain_outputs() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg || fail "mkdir -p $pkg failed"
  cat >$pkg/rule.bzl <<EOF
def _impl(ctx):
  ctx.file_action(
      output=ctx.outputs.out,
      content="Hello World!"
  )
  return []

demo_rule = rule(
  _impl,
  attrs = {
    'foo': attr.string(),
  },
  outputs = {
    'out': '%{foo}.txt'
  })
EOF

  cat >$pkg/BUILD <<EOF
load(':rule.bzl', 'demo_rule')

demo_rule(
  name = 'demo',
  foo = 'demo_output_name')
EOF

  bazel build //$pkg:demo &> $TEST_log || fail "Build failed"
  expect_log "demo_output_name.txt"
}

function test_function_outputs() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg || fail "mkdir -p $pkg failed"
  cat >$pkg/rule.bzl <<EOF
def _outputs(foo):
  return {
    'out': foo + '.txt',
  }

def _impl(ctx):
  ctx.file_action(
      output=ctx.outputs.out,
      content="Hello World!"
  )
  return []

demo_rule = rule(
  _impl,
  attrs = {
    'foo': attr.string(),
  },
  outputs = _outputs)
EOF

  cat >$pkg/BUILD <<EOF
load(':rule.bzl', 'demo_rule')

demo_rule(
  name = 'demo',
  foo = 'demo_output_name')
EOF

  bazel build //$pkg:demo &> $TEST_log || fail "Build failed"
  expect_log "demo_output_name.txt"
}

function test_output_select_error() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg || fail "mkdir -p $pkg failed"
  cat >$pkg/rule.bzl <<EOF
def _impl(ctx):
  ctx.file_action(
      output=ctx.outputs.out,
      content="Hello World!"
  )
  return []

demo_rule = rule(
  _impl,
  attrs = {
    'foo': attr.string(),
  },
  outputs = select({
    '//conditions:default': {
      'out': '%{foo}.txt'
    }
  }))
EOF

  cat >$pkg/BUILD <<EOF
load(':rule.bzl', 'demo_rule')

demo_rule(
  name = 'demo',
  foo = 'a_str')
EOF

  if bazel build //$pkg:demo &> $TEST_log; then
    fail "Build expected to fail"
  fi
  expect_log "expected value of type 'dict or NoneType or function' for parameter 'outputs', for call to function rule"
}

function test_configurable_output_error() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg || fail "mkdir -p $pkg failed"
  cat >$pkg/rule.bzl <<EOF
def _impl(ctx):
  ctx.file_action(
      output=ctx.outputs.out,
      content="Hello World!"
  )
  return []

demo_rule = rule(
  _impl,
  attrs = {
    'foo': attr.string(),
  },
  outputs = {
    'out': '%{foo}.txt'
  })
EOF

  cat >$pkg/BUILD <<EOF
load(':rule.bzl', 'demo_rule')

demo_rule(
  name = 'demo',
  foo = select({
    '//conditions:default': 'selectable_str',
  }))
EOF

  if bazel build //$pkg:demo &> $TEST_log; then
    fail "Build expected to fail"
  fi
  expect_log "Attribute foo is configurable and cannot be used in outputs"
}

run_suite "skylark outputs tests"
