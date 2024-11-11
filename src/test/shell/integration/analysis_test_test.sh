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
#
# Tests the examples provided in Bazel
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

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

function test_passing_test() {
  mkdir -p package
  cat > package/test.bzl <<EOF
def _rule_test_impl(ctx):
  return [AnalysisTestResultInfo(success = True,
                                 message = 'A success message!')]

my_rule_test = rule(
  implementation = _rule_test_impl,
  analysis_test = True,
)
EOF

  cat > package/BUILD <<EOF
load(":test.bzl", "my_rule_test")

my_rule_test(name = "r")
EOF

  bazel test package:r >& "$TEST_log" || fail "Unexpected failure"

  expect_log "PASSED"

  cat "${PRODUCT_NAME}-testlogs/package/r/test.log" > "$TEST_log"

  expect_log "A success message!"
}

function test_failing_test() {
  mkdir -p package
  cat > package/test.bzl <<EOF
def _rule_test_impl(ctx):
  return [AnalysisTestResultInfo(success = False,
                                 message = 'A failure message!')]

my_rule_test = rule(
  implementation = _rule_test_impl,
  analysis_test = True,
)
EOF

  cat > package/BUILD <<EOF
load(":test.bzl", "my_rule_test")

my_rule_test(name = "r")
EOF

  ! bazel test package:r >& "$TEST_log" || fail "Unexpected success"

  expect_log "FAILED"

  cat "${PRODUCT_NAME}-testlogs/package/r/test.log" > "$TEST_log"

  expect_log "A failure message!"
}

function test_failing_test_shell_escape_in_message() {
  mkdir -p package
  cat > package/test.bzl <<'EOF'
def _rule_test_impl(ctx):
  return [AnalysisTestResultInfo(success = False,
                                 message = 'Command is not "$1 copy $2"')]

my_rule_test = rule(
  implementation = _rule_test_impl,
  analysis_test = True,
)
EOF

  cat > package/BUILD <<EOF
load(":test.bzl", "my_rule_test")

my_rule_test(name = "r")
EOF

  ! bazel test package:r >& "$TEST_log" || fail "Unexpected success"

  expect_log "FAILED"

  cat "${PRODUCT_NAME}-testlogs/package/r/test.log" > "$TEST_log"

  expect_log 'Command is not "$1 copy $2"'
}

function test_failing_test_cmd_escape_in_message() {
  mkdir -p package
  cat > package/test.bzl <<'EOF'
def _rule_test_impl(ctx):
  return [
      AnalysisTestResultInfo(
          success = False,
          message = 'Command should contain "\\ & < > | ^ ! %FOO%"',
      ),
  ]

my_rule_test = rule(
  implementation = _rule_test_impl,
  analysis_test = True,
)
EOF

  cat > package/BUILD <<EOF
load(":test.bzl", "my_rule_test")

my_rule_test(name = "r")
EOF

  ! bazel test package:r >& "$TEST_log" || fail "Unexpected success"

  expect_log "FAILED"

  cat "${PRODUCT_NAME}-testlogs/package/r/test.log" > "$TEST_log"

  expect_log 'Command should contain "\\ & < > \| ^ ! %FOO%"'
}

function test_failing_test_eof_string_in_message() {
  mkdir -p package
  cat > package/test.bzl <<'EOF'
def _rule_test_impl(ctx):
  return [AnalysisTestResultInfo(success = False,
                                 message = '"\nEOF\n" not in command')]

my_rule_test = rule(
  implementation = _rule_test_impl,
  analysis_test = True,
)
EOF

  cat > package/BUILD <<EOF
load(":test.bzl", "my_rule_test")

my_rule_test(name = "r")
EOF

  ! bazel test package:r >& "$TEST_log" || fail "Unexpected success"

  expect_log "FAILED"

  cat "${PRODUCT_NAME}-testlogs/package/r/test.log" > "$TEST_log"

  # expect_log uses grep and looks at individual lines, but we can make sure
  # the part after \nEOF\n isn't cut off, as it was previously.
  expect_log "\" not in command"
}

function test_expected_failure_test() {
  mkdir -p package
  cat > package/test.bzl <<EOF
def _rule_test_impl(ctx):
  if AnalysisFailureInfo in ctx.attr.target_under_test[0]:
    dep_failure = ctx.attr.target_under_test[0][AnalysisFailureInfo]
    message = ("target_under_test failed: " +
        dep_failure.causes.to_list()[0].message)
    return [AnalysisTestResultInfo(success = True,
                                   message = message)]
  else:
    return [AnalysisTestResultInfo(success = False,
                                   message = "Expected dep failure")]

test_transition = analysis_test_transition(
  settings = {
      "//command_line_option:allow_analysis_failures" : "True" }
)

def _rule_impl(ctx):
    f = "foo".method_doesnt_exist()
    return []

my_rule = rule(
    implementation = _rule_impl,
)

my_rule_test = rule(
  implementation = _rule_test_impl,
  analysis_test = True,
  attrs = {
      'target_under_test': attr.label(cfg = test_transition),
  }
)
EOF

  cat > package/BUILD <<EOF
load(":test.bzl", "my_rule_test", "my_rule")

my_rule(name = "target_under_test")

my_rule_test(name = "test_target", target_under_test = ":target_under_test")
EOF

  bazel test package:test_target >& "$TEST_log" \
      || fail "Expected test to succeed"

  expect_log "PASSED"

  cat "${PRODUCT_NAME}-testlogs/package/test_target/test.log" > "$TEST_log"
  expect_log "target_under_test failed"
  expect_log "'string' value has no field or method 'method_doesnt_exist'"
}

run_suite "analysis_test rule tests"
