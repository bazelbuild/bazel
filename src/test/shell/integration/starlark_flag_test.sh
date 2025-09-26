#!/usr/bin/env bash
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

# Tests that the Starlark interpreter is reading flags passed in on the command
# line, in several different evaluation contexts.
#
# The --internal_starlark_flag_test_canary flag is built into
# BuildLanguageOptions specifically for this test suite.

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

# Text that will be appended to every print() output when the flag is enabled.
MARKER="<== Starlark flag test ==>"

sanity_fail_msg="Marker string '$MARKER' was seen even though "
sanity_fail_msg+="--internal_starlark_flag_test_canary wasn't passed"


function test_build_file() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<'EOF' || fail "couldn't create file"
print("In BUILD: ")

genrule(
    name = "dummy",
    cmd = "echo 'dummy' >$@",
    outs = ["dummy.txt"],
)
EOF

  # Sanity check.
  bazel build //$pkg:dummy \
      &>"$TEST_log" || fail "bazel build failed";
  expect_log "In BUILD: " "Did not find BUILD print output"
  expect_not_log "$MARKER" "$sanity_fail_msg"

  bazel build //$pkg:dummy \
      --internal_starlark_flag_test_canary \
      &>"$TEST_log" || fail "bazel build failed";
  expect_log "In BUILD: $MARKER" \
      "Starlark flags are not propagating to BUILD file evaluation"
}

function test_bzl_file_and_macro() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<'EOF' || fail "couldn't create file"
load(":test.bzl", "macro")

macro()
EOF
  cat >$pkg/test.bzl <<'EOF' || fail "couldn't create file"
print("In bzl: ")

def macro():
  print("In macro: ")
  native.genrule(
      name = "dummy",
      cmd = "echo 'dummy' >$@",
      outs = ["dummy.txt"],
  )
EOF

  # Sanity check.
  bazel build //$pkg:dummy \
      &>"$TEST_log" || fail "bazel build failed";
  expect_log "In bzl: " "Did not find .bzl print output"
  expect_log "In macro: " "Did not find macro print output"
  expect_not_log "$MARKER" "$sanity_fail_msg"

  bazel build //$pkg:dummy \
      --internal_starlark_flag_test_canary \
      &>"$TEST_log" || fail "bazel build failed";
  expect_log "In bzl: $MARKER" \
      "Starlark flags are not propagating to .bzl file evaluation"
  expect_log "In macro: $MARKER" \
      "Starlark flags are not propagating to macro evaluation"
}

function test_rule() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<'EOF' || fail "couldn't create file"
load(":test.bzl", "some_rule")

some_rule(
    name = "dummy",
)
EOF
  cat >$pkg/test.bzl <<'EOF' || fail "couldn't create file"
def _rule_impl(ctx):
  print("In rule: ")

some_rule = rule(
    implementation = _rule_impl,
)
EOF

  # Sanity check.
  bazel build //$pkg:dummy \
      &>"$TEST_log" || fail "bazel build failed";
  expect_log "In rule: " "Did not find rule print output"
  expect_not_log "$MARKER" "$sanity_fail_msg"

  bazel build //$pkg:dummy \
      --internal_starlark_flag_test_canary \
      &>"$TEST_log" || fail "bazel build failed";
  expect_log "In rule: $MARKER" \
      "Starlark flags are not propagating to rule implementation function evaluation"
}

# TODO(brandjon): Once we're no long dropping print() output in computed default
# functions, also test that we're propagating flags there. Alternatively, this
# could be tested by having conditional code that crashes while evaluating the
# Starlark function iff the flag is set.

function test_aspect() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<'EOF' || fail "couldn't create file"
load(":test.bzl", "some_rule")

some_rule(
    name = "dummy",
)
EOF
  cat >$pkg/test.bzl <<'EOF' || fail "couldn't create file"
def _rule_impl(ctx):
  pass

some_rule = rule(
    implementation = _rule_impl,
)

def _aspect_impl(target, ctx):
  print("In aspect: ")
  return []

some_aspect = aspect(
    implementation = _aspect_impl,
)
EOF

  # Sanity check.
  bazel build //$pkg:dummy --aspects $pkg/test.bzl%some_aspect \
      &>"$TEST_log" || fail "bazel build failed";
  expect_log "In aspect: " "Did not find aspect print output"
  expect_not_log "$MARKER" "$sanity_fail_msg"

  bazel build //$pkg:dummy --aspects $pkg/test.bzl%some_aspect \
      --internal_starlark_flag_test_canary \
      &>"$TEST_log" || fail "bazel build failed";
  expect_log "In aspect: $MARKER" \
      "Starlark flags are not propagating to aspect implementation function evaluation"
}


# Test a Starlark rule's interaction with the --nested_set_depth_limit flag.
function test_nested_set_depth() {
  mkdir -p test
  cat << EOF >> test/BUILD
load(":starlark.bzl", "test_rule")

test_rule(
    name = "test",
)
EOF

  cat << 'EOF' >> test/starlark.bzl
def _test_impl(ctx):
  x = depset([0])
  for i in range(1, 1000):
    x = depset([i], transitive = [x])
  x_list = x.to_list()
  return []

test_rule = rule(
    implementation=_test_impl,
)
EOF

  bazel build //test:test --nested_set_depth_limit=2000 &> $TEST_log \
      || fail "Build should have succeeded at depth limit 2000"

  bazel build //test:test --nested_set_depth_limit=500 &> $TEST_log \
      && fail "Build should have failed at depth limit 500"
  expect_log "depset depth 501 exceeds limit (500)"

  bazel build //test:test --nested_set_depth_limit=100 &> $TEST_log \
      && fail "Build should have failed at depth limit 100"
  expect_log "depset depth 101 exceeds limit (100)"

  bazel build //test:test --nested_set_depth_limit=3000 &> $TEST_log \
      || fail "Build should have succeeded at depth limit 3000"

  bazel build //test:test --nested_set_depth_limit=4000 &> $TEST_log \
      || fail "Build should have succeeded at depth limit 4000"
}


run_suite "starlark_flag_test"
