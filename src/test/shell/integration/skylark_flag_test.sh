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

# Tests that the Skylark interpreter is reading flags passed in on the command
# line, in several different evaluation contexts.
#
# The --internal_skylark_flag_test_canary flag is built into
# SkylarkSemanticsOptions specifically for this test suite.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# Text that will be appended to every print() output when the flag is enabled.
MARKER="<== skylark flag test ==>"

sanity_fail_msg="Marker string '$MARKER' was seen even though "
sanity_fail_msg+="--internal_skylark_flag_test_canary wasn't passed"


function test_build_file() {
  mkdir -p test
  cat > test/BUILD <<'EOF' || fail "couldn't create file"
print("In BUILD: ")

genrule(
    name = "dummy",
    cmd = "echo 'dummy' >$@",
    outs = ["dummy.txt"],
)
EOF

  # Sanity check.
  bazel build //test:dummy \
      &>"$TEST_log" || fail "bazel build failed";
  expect_log "In BUILD: " "Did not find BUILD print output"
  expect_not_log "$MARKER" "$sanity_fail_msg"

  bazel build //test:dummy \
      --internal_skylark_flag_test_canary \
      &>"$TEST_log" || fail "bazel build failed";
  expect_log "In BUILD: $MARKER" \
      "Skylark flags are not propagating to BUILD file evaluation"
}

function test_bzl_file_and_macro() {
  mkdir -p test
  cat > test/BUILD <<'EOF' || fail "couldn't create file"
load(":test.bzl", "macro")

macro()
EOF
  cat >test/test.bzl <<'EOF' || fail "couldn't create file"
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
  bazel build //test:dummy \
      &>"$TEST_log" || fail "bazel build failed";
  expect_log "In bzl: " "Did not find .bzl print output"
  expect_log "In macro: " "Did not find macro print output"
  expect_not_log "$MARKER" "$sanity_fail_msg"

  bazel build //test:dummy \
      --internal_skylark_flag_test_canary \
      &>"$TEST_log" || fail "bazel build failed";
  expect_log "In bzl: $MARKER" \
      "Skylark flags are not propagating to .bzl file evaluation"
  expect_log "In macro: $MARKER" \
      "Skylark flags are not propagating to macro evaluation"
}

function test_rule() {
  mkdir -p test
  cat > test/BUILD <<'EOF' || fail "couldn't create file"
load(":test.bzl", "some_rule")

some_rule(
    name = "dummy",
)
EOF
  cat >test/test.bzl <<'EOF' || fail "couldn't create file"
def _rule_impl(ctx):
  print("In rule: ")

some_rule = rule(
    implementation = _rule_impl,
)
EOF

  # Sanity check.
  bazel build //test:dummy \
      &>"$TEST_log" || fail "bazel build failed";
  expect_log "In rule: " "Did not find rule print output"
  expect_not_log "$MARKER" "$sanity_fail_msg"

  bazel build //test:dummy \
      --internal_skylark_flag_test_canary \
      &>"$TEST_log" || fail "bazel build failed";
  expect_log "In rule: $MARKER" \
      "Skylark flags are not propagating to rule implementation function evaluation"
}

# TODO(brandjon): Once we're no long dropping print() output in computed default
# functions, also test that we're propagating flags there. Alternatively, this
# could be tested by having conditional code that crashes while evaluating the
# Skylark function iff the flag is set.

function test_aspect() {
  mkdir -p test
  cat > test/BUILD <<'EOF' || fail "couldn't create file"
load(":test.bzl", "some_rule")

some_rule(
    name = "dummy",
)
EOF
  cat >test/test.bzl <<'EOF' || fail "couldn't create file"
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
  bazel build //test:dummy --aspects test/test.bzl%some_aspect \
      &>"$TEST_log" || fail "bazel build failed";
  expect_log "In aspect: " "Did not find aspect print output"
  expect_not_log "$MARKER" "$sanity_fail_msg"

  bazel build //test:dummy --aspects test/test.bzl%some_aspect \
      --internal_skylark_flag_test_canary \
      &>"$TEST_log" || fail "bazel build failed";
  expect_log "In aspect: $MARKER" \
      "Skylark flags are not propagating to aspect implementation function evaluation"
}


run_suite "skylark_flag_test"
