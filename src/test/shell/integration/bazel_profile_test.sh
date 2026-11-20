#!/usr/bin/env bash
#
# Copyright 2009 The Bazel Authors. All rights reserved.
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
# bazel_profile_test.sh: integration tests for Bazel --profile and
# --memory_profile.
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

function set_up() {
  cd "${WORKSPACE_DIR}"
  add_rules_cc "MODULE.bazel"
  mkdir -p foo
  cat >foo/BUILD <<'EOF'
genrule(
  name = "foo",
  outs = ["out"],
  srcs = ["in"],
  cmd = "cp $< $@",
)
EOF
  echo "echo TEST" >foo/in
}

#### TESTS #############################################################

function test_profiling() {
  bazel clean
  bazel build \
    --profile="$TEST_TMPDIR/profile.gz" foo:all >/dev/null 2>"$TEST_log" ||
      { fail "Expected successful build"; return 1; }
}

function test_memory_profiling() {
  # '"bazel build --memory_profile" is parseable by "awk"'
  local -r mp_dat="$TEST_TMPDIR/mp.dat"
  bazel build \
    --noexperimental_merged_skyframe_analysis_execution \
    --memory_profile="$mp_dat" foo:all >/dev/null 2>"$TEST_log" ||
      { fail "Expected successful build"; return 1; }
  cat "$mp_dat" >> "$TEST_log"
  assert_equals "8" "$(awk -F: '/Evaluate target patterns/ { print $4 }' "$mp_dat" \
    | tr -d '\r' | grep -cE '^-?[0-9]+$')" || \
      { fail "Expected successful parsing"; return 1; }
  assert_equals "1" "$(awk -F: '/Evaluate target patterns:heap:used/ { print $4 }' \
    "$mp_dat" | tr -d '\r' | grep -cE '^[0-9]+$')" || \
      { fail "Expected successful parsing"; return 1; }
  assert_equals "8" "$(awk -F: '/Build artifacts/ { print $4 }' "$mp_dat" \
    | tr -d '\r' | grep -cE '^-?[0-9]+$')" || \
      { fail "Expected successful parsing"; return 1; }
  assert_equals "1" "$(awk -F: '/Build artifacts:non-heap:commited/ \
    { print $4 }' "$mp_dat" | tr -d '\r' | grep -cE '^[0-9]+$')" || \
      { fail "Expected successful parsing"; return 1; }
}

function test_memory_profiling_skymeld() {
  # '"bazel build --memory_profile" is parseable by "awk"'
  local -r mp_dat="$TEST_TMPDIR/mp.dat"
  bazel build \
    --experimental_merged_skyframe_analysis_execution \
    --memory_profile="$mp_dat" foo:all >/dev/null 2>"$TEST_log" ||
      { fail "Expected successful build"; return 1; }
  cat "$mp_dat" >> "$TEST_log"
  assert_equals "8" "$(awk -F: '/Evaluate target patterns/ { print $4 }' "$mp_dat" \
    | tr -d '\r' | grep -cE '^-?[0-9]+$')" || \
      { fail "Expected successful parsing"; return 1; }
  assert_equals "1" "$(awk -F: '/Evaluate target patterns:heap:used/ { print $4 }' \
    "$mp_dat" | tr -d '\r' | grep -cE '^[0-9]+$')" || \
      { fail "Expected successful parsing"; return 1; }
  assert_equals "8" "$(awk -F: '/Load, analyze dependencies and build artifacts/ { print $4 }' "$mp_dat" \
    | tr -d '\r' | grep -cE '^-?[0-9]+$')" || \
      { fail "Expected successful parsing"; return 1; }
  assert_equals "1" "$(awk -F: '/Load, analyze dependencies and build artifacts:non-heap:commited/ \
    { print $4 }' "$mp_dat" | tr -d '\r' | grep -cE '^[0-9]+$')" || \
      { fail "Expected successful parsing"; return 1; }
}

# Regression test that checks that if an action fails to execute we do not crash and we record the
# profiling information correctly.
function test_profile_on_action_error() {
  mkdir -p action_error || die "Cannot create directory"
  cat > "action_error/BUILD" <<EOF
genrule(name = "wrong", outs = ["fail.out"], cmd = "cat I_do_not_exist > \$(OUTS)")
EOF
  bazel build action_error:wrong >&"$TEST_log" \
    --profile="$TEST_TMPDIR/profile.out" && fail "expected to fail"
  exit_code=$?
  assert_equals "1" $exit_code
  expect_not_log "java.lang.IllegalStateException"
  cat "$TEST_TMPDIR/profile.out" > "$TEST_log" || fail "Expected success"
  expect_log "Executing genrule //action_error:wrong"
}

# Regression test for b/34682860: "java.lang.IllegalStateException: Inconsistent
# Profiler.completeTask() call for the ACTION task."
function test_profile_on_action_error_with_cc_library() {
  mkdir -p action_error || die "Cannot create directory"
  cat > "action_error/BUILD" <<EOF
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(name = 'bad_cc', srcs = ['foo.cc'])
EOF
  # Note that we do not create foo.cc. Missing this input file seems to reliably trigger the
  # problem before the fix.
  bazel build action_error:bad_cc >& "$TEST_log" \
    --profile="$TEST_TMPDIR/profile.out" && fail "expected to fail"
  exit_code=$?
  assert_equals "1" $exit_code
  expect_not_log "java.lang.IllegalStateException"
}

function test_basic_profile_generation() {
  output_base=$(bazel info output_base)

  rm -f "$output_base/command.profile.gz"
  bazel build &> "$TEST_log" \
    || fail "expected to succeed"
  [[ -f "$output_base/command.profile.gz" ]] || fail "expected to generate profile"

  rm -f "$output_base/command.profile.gz"
  rm -f "$output_base/command.profile"
  bazel build \
    --noexperimental_generate_json_trace_profile \
    &> "$TEST_log" || fail "expected to succeed"
  [[ -f "$output_base/command.profile.gz" ]] && fail "expected not to generate profile"
  [[ -f "$output_base/command.profile" ]] && fail "expected not to generate profile"

  rm -f "$output_base/command.profile.gz"
  rm -f "$output_base/command.profile"
  bazel info \
    &> "$TEST_log" || fail "expected to succeed"
  expect_not_log "command.profile"
  [[ -f "$output_base/command.profile.gz" ]] && fail "expected not to generate profile"
  [[ -f "$output_base/command.profile" ]] && fail "expected not to generate profile"

  rm -f "$output_base/command.profile.gz"
  rm -f "$output_base/command.profile"
  bazel info \
    --experimental_generate_json_trace_profile \
    &> "$TEST_log" || fail "expected to succeed"
  [[ -f "$output_base/command.profile.gz" ]] || fail "expected to generate profile"
  [[ -f "$output_base/command.profile" ]] && fail "expected not to generate profile"

  rm -f "$output_base/command.profile.gz"
  rm -f "$output_base/command.profile"
  bazel clean \
    &> "$TEST_log" || fail "expected to succeed"
  expect_not_log "command.profile"
  [[ -f "$output_base/command.profile.gz" ]] && fail "expected not to generate profile"
  [[ -f "$output_base/command.profile" ]] && fail "expected not to generate profile"
  true  # reset $?, otherwise the test would be considered failed
}

function test_primary_output() {
  bazel build \
    --profile="$TEST_TMPDIR/foo.json" \
    --noslim_profile \
    --experimental_profile_include_primary_output &> "$TEST_log" \
    || fail "expected to succeed"
  grep '"out":' "$TEST_TMPDIR/foo.json" || \
    fail "expected primary output in profile"
}

function test_not_profile_command_log() {
  bazel build \
    --profile="$TEST_TMPDIR/foo.json" \
    --record_full_profiler_data &> "$TEST_log" \
    || fail "expected to succeed"
  cp "$TEST_TMPDIR/foo.json" "$TEST_log"
  expect_log "VFS delete.*/command.log"
  expect_not_log "VFS open.*/command.log"
  expect_not_log "VFS write.*/command.log"
}

function test_slim_profile_size_limit() {
  bazel clean

  # Generate many targets to force more events and flushes
  mkdir -p many
  echo > many/BUILD
  for i in {1..200}; do
    cat >> many/BUILD <<EOF
genrule(
  name = "g$i",
  outs = ["o$i"],
  cmd = "echo $i > \$@",
)
EOF
  done

  bazel build \
    --profile="$TEST_TMPDIR/profile.gz" \
    --slim_profile=100 \
    many:all >/dev/null 2>"$TEST_log" ||
      { fail "Expected successful build"; return 1; }

  # Decompress and check for merged events
  zcat "$TEST_TMPDIR/profile.gz" > "$TEST_TMPDIR/profile.json" || fail "Failed to decompress profile"
  if ! grep -q "various events" "$TEST_TMPDIR/profile.json"; then
    cat "$TEST_TMPDIR/profile.json"
    fail "Expected merged events in slim profile"
  fi
}

run_suite "Tests of Bazel profile."
