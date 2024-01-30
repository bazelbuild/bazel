#!/bin/bash
#
# Copyright 2023 The Bazel Authors. All rights reserved.
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
# An end-to-end test for the 'focus' command.

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
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

function set_up() {
  # Ensure we always start with a fresh server so that the following
  # env vars are picked up on startup. This could also be `bazel shutdown`,
  # but clean is useful for stateless tests.
  bazel clean --expunge

  # The focus command is currently implemented for InMemoryGraphImpl,
  # not SerializationCheckingGraph. This env var disables
  # SerializationCheckingGraph from being used as the evaluator.
  export DONT_SANITY_CHECK_SERIALIZATION=1
}

function test_must_be_used_after_a_build() {
  local -r pkg=${FUNCNAME[0]}
  mkdir ${pkg}|| fail "cannot mkdir ${pkg}"
  mkdir -p ${pkg}
  echo "input" > ${pkg}/in.txt
  cat > ${pkg}/BUILD <<EOF
genrule(
  name = "g",
  srcs = ["in.txt"],
  outs = ["out.txt"],
  cmd = "cp \$< \$@",
)
EOF

  bazel focus --experimental_working_set=${pkg}/in.txt >$TEST_log 2>&1 && "unexpected success"
  expect_log "Unable to focus without roots. Run a build first."
}

function test_correctly_rebuilds_after_using_focus() {
  local -r pkg=${FUNCNAME[0]}
  mkdir ${pkg}|| fail "cannot mkdir ${pkg}"
  mkdir -p ${pkg}
  echo "input" > ${pkg}/in.txt
  cat > ${pkg}/BUILD <<EOF
genrule(
  name = "g",
  srcs = ["in.txt"],
  outs = ["out.txt"],
  cmd = "cp \$< \$@",
)
EOF

  out=$(bazel info "${PRODUCT_NAME}-genfiles")/${pkg}/out.txt

  bazel build //${pkg}:g
  assert_contains "input" $out

  echo "a change" >> ${pkg}/in.txt
  bazel focus --experimental_working_set=${pkg}/in.txt

  bazel build //${pkg}:g
  assert_contains "a change" $out
}

function test_focus_command_prints_info_about_graph() {
  local -r pkg=${FUNCNAME[0]}
  mkdir ${pkg}|| fail "cannot mkdir ${pkg}"
  mkdir -p ${pkg}
  echo "input" > ${pkg}/in.txt
  cat > ${pkg}/BUILD <<EOF
genrule(
  name = "g",
  srcs = ["in.txt"],
  outs = ["out.txt"],
  cmd = "cp \$< \$@",
)
EOF

  out=$(bazel info "${PRODUCT_NAME}-genfiles")/${pkg}/out.txt
  bazel build //${pkg}:g

  bazel focus \
    --dump_used_heap_size_after_gc \
    --experimental_working_set=${pkg}/in.txt >$TEST_log 2>&1

  expect_log "Focusing on .\+ roots, .\+ leafs"
  expect_log "Nodes in reverse transitive closure from leafs: .\+"
  expect_log "Nodes in direct deps of reverse transitive closure: .\+"
  expect_log "Rdep edges: .\+ -> .\+"
  expect_log "Heap: .\+MB -> .\+MB (.\+% reduction)"
  expect_log "Node count: .\+ -> .\+ (.\+% reduction)"
}

function test_focus_command_dump_keys_prints_more_info_about_graph() {
  local -r pkg=${FUNCNAME[0]}
  mkdir ${pkg}|| fail "cannot mkdir ${pkg}"
  mkdir -p ${pkg}
  echo "input" > ${pkg}/in.txt
  cat > ${pkg}/BUILD <<EOF
genrule(
  name = "g",
  srcs = ["in.txt"],
  outs = ["out.txt"],
  cmd = "cp \$< \$@",
)
EOF

  out=$(bazel info "${PRODUCT_NAME}-genfiles")/${pkg}/out.txt
  bazel build //${pkg}:g  &> $TEST_log 2>&1

  bazel focus --dump_keys --experimental_working_set=${pkg}/in.txt >$TEST_log 2>&1

  expect_log "Focusing on .\+ roots, .\+ leafs"

  # additional info
  expect_log "Rdeps kept:"
  expect_log "BUILD_DRIVER:"

  expect_log "Deps kept:"
  expect_log "BUILD_CONFIGURATION:"

  expect_log "Summary of kept keys:"
  expect_log "BUILD_DRIVER"
}

function test_builds_new_target_after_using_focus() {
  local -r pkg=${FUNCNAME[0]}
  mkdir ${pkg}|| fail "cannot mkdir ${pkg}"
  mkdir -p ${pkg}
  echo "input" > ${pkg}/in.txt
  cat > ${pkg}/BUILD <<EOF
genrule(
  name = "g",
  srcs = ["in.txt"],
  outs = ["g.txt"],
  cmd = "cp \$< \$@",
)
genrule(
  name = "g2",
  srcs = ["in.txt"],
  outs = ["g2.txt"],
  cmd = "cp \$< \$@",
)
genrule(
  name = "g3",
  outs = ["g3.txt"],
  cmd = "touch \$@",
)
EOF

  outdir=$(bazel info "${PRODUCT_NAME}-genfiles")/${pkg}

  bazel build //${pkg}:g
  echo "a change" >> ${pkg}/in.txt

  bazel focus --experimental_working_set=${pkg}/in.txt
  bazel build //${pkg}:g
  bazel build //${pkg}:g2 || fail "cannot build //${pkg}:g2"
  bazel build //${pkg}:g3 || fail "cannot build //${pkg}:g3"
}

function test_focus_emits_profile_data() {
  local -r pkg=${FUNCNAME[0]}
  mkdir ${pkg}|| fail "cannot mkdir ${pkg}"
  mkdir -p ${pkg}
  echo "input" > ${pkg}/in.txt
  cat > ${pkg}/BUILD <<EOF
genrule(
  name = "g",
  srcs = ["in.txt"],
  outs = ["out.txt"],
  cmd = "cp \$< \$@",
)
EOF

  bazel build //${pkg}:g
  bazel focus --experimental_working_set=${pkg}/in.txt \
    --profile=/tmp/profile.log &> "$TEST_log" || fail "Expected success"
  grep '"ph":"X"' /tmp/profile.log > "$TEST_log" \
    || fail "Missing profile file."

  expect_log '"SkyframeFocuser"'
  expect_log '"focus.mark"'
  expect_log '"focus.sweep_nodes"'
  expect_log '"focus.sweep_edges"'
}

run_suite "Tests for the focus command"
