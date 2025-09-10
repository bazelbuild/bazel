#!/usr/bin/env bash
#
# Copyright 2025 The Bazel Authors. All rights reserved.
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
  add_rules_shell "MODULE.bazel"
  mkdir -p a b c

  cat > a/BUILD <<'EOF'
filegroup(
    name = "files",
    srcs = ["file.txt"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "rule_a",
    srcs = [":files"],
    outs = ["output_a.txt"],
    cmd = "cp $< $@",
)
EOF

  cat > a/file.txt <<'EOF'
content a
EOF

  cat > b/BUILD <<'EOF'
genrule(
    name = "rule_b",
    srcs = ["//a:files"],
    outs = ["output_b.txt"],
    cmd = "cp $< $@",
)
EOF

  cat > c/BUILD <<'EOF'
load("@rules_shell//shell:sh_test.bzl", "sh_test")

genrule(name = "x", outs = ["x.out"], cmd = "echo true > $@", executable = True)
sh_test(name = "test", srcs = ["x.out"])
EOF
}

function test_build_with_query_deps() {
  bazel build --query="//a:rule_a" >& "$TEST_log" || fail "Build with query failed"
  expect_log "//a:rule_a"
  [ -f "bazel-bin/a/output_a.txt" ] || fail "Output a/output_a.txt was not built"
}

function test_build_with_query_multiple() {
  bazel build --query="//a:rule_a + //b:rule_b" >& "$TEST_log" || fail "Build with query failed"
  [ -f "bazel-bin/a/output_a.txt" ] || fail "Output a/output_a.txt was not built"
  [ -f "bazel-bin/b/output_b.txt" ] || fail "Output b/output_b.txt was not built"
}

function test_build_with_query_pattern() {
  bazel build --query="//a:*" >& "$TEST_log" || fail "Build with pattern query failed"
  [ -f "bazel-bin/a/output_a.txt" ] || fail "Output a/output_a.txt was not built"
}

function test_build_with_query_file() {
  echo '//b:rule_b' > query.txt

  bazel build --query_file=query.txt >& "$TEST_log" || fail "Build with query_file failed"
  expect_log "//b:rule_b"
  [ -f "bazel-bin/b/output_b.txt" ] || fail "Output b/output_b.txt was not built"
}

function test_build_with_empty_query_result() {
  bazel build --query="//nonexistent:target" >& "$TEST_log" && fail "Should have failed with nonexistent target"
  expect_log "Error executing query"
}

function test_build_and_test_with_query() {
  bazel test --query="tests(//c/...)" >& "$TEST_log" || fail "Should have succeeded"
  expect_log "//c:test"
  [ -f "bazel-bin/c/x.out" ] || fail "Output c/x.out was not built"
}

run_suite "build --query tests"
