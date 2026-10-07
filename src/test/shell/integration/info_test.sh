#!/usr/bin/env bash
#
# Copyright 2020 The Bazel Authors. All rights reserved.
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
# An end-to-end test that Bazel info command reasonable output.

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


#### TESTS #############################################################

function test_info() {
  bazel info >$TEST_log \
    || fail "${PRODUCT_NAME} info failed"

  # Test some arbitrary keys.
  expect_log 'max-heap-size: [0-9]*MB'
  expect_log 'server_pid: [0-9]*'
  expect_log 'command_log: .*/command\.log'
  expect_log 'release: development version'

  # Make sure that hidden keys are not shown.
  expect_not_log 'used-heap-size-after-gc'
  expect_not_log 'starlark-semantics'
}

function test_server_pid() {
  bazel info server_pid >$TEST_log \
    || fail "${PRODUCT_NAME} info failed"
  expect_log '[0-9]*'
}

function test_used_heap_size_after_gc() {
  bazel info used-heap-size-after-gc >$TEST_log \
    || fail "${PRODUCT_NAME} info failed"
  expect_log '[0-9]*MB'
}

function test_starlark_semantics() {
  bazel info starlark-semantics >$TEST_log \
    || fail "${PRODUCT_NAME} info failed"
  expect_log 'StarlarkSemantics{.*}'
}

function test_multiple_keys() {
  bazel info release used-heap-size gc-count >$TEST_log \
    || fail "${PRODUCT_NAME} info failed"
  expect_log 'release: development version'
  expect_log 'used-heap-size: [0-9]*MB'
  expect_log 'gc-count: [0-9]*'
}

function test_multiple_keys_wrong_keys() {
  bazel info command_log foo used-heap-size-after-gc bar gc-count foo &>$TEST_log \
    && fail "expected ${PRODUCT_NAME} info to fail with unknown keys"

  # First test the valid keys.
  expect_log 'command_log: .*/command\.log'
  expect_log 'used-heap-size-after-gc: [0-9]*MB'
  expect_log 'gc-count: [0-9]*'

  # Then the error message.
  expect_log "ERROR: unknown key(s): 'foo', 'bar'"
}

# Regression test for https://github.com/bazelbuild/bazel/issues/24671
function test_invalid_flag_error() {
  # This type of loading error only happens with external dependencies.
  if [[ "$PRODUCT_NAME" != "bazel" ]]; then
    return 0
  fi
  bazel info --registry=foobarbaz &>$TEST_log \
    && fail "expected ${PRODUCT_NAME} to fail with an invalid registry"
  expect_not_log "crashed due to an internal error"
  expect_log "Invalid registry URL: foobarbaz"
}

function write_string_flag_bzl() {
  cat > "$1" <<'EOF'
string_flag = rule(
    implementation = lambda ctx: [],
    build_setting = config.string(flag = True),
)
EOF
}

# Regression test for https://github.com/bazelbuild/bazel/issues/25145
function test_label_flags_use_main_repo_mapping() {
  if [[ "$PRODUCT_NAME" != "bazel" ]]; then
    return 0
  fi
  local -r pkg=$FUNCNAME
  mkdir -p $pkg/dep
  cat > $(setup_module_dot_bazel "$pkg/MODULE.bazel") <<'EOF'
module(name = "my_module")
bazel_dep(name = "dep")
local_path_override(module_name = "dep", path = "dep")
my_repo = use_repo_rule("//:my_repo.bzl", "my_repo")
my_repo(name = "my_repo")
EOF
  cat > $pkg/my_repo.bzl <<'EOF'
def _my_repo_impl(rctx):
    rctx.file("BUILD", 'platform(name = "platform", visibility = ["//visibility:public"])')

my_repo = repository_rule(_my_repo_impl)
EOF
  cat > $pkg/BUILD <<'EOF'
platform(name = "platform")
EOF
  cat > $pkg/dep/MODULE.bazel <<'EOF'
module(name = "dep")
EOF
  write_string_flag_bzl $pkg/dep/flag.bzl
  cat > $pkg/dep/BUILD <<'EOF'
load(":flag.bzl", "string_flag")
platform(name = "platform", visibility = ["//visibility:public"])
string_flag(name = "flag", build_setting_default = "", visibility = ["//visibility:public"])
EOF
  cd $pkg

  bazel info --platforms=@my_repo//:platform bazel-bin &>$TEST_log \
    || fail "expected success with a platform in a use_repo_rule repo"
  bazel info --platforms=@my_module//:platform bazel-bin &>$TEST_log \
    || fail "expected success with a platform in the main repo"
  bazel info --platforms=@dep//:platform bazel-bin &>$TEST_log \
    || fail "expected success with a platform in a bazel_dep"
  # Regression test for https://github.com/bazelbuild/bazel/issues/29384
  bazel info --flag_alias=host_dep_flag=@dep//:flag bazel-bin &>$TEST_log \
    || fail "expected success with a flag alias for a flag in a bazel_dep"
}

function test_module_flag_alias_of_native_flag() {
  if [[ "$PRODUCT_NAME" != "bazel" ]]; then
    return 0
  fi
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $(setup_module_dot_bazel "$pkg/MODULE.bazel") <<'EOF'
flag_alias(name = "compilation_mode", starlark_flag = "//:compilation_mode")
EOF
  write_string_flag_bzl $pkg/flag.bzl
  cat > $pkg/BUILD <<'EOF'
load(":flag.bzl", "string_flag")
string_flag(name = "compilation_mode", build_setting_default = "")
EOF
  cd $pkg

  bazel info --compilation_mode=opt bazel-bin &>$TEST_log \
    || fail "${PRODUCT_NAME} info failed"
  # The alias sets the Starlark flag, so the native flag keeps its default.
  expect_log "-fastbuild/bin"
  expect_not_log "-opt/bin"
}

# Regression test for https://github.com/bazelbuild/bazel/issues/29176
function test_info_keeps_analysis_cache() {
  if [[ "$PRODUCT_NAME" != "bazel" ]]; then
    return 0
  fi
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  setup_module_dot_bazel "$pkg/MODULE.bazel" > /dev/null
  cat > $pkg/BUILD <<'EOF'
genrule(name = "gen", outs = ["out.txt"], cmd = "touch $@")
EOF
  cd $pkg
  local -r disk_cache="$TEST_TMPDIR/$FUNCNAME-disk-cache"

  # The second build ensures that the last one only reevaluates what info invalidated.
  for i in 1 2; do
    bazel build --disk_cache="$disk_cache" --build_event_json_file=bep.json //:gen &>$TEST_log \
      || fail "${PRODUCT_NAME} build failed"
  done
  bazel info --disk_cache="$disk_cache" bazel-bin &>$TEST_log \
    || fail "${PRODUCT_NAME} info failed"
  bazel build --disk_cache="$disk_cache" --build_event_json_file=bep.json //:gen &>$TEST_log \
    || fail "${PRODUCT_NAME} build failed"
  grep -o '"builtValues":\[[^]]*\]' bep.json > built_values.txt || true
  assert_not_contains '"CONFIGURED_TARGET"' built_values.txt
}

function test_keys_without_configuration_skip_module_resolution() {
  if [[ "$PRODUCT_NAME" != "bazel" ]]; then
    return 0
  fi
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $(setup_module_dot_bazel "$pkg/MODULE.bazel") <<'EOF'
fail("MODULE.bazel was evaluated")
EOF
  cd $pkg

  bazel info release output_base &>$TEST_log \
    || fail "expected ${PRODUCT_NAME} info to succeed without module resolution"
  bazel info bazel-bin &>$TEST_log \
    && fail "expected module resolution to fail"
  expect_log "MODULE.bazel was evaluated"
}

run_suite "Integration tests for ${PRODUCT_NAME} info."
