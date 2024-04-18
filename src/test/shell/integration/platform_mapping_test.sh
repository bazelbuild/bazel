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
# Test the providers and rules related to toolchains.
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

#### SETUP #############################################################

add_to_bazelrc "build --genrule_strategy=local"
add_to_bazelrc "test --test_strategy=standalone"

function set_up() {
  create_new_workspace

  mkdir -p package

  # Create shared platform definitions
  mkdir -p plat
  cat > plat/BUILD <<EOF
platform(
    name = 'platform1',
    constraint_values = [])

platform(
    name = 'platform2',
    constraint_values = [])
EOF

  # Create shared report rule for printing flag and platform info.
  mkdir -p report
  touch report/BUILD
  cat > report/report.bzl <<EOF
def _report_impl(ctx):
  print('copts: %s' % ctx.fragments.cpp.copts)
  print('platform: %s' % ctx.fragments.platform.platform)

report_flags = rule(
    implementation = _report_impl,
    attrs = {},
    fragments = ["cpp", "platform"]
)
EOF
}

#### TESTS #############################################################

function test_top_level_flags_to_platform_mapping() {
  cat > platform_mappings <<EOF
flags:
  --cpu=arm64
    //plat:platform1
EOF

  cat > package/BUILD <<EOF
load("//report:report.bzl", "report_flags")
report_flags(name = "report")
EOF

  bazel build \
    --platform_mappings=platform_mappings \
    --cpu=arm64 \
    package:report &> $TEST_log \
      || fail "Build failed unexpectedly"
  expect_log "platform: .*//plat:platform1"
}

function test_top_level_platform_to_flags_mapping() {
   cat > platform_mappings <<EOF
platforms:
  //plat:platform1
    --copt=foo
EOF

  cat > package/BUILD <<EOF
load("//report:report.bzl", "report_flags")
report_flags(name = "report")
EOF

  bazel build \
    --platform_mappings=platform_mappings \
    --platforms=//plat:platform1 \
    package:report &> $TEST_log \
      || fail "Build failed unexpectedly"
  expect_log "copts: \[\"foo\"\]"
}

function test_custom_platform_mapping_location() {
  mkdir custom
  cat > custom/platform_mappings <<EOF
flags:
  --cpu=arm64
    //plat:platform1
EOF

  cat > package/BUILD <<EOF
load("//report:report.bzl", "report_flags")
report_flags(name = "report")
EOF

  bazel build \
    --cpu=arm64 \
    --platform_mappings=custom/platform_mappings \
    package:report &> $TEST_log || fail "Build failed unexpectedly"
  expect_log "platform: .*//plat:platform1"
}

function test_custom_platform_mapping_location_after_exec_transition() {
  mkdir -p custom
  cat > custom/platform_mappings <<EOF
platforms:
  //plat:platform1
    --copt=foo
EOF

  cat > package/BUILD <<EOF
load("//report:report.bzl", "report_flags")
genrule(
    name = "genrule",
    outs = ["genrule.out"],
    cmd = "echo hello > \$@",
    tools = [
      ":report",
    ],
)
report_flags(name = "report")
EOF

  bazel build \
      --platform_mappings=custom/platform_mappings \
      --extra_execution_platforms=//plat:platform1 \
      package:genrule &> $TEST_log || fail "Build failed unexpectedly"
  expect_log "platform: .*//plat:platform1"
  expect_log "copts: \[\"foo\"\]"
}

function test_transition_platform_mapping() {
  cat > platform_mappings <<EOF
flags:
  --cpu=k8
    //plat:platform1
  --cpu=arm64
    //plat:platform2
EOF

  cat > package/rule.bzl <<EOF
def _my_transition_impl(settings, attrs):
  return {
    "//command_line_option:cpu": "arm64",
    "//command_line_option:copt": ["foo"],
    # Platforms *must* be wiped for transitions to correctly participate in
    # platform mapping.
    "//command_line_option:platforms": [],
  }


my_transition = transition(
  implementation = _my_transition_impl,
  inputs = [],
  outputs = [
      "//command_line_option:cpu",
      "//command_line_option:copt",
      "//command_line_option:platforms",
  ],
)


def _my_rule_impl(ctx):
  return []


my_rule = rule(
  implementation = _my_rule_impl,
  attrs = {
      "deps": attr.label_list(cfg = my_transition),
  }
)
EOF

  cat > package/BUILD <<EOF
load("//report:report.bzl", "report_flags")
load("//package:rule.bzl", "my_rule")

my_rule(
  name = "custom",
  deps = [ ":report" ]
)

report_flags(name = "report")
EOF

  bazel build \
    --platform_mappings=platform_mappings \
    --cpu=k8 \
    package:custom &> $TEST_log \
      || fail "Build failed unexpectedly"
  expect_not_log "platform: .*//plat:platform1"
  expect_log "platform: .*//plat:platform2"
}

function test_mapping_overrides_command_line() {
   cat > platform_mappings <<EOF
platforms:
  //plat:platform1
    --copt=foo
EOF

  cat > package/BUILD <<EOF
load("//report:report.bzl", "report_flags")
report_flags(name = "report")
EOF

  bazel build \
    --platform_mappings=platform_mappings \
    --platforms=//plat:platform1 \
    --copt=bar \
    package:report &> $TEST_log \
      || fail "Build failed unexpectedly"
  expect_log "copts: \[\"foo\"\]"
}

run_suite "platform mapping test"

