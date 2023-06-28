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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  create_new_workspace

  mkdir package

  # Create shared platform definitions
  mkdir plat
  cat > plat/BUILD <<EOF
platform(
    name = 'platform1',
    constraint_values = [])

platform(
    name = 'platform2',
    constraint_values = [])
EOF

  # Create shared report rule for printing flag and platform info.
  mkdir report
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

  bazel build --cpu=arm64 package:report &> $TEST_log \
      || fail "Build failed unexpectedly"
  expect_log "platform: @//plat:platform1"
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

  bazel build --platforms=//plat:platform1 package:report &> $TEST_log \
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

  bazel build --cpu=arm64 --platform_mappings=custom/platform_mappings \
      package:report &> $TEST_log || fail "Build failed unexpectedly"
  expect_log "platform: @//plat:platform1"
}

function test_custom_platform_mapping_location_after_exec_transition() {
  mkdir custom
  cat > custom/platform_mappings <<EOF
platforms:
  //plat:platform1
    --copt=foo
EOF

  cat > package/BUILD <<EOF
load("//report:report.bzl", "report_flags")
genrule(
    name = "genrule1",
    outs = ["genrule1.out"],
    cmd = "echo hello > \$@",
    tools = [
      ":genrule2",
    ],
)
genrule(
    name = "genrule2",
    outs = ["genrule2.out"],
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
      package:genrule1 &> $TEST_log || fail "Build failed unexpectedly"
  expect_log "platform: @//plat:platform1"
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
      "_allowlist_function_transition": attr.label(
          default = "@bazel_tools//tools/allowlists/function_transition_allowlist"),
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

  bazel build --cpu=k8 package:custom &> $TEST_log \
      || fail "Build failed unexpectedly"
  expect_not_log "platform: @//plat:platform1"
  expect_log "platform: @//plat:platform2"
}

run_suite "platform mapping test"

