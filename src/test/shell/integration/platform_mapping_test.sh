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
  cat > report/flag.bzl <<EOF
BuildSettingInfo = provider(fields = ["value"])

def _string_flag_impl(ctx):
    return [BuildSettingInfo(value = ctx.build_setting_value)]

string_flag = rule(
    implementation = _string_flag_impl,
    build_setting = config.string(flag = True)
)
EOF

  cat > report/BUILD <<EOF
load(":flag.bzl", "string_flag")

string_flag(
    name = "mapping_flag",
    build_setting_default = "from_default",
)
EOF
  cat > report/report.bzl <<EOF
load(":flag.bzl", "BuildSettingInfo")

def _report_impl(ctx):
  mapping_flag = ctx.attr._mapping_flag[BuildSettingInfo].value
  print('mapping_flag: %s' % mapping_flag)
  print('platform: %s' % ctx.fragments.platform.platform)

report_flags = rule(
    implementation = _report_impl,
    attrs = {
        "_mapping_flag": attr.label(default = "//report:mapping_flag"),
    },
    fragments = ["platform"]
)
EOF
}

#### TESTS #############################################################

function test_top_level_flags_to_platform_mapping() {
  cat > platform_mappings <<EOF
flags:
  --//report:mapping_flag=foo
    //plat:platform1
EOF

  cat > package/BUILD <<EOF
load("//report:report.bzl", "report_flags")
report_flags(name = "report")
EOF

  bazel build \
    --platform_mappings=platform_mappings \
    --//report:mapping_flag=foo \
    //package:report &> $TEST_log \
      || fail "Build failed unexpectedly"
  expect_log "platform: .*//plat:platform1"
}

function test_top_level_platform_to_flags_mapping() {
   cat > platform_mappings <<EOF
platforms:
  //plat:platform1
    --//report:mapping_flag=from_mapping
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
  expect_log "mapping_flag: from_mapping"
}

function test_custom_platform_mapping_location() {
  mkdir custom
  cat > custom/platform_mappings <<EOF
flags:
  --//report:mapping_flag=foo
    //plat:platform1
EOF

  cat > package/BUILD <<EOF
load("//report:report.bzl", "report_flags")
report_flags(name = "report")
EOF

  bazel build \
    --//report:mapping_flag=foo \
    --platform_mappings=custom/platform_mappings \
    package:report &> $TEST_log || fail "Build failed unexpectedly"
  expect_log "platform: .*//plat:platform1"
}

function test_custom_platform_mapping_location_after_exec_transition() {
  mkdir -p custom
  cat > custom/platform_mappings <<EOF
platforms:
  //plat:platform1
    --//report:mapping_flag=from_mapping
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
  expect_log "mapping_flag: from_mapping"
}

function test_transition_platform_mapping() {
  cat > platform_mappings <<EOF
flags:
  --//report:mapping_flag=foo
    //plat:platform1
  --//report:mapping_flag=bar
    //plat:platform2
EOF

  cat > package/rule.bzl <<EOF
def _my_transition_impl(settings, attrs):
  return {
    "//report:mapping_flag": "bar",
    # Platforms *must* be wiped for transitions to correctly participate in
    # platform mapping.
    "//command_line_option:platforms": [],
  }

my_transition = transition(
  implementation = _my_transition_impl,
  inputs = [],
  outputs = [
      "//report:mapping_flag",
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
    --//report:mapping_flag=foo \
    package:custom &> $TEST_log \
      || fail "Build failed unexpectedly"
  expect_not_log "platform: .*//plat:platform1"
  expect_log "platform: .*//plat:platform2"
}

function test_mapping_overrides_command_line() {
   cat > platform_mappings <<EOF
platforms:
  //plat:platform1
    --//report:mapping_flag=from_mapping
EOF

  cat > package/BUILD <<EOF
load("//report:report.bzl", "report_flags")
report_flags(name = "report")
EOF

  bazel build \
    --platform_mappings=platform_mappings \
    --platforms=//plat:platform1 \
    --//report:mapping_flag=from_cli \
    package:report &> $TEST_log \
      || fail "Build failed unexpectedly"
  expect_log "mapping_flag: from_mapping"
}

function test_repeatable_flag_doesnt_accumulate() {
  # Use a different flag for testing and reporting.
  mkdir -p repeatable
  cat > repeatable/flag.bzl <<EOF
FlagValue = provider(fields = ["value"])
def _impl(ctx):
    values = ctx.build_setting_value
    return [
        FlagValue(value = values),
    ]

repeatable_flag = rule(
    implementation = _impl,
    build_setting = config.string_list(flag = True, repeatable = True),
)
EOF
  cat > repeatable/BUILD <<EOF
load(":flag.bzl", "repeatable_flag")

package(default_visibility = ["//visibility:public"])

repeatable_flag(
    name = "repeatable_flag",
    build_setting_default = ['default'],
)
EOF
  cat > report/report.bzl <<EOF
load("//repeatable:flag.bzl", "FlagValue")

def _report_impl(ctx):
  flag_values = ctx.attr._flag[FlagValue].value
  print('repeatable_flag: %s' % flag_values)
  print('platform: %s' % ctx.fragments.platform.platform)

report_flags = rule(
    implementation = _report_impl,
    attrs = {
        "_flag": attr.label(default = "//repeatable:repeatable_flag"),
    },
    fragments = ["platform"]
)
EOF

   # Set up a platform mapping.
   cat > platform_mappings <<EOF
platforms:
  //plat:platform1
    --//repeatable:repeatable_flag=from_mapping
EOF

  cat > package/BUILD <<EOF
load("//report:report.bzl", "report_flags")
report_flags(name = "report")
EOF

  bazel build \
    --platform_mappings=platform_mappings \
    --platforms=//plat:platform1 \
    --//repeatable:repeatable_flag=from_cli \
    package:report &> $TEST_log \
      || fail "Build failed unexpectedly"
  expect_log 'repeatable_flag: \["from_mapping"\]'
}

run_suite "platform mapping test"
