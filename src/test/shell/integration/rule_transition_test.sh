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
# Test rule transition can inspect configurable attribute.

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

  # Define common starlark flags.
  mkdir -p settings
  touch settings/BUILD
  cat > settings/flag.bzl <<EOF
BuildSettingInfo = provider(fields = ["value"])

def _flag_impl(ctx):
    return [BuildSettingInfo(value = ctx.build_setting_value)]

bool_flag = rule(
    implementation = _flag_impl,
    build_setting = config.bool(flag = True)
)

string_list_flag = rule(
    implementation = _flag_impl,
    build_setting = config.string_list(),
)
EOF
}

function create_transitions() {
  local pkg="${1}"
  mkdir -p "${pkg}"
  cat > "${pkg}/def.bzl" <<EOF
load("//settings:flag.bzl", "BuildSettingInfo")

example_package = "${pkg}"

def _transition_impl(settings, attr):
    if getattr(attr, "apply_transition") and settings["//%s:transition_input_flag" % example_package]:
        return {"//%s:transition_output_flag" % example_package: True}
    return {"//%s:transition_output_flag" % example_package: False}

example_transition = transition(
    implementation = _transition_impl,
    inputs = ["//%s:transition_input_flag" % example_package],
    outputs = ["//%s:transition_output_flag" % example_package],
)

def _rule_impl(ctx):
    print("Flag value for %s: %s" % (
        ctx.label.name,
        ctx.attr._transition_output_flag[BuildSettingInfo].value,
    ))

transition_attached = rule(
    implementation = _rule_impl,
    cfg = example_transition,
    attrs = {
        "apply_transition": attr.bool(default = False),
        "deps": attr.label_list(),
        "_transition_output_flag": attr.label(default = "//%s:transition_output_flag" % example_package),
    },
)

transition_not_attached = rule(
    implementation = _rule_impl,
    attrs = {
        "deps": attr.label_list(),
        "_transition_output_flag": attr.label(default = "//%s:transition_output_flag" % example_package),
    },
)
EOF
}

function create_rules_with_incoming_transition_and_selects() {
  local pkg="${1}"
  mkdir -p "${pkg}"
  cat > "${pkg}/BUILD" <<EOF
load(
    "//${pkg}:def.bzl",
    "transition_attached",
    "transition_not_attached",
)
load("//settings:flag.bzl", "bool_flag")

bool_flag(
    name = "transition_input_flag",
    build_setting_default = True,
)

bool_flag(
    name = "transition_output_flag",
    build_setting_default = False,
)

config_setting(
    name = "select_setting",
    flag_values = {":transition_input_flag": "True"},
)

# All should print "False" if
# "--no//${pkg}:transition_input_flag" is
# specified on the command line

# bazel build :top_level will print the results for all of the targets below

transition_attached(
    name = "top_level",
    apply_transition = select({
        ":select_setting": True,
        "//conditions:default": False,
    }),
    deps = [
        ":transition_attached_dep",
        ":transition_not_attached_dep",
    ],
)

# Should print "False"
transition_attached(
    apply_transition = False,
    name = "transition_attached_dep",
    deps = [
        ":transition_not_attached_dep_of_dep",
    ],
)

# Should print "True" when building top_level, "False" otherwise
transition_not_attached(
    name = "transition_not_attached_dep",
)

# Should print "False"
transition_not_attached(
    name = "transition_not_attached_dep_of_dep",
)
EOF
}

function test_rule_transition_can_inspect_configure_attributes(){
  local -r pkg="${FUNCNAME[0]}"
  create_transitions "${pkg}"
  create_rules_with_incoming_transition_and_selects "${pkg}"

  bazel build "//${pkg}:top_level" &> $TEST_log || fail "Build failed"

  expect_log 'Flag value for transition_not_attached_dep: True'
  expect_log 'Flag value for transition_not_attached_dep_of_dep: False'
  expect_log 'Flag value for transition_attached_dep: False'
  expect_log 'Flag value for top_level: True'
}

function test_rule_transition_can_inspect_configure_attributes_with_flag(){
  local -r pkg="${FUNCNAME[0]}"

  create_transitions "${pkg}"
  create_rules_with_incoming_transition_and_selects "${pkg}"

  bazel build --no//${pkg}:transition_input_flag "//${pkg}:top_level" &> $TEST_log || fail "Build failed"

  expect_log 'Flag value for transition_not_attached_dep: False'
  expect_log 'Flag value for transition_not_attached_dep_of_dep: False'
  expect_log 'Flag value for transition_attached_dep: False'
  expect_log 'Flag value for top_level: False'
}

function test_rule_transition_can_not_inspect_configure_attribute() {
  local -r pkg="${FUNCNAME[0]}"

  # create transition definition
  mkdir -p "${pkg}"
  cat > "${pkg}/def.bzl" <<EOF
load("//settings:flag.bzl", "BuildSettingInfo")

example_package = "${pkg}"

def _transition_impl(settings, attr):
    if getattr(attr, "apply_transition") and settings["//%s:transition_input_flag" % example_package]:
        return {"//%s:transition_output_flag" % example_package: True}
    return {
      "//%s:transition_output_flag" % example_package: False,
      "//%s:transition_input_flag" % example_package: False
    }

example_transition = transition(
    implementation = _transition_impl,
    inputs = ["//%s:transition_input_flag" % example_package],
    outputs = [
      "//%s:transition_output_flag" % example_package,
      "//%s:transition_input_flag" % example_package,
    ],
)

def _rule_impl(ctx):
    print("Flag value for %s: %s" % (
        ctx.label.name,
        ctx.attr._transition_output_flag[BuildSettingInfo].value,
    ))

transition_attached = rule(
    implementation = _rule_impl,
    cfg = example_transition,
    attrs = {
        "apply_transition": attr.bool(default = False),
        "deps": attr.label_list(),
        "_transition_output_flag": attr.label(default = "//%s:transition_output_flag" % example_package),
    },
)
EOF

  # create rules with transition attached
  cat > "${pkg}/BUILD" <<EOF
load(
    "//${pkg}:def.bzl",
    "transition_attached",
)
load("//settings:flag.bzl", "bool_flag")

bool_flag(
    name = "transition_input_flag",
    build_setting_default = True,
)

bool_flag(
    name = "transition_output_flag",
    build_setting_default = False,
)

config_setting(
    name = "select_setting",
    flag_values = {":transition_input_flag": "True"},
)

# All should print "False" if
# "--no//${pkg}:transition_input_flag" is
# specified on the command line

# bazel build :top_level will print the results for all of the targets below

transition_attached(
    name = "top_level",
    apply_transition = select({
        ":select_setting": True,
        "//conditions:default": False,
    }),
)
EOF
  bazel build "//${pkg}:top_level" &> $TEST_log && fail "Build did NOT complete successfully"
  expect_log "No attribute 'apply_transition'. Either this attribute does not exist for this rule or the attribute was not resolved because it is set by a select that reads flags the transition may set."
}

function test_unresolvable_select_does_not_error_out_before_applying_rule_transition() {
  local -r pkg="${FUNCNAME[0]}"

  # create transition definition
  mkdir -p "${pkg}"
  cat > "${pkg}/def.bzl" <<EOF
load("//settings:flag.bzl", "BuildSettingInfo")

example_package = "${pkg}"

# transition that sets cpu so that unresolvable selects before rule transition
# becomes resolvable after rule transition
def _transition_impl(settings, attr):
    return {"//command_line_option:cpu": "ios_x86_64"}

example_transition = transition(
    implementation = _transition_impl,
    inputs = [],
    outputs = ["//command_line_option:cpu"],
)

def _rule_impl(ctx):
    print("Hello")

transition_attached = rule(
    implementation = _rule_impl,
    cfg = example_transition,
    attrs = {
        "cpu_name": attr.string(),
    },
)
EOF

  # create rules with transition attached
  cat > "${pkg}/BUILD" <<EOF
load(
    "//${pkg}:def.bzl",
    "transition_attached",
)

config_setting(
    name = "ios_x86_64",
    values = {
        "cpu": "ios_x86_64",
    },
)

config_setting(
    name = "darwin",
    values = {
        "cpu": "darwin",
    },
)

transition_attached(
    name = "top_level",
    cpu_name = select(
        {
            ":ios_x86_64": "ios_x86_64",
            ":darwin": "darwin",
        },
    ),
)
EOF
  bazel build "//${pkg}:top_level" &> $TEST_log || fail "Build failed"
}

function test_unresolvable_select_error_out_after_applying_rule_transition() {
  local -r pkg="${FUNCNAME[0]}"

  # create transition definition
  mkdir -p "${pkg}"
  cat > "${pkg}/def.bzl" <<EOF
load("//settings:flag.bzl", "BuildSettingInfo")

example_package = "${pkg}"

# dummy transition that does not set anything and we expect to see BUILD ERROR
# for unresolvable select after applying rule transition
def _transition_impl(settings, attr):
    return {}

example_transition = transition(
    implementation = _transition_impl,
    inputs = [],
    outputs = [],
)

def _rule_impl(ctx):
    print("Hello")

transition_attached = rule(
    implementation = _rule_impl,
    cfg = example_transition,
    attrs = {
        "cpu_name": attr.string(),
    },
)
EOF

  # create rules with transition attached
  cat > "${pkg}/BUILD" <<EOF
load(
    "//${pkg}:def.bzl",
    "transition_attached",
)

config_setting(
  name = "ios_x86_64",
  values = {
        "cpu": "ios_x86_64",
  },
)

config_setting(
  name = "darwin",
  values = {
        "cpu": "darwin",
  },
)

transition_attached(
    name = "top_level",
    cpu_name = select(
        {
            ":ios_x86_64": "ios_x86_64",
            ":darwin": "darwin",
        },
    ),
)
EOF
  bazel build "//${pkg}:top_level" &> $TEST_log && fail "Build did NOT complete successfully"
  expect_log "configurable attribute \"cpu_name\" in //test_unresolvable_select_error_out_after_applying_rule_transition:top_level doesn't match this configuration. Would a default condition help?"
}

# Regression test for b/338660045
function write_rule_list_transition() {
  local pkg="${1}"
  mkdir -p "${pkg}"

  # Create transition definition
  cat > "${pkg}/def.bzl" <<EOF
load("//settings:flag.bzl", "BuildSettingInfo")

example_package = "${pkg}"

# Transition that checks a list-typed attribute.
def _transition_impl(settings, attr):
    values = getattr(attr, "values")
    sorted_values = sorted(values)
    print("From transition: values = %s" % str(values))
    print("From transition: sorted values = %s" % str(sorted_values))
    return {"//%s:transition_output_flag" % example_package: sorted_values}

example_transition = transition(
    implementation = _transition_impl,
    inputs = [],
    outputs = ["//%s:transition_output_flag" % example_package],
)

def _rule_impl(ctx):
    attr_values = ctx.attr.values
    print("From rule attributes: values = %s" % str(attr_values))
    flag_values = ctx.attr._transition_output_flag[BuildSettingInfo].value
    print("From rule flag: values = %s" % str(flag_values))

    log = ctx.outputs.log
    content = "attr: %s\nflag: %s" % (str(attr_values), str(flag_values))
    ctx.actions.write(
        output = log,
        content = content,
    )

transition_attached = rule(
    implementation = _rule_impl,
    cfg = example_transition,
    attrs = {
        "values": attr.string_list(),
        "_transition_output_flag": attr.label(default = "//%s:transition_output_flag" % example_package),
        "log": attr.output(),
    },
)
EOF

  # Create rules with transition attached
  cat > "${pkg}/BUILD" <<EOF
load(
    "//${pkg}:def.bzl",
    "transition_attached",
)
load("//settings:flag.bzl", "bool_flag", "string_list_flag")

bool_flag(
    name = "select_flag",
    build_setting_default = True,
)

config_setting(
    name = "select_setting",
    flag_values = {":select_flag": "True"},
)

string_list_flag(
    name = "transition_output_flag",
    build_setting_default = [],
)

FRUITS = [
    # Deliberately not sorted.
    "banana",
    "grape",
    "apple",
]
ROCKS = [
    # Deliberately not sorted.
    "marble",
    "granite",
    "sandstone",
]

transition_attached(
    name = "top_level",
    log = "top_level.log",
    values = select({
        ":select_setting": FRUITS,
        "//conditions:default": ROCKS,
    }),
)
EOF
}

function test_inspect_attribute_list_direct() {
  local -r pkg="${FUNCNAME[0]}"

  write_rule_list_transition "${pkg}"

  bazel build --//"${pkg}":select_flag "//${pkg}:top_level" &> $TEST_log || fail "Build failed"
  expect_log 'From rule attributes: values = \["banana", "grape", "apple"\]'
  expect_log 'From rule flag: values = \["apple", "banana", "grape"\]'

  bazel build --//"${pkg}":select_flag=false "//${pkg}:top_level" &> $TEST_log || fail "Build failed"
  expect_log 'From rule attributes: values = \["marble", "granite", "sandstone"\]'
  expect_log 'From rule flag: values = \["granite", "marble", "sandstone"\]'
}

function test_inspect_attribute_list_via_output() {
  local -r pkg="${FUNCNAME[0]}"

  write_rule_list_transition "${pkg}"

  # Build via the output: this was the actual issue in b/338660045
  bazel build --//"${pkg}":select_flag "//${pkg}:top_level.log" &> $TEST_log || fail "Build failed"
  expect_log 'From rule attributes: values = \["banana", "grape", "apple"\]'
  expect_log 'From rule flag: values = \["apple", "banana", "grape"\]'

  bazel build --//"${pkg}":select_flag=false "//${pkg}:top_level.log" &> $TEST_log || fail "Build failed"
  expect_log 'From rule attributes: values = \["marble", "granite", "sandstone"\]'
  expect_log 'From rule flag: values = \["granite", "marble", "sandstone"\]'
}

run_suite "rule transition tests"
