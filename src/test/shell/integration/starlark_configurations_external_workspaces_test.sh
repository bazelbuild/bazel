#!/usr/bin/env bash
#
# Copyright 2019 The Bazel Authors. All rights reserved.
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
# starlark_configuration_test.sh: integration tests for starlark build
# configurations with external workspaces.

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

add_to_bazelrc "build --package_path=%workspace%"

#### HELPER FXNS #######################################################

function write_build_setting_bzl() {
  declare -r at_workspace="${1:-}"

  cat > $pkg/provider.bzl <<EOF
BuildSettingInfo = provider(fields = ['name', 'value'])
EOF

  cat > $pkg/build_setting.bzl <<EOF
load("@${WORKSPACE_NAME}//$pkg:provider.bzl", "BuildSettingInfo")

def _build_setting_impl(ctx):
  return [BuildSettingInfo(name = ctx.attr.name, value = ctx.build_setting_value)]

drink_attribute = rule(
  implementation = _build_setting_impl,
  build_setting = config.string(flag = True),
)
EOF

  cat > $pkg/rules.bzl <<EOF
load("@${WORKSPACE_NAME}//$pkg:provider.bzl", "BuildSettingInfo")

def _impl(ctx):
  _type_name = ctx.attr._type[BuildSettingInfo].name
  _type_setting = ctx.attr._type[BuildSettingInfo].value
  print(_type_name + "=" + str(_type_setting))
  _temp_name = ctx.attr._temp[BuildSettingInfo].name
  _temp_setting = ctx.attr._temp[BuildSettingInfo].value
  print(_temp_name + "=" + str(_temp_setting))
  print("strict_java_deps=" + ctx.fragments.java.strict_java_deps)

drink = rule(
  implementation = _impl,
  attrs = {
    "_type":attr.label(default = Label("$at_workspace//$pkg:type")),
    "_temp":attr.label(default = Label("$at_workspace//$pkg:temp")),
  },
  fragments = ["java"],
)
EOF

  cat > $pkg/BUILD <<EOF
load("//$pkg:build_setting.bzl", "drink_attribute")
load("//$pkg:rules.bzl", "drink")

drink(name = 'my_drink')

drink_attribute(
  name = 'type',
  build_setting_default = 'unknown',
  visibility = ['//visibility:public'],
)
drink_attribute(
  name = 'temp',
  build_setting_default = 'unknown',
  visibility = ['//visibility:public'],
)
EOF
}

#### TESTS #############################################################


function test_reference_inner_repository_flags() {
  local -r pkg=$FUNCNAME
  local -r subpkg="$pkg/sub"
  mkdir -p $subpkg

  ## set up outer repo
  cat > $(setup_module_dot_bazel "$pkg/MODULE.bazel") <<EOF
local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
local_repository(
  name = "sub",
  path = "./sub")
EOF

  ## set up inner repo
  cat > $subpkg/BUILD <<EOF
load(":rules.bzl", "rule_with_transition", "my_flag")

my_flag(
    name = "my_flag",
    build_setting_default = "saguaro",
)

rule_with_transition(
    name = "my_target",
    src = ":my_flag",
)
EOF

  cat > $subpkg/rules.bzl <<EOF
BuildSettingInfo = provider(fields = ['value'])

def _flag_impl(ctx):
 return BuildSettingInfo(value = ctx.build_setting_value)

my_flag = rule(
  implementation = _flag_impl,
  build_setting = config.string(flag = True),
)

def _my_transition_impl(settings, attr):
    print("value before transition: " + settings["@sub//:my_flag"])
    return {"@sub//:my_flag": "prickly-pear"}

my_transition = transition(
    implementation = _my_transition_impl,
    inputs = ["@sub//:my_flag"],
    outputs = ["@sub//:my_flag"],
)

def _rule_impl(ctx):
    print("value after transition: " + ctx.attr.src[BuildSettingInfo].value)

rule_with_transition = rule(
    implementation = _rule_impl,
    cfg = my_transition,
    attrs = {
        "src": attr.label(allow_files = True),
    },
)
EOF

  cat > $(setup_module_dot_bazel "$subpkg/MODULE.bazel") <<EOF
module(name = "sub")
EOF

  # from the outer repo
  cd $pkg
  bazel build @sub//:my_target \
      > output 2>"$TEST_log" || fail "Expected success"
  expect_log "value before transition: saguaro"
  expect_log "value after transition: prickly-pear"

  bazel build @sub//:my_target --@sub//:my_flag=prickly-pear \
       > output 2>"$TEST_log" || fail "Expected success"
  expect_log "value before transition: prickly-pear"
  expect_log "value after transition: prickly-pear"

  # from the inner repo
  cd sub
  bazel build :my_target \
      > output 2>"$TEST_log" || fail "Expected success"
  expect_log "value before transition: saguaro"
  expect_log "value after transition: prickly-pear"

  bazel build :my_target --//:my_flag=prickly-pear \
      > output 2>"$TEST_log" || fail "Expected success"
  expect_log "value before transition: prickly-pear"
  expect_log "value after transition: prickly-pear"

  bazel clean 2>"$TEST_log" || fail "Clean failed"

  bazel build :my_target --@sub//:my_flag=prickly-pear \
      > output 2>"$TEST_log" || fail "Expected success"
  expect_log "value before transition: prickly-pear"
  expect_log "value after transition: prickly-pear"

  bazel clean 2>"$TEST_log" || fail "Clean failed"

  bazel build :my_target --@//:my_flag=prickly-pear \
      > output 2>"$TEST_log" || fail "Expected success"
  expect_log "value before transition: prickly-pear"
  expect_log "value after transition: prickly-pear"
}

function test_rc_flag_alias_external_repo() {
  local -r pkg=$FUNCNAME
  local -r subpkg="$pkg/sub"
  mkdir -p $subpkg

  ## set up outer repo
  cat > $(setup_module_dot_bazel "$pkg/MODULE.bazel") <<EOF
local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
local_repository(
  name = "sub",
  path = "./sub")
EOF

  cat > $pkg/BUILD <<EOF
load("@sub//:rules.bzl", "flag_reader")
flag_reader(
    name = "reader",
    flag = "@sub//:my_flag",
)
EOF

  ## set up inner repo with a flag and a rule that reads it
  cat > $subpkg/BUILD <<EOF
load(":rules.bzl", "my_flag")
my_flag(
    name = "my_flag",
    build_setting_default = "default_value",
    visibility = ["//visibility:public"],
)
EOF

  cat > $subpkg/rules.bzl <<EOF
BuildSettingInfo = provider(fields = ['value'])

def _flag_impl(ctx):
    return BuildSettingInfo(value = ctx.build_setting_value)

my_flag = rule(
    implementation = _flag_impl,
    build_setting = config.string(flag = True),
)

def _flag_reader_impl(ctx):
    print("flag value: " + ctx.attr.flag[BuildSettingInfo].value)
    return []

flag_reader = rule(
    implementation = _flag_reader_impl,
    attrs = {
        "flag": attr.label(),
    },
)
EOF

  cat > $(setup_module_dot_bazel "$subpkg/MODULE.bazel") <<EOF
module(name = "sub")
EOF

  cd $pkg

  # Test that flag alias for external repo flag works correctly.
  bazel build :reader --flag_alias=myflag=@sub//:my_flag --myflag=custom_value \
      >& "$TEST_log" || fail "Expected success"
  expect_log "flag value: custom_value"
}

function test_bzl_module_flag_alias_function() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  # Set up rule, flag definitions.
  cat > $pkg/rules.bzl <<EOF
BuildSettingInfo = provider(fields = ['value'])
simple_flag = rule(
  implementation = lambda ctx: BuildSettingInfo(value = ctx.build_setting_value),
  build_setting = config.string(flag = True),
)
simple_rule = rule(
    implementation = lambda ctx: [],
    attrs = {}
)
EOF

  # Set up rule, flag instances.
  cat > $pkg/BUILD <<EOF
load(":rules.bzl", "simple_rule", "simple_flag")
simple_flag(
  name = "my_flag",
  build_setting_default = "default",
)
simple_rule(name = "buildme")
EOF

  # Set up root workspace's MODULE.bazel.
  cat > $(setup_module_dot_bazel "$pkg/MODULE.bazel") <<EOF
flag_alias(name = "compilation_mode", starlark_flag = "//:my_flag")
EOF

  cd $pkg
  # cquery a target with the alias-mapped flag.
  bazel cquery :buildme --compilation_mode=opt \
      --flag_alias=user_set=//:fake_flag > cquery_output 2>"$TEST_log" \
      || fail "Expected success"
  # Find the cquery target's configuration hash.
  config_hash=$(cat cquery_output  | grep -o '([a-zA-Z0-9]\+)' | tr -d '()')
  # Get the configuration.
  bazel config $config_hash > "$TEST_log" 2>&1 || fail "Expected success"
  expect_log "//:my_flag: opt" "Expected Starlark flag to have user value"
  expect_log "compilation_mode: fastbuild" \
      "Expected native flag to have default value"
  # This is important because select() and transitions read --flag_alias to
  # correctly map aliases.
  # This regex checks that the line starts with "flag_alias:" and contains
  # both "compilation_mode=@//:my_flag" and "user_set=//:fake_flag" in any order.
  local flag_alias_regex="flag_alias:.*\\("
  flag_alias_regex+="compilation_mode=@//:my_flag.*user_set=//:fake_flag"
  flag_alias_regex+="\\|"
  flag_alias_regex+="user_set=//:fake_flag.*compilation_mode=@//:my_flag"
  flag_alias_regex+="\\)"

  expect_log "$flag_alias_regex" \
      "Expected both aliases to be in --flag_alias option value list"
}

run_suite "${PRODUCT_NAME} starlark configurations tests"
