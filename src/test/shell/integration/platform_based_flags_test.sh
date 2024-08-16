#!/bin/bash
#
# Copyright 2024 The Bazel Authors. All rights reserved.
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
# An end-to-end test for Bazel's Platform-based Flags API.

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
  # Clean the bazelrc since some tests modify it
  write_default_bazelrc
}

function write_flags() {
  local -r pkg="$1"

  # Create some flags for testing, and some rules to check the values.
  mkdir -p "${pkg}"
  cat > "${pkg}"/sample_flag.bzl <<EOF
BuildSettingInfo = provider(fields = ["value"])

def _sample_flag_impl(ctx):
    return BuildSettingInfo(value = ctx.build_setting_value)

sample_flag = rule(
    implementation = _sample_flag_impl,
    build_setting = config.string(flag = True),
)

def _show_sample_flag_impl(ctx):
    value = ctx.attr.flag[BuildSettingInfo].value
    print("%s: value = \"%s\""% (ctx.label, value))

show_sample_flag = rule(
    implementation = _show_sample_flag_impl,
    attrs = {
        "flag": attr.label(providers = [BuildSettingInfo]),
    },
)
EOF

  # Create a transition on platform.
  cat > "${pkg}"/transition.bzl <<EOF
def _platform_transition_impl(settings, attr):
    # Change the platform.
    new_platform = attr.platform
    result = {
        "//command_line_option:platforms": [new_platform],
    }

    # Possibly change the flag value, too.
    new_flag_value = attr.flag_value
    if new_flag_value:
        result["//pbf:flag"] = new_flag_value
    else:
        result["//pbf:flag"] = settings["//pbf:flag"]

    return result

_platform_transition = transition(
    implementation = _platform_transition_impl,
    inputs = [
        "//pbf:flag",
    ],
    outputs = [
        "//command_line_option:platforms",
        "//pbf:flag",
    ],
)

def _change_platform_impl(ctx):
    pass

change_platform = rule(
    implementation = _change_platform_impl,
    attrs = {
        "target": attr.label(cfg = _platform_transition),
        "flag_value": attr.string(),
        "platform": attr.label(providers = [platform_common.PlatformInfo]),
    },
)
EOF

  cat > "${pkg}"/BUILD <<EOF
load(":sample_flag.bzl", "sample_flag")

package(
    default_visibility = ["//visibility:public"],
)

exports_files([
    "sample_flag.bzl",
    "transition.bzl",
])

sample_flag(
    name = "flag",
    build_setting_default = "default_value_for_flag",
)
EOF
}

# No platform flags, just verifying the flags.
function test_sample_flag() {
  local -r pkg="$FUNCNAME"
  write_flags "pbf"
  mkdir -p "$pkg"

  cat > "$pkg/BUILD" <<EOF
load("//pbf:sample_flag.bzl", "show_sample_flag")

show_sample_flag(
    name = "show",
    flag = "//pbf:flag",
)
EOF

  bazel build //$pkg:show &> $TEST_log || fail "bazel failed"
  expect_log "//$pkg:show: value = \"default_value_for_flag\""

  bazel build --//pbf:flag=cli //$pkg:show &> $TEST_log || fail "bazel failed"
  expect_log "//$pkg:show: value = \"cli\""
}

# Set the platform at the CLI, see changed flag value.
function test_platform_flag() {
  local -r pkg="$FUNCNAME"
  write_flags "pbf"
  mkdir -p "$pkg"

  cat > "$pkg/BUILD" <<EOF
load("//pbf:sample_flag.bzl", "show_sample_flag")

platform(
    name = "pbf_demo",
    flags = [
        "--//pbf:flag=platform",
    ],
)

show_sample_flag(
    name = "show",
    flag = "//pbf:flag",
)
EOF

  bazel build --platforms="//$pkg:pbf_demo" //$pkg:show &> $TEST_log || fail "bazel failed"
  expect_log "//${pkg}:show: value = \"platform\""
}

# Set the platform and the flag at the CLI, see the platform value.
function test_platform_flag_and_override() {
  local -r pkg="$FUNCNAME"
  write_flags "pbf"
  mkdir -p "$pkg"

  cat > "$pkg/BUILD" <<EOF
load("//pbf:sample_flag.bzl", "show_sample_flag")

platform(
    name = "pbf_demo",
    flags = [
        "--//pbf:flag=platform",
    ],
)

show_sample_flag(
    name = "show",
    flag = "//pbf:flag",
)
EOF

  bazel build --//pbf:flag=cli --platforms="//$pkg:pbf_demo" //$pkg:show &> $TEST_log || fail "bazel failed"
  expect_log "//$pkg:show: value = \"platform\""
}

# Inherit the flags from a parent platform.
function test_inherit() {
  local -r pkg="$FUNCNAME"
  write_flags "pbf"
  mkdir -p "$pkg"

  cat > "$pkg/BUILD" <<EOF
load("//pbf:sample_flag.bzl", "show_sample_flag")

platform(
    name = "parent",
    flags = [
        "--//pbf:flag=parent",
    ],
)

platform(
    name = "pbf_demo",
    parents = [":parent"],
)

show_sample_flag(
    name = "show",
    flag = "//pbf:flag",
)
EOF

  bazel build --platforms="//$pkg:pbf_demo" //$pkg:show &> $TEST_log || fail "bazel failed"
  expect_log "//$pkg:show: value = \"parent\""
}

# Inherit the flags from a parent platform but override.
function test_inherit_override() {
  local -r pkg="$FUNCNAME"
  write_flags "pbf"
  mkdir -p "$pkg"

  mkdir -p "$pkg/flag"
  cat > $pkg/flag/other_flag.bzl <<EOF
load('//pbf:sample_flag.bzl', BuildSettingInfo1 = 'BuildSettingInfo')
BuildSettingInfo2 = provider(fields = ["value"])

def _other_flag_impl(ctx):
    return BuildSettingInfo2(value = ctx.build_setting_value)

other_flag = rule(
    implementation = _other_flag_impl,
    build_setting = config.string(flag = True),
)

def _show_other_flag_impl(ctx):
    value1 = ctx.attr.flag1[BuildSettingInfo1].value
    value2 = ctx.attr.flag2[BuildSettingInfo2].value
    print("%s: value1 = \"%s\""% (ctx.label, value1))
    print("%s: value2 = \"%s\""% (ctx.label, value2))

show_other_flag = rule(
    implementation = _show_other_flag_impl,
    attrs = {
        "flag1": attr.label(providers = [BuildSettingInfo1]),
        "flag2": attr.label(providers = [BuildSettingInfo2]),
    },
)
EOF

  cat > $pkg/flag/BUILD <<EOF
load(":other_flag.bzl", "other_flag")

package(
    default_visibility = ["//visibility:public"],
)

other_flag(
    name = "other",
    build_setting_default = "default_value_for_flag2",
)
EOF

  cat > "$pkg/BUILD" <<EOF
load("//$pkg/flag:other_flag.bzl", "show_other_flag")

platform(
    name = "parent",
    flags = [
        "--//pbf:flag=parent",
        "--//${pkg}/flag:other=parent",
    ],
)

platform(
    name = "pbf_demo",
    parents = [":parent"],
    flags = [
        "--//pbf:flag=child",
    ],
)

show_other_flag(
    name = "show_other",
    flag1 = "//pbf:flag",
    flag2 = "//${pkg}/flag:other",
)
EOF

  bazel build --platforms="//$pkg:pbf_demo" //$pkg:show_other &> $TEST_log || fail "bazel failed"
  expect_log "//$pkg:show_other: value1 = \"child\""
  expect_log "//$pkg:show_other: value2 = \"parent\""
}

# Set the platform in an rcfile, see changed flag value.
function test_rcfile() {
  local -r pkg="$FUNCNAME"
  write_flags "pbf"
  mkdir -p "$pkg"

  cat > "$pkg/BUILD" <<EOF
load("//pbf:sample_flag.bzl", "show_sample_flag")

platform(
    name = "pbf_demo",
    flags = [
        "--//pbf:flag=platform",
    ],
)

show_sample_flag(
    name = "show",
    flag = "//pbf:flag",
)
EOF

  add_to_bazelrc "common --platforms=//$pkg:pbf_demo"

  bazel build //$pkg:show &> $TEST_log || fail "bazel failed"
  expect_log "//$pkg:show: value = \"platform\""
}

# Set the platform in an rcfile, see changed flag value.
function test_rcfile_config() {
  local -r pkg="$FUNCNAME"
  write_flags "pbf"
  mkdir -p "$pkg"

  cat > "$pkg/BUILD" <<EOF
load("//pbf:sample_flag.bzl", "show_sample_flag")

platform(
    name = "pbf_demo",
    flags = [
        "--//pbf:flag=platform",
    ],
)

show_sample_flag(
    name = "show",
    flag = "//pbf:flag",
)
EOF

  add_to_bazelrc "common:pbf --platforms=//$pkg:pbf_demo"

  bazel build //$pkg:show &> $TEST_log || fail "bazel failed"
  expect_log "//$pkg:show: value = \"default_value_for_flag\""

  bazel build --config=pbf //$pkg:show &> $TEST_log || fail "bazel failed"
  expect_log "//$pkg:show: value = \"platform\""
}

# Change the platform in a transition, see changed flag value.
function test_transition() {
  local -r pkg="$FUNCNAME"
  write_flags "pbf"
  mkdir -p "$pkg"

  cat > "$pkg/BUILD" <<EOF
load("//pbf:sample_flag.bzl", "show_sample_flag")
load("//pbf:transition.bzl", "change_platform")

platform(
    name = "pbf_demo",
    flags = [
        "--//pbf:flag=via_transition",
    ],
)

change_platform(
    name = "with_transition",
    platform = ":pbf_demo",
    target = ":show",
)

show_sample_flag(
    name = "show",
    flag = "//pbf:flag",
)
EOF

  bazel build //$pkg:with_transition &> $TEST_log || fail "bazel failed"
  expect_log "//$pkg:show: value = \"via_transition\""
}

# Change the platform and the flag in a transition, see direct value.
function test_transition_override() {
  local -r pkg="$FUNCNAME"
  write_flags "pbf"
  mkdir -p "$pkg"

  cat > "$pkg/BUILD" <<EOF
load("//pbf:sample_flag.bzl", "show_sample_flag")
load("//pbf:transition.bzl", "change_platform")

platform(
    name = "pbf_demo",
    flags = [
        "--//pbf:flag=via_transition",
    ],
)

change_platform(
    name = "with_transition",
    platform = ":pbf_demo",
    flag_value = "direct",
    target = ":show",
)

show_sample_flag(
    name = "show",
    flag = "//pbf:flag",
)
EOF

  bazel build //$pkg:with_transition &> $TEST_log || fail "bazel failed"
  expect_log "//$pkg:show: value = \"via_transition\""
}

# Change the platform in a transition, mapping is not applied.
function test_transition_ignores_mapping() {
  local -r pkg="$FUNCNAME"
  write_flags "pbf"
  mkdir -p "$pkg"

  cat > "$pkg/BUILD" <<EOF
load("//pbf:sample_flag.bzl", "show_sample_flag")
load("//pbf:transition.bzl", "change_platform")

platform(
    name = "pbf_demo",
    flags = [
        "--//pbf:flag=via_transition",
    ],
)

change_platform(
    name = "with_transition",
    platform = ":pbf_demo",
    target = ":show",
)

show_sample_flag(
    name = "show",
    flag = "//pbf:flag",
)
EOF

  cat > "$pkg/platform_mappings" << EOF
platforms:
    //$pkg:pbf_demo
      --//pbf:flag=from_mapping
EOF

  bazel build --platform_mappings="$pkg/platform_mappings" //$pkg:with_transition &> $TEST_log || fail "bazel failed"
  expect_log "//$pkg:show: value = \"via_transition\""
}

# Regression test for https://github.com/bazelbuild/bazel/issues/22995
function test_cache_invalidation() {
  local -r pkg="$FUNCNAME"
  write_flags "pbf"
  mkdir -p "$pkg"

  cat > "$pkg/BUILD" <<EOF
load("//pbf:sample_flag.bzl", "show_sample_flag")

platform(
    name = "pbf_demo",
    flags = [
        "--//pbf:flag=first",
    ],
)

show_sample_flag(
    name = "show",
    flag = "//pbf:flag",
)
EOF

  bazel build --platforms="//$pkg:pbf_demo" //$pkg:show &> $TEST_log || fail "bazel failed"
  expect_log "//$pkg:show: value = \"first\""

  # Now change the platform definition.
  cat > "$pkg/BUILD" <<EOF
load("//pbf:sample_flag.bzl", "show_sample_flag")

platform(
    name = "pbf_demo",
    flags = [
        "--//pbf:flag=second",
    ],
)

show_sample_flag(
    name = "show",
    flag = "//pbf:flag",
)
EOF

  bazel build --platforms="//$pkg:pbf_demo" //$pkg:show &> $TEST_log || fail "bazel failed"
  expect_log "//$pkg:show: value = \"second\""
}

# Regression test for https://github.com/bazelbuild/bazel/issues/23147
function test_reset_starlark_flag_to_default() {
  local -r pkg="$FUNCNAME"
  write_flags "pbf"
  mkdir -p "$pkg"

  cat > "$pkg/BUILD" <<EOF
load("//pbf:sample_flag.bzl", "show_sample_flag")

platform(
    name = "pbf_demo",
    flags = [
        # Reset the flag to the default value.
        "--//pbf:flag=default_value_for_flag",
    ],
)

show_sample_flag(
    name = "show",
    flag = "//pbf:flag",
)
EOF

  bazel build --//pbf:flag=cli --platforms="//$pkg:pbf_demo" //$pkg:show &> $TEST_log || fail "bazel failed"

  # Ensure the default value is seen.
  expect_not_log "//$pkg:show: value = \"platform\""
  expect_log "//$pkg:show: value = \"default_value_for_flag\""

  # Check that the now-default Starlark flag is not present in the
  # configuration.
  for config in $(bazel config | tail -n +2 | cut -d ' ' -f 1); do
    bazel config "${config}" &> $TEST_log
    expect_not_log "//pbf:flag:.*default_value_for_flag" "Found default value for //pbf:flag in config ${config}"
  done
}

function is_bazel() {
  [ $TEST_WORKSPACE == "_main" ]
}

# Regression test for https://github.com/bazelbuild/bazel/issues/23115
function test_module_label_flag() {
  # Only bazel supports modules.
  is_bazel || return 0

  local -r pkg="$FUNCNAME"
  mkdir -p "$pkg"

  # Create another module that defines a label flag.
  mkdir -p "$pkg/flags"
  cat > "$pkg/flags/MODULE.bazel" <<EOF
module(name = "flags")
EOF

  write_flags "${pkg}/flags/pbf"
  cat >>${pkg}/flags/rule.bzl <<EOF
load('//pbf:sample_flag.bzl', 'BuildSettingInfo')
def _impl(ctx):
    value = ctx.attr.value
    return BuildSettingInfo(value = value)

sample_value = rule(
    implementation = _impl,
    attrs = {
        "value": attr.string(),
    },
)
EOF
  cat >>${pkg}/flags/BUILD <<EOF
load(":rule.bzl", "sample_value")

package(default_visibility = ["//visibility:public"])

label_flag(
    name = "label_flag",
    build_setting_default = ":default",
)

sample_value(
    name = "default",
    value = "default",
)

sample_value(
    name = "alt",
    value = "alt",
)
EOF

  # Refer to the module from the base.
  cat > MODULE.bazel <<EOF
module(name = "main")

bazel_dep(name = "flags")

local_path_override(
    module_name = "flags",
    path = "${pkg}/flags",
)
EOF

  cat >> $pkg/BUILD <<EOF
load("@flags//pbf:sample_flag.bzl", "show_sample_flag")

platform(
    name = "pbf_demo",
    flags = [
        # Test setting the value to a label in the module.
        "--@flags//:label_flag=@flags//:alt",
    ],
)

show_sample_flag(
    name = "show",
    flag = "@flags//:label_flag",
)
EOF

  bazel build --platforms="//$pkg:pbf_demo" //$pkg:show &> $TEST_log || fail "bazel failed"
  expect_log "//${pkg}:show: value = \"alt\""
}

run_suite "Tests for platform based flags"
