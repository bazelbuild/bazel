#!/bin/bash
#
# Copyright 2018 The Bazel Authors. All rights reserved.
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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  create_new_workspace
}

function write_test_target() {
  cat > BUILD <<EOF
load("@bazel_tools//tools/cpp:cc_flags_supplier.bzl", "cc_flags_supplier")
cc_flags_supplier(name = "cc_flags")
load(":display.bzl", "display")
display(
    name = "display",
    toolchains = [":cc_flags"],
)
EOF

  # Add a rule to display CC_FLAGS
  cat >>display.bzl <<EOF
def _impl(ctx):
  cc_flags = ctx.var["CC_FLAGS"]
  print("CC_FLAGS: %s" % cc_flags)
  return []
display = rule(
  implementation = _impl,
)
EOF
}

function write_crosstool() {
  action_configs="$1"
  make_variables="$2"
  builtin_sysroot="$3"

  mkdir -p setup
  cat > setup/BUILD <<EOF
package(default_visibility = ["//visibility:public"])

load(":cc_toolchain_config.bzl", "cc_toolchain_config")
cc_library(
    name = "malloc",
)

filegroup(
    name = "empty",
    srcs = [],
)

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "local|local": ":local",
        "local": ":local",
    },
)

cc_toolchain_config(
    name = "local_config",
    cpu = "local",
    toolchain_identifier = "local",
    host_system_name = "local",
    target_system_name = "local",
    target_libc = "local",
    compiler = "local",
    abi_version = "local",
    abi_libc_version = "local",
)

cc_toolchain(
    name = "local",
    toolchain_identifier = "local",
    toolchain_config = ":local_config",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
)

toolchain(
    name = "cc-toolchain-local",
    exec_compatible_with = [
    ],
    target_compatible_with = [
    ],
    toolchain = ":local",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)
EOF

  cat >> WORKSPACE <<EOF
register_toolchains("//setup:cc-toolchain-local")
EOF
  cat > setup/cc_toolchain_config.bzl <<EOF
load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
    "make_variable",
    "flag_set",
    "flag_group",
)

def _get_make_variables():
    return [${make_variables}]

def _get_action_configs():
    return [${action_configs}]

def _impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(out, "Fake executable")
    return [
        cc_common.create_cc_toolchain_config_info(
            ctx = ctx,
            target_cpu = ctx.attr.cpu,
            compiler = ctx.attr.compiler,
            toolchain_identifier = ctx.attr.toolchain_identifier,
            host_system_name = ctx.attr.host_system_name,
            target_system_name = ctx.attr.target_system_name,
            target_libc = ctx.attr.target_libc,
            abi_version = ctx.attr.abi_version,
            abi_libc_version = ctx.attr.abi_libc_version,
            builtin_sysroot = ${builtin_sysroot},
            action_configs = _get_action_configs(),
            make_variables = _get_make_variables(),
        ),
        DefaultInfo(
            executable = out,
        ),
    ]


cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory = True),
        "compiler": attr.string(mandatory = True),
        "toolchain_identifier": attr.string(mandatory = True),
        "host_system_name": attr.string(mandatory = True),
        "target_system_name": attr.string(mandatory = True),
        "target_libc": attr.string(mandatory = True),
        "abi_version": attr.string(mandatory = True),
        "abi_libc_version": attr.string(mandatory = True),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)
EOF
}

function test_legacy_make_variable() {
  write_crosstool \
    "" \
    "make_variable(name = 'CC_FLAGS', value = '-test-cflag1 -test-cflag2')" \
    "''"

  write_test_target
  bazel build --cpu local --crosstool_top=//setup:toolchain //:display &> "$TEST_log" || fail "Build failed"
  expect_log "CC_FLAGS: -test-cflag1 -test-cflag2"
}

function test_sysroot() {
  write_crosstool \
    "" \
    "" \
    "'/sys/root'"

  write_test_target
  bazel build --cpu local --crosstool_top=//setup:toolchain //:display &> "$TEST_log" || fail "Build failed"
  expect_log "CC_FLAGS: --sysroot=/sys/root"
}

function test_feature_config() {
  write_crosstool \
    "
        action_config(
            action_name = 'cc-flags-make-variable',
            flag_sets = [
                flag_set(
                    flag_groups=[flag_group(flags=['foo', 'bar', 'baz'])],
                ),
            ],
        ),
    " \
    "" \
    "''"

  write_test_target
  bazel build --cpu local --crosstool_top=//setup:toolchain //:display &> "$TEST_log" || fail "Build failed"
  expect_log "CC_FLAGS: foo bar baz"
}

function test_all_sources() {
  write_crosstool \
    "
        action_config(
            action_name = 'cc-flags-make-variable',
            flag_sets = [
                flag_set(
                    flag_groups=[flag_group(flags=['foo', 'bar', 'baz'])],
                ),
            ],
        ),
    " \
    "make_variable(name = 'CC_FLAGS', value = '-test-cflag1 -test-cflag2')" \
     "'/sys/root'"

  write_test_target
  bazel build --cpu local --crosstool_top=//setup:toolchain //:display &> "$TEST_log" || fail "Build failed"
  expect_log "CC_FLAGS: -test-cflag1 -test-cflag2 --sysroot=/sys/root foo bar baz"
}

run_suite "cc_flag_supplier tests"
