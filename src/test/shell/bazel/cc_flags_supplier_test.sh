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
  extra_crosstool_content="$1"
  mkdir -p setup
  cat > setup/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "malloc",
)

cc_library(
    name = "stl",
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

cc_toolchain(
    name = "local",
    toolchain_identifier = "local",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    cpu = "local",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":empty",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
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
  cat > setup/CROSSTOOL <<EOF
major_version: "local"
minor_version: ""

toolchain {
  abi_version: "local"
  abi_libc_version: "local"
  compiler: "compiler"
  host_system_name: "local"
  target_libc: "local"
  target_cpu: "local"
  target_system_name: "local"
  toolchain_identifier: "local"

  ${extra_crosstool_content}
}
EOF
}

function test_legacy_make_variable() {
  write_crosstool "
make_variable {
  name: 'CC_FLAGS'
  value: '-test-cflag1 -test-cflag2'
}"

  write_test_target
  bazel build --cpu local --crosstool_top=//setup:toolchain //:display &> "$TEST_log" || fail "Build failed"
  expect_log "CC_FLAGS: -test-cflag1 -test-cflag2"
}

function test_sysroot() {
  write_crosstool "builtin_sysroot: '/sys/root'"

  write_test_target
  bazel build --cpu local --crosstool_top=//setup:toolchain //:display &> "$TEST_log" || fail "Build failed"
  expect_log "CC_FLAGS: --sysroot=/sys/root"
}

function test_feature_config() {
  write_crosstool "
action_config {
  action_name: 'cc-flags-make-variable'
  config_name: 'cc-flags-make-variable'
  flag_set {
    flag_group {
      flag: 'foo'
      flag: 'bar'
      flag: 'baz'
    }
  }
}"

  write_test_target
  bazel build --cpu local --crosstool_top=//setup:toolchain //:display &> "$TEST_log" || fail "Build failed"
  expect_log "CC_FLAGS: foo bar baz"
}

function test_all_sources() {
  write_crosstool "
make_variable {
  name: 'CC_FLAGS'
  value: '-test-cflag1 -test-cflag2'
}
builtin_sysroot: '/sys/root'
action_config {
  action_name: 'cc-flags-make-variable'
  config_name: 'cc-flags-make-variable'
  flag_set {
    flag_group {
      flag: 'foo'
      flag: 'bar'
      flag: 'baz'
    }
  }
}"

  write_test_target
  bazel build --cpu local --crosstool_top=//setup:toolchain //:display &> "$TEST_log" || fail "Build failed"
  expect_log "CC_FLAGS: -test-cflag1 -test-cflag2 --sysroot=/sys/root foo bar baz"
}

run_suite "cc_flag_supplier tests"
