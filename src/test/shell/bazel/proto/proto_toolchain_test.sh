#!/bin/bash
#
# Copyright 2021 The Bazel Authors. All rights reserved.
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
# Tests for `proto_toolchain` provided by `@bazel_tools`.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function setup_workspace() {
  cat >> WORKSPACE <<EOF
# TODO(#9029): May require some adjustment if/when we depend on the real
# @rules_python in the real source tree, since this third_party/ package won't
# be available.
new_local_repository(
    name = "rules_python",
    path = "$(dirname $(rlocation io_bazel/third_party/rules_python/rules_python.WORKSPACE))",
    build_file = "$(rlocation io_bazel/third_party/rules_python/BUILD)",
    workspace_file = "$(rlocation io_bazel/third_party/rules_python/rules_python.WORKSPACE)",
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
EOF
  cat $(rlocation io_bazel/src/tests/shell/bazel/rules_proto_stanza.txt) >> WORKSPACE
  cat >> WORKSPACE << EOF
load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
rules_proto_dependencies()
rules_proto_toolchains()

# @com_google_protobuf//:protoc depends on @io_bazel//third_party/zlib.
new_local_repository(
    name = "io_bazel",
    path = "$(dirname $(rlocation io_bazel/third_party/rules_python/rules_python.WORKSPACE))/../..",
    build_file_content = "# Intentionally left empty.",
    workspace_file_content = "workspace(name = 'io_bazel')",
)

# TODO(yannic): Remove when the toolchain is registered in a WORKSPACE suffix.
register_toolchains(
    "@bazel_tools//tools/proto/private:default_toolchain",
)
EOF
}

function define_print_compiler_flags() {
  cat >> print_compiler_flags.bzl <<EOF
def _print_compiler_flags_impl(ctx):
    toolchain = ctx.toolchains["@bazel_tools//tools/proto:toolchain_type"]
    opts = toolchain.proto.compiler_options
    print("compiler_options({}): {}".format(len(opts), " ".join(opts)))

print_compiler_flags = rule(
    implementation = _print_compiler_flags_impl,
    toolchains = [
        "@bazel_tools//tools/proto:toolchain_type",
    ],
)

def _print_compiler_flags_from_alias_impl(ctx):
    opts = ctx.attr._alias[ProtoToolchainInfo].compiler_options
    print("compiler_options({}): {}".format(len(opts), " ".join(opts)))

print_compiler_flags_from_alias = rule(
    implementation = _print_compiler_flags_from_alias_impl,
    attrs = {
        "_alias": attr.label(
            default = "@bazel_tools//tools/proto/private:toolchain_alias",
        ),
    },
)
EOF

  cat >> BUILD <<EOF
load(
    ":print_compiler_flags.bzl",
    "print_compiler_flags",
    "print_compiler_flags_from_alias",
)

print_compiler_flags(name = "print_compiler_flags")
print_compiler_flags_from_alias(name = "print_compiler_flags_from_alias")
EOF
}

function define_custom_proto_toolchain() {
  mkdir -p proto
  cat >> proto/BUILD <<EOF
load("@bazel_tools//tools/proto:defs.bzl", "proto_toolchain")

proto_toolchain(
    name = "impl",
    compiler = "@com_google_protobuf//:protoc",
    compiler_options = [
        "--hello=world",
    ],
)

toolchain(
    name = "toolchain",
    toolchain = ":impl",
    toolchain_type = "@bazel_tools//tools/proto:toolchain_type",
)
EOF
}

function test_default_proto_toolchain() {
  setup_workspace
  define_print_compiler_flags

  # No command-line protoc compiler option.
  bazel build //:print_compiler_flags >& "$TEST_log" || fail "build failed"
  expect_log "compiler_options(0):"

  # Single command-line protoc compiler option.
  bazel build \
    --protocopt=--foo \
    //:print_compiler_flags >& "$TEST_log" || fail "build failed"
  expect_log "compiler_options(1): --foo"

  # Multiple command-line protoc compiler option.
  bazel build \
    --protocopt=--foo \
    --protocopt=--bar=baz \
    --protocopt=--foo \
    //:print_compiler_flags >& "$TEST_log" || fail "build failed"
  expect_log "compiler_options(3): --foo --bar=baz --foo"
}

function test_custom_proto_toolchain() {
  setup_workspace
  define_print_compiler_flags
  define_custom_proto_toolchain

  # No command-line protoc compiler option.
  bazel build \
    --extra_toolchains=//proto:toolchain \
    //:print_compiler_flags >& "$TEST_log" || fail "build failed"
  expect_log "compiler_options(1): --hello=world"

  # Single command-line protoc compiler option.
  bazel build \
    --extra_toolchains=//proto:toolchain \
    --protocopt=--foo \
    //:print_compiler_flags >& "$TEST_log" || fail "build failed"
  expect_log "compiler_options(2): --hello=world --foo"

  # Multiple command-line protoc compiler option.
  bazel build \
    --extra_toolchains=//proto:toolchain \
    --protocopt=--foo \
    --protocopt=--bar=baz \
    --protocopt=--foo \
    //:print_compiler_flags >& "$TEST_log" || fail "build failed"
  expect_log "compiler_options(4): --hello=world --foo --bar=baz --foo"
}

function test_proto_toolchain_alias() {
  setup_workspace
  define_print_compiler_flags
  define_custom_proto_toolchain

  # Default toolchain
  bazel build \
    --protocopt=--foo \
    --protocopt=--bar=baz \
    --protocopt=--foo \
    //:print_compiler_flags_from_alias >& "$TEST_log" || fail "build failed"
  expect_log "compiler_options(3): --foo --bar=baz --foo"

  # Custom toolchain
  bazel build \
    --extra_toolchains=//proto:toolchain \
    --protocopt=--foo \
    --protocopt=--bar=baz \
    --protocopt=--foo \
    //:print_compiler_flags_from_alias >& "$TEST_log" || fail "build failed"
  expect_log "compiler_options(4): --hello=world --foo --bar=baz --foo"
}

run_suite "Integration tests for proto_toolchain"
