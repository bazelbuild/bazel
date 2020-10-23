#!/bin/bash
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
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

if ! type try_with_timeout >&/dev/null; then
  # Bazel's testenv.sh defines try_with_timeout but the Google-internal version
  # uses a different testenv.sh.
  function try_with_timeout() { $* ; }
fi

#### TESTS #############################################################

function test_no_rebuild_on_irrelevant_header_change() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
cc_binary(name="a", srcs=["a.cc"], deps=["b"])
cc_library(name="b", srcs=["b1.h", "b2.h"])
EOF

  cat > $pkg/a.cc <<EOF
#include "$pkg/b1.h"

int main(void) {
  return B_RETURN_VALUE;
}
EOF

  cat > $pkg/b1.h <<EOF
#define B_RETURN_VALUE 31
EOF

  cat > $pkg/b2.h <<EOF
=== BANANA ===
EOF

  bazel build //$pkg:a || fail "build failed"
  echo "CHERRY" > $pkg/b2.h
  bazel build //$pkg:a >& $TEST_log || fail "build failed"
  expect_not_log "Compiling $pkg/a.cc"
}

function test_new_header_is_required() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
cc_binary(name="a", srcs=["a.cc"], deps=[":b"])
cc_library(name="b", srcs=["b1.h", "b2.h"])
EOF

  cat > $pkg/a.cc << EOF
#include "$pkg/b1.h"

int main(void) {
    return B1;
}
EOF

  cat > $pkg/b1.h <<EOF
#define B1 3
EOF

  cat > $pkg/b2.h <<EOF
#define B2 4
EOF

  bazel build //$pkg:a || fail "build failed"
  cat > $pkg/a.cc << EOF
#include "$pkg/b1.h"
#include "$pkg/b2.h"

int main(void) {
    return B1 + B2;
}
EOF

  bazel build //$pkg:a || fail "build failled"
}

function test_no_recompile_on_shutdown() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
cc_binary(name="a", srcs=["a.cc"], deps=["b"])
cc_library(name="b", includes=["."], hdrs=["b.h"])
EOF

  cat > $pkg/a.cc <<EOF
#include "b.h"

int main(void) {
  return B_RETURN_VALUE;
}
EOF

  cat > $pkg/b.h <<EOF
#define B_RETURN_VALUE 31
EOF

  bazel build -s //$pkg:a >& $TEST_log || fail "build failed"
  expect_log "Compiling $pkg/a.cc"
  try_with_timeout bazel shutdown || fail "shutdown failed"
  bazel build -s //$pkg:a >& $TEST_log || fail "build failed"
  expect_not_log "Compiling $pkg/a.cc"
}

# host_crosstool_top should default to the initial value of crosstool_top,
# not the value after a transition.
function test_default_host_crosstool_top() {
  local -r pkg=$FUNCNAME

  # Define two different toolchain suites to use with crosstool_top.
  mkdir -p $pkg/toolchain
  cat >> $pkg/toolchain/BUILD <<EOF
package(default_visibility = ["//visibility:public"])

load(":toolchain.bzl", "toolchains")

cc_library(
    name = "malloc",
)

filegroup(
    name = "empty",
    srcs = [],
)

[toolchains(id) for id in [
    "alpha",
    "beta",
]]
EOF
  cat >> $pkg/toolchain/toolchain.bzl <<EOF
load(
    "${TOOLS_REPOSITORY}//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
    "flag_group",
    "flag_set",
    "make_variable",
)

def _get_make_variables():
    return []

def _get_action_configs():
    return []

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

def toolchains(id):
    native.cc_toolchain_suite(
        name = "%s" % id,
        toolchains = {
            "fake": ":cc-toolchain-%s" % id,
        },
    )

    name = "toolchain-%s" % id
    cc_toolchain_config(
        name = "cc-toolchain-config-%s" % id,
        abi_libc_version = name,
        abi_version = name,
        compiler = name,
        cpu = name,
        host_system_name = name,
        target_libc = name,
        target_system_name = name,
        toolchain_identifier = name,
    )

    native.cc_toolchain(
        name = "cc-toolchain-%s" % id,
        all_files = ":empty",
        ar_files = ":empty",
        as_files = ":empty",
        compiler_files = ":empty",
        dwp_files = ":empty",
        linker_files = ":empty",
        objcopy_files = ":empty",
        strip_files = ":empty",
        toolchain_config = ":cc-toolchain-config-%s" % id,
        toolchain_identifier = name,
    )
EOF

  # Create an outer target, which uses a transition to depend on an inner.
  # The inner then depends on a tool in the exec configuration.
  # Even though te outer->inner dependency changes the value of crosstool_top,
  # the tool should use the initial crosstool.
  cat >> $pkg/BUILD <<EOF
load(":display.bzl", "inner", "outer", "tool")

outer(
    name = "outer",
    inner = ":inner",
)

inner(
    name = "inner",
    tool = ":tool",
)

tool(name = "tool")
EOF
  cat >> $pkg/display.bzl <<EOF
def find_cc_toolchain(ctx):
    return ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]

# Custom transition to change the crosstool.
def _crosstool_transition_impl(settings, attr):
    _ignore = (settings, attr)
    return {"//command_line_option:crosstool_top": "//${pkg}/toolchain:beta"}

crosstool_transition = transition(
    implementation = _crosstool_transition_impl,
    inputs = [],
    outputs = ["//command_line_option:crosstool_top"],
)

# Outer display rule.
def _outer_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)
    print("Outer %s found cc toolchain %s" % (ctx.label, cc_toolchain.target_gnu_system_name))
    pass

outer = rule(
    implementation = _outer_impl,
    attrs = {
        "inner": attr.label(mandatory = True, cfg = crosstool_transition),
        "_cc_toolchain": attr.label(
            default = Label(
                "${TOOLS_REPOSITORY}//tools/cpp:current_cc_toolchain",
            ),
        ),
        "_whitelist_function_transition": attr.label(default = "${TOOLS_REPOSITORY}//tools/whitelists/function_transition_whitelist"),
    },
)

# Inner display rule.
def _inner_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)
    print("Inner %s found cc toolchain %s" % (ctx.label, cc_toolchain.target_gnu_system_name))
    pass

inner = rule(
    implementation = _inner_impl,
    attrs = {
        "tool": attr.label(mandatory = True, cfg = "exec"),
        "_cc_toolchain": attr.label(
            default = Label(
                "${TOOLS_REPOSITORY}//tools/cpp:current_cc_toolchain",
            ),
        ),
    },
)

# Tool rule.
def _tool_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)
    print("Tool %s found cc toolchain %s" % (ctx.label, cc_toolchain.target_gnu_system_name))
    pass

tool = rule(
    implementation = _tool_impl,
    attrs = {
        "_cc_toolchain": attr.label(
            default = Label(
                "${TOOLS_REPOSITORY}//tools/cpp:current_cc_toolchain",
            ),
        ),
    },
)
EOF


  bazel build \
    --cpu=fake --host_cpu=fake \
    --crosstool_top=//$pkg/toolchain:alpha \
    //$pkg:outer >& $TEST_log || fail "build failed"
  expect_log "Outer //$pkg:outer found cc toolchain toolchain-alpha"
  expect_log "Inner //$pkg:inner found cc toolchain toolchain-beta"
  expect_log "Tool //$pkg:tool found cc toolchain toolchain-alpha"
}

run_suite "Tests for Bazel's C++ rules"
