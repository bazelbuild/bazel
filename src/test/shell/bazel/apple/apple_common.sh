#!/bin/bash
#
# Copyright 2022 The Bazel Authors. All rights reserved.
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
# Common Bash functions to test Apple rules in Bazel.
#

function make_starlark_apple_binary_rule_in() {
  local dir="$1"; shift

  # TODO(b/63092114): Add lipo logic here, too. Based on a reduced subset of
  # lipo.bzl, avoiding the apple_support dependency.

  # All of the attributes below, except for `stamp`, are required as part of the
  # implied contract of `apple_common.link_multi_arch_binary` since it asks for
  # attributes directly from the rule context. As these requirements are changed
  # from implied attributes to function arguments, they can be removed.
  cat >> "${dir}/starlark_apple_binary.bzl" <<EOF
def _starlark_apple_binary_impl(ctx):
    link_result = apple_common.link_multi_arch_binary(
        ctx = ctx,
        should_lipo = True,
        stamp = ctx.attr.stamp,
    )
    return [
        DefaultInfo(files = depset([link_result.binary_provider.binary])),
        OutputGroupInfo(**link_result.output_groups),
        link_result.binary_provider,
        link_result.debug_outputs_provider,
    ]

starlark_apple_binary = rule(
    attrs = {
        "_child_configuration_dummy": attr.label(
            cfg = apple_common.multi_arch_split,
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
        "_xcode_config": attr.label(
            default = configuration_field(
                fragment = "apple",
                name = "xcode_config_label",
            ),
        ),
        "_xcrunwrapper": attr.label(
            cfg = "host",
            default = Label("@bazel_tools//tools/objc:xcrunwrapper"),
            executable = True,
        ),
        "binary_type": attr.string(default = "executable"),
        "bundle_loader": attr.label(),
        "deps": attr.label_list(
            cfg = apple_common.multi_arch_split,
        ),
        "dylibs": attr.label_list(),
        "linkopts": attr.string_list(),
        "minimum_os_version": attr.string(),
        "platform_type": attr.string(),
        "stamp": attr.int(default = -1, values = [-1, 0, 1]),
    },
    fragments = ["apple", "objc", "cpp"],
    implementation = _starlark_apple_binary_impl,
    outputs = {
        # Provided for compatibility with apple_binary tests only.
        "lipobin": "%{name}_lipobin",
    },
)
EOF
}
