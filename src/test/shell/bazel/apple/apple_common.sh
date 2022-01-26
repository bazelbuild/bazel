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

  # All of the attributes below, except for `stamp`, are required as part of the
  # implied contract of `apple_common.link_multi_arch_binary` since it asks for
  # attributes directly from the rule context. As these requirements are changed
  # from implied attributes to function arguments, they can be removed.
  cat >> "${dir}/starlark_apple_binary.bzl" <<EOF
def _starlark_apple_binary_impl(ctx):
    link_result = apple_common.link_multi_arch_binary(
        ctx = ctx,
        stamp = ctx.attr.stamp,
    )
    processed_binary = ctx.actions.declare_file(
        '{}_lipobin'.format(ctx.label.name)
    )
    lipo_inputs = [output.binary for output in link_result.outputs]
    if len(lipo_inputs) > 1:
        apple_env = {}
        xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]
        apple_env.update(apple_common.apple_host_system_env(xcode_config))
        apple_env.update(
            apple_common.target_apple_env(
                xcode_config,
                ctx.fragments.apple.single_arch_platform,
            ),
        )
        args = ctx.actions.args()
        args.add('-create')
        args.add_all(lipo_inputs)
        args.add('-output', processed_binary)
        ctx.actions.run(
            arguments = [args],
            env = apple_env,
            executable = '/usr/bin/lipo',
            execution_requirements = xcode_config.execution_info(),
            inputs = lipo_inputs,
            outputs = [processed_binary],
        )
    else:
        ctx.actions.symlink(
            target_file = lipo_inputs[0],
            output = processed_binary,
        )
    return [
        DefaultInfo(files = depset([processed_binary])),
        OutputGroupInfo(**link_result.output_groups),
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
