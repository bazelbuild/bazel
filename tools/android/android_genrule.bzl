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

"""Allows building Android-specific dependencies with the correct configuration."""

def _android_transition_impl(settings, attr):
    return {
        "//command_line_option:platforms": str(attr.platform),
        "//command_line_option:android_platforms": str(attr.platform),
    }

_android_transition = transition(
    implementation = _android_transition_impl,
    inputs = [],
    outputs = [
        "//command_line_option:platforms",
        "//command_line_option:android_platforms",
    ],
)

# TODO(blaze-configurability-team): Consider replacing with platform_data when that is available.
def _android_platform_data_impl(ctx):
    result = []

    new_file = ctx.outputs.out
    original_file = ctx.file.target

    ctx.actions.symlink(
        output = new_file,
        target_file = original_file,
    )

    files = depset(direct = [new_file])

    result.append(
        DefaultInfo(
            files = files,
        ),
    )

    return result

_android_platform_data = rule(
    implementation = _android_platform_data_impl,
    attrs = {
        "target": attr.label(
            allow_single_file = True,
            cfg = _android_transition,
        ),
        "platform": attr.label(
            mandatory = True,
        ),
        "out": attr.output(),
    },
)

def android_genrule(
        name = "",
        platform = "",
        out = "",
        tags = [],
        visibility = [],
        **kwargs):
    # TODO(jcater): Check for empty name, platform, and outs.

    # Transition to an Android config.
    _android_platform_data(
        name = name,
        target = ":%s_for_android_%s" % (name, out),
        out = out,
        platform = platform,
        visibility = visibility,
    )

    # The actual genrule.
    native.genrule(
        name = "%s_for_android" % name,
        outs = [
            ":%s_for_android_%s" % (name, out),
        ],
        tags = tags + ["manual"],
        visibility = ["//visibility:private"],
        **kwargs
    )
