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

"""Definition for the `xcode_config_alias` rule.

This rule is an alias to the `xcode_config` rule currently in use, which in turn
depends on the current configuration; in particular, the value of the
`--xcode_version_config`.

This is intentionally undocumented for users; the workspace is expected to
contain exactly one instance of this rule under `@bazel_tools//tools/osx` and
people who want to get data this rule provides should depend on that one.
"""

load(":common/objc/apple_common.bzl", "apple_common")

def _xcode_config_alias_impl(ctx):
    xcode_config = ctx.attr._xcode_config
    return [
        xcode_config[apple_common.XcodeProperties],
        xcode_config[_builtins.internal.apple_common.XcodeVersionConfig],
    ]

xcode_config_alias = rule(
    attrs = {
        "_xcode_config": attr.label(
            default = configuration_field(
                fragment = "apple",
                name = "xcode_config_label",
            ),
        ),
    },
    fragments = ["apple"],
    implementation = _xcode_config_alias_impl,
)
