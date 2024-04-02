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

"""Rule definition for the available_xcodes rule."""

def _available_xcodes_impl(ctx):
    available_versions = [
        target[_builtins.internal.XcodeVersionRuleData]
        for target in ctx.attr.versions
    ]
    default_version = ctx.attr.default[_builtins.internal.XcodeVersionRuleData]

    return [
        _builtins.internal.AvailableXcodesInfo(
            default = default_version,
            versions = available_versions,
        ),
    ]

available_xcodes = rule(
    attrs = {
        "default": attr.label(
            doc = "The default xcode version for this platform.",
            mandatory = True,
            providers = [[_builtins.internal.XcodeVersionRuleData]],
            flags = ["NONCONFIGURABLE"],
        ),
        "versions": attr.label_list(
            doc = "The xcode versions that are available on this platform.",
            providers = [[_builtins.internal.XcodeVersionRuleData]],
            flags = ["NONCONFIGURABLE"],
        ),
    },
    doc = """\
Two targets of this rule can be depended on by an `xcode_config` rule instance
to indicate the remotely and locally available xcode versions. This allows
selection of an official xcode version from the collectively available xcodes.
        """,
    implementation = _available_xcodes_impl,
)
