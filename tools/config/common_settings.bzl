# Copyright 2019 The Bazel Authors. All rights reserved.
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
"""Common build setting rules for use in Bazel tools

Warning: The content of this file is copied from
https://github.com/bazelbuild/bazel-skylib/rules/common_settings.bzl
We maintain this private copy only to prevent taking a dependency on skylib.
Please do not innovate make modifications here.

These rules return a BuildSettingInfo with the value of the build setting.
For label-typed settings, use the native label_flag and label_setting rules.

More documentation on how to use build settings at
https://docs.bazel.build/versions/master/skylark/config.html#user-defined-build-settings
"""

BuildSettingInfo = provider(
    doc = "A singleton provider that contains the raw value of a build setting",
    fields = {
        "value": "The value of the build setting in the current configuration. " +
                 "This value may come from the command line or an upstream transition, " +
                 "or else it will be the build setting's default.",
    },
)

def _impl(ctx):
    return BuildSettingInfo(value = ctx.build_setting_value)

int_flag = rule(
    implementation = _impl,
    build_setting = config.int(flag = True),
    doc = "An int-typed build setting that can be set on the command line",
)

int_setting = rule(
    implementation = _impl,
    build_setting = config.int(),
    doc = "An int-typed build setting that cannot be set on the command line",
)

bool_flag = rule(
    implementation = _impl,
    build_setting = config.bool(flag = True),
    doc = "A bool-typed build setting that can be set on the command line",
)

bool_setting = rule(
    implementation = _impl,
    build_setting = config.bool(),
    doc = "A bool-typed build setting that cannot be set on the command line",
)

string_list_flag = rule(
    implementation = _impl,
    build_setting = config.string_list(flag = True),
    doc = "A string list-typed build setting that can be set on the command line",
)

string_list_setting = rule(
    implementation = _impl,
    build_setting = config.string_list(),
    doc = "A string list-typed build setting that cannot be set on the command line",
)

def _string_impl(ctx):
    allowed_values = ctx.attr.values
    value = ctx.build_setting_value
    if len(allowed_values) == 0 or value in ctx.attr.values:
        return BuildSettingInfo(value = value)
    else:
        fail("Error setting " + str(ctx.label) + ": invalid value '" + value + "'. Allowed values are " + str(allowed_values))

string_flag = rule(
    implementation = _string_impl,
    build_setting = config.string(flag = True),
    attrs = {
        "values": attr.string_list(
            doc = "The list of allowed values for this setting. An error is raised if any other value is given.",
        ),
    },
    doc = "A string-typed build setting that can be set on the command line",
)

string_setting = rule(
    implementation = _string_impl,
    build_setting = config.string(),
    attrs = {
        "values": attr.string_list(
            doc = "The list of allowed values for this setting. An error is raised if any other value is given.",
        ),
    },
    doc = "A string-typed build setting that cannot be set on the command line",
)
