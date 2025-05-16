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

"""Temporary implementation of a rule that aliases the value of --java_launcher flag"""

load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

_providers = (
    [CcInfo, getattr(cc_common, "launcher_provider")] if hasattr(cc_common, "launcher_provider") else [CcInfo]
)

def _impl(ctx):
    if not ctx.attr._launcher:
        return None
    launcher = ctx.attr._launcher
    providers = [ctx.attr._launcher[p] for p in _providers]
    providers.append(DefaultInfo(files = launcher[DefaultInfo].files, runfiles = launcher[DefaultInfo].default_runfiles))
    return providers

launcher_flag_alias = rule(
    implementation = _impl,
    attrs = {
        "_launcher": attr.label(
            default = configuration_field(
                fragment = "java",
                name = "launcher",
            ),
            providers = _providers,
        ),
    },
)
