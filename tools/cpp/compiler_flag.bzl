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

"""Rule that allows select() to differentiate between compilers."""

load("//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")

def _compiler_flag_impl(ctx):
    toolchain = find_cpp_toolchain(ctx)
    return [config_common.FeatureFlagInfo(value = toolchain.compiler)]

compiler_flag = rule(
    implementation = _compiler_flag_impl,
    attrs = {
        "_cc_toolchain": attr.label(default = Label("//tools/cpp:current_cc_toolchain")),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
)
