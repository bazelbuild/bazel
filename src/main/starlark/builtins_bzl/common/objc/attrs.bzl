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

"""Attributes common to Objc rules"""

load("@_builtins//:common/objc/semantics.bzl", "semantics")
load(":common/cc/cc_info.bzl", "CcInfo")

# Private attribute required by `objc_internal.expand_toolchain_and_ctx_variables`
_CC_TOOLCHAIN_RULE = {
    "_cc_toolchain": attr.label(
        default = "@" + semantics.get_repo() + "//tools/cpp:current_cc_toolchain",
    ),
}

_COMPILING_RULE = {
    "srcs": attr.label_list(
        allow_files = True,
        flags = ["DIRECT_COMPILE_TIME_INPUT"],
    ),
    "non_arc_srcs": attr.label_list(
        allow_files = True,
        flags = ["DIRECT_COMPILE_TIME_INPUT"],
    ),
    "pch": attr.label(
        allow_single_file = [".pch"],
        flags = ["DIRECT_COMPILE_TIME_INPUT"],
    ),
    "defines": attr.string_list(),
    "enable_modules": attr.bool(),
    "linkopts": attr.string_list(),
    "module_map": attr.label(allow_files = [".modulemap"]),
    "module_name": attr.string(),
    # How many rules use this in the depot?
    "stamp": attr.bool(),
}

_COMPILE_DEPENDENCY_RULE = {
    "hdrs": attr.label_list(
        allow_files = True,
        flags = ["DIRECT_COMPILE_TIME_INPUT"],
    ),
    "textual_hdrs": attr.label_list(
        allow_files = True,
        flags = ["DIRECT_COMPILE_TIME_INPUT"],
    ),
    "includes": attr.string_list(),
    "sdk_includes": attr.string_list(),
    "deps": attr.label_list(
        providers = [CcInfo],
    ),
}

_INCLUDE_SCANNING_RULE = {
    "_grep_includes": attr.label(
        allow_single_file = True,
        cfg = "exec",
        default = "@" + semantics.get_repo() + "//tools/cpp:grep-includes",
        executable = True,
    ),
}

_SDK_FRAMEWORK_DEPENDER_RULE = {
    "sdk_frameworks": attr.string_list(),
    "weak_sdk_frameworks": attr.string_list(),
    "sdk_dylibs": attr.string_list(),
}

_COPTS_RULE = {
    "copts": attr.string_list(),
}

_ALWAYSLINK_RULE = {
    "alwayslink": attr.bool(),
}

def _union(*dictionaries):
    result = {}
    for dictionary in dictionaries:
        result.update(dictionary)
    return result

common_attrs = struct(
    union = _union,
    ALWAYSLINK_RULE = _ALWAYSLINK_RULE,
    CC_TOOLCHAIN_RULE = _CC_TOOLCHAIN_RULE,
    COMPILING_RULE = _COMPILING_RULE,
    COMPILE_DEPENDENCY_RULE = _COMPILE_DEPENDENCY_RULE,
    COPTS_RULE = _COPTS_RULE,
    INCLUDE_SCANNING_RULE = _INCLUDE_SCANNING_RULE,
    LICENSES = semantics.get_licenses_attr(),
    SDK_FRAMEWORK_DEPENDER_RULE = _SDK_FRAMEWORK_DEPENDER_RULE,
)
