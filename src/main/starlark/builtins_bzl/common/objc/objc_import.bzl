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

"""objc_import Starlark implementation replacing native"""

load("@_builtins//:common/objc/semantics.bzl", "semantics")
load("@_builtins//:common/objc/attrs.bzl", "common_attrs")

objc_internal = _builtins.internal.objc_internal
CcInfo = _builtins.toplevel.CcInfo

def _objc_import_impl(ctx):
    compilation_attributes = objc_internal.create_compilation_attributes(ctx = ctx)
    intermediate_artifacts = objc_internal.create_intermediate_artifacts(ctx = ctx)
    common = objc_internal.create_common(
        purpose = "COMPILE_AND_LINK",
        ctx = ctx,
        compilation_attributes = compilation_attributes,
        deps = ctx.attr.deps,
        intermediate_artifacts = intermediate_artifacts,
        alwayslink = ctx.attr.alwayslink,
        has_module_map = True,
        extra_import_libraries = ctx.files.archives,
    )

    compilation_support = objc_internal.create_compilation_support(
        ctx = ctx,
        semantics = semantics.get_semantics(),
    )

    compilation_support.register_compile_and_archive_actions(common = common)
    compilation_support.validate_attributes()

    return [
        CcInfo(compilation_context = compilation_support.compilation_context),
        common.objc_provider,
    ]

objc_import = rule(
    implementation = _objc_import_impl,
    attrs = common_attrs.union(
        {
            "archives": attr.label_list(allow_empty = False, mandatory = True, allow_files = [".a"]),
            "_cc_toolchain": attr.label(
                default = "@" + semantics.get_repo() + "//tools/cpp:current_cc_toolchain",
            ),
        },
        common_attrs.COMPILING_RULE,
        common_attrs.COMPILE_DEPENDENCY_RULE,
        common_attrs.INCLUDE_SCANNING_RULE,
        common_attrs.SDK_FRAMEWORK_DEPENDER_RULE,
        common_attrs.COPTS_RULE,
        common_attrs.X_C_RUNE_RULE,
    ),
    fragments = ["objc", "apple", "cpp"],
    toolchains = ["@" + semantics.get_repo() + "//tools/cpp:toolchain_type"],
    incompatible_use_toolchain_transition = True,
)
