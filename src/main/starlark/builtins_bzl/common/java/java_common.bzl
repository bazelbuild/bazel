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

"""
Common code for reuse across java_* rules
"""

load(":common/rule_util.bzl", "create_composite_dep")
load(":common/java/android_lint.bzl", "ANDROID_LINT_ACTION")
load(":common/java/compile_action.bzl", "COMPILE_ACTION")

coverage_common = _builtins.toplevel.coverage_common

def _filter_srcs(srcs, ext):
    return [f for f in srcs if f.extension == ext]

def _base_common_impl(
        ctx,
        extra_resources,
        output_prefix,
        enable_compile_jar_action = True,
        extra_runtime_jars = [],
        coverage_config = None):
    srcs = ctx.files.srcs
    source_files = _filter_srcs(srcs, "java")
    source_jars = _filter_srcs(srcs, "srcjar")

    java_info, default_info, compilation_info = COMPILE_ACTION.call(
        ctx,
        extra_resources,
        source_files,
        source_jars,
        output_prefix,
        enable_compile_jar_action,
        extra_runtime_jars,
        extra_deps = [coverage_config.runner] if coverage_config else [],
    )
    output_groups = dict(
        compilation_outputs = compilation_info.outputs,
        _source_jars = java_info.transitive_source_jars,
        _direct_source_jars = java_info.source_jars,
    )

    lint_output = ANDROID_LINT_ACTION.call(ctx, java_info, source_files, source_jars, compilation_info)
    if lint_output:
        output_groups["_validation"] = [lint_output]

    instrumented_files_info = coverage_common.instrumented_files_info(
        ctx,
        source_attributes = ["srcs"],
        dependency_attributes = ["deps", "data", "resources", "resource_jars", "exports", "runtime_deps", "jars"],
        coverage_support_files = coverage_config.support_files if coverage_config else depset(),
        coverage_environment = coverage_config.env if coverage_config else {},
    )

    return struct(
        java_info = java_info,
        default_info = default_info,
        instrumented_files_info = instrumented_files_info,
        output_groups = output_groups,
        extra_providers = [],
    )

JAVA_COMMON_DEP = create_composite_dep(
    _base_common_impl,
    COMPILE_ACTION,
    ANDROID_LINT_ACTION,
)
