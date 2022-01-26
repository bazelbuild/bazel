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
load(":common/java/java_semantics.bzl", "semantics")

java_common = _builtins.toplevel.java_common
coverage_common = _builtins.toplevel.coverage_common

JavaInfo = _builtins.toplevel.JavaInfo
JavaPluginInfo = _builtins.toplevel.JavaPluginInfo
ProtoInfo = _builtins.toplevel.ProtoInfo
CcInfo = _builtins.toplevel.CcInfo

def _filter_srcs(srcs, ext):
    return [f for f in srcs if f.extension == ext]

def _filter_provider(provider, *attrs):
    return [dep[provider] for attr in attrs for dep in attr if provider in dep]

def _get_attr_safe(ctx, attr, default):
    return getattr(ctx.attr, attr) if hasattr(ctx.attr, attr) else default

# TODO(b/11285003): disallow jar files in deps, require java_import instead
def _filter_javainfo_and_legacy_jars(attr):
    dep_list = []

    # Native code collected data into a NestedSet, using add for legacy jars and
    # addTransitive for JavaInfo. This resulted in legacy jars being first in the list.
    for dep in attr:
        kind = java_common.target_kind(dep)
        if not JavaInfo in dep or kind == "java_binary" or kind == "java_test":
            for file in dep[DefaultInfo].files.to_list():
                if file.extension == "jar":
                    # Native doesn't construct JavaInfo
                    java_info = JavaInfo(output_jar = file, compile_jar = file)
                    dep_list.append(java_info)

    for dep in attr:
        if JavaInfo in dep:
            dep_list.append(dep[JavaInfo])
    return dep_list

def _base_common_impl(
        ctx,
        extra_resources,
        enable_compile_jar_action = True,
        extra_runtime_jars = [],
        classpath_resources = [],
        extra_runtime_deps = [],
        coverage_config = None):
    srcs = ctx.files.srcs
    source_files = _filter_srcs(srcs, "java")
    source_jars = _filter_srcs(srcs, "srcjar")

    resources = []
    if semantics.COLLECT_SRCS_FROM_PROTO_LIBRARY:
        for resource in ctx.attr.resources:
            if ProtoInfo in resource:
                resources.extend(resource[ProtoInfo].transitive_sources.to_list())
            else:
                resources.extend(resource[DefaultInfo].files.to_list())
    else:
        resources.extend(ctx.files.resources)
    resources.extend(_filter_srcs(ctx.files.srcs, "properties"))
    resources.extend(extra_resources)

    plugins = _filter_provider(JavaPluginInfo, ctx.attr.plugins)
    plugins.append(ctx.attr._java_plugins[JavaPluginInfo])

    deps = []
    deps.extend(ctx.attr.deps)
    if coverage_config:
        deps.append(coverage_config.runner)

    runtime_deps = _get_attr_safe(ctx, "runtime_deps", []) + extra_runtime_deps
    exports = _get_attr_safe(ctx, "exports", [])
    exported_plugins = _get_attr_safe(ctx, "exported_plugins", [])

    deps_javainfo = _filter_javainfo_and_legacy_jars(deps)
    runtime_deps_javainfo = _filter_javainfo_and_legacy_jars(runtime_deps)
    runtime_deps_javainfo.extend([JavaInfo(jar, None) for jar in extra_runtime_jars])
    exports_javainfo = _filter_javainfo_and_legacy_jars(exports)

    java_info, compilation_info = COMPILE_ACTION.call(
        ctx,
        output_class_jar = ctx.outputs.classjar,
        output_source_jar = ctx.outputs.sourcejar,
        source_files = source_files,
        source_jars = source_jars,
        deps = deps_javainfo,
        runtime_deps = runtime_deps_javainfo,
        plugins = plugins,
        exports = exports_javainfo,
        exported_plugins = _filter_provider(JavaPluginInfo, exported_plugins),
        resources = resources,
        classpath_resources = classpath_resources,
        native_libraries = _filter_provider(CcInfo, deps, runtime_deps, exports),
        javacopts = ctx.attr.javacopts,
        neverlink = ctx.attr.neverlink,
        strict_deps = ctx.fragments.java.strict_java_deps,
        enable_compile_jar_action = enable_compile_jar_action,
    )

    files = depset(compilation_info.output_class_jars)
    if ctx.attr.neverlink:
        runfiles = None
    else:
        has_sources = source_files or source_jars
        run_files = files if has_sources or resources else None
        runfiles = ctx.runfiles(transitive_files = run_files, collect_default = True)
        runfiles = runfiles.merge_all([dep[DefaultInfo].default_runfiles for attr in [runtime_deps, exports] for dep in attr])
    default_info = DefaultInfo(
        files = files,
        runfiles = runfiles,
    )

    output_groups = dict(
        compilation_outputs = compilation_info.output_class_jars,
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
