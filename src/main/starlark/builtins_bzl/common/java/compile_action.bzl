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
Java compile action
"""

load(":common/rule_util.bzl", "create_dep")
load(":common/java/java_semantics.bzl", "semantics")

java_common = _builtins.toplevel.java_common
ProtoInfo = _builtins.toplevel.ProtoInfo
DefaultInfo = _builtins.toplevel.DefaultInfo
CcInfo = _builtins.toplevel.CcInfo
JavaInfo = _builtins.toplevel.JavaInfo
JavaPluginInfo = _builtins.toplevel.JavaPluginInfo

def _get_attr_safe(ctx, attr, default):
    return getattr(ctx.attr, attr) if hasattr(ctx.attr, attr) else default

def _filter_srcs(srcs, ext):
    return [f for f in srcs if f.extension == ext]

def _filter_provider(provider, *attrs):
    return [dep[provider] for attr in attrs for dep in attr if provider in dep]

def _filter_strict_deps(mode):
    return "error" if mode in ["strict", "default"] else mode

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

def _collect_plugins(deps, plugins):
    transitive_processor_jars = []
    transitive_processor_data = []
    for plugin in plugins:
        transitive_processor_jars.append(plugin.plugins.processor_jars)
        transitive_processor_data.append(plugin.plugins.processor_data)
    for dep in deps:
        transitive_processor_jars.append(dep.plugins.processor_jars)
        transitive_processor_data.append(dep.plugins.processor_data)
    return struct(
        processor_jars = depset(transitive = transitive_processor_jars),
        processor_data = depset(transitive = transitive_processor_data),
    )

def _compile_action(
        ctx,
        extra_resources,
        classpath_resources,
        source_files,
        source_jars,
        output_prefix,
        enable_compile_jar_action = True,
        extra_runtime_jars = [],
        extra_runtime_deps = [],
        extra_deps = []):
    if extra_deps:
        deps = []
        deps.extend(ctx.attr.deps)
        deps.extend(extra_deps)
    else:
        deps = ctx.attr.deps

    runtime_deps = _get_attr_safe(ctx, "runtime_deps", []) + extra_runtime_deps
    exports = _get_attr_safe(ctx, "exports", [])
    exported_plugins = _get_attr_safe(ctx, "exported_plugins", [])

    srcs = ctx.files.srcs

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

    deps_javainfo = _filter_javainfo_and_legacy_jars(deps)
    runtime_deps_javainfo = _filter_javainfo_and_legacy_jars(runtime_deps)
    runtime_deps_javainfo.extend([JavaInfo(jar, None) for jar in extra_runtime_jars])
    exports_javainfo = _filter_javainfo_and_legacy_jars(exports)

    output = ctx.outputs.classjar
    output_source_jar = ctx.outputs.sourcejar

    java_info = java_common.compile(
        ctx,
        source_files = source_files,
        source_jars = source_jars,
        resources = resources,
        classpath_resources = classpath_resources,
        plugins = plugins,
        deps = deps_javainfo,
        native_libraries = _filter_provider(CcInfo, deps, runtime_deps, exports),
        runtime_deps = runtime_deps_javainfo,
        exports = exports_javainfo,
        exported_plugins = _filter_provider(JavaPluginInfo, exported_plugins),
        javac_opts = [ctx.expand_location(opt) for opt in ctx.attr.javacopts],
        neverlink = ctx.attr.neverlink,
        java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],
        output = output,
        output_source_jar = output_source_jar,
        strict_deps = _filter_strict_deps(ctx.fragments.java.strict_java_deps),
        enable_compile_jar_action = enable_compile_jar_action,
    )

    files = [out.class_jar for out in java_info.java_outputs]
    files_depset = depset(files)

    if ctx.attr.neverlink:
        runfiles = None
    else:
        has_sources = source_files or source_jars
        run_files = files_depset if has_sources or resources else None
        runfiles = ctx.runfiles(transitive_files = run_files, collect_default = True)
        runfiles = runfiles.merge_all([dep[DefaultInfo].default_runfiles for attr in [runtime_deps, exports] for dep in attr])

    default_info = DefaultInfo(
        files = files_depset,
        runfiles = runfiles,
    )

    compilation_info = struct(
        plugins = _collect_plugins(deps_javainfo, plugins),
        outputs = files,
    )

    return java_info, default_info, compilation_info

COMPILE_ACTION = create_dep(
    _compile_action,
    attrs = {
        "srcs": attr.label_list(
            allow_files = [".java", ".srcjar", ".properties"] + semantics.EXTRA_SRCS_TYPES,
            flags = ["DIRECT_COMPILE_TIME_INPUT", "ORDER_INDEPENDENT"],
        ),
        "data": attr.label_list(
            allow_files = True,
            flags = ["SKIP_CONSTRAINTS_OVERRIDE"],
        ),
        "resources": attr.label_list(
            allow_files = True,
            flags = ["SKIP_CONSTRAINTS_OVERRIDE", "ORDER_INDEPENDENT"],
        ),
        "plugins": attr.label_list(
            providers = [JavaPluginInfo],
            allow_files = True,
            cfg = "exec",
        ),
        "deps": attr.label_list(
            allow_files = [".jar"],
            allow_rules = semantics.ALLOWED_RULES_IN_DEPS + semantics.ALLOWED_RULES_IN_DEPS_WITH_WARNING,
            providers = [
                [CcInfo],
                [JavaInfo],
            ],
            flags = ["SKIP_ANALYSIS_TIME_FILETYPE_CHECK"],
        ),
        "runtime_deps": attr.label_list(
            allow_files = [".jar"],
            allow_rules = semantics.ALLOWED_RULES_IN_DEPS,
            providers = [[CcInfo], [JavaInfo]],
            flags = ["SKIP_ANALYSIS_TIME_FILETYPE_CHECK"],
        ),
        "exports": attr.label_list(
            allow_rules = semantics.ALLOWED_RULES_IN_DEPS,
            providers = [[JavaInfo], [CcInfo]],
        ),
        "exported_plugins": attr.label_list(
            providers = [JavaPluginInfo],
            cfg = "exec",
        ),
        "javacopts": attr.string_list(),
        "neverlink": attr.bool(),
        "_java_toolchain": attr.label(
            default = semantics.JAVA_TOOLCHAIN_LABEL,
            providers = [java_common.JavaToolchainInfo],
        ),
        "_java_plugins": attr.label(
            default = semantics.JAVA_PLUGINS_FLAG_ALIAS_LABEL,
            providers = [JavaPluginInfo],
        ),
    },
    fragments = ["java", "cpp"],
    mandatory_attrs = ["srcs", "deps", "resources", "plugins", "javacopts", "neverlink"],
)
