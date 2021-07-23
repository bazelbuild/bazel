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
load(
    ":common/java/java_semantics.bzl",
    "ALLOWED_RULES_IN_DEPS",
    "ALLOWED_RULES_IN_DEPS_WITH_WARNING",
    "COLLECT_SRCS_FROM_PROTO_LIBRARY",
    "EXPERIMENTAL_USE_FILEGROUPS_IN_JAVALIBRARY",
    "EXTRA_SRCS_TYPES",
)

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
def _append_legacy_jars(attr, dep_list):
    for dep in attr:
        if not JavaInfo in dep or dep.kind == "java_binary" or dep.kind == "java_test":
            for file in dep[DefaultInfo].files.to_list():
                if file.extension == "jar":
                    # Native doesn't construct JavaInfo
                    java_info = JavaInfo(output_jar = file, compile_jar = file)
                    dep_list.append(java_info)

def _compile_action(ctx, extra_resources, source_files, source_jars, output_prefix):
    deps = ctx.attr.deps
    runtime_deps = _get_attr_safe(ctx, "runtime_deps", [])
    exports = _get_attr_safe(ctx, "exports", [])
    exported_plugins = _get_attr_safe(ctx, "exported_plugins", [])

    srcs = ctx.files.srcs

    resources = []
    if COLLECT_SRCS_FROM_PROTO_LIBRARY:
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

    deps_javainfo = _filter_provider(JavaInfo, deps)
    _append_legacy_jars(deps, deps_javainfo)
    runtime_deps_javainfo = _filter_provider(JavaInfo, runtime_deps)
    _append_legacy_jars(runtime_deps, runtime_deps_javainfo)
    exports_javainfo = _filter_provider(JavaInfo, exports)
    _append_legacy_jars(exports, exports_javainfo)

    java_info = java_common.compile(
        ctx,
        source_files = source_files,
        source_jars = source_jars,
        resources = resources,
        plugins = plugins,
        deps = deps_javainfo,
        native_libraries = _filter_provider(CcInfo, deps, runtime_deps, exports),
        runtime_deps = runtime_deps_javainfo,
        exports = exports_javainfo,
        exported_plugins = _filter_provider(JavaPluginInfo, exported_plugins),
        javac_opts = [ctx.expand_location(opt) for opt in ctx.attr.javacopts],
        neverlink = ctx.attr.neverlink,
        java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],
        output = ctx.actions.declare_file(output_prefix + "%s.jar" % ctx.attr.name) if EXPERIMENTAL_USE_FILEGROUPS_IN_JAVALIBRARY else ctx.outputs.classjar,
        output_source_jar = ctx.actions.declare_file(output_prefix + "%s-src.jar" % ctx.attr.name) if EXPERIMENTAL_USE_FILEGROUPS_IN_JAVALIBRARY else ctx.outputs.sourcejar,
        strict_deps = _filter_strict_deps(ctx.fragments.java.strict_java_deps),
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
    return java_info, default_info, files

COMPILE_ACTION = create_dep(
    _compile_action,
    {
        "srcs": attr.label_list(
            allow_files = [".java", ".srcjar", ".properties"] + EXTRA_SRCS_TYPES,
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
            allow_rules = ALLOWED_RULES_IN_DEPS + ALLOWED_RULES_IN_DEPS_WITH_WARNING,
            providers = [
                [CcInfo],
                [JavaInfo],
            ],
            flags = ["SKIP_ANALYSIS_TIME_FILETYPE_CHECK"],
        ),
        "javacopts": attr.string_list(),
        "neverlink": attr.bool(),
        "_java_toolchain": attr.label(
            default = "@//tools/jdk:current_java_toolchain",
            providers = [java_common.JavaToolchainInfo],
        ),
        "_java_plugins": attr.label(
            default = "@//tools/jdk:java_plugins_flag_alias",
            providers = [JavaPluginInfo],
        ),
    },
    ["java", "google_java", "cpp"],
)
