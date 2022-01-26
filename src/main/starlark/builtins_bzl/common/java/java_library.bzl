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
Definition of java_library rule.
"""

load(":common/java/java_common.bzl", "BASIC_JAVA_LIBRARY_IMPLICIT_ATTRS", "basic_java_library", "construct_defaultinfo")
load(":common/rule_util.bzl", "merge_attrs")
load(":common/java/java_semantics.bzl", "semantics")
load(":common/java/proguard_validation.bzl", "VALIDATE_PROGUARD_SPECS_IMPLICIT_ATTRS", "validate_proguard_specs")

JavaInfo = _builtins.toplevel.JavaInfo
JavaPluginInfo = _builtins.toplevel.JavaPluginInfo
CcInfo = _builtins.toplevel.CcInfo

def bazel_java_library_rule(
        ctx,
        srcs = [],
        deps = [],
        runtime_deps = [],
        plugins = [],
        exports = [],
        exported_plugins = [],
        resources = [],
        javacopts = [],
        neverlink = False,
        proguard_specs = []):
    """Implements java_library.

    Use this call when you need to produce a fully fledged java_library from
    another rule's implementation.

    Args:
      ctx: (RuleContext) Used to register the actions.
      srcs: (list[File]) The list of source files that are processed to create the target.
      deps: (list[Target]) The list of other libraries to be linked in to the target.
      runtime_deps: (list[Target]) Libraries to make available to the final binary or test at runtime only.
      plugins: (list[Target]) Java compiler plugins to run at compile-time.
      exports: (list[Target]) Exported libraries.
      exported_plugins: (list[Target]) The list of `java_plugin`s (e.g. annotation
        processors) to export to libraries that directly depend on this library.
      resources: (list[File]) A list of data files to include in a Java jar.
      javacopts: (list[str]) Extra compiler options for this library.
      neverlink: (bool) Whether this library should only be used for compilation and not at runtime.
      proguard_specs: (list[File]) Files to be used as Proguard specification.
    Returns:
      (list[provider]) A list containing DefaultInfo, JavaInfo,
        InstrumentedFilesInfo, OutputGroupsInfo, ProguardSpecProvider providers.
    """
    if not srcs and deps:
        fail("deps not allowed without srcs; move to runtime_deps?")

    base_info = basic_java_library(
        ctx,
        srcs,
        deps,
        runtime_deps,
        plugins,
        exports,
        exported_plugins,
        resources,
        [],  # class_pathresources
        javacopts,
        neverlink,
    )

    proguard_specs_provider = validate_proguard_specs(
        ctx,
        proguard_specs,
        [deps, runtime_deps, exports, plugins, exported_plugins],
    )
    base_info.output_groups["_hidden_top_level_INTERNAL_"] = proguard_specs_provider.specs
    base_info.extra_providers["ProguardSpecProvider"] = proguard_specs_provider

    default_info = construct_defaultinfo(
        ctx,
        base_info.files_to_build,
        neverlink,
        base_info.has_sources_or_resources,
        exports,
        runtime_deps,
    )

    return dict({
        "DefaultInfo": default_info,
        "JavaInfo": base_info.java_info,
        "InstrumentedFilesInfo": base_info.instrumented_files_info,
        "OutputGroupInfo": OutputGroupInfo(**base_info.output_groups),
    }, **base_info.extra_providers)

def _proxy(ctx):
    return bazel_java_library_rule(
        ctx,
        ctx.files.srcs,
        ctx.attr.deps,
        ctx.attr.runtime_deps,
        ctx.attr.plugins,
        ctx.attr.exports,
        ctx.attr.exported_plugins,
        ctx.files.resources,
        ctx.attr.javacopts,
        ctx.attr.neverlink,
        ctx.files.proguard_specs,
    ).values()

JAVA_LIBRARY_IMPLICIT_ATTRS = merge_attrs(
    BASIC_JAVA_LIBRARY_IMPLICIT_ATTRS,
    VALIDATE_PROGUARD_SPECS_IMPLICIT_ATTRS,
)

JAVA_LIBRARY_ATTRS = merge_attrs(
    JAVA_LIBRARY_IMPLICIT_ATTRS,
    {
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
        "resource_strip_prefix": attr.string(),
        "proguard_specs": attr.label_list(allow_files = True),
        "licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
    },
)

java_library = rule(
    _proxy,
    attrs = JAVA_LIBRARY_ATTRS,
    provides = [JavaInfo],
    outputs = {
        "classjar": "lib%{name}.jar",
        "sourcejar": "lib%{name}-src.jar",
    },
    fragments = ["java", "cpp"],
    compile_one_filetype = [".java"],
)
