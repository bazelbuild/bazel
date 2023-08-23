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

load(":common/java/basic_java_library.bzl", "BASIC_JAVA_LIBRARY_IMPLICIT_ATTRS", "basic_java_library", "construct_defaultinfo")
load(":common/rule_util.bzl", "merge_attrs")
load(":common/java/java_semantics.bzl", "semantics")
load(":common/cc/cc_info.bzl", "CcInfo")
load(":common/java/java_info.bzl", "JavaInfo", "JavaPluginInfo")

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
        proguard_specs = [],
        add_exports = [],
        add_opens = []):
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
      add_exports: (list[str]) Allow this library to access the given <module>/<package>.
      add_opens: (list[str]) Allow this library to reflectively access the given <module>/<package>.
    Returns:
      (dict[str, provider]) A list containing DefaultInfo, JavaInfo,
        InstrumentedFilesInfo, OutputGroupsInfo, ProguardSpecProvider providers.
    """
    if not srcs and deps:
        fail("deps not allowed without srcs; move to runtime_deps?")

    target, base_info = basic_java_library(
        ctx,
        srcs,
        deps,
        runtime_deps,
        plugins,
        exports,
        exported_plugins,
        resources,
        [],  # resource_jars
        [],  # class_pathresources
        javacopts,
        neverlink,
        proguard_specs = proguard_specs,
        add_exports = add_exports,
        add_opens = add_opens,
    )

    target["DefaultInfo"] = construct_defaultinfo(
        ctx,
        base_info.files_to_build,
        base_info.runfiles,
        neverlink,
        exports,
        runtime_deps,
    )
    target["OutputGroupInfo"] = OutputGroupInfo(**base_info.output_groups)

    return target

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
        ctx.attr.add_exports,
        ctx.attr.add_opens,
    ).values()

JAVA_LIBRARY_IMPLICIT_ATTRS = BASIC_JAVA_LIBRARY_IMPLICIT_ATTRS

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
        "add_exports": attr.string_list(),
        "add_opens": attr.string_list(),
        "licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
        "_java_toolchain_type": attr.label(default = semantics.JAVA_TOOLCHAIN_TYPE),
    },
)

java_library = rule(
    _proxy,
    attrs = merge_attrs(
        JAVA_LIBRARY_ATTRS,
        {"_use_auto_exec_groups": attr.bool(default = True)},
    ),
    provides = [JavaInfo],
    outputs = {
        "classjar": "lib%{name}.jar",
        "sourcejar": "lib%{name}-src.jar",
    },
    fragments = ["java", "cpp"],
    toolchains = [semantics.JAVA_TOOLCHAIN],
)
