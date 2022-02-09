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
Definition of java_plugin rule.
"""

load(":common/java/java_common.bzl", "basic_java_library", "construct_defaultinfo")
load(":common/java/java_library.bzl", "JAVA_LIBRARY_ATTRS", "JAVA_LIBRARY_IMPLICIT_ATTRS")
load(":common/rule_util.bzl", "merge_attrs")
load(":common/java/proguard_validation.bzl", "validate_proguard_specs")

JavaPluginInfo = _builtins.toplevel.JavaPluginInfo

def bazel_java_plugin_rule(
        ctx,
        srcs = [],
        data = [],
        generates_api = False,
        processor_class = "",
        deps = [],
        plugins = [],
        resources = [],
        javacopts = [],
        neverlink = False,
        proguard_specs = []):
    """Implements java_plugin rule.

    Use this call when you need to produce a fully fledged java_plugin from
    another rule's implementation.

    Args:
      ctx: (RuleContext) Used to register the actions.
      srcs: (list[File]) The list of source files that are processed to create the target.
      data: (list[File]) The list of files needed by this plugin at runtime.
      generates_api: (bool) This attribute marks annotation processors that generate API code.
      processor_class: (str) The processor class is the fully qualified type of
        the class that the Java compiler should use as entry point to the annotation processor.
      deps: (list[Target]) The list of other libraries to be linked in to the target.
      plugins: (list[Target]) Java compiler plugins to run at compile-time.
      resources: (list[File]) A list of data files to include in a Java jar.
      javacopts: (list[str]) Extra compiler options for this library.
      neverlink: (bool) Whether this library should only be used for compilation and not at runtime.
      proguard_specs: (list[File]) Files to be used as Proguard specification.
    Returns:
      (list[provider]) A list containing DefaultInfo, JavaInfo,
        InstrumentedFilesInfo, OutputGroupsInfo, ProguardSpecProvider providers.
    """
    base_info = basic_java_library(
        ctx,
        srcs,
        deps,
        [],  # runtime_deps
        plugins,
        [],  # exports
        [],  # exported_plugins
        resources,
        [],  # classpath_resources
        javacopts,
        neverlink,
    )

    proguard_specs_provider = validate_proguard_specs(ctx, proguard_specs, [deps, plugins])
    base_info.output_groups["_hidden_top_level_INTERNAL_"] = proguard_specs_provider.specs
    base_info.extra_providers["ProguardSpecProvider"] = proguard_specs_provider

    java_plugin_info = JavaPluginInfo(
        runtime_deps = [base_info.java_info],
        processor_class = processor_class if processor_class else None,  # ignore empty string (default)
        data = data,
        generates_api = generates_api,
    )

    default_info = construct_defaultinfo(
        ctx,
        base_info.files_to_build,
        base_info.runfiles,
        neverlink,
    )

    return dict({
        "DefaultInfo": default_info,
        "JavaPluginInfo": java_plugin_info,
        "InstrumentedFilesInfo": base_info.instrumented_files_info,
        "OutputGroupInfo": OutputGroupInfo(**base_info.output_groups),
    }, **base_info.extra_providers)

def _proxy(ctx):
    return bazel_java_plugin_rule(
        ctx,
        ctx.files.srcs,
        ctx.files.data,
        ctx.attr.generates_api,
        ctx.attr.processor_class,
        ctx.attr.deps,
        ctx.attr.plugins,
        ctx.files.resources,
        ctx.attr.javacopts,
        ctx.attr.neverlink,
        ctx.files.proguard_specs,
    ).values()

JAVA_PLUGIN_ATTRS = merge_attrs(
    JAVA_LIBRARY_ATTRS,
    {
        "generates_api": attr.bool(),
        "processor_class": attr.string(),
        "output_licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
    },
    remove_attrs = ["runtime_deps", "exports", "exported_plugins"],
)

JAVA_PLUGIN_IMPLICIT_ATTRS = JAVA_LIBRARY_IMPLICIT_ATTRS

java_plugin = rule(
    _proxy,
    attrs = merge_attrs(
        JAVA_PLUGIN_ATTRS,
        JAVA_PLUGIN_IMPLICIT_ATTRS,
    ),
    provides = [JavaPluginInfo],
    outputs = {
        "classjar": "lib%{name}.jar",
        "sourcejar": "lib%{name}-src.jar",
    },
    fragments = ["java", "cpp"],
)
