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

load(":common/java/java_common.bzl", "JAVA_COMMON_DEP", "collect_resources", "construct_defaultinfo")
load(":common/rule_util.bzl", "create_rule")
load(":common/java/java_semantics.bzl", "semantics")
load(":common/java/proguard_validation.bzl", "VALIDATE_PROGUARD_SPECS")

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
    base_info = JAVA_COMMON_DEP.call(
        ctx,
        srcs = srcs,
        resources = resources,
        plugins = plugins,
        deps = deps,
        javacopts = javacopts,
        neverlink = neverlink,
    )

    proguard_specs_provider = VALIDATE_PROGUARD_SPECS.call(ctx, proguard_specs = proguard_specs, transitive_attrs = [deps, plugins])
    base_info.output_groups["_hidden_top_level_INTERNAL_"] = proguard_specs_provider.specs
    base_info.extra_providers["ProguardSpecProvider"] = proguard_specs_provider

    java_info, extra_files = semantics.postprocess_plugin(ctx, base_info)

    java_plugin_info = JavaPluginInfo(
        runtime_deps = [java_info],
        processor_class = processor_class if processor_class else None,  # ignore empty string (default)
        data = data,
        generates_api = generates_api,
    )

    default_info = construct_defaultinfo(
        ctx,
        base_info.files_to_build + extra_files,
        neverlink,
        base_info.has_sources_or_resources,
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
        srcs = ctx.files.srcs,
        data = ctx.files.data,
        generates_api = ctx.attr.generates_api,
        processor_class = ctx.attr.processor_class,
        deps = ctx.attr.deps,
        plugins = ctx.attr.plugins,
        resources = collect_resources(ctx),
        javacopts = ctx.attr.javacopts,
        neverlink = ctx.attr.neverlink,
        proguard_specs = ctx.files.proguard_specs,
    ).values()

java_plugin = create_rule(
    _proxy,
    attrs = dict(
        {
            "generates_api": attr.bool(),
            "processor_class": attr.string(),
            "licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
            "output_licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
        },
        **semantics.EXTRA_PLUGIN_ATTRIBUTES
    ),
    deps = [
        JAVA_COMMON_DEP,
        VALIDATE_PROGUARD_SPECS,
    ],
    provides = [JavaPluginInfo],
    outputs = {
        "classjar": "lib%{name}.jar",
        "sourcejar": "lib%{name}-src.jar",
    },
    remove_attrs = ["runtime_deps", "exports", "exported_plugins"],
)
