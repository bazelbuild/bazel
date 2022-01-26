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

def _java_plugin_rule_impl(ctx):
    semantics.check_rule(ctx)
    semantics.check_dependency_rule_kinds(ctx, "java_plugin")

    extra_resources = semantics.preprocess(ctx)

    base_info = JAVA_COMMON_DEP.call(
        ctx,
        srcs = ctx.files.srcs,
        deps = ctx.attr.deps,
        resources = collect_resources(ctx, extra_resources),
        plugins = ctx.attr.plugins,
        javacopts = ctx.attr.javacopts,
        neverlink = ctx.attr.neverlink,
    )

    proguard_specs_provider = VALIDATE_PROGUARD_SPECS.call(ctx)
    base_info.output_groups["_hidden_top_level_INTERNAL_"] = proguard_specs_provider.specs
    base_info.extra_providers.append(proguard_specs_provider)

    java_info, extra_files = semantics.postprocess_plugin(ctx, base_info)

    java_plugin_info = JavaPluginInfo(
        runtime_deps = [java_info],
        processor_class = ctx.attr.processor_class if ctx.attr.processor_class else None,  # ignore empty string (default)
        data = ctx.files.data,
        generates_api = ctx.attr.generates_api,
    )

    default_info = construct_defaultinfo(
        ctx,
        base_info.files_to_build + extra_files,
        ctx.attr.neverlink,
        base_info.has_sources_or_resources,
    )

    return [
        default_info,
        java_plugin_info,
        base_info.instrumented_files_info,
        OutputGroupInfo(**base_info.output_groups),
    ] + base_info.extra_providers

java_plugin = create_rule(
    _java_plugin_rule_impl,
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
    ] + semantics.EXTRA_PLUGIN_DEPS,
    provides = [JavaPluginInfo],
    outputs = {
        "classjar": "lib%{name}.jar",
        "sourcejar": "lib%{name}-src.jar",
    },
    remove_attrs = ["runtime_deps", "exports", "exported_plugins"],
)
