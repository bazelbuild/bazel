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

"""A Starlark implementation of the java_lite_proto_library rule."""

load(":common/proto/proto_info.bzl", "ProtoInfo")
load(":common/java/java_semantics.bzl", "semantics")
load(":common/proto/proto_common.bzl", "ProtoLangToolchainInfo", proto_common = "proto_common_do_not_use")
load(":common/java/proto/java_proto_library.bzl", "JavaProtoAspectInfo", "bazel_java_proto_library_rule", "java_compile_for_protos")
load(":common/java/java_info.bzl", "JavaInfo")
load(":common/java/java_common.bzl", "java_common")

PROTO_TOOLCHAIN_ATTR = "_aspect_proto_toolchain_for_javalite"

ProguardSpecProvider = _builtins.toplevel.ProguardSpecProvider

def _aspect_impl(target, ctx):
    """Generates and compiles Java code for a proto_library dependency graph.

    Args:
      target: (Target) The `proto_library` target.
      ctx: (RuleContext) The rule context.

    Returns:
      ([JavaInfo, JavaProtoAspectInfo]) A JavaInfo describing compiled Java
      version of`proto_library` and `JavaProtoAspectInfo` with all source and
      runtime jars.
    """

    deps = [dep[JavaInfo] for dep in ctx.rule.attr.deps]
    exports = [exp[JavaInfo] for exp in ctx.rule.attr.exports]
    proto_toolchain_info = ctx.attr._aspect_proto_toolchain_for_javalite[ProtoLangToolchainInfo]
    source_jar = None

    if proto_common.experimental_should_generate_code(target[ProtoInfo], proto_toolchain_info, "java_lite_proto_library", target.label):
        source_jar = ctx.actions.declare_file(ctx.label.name + "-lite-src.jar")
        proto_common.compile(
            ctx.actions,
            target[ProtoInfo],
            proto_toolchain_info,
            [source_jar],
            experimental_output_files = "single",
        )
        runtime = proto_toolchain_info.runtime
        if runtime:
            deps.append(runtime[JavaInfo])

    java_info, jars = java_compile_for_protos(
        ctx,
        "-lite.jar",
        source_jar,
        deps,
        exports,
        injecting_rule_kind = "java_lite_proto_library",
    )
    transitive_jars = [dep[JavaProtoAspectInfo].jars for dep in ctx.rule.attr.deps]

    return [
        java_info,
        JavaProtoAspectInfo(jars = depset(jars, transitive = transitive_jars)),
    ]

java_lite_proto_aspect = aspect(
    implementation = _aspect_impl,
    attr_aspects = ["deps", "exports"],
    attrs = {
        PROTO_TOOLCHAIN_ATTR: attr.label(
            default = configuration_field(fragment = "proto", name = "proto_toolchain_for_java_lite"),
        ),
    },
    fragments = ["java"],
    required_providers = [ProtoInfo],
    provides = [JavaInfo, JavaProtoAspectInfo],
    toolchains = [semantics.JAVA_TOOLCHAIN],
)

def _rule_impl(ctx):
    """Merges results of `java_proto_aspect` in `deps`.

    `java_lite_proto_library` is identical to `java_proto_library` in every respect, except it
    builds JavaLite protos.
    Implementation of this rule is built on the implementation of `java_proto_library`.

    Args:
      ctx: (RuleContext) The rule context.
    Returns:
      ([JavaInfo, DefaultInfo, OutputGroupInfo, ProguardSpecProvider])
    """

    proto_toolchain_info = ctx.attr._aspect_proto_toolchain_for_javalite[ProtoLangToolchainInfo]
    for dep in ctx.attr.deps:
        proto_common.check_collocated(ctx.label, dep[ProtoInfo], proto_toolchain_info)

    runtime = proto_toolchain_info.runtime

    if runtime:
        proguard_provider_specs = runtime[ProguardSpecProvider]
    else:
        proguard_provider_specs = ProguardSpecProvider(depset())

    java_info, DefaultInfo, OutputGroupInfo = bazel_java_proto_library_rule(ctx)

    if hasattr(java_common, "add_constraints"):
        java_info = java_common.add_constraints(java_info, constraints = ["android"])

    return [
        java_info,
        DefaultInfo,
        OutputGroupInfo,
        proguard_provider_specs,
    ]

java_lite_proto_library = rule(
    implementation = _rule_impl,
    attrs = {
        "deps": attr.label_list(providers = [ProtoInfo], aspects = [java_lite_proto_aspect]),
        PROTO_TOOLCHAIN_ATTR: attr.label(
            default = configuration_field(fragment = "proto", name = "proto_toolchain_for_java_lite"),
        ),
    },
    provides = [JavaInfo],
)
