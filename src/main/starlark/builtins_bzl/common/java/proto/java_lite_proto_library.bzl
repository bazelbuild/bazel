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

load(":common/java/java_semantics.bzl", "semantics")
load(":common/proto/proto_common.bzl", "ProtoLangToolchainInfo", proto_common = "proto_common_do_not_use")

PROTO_TOOLCHAIN_ATTR = "_aspect_proto_toolchain_for_javalite"
PROTO_JAVACOPTS_KEY = "proto"
JAVA_TOOLCHAIN_ATTR = "_java_toolchain"

java_common = _builtins.toplevel.java_common
ProtoInfo = _builtins.toplevel.ProtoInfo
JavaInfo = _builtins.toplevel.JavaInfo
ProguardSpecProvider = _builtins.toplevel.ProguardSpecProvider

_JavaProtoAspectInfo = provider("JavaProtoAspectInfo", fields = ["jars"])

def _rule_impl(ctx):
    proto_toolchain_info = ctx.attr._aspect_proto_toolchain_for_javalite[ProtoLangToolchainInfo]
    runtime = proto_toolchain_info.runtime

    if runtime:
        proguard_provider_specs = runtime[ProguardSpecProvider]
    else:
        proguard_provider_specs = ProguardSpecProvider(depset())

    # Merging the retrieved list of aspect providers from the dependencies and runtime JavaInfo providers.
    java_info = java_common.merge(
        [],
        exports = [dep[JavaInfo] for dep in ctx.attr.deps],
        include_source_jars_from_exports = True,
    )

    # Collect the aspect output files.
    files_to_build = depset(transitive = [dep[_JavaProtoAspectInfo].jars for dep in ctx.attr.deps])

    java_info = semantics.add_constraints(java_info, ["android"])

    return [
        DefaultInfo(
            files = files_to_build,  # files to build
            runfiles = ctx.runfiles(
                transitive_files = depset(transitive = [java_info.transitive_runtime_jars]),
            ),
        ),
        java_info,
        OutputGroupInfo(default = depset()),
        proguard_provider_specs,
    ]

def _aspect_impl(target, ctx):
    """Stub aspect that collect all the proto source jars.

    Args:
      target: The configured target.
      ctx: The rule context.

    Returns:
      A source_jars_provider containing all the propagated dependency source jars.
    """
    transitive_files_to_build = [dep[_JavaProtoAspectInfo].jars for dep in ctx.rule.attr.deps]
    files_to_build = []

    # Collect the dependencies' aspect providers.
    deps = [dep[JavaInfo] for dep in ctx.rule.attr.deps]

    # Collect the exports' aspect providers.
    exports = [exp[JavaInfo] for exp in ctx.rule.attr.exports]
    proto_toolchain_info = ctx.attr._aspect_proto_toolchain_for_javalite[ProtoLangToolchainInfo]
    if proto_common.experimental_should_generate_code(target, proto_toolchain_info, "java_lite_proto_library"):
        source_jar = ctx.actions.declare_file(ctx.label.name + "-lite-src.jar")
        proto_common.compile(
            ctx.actions,
            target,
            proto_toolchain_info,
            [source_jar],
            source_jar,
        )

        output_jar = ctx.actions.declare_file("lib" + ctx.label.name + "-lite.jar")

        files_to_build.extend([source_jar, output_jar])

        # This returns a java provider or None.
        runtime = proto_toolchain_info.runtime
        if runtime:
            deps.append(runtime[JavaInfo])

        java_info = java_common.compile(
            ctx,
            injecting_rule_kind = "java_lite_proto_library",
            source_jars = [source_jar],
            output = output_jar,
            output_source_jar = source_jar,
            deps = deps,
            exports = exports,
            java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],
            javac_opts = ctx.attr._java_toolchain[java_common.JavaToolchainInfo].compatible_javacopts(PROTO_JAVACOPTS_KEY),
            enable_jspecify = False,
            create_output_source_jar = False,
        )
    else:
        # If there are no proto sources just pass along the compilation dependencies.
        java_info = java_common.merge([], exports = deps + exports)

    return [
        java_info,
        _JavaProtoAspectInfo(jars = depset(files_to_build, transitive = transitive_files_to_build)),
    ]

java_lite_proto_aspect = aspect(
    implementation = _aspect_impl,
    attr_aspects = ["deps", "exports"],
    attrs = {
        PROTO_TOOLCHAIN_ATTR: attr.label(
            default = configuration_field(fragment = "proto", name = "proto_toolchain_for_java_lite"),
        ),
        JAVA_TOOLCHAIN_ATTR: attr.label(
            default = Label(semantics.JAVA_TOOLCHAIN_LABEL),
        ),
    },
    fragments = ["java"],
    required_providers = [ProtoInfo],
    provides = [JavaInfo, _JavaProtoAspectInfo],
)

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
