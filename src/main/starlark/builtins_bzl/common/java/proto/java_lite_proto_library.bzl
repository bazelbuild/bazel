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

"""A Starlark implementation of the java_lite_proto_library rule.

  TODO(elenairina):
  * Return ProguardSpecProvider from the rule via JavaProvider.
  * Return proto_java from the aspect.
"""

load(":common/java/java_semantics.bzl", "semantics")

PROTO_TOOLCHAIN_ATTR = "_aspect_proto_toolchain_for_javalite"
PROTO_JAVACOPTS_KEY = "proto"
JAVA_TOOLCHAIN_ATTR = "_java_toolchain"

java_common = _builtins.toplevel.java_common
java_proto_common = _builtins.toplevel.java_proto_common
JavaInfo = _builtins.toplevel.JavaInfo
ProguardSpecProvider = _builtins.toplevel.ProguardSpecProvider

_JavaProtoAspectInfo = provider("JavaProtoAspectInfo", fields = ["jars"])

def _rule_impl(ctx):
    runtime = java_proto_common.get_runtime(
        ctx,
        proto_toolchain_attr = PROTO_TOOLCHAIN_ATTR,
    )
    proguard_provider_specs = []
    if runtime:
        proguard_provider_specs = [runtime[ProguardSpecProvider].specs]

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
        ProguardSpecProvider(
            depset(transitive = proguard_provider_specs),
        ),
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

    if java_proto_common.has_proto_sources(target):
        source_jar = ctx.actions.declare_file(ctx.label.name + "-lite-src.jar")
        java_proto_common.create_java_lite_proto_compile_action(
            ctx,
            target,
            src_jar = source_jar,
            proto_toolchain_attr = PROTO_TOOLCHAIN_ATTR,
            flavour = "javalite",
        )

        output_jar = ctx.actions.declare_file("lib" + ctx.label.name + "-lite.jar")

        files_to_build.extend([source_jar, output_jar])

        # This returns a java provider or None.
        runtime = java_proto_common.get_runtime(
            ctx,
            proto_toolchain_attr = PROTO_TOOLCHAIN_ATTR,
        )
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
    fragments = ["proto", "java"],
    provides = [JavaInfo],
)

java_lite_proto_library = rule(
    implementation = _rule_impl,
    attrs = {
        "deps": attr.label_list(aspects = [java_lite_proto_aspect]),
        PROTO_TOOLCHAIN_ATTR: attr.label(
            default = configuration_field(fragment = "proto", name = "proto_toolchain_for_java_lite"),
        ),
    },
    provides = [JavaInfo],
)
