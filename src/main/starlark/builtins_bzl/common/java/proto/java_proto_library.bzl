# Copyright 2022 The Bazel Authors. All rights reserved.
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

"""The implementation of the `java_proto_library` rule and its aspect."""

load(":common/java/java_semantics.bzl", "semantics")
load(":common/proto/proto_common.bzl", proto_common = "proto_common_do_not_use")
load(":common/proto/providers.bzl", "ProtoLangToolchainInfo")

java_common = _builtins.toplevel.java_common
JavaInfo = _builtins.toplevel.JavaInfo
ProtoInfo = _builtins.toplevel.ProtoInfo

# The provider is used to collect source and runtime jars in the `proto_library` dependency graph.
JavaProtoAspectInfo = provider("JavaProtoAspectInfo", fields = ["jars"])

def _filter_provider(provider, *attrs):
    return [dep[provider] for attr in attrs for dep in attr if provider in dep]

def _bazel_java_proto_aspect_impl(target, ctx):
    """Generates and compiles Java code for a proto_library.

    The function runs protobuf compiler on the `proto_library` target using
    `proto_lang_toolchain` specified by `--proto_toolchain_for_java` flag.
    This generates a source jar.

    After that the source jar is compiled, respecting `deps` and `exports` of
    the `proto_library`.

    Args:
      target: (Target) The `proto_library` target (any target providing `ProtoInfo`.
      ctx: (RuleContext) The rule context.

    Returns:
      ([JavaInfo, JavaProtoAspectInfo]) A JavaInfo describing compiled Java
      version of`proto_library` and `JavaProtoAspectInfo` with all source and
      runtime jars.
    """

    proto_toolchain_info = ctx.attr._java_proto_toolchain[ProtoLangToolchainInfo]
    source_jar = None
    if proto_common.experimental_should_generate_code(target, proto_toolchain_info, "java_proto_library"):
        # Generate source jar using proto compiler.
        source_jar = ctx.actions.declare_file(ctx.label.name + "-speed-src.jar")
        proto_common.compile(
            ctx.actions,
            target,
            proto_toolchain_info,
            [source_jar],
            source_jar,
        )

    # Compile Java sources (or just merge if there aren't any)
    deps = _filter_provider(JavaInfo, ctx.rule.attr.deps)
    exports = _filter_provider(JavaInfo, ctx.rule.attr.exports)
    if source_jar and proto_toolchain_info.runtime:
        deps.append(proto_toolchain_info.runtime[JavaInfo])
    java_info, jars = java_compile_for_protos(
        ctx,
        "lib" + ctx.label.name + "-speed.jar",
        source_jar,
        deps,
        exports,
    )

    transitive_jars = [dep[JavaProtoAspectInfo].jars for dep in ctx.rule.attr.deps]
    return [
        java_info,
        JavaProtoAspectInfo(jars = depset(jars, transitive = transitive_jars)),
    ]

def java_compile_for_protos(ctx, output_jar_name, source_jar = None, deps = [], exports = []):
    """Compiles Java source jar returned by proto compiler.

    Use this call for java_xxx_proto_library. It uses java_common.compile with
    some checks disabled (via javacopts) and jspecify disabled, so that the
    generated code passes.

    It also takes care that input source jar is not repackaged with a different
    name.

    When `source_jar` is `None`, the function only merges `deps` and `exports`.

    Args:
      ctx: (RuleContext) Used to call `java_common.compile`
      output_jar_name: (str) How to name the output jar.
      source_jar: (File) Input source jar (may be `None`).
      deps: (list[JavaInfo]) `deps` of the `proto_library`.
      exports: (list[JavaInfo]) `exports` of the `proto_library`.
    Returns:
      ((JavaInfo, list[File])) JavaInfo of this target and list containing source
      and runtime jar, when they are created.
    """
    if source_jar != None:
        output_jar = ctx.actions.declare_file(output_jar_name)
        java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo]
        java_info = java_common.compile(
            ctx,
            source_jars = [source_jar],
            deps = deps,
            exports = exports,
            output = output_jar,
            output_source_jar = source_jar,
            injecting_rule_kind = "java_proto_library",
            javac_opts = java_toolchain.compatible_javacopts("proto"),
            enable_jspecify = False,
            create_output_source_jar = False,
            java_toolchain = java_toolchain,
        )
        jars = [source_jar, output_jar]
    else:
        # If there are no proto sources just pass along the compilation dependencies.
        java_info = java_common.merge([], exports = deps + exports)
        jars = []
    return java_info, jars

bazel_java_proto_aspect = aspect(
    implementation = _bazel_java_proto_aspect_impl,
    attrs = {
        "_java_proto_toolchain": attr.label(
            default = configuration_field(fragment = "proto", name = "proto_toolchain_for_java"),
        ),
        "_java_toolchain": attr.label(
            default = Label(semantics.JAVA_TOOLCHAIN_LABEL),
        ),
    },
    attr_aspects = ["deps", "exports"],
    required_providers = [ProtoInfo],
    provides = [JavaInfo, JavaProtoAspectInfo],
    fragments = ["java"],
)

def bazel_java_proto_library_rule(ctx):
    """Merges results of `java_proto_aspect` in `deps`.

    Args:
      ctx: (RuleContext) The rule context.
    Returns:
      ([JavaInfo, DefaultInfo, OutputGroupInfo])
    """

    java_info = java_common.merge(
        [],
        exports = [dep[JavaInfo] for dep in ctx.attr.deps],
        include_source_jars_from_exports = True,
    )

    transitive_src_and_runtime_jars = depset(transitive = [dep[JavaProtoAspectInfo].jars for dep in ctx.attr.deps])
    transitive_runtime_jars = depset(transitive = [java_info.transitive_runtime_jars])

    return [
        java_info,
        DefaultInfo(
            files = transitive_src_and_runtime_jars,
            runfiles = ctx.runfiles(transitive_files = transitive_runtime_jars),
        ),
        OutputGroupInfo(default = depset()),
    ]

java_proto_library = rule(
    implementation = bazel_java_proto_library_rule,
    attrs = {
        "deps": attr.label_list(providers = [ProtoInfo], aspects = [bazel_java_proto_aspect]),
    },
    provides = [JavaInfo],
)
