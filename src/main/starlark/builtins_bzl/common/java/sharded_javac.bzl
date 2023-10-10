# Copyright 2023 The Bazel Authors. All rights reserved.
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

""" Highly experimental/unstable implementation of sharded compilation """

load(":common/java/java_helper.bzl", "helper")

_java_common_internal = _builtins.internal.java_common_internal_do_not_use

def use_sharded_javac(ctx):
    """Whether this target should use sharded javac compilation

    Args:
        ctx: the rule context

    Returns:
        (bool)
     """
    if not hasattr(ctx.attr, "experimental_javac_shard_size"):
        return False
    if ctx.attr.experimental_javac_shard_size <= 0:
        fail("invalid value for javac shard size, expected a value > 0")
    return True

def experimental_sharded_javac(
        ctx,
        java_toolchain,
        output,
        compile_jar,
        plugin_info,
        compilation_classpath,
        direct_jars,
        bootclasspath,
        compile_time_java_deps,
        all_javac_opts,
        strict_deps,
        source_files,
        source_jars,
        resources,
        resource_jars):
    """ Performs multiple javac compilations and merges the results with singlejar

    Size of each shard is determined by ctx.attr.experimental_javac_shard_size,
    and there is one action for all resources, source_jars and resource_jars.

    Args:
        ctx: the rule context
        java_toolchain: the java toolchain
        output: (File) the final class jar to produce
        compile_jar: (File) the output of the header jar compilation for this target
        plugin_info: (JavaPluginInfo) information about plugins
        compilation_classpath: (depset[File]) the transitive jars required for compilation
        direct_jars: (depset[File]) the direct jars required for compilation
        bootclasspath: (BootClassPathInfo) the bootclasspath to use
        compile_time_java_deps: (depset[File])
        all_javac_opts: (depset[str]) options to pass to javabuilder
        strict_deps: (str) the strict deps mode
        source_files: [File] list of all the sources files to compile
        source_jars: [File] the list of jars containing sources
        resources: [File] the list of resource files
        resource_jars: [File] the list of jars containing resources
    """
    shard_direct_jars = depset(direct = [compile_jar], transitive = [direct_jars])
    shard_compilation_classpath = depset(direct = [compile_jar], transitive = [compilation_classpath])

    sharded_outputs = []
    shard_size = ctx.attr.experimental_javac_shard_size
    shard_count = (len(source_files) + shard_size - 1) // shard_size
    for shard_idx in range(shard_count):
        start = shard_idx * shard_size
        sources_for_shard = source_files[start:start + shard_size]
        shard_output = helper.derive_output_file(ctx, output, name_suffix = "_shard_%s" % shard_idx)
        sharded_outputs.append(shard_output)
        _java_common_internal.create_compilation_action(
            ctx,
            java_toolchain,
            shard_output,
            helper.derive_output_file(ctx, shard_output, extension_suffix = "_manifest_proto"),  # manifest_proto
            plugin_info,
            shard_compilation_classpath,
            shard_direct_jars,
            bootclasspath,
            compile_time_java_deps,
            all_javac_opts,
            strict_deps,
            ctx.label,
            sources = depset(sources_for_shard),
        )
    if resources or resource_jars or source_jars:
        shard_output = helper.derive_output_file(ctx, output, name_suffix = "_shard_resources")
        sharded_outputs.append(shard_output)
        _java_common_internal.create_compilation_action(
            ctx,
            java_toolchain,
            shard_output,
            helper.derive_output_file(ctx, shard_output, extension_suffix = "_manifest_proto"),  # manifest_proto
            plugin_info,
            shard_compilation_classpath,
            shard_direct_jars,
            bootclasspath,
            compile_time_java_deps,
            all_javac_opts,
            strict_deps,
            ctx.label,
            source_jars = source_jars,
            resources = resources,
            resource_jars = depset(resource_jars),
        )
    helper.create_single_jar(
        ctx.actions,
        toolchain = java_toolchain,
        output = output,
        sources = depset(sharded_outputs),
        progress_message = "Merging shards into %{output}",
        mnemonic = "JavaSingleJar",
        build_target = ctx.label,
        output_creator = "bazel",
    )
