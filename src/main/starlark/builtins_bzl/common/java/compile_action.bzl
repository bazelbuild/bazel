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
Java compile action
"""

load(":common/java/java_semantics.bzl", "semantics")
load(":common/java/java_common_internal_for_builtins.bzl", _compile_private_for_builtins = "compile")

def _filter_strict_deps(mode):
    return "error" if mode in ["strict", "default"] else mode

def _collect_plugins(deps, plugins):
    transitive_processor_jars = []
    transitive_processor_data = []
    for plugin in plugins:
        transitive_processor_jars.append(plugin.plugins.processor_jars)
        transitive_processor_data.append(plugin.plugins.processor_data)
    for dep in deps:
        transitive_processor_jars.append(dep.plugins.processor_jars)
        transitive_processor_data.append(dep.plugins.processor_data)
    return struct(
        processor_jars = depset(transitive = transitive_processor_jars),
        processor_data = depset(transitive = transitive_processor_data),
    )

def compile_action(
        ctx,
        output_class_jar,
        output_source_jar,
        source_files = [],
        source_jars = [],
        deps = [],
        runtime_deps = [],
        plugins = [],
        exports = [],
        exported_plugins = [],
        resources = [],
        resource_jars = [],
        classpath_resources = [],
        native_libraries = [],
        javacopts = [],
        neverlink = False,
        strict_deps = "ERROR",
        enable_compile_jar_action = True,
        add_exports = [],
        add_opens = []):
    """
    Creates actions that compile Java sources, produce source jar, and produce header jar and returns JavaInfo.

    Use this call when you need the most basic and consistent Java compilation.

    Most parameters correspond to attributes on a java_library (srcs, deps,
    plugins, resources ...) except they are more strict, for example:

    - Where java_library's srcs attribute allows mixing of .java, .srcjar, and
     .properties files the arguments accepted by this call should be strictly
     separated into source_files, source_jars, and resources parameter.
    - deps parameter accepts only JavaInfo providers and plugins parameter only
     JavaPluginInfo

    The call creates following actions and files:
    - compiling Java sources to a class jar (output_class_jar parameter)
    - a source jar (output_source_jar parameter)
    - optionally a jar containing plugin generated classes when plugins are present
    - optionally a jar containing plugin generated sources
    - jdeps file containing dependencies used during compilation
    - other files used to speed up incremental builds:
         - a header jar - a jar containing only method signatures without implementation
         - compile jdeps - dependencies used during header compilation

    The returned JavaInfo provider may be used as a "fully-qualified" dependency
    to a java_library.

    Args:
      ctx: (RuleContext) Used to register the actions.
      output_class_jar: (File) Output class .jar file. The file needs to be declared.
      output_source_jar: (File) Output source .jar file. The file needs to be declared.
      source_files: (list[File]) A list of .java source files to compile.
        At least one of source_files or source_jars parameter must be specified.
      source_jars: (list[File]) A list of .jar or .srcjar files containing
        source files to compile.
        At least one of source_files or source_jars parameter must be specified.
      deps: (list[JavaInfo]) A list of dependencies.
      runtime_deps: (list[JavaInfo]) A list of runtime dependencies.
      plugins: (list[JavaPluginInfo]) A list of plugins.
      exports: (list[JavaInfo]) A list of exports.
      exported_plugins: (list[JavaInfo]) A list of exported plugins.
      resources: (list[File]) A list of resources.
      resource_jars: (list[File]) A list of jars to unpack.
      classpath_resources: (list[File]) A list of classpath resources.
      native_libraries: (list[CcInfo]) C++ native library dependencies that are
        needed for this library.
      javacopts: (list[str]) A list of the desired javac options. The options
        may contain `$(location ..)` templates that will be expanded.
      neverlink: (bool) Whether or not this library should be used only for
        compilation and not at runtime.
      strict_deps: (str) A string that specifies how to handle strict deps.
        Possible values: 'OFF', 'ERROR', 'WARN' and 'DEFAULT'. For more details
        see https://bazel.build/docs/user-manual#strict-java-deps.
        By default 'ERROR'.
      enable_compile_jar_action: (bool) Enables header compilation or ijar
        creation. If set to False, it forces use of the full class jar in the
        compilation classpaths of any dependants. Doing so is intended for use
        by non-library targets such as binaries that do not have dependants.
      add_exports: (list[str]) Allow this library to access the given <module>/<package>.
      add_opens: (list[str]) Allow this library to reflectively access the given <module>/<package>.

    Returns:
      ((JavaInfo, {files_to_build: list[File],
                   runfiles: list[File],
                   compilation_classpath: list[File],
                   plugins: {processor_jars,
                             processor_data: depset[File]}}))
      A tuple with JavaInfo provider and additional compilation info.

      Files_to_build may include an empty .jar file when there are no sources
      or resources present, whereas runfiles in this case are empty.
    """

    java_info = _compile_private_for_builtins(
        ctx,
        output = output_class_jar,
        java_toolchain = semantics.find_java_toolchain(ctx),
        source_files = source_files,
        source_jars = source_jars,
        resources = resources,
        resource_jars = resource_jars,
        classpath_resources = classpath_resources,
        plugins = plugins,
        deps = deps,
        native_libraries = native_libraries,
        runtime_deps = runtime_deps,
        exports = exports,
        exported_plugins = exported_plugins,
        javac_opts = [ctx.expand_location(opt) for opt in javacopts],
        neverlink = neverlink,
        output_source_jar = output_source_jar,
        strict_deps = _filter_strict_deps(strict_deps),
        enable_compile_jar_action = enable_compile_jar_action,
        add_exports = add_exports,
        add_opens = add_opens,
    )

    compilation_info = struct(
        files_to_build = [output_class_jar],
        runfiles = [output_class_jar] if source_files or source_jars or resources else [],
        # TODO(ilist): collect compile_jars from JavaInfo in deps & exports
        compilation_classpath = java_info.compilation_info.compilation_classpath,
        javac_options = java_info.compilation_info.javac_options,
        plugins = _collect_plugins(deps, plugins),
    )

    return java_info, compilation_info
