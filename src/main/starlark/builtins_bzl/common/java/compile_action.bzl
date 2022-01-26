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

load(":common/rule_util.bzl", "create_dep")
load(":common/java/java_semantics.bzl", "semantics")

java_common = _builtins.toplevel.java_common

CcInfo = _builtins.toplevel.CcInfo
JavaInfo = _builtins.toplevel.JavaInfo
JavaPluginInfo = _builtins.toplevel.JavaPluginInfo

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

def _compile_action(
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
        classpath_resources = [],
        native_libraries = [],
        javacopts = [],
        neverlink = False,
        strict_deps = "ERROR",
        enable_compile_jar_action = True):
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
      classpath_resources: (list[File]) A list of classpath resources.
      native_libraries: (list[CcInfo]) C++ native library dependencies that are
        needed for this library.
      javacopts: (list[str]) A list of the desired javac options. The options
        may contain `$(location ..)` templates that will be expanded.
      neverlink: (bool) Whether or not this library should be used only for
        compilation and not at runtime.
      strict_deps: (str) A string that specifies how to handle strict deps.
        Possible values: 'OFF', 'ERROR', 'WARN' and 'DEFAULT'. For more details
        see https://docs.bazel.build/versions/main/bazel-user-manual.html#flag--strict_java_deps.
        By default 'ERROR'.
      enable_compile_jar_action: (bool) Enables header compilation or ijar
        creation. If set to False, it forces use of the full class jar in the
        compilation classpaths of any dependants. Doing so is intended for use
        by non-library targets such as binaries that do not have dependants.

    Returns:
      ((JavaInfo, {output_class_jars: list[File], plugins: {processor_jars, processor_data: depset[File]}})
      A tuple with JavaInfo provider and additional compilation info.
    """

    java_info = java_common.compile(
        ctx,
        source_files = source_files,
        source_jars = source_jars,
        resources = resources,
        classpath_resources = classpath_resources,
        plugins = plugins,
        deps = deps,
        native_libraries = native_libraries,
        runtime_deps = runtime_deps,
        exports = exports,
        exported_plugins = exported_plugins,
        javac_opts = [ctx.expand_location(opt) for opt in javacopts],
        neverlink = neverlink,
        java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],
        output = output_class_jar,
        output_source_jar = output_source_jar,
        strict_deps = _filter_strict_deps(strict_deps),
        enable_compile_jar_action = enable_compile_jar_action,
    )

    # TODO(b/213551463): Can `output_class_jars = [output_class_jar]` be used here, if not document.
    output_class_jars = [out.class_jar for out in java_info.java_outputs]

    compilation_info = struct(
        output_class_jars = output_class_jars,
        plugins = _collect_plugins(deps, plugins),
    )

    return java_info, compilation_info

COMPILE_ACTION = create_dep(
    _compile_action,
    attrs = {
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
        "_java_toolchain": attr.label(
            default = semantics.JAVA_TOOLCHAIN_LABEL,
            providers = [java_common.JavaToolchainInfo],
        ),
        "_java_plugins": attr.label(
            default = semantics.JAVA_PLUGINS_FLAG_ALIAS_LABEL,
            providers = [JavaPluginInfo],
        ),
    },
    fragments = ["java", "cpp"],
    mandatory_attrs = ["srcs", "deps", "resources", "plugins", "javacopts", "neverlink"],
)
