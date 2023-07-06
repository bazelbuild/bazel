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
Common code for reuse across java_* rules
"""

load(":common/rule_util.bzl", "merge_attrs")
load(":common/java/android_lint.bzl", "android_lint_action")
load(":common/java/compile_action.bzl", "compile_action")
load(":common/java/java_semantics.bzl", "semantics")
load(":common/java/proguard_validation.bzl", "validate_proguard_specs")
load(":common/cc/cc_info.bzl", "CcInfo")
load(":common/java/java_info.bzl", "JavaInfo", "JavaPluginInfo")
load(":common/java/java_common.bzl", "java_common")
load(":common/java/java_common_internal_for_builtins.bzl", "target_kind")

coverage_common = _builtins.toplevel.coverage_common

def _filter_srcs(srcs, ext):
    return [f for f in srcs if f.extension == ext]

def _filter_provider(provider, *attrs):
    return [dep[provider] for attr in attrs for dep in attr if provider in dep]

def _get_attr_safe(ctx, attr, default):
    return getattr(ctx.attr, attr) if hasattr(ctx.attr, attr) else default

# TODO(b/11285003): disallow jar files in deps, require java_import instead
def _filter_javainfo_and_legacy_jars(attr):
    dep_list = []

    # Native code collected data into a NestedSet, using add for legacy jars and
    # addTransitive for JavaInfo. This resulted in legacy jars being first in the list.
    for dep in attr:
        kind = target_kind(dep, dereference_aliases = True)
        if not JavaInfo in dep or kind == "java_binary" or kind == "java_test":
            for file in dep[DefaultInfo].files.to_list():
                if file.extension == "jar":
                    # Native doesn't construct JavaInfo
                    java_info = JavaInfo(output_jar = file, compile_jar = file)
                    dep_list.append(java_info)

    for dep in attr:
        if JavaInfo in dep:
            dep_list.append(dep[JavaInfo])
    return dep_list

def basic_java_library(
        ctx,
        srcs,
        deps = [],
        runtime_deps = [],
        plugins = [],
        exports = [],
        exported_plugins = [],
        resources = [],
        resource_jars = [],
        classpath_resources = [],
        javacopts = [],
        neverlink = False,
        enable_compile_jar_action = True,
        coverage_config = None,
        proguard_specs = None,
        add_exports = [],
        add_opens = []):
    """
    Creates actions that compile and lint Java sources, sets up coverage and returns JavaInfo, InstrumentedFilesInfo and output groups.

    The call creates actions and providers needed and shared by `java_library`,
    `java_plugin`,`java_binary`, and `java_test` rules and it is primarily
    intended to be used in those rules.

    Before compilation coverage.runner is added to the dependencies and if
    present plugins are extended with the value of `--plugin` flag.

    Args:
      ctx: (RuleContext) Used to register the actions.
      srcs: (list[File]) The list of source files that are processed to create the target.
      deps: (list[Target]) The list of other libraries to be linked in to the target.
      runtime_deps: (list[Target]) Libraries to make available to the final binary or test at runtime only.
      plugins: (list[Target]) Java compiler plugins to run at compile-time.
      exports: (list[Target]) Exported libraries.
      exported_plugins: (list[Target]) The list of `java_plugin`s (e.g. annotation
        processors) to export to libraries that directly depend on this library.
      resources: (list[File]) A list of data files to include in a Java jar.
      resource_jars: (list[File]) A list of jar files to unpack and include in a
        Java jar.
      classpath_resources: (list[File])
      javacopts: (list[str])
      neverlink: (bool) Whether this library should only be used for compilation and not at runtime.
      enable_compile_jar_action: (bool) Enables header compilation or ijar creation.
      coverage_config: (struct{runner:JavaInfo, support_files:list[File]|depset[File], env:dict[str,str]})
        Coverage configuration. `runner` is added to dependencies during
        compilation, `support_files` and `env` is returned in InstrumentedFilesInfo.
      proguard_specs: (list[File]) Files to be used as Proguard specification.
        Proguard validation is done only when the parameter is set.
      add_exports: (list[str]) Allow this library to access the given <module>/<package>.
      add_opens: (list[str]) Allow this library to reflectively access the given <module>/<package>.
    Returns:
      (dict[str, Provider],
        {files_to_build: list[File],
         runfiles: list[File],
         output_groups: dict[str,list[File]]})
    """
    source_files = _filter_srcs(srcs, "java")
    source_jars = _filter_srcs(srcs, "srcjar")

    plugins_javaplugininfo = _collect_plugins(plugins)
    plugins_javaplugininfo.append(ctx.attr._java_plugins[JavaPluginInfo])

    properties = _filter_srcs(srcs, "properties")
    if properties:
        resources = list(resources)
        resources.extend(properties)

    java_info, compilation_info = compile_action(
        ctx,
        ctx.outputs.classjar,
        ctx.outputs.sourcejar,
        source_files,
        source_jars,
        collect_deps(deps) + ([coverage_config.runner] if coverage_config and coverage_config.runner else []),
        collect_deps(runtime_deps),
        plugins_javaplugininfo,
        collect_deps(exports),
        _collect_plugins(exported_plugins),
        resources,
        resource_jars,
        classpath_resources,
        _collect_native_libraries(deps, runtime_deps, exports),
        javacopts,
        neverlink,
        ctx.fragments.java.strict_java_deps,
        enable_compile_jar_action,
        add_exports = add_exports,
        add_opens = add_opens,
    )
    target = {"JavaInfo": java_info}

    output_groups = dict(
        compilation_outputs = compilation_info.files_to_build,
        _source_jars = java_info.transitive_source_jars,
        _direct_source_jars = java_info.source_jars,
    )

    if ctx.fragments.java.run_android_lint:
        generated_source_jars = [
            output.generated_source_jar
            for output in java_info.java_outputs
            if output.generated_source_jar != None
        ]
        lint_output = android_lint_action(
            ctx,
            source_files,
            source_jars + generated_source_jars,
            compilation_info,
        )
        if lint_output:
            output_groups["_validation"] = [lint_output]

    target["InstrumentedFilesInfo"] = coverage_common.instrumented_files_info(
        ctx,
        source_attributes = ["srcs"],
        dependency_attributes = ["deps", "data", "resources", "resource_jars", "exports", "runtime_deps", "jars"],
        coverage_support_files = coverage_config.support_files if coverage_config else depset(),
        coverage_environment = coverage_config.env if coverage_config else {},
    )

    if proguard_specs != None:
        target["ProguardSpecProvider"] = validate_proguard_specs(
            ctx,
            proguard_specs,
            [deps, runtime_deps, exports],
        )
        output_groups["_hidden_top_level_INTERNAL_"] = target["ProguardSpecProvider"].specs

    return target, struct(
        files_to_build = compilation_info.files_to_build,
        runfiles = compilation_info.runfiles,
        output_groups = output_groups,
    )

def _collect_plugins(plugins):
    """Collects plugins from an attribute.

    Use this call to collect plugins from `plugins` or `exported_plugins` attribute.

    The call simply extracts JavaPluginInfo provider.

    Args:
      plugins: (list[Target]) Attribute to collect plugins from.
    Returns:
      (list[JavaPluginInfo]) The plugins.
    """
    return _filter_provider(JavaPluginInfo, plugins)

def collect_deps(deps):
    """Collects dependencies from an attribute.

    Use this call to collect plugins from `deps`, `runtime_deps`, or `exports` attribute.

    The call extracts JavaInfo and additionaly also "legacy jars". "legacy jars"
    are wrapped into a JavaInfo.

    Args:
      deps: (list[Target]) Attribute to collect dependencies from.
    Returns:
      (list[JavaInfo]) The dependencies.
    """
    return _filter_javainfo_and_legacy_jars(deps)

def _collect_native_libraries(*attrs):
    """Collects native libraries from a list of attributes.

    Use this call to collect native libraries from `deps`, `runtime_deps`, or `exports` attributes.

    The call simply extracts CcInfo provider.
    Args:
      *attrs: (*list[Target]) Attribute to collect native libraries from.
    Returns:
      (list[CcInfo]) The native library dependencies.
    """
    return _filter_provider(CcInfo, *attrs)

def construct_defaultinfo(ctx, files_to_build, files, neverlink, *extra_attrs):
    """Constructs DefaultInfo for Java library like rule.

    Args:
      ctx: (RuleContext) Used to construct the runfiles.
      files_to_build: (list[File]) List of the files built by the rule.
      files: (list[File]) List of the files include in runfiles.
      neverlink: (bool) When true empty runfiles are constructed.
      *extra_attrs: (list[Target]) Extra attributes to merge runfiles from.

    Returns:
      (DefaultInfo) DefaultInfo provider.
    """
    if neverlink:
        runfiles = None
    else:
        runfiles = ctx.runfiles(files = files, collect_default = True)
        runfiles = runfiles.merge_all([dep[DefaultInfo].default_runfiles for attr in extra_attrs for dep in attr])
    default_info = DefaultInfo(
        files = depset(files_to_build),
        runfiles = runfiles,
    )
    return default_info

BASIC_JAVA_LIBRARY_IMPLICIT_ATTRS = merge_attrs(
    {
        "_java_plugins": attr.label(
            default = semantics.JAVA_PLUGINS_FLAG_ALIAS_LABEL,
            providers = [JavaPluginInfo],
        ),
        # TODO(b/245144242): Used by IDE integration, remove when toolchains are used
        "_java_toolchain": attr.label(
            default = semantics.JAVA_TOOLCHAIN_LABEL,
            providers = [java_common.JavaToolchainInfo],
        ),
    },
)
