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

""" Private utilities for Java compilation support in Starlark. """

load(":common/java/java_semantics.bzl", "semantics")
load(":common/paths.bzl", "paths")
load(
    ":common/java/java_info.bzl",
    "JavaPluginInfo",
    "disable_plugin_info_annotation_processing",
    "java_info_for_compilation",
    "merge_plugin_info_without_outputs",
)
load(":common/java/java_helper.bzl", "helper")

_java_common_internal = _builtins.internal.java_common_internal_do_not_use

def _derive_output_file(ctx, base_file, *, name_suffix = "", extension = None, extension_suffix = ""):
    """Declares a new file whose name is derived from the given file

    This method allows appending a suffix to the name (before extension), changing
    the extension or appending a suffix after the extension. The new file is declared
    as a sibling of the given base file. At least one of the three options must be
    specified. It is an error to specify both `extension` and `extension_suffix`.

    Args:
        ctx: (RuleContext) the rule context.
        base_file: (File) the file from which to derive the resultant file.
        name_suffix: (str) Optional. The suffix to append to the name before the
        extension.
        extension: (str) Optional. The new extension to use (without '.'). By default,
        the base_file's extension is used.
        extension_suffix: (str) Optional. The suffix to append to the base_file's extension

    Returns:
        (File) the derived file
    """
    if not name_suffix and not extension_suffix and not extension:
        fail("At least one of name_suffix, extension or extension_suffix is required")
    if extension and extension_suffix:
        fail("only one of extension or extension_suffix can be specified")
    if extension == None:
        extension = base_file.extension
    new_basename = paths.replace_extension(base_file.basename, name_suffix + "." + extension + extension_suffix)
    return ctx.actions.declare_file(new_basename, sibling = base_file)

def compile(
        ctx,
        output,
        java_toolchain,
        source_jars = [],
        source_files = [],
        output_source_jar = None,
        javac_opts = [],
        deps = [],
        runtime_deps = [],
        exports = [],
        plugins = [],
        exported_plugins = [],
        native_libraries = [],
        annotation_processor_additional_inputs = [],
        annotation_processor_additional_outputs = [],
        strict_deps = "ERROR",
        bootclasspath = None,
        sourcepath = [],
        resources = [],
        add_exports = [],
        add_opens = [],
        neverlink = False,
        enable_annotation_processing = True,
        # private to @_builtins:
        enable_compile_jar_action = True,
        enable_jspecify = True,
        include_compilation_info = True,
        classpath_resources = [],
        resource_jars = [],
        injecting_rule_kind = None):
    """Compiles Java source files/jars from the implementation of a Starlark rule

    The result is a provider that represents the results of the compilation and can be added to the
    set of providers emitted by this rule.

    Args:
        ctx: (RuleContext) The rule context
        output: (File) The output of compilation
        java_toolchain: (JavaToolchainInfo) Toolchain to be used for this compilation. Mandatory.
        source_jars: ([File]) A list of the jars to be compiled. At least one of source_jars or
            source_files should be specified.
        source_files: ([File]) A list of the Java source files to be compiled. At least one of
            source_jars or source_files should be specified.
        output_source_jar: (File) The output source jar. Optional. Defaults to
            `{output_jar}-src.jar` if unset.
        javac_opts: ([str]) A list of the desired javac options. Optional.
        deps: ([JavaInfo]) A list of dependencies. Optional.
        runtime_deps: ([JavaInfo]) A list of runtime dependencies. Optional.
        exports: ([JavaInfo]) A list of exports. Optional.
        plugins: ([JavaPluginInfo|JavaInfo]) A list of plugins. Optional.
        exported_plugins: ([JavaPluginInfo|JavaInfo]) A list of exported plugins. Optional.
        native_libraries: ([CcInfo]) CC library dependencies that are needed for this library.
        annotation_processor_additional_inputs: ([File]) A list of inputs that the Java compilation
            action will take in addition to the Java sources for annotation processing.
        annotation_processor_additional_outputs: ([File]) A list of outputs that the Java
            compilation action will output in addition to the class jar from annotation processing.
        strict_deps: (str) A string that specifies how to handle strict deps. Possible values:
            'OFF', 'ERROR', 'WARN' and 'DEFAULT'.
        bootclasspath: (BootClassPathInfo) If present, overrides the bootclasspath associated with
            the provided java_toolchain. Optional.
        sourcepath: ([File])
        resources: ([File])
        resource_jars: ([File])
        classpath_resources: ([File])
        neverlink: (bool)
        enable_annotation_processing: (bool) Disables annotation processing in this compilation,
            causing any annotation processors provided in plugins or in exported_plugins of deps to
            be ignored.
        enable_compile_jar_action: (bool) Enables header compilation or ijar creation. If set to
            False, it forces use of the full class jar in the compilation classpaths of any
            dependants. Doing so is intended for use by non-library targets such as binaries that
            do not have dependants.
        enable_jspecify: (bool)
        include_compilation_info: (bool)
        injecting_rule_kind: (str|None)
        add_exports: ([str]) Allow this library to access the given <module>/<package>. Optional.
        add_opens: ([str]) Allow this library to reflectively access the given <module>/<package>.
             Optional.

    Returns:
        (JavaInfo)
    """
    _java_common_internal.check_provider_instances(plugins, "plugins", JavaPluginInfo)

    plugin_info = merge_plugin_info_without_outputs(plugins + deps)

    all_javac_opts = []
    all_javac_opts.extend(_java_common_internal.default_javac_opts(java_toolchain = java_toolchain))

    all_javac_opts.extend(ctx.fragments.java.default_javac_flags)
    all_javac_opts.extend(semantics.compatible_javac_options(ctx, java_toolchain, _java_common_internal))

    if "com.google.devtools.build.runfiles.AutoBazelRepositoryProcessor" in plugin_info.plugins.processor_classes.to_list():
        all_javac_opts.append("-Abazel.repository=" + ctx.label.workspace_name)
    for package_config in java_toolchain.package_configuration():
        if package_config.matches(ctx.label):
            all_javac_opts.extend(package_config.javac_opts())

    all_javac_opts.extend(["--add-exports=%s=ALL-UNNAMED" % x for x in add_exports])
    all_javac_opts.extend(["--add-opens=%s=ALL-UNNAMED" % x for x in add_opens])
    all_javac_opts.extend([token for x in javac_opts for token in ctx.tokenize(x)])

    # Optimization: skip this if there are no annotation processors, to avoid unnecessarily
    # disabling the direct classpath optimization if `enable_annotation_processor = False`
    # but there aren't any annotation processors.
    enable_direct_classpath = True
    if not enable_annotation_processing and plugin_info.plugins.processor_classes:
        plugin_info = disable_plugin_info_annotation_processing(plugin_info)
        enable_direct_classpath = False

    uses_annotation_processing = False
    if "-processor" in all_javac_opts or plugin_info.plugins.processor_classes:
        uses_annotation_processing = True

    has_sources = source_files or source_jars
    has_resources = resources or resource_jars

    native_headers_jar = _derive_output_file(ctx, output, name_suffix = "-native-header")
    manifest_proto = _derive_output_file(ctx, output, extension_suffix = "_manifest_proto")
    deps_proto = None
    if ctx.fragments.java.generate_java_deps() and has_sources:
        deps_proto = _derive_output_file(ctx, output, extension = "jdeps")
    generated_class_jar = None
    generated_source_jar = None
    if uses_annotation_processing:
        generated_class_jar = _derive_output_file(ctx, output, name_suffix = "-gen")
        generated_source_jar = _derive_output_file(ctx, output, name_suffix = "-gensrc")

    is_strict_mode = strict_deps != "OFF"
    classpath_mode = ctx.fragments.java.reduce_java_classpath()

    direct_jars = depset()
    if is_strict_mode:
        direct_jars = depset(order = "preorder", transitive = [dep.compile_jars for dep in deps])
    compilation_classpath = depset(
        order = "preorder",
        transitive = [direct_jars] + [dep.transitive_compile_time_jars for dep in deps],
    )
    compile_time_java_deps = depset()
    if is_strict_mode and classpath_mode != "OFF":
        compile_time_java_deps = depset(transitive = [dep._compile_time_java_dependencies for dep in deps])

    _java_common_internal.create_compilation_action(
        ctx,
        java_toolchain,
        output,
        deps_proto,
        generated_class_jar,
        generated_source_jar,
        manifest_proto,
        native_headers_jar,
        plugin_info,
        depset(source_files),
        source_jars,
        resources,
        depset(resource_jars),
        compilation_classpath,
        classpath_resources,
        sourcepath,
        direct_jars,
        bootclasspath,
        compile_time_java_deps,
        all_javac_opts,
        strict_deps,
        ctx.label,
        injecting_rule_kind,
        enable_jspecify,
        enable_direct_classpath,
        annotation_processor_additional_inputs,
        annotation_processor_additional_outputs,
    )

    # create compile time jar action
    if not has_sources:
        compile_jar = None
        compile_deps_proto = None
    elif not enable_compile_jar_action:
        compile_jar = output
        compile_deps_proto = None
    elif _should_use_header_compilation(ctx, java_toolchain):
        compile_jar = _derive_output_file(ctx, output, name_suffix = "-hjar", extension = "jar")
        compile_deps_proto = _derive_output_file(ctx, output, name_suffix = "-hjar", extension = "jdeps")
        _java_common_internal.create_header_compilation_action(
            ctx,
            java_toolchain,
            compile_jar,
            compile_deps_proto,
            plugin_info,
            depset(source_files),
            source_jars,
            compilation_classpath,
            direct_jars,
            bootclasspath,
            compile_time_java_deps,
            all_javac_opts,
            strict_deps,
            ctx.label,
            injecting_rule_kind,
            enable_direct_classpath,
            annotation_processor_additional_inputs,
        )
    elif ctx.fragments.java.use_ijars():
        compile_jar = run_ijar(
            ctx.actions,
            output,
            java_toolchain,
            target_label = ctx.label,
            injecting_rule_kind = injecting_rule_kind,
        )
        compile_deps_proto = None
    else:
        compile_jar = output
        compile_deps_proto = None

    create_output_source_jar = len(source_files) > 0 or source_jars != [output_source_jar]
    if not output_source_jar:
        output_source_jar = _derive_output_file(ctx, output, name_suffix = "-src", extension = "jar")
    if create_output_source_jar:
        helper.create_single_jar(
            ctx.actions,
            toolchain = java_toolchain,
            output = output_source_jar,
            sources = depset(source_jars + ([generated_source_jar] if generated_source_jar else [])),
            resources = depset(source_files),
            progress_message = "Building source jar %{output}",
            mnemonic = "JavaSourceJar",
        )

    if has_sources or has_resources:
        direct_runtime_jars = [output]
    else:
        direct_runtime_jars = []

    compilation_info = struct(
        javac_options = all_javac_opts,
        # needs to be flattened because the public API is a list
        boot_classpath = (bootclasspath.bootclasspath if bootclasspath else java_toolchain.bootclasspath).to_list(),
        # we only add compile time jars from deps, and not exports
        compilation_classpath = compilation_classpath,
        runtime_classpath = depset(
            order = "preorder",
            direct = direct_runtime_jars,
            transitive = [dep.transitive_runtime_jars for dep in runtime_deps + deps],
        ),
        uses_annotation_processing = uses_annotation_processing,
    ) if include_compilation_info else None

    return java_info_for_compilation(
        output_jar = output,
        compile_jar = compile_jar,
        source_jar = output_source_jar,
        generated_class_jar = generated_class_jar,
        generated_source_jar = generated_source_jar,
        plugin_info = plugin_info,
        deps = deps,
        runtime_deps = runtime_deps,
        exports = exports,
        exported_plugins = exported_plugins,
        compile_jdeps = compile_deps_proto if compile_deps_proto else deps_proto,
        jdeps = deps_proto if include_compilation_info else None,
        native_headers_jar = native_headers_jar,
        manifest_proto = manifest_proto,
        native_libraries = native_libraries,
        neverlink = neverlink,
        add_exports = add_exports,
        add_opens = add_opens,
        direct_runtime_jars = direct_runtime_jars,
        compilation_info = compilation_info,
    )

def _should_use_header_compilation(ctx, toolchain):
    if not ctx.fragments.java.use_header_compilation():
        return False
    if toolchain.forcibly_disable_header_compilation():
        return False
    if not toolchain.has_header_compiler():
        fail(
            "header compilation was requested but it is not supported by the " +
            "current Java toolchain '" + str(toolchain.label) +
            "'; see the java_toolchain.header_compiler attribute",
        )
    if not toolchain.has_header_compiler_direct():
        fail(
            "header compilation was requested but it is not supported by the " +
            "current Java toolchain '" + str(toolchain.label) +
            "'; see the java_toolchain.header_compiler_direct attribute",
        )
    return True

def run_ijar(
        actions,
        jar,
        java_toolchain,
        target_label = None,
        # private to @_builtins:
        output = None,
        injecting_rule_kind = None):
    """Runs ijar on a jar, stripping it of its method bodies.

    This helps reduce rebuilding of dependent jars during any recompiles consisting only of simple
    changes to method implementations. The return value is typically passed to JavaInfo.compile_jar

    Args:
        actions: (actions) ctx.actions
        jar: (File) The jar to run ijar on.
        java_toolchain: (JavaToolchainInfo) The toolchain to used to find the ijar tool.
        target_label: (Label|None) A target label to stamp the jar with. Used for `add_dep` support.
            Typically, you would pass `ctx.label` to stamp the jar with the current rule's label.
        output: (File) Optional.
        injecting_rule_kind: (str) the rule class of the current target
    Returns:
        (File) The output artifact
    """
    if not output:
        output = actions.declare_file(paths.replace_extension(jar.basename, "-ijar.jar"), sibling = jar)
    args = actions.args()
    args.add(jar)
    args.add(output)
    if target_label != None:
        args.add("--target_label", target_label)
    if injecting_rule_kind != None:
        args.add("--injecting_rule_kind", injecting_rule_kind)

    actions.run(
        mnemonic = "JavaIjar",
        inputs = [jar],
        outputs = [output],
        executable = java_toolchain.ijar,
        arguments = [args],
        progress_message = "Extracting interface for jar %{input}",
        toolchain = semantics.JAVA_TOOLCHAIN_TYPE,
        use_default_shell_env = True,
    )
    return output

def target_kind(target, dereference_aliases = False):
    """Get the rule class string for a target

    Args:
        target: (Target)
        dereference_aliases: (bool) resolve the actual target rule class if an
            alias

    Returns:
        (str) The rule class string of the target
    """
    return _java_common_internal.target_kind(
        target,
        dereference_aliases = dereference_aliases,
    )

def get_build_info(ctx, is_stamping_enabled):
    """Get the artifacts representing the workspace status for this build

    Args:
        ctx: (RuleContext) The rule context
        is_stamping_enabled: (bool) If stamping is enabled

    Returns
        ([File]) The build info artifacts
    """
    return _java_common_internal.get_build_info(ctx, is_stamping_enabled)

def collect_native_deps_dirs(libraries):
    """Collect the set of root-relative paths containing native libraries

    Args:
        libraries: (depset[LibraryToLink]) set of native libraries

    Returns:
        ([String]) A set of root-relative paths as a list
    """
    return _java_common_internal.collect_native_deps_dirs(libraries)

def get_runtime_classpath_for_archive(jars, excluded_jars):
    """Filters a classpath to remove certain entries

    Args
        jars: (depset[File]) The classpath to filter
        excluded_jars: (depset[File]) The files to remove

    Returns:
        (depset[File]) The filtered classpath
    """
    return _java_common_internal.get_runtime_classpath_for_archive(
        jars,
        excluded_jars,
    )

def filter_protos_for_generated_extension_registry(runtime_jars, deploy_env):
    """Get proto artifacts from runtime_jars excluding those in deploy_env

    Args:
        runtime_jars: (depset[File]) the artifacts to scan
        deploy_env: (depset[File]) the artifacts to exclude

    Returns
        (depset[File], bool) A tuple of the filtered protos and whether all protos are 'lite'
            flavored
    """
    return _java_common_internal.filter_protos_for_generated_extension_registry(
        runtime_jars,
        deploy_env,
    )
