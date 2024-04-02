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

""" Implementation of java_binary for bazel """

load(":common/cc/cc_common.bzl", "cc_common")
load(":common/cc/cc_info.bzl", "CcInfo")
load(":common/cc/semantics.bzl", cc_semantics = "semantics")
load(":common/java/basic_java_library.bzl", "BASIC_JAVA_LIBRARY_IMPLICIT_ATTRS", "basic_java_library", "collect_deps")
load(":common/java/boot_class_path_info.bzl", "BootClassPathInfo")
load(":common/java/java_binary_deploy_jar.bzl", "create_deploy_archive")
load(":common/java/java_common.bzl", "java_common")
load(
    ":common/java/java_common_internal_for_builtins.bzl",
    "collect_native_deps_dirs",
    "get_runtime_classpath_for_archive",
)
load(":common/java/java_helper.bzl", "helper")
load(":common/java/java_info.bzl", "JavaCompilationInfo", "JavaInfo", "JavaPluginInfo", "to_java_binary_info")
load(":common/java/java_semantics.bzl", "semantics")
load(":common/paths.bzl", "paths")
load(":common/proto/proto_info.bzl", "ProtoInfo")
load(":common/rule_util.bzl", "merge_attrs")

CcLauncherInfo = _builtins.internal.cc_internal.launcher_provider

InternalDeployJarInfo = provider(
    "Provider for passing info to deploy jar rule",
    fields = [
        "java_attrs",
        "launcher_info",
        "shared_archive",
        "main_class",
        "coverage_main_class",
        "strip_as_default",
        "stamp",
        "hermetic",
        "add_exports",
        "add_opens",
        "manifest_lines",
    ],
)

def basic_java_binary(
        ctx,
        deps,
        runtime_deps,
        resources,
        main_class,
        coverage_main_class,
        coverage_config,
        launcher_info,
        executable,
        strip_as_default,
        extension_registry_provider = None,
        is_test_rule_class = False):
    """Creates actions for compiling and linting java sources, coverage support, and sources jar (_deploy-src.jar).

    Args:
        ctx: (RuleContext) The rule context
        deps: (list[Target]) The list of other targets to be compiled with
        runtime_deps: (list[Target]) The list of other targets to be linked in
        resources: (list[File]) The list of data files to be included in the class jar
        main_class: (String) FQN of the java main class
        coverage_main_class: (String) FQN of the actual main class if coverage is enabled
        coverage_config: (Struct|None) If coverage is enabled, a struct with fields (runner, manifest, env, support_files), None otherwise
        launcher_info: (Struct) Structure with fields (launcher, unstripped_launcher, runfiles, runtime_jars, jvm_flags, classpath_resources)
        executable: (File) The executable output of the rule
        strip_as_default: (bool) Whether this target outputs a stripped launcher and deploy jar
        extension_registry_provider: (GeneratedExtensionRegistryProvider) internal param, do not use
        is_test_rule_class: (bool) Whether this rule is a test rule

    Returns:
        Tuple(
            dict[str, Provider],    // providers
            Struct(                 // default info
                files_to_build: depset(File),
                runfiles: Runfiles,
                executable: File
            ),
            list[String]            // jvm flags
          )

    """
    if not ctx.attr.create_executable and (ctx.attr.launcher and cc_common.launcher_provider in ctx.attr.launcher):
        fail("launcher specified but create_executable is false")
    if not ctx.attr.use_launcher and (ctx.attr.launcher and ctx.attr.launcher.label != semantics.LAUNCHER_FLAG_LABEL):
        fail("launcher specified but use_launcher is false")

    if not ctx.attr.srcs and ctx.attr.deps:
        fail("deps not allowed without srcs; move to runtime_deps?")

    module_flags = [dep[JavaInfo].module_flags_info for dep in runtime_deps if JavaInfo in dep]
    add_exports = depset(ctx.attr.add_exports, transitive = [m.add_exports for m in module_flags])
    add_opens = depset(ctx.attr.add_opens, transitive = [m.add_opens for m in module_flags])

    classpath_resources = []
    classpath_resources.extend(launcher_info.classpath_resources)
    if hasattr(ctx.files, "classpath_resources"):
        classpath_resources.extend(ctx.files.classpath_resources)

    toolchain = semantics.find_java_toolchain(ctx)
    timezone_data = [toolchain._timezone_data] if toolchain._timezone_data else []
    target, common_info = basic_java_library(
        ctx,
        srcs = ctx.files.srcs,
        deps = deps,
        runtime_deps = runtime_deps,
        plugins = ctx.attr.plugins,
        resources = resources,
        resource_jars = timezone_data,
        classpath_resources = classpath_resources,
        javacopts = ctx.attr.javacopts,
        neverlink = ctx.attr.neverlink,
        enable_compile_jar_action = False,
        coverage_config = coverage_config,
        add_exports = ctx.attr.add_exports,
        add_opens = ctx.attr.add_opens,
        bootclasspath = ctx.attr.bootclasspath,
    )
    java_info = target["JavaInfo"]
    compilation_info = java_info.compilation_info
    runtime_classpath = depset(
        order = "preorder",
        transitive = [
            java_info.transitive_runtime_jars
            for java_info in (
                collect_deps(ctx.attr.runtime_deps + deps) +
                ([coverage_config.runner] if coverage_config and coverage_config.runner else [])
            )
        ],
    )
    if extension_registry_provider:
        runtime_classpath = depset(order = "preorder", direct = [extension_registry_provider.class_jar], transitive = [runtime_classpath])
        java_info = java_common.merge(
            [
                java_info,
                JavaInfo(
                    output_jar = extension_registry_provider.class_jar,
                    compile_jar = None,
                    source_jar = extension_registry_provider.src_jar,
                ),
            ],
        )
        compilation_info = JavaCompilationInfo(
            compilation_classpath = compilation_info.compilation_classpath,
            runtime_classpath = runtime_classpath,
            boot_classpath = compilation_info.boot_classpath,
            javac_options = compilation_info.javac_options,
        )

    java_attrs = _collect_attrs(ctx, runtime_classpath, classpath_resources)

    jvm_flags = []

    jvm_flags.extend(launcher_info.jvm_flags)

    native_libs_depsets = []
    for dep in runtime_deps:
        if JavaInfo in dep:
            native_libs_depsets.append(dep[JavaInfo].transitive_native_libraries)
        if CcInfo in dep:
            native_libs_depsets.append(dep[CcInfo].transitive_native_libraries())
    native_libs_dirs = collect_native_deps_dirs(depset(transitive = native_libs_depsets))
    if native_libs_dirs:
        prefix = "${JAVA_RUNFILES}/" + ctx.workspace_name + "/"
        jvm_flags.append("-Djava.library.path=%s" % (
            ":".join([prefix + d for d in native_libs_dirs])
        ))

    jvm_flags.extend(ctx.fragments.java.default_jvm_opts)
    jvm_flags.extend([ctx.expand_make_variables(
        "jvm_flags",
        ctx.expand_location(flag, ctx.attr.data, short_paths = True),
        {},
    ) for flag in ctx.attr.jvm_flags])

    # TODO(cushon): make string formatting lazier once extend_template support is added
    # https://github.com/bazelbuild/proposals#:~:text=2022%2D04%2D25,Starlark
    jvm_flags.extend(["--add-exports=%s=ALL-UNNAMED" % x for x in add_exports.to_list()])
    jvm_flags.extend(["--add-opens=%s=ALL-UNNAMED" % x for x in add_opens.to_list()])

    files_to_build = []

    if executable:
        files_to_build.append(executable)

    output_groups = common_info.output_groups

    if coverage_config:
        _generate_coverage_manifest(ctx, coverage_config.manifest, java_attrs.runtime_classpath)
        files_to_build.append(coverage_config.manifest)

    shared_archive = _create_shared_archive(ctx, java_attrs)

    if extension_registry_provider:
        files_to_build.append(extension_registry_provider.class_jar)
        output_groups["_direct_source_jars"] = (
            output_groups["_direct_source_jars"] + [extension_registry_provider.src_jar]
        )
        output_groups["_source_jars"] = depset(
            direct = [extension_registry_provider.src_jar],
            transitive = [output_groups["_source_jars"]],
        )

    if (ctx.fragments.java.one_version_enforcement_on_java_tests or not is_test_rule_class):
        one_version_output = _create_one_version_check(ctx, java_attrs.runtime_classpath, is_test_rule_class)
    else:
        one_version_output = None

    validation_outputs = [one_version_output] if one_version_output else []

    _create_deploy_sources_jar(ctx, output_groups["_source_jars"])

    files = depset(files_to_build + common_info.files_to_build)

    transitive_runfiles_artifacts = depset(transitive = [
        files,
        java_attrs.runtime_classpath,
        depset(transitive = launcher_info.runfiles),
    ])

    runfiles = ctx.runfiles(
        transitive_files = transitive_runfiles_artifacts,
        collect_default = True,
    )

    if launcher_info.launcher:
        default_launcher = helper.filter_launcher_for_target(ctx)
        default_launcher_artifact = helper.launcher_artifact_for_target(ctx)
        default_launcher_runfiles = default_launcher[DefaultInfo].default_runfiles
        if default_launcher_artifact == launcher_info.launcher:
            runfiles = runfiles.merge(default_launcher_runfiles)
        else:
            # N.B. The "default launcher" referred to here is the launcher target specified through
            # an attribute or flag. We wish to retain the runfiles of the default launcher, *except*
            # for the original cc_binary artifact, because we've swapped it out with our custom
            # launcher. Hence, instead of calling builder.addTarget(), or adding an odd method
            # to Runfiles.Builder, we "unravel" the call and manually add things to the builder.
            # Because the NestedSet representing each target's launcher runfiles is re-built here,
            # we may see increased memory consumption for representing the target's runfiles.
            runfiles = runfiles.merge(
                ctx.runfiles(
                    files = [launcher_info.launcher],
                    transitive_files = depset([
                        file
                        for file in default_launcher_runfiles.files.to_list()
                        if file != default_launcher_artifact
                    ]),
                    symlinks = default_launcher_runfiles.symlinks,
                    root_symlinks = default_launcher_runfiles.root_symlinks,
                ),
            )

    runfiles = runfiles.merge_all([
        dep[DefaultInfo].default_runfiles
        for dep in ctx.attr.runtime_deps
        if DefaultInfo in dep
    ])

    if validation_outputs:
        output_groups["_validation"] = output_groups.get("_validation", []) + validation_outputs

    _filter_validation_output_group(ctx, output_groups)

    java_binary_info = to_java_binary_info(java_info, compilation_info)

    internal_deploy_jar_info = InternalDeployJarInfo(
        java_attrs = java_attrs,
        launcher_info = struct(
            runtime_jars = launcher_info.runtime_jars,
            launcher = launcher_info.launcher,
            unstripped_launcher = launcher_info.unstripped_launcher,
        ),
        shared_archive = shared_archive,
        main_class = main_class,
        coverage_main_class = coverage_main_class,
        strip_as_default = strip_as_default,
        stamp = ctx.attr.stamp,
        hermetic = hasattr(ctx.attr, "hermetic") and ctx.attr.hermetic,
        add_exports = add_exports,
        add_opens = add_opens,
        manifest_lines = ctx.attr.deploy_manifest_lines,
    )

    # "temporary" workaround for https://github.com/bazelbuild/intellij/issues/5845
    extra_files = []
    if is_test_rule_class and ctx.fragments.java.auto_create_java_test_deploy_jars():
        extra_files.append(_auto_create_deploy_jar(ctx, internal_deploy_jar_info))

    default_info = struct(
        files = depset(extra_files, transitive = [files]),
        runfiles = runfiles,
        executable = executable,
    )

    return {
        "OutputGroupInfo": OutputGroupInfo(**output_groups),
        "JavaInfo": java_binary_info,
        "InstrumentedFilesInfo": target["InstrumentedFilesInfo"],
        "JavaRuntimeClasspathInfo": java_common.JavaRuntimeClasspathInfo(runtime_classpath = java_info.transitive_runtime_jars),
        "InternalDeployJarInfo": internal_deploy_jar_info,
    }, default_info, jvm_flags

def _collect_attrs(ctx, runtime_classpath, classpath_resources):
    deploy_env_jars = depset(transitive = [
        dep[java_common.JavaRuntimeClasspathInfo].runtime_classpath
        for dep in ctx.attr.deploy_env
    ]) if hasattr(ctx.attr, "deploy_env") else depset()

    runtime_classpath_for_archive = get_runtime_classpath_for_archive(runtime_classpath, deploy_env_jars)
    runtime_jars = [ctx.outputs.classjar]

    resources = [p for p in ctx.files.srcs if p.extension == "properties"]
    transitive_resources = []
    for r in ctx.attr.resources:
        transitive_resources.append(
            r[ProtoInfo].transitive_sources if ProtoInfo in r else r.files,
        )

    resource_names = dict()
    for r in classpath_resources:
        if r.basename in resource_names:
            fail("entries must have different file names (duplicate: %s)" % r.basename)
        resource_names[r.basename] = None

    return struct(
        runtime_jars = depset(runtime_jars),
        runtime_classpath_for_archive = runtime_classpath_for_archive,
        classpath_resources = depset(classpath_resources),
        runtime_classpath = depset(order = "preorder", direct = runtime_jars, transitive = [runtime_classpath]),
        resources = depset(resources, transitive = transitive_resources),
    )

def _generate_coverage_manifest(ctx, output, runtime_classpath):
    ctx.actions.write(
        output = output,
        content = "\n".join([file.short_path for file in runtime_classpath.to_list()]),
    )

#TODO(hvd): not needed in bazel
def _create_shared_archive(ctx, java_attrs):
    classlist = ctx.file.classlist if hasattr(ctx.file, "classlist") else None
    if not classlist:
        return None
    runtime = semantics.find_java_runtime_toolchain(ctx)
    jsa = ctx.actions.declare_file("%s.jsa" % ctx.label.name)
    merged = ctx.actions.declare_file(jsa.dirname + "/" + helper.strip_extension(jsa) + "-merged.jar")
    helper.create_single_jar(
        ctx.actions,
        toolchain = semantics.find_java_toolchain(ctx),
        output = merged,
        sources = depset(transitive = [java_attrs.runtime_jars, java_attrs.runtime_classpath_for_archive]),
    )

    args = ctx.actions.args()
    args.add("-Xshare:dump")
    args.add(jsa, format = "-XX:SharedArchiveFile=%s")
    args.add(classlist, format = "-XX:SharedClassListFile=%s")

    input_files = [classlist, merged]

    config_file = ctx.file.cds_config_file if hasattr(ctx.file, "cds_config_file") else None
    if config_file:
        args.add(config_file, format = "-XX:SharedArchiveConfigFile=%s")
        input_files.append(config_file)

    args.add("-cp", merged)

    if hasattr(ctx.attr, "jvm_flags_for_cds_image_creation") and ctx.attr.jvm_flags_for_cds_image_creation:
        args.add_all([
            ctx.expand_location(flag, ctx.attr.data)
            for flag in ctx.attr.jvm_flags_for_cds_image_creation
        ])
        input_files.extend(ctx.files.data)

    ctx.actions.run(
        mnemonic = "JavaJSA",
        progress_message = "Dumping Java Shared Archive %s" % jsa.short_path,
        executable = runtime.java_executable_exec_path,
        toolchain = semantics.JAVA_RUNTIME_TOOLCHAIN_TYPE,
        inputs = depset(input_files, transitive = [runtime.files]),
        outputs = [jsa],
        arguments = [args],
    )
    return jsa

def _create_one_version_check(ctx, inputs, is_test_rule_class):
    one_version_level = ctx.fragments.java.one_version_enforcement_level
    if one_version_level == "OFF":
        return None
    tool = helper.check_and_get_one_version_attribute(ctx, "_one_version_tool")

    if is_test_rule_class:
        toolchain = semantics.find_java_toolchain(ctx)
        allowlist = toolchain._one_version_allowlist_for_tests
    else:
        allowlist = helper.check_and_get_one_version_attribute(ctx, "_one_version_allowlist")

    if not tool or not allowlist:  # On Mac oneversion tool is not available
        return None

    output = ctx.actions.declare_file("%s-one-version.txt" % ctx.label.name)

    args = ctx.actions.args()
    args.set_param_file_format("shell").use_param_file("@%s", use_always = True)

    args.add("--output", output)
    args.add("--whitelist", allowlist)
    if one_version_level == "WARNING":
        args.add("--succeed_on_found_violations")
    args.add_all(
        "--inputs",
        inputs,
        map_each = helper.jar_and_target_arg_mapper,
    )

    ctx.actions.run(
        mnemonic = "JavaOneVersion",
        progress_message = "Checking for one-version violations in %{label}",
        executable = tool,
        toolchain = semantics.JAVA_TOOLCHAIN_TYPE,
        inputs = depset([allowlist], transitive = [inputs]),
        tools = [tool],
        outputs = [output],
        arguments = [args],
    )

    return output

def _create_deploy_sources_jar(ctx, sources):
    helper.create_single_jar(
        ctx.actions,
        toolchain = semantics.find_java_toolchain(ctx),
        output = ctx.outputs.deploysrcjar,
        sources = sources,
    )

def _filter_validation_output_group(ctx, output_group):
    to_exclude = depset(transitive = [
        dep[OutputGroupInfo]._validation
        for dep in ctx.attr.deploy_env
        if OutputGroupInfo in dep and hasattr(dep[OutputGroupInfo], "_validation")
    ]) if hasattr(ctx.attr, "deploy_env") else depset()
    if to_exclude:
        transitive_validations = depset(transitive = [
            _get_validations_from_attr(ctx, attr_name)
            for attr_name in dir(ctx.attr)
            # we also exclude implicit, cfg=host/exec and tool attributes
            if not attr_name.startswith("_") and
               attr_name not in [
                   "deploy_env",
                   "applicable_licenses",
                   "package_metadata",
                   "plugins",
                   "translations",
                   # special ignored attributes
                   "compatible_with",
                   "restricted_to",
                   "exec_compatible_with",
                   "target_compatible_with",
               ]
        ])
        if not ctx.attr.create_executable:
            excluded_set = {x: None for x in to_exclude.to_list()}
            transitive_validations = [
                x
                for x in transitive_validations.to_list()
                if x not in excluded_set
            ]
        output_group["_validation_transitive"] = transitive_validations

def _get_validations_from_attr(ctx, attr_name):
    attr = getattr(ctx.attr, attr_name)
    if type(attr) == "list":
        return depset(transitive = [_get_validations_from_target(t) for t in attr])
    else:
        return _get_validations_from_target(attr)

def _get_validations_from_target(target):
    if (
        type(target) == "Target" and
        OutputGroupInfo in target and
        hasattr(target[OutputGroupInfo], "_validation")
    ):
        return target[OutputGroupInfo]._validation
    else:
        return depset()

# TODO: bazelbuild/intellij/issues/5845 - remove this once no longer required
# this need not be completely identical to the regular deploy jar since we only
# care about packaging the classpath
def _auto_create_deploy_jar(ctx, info):
    output = ctx.actions.declare_file(ctx.label.name + "_auto_deploy.jar")
    java_attrs = info.java_attrs
    runtime_classpath = depset(
        direct = info.launcher_info.runtime_jars,
        transitive = [
            java_attrs.runtime_jars,
            java_attrs.runtime_classpath_for_archive,
        ],
        order = "preorder",
    )
    create_deploy_archive(
        ctx,
        launcher = info.launcher_info.launcher,
        main_class = info.main_class,
        coverage_main_class = info.coverage_main_class,
        resources = java_attrs.resources,
        classpath_resources = java_attrs.classpath_resources,
        runtime_classpath = runtime_classpath,
        manifest_lines = info.manifest_lines,
        build_info_files = [],
        build_target = str(ctx.label),
        output = output,
        shared_archive = info.shared_archive,
        one_version_level = ctx.fragments.java.one_version_enforcement_level,
        one_version_allowlist = helper.check_and_get_one_version_attribute(ctx, "_one_version_allowlist"),
        multi_release = ctx.fragments.java.multi_release_deploy_jars,
        hermetic = info.hermetic,
        add_exports = info.add_exports,
        add_opens = info.add_opens,
    )
    return output

BASIC_JAVA_BINARY_ATTRIBUTES = merge_attrs(
    BASIC_JAVA_LIBRARY_IMPLICIT_ATTRS,
    {
        "srcs": attr.label_list(
            allow_files = [".java", ".srcjar", ".properties"] + semantics.EXTRA_SRCS_TYPES,
            flags = ["DIRECT_COMPILE_TIME_INPUT", "ORDER_INDEPENDENT"],
            doc = """
The list of source files that are processed to create the target.
This attribute is almost always required; see exceptions below.
<p>
Source files of type <code>.java</code> are compiled. In case of generated
<code>.java</code> files it is generally advisable to put the generating rule's name
here instead of the name of the file itself. This not only improves readability but
makes the rule more resilient to future changes: if the generating rule generates
different files in the future, you only need to fix one place: the <code>outs</code> of
the generating rule. You should not list the generating rule in <code>deps</code>
because it is a no-op.
</p>
<p>
Source files of type <code>.srcjar</code> are unpacked and compiled. (This is useful if
you need to generate a set of <code>.java</code> files with a genrule.)
</p>
<p>
Rules: if the rule (typically <code>genrule</code> or <code>filegroup</code>) generates
any of the files listed above, they will be used the same way as described for source
files.
</p>

<p>
This argument is almost always required, except if a
<a href="#java_binary.main_class"><code>main_class</code></a> attribute specifies a
class on the runtime classpath or you specify the <code>runtime_deps</code> argument.
</p>
            """,
        ),
        "deps": attr.label_list(
            allow_files = [".jar"],
            allow_rules = semantics.ALLOWED_RULES_IN_DEPS + semantics.ALLOWED_RULES_IN_DEPS_WITH_WARNING,
            providers = [
                [CcInfo],
                [JavaInfo],
            ],
            flags = ["SKIP_ANALYSIS_TIME_FILETYPE_CHECK"],
            doc = """
The list of other libraries to be linked in to the target.
See general comments about <code>deps</code> at
<a href="common-definitions.html#typical-attributes">Typical attributes defined by
most build rules</a>.
            """,
        ),
        "resources": attr.label_list(
            allow_files = True,
            flags = ["SKIP_CONSTRAINTS_OVERRIDE", "ORDER_INDEPENDENT"],
            doc = """
A list of data files to include in a Java jar.

<p>
Resources may be source files or generated files.
</p>
            """ + semantics.DOCS.for_attribute("resources"),
        ),
        "runtime_deps": attr.label_list(
            allow_files = [".jar"],
            allow_rules = semantics.ALLOWED_RULES_IN_DEPS,
            providers = [[CcInfo], [JavaInfo]],
            flags = ["SKIP_ANALYSIS_TIME_FILETYPE_CHECK"],
            doc = """
Libraries to make available to the final binary or test at runtime only.
Like ordinary <code>deps</code>, these will appear on the runtime classpath, but unlike
them, not on the compile-time classpath. Dependencies needed only at runtime should be
listed here. Dependency-analysis tools should ignore targets that appear in both
<code>runtime_deps</code> and <code>deps</code>.
            """,
        ),
        "data": attr.label_list(
            allow_files = True,
            flags = ["SKIP_CONSTRAINTS_OVERRIDE"],
            doc = """
The list of files needed by this library at runtime.
See general comments about <code>data</code>
at <a href="${link common-definitions#typical-attributes}">Typical attributes defined by
most build rules</a>.
            """ + semantics.DOCS.for_attribute("data"),
        ),
        "plugins": attr.label_list(
            providers = [JavaPluginInfo],
            allow_files = True,
            cfg = "exec",
            doc = """
Java compiler plugins to run at compile-time.
Every <code>java_plugin</code> specified in this attribute will be run whenever this rule
is built. A library may also inherit plugins from dependencies that use
<code><a href="#java_library.exported_plugins">exported_plugins</a></code>. Resources
generated by the plugin will be included in the resulting jar of this rule.
            """,
        ),
        "deploy_env": attr.label_list(
            providers = [java_common.JavaRuntimeClasspathInfo],
            allow_files = False,
            doc = """
A list of other <code>java_binary</code> targets which represent the deployment
environment for this binary.
Set this attribute when building a plugin which will be loaded by another
<code>java_binary</code>.<br/> Setting this attribute excludes all dependencies from
the runtime classpath (and the deploy jar) of this binary that are shared between this
binary and the targets specified in <code>deploy_env</code>.
            """,
        ),
        "launcher": attr.label(
            # TODO(b/295221112): add back CcLauncherInfo
            allow_files = False,
            doc = """
Specify a binary that will be used to run your Java program instead of the
normal <code>bin/java</code> program included with the JDK.
The target must be a <code>cc_binary</code>. Any <code>cc_binary</code> that
implements the
<a href="http://docs.oracle.com/javase/7/docs/technotes/guides/jni/spec/invocation.html">
Java Invocation API</a> can be specified as a value for this attribute.

<p>By default, Bazel will use the normal JDK launcher (bin/java or java.exe).</p>

<p>The related <a href="${link user-manual#flag--java_launcher}"><code>
--java_launcher</code></a> Bazel flag affects only those
<code>java_binary</code> and <code>java_test</code> targets that have
<i>not</i> specified a <code>launcher</code> attribute.</p>

<p>Note that your native (C++, SWIG, JNI) dependencies will be built differently
depending on whether you are using the JDK launcher or another launcher:</p>

<ul>
<li>If you are using the normal JDK launcher (the default), native dependencies are
built as a shared library named <code>{name}_nativedeps.so</code>, where
<code>{name}</code> is the <code>name</code> attribute of this java_binary rule.
Unused code is <em>not</em> removed by the linker in this configuration.</li>

<li>If you are using any other launcher, native (C++) dependencies are statically
linked into a binary named <code>{name}_nativedeps</code>, where <code>{name}</code>
is the <code>name</code> attribute of this java_binary rule. In this case,
the linker will remove any code it thinks is unused from the resulting binary,
which means any C++ code accessed only via JNI may not be linked in unless
that <code>cc_library</code> target specifies <code>alwayslink = 1</code>.</li>
</ul>

<p>When using any launcher other than the default JDK launcher, the format
of the <code>*_deploy.jar</code> output changes. See the main
<a href="#java_binary">java_binary</a> docs for details.</p>
            """,
        ),
        "bootclasspath": attr.label(
            providers = [BootClassPathInfo],
            flags = ["SKIP_CONSTRAINTS_OVERRIDE"],
            doc = "Restricted API, do not use!",
        ),
        "neverlink": attr.bool(),
        "javacopts": attr.string_list(
            doc = """
Extra compiler options for this binary.
Subject to <a href="make-variables.html">"Make variable"</a> substitution and
<a href="common-definitions.html#sh-tokenization">Bourne shell tokenization</a>.
<p>These compiler options are passed to javac after the global compiler options.</p>
            """,
        ),
        "add_exports": attr.string_list(
            doc = """
Allow this library to access the given <code>module</code> or <code>package</code>.
<p>
This corresponds to the javac and JVM --add-exports= flags.
            """,
        ),
        "add_opens": attr.string_list(
            doc = """
Allow this library to reflectively access the given <code>module</code> or
<code>package</code>.
<p>
This corresponds to the javac and JVM --add-opens= flags.
            """,
        ),
        "main_class": attr.string(
            doc = """
Name of class with <code>main()</code> method to use as entry point.
If a rule uses this option, it does not need a <code>srcs=[...]</code> list.
Thus, with this attribute one can make an executable from a Java library that already
contains one or more <code>main()</code> methods.
<p>
The value of this attribute is a class name, not a source file. The class must be
available at runtime: it may be compiled by this rule (from <code>srcs</code>) or
provided by direct or transitive dependencies (through <code>runtime_deps</code> or
<code>deps</code>). If the class is unavailable, the binary will fail at runtime; there
is no build-time check.
</p>
            """,
        ),
        "jvm_flags": attr.string_list(
            doc = """
A list of flags to embed in the wrapper script generated for running this binary.
Subject to <a href="${link make-variables#location}">$(location)</a> and
<a href="make-variables.html">"Make variable"</a> substitution, and
<a href="common-definitions.html#sh-tokenization">Bourne shell tokenization</a>.

<p>The wrapper script for a Java binary includes a CLASSPATH definition
(to find all the dependent jars) and invokes the right Java interpreter.
The command line generated by the wrapper script includes the name of
the main class followed by a <code>"$@"</code> so you can pass along other
arguments after the classname.  However, arguments intended for parsing
by the JVM must be specified <i>before</i> the classname on the command
line.  The contents of <code>jvm_flags</code> are added to the wrapper
script before the classname is listed.</p>

<p>Note that this attribute has <em>no effect</em> on <code>*_deploy.jar</code>
outputs.</p>
            """,
        ),
        "deploy_manifest_lines": attr.string_list(
            doc = """
A list of lines to add to the <code>META-INF/manifest.mf</code> file generated for the
<code>*_deploy.jar</code> target. The contents of this attribute are <em>not</em> subject
to <a href="make-variables.html">"Make variable"</a> substitution.
            """,
        ),
        "stamp": attr.int(
            default = -1,
            values = [-1, 0, 1],
            doc = """
Whether to encode build information into the binary. Possible values:
<ul>
<li>
  <code>stamp = 1</code>: Always stamp the build information into the binary, even in
  <a href="${link user-manual#flag--stamp}"><code>--nostamp</code></a> builds. <b>This
  setting should be avoided</b>, since it potentially kills remote caching for the
  binary and any downstream actions that depend on it.
</li>
<li>
  <code>stamp = 0</code>: Always replace build information by constant values. This
  gives good build result caching.
</li>
<li>
  <code>stamp = -1</code>: Embedding of build information is controlled by the
  <a href="${link user-manual#flag--stamp}"><code>--[no]stamp</code></a> flag.
</li>
</ul>
<p>Stamped binaries are <em>not</em> rebuilt unless their dependencies change.</p>
            """,
        ),
        "use_testrunner": attr.bool(
            default = False,
            doc = semantics.DOCS.for_attribute("use_testrunner") + """
<br/>
You can use this to override the default
behavior, which is to use test runner for
<code>java_test</code> rules,
and not use it for <code>java_binary</code> rules.  It is unlikely
you will want to do this.  One use is for <code>AllTest</code>
rules that are invoked by another rule (to set up a database
before running the tests, for example).  The <code>AllTest</code>
rule must be declared as a <code>java_binary</code>, but should
still use the test runner as its main entry point.

The name of a test runner class can be overridden with <code>main_class</code> attribute.
            """,
        ),
        "use_launcher": attr.bool(
            default = True,
            doc = """
Whether the binary should use a custom launcher.

<p>If this attribute is set to false, the
<a href="${link java_binary.launcher}">launcher</a> attribute  and the related
<a href="${link user-manual#flag--java_launcher}"><code>--java_launcher</code></a> flag
will be ignored for this target.
            """,
        ),
        "env": attr.string_dict(),
        "classpath_resources": attr.label_list(
            allow_files = True,
            doc = """
<em class="harmful">DO NOT USE THIS OPTION UNLESS THERE IS NO OTHER WAY)</em>
<p>
A list of resources that must be located at the root of the java tree. This attribute's
only purpose is to support third-party libraries that require that their resources be
found on the classpath as exactly <code>"myconfig.xml"</code>. It is only allowed on
binaries and not libraries, due to the danger of namespace conflicts.
</p>
            """,
        ),
        "licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
        "_stub_template": attr.label(
            default = semantics.JAVA_STUB_TEMPLATE_LABEL,
            allow_single_file = True,
        ),
        "_java_toolchain_type": attr.label(default = semantics.JAVA_TOOLCHAIN_TYPE),
        "_windows_constraints": attr.label_list(
            default = ["@" + paths.join(cc_semantics.get_platforms_root(), "os:windows")],
        ),
    } | ({} if _builtins.internal.java_common_internal_do_not_use.incompatible_disable_non_executable_java_binary() else {"create_executable": attr.bool(default = True, doc = "Deprecated, use <code>java_single_jar</code> instead.")}),
)

BASE_TEST_ATTRIBUTES = {
    "test_class": attr.string(
        doc = """
The Java class to be loaded by the test runner.<br/>
<p>
  By default, if this argument is not defined then the legacy mode is used and the
  test arguments are used instead. Set the <code>--nolegacy_bazel_java_test</code> flag
  to not fallback on the first argument.
</p>
<p>
  This attribute specifies the name of a Java class to be run by
  this test. It is rare to need to set this. If this argument is omitted,
  it will be inferred using the target's <code>name</code> and its
  source-root-relative path. If the test is located outside a known
  source root, Bazel will report an error if <code>test_class</code>
  is unset.
</p>
<p>
  For JUnit3, the test class needs to either be a subclass of
  <code>junit.framework.TestCase</code> or it needs to have a public
  static <code>suite()</code> method that returns a
  <code>junit.framework.Test</code> (or a subclass of <code>Test</code>).
  For JUnit4, the class needs to be annotated with
  <code>org.junit.runner.RunWith</code>.
</p>
<p>
  This attribute allows several <code>java_test</code> rules to
  share the same <code>Test</code>
  (<code>TestCase</code>, <code>TestSuite</code>, ...).  Typically
  additional information is passed to it
  (e.g. via <code>jvm_flags=['-Dkey=value']</code>) so that its
  behavior differs in each case, such as running a different
  subset of the tests.  This attribute also enables the use of
  Java tests outside the <code>javatests</code> tree.
</p>
        """,
    ),
    "env_inherit": attr.string_list(),
    "_apple_constraints": attr.label_list(
        default = [
            "@" + paths.join(cc_semantics.get_platforms_root(), "os:ios"),
            "@" + paths.join(cc_semantics.get_platforms_root(), "os:macos"),
            "@" + paths.join(cc_semantics.get_platforms_root(), "os:tvos"),
            "@" + paths.join(cc_semantics.get_platforms_root(), "os:visionos"),
            "@" + paths.join(cc_semantics.get_platforms_root(), "os:watchos"),
        ],
    ),
    "_legacy_any_type_attrs": attr.string_list(default = ["stamp"]),
}
