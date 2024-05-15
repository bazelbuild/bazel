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

"""Auxiliary rule to create the deploy archives for java_binary"""

load(":common/java/java_helper.bzl", "helper")
load(":common/java/java_semantics.bzl", "semantics")

InstrumentedFilesInfo = _builtins.toplevel.InstrumentedFilesInfo

def _stamping_enabled(ctx, stamp):
    if ctx.configuration.is_tool_configuration():
        stamp = 0
    return (stamp == 1) or (stamp == -1 and ctx.configuration.stamp_binaries())

def get_build_info(ctx, stamp):
    if _stamping_enabled(ctx, stamp):
        # Makes the target depend on BUILD_INFO_KEY, which helps to discover stamped targets
        # See b/326620485 for more details.
        ctx.version_file  # buildifier: disable=no-effect
        return ctx.attr._build_info_translator[OutputGroupInfo].non_redacted_build_info_files.to_list()
    else:
        return ctx.attr._build_info_translator[OutputGroupInfo].redacted_build_info_files.to_list()

def create_deploy_archives(
        ctx,
        java_attrs,
        launcher_info,
        main_class,
        coverage_main_class,
        strip_as_default,
        hermetic = False,
        add_exports = depset(),
        add_opens = depset(),
        shared_archive = None,
        one_version_level = "OFF",
        one_version_allowlist = None,
        extra_args = [],
        extra_manifest_lines = []):
    """ Registers actions for _deploy.jar and _deploy.jar.unstripped

    Args:
        ctx: (RuleContext) The rule context
        java_attrs: (Struct) Struct of (classpath_resources, runtime_jars, runtime_classpath_for_archive, resources)
        launcher_info: (Struct) Struct of (runtime_jars, launcher, unstripped_launcher)
        main_class: (String) FQN of the entry point for execution
        coverage_main_class: (String) FQN of the entry point for coverage collection
        build_target: (String) Name of the build target for stamping
        strip_as_default: (bool) Whether to create unstripped deploy jar
        hermetic: (bool)
        add_exports: (depset)
        add_opens: (depset)
        shared_archive: (File) Optional .jsa artifact
        one_version_level: (String) Optional one version check level, default OFF
        one_version_allowlist: (File) Optional allowlist for one version check
        extra_args: (list[Args]) Optional arguments for the deploy jar action
        extra_manifest_lines: (list[String]) Optional lines added to the jar manifest
    """
    classpath_resources = java_attrs.classpath_resources

    runtime_classpath = depset(
        direct = launcher_info.runtime_jars,
        transitive = [
            java_attrs.runtime_jars,
            java_attrs.runtime_classpath_for_archive,
        ],
        order = "preorder",
    )
    multi_release = ctx.fragments.java.multi_release_deploy_jars
    build_info_files = get_build_info(ctx, ctx.attr.stamp)
    build_target = str(ctx.label)
    manifest_lines = ctx.attr.deploy_manifest_lines + extra_manifest_lines
    create_deploy_archive(
        ctx,
        launcher_info.launcher,
        main_class,
        coverage_main_class,
        java_attrs.resources,
        classpath_resources,
        runtime_classpath,
        manifest_lines,
        build_info_files,
        build_target,
        output = ctx.outputs.deployjar,
        shared_archive = shared_archive,
        one_version_level = one_version_level,
        one_version_allowlist = one_version_allowlist,
        multi_release = multi_release,
        hermetic = hermetic,
        add_exports = add_exports,
        add_opens = add_opens,
        extra_args = extra_args,
    )

    if strip_as_default:
        create_deploy_archive(
            ctx,
            launcher_info.unstripped_launcher,
            main_class,
            coverage_main_class,
            java_attrs.resources,
            classpath_resources,
            runtime_classpath,
            manifest_lines,
            build_info_files,
            build_target,
            output = ctx.outputs.unstrippeddeployjar,
            multi_release = multi_release,
            hermetic = hermetic,
            add_exports = add_exports,
            add_opens = add_opens,
            extra_args = extra_args,
        )
    else:
        ctx.actions.write(ctx.outputs.unstrippeddeployjar, "")

def create_deploy_archive(
        ctx,
        launcher,
        main_class,
        coverage_main_class,
        resources,
        classpath_resources,
        runtime_classpath,
        manifest_lines,
        build_info_files,
        build_target,
        output,
        shared_archive = None,
        one_version_level = "OFF",
        one_version_allowlist = None,
        multi_release = False,
        hermetic = False,
        add_exports = [],
        add_opens = [],
        extra_args = []):
    """ Creates a deploy jar

    Requires a Java runtime toolchain if and only if hermetic is True.

    Args:
        ctx: (RuleContext) The rule context
        launcher: (File) the launcher artifact
        main_class: (String) FQN of the entry point for execution
        coverage_main_class: (String) FQN of the entry point for coverage collection
        resources: (Depset) resource inputs
        classpath_resources: (Depset) classpath resource inputs
        runtime_classpath: (Depset) source files to add to the jar
        build_target: (String) Name of the build target for stamping
        manifest_lines: (list[String]) Optional lines added to the jar manifest
        build_info_files: (list[File]) build info files for stamping
        build_target: (String) the owner build target label name string
        output: (File) the output jar artifact
        shared_archive: (File) Optional .jsa artifact
        one_version_level: (String) Optional one version check level, default OFF
        one_version_allowlist: (File) Optional allowlist for one version check
        multi_release: (bool)
        hermetic: (bool)
        add_exports: (depset)
        add_opens: (depset)
        extra_args: (list[Args]) Optional arguments for the deploy jar action
    """
    input_files = []
    input_files.extend(build_info_files)

    transitive_input_files = [
        resources,
        classpath_resources,
        runtime_classpath,
    ]

    single_jar = semantics.find_java_toolchain(ctx).single_jar

    manifest_lines = list(manifest_lines)
    if ctx.configuration.coverage_enabled:
        manifest_lines.append("Coverage-Main-Class: %s" % coverage_main_class)

    args = ctx.actions.args()
    args.set_param_file_format("shell").use_param_file("@%s", use_always = True)

    args.add("--output", output)
    args.add("--build_target", build_target)
    args.add("--normalize")
    args.add("--compression")
    if main_class:
        args.add("--main_class", main_class)
    args.add_all("--deploy_manifest_lines", manifest_lines)
    args.add_all(build_info_files, before_each = "--build_info_file")
    if launcher:
        input_files.append(launcher)
        args.add("--java_launcher", launcher)
    args.add_all("--classpath_resources", classpath_resources)
    args.add_all(
        "--sources",
        runtime_classpath,
        map_each = helper.jar_and_target_arg_mapper,
    )

    if one_version_level != "OFF" and one_version_allowlist:
        input_files.append(one_version_allowlist)
        args.add("--enforce_one_version")
        args.add("--one_version_whitelist", one_version_allowlist)
        if one_version_level == "WARNING":
            args.add("--succeed_on_found_violations")

    if multi_release:
        args.add("--multi_release")

    if hermetic:
        runtime = ctx.toolchains["@//tools/jdk/hermetic:hermetic_runtime_toolchain_type"].java_runtime
        if runtime.lib_modules != None:
            java_home = runtime.java_home
            lib_modules = runtime.lib_modules
            hermetic_files = runtime.hermetic_files
            args.add("--hermetic_java_home", java_home)
            args.add("--jdk_lib_modules", lib_modules)
            args.add_all("--resources", hermetic_files)
            input_files.append(lib_modules)
            transitive_input_files.append(hermetic_files)

            if shared_archive == None:
                shared_archive = runtime.default_cds

    if shared_archive:
        input_files.append(shared_archive)
        args.add("--cds_archive", shared_archive)

    args.add_all("--add_exports", add_exports)
    args.add_all("--add_opens", add_opens)

    inputs = depset(input_files, transitive = transitive_input_files)

    ctx.actions.run(
        mnemonic = "JavaDeployJar",
        progress_message = "Building deploy jar %s" % output.short_path,
        executable = single_jar,
        inputs = inputs,
        tools = [single_jar],
        outputs = [output],
        arguments = [args] + extra_args,
        use_default_shell_env = True,
        toolchain = semantics.JAVA_TOOLCHAIN_TYPE,
    )
