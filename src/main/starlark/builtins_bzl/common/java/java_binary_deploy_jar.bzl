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

"""Auxiliary rule to create the deploy archives for java_binary

This needs to be a separate rule because we need to add the runfiles manifest as an input to
the generating actions, so that the runfiles symlink tree is staged for the deploy jars.
"""

load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/java/java_semantics.bzl", "semantics")
load(":common/cc/semantics.bzl", cc_semantics = "semantics")
load(":common/java/java_helper.bzl", "util")

InstrumentedFilesInfo = _builtins.toplevel.InstrumentedFilesInfo
java_common = _builtins.toplevel.java_common

def create_deploy_archives(
        ctx,
        java_attrs,
        launcher_info,
        runfiles,
        main_class,
        coverage_main_class,
        strip_as_default,
        build_info_files,
        build_target,
        hermetic = False,
        add_exports = depset(),
        add_opens = depset(),
        shared_archive = None,
        one_version_level = "OFF",
        one_version_allowlist = None,
        extra_args = [],
        manifest_lines = []):
    """ Registers actions for _deploy.jar and _deploy.jar.unstripped

    Args:
        ctx: (RuleContext) The rule context
        java_attrs: (Struct) Struct of (classpath_resources, runtime_jars, runtime_classpath_for_archive, resources)
        launcher_info: (Struct) Struct of (runtime_jars, launcher, unstripped_launcher)
        runfiles: (Depset) the runfiles for the deploy jar
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
        manifest_lines: (list[String]) Optional lines added to the jar manifest
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

    _create_deploy_archive(
        ctx,
        launcher_info.launcher,
        runfiles,
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
        _create_deploy_archive(
            ctx,
            launcher_info.unstripped_launcher,
            runfiles,
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
            extra_args = extra_args,
        )
    else:
        ctx.actions.write(ctx.outputs.unstrippeddeployjar, "")

def _create_deploy_archive(
        ctx,
        launcher,
        runfiles,
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
    runtime = semantics.find_java_runtime_toolchain(ctx)

    input_files = []
    input_files.extend(build_info_files)

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
        map_each = util.jar_and_target_arg_mapper,
    )

    if one_version_level != "OFF" and one_version_allowlist:
        input_files.append(one_version_allowlist)
        args.add("--enforce_one_version")
        args.add("--one_version_whitelist", one_version_allowlist)
        if one_version_level == "WARNING":
            args.add("--succeed_on_found_violations")

    if multi_release:
        args.add("--multi_release")

    hermetic_files = runtime.hermetic_files
    if hermetic and runtime.lib_modules != None and hermetic_files != None:
        java_home = runtime.java_home
        lib_modules = runtime.lib_modules
        args.add("--hermetic_java_home", java_home)
        args.add("--jdk_lib_modules", lib_modules)
        args.add_all("--resources", hermetic_files)
        input_files.append(lib_modules)

        if shared_archive == None:
            shared_archive = runtime.default_cds

    if shared_archive:
        input_files.append(shared_archive)
        args.add("--cds_archive", shared_archive)

    args.add_all("--add_exports", add_exports)
    args.add_all("--add_opens", add_opens)

    inputs = depset(input_files, transitive = [
        resources,
        classpath_resources,
        runtime_classpath,
        runfiles,
        hermetic_files,
    ])

    ctx.actions.run(
        mnemonic = "JavaDeployJar",
        progress_message = "Building deploy jar %s" % output.short_path,
        executable = single_jar,
        inputs = inputs,
        tools = [single_jar],
        outputs = [output],
        arguments = [args] + extra_args,
        use_default_shell_env = True,
    )
    return output

def _implicit_outputs(binary):
    binary_name = binary.name
    return {
        "deployjar": "%s_deploy.jar" % binary_name,
        "unstrippeddeployjar": "%s_deploy.jar.unstripped" % binary_name,
    }

def make_deploy_jars_rule(implementation):
    """Creates the deploy jar auxiliary rule for java_binary

    Args:
        implementation: (Function) The rule implementation function

    Returns:
        The deploy jar rule class
    """
    return rule(
        implementation = implementation,
        attrs = {
            "binary": attr.label(mandatory = True),
            # TODO(b/245144242): Used by IDE integration, remove when toolchains are used
            "_java_toolchain": attr.label(
                default = semantics.JAVA_TOOLCHAIN_LABEL,
                providers = [java_common.JavaToolchainInfo],
            ),
            "_cc_toolchain": attr.label(default = "@" + cc_semantics.get_repo() + "//tools/cpp:current_cc_toolchain"),
            "_java_toolchain_type": attr.label(default = semantics.JAVA_TOOLCHAIN_TYPE),
            "_java_runtime_toolchain_type": attr.label(default = semantics.JAVA_RUNTIME_TOOLCHAIN_TYPE),
        },
        outputs = _implicit_outputs,
        fragments = ["java"],
        toolchains = [semantics.JAVA_TOOLCHAIN, semantics.JAVA_RUNTIME_TOOLCHAIN] + cc_helper.use_cpp_toolchain(),
    )
