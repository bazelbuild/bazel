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

load(":common/rule_util.bzl", "merge_attrs")
load(":common/java/java_semantics.bzl", "semantics")
load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/java/java_helper.bzl", "helper")
load(":common/java/java_binary.bzl", "BASE_TEST_ATTRIBUTES", "BASIC_JAVA_BINARY_ATTRIBUTES", "basic_java_binary")
load(":common/paths.bzl", "paths")
load(":common/java/java_info.bzl", "JavaInfo")

def _bazel_java_binary_impl(ctx):
    return _bazel_base_binary_impl(ctx, is_test_rule_class = False) + helper.executable_providers(ctx)

def _bazel_java_test_impl(ctx):
    return _bazel_base_binary_impl(ctx, is_test_rule_class = True) + helper.test_providers(ctx)

def _bazel_base_binary_impl(ctx, is_test_rule_class):
    deps = _collect_all_targets_as_deps(ctx, classpath_type = "compile_only")
    runtime_deps = _collect_all_targets_as_deps(ctx)

    main_class = _check_and_get_main_class(ctx)
    coverage_main_class = main_class
    coverage_config = helper.get_coverage_config(ctx, _get_coverage_runner(ctx))
    if coverage_config:
        main_class = coverage_config.main_class

    launcher_info = _get_launcher_info(ctx)

    executable = _get_executable(ctx)

    feature_config = helper.get_feature_config(ctx)
    strip_as_default = helper.should_strip_as_default(ctx, feature_config)

    providers, default_info, jvm_flags = basic_java_binary(
        ctx,
        deps,
        runtime_deps,
        ctx.files.resources,
        main_class,
        coverage_main_class,
        coverage_config,
        launcher_info,
        executable,
        feature_config,
        strip_as_default,
        is_test_rule_class = is_test_rule_class,
    )

    if ctx.attr.use_testrunner:
        if semantics.find_java_runtime_toolchain(ctx).version >= 17:
            jvm_flags.append("-Djava.security.manager=allow")
        test_class = ctx.attr.test_class if hasattr(ctx.attr, "test_class") else ""
        if test_class == "":
            test_class = helper.primary_class(ctx)
        if test_class == None:
            fail("cannot determine test class")
        jvm_flags.extend([
            "-ea",
            "-Dbazel.test_suite=" + helper.shell_quote(test_class),
        ])

    java_attrs = providers["InternalDeployJarInfo"].java_attrs

    if executable:
        _create_stub(ctx, java_attrs, launcher_info.launcher, executable, jvm_flags, main_class, coverage_main_class)

    runfiles = default_info.runfiles

    if executable:
        runtime_toolchain = semantics.find_java_runtime_toolchain(ctx)
        runfiles = runfiles.merge(ctx.runfiles(transitive_files = runtime_toolchain.files))

    test_support = helper.get_test_support(ctx)
    if test_support:
        runfiles = runfiles.merge(test_support[DefaultInfo].default_runfiles)

    providers["DefaultInfo"] = DefaultInfo(
        files = default_info.files,
        runfiles = runfiles,
        executable = default_info.executable,
    )

    return providers.values()

def _get_coverage_runner(ctx):
    if ctx.configuration.coverage_enabled and ctx.attr.create_executable:
        toolchain = semantics.find_java_toolchain(ctx)
        runner = toolchain.jacocorunner
        if not runner:
            fail("jacocorunner not set in java_toolchain: %s" % toolchain.label)
        runner_jar = runner.executable

        # wrap the jar in JavaInfo so we can add it to deps for java_common.compile()
        return JavaInfo(output_jar = runner_jar, compile_jar = runner_jar)

    return None

def _collect_all_targets_as_deps(ctx, classpath_type = "all"):
    deps = helper.collect_all_targets_as_deps(ctx, classpath_type = classpath_type)

    if classpath_type == "compile_only" and ctx.fragments.java.enforce_explicit_java_test_deps():
        return deps

    test_support = helper.get_test_support(ctx)
    if test_support:
        deps.append(test_support)
    return deps

def _check_and_get_main_class(ctx):
    create_executable = ctx.attr.create_executable
    main_class = _get_main_class(ctx)

    if not create_executable and main_class:
        fail("main class must not be specified when executable is not created")
    if create_executable and not main_class:
        if not ctx.attr.srcs:
            fail("need at least one of 'main_class' or Java source files")
        main_class = helper.primary_class(ctx)
        if main_class == None:
            fail("main_class was not provided and cannot be inferred: " +
                 "source path doesn't include a known root (java, javatests, src, testsrc)")

    return _get_main_class(ctx)

def _get_main_class(ctx):
    if not ctx.attr.create_executable:
        return None

    main_class = _get_main_class_from_rule(ctx)

    if main_class == "":
        main_class = helper.primary_class(ctx)
    return main_class

def _get_main_class_from_rule(ctx):
    main_class = ctx.attr.main_class
    if main_class:
        return main_class
    if ctx.attr.use_testrunner:
        return "com.google.testing.junit.runner.BazelTestRunner"
    return main_class

def _get_launcher_info(ctx):
    launcher = helper.launcher_artifact_for_target(ctx)
    return struct(
        launcher = launcher,
        unstripped_launcher = launcher,
        runfiles = [],
        runtime_jars = [],
        jvm_flags = [],
        classpath_resources = [],
    )

def _get_executable(ctx):
    if not ctx.attr.create_executable:
        return None
    executable_name = ctx.label.name
    if helper.is_windows(ctx):
        executable_name = executable_name + ".exe"

    return ctx.actions.declare_file(executable_name)

def _create_stub(ctx, java_attrs, launcher, executable, jvm_flags, main_class, coverage_main_class):
    java_runtime_toolchain = semantics.find_java_runtime_toolchain(ctx)
    java_executable = helper.get_java_executable(ctx, java_runtime_toolchain, launcher)
    workspace_name = ctx.workspace_name
    workspace_prefix = workspace_name + ("/" if workspace_name else "")
    runfiles_enabled = helper.runfiles_enabled(ctx)
    coverage_enabled = ctx.configuration.coverage_enabled

    test_support = helper.get_test_support(ctx)
    test_support_jars = test_support[JavaInfo].transitive_runtime_jars if test_support else depset()
    classpath = depset(
        transitive = [
            java_attrs.runtime_classpath,
            test_support_jars if ctx.fragments.java.enforce_explicit_java_test_deps() else depset(),
        ],
    )

    if helper.is_windows(ctx):
        jvm_flags_for_launcher = []
        for flag in jvm_flags:
            jvm_flags_for_launcher.extend(ctx.tokenize(flag))
        return _create_windows_exe_launcher(ctx, java_executable, classpath, main_class, jvm_flags_for_launcher, runfiles_enabled, executable)

    if runfiles_enabled:
        prefix = "" if helper.is_absolute_path(ctx, java_executable) else "${JAVA_RUNFILES}/"
        java_bin = "JAVABIN=${JAVABIN:-" + prefix + java_executable + "}"
    else:
        java_bin = "JAVABIN=${JAVABIN:-$(rlocation " + java_executable + ")}"

    td = ctx.actions.template_dict()
    td.add_joined(
        "%classpath%",
        classpath,
        map_each = lambda file: _format_classpath_entry(runfiles_enabled, workspace_prefix, file),
        join_with = ctx.configuration.host_path_separator,
        format_joined = "\"%s\"",
        allow_closure = True,
    )

    ctx.actions.expand_template(
        template = ctx.file._stub_template,
        output = executable,
        substitutions = {
            "%runfiles_manifest_only%": "" if runfiles_enabled else "1",
            "%workspace_prefix%": workspace_prefix,
            "%javabin%": java_bin,
            "%needs_runfiles%": "0" if helper.is_absolute_path(ctx, java_runtime_toolchain.java_executable_exec_path) else "1",
            "%set_jacoco_metadata%": "",
            "%set_jacoco_main_class%": "export JACOCO_MAIN_CLASS=" + coverage_main_class if coverage_enabled else "",
            "%set_jacoco_java_runfiles_root%": "export JACOCO_JAVA_RUNFILES_ROOT=${JAVA_RUNFILES}/" + workspace_prefix if coverage_enabled else "",
            "%java_start_class%": helper.shell_quote(main_class),
            "%jvm_flags%": " ".join(jvm_flags),
        },
        computed_substitutions = td,
        is_executable = True,
    )
    return executable

def _format_classpath_entry(runfiles_enabled, workspace_prefix, file):
    if runfiles_enabled:
        return "${RUNPATH}" + file.short_path

    return "$(rlocation " + paths.normalize(workspace_prefix + file.short_path) + ")"

def _create_windows_exe_launcher(ctx, java_executable, classpath, main_class, jvm_flags_for_launcher, runfiles_enabled, executable):
    launch_info = ctx.actions.args().use_param_file("%s", use_always = True).set_param_file_format("multiline")
    launch_info.add("binary_type=Java")
    launch_info.add(ctx.workspace_name, format = "workspace_name=%s")
    launch_info.add("1" if runfiles_enabled else "0", format = "symlink_runfiles_enabled=%s")
    launch_info.add(java_executable, format = "java_bin_path=%s")
    launch_info.add(main_class, format = "java_start_class=%s")
    launch_info.add_joined(classpath, map_each = _short_path, join_with = ";", format_joined = "classpath=%s", omit_if_empty = False)
    launch_info.add_joined(jvm_flags_for_launcher, join_with = "\t", format_joined = "jvm_flags=%s", omit_if_empty = False)
    jar_bin_path = semantics.find_java_runtime_toolchain(ctx).java_home + "/bin/jar.exe"
    launch_info.add(jar_bin_path, format = "jar_bin_path=%s")
    launcher_artifact = ctx.executable._launcher
    ctx.actions.run(
        executable = ctx.executable._windows_launcher_maker,
        inputs = [launcher_artifact],
        outputs = [executable],
        arguments = [launcher_artifact.path, launch_info, executable.path],
        use_default_shell_env = True,
    )
    return executable

def _short_path(file):
    return file.short_path

def _compute_test_support(use_testrunner):
    return Label(semantics.JAVA_TEST_RUNNER_LABEL) if use_testrunner else None

def _compute_launcher_attr(launcher):
    return launcher

def _make_binary_rule(implementation, attrs, executable = False, test = False):
    return rule(
        implementation = implementation,
        attrs = attrs,
        executable = executable,
        test = test,
        fragments = ["cpp", "java"],
        provides = [JavaInfo],
        toolchains = [semantics.JAVA_TOOLCHAIN] + cc_helper.use_cpp_toolchain() + (
            [semantics.JAVA_RUNTIME_TOOLCHAIN] if executable or test else []
        ),
        # TODO(hvd): replace with filegroups?
        outputs = {
            "classjar": "%{name}.jar",
            "sourcejar": "%{name}-src.jar",
            "deploysrcjar": "%{name}_deploy-src.jar",
        },
        exec_groups = {
            "cpp_link": exec_group(toolchains = cc_helper.use_cpp_toolchain()),
        },
    )

_BASE_BINARY_ATTRS = merge_attrs(
    BASIC_JAVA_BINARY_ATTRIBUTES,
    {
        "_test_support": attr.label(default = _compute_test_support),
        "_launcher": attr.label(
            cfg = "exec",
            executable = True,
            default = "@bazel_tools//tools/launcher:launcher",
        ),
        "resource_strip_prefix": attr.string(),
        "_windows_launcher_maker": attr.label(
            default = "@bazel_tools//tools/launcher:launcher_maker",
            cfg = "exec",
            executable = True,
        ),
    },
)

def make_java_binary(executable, resolve_launcher_flag, has_launcher = False):
    return _make_binary_rule(
        _bazel_java_binary_impl,
        merge_attrs(
            _BASE_BINARY_ATTRS,
            {
                "_java_launcher": attr.label(
                    default = configuration_field(
                        fragment = "java",
                        name = "launcher",
                    ) if resolve_launcher_flag else (_compute_launcher_attr if has_launcher else None),
                ),
                "_use_auto_exec_groups": attr.bool(default = True),
            },
            ({} if executable else {
                "args": attr.string_list(),
                "output_licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
            }),
            remove_attrs = [] if executable else ["_java_runtime_toolchain_type"],
        ),
        executable = executable,
    )

java_binary = make_java_binary(executable = True, resolve_launcher_flag = True)

def make_java_test(resolve_launcher_flag, has_launcher = False):
    return _make_binary_rule(
        _bazel_java_test_impl,
        merge_attrs(
            BASE_TEST_ATTRIBUTES,
            _BASE_BINARY_ATTRS,
            {
                "_java_launcher": attr.label(
                    default = configuration_field(
                        fragment = "java",
                        name = "launcher",
                    ) if resolve_launcher_flag else (_compute_launcher_attr if has_launcher else None),
                ),
                "_lcov_merger": attr.label(
                    cfg = "exec",
                    default = configuration_field(
                        fragment = "coverage",
                        name = "output_generator",
                    ),
                ),
                "_collect_cc_coverage": attr.label(
                    cfg = "exec",
                    allow_single_file = True,
                    default = "@bazel_tools//tools/test:collect_cc_coverage",
                ),
            },
            override_attrs = {
                "use_testrunner": attr.bool(default = True),
                "stamp": attr.int(default = 0, values = [-1, 0, 1]),
            },
            remove_attrs = ["deploy_env"],
        ),
        test = True,
    )

java_test = make_java_test(True)
