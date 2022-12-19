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
"""Implementation for Bazel Python executable."""

load(":common/python/attributes_bazel.bzl", "IMPORTS_ATTRS")
load(":common/python/common_bazel.bzl", "collect_cc_info")
load(":common/python/providers.bzl", "DEFAULT_STUB_SHEBANG")
load(":common/python/py_executable.bzl", "py_executable_base_impl")
load(":common/python/semantics.bzl", "TOOLS_REPO")

BAZEL_EXECUTABLE_ATTRS = union_attrs(
    IMPORTS_ATTRS,
    {
        "_zipper": attr.label(cfg = "exec"),
        "_launcher": attr.label(
            cfg = "target",
            default = "@" + TOOLS_REPO + "//tools/launcher:launcher",
        ),
        "_windows_launcher_maker": attr.label(
            default = "@" + TOOLS_REPO + "//tools/launcher:launcher_maker",
            cfg = "exec",
            executable = True,
        ),
        # TODO: Remove this attribute; it's basically a no-op in Bazel due
        # to toolchain resolution.
        "_py_interpreter": attr.label(),
    },
)

def create_executable_rule(*, attrs, **kwargs):
    return create_base_executable_rule(
        attrs = BAZEL_EXECUTABLE_ATTRS | attrs,
        fragments = ["py", "bazel_py"],
        **kwargs
    )

def py_executable_impl(ctx, *, is_test, inherited_environment):
    result = py_executable_base_impl(
        ctx = ctx,
        semantics = create_binary_semantics_bazel(),
        is_test = is_test,
        inherited_environment = inherited_environment,
    )
    return struct(
        providers = result.providers,
        **result.legacy_providers
    )

def create_binary_semantics_bazel():
    return create_binary_semantics_bazel_common(
        # keep-sorted start
        create_executable = _create_executable,
        get_cc_details_for_binary = _get_cc_details_for_binary,
        get_interpreter_path = _get_interpreter_path,
        get_native_deps_dso_name = _get_native_deps_dso_name,
        get_native_deps_user_link_flags = _get_native_deps_user_link_flags,
        should_build_native_deps_dso = _should_build_native_deps_dso,
        # keep-sorted end
    )

def _create_executable(
        ctx,
        *,
        executable,
        main_py,
        imports,
        is_test,
        runtime_details,
        cc_details,
        native_deps_details,
        runfiles_details):
    _ = is_test, cc_details, native_deps_details  # @unused

    common_bootstrap_template_kwargs = dict(
        main_py = main_py,
        imports = imports,
        runtime_details = runtime_details,
    )

    # TODO: This should use the configuration instead of the Bazel OS.
    # This is just legacy behavior.
    is_windows = _py_builtins.get_current_os_name() == "windows"

    if is_windows:
        if not executable.name.extension == "exe":
            fail("Should not happen: somehow we are generating a non-.exe file on windows")
        base_executable_name = executable.name[0:-4]
    else:
        base_executable_name = executable.name

    zip_bootstrap = ctx.actions.declare(base_executable_name + ".temp")
    zip_file = ctx.actions.declare(base_executable_name + ".zip")

    _expand_bootstrap_template(
        ctx,
        output = zip_bootstrap,
        is_for_zip = True,
        **common_bootstrap_template_kwargs
    )
    _create_zip_file(
        ctx,
        output = zip_file,
        original_nonzip_executable = executable,
        zipfile_executable = zip_bootstrap,
        runfiles = runfiles_details.default_runfiles,
    )

    build_zip_enabled = ctx.fragments.py.build_python_zip

    # When --build_python_zip is enabled, then the zip file becomes
    # one of the default outputs.
    if build_zip_enabled:
        extra_files_to_build.append(zip_file)

    # The logic here is a bit convoluted. Essentially, there are 3 types of
    # executables produced:
    # 1. A bootstrap template based program.
    # 2. A self-executable zip file of a bootstrap template based program.
    # 3. For Windows, a native Windows executable that finds and launches
    #    the actual underlying Bazel program (one of the above). Note that
    #    it implicitly assumes one of the above is located next to it.

    should_create_executable_zip = False
    bootstrap_output = None
    if not is_windows:
        if build_zip_enabled:
            should_create_executable_zip = True
        else:
            bootstrap_output = executable
    else:
        _create_windows_exe_launcher(
            ctx,
            output = executable,
            is_for_zip = build_zip_enabled,
            python_binary_path = runtime_details.executable_interpreter_path,
        )
        if not build_zip_enabled:
            # On Windows, the main executable has an "exe" extension, so
            # here we re-use the un-extensioned name for the bootstrap output.
            bootstrap_output = ctx.actions.declare_file(base_executable_name)

            # The launcher looks for the non-zip executable next to
            # itself, so add it to the default outputs.
            extra_files_to_build.append(bootstrap_output)

    if should_create_executable_zip:
        if bootstrap_template != None:
            fail("Should not occur: bootstrap_template should not be used " +
                 "when creating an executable zip")
        _create_executable_zip_file(ctx, output = executable, zip_file = zip_file)
    else:
        if bootstrap_template == None:
            fail("Should not occur: bootstrap_template should set when " +
                 "build a bootstrap-template-based executable")
        _expand_bootstrap_template(
            ctx,
            output = bootstrap_template,
            is_for_zip = build_zip_enabled,
            **common_bootstrap_template_kwargs
        )

    return create_executable_result_struct(
        extra_files_to_build = depset(extra_files_to_build),
        output_groups = {"python_zip_file": depset([zip_file])},
    )

def _expand_bootstrap_template(
        ctx,
        *,
        output,
        main_py,
        imports,
        is_for_zip,
        runtime_details):
    runtime = runtime_details.effective_runtime
    if (ctx.configuration.coverage_enabled and
        runtime and
        runtime.coverage_tool):
        coverage_tool_runfiles_path = "{}/{}".format(
            ctx.workspace_name,
            runtime.coverage_tool.short_path,
        )
    else:
        coverage_tool_runfiles_path = ""

    if runtime:
        shebang = runtime.stub_shebang
    else:
        shebang = DEFAULT_STUB_SHEBANG

    ctx.actions.expand_template(
        template = ctx.file._bootstrap_template,
        output = output,
        substitutions = {
            "%shebang": shebang,
            "%main": main_py.short_path,
            "%python_binary%": runtime_details.executable_interpreter_path,
            "%coverage_tool%": coverage_tool_runfiles_path,
            "%imports%": ":".join(imports.to_list()),
            "%workspace_name%": ctx.workspace_name,
            "%is_zipfile%": "True" if is_for_zip else "False",
            "%import_all%": "True" if ctx.fragments.bazel_py.python_import_all_repositories else "False",
            "%target%": str(ctx.label),
        },
        is_executable = True,
    )

def _create_windows_exe_launcher(
        ctx,
        *,
        output,
        python_binary_path,
        use_zip_file):
    launch_info = ctx.actions.args()
    launch_info.use_param_file("%s", use_always = True)
    launch_info.set_param_file_format("multiline")
    launch_info.add("binary_type=Python")
    launch_info.add(ctx.workspace_name, format = "workspace_name=%s")
    launch_info.add(
        "1" if ctx.configuration.runfiles_enabled() else "0",
        format = "symlink_runfiles_enabled=%s",
    )
    launch_info.add(python_binary_path, format = "python_bin_path=%s")
    launch_info.add("1" if use_zip_file else "0", format = "use_zip_file=%s")

    ctx.actions.run(
        executable = ctx.executable._windows_launcher_maker,
        arguments = [launch_info],
        inputs = [ctx.file._launcher],
        outputs = [output],
        mnemonic = "PyBuildLauncher",
        progress_message = "Creating launcher for %{label}",
    )

def _create_zip_file(ctx, *, zip_file, original_nonzip_executable, zip_executable, runfiles):
    workspace_name = ctx.workspace_name
    legacy_external_runfiles = _py_builtins.get_legacy_external_runfiles(ctx)

    manifest = ctx.actions.args()
    manifest.use_param_file("%s", use_always = True)
    manifest.set_param_file_format("multiline")

    manifest.add("__main__.py=%s", zip_executable)
    manifest.add("__init__.py=")
    manifest.add(
        "%s=",
        _get_zip_runfiles_path("__init__.py", workspace_name, legacy_external_runfiles),
    )
    for path in runfiles.empty_filenames.to_list():
        manifest.add("%s=", _get_zip_runfiles_path(path, workspace_name, legacy_external_runfiles))

    def map_zip_runfiles(file):
        if file != original_nonzip_executable and file != zip_file:
            return "{}={}".format(
                _get_zip_runfiles_path(file.short_path, workspace_name, legacy_external_runfiles),
                file.path,
            )
        else:
            return None

    manifest.add_all(runfiles.files, map_each = map_zip_runfiles, allow_closure = True)
    inputs = []
    for artifact in runfiles.files.to_list():
        # Don't include the original executable because it isn't used by the
        # zip file, so no need to build it for the action.
        # Don't include the zipfile itself because it's an output.
        if artifact != original_nonzip_executable and artifact != zip_file:
            inputs.append(artifact)

    zip_cli_args = ctx.actions.args()
    zip_cli_args.add("cC")
    zip_cli_args.add(zip_file)
    ctx.actions.run(
        executable = ctx.executable._zipper,
        arguments = [zip_cli_args, manifest],
        inputs = depset(inputs),
        outputs = [zip_file],
        use_default_shell_env = True,
        mnemonic = "PythonZipper",
        progress_message = "Building Python zip: %{label}",
    )

def _get_zip_runfiles_path(path, workspace_name, legacy_external_runfiles):
    if legacy_external_runfiles and path.startswith(_EXTERNAL_PATH_PREFIX):
        zip_runfiles_path = path.relativeTo(EXTERNAL_PATH_PREFIX)
    else:
        zip_runfiles_path = "{}/{}".format(workspace_name, path)
    return ZIP_RUNFILES_DIRECTORY_NAME.getRelative(zip_runfiles_path)

def _create_executable_zip_file(ctx, *, output, zip_file):
    ctx.actions.run_shell(
        command = "echo '{shebang}' | cat - {zip} > {output}".format(
            shebang = "#!/usr/bin/env python3",
            zip = zip_file.path,
            output = output.path,
        ),
        inputs = [zip_file],
        outputs = [output],
        use_default_shell_env = True,
        mnemonic = "BuildBinary",
        progress_message = "Build Python zip executable: %{label}",
    )

def _get_cc_details_for_binary(ctx, extra_deps):
    cc_info = collect_cc_info(ctx, extra_deps = extra_deps)
    return create_cc_details_struct(
        cc_info_for_propagating = cc_info,
        cc_info_for_self_link = cc_info,
        cc_info_with_extra_link_time_libraries = None,
        extra_runfiles = ctx.runfiles(),
        # Though the rules require the CcToolchain, it isn't actually used.
        cc_toolchain = None,
    )

def _get_interpreter_path(ctx, *, runtime, flag_interpreter_path):
    _ = ctx  # @unused
    if runtime:
        if runtime.interpreter_path:
            interpreter_path = runtime.interpreter_path
        else:
            interpreter_path = runtime.interpreter.short_path
    elif flag_interpreter_path:
        interpreter_path = flag_interpreter_path
    else:
        fail("Unable to determine interpreter path")

    return interpreter_path

def _should_build_native_deps_dso(ctx):
    _ = ctx  # @unused
    return False

def _get_native_deps_dso_name(ctx):
    _ = ctx  # @unused
    fail("Building native deps DSO not supported.")

def _get_native_deps_user_link_flags(ctx):
    _ = ctx  # @unused
    fail("Building native deps DSO not supported.")
