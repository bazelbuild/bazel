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

load(":common/paths.bzl", "paths")
load(":common/python/attributes_bazel.bzl", "IMPORTS_ATTRS")
load(
    ":common/python/common.bzl",
    "create_binary_semantics_struct",
    "create_cc_details_struct",
    "create_executable_result_struct",
    "union_attrs",
)
load(":common/python/common_bazel.bzl", "collect_cc_info", "get_imports", "maybe_precompile")
load(":common/python/providers.bzl", "DEFAULT_STUB_SHEBANG")
load(
    ":common/python/py_executable.bzl",
    "create_base_executable_rule",
    "py_executable_base_impl",
)
load(":common/python/semantics.bzl", "TOOLS_REPO")

_py_builtins = _builtins.internal.py_builtins
_EXTERNAL_PATH_PREFIX = "external"
_ZIP_RUNFILES_DIRECTORY_NAME = "runfiles"

BAZEL_EXECUTABLE_ATTRS = union_attrs(
    IMPORTS_ATTRS,
    {
        "legacy_create_init": attr.int(
            default = -1,
            values = [-1, 0, 1],
            doc = """\
Whether to implicitly create empty `__init__.py` files in the runfiles tree.
These are created in every directory containing Python source code or shared
libraries, and every parent directory of those directories, excluding the repo
root directory. The default, `-1` (auto), means true unless
`--incompatible_default_to_explicit_init_py` is used. If false, the user is
responsible for creating (possibly empty) `__init__.py` files and adding them to
the `srcs` of Python targets as required.
                                       """,
        ),
        "_zipper": attr.label(
            cfg = "exec",
            executable = True,
            default = "@" + TOOLS_REPO + "//tools/zip:zipper",
        ),
        "_launcher": attr.label(
            cfg = "target",
            default = "@" + TOOLS_REPO + "//tools/launcher:launcher",
            executable = True,
        ),
        "_windows_launcher_maker": attr.label(
            default = "@" + TOOLS_REPO + "//tools/launcher:launcher_maker",
            cfg = "exec",
            executable = True,
        ),
        "_py_interpreter": attr.label(
            default = configuration_field(
                fragment = "bazel_py",
                name = "python_top",
            ),
        ),
        # TODO: This appears to be vestigial. It's only added because
        # GraphlessQueryTest.testLabelsOperator relies on it to test for
        # query behavior of implicit dependencies.
        "_py_toolchain_type": attr.label(
            default = "@" + TOOLS_REPO + "//tools/python:toolchain_type",
        ),
        "_bootstrap_template": attr.label(
            allow_single_file = True,
            default = "@" + TOOLS_REPO + "//tools/python:python_bootstrap_template.txt",
        ),
    },
)

def create_executable_rule(*, attrs, **kwargs):
    return create_base_executable_rule(
        attrs = BAZEL_EXECUTABLE_ATTRS | attrs,
        fragments = ["py", "bazel_py"],
        **kwargs
    )

def py_executable_bazel_impl(ctx, *, is_test, inherited_environment):
    """Common code for executables for Baze."""
    return py_executable_base_impl(
        ctx = ctx,
        semantics = create_binary_semantics_bazel(),
        is_test = is_test,
        inherited_environment = inherited_environment,
    )

def create_binary_semantics_bazel():
    return create_binary_semantics_struct(
        # keep-sorted start
        create_executable = _create_executable,
        get_cc_details_for_binary = _get_cc_details_for_binary,
        get_central_uncachable_version_file = lambda ctx: None,
        get_coverage_deps = _get_coverage_deps,
        get_debugger_deps = _get_debugger_deps,
        get_extra_common_runfiles_for_binary = lambda ctx: ctx.runfiles(),
        get_extra_providers = _get_extra_providers,
        get_extra_write_build_data_env = lambda ctx: {},
        get_imports = get_imports,
        get_interpreter_path = _get_interpreter_path,
        get_native_deps_dso_name = _get_native_deps_dso_name,
        get_native_deps_user_link_flags = _get_native_deps_user_link_flags,
        get_stamp_flag = _get_stamp_flag,
        maybe_precompile = maybe_precompile,
        should_build_native_deps_dso = lambda ctx: False,
        should_create_init_files = _should_create_init_files,
        should_include_build_data = lambda ctx: False,
        # keep-sorted end
    )

def _get_coverage_deps(ctx, runtime_details):
    _ = ctx, runtime_details  # @unused
    return []

def _get_debugger_deps(ctx, runtime_details):
    _ = ctx, runtime_details  # @unused
    return []

def _get_extra_providers(ctx, main_py, runtime_details):
    _ = ctx, main_py, runtime_details  # @unused
    return []

def _get_stamp_flag(ctx):
    # NOTE: Undocumented API; private to builtins
    return ctx.configuration.stamp_binaries

def _should_create_init_files(ctx):
    if ctx.attr.legacy_create_init == -1:
        return not ctx.fragments.py.default_to_explicit_init_py
    else:
        return bool(ctx.attr.legacy_create_init)

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
        if not executable.extension == "exe":
            fail("Should not happen: somehow we are generating a non-.exe file on windows")
        base_executable_name = executable.basename[0:-4]
    else:
        base_executable_name = executable.basename

    zip_bootstrap = ctx.actions.declare_file(base_executable_name + ".temp", sibling = executable)
    zip_file = ctx.actions.declare_file(base_executable_name + ".zip", sibling = executable)

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
        executable_for_zip_file = zip_bootstrap,
        runfiles = runfiles_details.default_runfiles,
    )

    extra_files_to_build = []

    # NOTE: --build_python_zip defauls to true on Windows
    build_zip_enabled = ctx.fragments.py.build_python_zip

    # When --build_python_zip is enabled, then the zip file becomes
    # one of the default outputs.
    if build_zip_enabled:
        extra_files_to_build.append(zip_file)

    # The logic here is a bit convoluted. Essentially, there are 3 types of
    # executables produced:
    # 1. (non-Windows) A bootstrap template based program.
    # 2. (non-Windows) A self-executable zip file of a bootstrap template based program.
    # 3. (Windows) A native Windows executable that finds and launches
    #    the actual underlying Bazel program (one of the above). Note that
    #    it implicitly assumes one of the above is located next to it, and
    #    that --build_python_zip defaults to true for Windows.

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
            use_zip_file = build_zip_enabled,
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
        if bootstrap_output != None:
            fail("Should not occur: bootstrap_output should not be used " +
                 "when creating an executable zip")
        _create_executable_zip_file(ctx, output = executable, zip_file = zip_file)
    elif bootstrap_output:
        _expand_bootstrap_template(
            ctx,
            output = bootstrap_output,
            is_for_zip = build_zip_enabled,
            **common_bootstrap_template_kwargs
        )
    else:
        # Otherwise, this should be the Windows case of launcher + zip.
        # Double check this just to make sure.
        if not is_windows or not build_zip_enabled:
            fail(("Should not occur: The non-executable-zip and " +
                  "non-boostrap-template case should have windows and zip " +
                  "both true, but got " +
                  "is_windows={is_windows} " +
                  "build_zip_enabled={build_zip_enabled}").format(
                is_windows = is_windows,
                build_zip_enabled = build_zip_enabled,
            ))

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
        template = runtime.bootstrap_template
    else:
        shebang = DEFAULT_STUB_SHEBANG
        template = ctx.file._bootstrap_template

    ctx.actions.expand_template(
        template = template,
        output = output,
        substitutions = {
            "%shebang%": shebang,
            "%main%": "{}/{}".format(
                ctx.workspace_name,
                main_py.short_path,
            ),
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
        arguments = [ctx.executable._launcher.path, launch_info, output.path],
        inputs = [ctx.executable._launcher],
        outputs = [output],
        mnemonic = "PyBuildLauncher",
        progress_message = "Creating launcher for %{label}",
        # Needed to inherit PATH when using non-MSVC compilers like MinGW
        use_default_shell_env = True,
    )

def _create_zip_file(ctx, *, output, original_nonzip_executable, executable_for_zip_file, runfiles):
    workspace_name = ctx.workspace_name
    legacy_external_runfiles = _py_builtins.get_legacy_external_runfiles(ctx)

    manifest = ctx.actions.args()
    manifest.use_param_file("@%s", use_always = True)
    manifest.set_param_file_format("multiline")

    manifest.add("__main__.py={}".format(executable_for_zip_file.path))
    manifest.add("__init__.py=")
    manifest.add(
        "{}=".format(
            _get_zip_runfiles_path("__init__.py", workspace_name, legacy_external_runfiles),
        ),
    )
    for path in runfiles.empty_filenames.to_list():
        manifest.add("{}=".format(_get_zip_runfiles_path(path, workspace_name, legacy_external_runfiles)))

    def map_zip_runfiles(file):
        if file != original_nonzip_executable and file != output:
            return "{}={}".format(
                _get_zip_runfiles_path(file.short_path, workspace_name, legacy_external_runfiles),
                file.path,
            )
        else:
            return None

    manifest.add_all(runfiles.files, map_each = map_zip_runfiles, allow_closure = True)

    inputs = [executable_for_zip_file]
    if _py_builtins.is_bzlmod_enabled(ctx):
        zip_repo_mapping_manifest = ctx.actions.declare_file(
            output.basename + ".repo_mapping",
            sibling = output,
        )
        _py_builtins.create_repo_mapping_manifest(
            ctx = ctx,
            runfiles = runfiles,
            output = zip_repo_mapping_manifest,
        )
        manifest.add("{}/_repo_mapping={}".format(
            _ZIP_RUNFILES_DIRECTORY_NAME,
            zip_repo_mapping_manifest.path,
        ))
        inputs.append(zip_repo_mapping_manifest)

    for artifact in runfiles.files.to_list():
        # Don't include the original executable because it isn't used by the
        # zip file, so no need to build it for the action.
        # Don't include the zipfile itself because it's an output.
        if artifact != original_nonzip_executable and artifact != output:
            inputs.append(artifact)

    zip_cli_args = ctx.actions.args()
    zip_cli_args.add("cC")
    zip_cli_args.add(output)

    ctx.actions.run(
        executable = ctx.executable._zipper,
        arguments = [zip_cli_args, manifest],
        inputs = depset(inputs),
        outputs = [output],
        use_default_shell_env = True,
        mnemonic = "PythonZipper",
        progress_message = "Building Python zip: %{label}",
    )

def _get_zip_runfiles_path(path, workspace_name, legacy_external_runfiles):
    if legacy_external_runfiles and path.startswith(_EXTERNAL_PATH_PREFIX):
        zip_runfiles_path = paths.relativize(path, _EXTERNAL_PATH_PREFIX)
    else:
        # NOTE: External runfiles (artifacts in other repos) will have a leading
        # path component of "../" so that they refer outside the main workspace
        # directory and into the runfiles root. By normalizing, we simplify e.g.
        # "workspace/../foo/bar" to simply "foo/bar".
        zip_runfiles_path = paths.normalize("{}/{}".format(workspace_name, path))
    return "{}/{}".format(_ZIP_RUNFILES_DIRECTORY_NAME, zip_runfiles_path)

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
    if runtime:
        if runtime.interpreter_path:
            interpreter_path = runtime.interpreter_path
        else:
            interpreter_path = "{}/{}".format(
                ctx.workspace_name,
                runtime.interpreter.short_path,
            )

            # NOTE: External runfiles (artifacts in other repos) will have a
            # leading path component of "../" so that they refer outside the
            # main workspace directory and into the runfiles root. By
            # normalizing, we simplify e.g. "workspace/../foo/bar" to simply
            # "foo/bar"
            interpreter_path = paths.normalize(interpreter_path)

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
