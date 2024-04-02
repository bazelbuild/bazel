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
"""Common functionality between test/binary executables."""

load(":common/cc/cc_common.bzl", _cc_common = "cc_common")
load(":common/cc/cc_helper.bzl", "cc_helper")
load(
    ":common/python/attributes.bzl",
    "AGNOSTIC_EXECUTABLE_ATTRS",
    "COMMON_ATTRS",
    "PY_SRCS_ATTRS",
    "SRCS_VERSION_ALL_VALUES",
    "create_srcs_attr",
    "create_srcs_version_attr",
)
load(
    ":common/python/common.bzl",
    "TOOLCHAIN_TYPE",
    "check_native_allowed",
    "collect_imports",
    "collect_runfiles",
    "create_instrumented_files_info",
    "create_output_group_info",
    "create_py_info",
    "csv",
    "filter_to_py_srcs",
    "union_attrs",
)
load(
    ":common/python/providers.bzl",
    "PyCcLinkParamsProvider",
    "PyRuntimeInfo",
)
load(
    ":common/python/semantics.bzl",
    "ALLOWED_MAIN_EXTENSIONS",
    "BUILD_DATA_SYMLINK_PATH",
    "IS_BAZEL",
    "PY_RUNTIME_ATTR_NAME",
)

_py_builtins = _builtins.internal.py_builtins

# Non-Google-specific attributes for executables
# These attributes are for rules that accept Python sources.
EXECUTABLE_ATTRS = union_attrs(
    COMMON_ATTRS,
    AGNOSTIC_EXECUTABLE_ATTRS,
    PY_SRCS_ATTRS,
    {
        # TODO(b/203567235): In the Java impl, any file is allowed. While marked
        # label, it is more treated as a string, and doesn't have to refer to
        # anything that exists because it gets treated as suffix-search string
        # over `srcs`.
        "main": attr.label(
            allow_single_file = True,
            doc = """\
Optional; the name of the source file that is the main entry point of the
application. This file must also be listed in `srcs`. If left unspecified,
`name`, with `.py` appended, is used instead. If `name` does not match any
filename in `srcs`, `main` must be specified.
""",
        ),
        # TODO(b/203567235): In Google, this attribute is deprecated, and can
        # only effectively be PY3. Externally, with Bazel, this attribute has
        # a separate story.
        "python_version": attr.string(
            # TODO(b/203567235): In the Java impl, the default comes from
            # --python_version. Not clear what the Starlark equivalent is.
            default = "PY3",
            # NOTE: Some tests care about the order of these values.
            values = ["PY2", "PY3"],
        ),
    },
    create_srcs_version_attr(values = SRCS_VERSION_ALL_VALUES),
    create_srcs_attr(mandatory = True, doc = """
        The list of source (<code>.py</code>) files that are processed to create the target.
        This includes all your checked-in code and any generated source files. Library targets
        belong in <code>deps</code> instead, while other binary files needed at runtime belong in
        <code>data</code>.
    """),
    allow_none = True,
)

def py_executable_base_impl(ctx, *, semantics, is_test, inherited_environment = []):
    """Base rule implementation for a Python executable.

    Google and Bazel call this common base and apply customizations using the
    semantics object.

    Args:
        ctx: The rule ctx
        semantics: BinarySemantics struct; see create_binary_semantics_struct()
        is_test: bool, True if the rule is a test rule (has `test=True`),
            False if not (has `executable=True`)
        inherited_environment: List of str; additional environment variable
            names that should be inherited from the runtime environment when the
            executable is run.
    Returns:
        DefaultInfo provider for the executable
    """
    _validate_executable(ctx)

    main_py = determine_main(ctx)
    direct_sources = filter_to_py_srcs(ctx.files.srcs)
    output_sources = semantics.maybe_precompile(ctx, direct_sources)
    imports = collect_imports(ctx, semantics)
    executable, files_to_build = _compute_outputs(ctx, output_sources)

    runtime_details = _get_runtime_details(ctx, semantics)
    if ctx.configuration.coverage_enabled:
        extra_deps = semantics.get_coverage_deps(ctx, runtime_details)
    else:
        extra_deps = []

    # The debugger dependency should be prevented by select() config elsewhere,
    # but just to be safe, also guard against adding it to the output here.
    if not _is_tool_config(ctx):
        extra_deps.extend(semantics.get_debugger_deps(ctx, runtime_details))

    cc_details = semantics.get_cc_details_for_binary(ctx, extra_deps = extra_deps)
    native_deps_details = _get_native_deps_details(
        ctx,
        semantics = semantics,
        cc_details = cc_details,
        is_test = is_test,
    )
    runfiles_details = _get_base_runfiles_for_binary(
        ctx,
        executable = executable,
        extra_deps = extra_deps,
        files_to_build = files_to_build,
        extra_common_runfiles = [
            runtime_details.runfiles,
            cc_details.extra_runfiles,
            native_deps_details.runfiles,
            semantics.get_extra_common_runfiles_for_binary(ctx),
        ],
        semantics = semantics,
    )
    exec_result = semantics.create_executable(
        ctx,
        executable = executable,
        main_py = main_py,
        imports = imports,
        is_test = is_test,
        runtime_details = runtime_details,
        cc_details = cc_details,
        native_deps_details = native_deps_details,
        runfiles_details = runfiles_details,
    )
    files_to_build = depset(transitive = [
        exec_result.extra_files_to_build,
        files_to_build,
    ])
    extra_exec_runfiles = ctx.runfiles(transitive_files = files_to_build)
    runfiles_details = struct(
        default_runfiles = runfiles_details.default_runfiles.merge(extra_exec_runfiles),
        data_runfiles = runfiles_details.data_runfiles.merge(extra_exec_runfiles),
    )

    return _create_providers(
        ctx = ctx,
        executable = executable,
        runfiles_details = runfiles_details,
        main_py = main_py,
        imports = imports,
        direct_sources = direct_sources,
        files_to_build = files_to_build,
        runtime_details = runtime_details,
        cc_info = cc_details.cc_info_for_propagating,
        inherited_environment = inherited_environment,
        semantics = semantics,
        output_groups = exec_result.output_groups,
    )

def _validate_executable(ctx):
    if ctx.attr.python_version != "PY3":
        fail("It is not allowed to use Python 2")
    check_native_allowed(ctx)

def _compute_outputs(ctx, output_sources):
    # TODO: This should use the configuration instead of the Bazel OS.
    if _py_builtins.get_current_os_name() == "windows":
        executable = ctx.actions.declare_file(ctx.label.name + ".exe")
    else:
        executable = ctx.actions.declare_file(ctx.label.name)

    # TODO(b/208657718): Remove output_sources from the default outputs
    # once the depot is cleaned up.
    return executable, depset([executable] + output_sources)

def _get_runtime_details(ctx, semantics):
    """Gets various information about the Python runtime to use.

    While most information comes from the toolchain, various legacy and
    compatibility behaviors require computing some other information.

    Args:
        ctx: Rule ctx
        semantics: A `BinarySemantics` struct; see `create_binary_semantics_struct`

    Returns:
        A struct; see inline-field comments of the return value for details.
    """

    # Bazel has --python_path. This flag has a computed default of "python" when
    # its actual default is null (see
    # BazelPythonConfiguration.java#getPythonPath). This flag is only used if
    # toolchains are not enabled and `--python_top` isn't set. Note that Google
    # used to have a variant of this named --python_binary, but it has since
    # been removed.
    #
    # TOOD(bazelbuild/bazel#7901): Remove this once --python_path flag is removed.

    if IS_BAZEL:
        flag_interpreter_path = ctx.fragments.bazel_py.python_path
        toolchain_runtime, effective_runtime = _maybe_get_runtime_from_ctx(ctx)
        if not effective_runtime:
            # Clear these just in case
            toolchain_runtime = None
            effective_runtime = None

    else:  # Google code path
        flag_interpreter_path = None
        toolchain_runtime, effective_runtime = _maybe_get_runtime_from_ctx(ctx)
        if not effective_runtime:
            fail("Unable to find Python runtime")

    if effective_runtime:
        direct = []  # List of files
        transitive = []  # List of depsets
        if effective_runtime.interpreter:
            direct.append(effective_runtime.interpreter)
            transitive.append(effective_runtime.files)

        if ctx.configuration.coverage_enabled:
            if effective_runtime.coverage_tool:
                direct.append(effective_runtime.coverage_tool)
            if effective_runtime.coverage_files:
                transitive.append(effective_runtime.coverage_files)
        runtime_files = depset(direct = direct, transitive = transitive)
    else:
        runtime_files = depset()

    executable_interpreter_path = semantics.get_interpreter_path(
        ctx,
        runtime = effective_runtime,
        flag_interpreter_path = flag_interpreter_path,
    )

    return struct(
        # Optional PyRuntimeInfo: The runtime found from toolchain resolution.
        # This may be None because, within Google, toolchain resolution isn't
        # yet enabled.
        toolchain_runtime = toolchain_runtime,
        # Optional PyRuntimeInfo: The runtime that should be used. When
        # toolchain resolution is enabled, this is the same as
        # `toolchain_resolution`. Otherwise, this probably came from the
        # `_python_top` attribute that the Google implementation still uses.
        # This is separate from `toolchain_runtime` because toolchain_runtime
        # is propagated as a provider, while non-toolchain runtimes are not.
        effective_runtime = effective_runtime,
        # str; Path to the Python interpreter to use for running the executable
        # itself (not the bootstrap script). Either an absolute path (which
        # means it is platform-specific), or a runfiles-relative path (which
        # means the interpreter should be within `runtime_files`)
        executable_interpreter_path = executable_interpreter_path,
        # runfiles: Additional runfiles specific to the runtime that should
        # be included. For in-build runtimes, this shold include the interpreter
        # and any supporting files.
        runfiles = ctx.runfiles(transitive_files = runtime_files),
    )

def _maybe_get_runtime_from_ctx(ctx):
    """Finds the PyRuntimeInfo from the toolchain or attribute, if available.

    Returns:
        2-tuple of toolchain_runtime, effective_runtime
    """
    if ctx.fragments.py.use_toolchains:
        toolchain = ctx.toolchains[TOOLCHAIN_TYPE]

        if not hasattr(toolchain, "py3_runtime"):
            fail("Python toolchain field 'py3_runtime' is missing")
        if not toolchain.py3_runtime:
            fail("Python toolchain missing py3_runtime")
        py3_runtime = toolchain.py3_runtime

        # Hack around the fact that the autodetecting Python toolchain, which is
        # automatically registered, does not yet support Windows. In this case,
        # we want to return null so that _get_interpreter_path falls back on
        # --python_path. See tools/python/toolchain.bzl.
        # TODO(#7844): Remove this hack when the autodetecting toolchain has a
        # Windows implementation.
        if py3_runtime.interpreter_path == "/_magic_pyruntime_sentinel_do_not_use":
            return None, None

        if py3_runtime.python_version != "PY3":
            fail("Python toolchain py3_runtime must be python_version=PY3, got {}".format(
                py3_runtime.python_version,
            ))
        toolchain_runtime = toolchain.py3_runtime
        effective_runtime = toolchain_runtime
    else:
        toolchain_runtime = None
        attr_target = getattr(ctx.attr, PY_RUNTIME_ATTR_NAME)

        # In Bazel, --python_top is null by default.
        if attr_target and PyRuntimeInfo in attr_target:
            effective_runtime = attr_target[PyRuntimeInfo]
        else:
            return None, None

    return toolchain_runtime, effective_runtime

def _get_base_runfiles_for_binary(
        ctx,
        *,
        executable,
        extra_deps,
        files_to_build,
        extra_common_runfiles,
        semantics):
    """Returns the set of runfiles necessary prior to executable creation.

    NOTE: The term "common runfiles" refers to the runfiles that both the
    default and data runfiles have in common.

    Args:
        ctx: The rule ctx.
        executable: The main executable output.
        extra_deps: List of Targets; additional targets whose runfiles
            will be added to the common runfiles.
        files_to_build: depset of File of the default outputs to add into runfiles.
        extra_common_runfiles: List of runfiles; additional runfiles that
            will be added to the common runfiles.
        semantics: A `BinarySemantics` struct; see `create_binary_semantics_struct`.

    Returns:
        struct with attributes:
        * default_runfiles: The default runfiles
        * data_runfiles: The data runfiles
    """
    common_runfiles = collect_runfiles(ctx, depset(
        direct = [executable],
        transitive = [files_to_build],
    ))
    if extra_deps:
        common_runfiles = common_runfiles.merge_all([
            t[DefaultInfo].default_runfiles
            for t in extra_deps
        ])
    common_runfiles = common_runfiles.merge_all(extra_common_runfiles)

    if semantics.should_create_init_files(ctx):
        common_runfiles = _py_builtins.merge_runfiles_with_generated_inits_empty_files_supplier(
            ctx = ctx,
            runfiles = common_runfiles,
        )

    # Don't include build_data.txt in data runfiles. This allows binaries to
    # contain other binaries while still using the same fixed location symlink
    # for the build_data.txt file. Really, the fixed location symlink should be
    # removed and another way found to locate the underlying build data file.
    data_runfiles = common_runfiles

    if is_stamping_enabled(ctx, semantics) and semantics.should_include_build_data(ctx):
        default_runfiles = common_runfiles.merge(_create_runfiles_with_build_data(
            ctx,
            semantics.get_central_uncachable_version_file(ctx),
            semantics.get_extra_write_build_data_env(ctx),
        ))
    else:
        default_runfiles = common_runfiles

    return struct(
        default_runfiles = default_runfiles,
        data_runfiles = data_runfiles,
    )

def _create_runfiles_with_build_data(
        ctx,
        central_uncachable_version_file,
        extra_write_build_data_env):
    return ctx.runfiles(
        symlinks = {
            BUILD_DATA_SYMLINK_PATH: _write_build_data(
                ctx,
                central_uncachable_version_file,
                extra_write_build_data_env,
            ),
        },
    )

def _write_build_data(ctx, central_uncachable_version_file, extra_write_build_data_env):
    # TODO: Remove this logic when a central file is always available
    if not central_uncachable_version_file:
        version_file = ctx.actions.declare_file(ctx.label.name + "-uncachable_version_file.txt")
        _py_builtins.copy_without_caching(
            ctx = ctx,
            read_from = ctx.version_file,
            write_to = version_file,
        )
    else:
        version_file = central_uncachable_version_file

    direct_inputs = [ctx.info_file, version_file]

    # A "constant metadata" file is basically a special file that doesn't
    # support change detection logic and reports that it is unchanged. i.e., it
    # behaves like ctx.version_file and is ignored when computing "what inputs
    # changed" (see https://bazel.build/docs/user-manual#workspace-status).
    #
    # We do this so that consumers of the final build data file don't have
    # to transitively rebuild everything -- the `uncachable_version_file` file
    # isn't cachable, which causes the build data action to always re-run.
    #
    # While this technically means a binary could have stale build info,
    # it ends up not mattering in practice because the volatile information
    # doesn't meaningfully effect other outputs.
    #
    # This is also done for performance and Make It work reasons:
    #   * Passing the transitive dependencies into the action requires passing
    #     the runfiles, but actions don't directly accept runfiles. While
    #     flattening the depsets can be deferred, accessing the
    #     `runfiles.empty_filenames` attribute will will invoke the empty
    #     file supplier a second time, which is too much of a memory and CPU
    #     performance hit.
    #   * Some targets specify a directory in `data`, which is unsound, but
    #     mostly works. Google's RBE, unfortunately, rejects it.
    #   * A binary's transitive closure may be so large that it exceeds
    #     Google RBE limits for action inputs.
    build_data = _py_builtins.declare_constant_metadata_file(
        ctx = ctx,
        name = ctx.label.name + ".build_data.txt",
        root = ctx.bin_dir,
    )

    ctx.actions.run(
        executable = ctx.executable._build_data_gen,
        env = {
            "TARGET": str(ctx.label),
            "OUTPUT": build_data.path,
            "VERSION_FILE": version_file.path,
            # NOTE: ctx.info_file is undocumented; see
            # https://github.com/bazelbuild/bazel/issues/9363
            "INFO_FILE": ctx.info_file.path,
            "PLATFORM": cc_helper.find_cpp_toolchain(ctx).toolchain_id,
        } | extra_write_build_data_env,
        inputs = depset(
            direct = direct_inputs,
        ),
        outputs = [build_data],
        mnemonic = "PyWriteBuildData",
        progress_message = "Generating %{label} build_data.txt",
    )
    return build_data

def _get_native_deps_details(ctx, *, semantics, cc_details, is_test):
    if not semantics.should_build_native_deps_dso(ctx):
        return struct(dso = None, runfiles = ctx.runfiles())

    cc_info = cc_details.cc_info_for_self_link

    if not cc_info.linking_context.linker_inputs:
        return struct(dso = None, runfiles = ctx.runfiles())

    dso = ctx.actions.declare_file(semantics.get_native_deps_dso_name(ctx))
    share_native_deps = ctx.fragments.cpp.share_native_deps()
    cc_feature_config = cc_configure_features(
        ctx,
        cc_toolchain = cc_details.cc_toolchain,
        # See b/171276569#comment18: this feature string is just to allow
        # Google's RBE to know the link action is for the Python case so it can
        # take special actions (though as of Jun 2022, no special action is
        # taken).
        extra_features = ["native_deps_link"],
    )
    if share_native_deps:
        linked_lib = _create_shared_native_deps_dso(
            ctx,
            cc_toolchain = cc_details.cc_toolchain,
            cc_info = cc_info,
            is_test = is_test,
            requested_features = cc_feature_config.requested_features,
            feature_configuration = cc_feature_config.feature_configuration,
        )
        ctx.actions.symlink(
            output = dso,
            target_file = linked_lib,
            progress_message = "Symlinking shared native deps for %{label}",
        )
    else:
        linked_lib = dso
    _cc_common.link(
        name = ctx.label.name,
        actions = ctx.actions,
        linking_contexts = [cc_info.linking_context],
        output_type = "dynamic_library",
        never_link = True,
        native_deps = True,
        feature_configuration = cc_feature_config.feature_configuration,
        cc_toolchain = cc_details.cc_toolchain,
        test_only_target = is_test,
        stamp = 1 if is_stamping_enabled(ctx, semantics) else 0,
        main_output = linked_lib,
        use_shareable_artifact_factory = True,
        # NOTE: Only flags not captured by cc_info.linking_context need to
        # be manually passed
        user_link_flags = semantics.get_native_deps_user_link_flags(ctx),
    )
    return struct(
        dso = dso,
        runfiles = ctx.runfiles(files = [dso]),
    )

def _create_shared_native_deps_dso(
        ctx,
        *,
        cc_toolchain,
        cc_info,
        is_test,
        feature_configuration,
        requested_features):
    linkstamps = cc_info.linking_context.linkstamps()

    partially_disabled_thin_lto = (
        _cc_common.is_enabled(
            feature_name = "thin_lto_linkstatic_tests_use_shared_nonlto_backends",
            feature_configuration = feature_configuration,
        ) and not _cc_common.is_enabled(
            feature_name = "thin_lto_all_linkstatic_use_shared_nonlto_backends",
            feature_configuration = feature_configuration,
        )
    )
    if not linkstamps:
        build_info_artifacts = []
    elif cc_helper.is_stamping_enabled(ctx):
        build_info_artifacts = cc_toolchain._build_info_files.non_redacted_build_info_files.to_list()
    else:
        build_info_artifacts = cc_toolchain._build_info_files.redacted_build_info_files.to_list()
    dso_hash = _get_shared_native_deps_hash(
        linker_inputs = cc_helper.get_static_mode_params_for_dynamic_library_libraries(
            depset([
                lib
                for linker_input in cc_info.linking_context.linker_inputs.to_list()
                for lib in linker_input.libraries
            ]),
        ),
        link_opts = [
            flag
            for input in cc_info.linking_context.linker_inputs.to_list()
            for flag in input.user_link_flags
        ],
        linkstamps = [linkstamp.file() for linkstamp in linkstamps.to_list()],
        build_info_artifacts = build_info_artifacts,
        features = requested_features,
        is_test_target_partially_disabled_thin_lto = is_test and partially_disabled_thin_lto,
    )
    return ctx.actions.declare_shareable_artifact("_nativedeps/%x.so" % dso_hash)

# This is a minimal version of NativeDepsHelper.getSharedNativeDepsPath, see
# com.google.devtools.build.lib.rules.nativedeps.NativeDepsHelper#getSharedNativeDepsPath
# The basic idea is to take all the inputs that affect linking and encode (via
# hashing) them into the filename.
# TODO(b/234232820): The settings that affect linking must be kept in sync with the actual
# C++ link action. For more information, see the large descriptive comment on
# NativeDepsHelper#getSharedNativeDepsPath.
def _get_shared_native_deps_hash(
        *,
        linker_inputs,
        link_opts,
        linkstamps,
        build_info_artifacts,
        features,
        is_test_target_partially_disabled_thin_lto):
    # NOTE: We use short_path because the build configuration root in which
    # files are always created already captures the configuration-specific
    # parts, so no need to include them manually.
    parts = []
    for artifact in linker_inputs:
        parts.append(artifact.short_path)
    parts.append(str(len(link_opts)))
    parts.extend(link_opts)
    for artifact in linkstamps:
        parts.append(artifact.short_path)
    for artifact in build_info_artifacts:
        parts.append(artifact.short_path)
    parts.extend(sorted(features))

    # Sharing of native dependencies may cause an {@link
    # ActionConflictException} when ThinLTO is disabled for test and test-only
    # targets that are statically linked, but enabled for other statically
    # linked targets. This happens in case the artifacts for the shared native
    # dependency are output by {@link Action}s owned by the non-test and test
    # targets both. To fix this, we allow creation of multiple artifacts for the
    # shared native library - one shared among the test and test-only targets
    # where ThinLTO is disabled, and the other shared among other targets where
    # ThinLTO is enabled. See b/138118275
    parts.append("1" if is_test_target_partially_disabled_thin_lto else "0")

    return hash("".join(parts))

def determine_main(ctx):
    """Determine the main entry point .py source file.

    Args:
        ctx: The rule ctx.

    Returns:
        Artifact; the main file. If one can't be found, an error is raised.
    """
    if ctx.attr.main:
        proposed_main = ctx.attr.main.label.name
        if not proposed_main.endswith(tuple(ALLOWED_MAIN_EXTENSIONS)):
            fail("main must end in '.py'")
    else:
        if ctx.label.name.endswith(".py"):
            fail("name must not end in '.py'")
        proposed_main = ctx.label.name + ".py"

    main_files = [src for src in ctx.files.srcs if _path_endswith(src.short_path, proposed_main)]
    if not main_files:
        if ctx.attr.main:
            fail("could not find '{}' as specified by 'main' attribute".format(proposed_main))
        else:
            fail(("corresponding default '{}' does not appear in srcs. Add " +
                  "it or override default file name with a 'main' attribute").format(
                proposed_main,
            ))

    elif len(main_files) > 1:
        if ctx.attr.main:
            fail(("file name '{}' specified by 'main' attributes matches multiple files. " +
                  "Matches: {}").format(
                proposed_main,
                csv([f.short_path for f in main_files]),
            ))
        else:
            fail(("default main file '{}' matches multiple files in srcs. Perhaps specify " +
                  "an explicit file with 'main' attribute? Matches were: {}").format(
                proposed_main,
                csv([f.short_path for f in main_files]),
            ))
    return main_files[0]

def _path_endswith(path, endswith):
    # Use slash to anchor each path to prevent e.g.
    # "ab/c.py".endswith("b/c.py") from incorrectly matching.
    return ("/" + path).endswith("/" + endswith)

def is_stamping_enabled(ctx, semantics):
    """Tells if stamping is enabled or not.

    Args:
        ctx: The rule ctx
        semantics: a semantics struct (see create_semantics_struct).
    Returns:
        bool; True if stamping is enabled, False if not.
    """
    if _is_tool_config(ctx):
        return False

    stamp = ctx.attr.stamp
    if stamp == 1:
        return True
    elif stamp == 0:
        return False
    elif stamp == -1:
        return semantics.get_stamp_flag(ctx)
    else:
        fail("Unsupported `stamp` value: {}".format(stamp))

def _is_tool_config(ctx):
    # NOTE: The is_tool_configuration() function is only usable by builtins.
    # See https://github.com/bazelbuild/bazel/issues/14444 for the FR for
    # a more public API. Outside of builtins, ctx.bin_dir.path can be
    # checked for `/host/` or `-exec-`.
    return ctx.configuration.is_tool_configuration()

def _create_providers(
        *,
        ctx,
        executable,
        main_py,
        direct_sources,
        files_to_build,
        runfiles_details,
        imports,
        cc_info,
        inherited_environment,
        runtime_details,
        output_groups,
        semantics):
    """Creates the providers an executable should return.

    Args:
        ctx: The rule ctx.
        executable: File; the target's executable file.
        main_py: File; the main .py entry point.
        direct_sources: list of Files; the direct, raw `.py` sources for the target.
            This should only be Python source files. It should not include pyc
            files.
        files_to_build: depset of Files; the files for DefaultInfo.files
        runfiles_details: runfiles that will become the default  and data runfiles.
        imports: depset of strings; the import paths to propagate
        cc_info: optional CcInfo; Linking information to propagate as
            PyCcLinkParamsProvider. Note that only the linking information
            is propagated, not the whole CcInfo.
        inherited_environment: list of strings; Environment variable names
            that should be inherited from the environment the executuble
            is run within.
        runtime_details: struct of runtime information; see _get_runtime_details()
        output_groups: dict[str, depset[File]]; used to create OutputGroupInfo
        semantics: BinarySemantics struct; see create_binary_semantics()

    Returns:
        A two-tuple of:
        1. A dict of legacy providers.
        2. A list of modern providers.
    """
    providers = [
        DefaultInfo(
            executable = executable,
            files = files_to_build,
            default_runfiles = _py_builtins.make_runfiles_respect_legacy_external_runfiles(
                ctx,
                runfiles_details.default_runfiles,
            ),
            data_runfiles = _py_builtins.make_runfiles_respect_legacy_external_runfiles(
                ctx,
                runfiles_details.data_runfiles,
            ),
        ),
        create_instrumented_files_info(ctx),
        _create_run_environment_info(ctx, inherited_environment),
    ]

    # TODO(b/265840007): Make this non-conditional once Google enables
    # --incompatible_use_python_toolchains.
    if runtime_details.toolchain_runtime:
        providers.append(runtime_details.toolchain_runtime)

    # TODO(b/163083591): Remove the PyCcLinkParamsProvider once binaries-in-deps
    # are cleaned up.
    if cc_info:
        providers.append(
            PyCcLinkParamsProvider(cc_info = cc_info),
        )

    py_info, deps_transitive_sources = create_py_info(
        ctx,
        direct_sources = depset(direct_sources),
        imports = imports,
    )

    # TODO(b/253059598): Remove support for extra actions; https://github.com/bazelbuild/bazel/issues/16455
    listeners_enabled = _py_builtins.are_action_listeners_enabled(ctx)
    if listeners_enabled:
        _py_builtins.add_py_extra_pseudo_action(
            ctx = ctx,
            dependency_transitive_python_sources = deps_transitive_sources,
        )

    providers.append(py_info)
    providers.append(create_output_group_info(py_info.transitive_sources, output_groups))

    extra_providers = semantics.get_extra_providers(
        ctx,
        main_py = main_py,
        runtime_details = runtime_details,
    )
    providers.extend(extra_providers)
    return providers

def _create_run_environment_info(ctx, inherited_environment):
    expanded_env = {}
    for key, value in ctx.attr.env.items():
        expanded_env[key] = _py_builtins.expand_location_and_make_variables(
            ctx = ctx,
            attribute_name = "env[{}]".format(key),
            expression = value,
            targets = ctx.attr.data,
        )
    return RunEnvironmentInfo(
        environment = expanded_env,
        inherited_environment = inherited_environment,
    )

def create_base_executable_rule(*, attrs, fragments = [], **kwargs):
    """Create a function for defining for Python binary/test targets.

    Args:
        attrs: Rule attributes
        fragments: List of str; extra config fragments that are required.
        **kwargs: Additional args to pass onto `rule()`

    Returns:
        A rule function
    """
    if "py" not in fragments:
        # The list might be frozen, so use concatentation
        fragments = fragments + ["py"]
    return rule(
        # TODO: add ability to remove attrs, i.e. for imports attr
        attrs = EXECUTABLE_ATTRS | attrs,
        toolchains = [TOOLCHAIN_TYPE] + cc_helper.use_cpp_toolchain(),
        fragments = fragments,
        **kwargs
    )

def cc_configure_features(ctx, *, cc_toolchain, extra_features):
    """Configure C++ features for Python purposes.

    Args:
        ctx: Rule ctx
        cc_toolchain: The CcToolchain the target is using.
        extra_features: list of strings; additional features to request be
            enabled.

    Returns:
        struct of the feature configuration and all requested features.
    """
    requested_features = ["static_linking_mode"]
    requested_features.extend(extra_features)
    requested_features.extend(ctx.features)
    if "legacy_whole_archive" not in ctx.disabled_features:
        requested_features.append("legacy_whole_archive")
    feature_configuration = _cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = requested_features,
        unsupported_features = ctx.disabled_features,
    )
    return struct(
        feature_configuration = feature_configuration,
        requested_features = requested_features,
    )

only_exposed_for_google_internal_reason = struct(
    create_runfiles_with_build_data = _create_runfiles_with_build_data,
)
