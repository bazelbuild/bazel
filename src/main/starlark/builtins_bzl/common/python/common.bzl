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
"""Various things common to Bazel and Google rule implementations."""

load(":common/cc/cc_helper.bzl", "cc_helper")
load(
    ":common/python/providers.bzl",
    "PyInfo",
)
load(
    ":common/python/semantics.bzl",
    "NATIVE_RULES_MIGRATION_FIX_CMD",
    "NATIVE_RULES_MIGRATION_HELP_URL",
    "TOOLS_REPO",
)

_testing = _builtins.toplevel.testing
_platform_common = _builtins.toplevel.platform_common
_coverage_common = _builtins.toplevel.coverage_common
_py_builtins = _builtins.internal.py_builtins
PackageSpecificationInfo = _builtins.toplevel.PackageSpecificationInfo

TOOLCHAIN_TYPE = "@" + TOOLS_REPO + "//tools/python:toolchain_type"

# Extensions without the dot
_PYTHON_SOURCE_EXTENSIONS = ["py"]

# NOTE: Must stay in sync with the value used in rules_python
_MIGRATION_TAG = "__PYTHON_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__"

def create_binary_semantics_struct(
        *,
        create_executable,
        get_cc_details_for_binary,
        get_central_uncachable_version_file,
        get_coverage_deps,
        get_debugger_deps,
        get_extra_common_runfiles_for_binary,
        get_extra_providers,
        get_extra_write_build_data_env,
        get_interpreter_path,
        get_imports,
        get_native_deps_dso_name,
        get_native_deps_user_link_flags,
        get_stamp_flag,
        maybe_precompile,
        should_build_native_deps_dso,
        should_create_init_files,
        should_include_build_data):
    """Helper to ensure a semantics struct has all necessary fields.

    Call this instead of a raw call to `struct(...)`; it'll help ensure all
    the necessary functions are being correctly provided.

    Args:
        create_executable: Callable; creates a binary's executable output. See
            py_executable.bzl#py_executable_base_impl for details.
        get_cc_details_for_binary: Callable that returns a `CcDetails` struct; see
            `create_cc_detail_struct`.
        get_central_uncachable_version_file: Callable that returns an optional
            Artifact; this artifact is special: it is never cached and is a copy
            of `ctx.version_file`; see py_builtins.copy_without_caching
        get_coverage_deps: Callable that returns a list of Targets for making
            coverage work; only called if coverage is enabled.
        get_debugger_deps: Callable that returns a list of Targets that provide
            custom debugger support; only called for target-configuration.
        get_extra_common_runfiles_for_binary: Callable that returns a runfiles
            object of extra runfiles a binary should include.
        get_extra_providers: Callable that returns extra providers; see
            py_executable.bzl#_create_providers for details.
        get_extra_write_build_data_env: Callable that returns a dict[str, str]
            of additional environment variable to pass to build data generation.
        get_interpreter_path: Callable that returns an optional string, which is
            the path to the Python interpreter to use for running the binary.
        get_imports: Callable that returns a list of the target's import
            paths (from the `imports` attribute, so just the target's own import
            path strings, not from dependencies).
        get_native_deps_dso_name: Callable that returns a string, which is the
            basename (with extension) of the native deps DSO library.
        get_native_deps_user_link_flags: Callable that returns a list of strings,
            which are any extra linker flags to pass onto the native deps DSO
            linking action.
        get_stamp_flag: Callable that returns bool of if the --stamp flag was
            enabled or not.
        maybe_precompile: Callable that may optional precompile the input `.py`
            sources and returns the full set of desired outputs derived from
            the source files (e.g., both py and pyc, only one of them, etc).
        should_build_native_deps_dso: Callable that returns bool; True if
            building a native deps DSO is supported, False if not.
        should_create_init_files: Callable that returns bool; True if
            `__init__.py` files should be generated, False if not.
        should_include_build_data: Callable that returns bool; True if
            build data should be generated, False if not.
    Returns:
        A "BinarySemantics" struct.
    """
    return struct(
        # keep-sorted
        create_executable = create_executable,
        get_cc_details_for_binary = get_cc_details_for_binary,
        get_central_uncachable_version_file = get_central_uncachable_version_file,
        get_coverage_deps = get_coverage_deps,
        get_debugger_deps = get_debugger_deps,
        get_extra_common_runfiles_for_binary = get_extra_common_runfiles_for_binary,
        get_extra_providers = get_extra_providers,
        get_extra_write_build_data_env = get_extra_write_build_data_env,
        get_imports = get_imports,
        get_interpreter_path = get_interpreter_path,
        get_native_deps_dso_name = get_native_deps_dso_name,
        get_native_deps_user_link_flags = get_native_deps_user_link_flags,
        get_stamp_flag = get_stamp_flag,
        maybe_precompile = maybe_precompile,
        should_build_native_deps_dso = should_build_native_deps_dso,
        should_create_init_files = should_create_init_files,
        should_include_build_data = should_include_build_data,
    )

def create_library_semantics_struct(
        *,
        get_cc_info_for_library,
        get_imports,
        maybe_precompile):
    """Create a `LibrarySemantics` struct.

    Call this instead of a raw call to `struct(...)`; it'll help ensure all
    the necessary functions are being correctly provided.

    Args:
        get_cc_info_for_library: Callable that returns a CcInfo for the library;
            see py_library_impl for arg details.
        get_imports: Callable; see create_binary_semantics_struct.
        maybe_precompile: Callable; see create_binary_semantics_struct.
    Returns:
        a `LibrarySemantics` struct.
    """
    return struct(
        # keep sorted
        get_cc_info_for_library = get_cc_info_for_library,
        get_imports = get_imports,
        maybe_precompile = maybe_precompile,
    )

def create_cc_details_struct(
        *,
        cc_info_for_propagating,
        cc_info_for_self_link,
        cc_info_with_extra_link_time_libraries,
        extra_runfiles,
        cc_toolchain):
    """Creates a CcDetails struct.

    Args:
        cc_info_for_propagating: CcInfo that is propagated out of the target
            by returning it within a PyCcLinkParamsProvider object.
        cc_info_for_self_link: CcInfo that is used when linking for the
            binary (or its native deps DSO) itself. This may include extra
            information that isn't propagating (e.g. a custom malloc)
        cc_info_with_extra_link_time_libraries: CcInfo of extra link time
            libraries that MUST come after `cc_info_for_self_link` (or possibly
            always last; not entirely clear) when passed to
            `link.linking_contexts`.
        extra_runfiles: runfiles of extra files needed at runtime, usually as
            part of `cc_info_with_extra_link_time_libraries`; should be added to
            runfiles.
        cc_toolchain: CcToolchain that should be used when building.

    Returns:
        A `CcDetails` struct.
    """
    return struct(
        cc_info_for_propagating = cc_info_for_propagating,
        cc_info_for_self_link = cc_info_for_self_link,
        cc_info_with_extra_link_time_libraries = cc_info_with_extra_link_time_libraries,
        extra_runfiles = extra_runfiles,
        cc_toolchain = cc_toolchain,
    )

def create_executable_result_struct(*, extra_files_to_build, output_groups):
    """Creates a `CreateExecutableResult` struct.

    This is the return value type of the semantics create_executable function.

    Args:
        extra_files_to_build: depset of File; additional files that should be
            included as default outputs.
        output_groups: dict[str, depset[File]]; additional output groups that
            should be returned.

    Returns:
        A `CreateExecutableResult` struct.
    """
    return struct(
        extra_files_to_build = extra_files_to_build,
        output_groups = output_groups,
    )

def union_attrs(*attr_dicts, allow_none = False):
    """Helper for combining and building attriute dicts for rules.

    Similar to dict.update, except:
      * Duplicate keys raise an error if they aren't equal. This is to prevent
        unintentionally replacing an attribute with a potentially incompatible
        definition.
      * None values are special: They mean the attribute is required, but the
        value should be provided by another attribute dict (depending on the
        `allow_none` arg).
    Args:
        *attr_dicts: The dicts to combine.
        allow_none: bool, if True, then None values are allowed. If False,
            then one of `attrs_dicts` must set a non-None value for keys
            with a None value.

    Returns:
        dict of attributes.
    """
    result = {}
    missing = {}
    for attr_dict in attr_dicts:
        for attr_name, value in attr_dict.items():
            if value == None and not allow_none:
                if attr_name not in result:
                    missing[attr_name] = None
            else:
                if attr_name in missing:
                    missing.pop(attr_name)

                if attr_name not in result or result[attr_name] == None:
                    result[attr_name] = value
                elif value != None and result[attr_name] != value:
                    fail("Duplicate attribute name: '{}': existing={}, new={}".format(
                        attr_name,
                        result[attr_name],
                        value,
                    ))

                # Else, they're equal, so do nothing. This allows merging dicts
                # that both define the same key from a common place.

    if missing and not allow_none:
        fail("Required attributes missing: " + csv(missing.keys()))
    return result

def csv(values):
    """Convert a list of strings to comma separated value string."""
    return ", ".join(sorted(values))

def filter_to_py_srcs(srcs):
    """Filters .py files from the given list of files"""

    # TODO(b/203567235): Get the set of recognized extensions from
    # elsewhere, as there may be others. e.g. Bazel recognizes .py3
    # as a valid extension.
    return [f for f in srcs if f.extension == "py"]

def collect_imports(ctx, semantics):
    return depset(direct = semantics.get_imports(ctx), transitive = [
        dep[PyInfo].imports
        for dep in ctx.attr.deps
        if PyInfo in dep
    ])

def collect_runfiles(ctx, files):
    """Collects the necessary files from the rule's context.

    This presumes the ctx is for a py_binary, py_test, or py_library rule.

    Args:
        ctx: rule ctx
        files: depset of extra files to include in the runfiles.
    Returns:
        runfiles necessary for the ctx's target.
    """
    return ctx.runfiles(
        transitive_files = files,
        # This little arg carries a lot of weight, but because Starlark doesn't
        # have a way to identify if a target is just a File, the equivalent
        # logic can't be re-implemented in pure-Starlark.
        #
        # Under the hood, it calls the Java `Runfiles#addRunfiles(ctx,
        # DEFAULT_RUNFILES)` method, which is the what the Java implementation
        # of the Python rules originally did, and the details of how that method
        # works have become relied on in various ways. Specifically, what it
        # does is visit the srcs, deps, and data attributes in the following
        # ways:
        #
        # For each target in the "data" attribute...
        #   If the target is a File, then add that file to the runfiles.
        #   Otherwise, add the target's **data runfiles** to the runfiles.
        #
        # Note that, contray to best practice, the default outputs of the
        # targets in `data` are *not* added, nor are the default runfiles.
        #
        # This ends up being important for several reasons, some of which are
        # specific to Google-internal features of the rules.
        #   * For Python executables, we have to use `data_runfiles` to avoid
        #     conflicts for the build data files. Such files have
        #     target-specific content, but uses a fixed location, so if a
        #     binary has another binary in `data`, and both try to specify a
        #     file for that file path, then a warning is printed and an
        #     arbitrary one will be used.
        #   * For rules with _entirely_ different sets of files in data runfiles
        #     vs default runfiles vs default outputs. For example,
        #     proto_library: documented behavior of this rule is that putting it
        #     in the `data` attribute will cause the transitive closure of
        #     `.proto` source files to be included. This set of sources is only
        #     in the `data_runfiles` (`default_runfiles` is empty).
        #   * For rules with a _subset_ of files in data runfiles. For example,
        #     a certain Google rule used for packaging arbitrary binaries will
        #     generate multiple versions of a binary (e.g. different archs,
        #     stripped vs un-stripped, etc) in its default outputs, but only
        #     one of them in the runfiles; this helps avoid large, unused
        #     binaries contributing to remote executor input limits.
        #
        # Unfortunately, the above behavior also results in surprising behavior
        # in some cases. For example, simple custom rules that only return their
        # files in their default outputs won't have their files included. Such
        # cases must either return their files in runfiles, or use `filegroup()`
        # which will do so for them.
        #
        # For each target in "srcs" and "deps"...
        #   Add the default runfiles of the target to the runfiles. While this
        #   is desirable behavior, it also ends up letting a `py_library`
        #   be put in `srcs` and still mostly work.
        # TODO(b/224640180): Reject py_library et al rules in srcs.
        collect_default = True,
    )

def create_py_info(ctx, *, direct_sources, imports):
    """Create PyInfo provider.

    Args:
        ctx: rule ctx.
        direct_sources: depset of Files; the direct, raw `.py` sources for the
            target. This should only be Python source files. It should not
            include pyc files.
        imports: depset of strings; the import path values to propagate.

    Returns:
        A tuple of the PyInfo instance and a depset of the
        transitive sources collected from dependencies (the latter is only
        necessary for deprecated extra actions support).
    """
    uses_shared_libraries = False
    has_py2_only_sources = ctx.attr.srcs_version in ("PY2", "PY2ONLY")
    has_py3_only_sources = ctx.attr.srcs_version in ("PY3", "PY3ONLY")
    transitive_sources_depsets = []  # list of depsets
    transitive_sources_files = []  # list of Files
    for target in ctx.attr.deps:
        # PyInfo may not be present for e.g. cc_library rules.
        if PyInfo in target:
            info = target[PyInfo]
            transitive_sources_depsets.append(info.transitive_sources)
            uses_shared_libraries = uses_shared_libraries or info.uses_shared_libraries
            has_py2_only_sources = has_py2_only_sources or info.has_py2_only_sources
            has_py3_only_sources = has_py3_only_sources or info.has_py3_only_sources
        else:
            # TODO(b/228692666): Remove this once non-PyInfo targets are no
            # longer supported in `deps`.
            files = target[DefaultInfo].files.to_list()
            for f in files:
                if f.extension == "py":
                    transitive_sources_files.append(f)
                uses_shared_libraries = (
                    uses_shared_libraries or
                    cc_helper.is_valid_shared_library_artifact(f)
                )
    deps_transitive_sources = depset(
        direct = transitive_sources_files,
        transitive = transitive_sources_depsets,
    )

    # We only look at data to calculate uses_shared_libraries, if it's already
    # true, then we don't need to waste time looping over it.
    if not uses_shared_libraries:
        # Similar to the above, except we only calculate uses_shared_libraries
        for target in ctx.attr.data:
            # TODO(b/234730058): Remove checking for PyInfo in data once depot
            # cleaned up.
            if PyInfo in target:
                info = target[PyInfo]
                uses_shared_libraries = info.uses_shared_libraries
            else:
                files = target[DefaultInfo].files.to_list()
                for f in files:
                    uses_shared_libraries = cc_helper.is_valid_shared_library_artifact(f)
                    if uses_shared_libraries:
                        break
            if uses_shared_libraries:
                break

    # TODO(b/203567235): Set `uses_shared_libraries` field, though the Bazel
    # docs indicate it's unused in Bazel and may be removed.
    py_info = PyInfo(
        transitive_sources = depset(
            transitive = [deps_transitive_sources, direct_sources],
        ),
        imports = imports,
        # NOTE: This isn't strictly correct, but with Python 2 gone,
        # the srcs_version logic is largely defunct, so shouldn't matter in
        # practice.
        has_py2_only_sources = has_py2_only_sources,
        has_py3_only_sources = has_py3_only_sources,
        uses_shared_libraries = uses_shared_libraries,
    )
    return py_info, deps_transitive_sources

def create_instrumented_files_info(ctx):
    return _coverage_common.instrumented_files_info(
        ctx,
        source_attributes = ["srcs"],
        dependency_attributes = ["deps", "data"],
        extensions = _PYTHON_SOURCE_EXTENSIONS,
    )

def create_output_group_info(transitive_sources, extra_groups):
    return OutputGroupInfo(
        compilation_prerequisites_INTERNAL_ = transitive_sources,
        compilation_outputs = transitive_sources,
        **extra_groups
    )

def maybe_add_test_execution_info(providers, ctx):
    """Adds ExecutionInfo, if necessary for proper test execution.

    Args:
        providers: Mutable list of providers; may have ExecutionInfo
            provider appended.
        ctx: Rule ctx.
    """

    # When built for Apple platforms, require the execution to be on a Mac.
    # TODO(b/176993122): Remove when bazel automatically knows to run on darwin.
    if target_platform_has_any_constraint(ctx, ctx.attr._apple_constraints):
        providers.append(_testing.ExecutionInfo({"requires-darwin": ""}))

_BOOL_TYPE = type(True)

def is_bool(v):
    return type(v) == _BOOL_TYPE

def target_platform_has_any_constraint(ctx, constraints):
    """Check if target platform has any of a list of constraints.

    Args:
      ctx: rule context.
      constraints: label_list of constraints.

    Returns:
      True if target platform has at least one of the constraints.
    """
    for constraint in constraints:
        constraint_value = constraint[_platform_common.ConstraintValueInfo]
        if ctx.target_platform_has_constraint(constraint_value):
            return True
    return False

def check_native_allowed(ctx):
    """Check if the usage of the native rule is allowed.

    Args:
        ctx: rule context to check
    """
    if not ctx.fragments.py.disallow_native_rules:
        return

    if _MIGRATION_TAG in ctx.attr.tags:
        return

    # NOTE: The main repo name is empty in *labels*, but not in
    # ctx.workspace_name
    is_main_repo = not bool(ctx.label.workspace_name)
    if is_main_repo:
        check_label = ctx.label
    else:
        # package_group doesn't allow @repo syntax, so we work around that
        # by prefixing external repos with a fake package path. This also
        # makes it easy to enable or disable all external repos.
        check_label = Label("@//__EXTERNAL_REPOS__/{workspace}{package}".format(
            workspace = ctx.label.workspace_name,
            # Prevent a label with trailing slash, which is malformed.
            package = "/" + ctx.label.package if ctx.label.package else "",
        ))
    allowlist = ctx.attr._native_rules_allowlist
    if allowlist:
        allowed = ctx.attr._native_rules_allowlist[PackageSpecificationInfo].contains(check_label)
        allowlist_help = str(allowlist.label).replace("@//", "//")
    else:
        allowed = False
        allowlist_help = ("no allowlist specified; all disallowed; specify one " +
                          "with --python_native_rules_allowlist")
    if not allowed:
        if ctx.attr.generator_function:
            generator = "{generator_function}(name={generator_name}) in {generator_location}".format(
                generator_function = ctx.attr.generator_function,
                generator_name = ctx.attr.generator_name,
                generator_location = ctx.attr.generator_location,
            )
        else:
            generator = "No generator (called directly in BUILD file)"

        msg = (
            "{target} not allowed to use native.{rule}\n" +
            "Generated by: {generator}\n" +
            "Allowlist: {allowlist}\n" +
            "Migrate to using @rules_python, see {help_url}\n" +
            "FIXCMD: {fix_cmd} --target={target} --rule={rule} " +
            "--generator_name={generator_name} --location={generator_location}"
        )
        fail(msg.format(
            target = str(ctx.label).replace("@//", "//"),
            rule = _py_builtins.get_rule_name(ctx),
            generator = generator,
            allowlist = allowlist_help,
            generator_name = ctx.attr.generator_name,
            generator_location = ctx.attr.generator_location,
            help_url = NATIVE_RULES_MIGRATION_HELP_URL,
            fix_cmd = NATIVE_RULES_MIGRATION_FIX_CMD,
        ))
