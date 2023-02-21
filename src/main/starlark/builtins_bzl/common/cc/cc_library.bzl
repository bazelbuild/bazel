# Copyright 2020 The Bazel Authors. All rights reserved.
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

"""cc_library Starlark implementation replacing native"""

load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/cc/semantics.bzl", "semantics")
load(":common/cc/cc_info.bzl", "CcInfo")

cc_common = _builtins.toplevel.cc_common
cc_internal = _builtins.internal.cc_internal

def _cc_library_impl(ctx):
    cc_helper.check_srcs_extensions(ctx, ALLOWED_SRC_FILES, "cc_library", True)

    common = cc_internal.create_common(ctx = ctx)
    cc_toolchain = cc_helper.find_cpp_toolchain(ctx)
    cc_helper.report_invalid_options(cc_toolchain, ctx.fragments.cpp)

    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    precompiled_files = cc_helper.build_precompiled_files(ctx = ctx)

    semantics.validate_attributes(ctx = ctx)
    _check_no_repeated_srcs(ctx)

    semantics.check_can_use_implementation_deps(ctx)
    interface_deps = ctx.attr.deps + semantics.get_cc_runtimes(ctx, True)
    compilation_contexts = cc_helper.get_compilation_contexts_from_deps(interface_deps)
    implementation_compilation_contexts = cc_helper.get_compilation_contexts_from_deps(ctx.attr.implementation_deps)

    additional_make_variable_substitutions = cc_helper.get_toolchain_global_make_variables(cc_toolchain)
    additional_make_variable_substitutions.update(cc_helper.get_cc_flags_make_variable(ctx, feature_configuration, cc_toolchain))

    (compilation_context, srcs_compilation_outputs) = cc_common.compile(
        actions = ctx.actions,
        name = ctx.label.name,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        user_compile_flags = cc_helper.get_copts(ctx, feature_configuration, additional_make_variable_substitutions),
        defines = cc_helper.defines(ctx, additional_make_variable_substitutions),
        local_defines = cc_helper.local_defines(ctx, additional_make_variable_substitutions) + cc_helper.get_local_defines_for_runfiles_lookup(ctx),
        loose_includes = common.loose_include_dirs,
        system_includes = cc_helper.system_include_dirs(ctx, additional_make_variable_substitutions),
        copts_filter = cc_helper.copts_filter(ctx, additional_make_variable_substitutions),
        purpose = "cc_library-compile",
        srcs = cc_helper.get_srcs(ctx),
        private_hdrs = cc_helper.get_private_hdrs(ctx),
        public_hdrs = cc_helper.get_public_hdrs(ctx),
        code_coverage_enabled = cc_helper.is_code_coverage_enabled(ctx),
        compilation_contexts = compilation_contexts,
        implementation_compilation_contexts = implementation_compilation_contexts,
        hdrs_checking_mode = semantics.determine_headers_checking_mode(ctx),
        grep_includes = ctx.executable._grep_includes,
        textual_hdrs = ctx.files.textual_hdrs,
        include_prefix = ctx.attr.include_prefix,
        strip_include_prefix = ctx.attr.strip_include_prefix,
    )

    precompiled_objects = cc_common.create_compilation_outputs(
        # TODO(bazel-team): Perhaps this should be objects, leaving as it is in the original
        # Java code for now. Changing it might cause breakages.
        objects = depset(precompiled_files[1]),
        pic_objects = depset(precompiled_files[1]),
    )

    compilation_outputs = cc_common.merge_compilation_outputs(
        compilation_outputs = [precompiled_objects, srcs_compilation_outputs],
    )

    supports_dynamic_linker = cc_common.is_enabled(
        feature_configuration = feature_configuration,
        feature_name = "supports_dynamic_linker",
    )

    create_dynamic_library = (not ctx.attr.linkstatic and
                              supports_dynamic_linker and
                              (not cc_helper.is_compilation_outputs_empty(compilation_outputs) or
                               cc_common.is_enabled(
                                   feature_configuration = feature_configuration,
                                   feature_name = "header_module_codegen",
                               )))

    output_group_builder = {}

    has_compilation_outputs = not cc_helper.is_compilation_outputs_empty(compilation_outputs)
    linking_context = CcInfo().linking_context
    empty_archive_linking_context = CcInfo().linking_context

    linking_contexts = cc_helper.get_linking_contexts_from_deps(ctx.attr.deps)
    linking_contexts.extend(cc_helper.get_linking_contexts_from_deps(ctx.attr.implementation_deps))
    if ctx.file.linkstamp != None:
        linkstamps = []
        linkstamps.append(cc_internal.create_linkstamp(
            actions = ctx.actions,
            linkstamp = ctx.file.linkstamp,
            compilation_context = compilation_context,
        ))
        linkstamps_linker_input = cc_common.create_linker_input(
            owner = ctx.label,
            linkstamps = depset(linkstamps),
        )
        linkstamps_linking_context = cc_common.create_linking_context(
            linker_inputs = depset([linkstamps_linker_input]),
        )
        linking_contexts.append(linkstamps_linking_context)

    if has_compilation_outputs:
        dll_name_suffix = ""
        win_def_file = None
        is_windows_enabled = cc_common.is_enabled(feature_configuration = feature_configuration, feature_name = "targets_windows")
        if is_windows_enabled:
            dll_name_suffix = cc_helper.dll_hash_suffix(ctx, feature_configuration, ctx.fragments.cpp)
            generated_def_file = None

            def_parser = ctx.file._def_parser
            if def_parser != None:
                generated_def_file = cc_helper.generate_def_file(ctx, def_parser, compilation_outputs.objects, ctx.label.name + dll_name_suffix)
                output_group_builder["def_file"] = depset([generated_def_file])

            win_def_file = cc_helper.get_windows_def_file_for_linking(ctx, ctx.file.win_def_file, generated_def_file, feature_configuration)

        (
            linking_context,
            linking_outputs,
        ) = cc_common.create_linking_context_from_compilation_outputs(
            actions = ctx.actions,
            name = ctx.label.name,
            compilation_outputs = compilation_outputs,
            cc_toolchain = cc_toolchain,
            feature_configuration = feature_configuration,
            additional_inputs = _filter_linker_scripts(ctx.files.deps),
            linking_contexts = linking_contexts,
            grep_includes = ctx.executable._grep_includes,
            user_link_flags = cc_helper.linkopts(ctx, additional_make_variable_substitutions, cc_toolchain),
            alwayslink = ctx.attr.alwayslink,
            disallow_dynamic_library = not create_dynamic_library or is_windows_enabled and win_def_file == None,
            linked_dll_name_suffix = dll_name_suffix,
            win_def_file = win_def_file,
        )
    elif semantics.should_create_empty_archive():
        precompiled_files_count = 0
        for precompiled_files_entry in precompiled_files:
            precompiled_files_count += len(precompiled_files_entry)

        (
            linking_context,
            linking_outputs,
        ) = cc_common.create_linking_context_from_compilation_outputs(
            actions = ctx.actions,
            name = ctx.label.name,
            cc_toolchain = cc_toolchain,
            compilation_outputs = cc_common.create_compilation_outputs(),
            feature_configuration = feature_configuration,
            grep_includes = ctx.executable._grep_includes,
            disallow_dynamic_library = True,
            alwayslink = ctx.attr.alwayslink,
        )

        if precompiled_files_count == 0:
            empty_archive_linking_context = linking_context
    else:
        linking_outputs = struct(library_to_link = None)

    _add_linker_artifacts_output_groups(output_group_builder, linking_outputs)

    precompiled_libraries = _convert_precompiled_libraries_to_library_to_link(
        ctx,
        cc_toolchain,
        feature_configuration,
        ctx.fragments.cpp.force_pic(),
        precompiled_files,
    )

    if not cc_helper.is_compilation_outputs_empty(compilation_outputs):
        _check_if_link_outputs_colliding_with_precompiled_files(
            ctx,
            linking_outputs,
            precompiled_libraries,
        )

    precompiled_linking_context = cc_helper.build_linking_context_from_libraries(ctx, precompiled_libraries)

    contexts_to_merge = [precompiled_linking_context, empty_archive_linking_context]
    if has_compilation_outputs:
        contexts_to_merge.append(linking_context)
    else:
        user_link_flags = cc_helper.linkopts(ctx, additional_make_variable_substitutions, cc_toolchain)
        linker_scripts = _filter_linker_scripts(ctx.files.deps)
        if len(user_link_flags) > 0 or len(linker_scripts) > 0 or not semantics.should_create_empty_archive():
            linker_input = cc_common.create_linker_input(
                owner = ctx.label,
                user_link_flags = user_link_flags,
                additional_inputs = depset(linker_scripts),
            )
            contexts_to_merge.append(cc_common.create_linking_context(linker_inputs = depset([linker_input])))

        contexts_to_merge.extend(linking_contexts)

    linking_context = cc_common.merge_linking_contexts(
        linking_contexts = contexts_to_merge,
    )

    libraries_to_link = _create_libraries_to_link_list(
        linking_outputs.library_to_link,
        precompiled_libraries,
    )

    linking_context_for_runfiles = cc_helper.build_linking_context_from_libraries(ctx, libraries_to_link)

    cc_native_library_info = cc_helper.collect_native_cc_libraries(
        deps = ctx.attr.deps,
        libraries = libraries_to_link,
    )

    files_builder = []
    if linking_outputs.library_to_link != None:
        artifacts_to_build = linking_outputs.library_to_link
        if artifacts_to_build.static_library != None:
            files_builder.append(artifacts_to_build.static_library)

        if artifacts_to_build.pic_static_library != None:
            files_builder.append(artifacts_to_build.pic_static_library)

        if not cc_common.is_enabled(
            feature_configuration = feature_configuration,
            feature_name = "targets_windows",
        ):
            if artifacts_to_build.resolved_symlink_dynamic_library != None:
                files_builder.append(artifacts_to_build.resolved_symlink_dynamic_library)
            elif artifacts_to_build.dynamic_library != None:
                files_builder.append(artifacts_to_build.dynamic_library)

            if artifacts_to_build.resolved_symlink_interface_library != None:
                files_builder.append(artifacts_to_build.resolved_symlink_interface_library)
            elif artifacts_to_build.interface_library != None:
                files_builder.append(artifacts_to_build.interface_library)

    instrumented_files_info = cc_helper.create_cc_instrumented_files_info(
        ctx = ctx,
        cc_config = ctx.fragments.cpp,
        cc_toolchain = cc_toolchain,
        metadata_files = compilation_outputs.gcno_files() + compilation_outputs.pic_gcno_files(),
    )

    runfiles_list = []
    for data_dep in ctx.attr.data:
        if data_dep[DefaultInfo].data_runfiles.files:
            runfiles_list.append(data_dep[DefaultInfo].data_runfiles)
        else:
            # This branch ensures interop with custom Starlark rules following
            # https://bazel.build/extending/rules#runfiles_features_to_avoid
            runfiles_list.append(ctx.runfiles(transitive_files = data_dep[DefaultInfo].files))
            runfiles_list.append(data_dep[DefaultInfo].default_runfiles)

    for src in ctx.attr.srcs:
        runfiles_list.append(src[DefaultInfo].default_runfiles)

    for dep in ctx.attr.deps:
        runfiles_list.append(dep[DefaultInfo].default_runfiles)

    runfiles = ctx.runfiles().merge_all(runfiles_list)

    default_runfiles = ctx.runfiles(files = cc_helper.get_dynamic_libraries_for_runtime(linking_context_for_runfiles, True))
    default_runfiles = runfiles.merge(default_runfiles)

    data_runfiles = ctx.runfiles(files = cc_helper.get_dynamic_libraries_for_runtime(linking_context_for_runfiles, False))
    data_runfiles = runfiles.merge(data_runfiles)

    current_output_groups = cc_helper.build_output_groups_for_emitting_compile_providers(
        compilation_outputs,
        compilation_context,
        ctx.fragments.cpp,
        cc_toolchain,
        feature_configuration,
        ctx,
        generate_hidden_top_level_group = True,
    )
    providers = []

    providers.append(DefaultInfo(
        files = depset(files_builder),
        default_runfiles = default_runfiles,
        data_runfiles = data_runfiles,
    ))

    debug_context = cc_helper.merge_cc_debug_contexts(compilation_outputs, cc_helper.get_providers(ctx.attr.deps, CcInfo))
    cc_info = CcInfo(
        compilation_context = compilation_context,
        linking_context = linking_context,
        debug_context = debug_context,
        cc_native_library_info = cc_native_library_info,
    )

    merged_output_groups = cc_helper.merge_output_groups(
        [current_output_groups, output_group_builder],
    )

    providers.append(cc_info)
    providers.append(OutputGroupInfo(**merged_output_groups))
    providers.append(instrumented_files_info)

    return providers

def _add_linker_artifacts_output_groups(output_group_builder, linking_outputs):
    archive_file = []
    dynamic_library = []

    lib = linking_outputs.library_to_link

    if lib == None:
        return

    if lib.static_library != None:
        archive_file.append(lib.static_library)
    elif lib.pic_static_library != None:
        archive_file.append(lib.pic_static_library)

    if lib.resolved_symlink_dynamic_library != None:
        dynamic_library.append(lib.resolved_symlink_dynamic_library)
    elif lib.dynamic_library != None:
        dynamic_library.append(lib.dynamic_library)

    if lib.resolved_symlink_interface_library != None:
        dynamic_library.append(lib.resolved_symlink_interface_library)
    elif lib.interface_library != None:
        dynamic_library.append(lib.interface_library)

    output_group_builder["archive"] = depset(archive_file)
    output_group_builder["dynamic_library"] = depset(dynamic_library)

def _convert_precompiled_libraries_to_library_to_link(
        ctx,
        cc_toolchain,
        feature_configuration,
        force_pic,
        precompiled_files):
    static_libraries = _build_map_identifier_to_artifact(precompiled_files[2])
    pic_static_libraries = _build_map_identifier_to_artifact(precompiled_files[3])
    alwayslink_static_libraries = _build_map_identifier_to_artifact(precompiled_files[4])
    alwayslink_pic_static_libraries = _build_map_identifier_to_artifact(precompiled_files[5])
    dynamic_libraries = _build_map_identifier_to_artifact(precompiled_files[6])

    libraries = []

    identifiers_used = {}
    static_libraries_it = []
    static_libraries_it.extend(static_libraries.items())
    static_libraries_it.extend(alwayslink_static_libraries.items())
    for identifier, v in static_libraries_it:
        static_library = None
        pic_static_library = None
        dynamic_library = None
        interface_library = None

        has_pic = identifier in pic_static_libraries
        has_always_pic = identifier in alwayslink_pic_static_libraries
        if has_pic or has_always_pic:
            if has_pic:
                pic_static_library = pic_static_libraries[identifier]
            else:
                pic_static_library = alwayslink_pic_static_libraries[identifier]
        if not force_pic or not (has_pic or has_always_pic):
            static_library = v

        if identifier in dynamic_libraries:
            dynamic_library = dynamic_libraries[identifier]

        identifiers_used[identifier] = True

        library = cc_common.create_library_to_link(
            actions = ctx.actions,
            feature_configuration = feature_configuration,
            cc_toolchain = cc_toolchain,
            static_library = static_library,
            pic_static_library = pic_static_library,
            dynamic_library = dynamic_library,
            alwayslink = identifier in alwayslink_static_libraries,
        )
        libraries.append(library)

    pic_static_libraries_it = []
    pic_static_libraries_it.extend(pic_static_libraries.items())
    pic_static_libraries_it.extend(alwayslink_pic_static_libraries.items())
    for identifier, v in pic_static_libraries_it:
        if identifier in identifiers_used:
            continue

        pic_static_library = v
        if identifier in dynamic_libraries:
            dynamic_library = dynamic_libraries[identifier]

        identifiers_used[identifier] = True

        library = cc_common.create_library_to_link(
            actions = ctx.actions,
            feature_configuration = feature_configuration,
            cc_toolchain = cc_toolchain,
            pic_static_library = pic_static_library,
            alwayslink = identifier in alwayslink_static_libraries,
        )
        libraries.append(library)

    for identifier, v in dynamic_libraries.items():
        if identifier in identifiers_used:
            continue

        dynamic_library = dynamic_libraries[identifier]

        library = cc_common.create_library_to_link(
            actions = ctx.actions,
            feature_configuration = feature_configuration,
            cc_toolchain = cc_toolchain,
            dynamic_library = dynamic_library,
        )
        libraries.append(library)

    return libraries

def _build_map_identifier_to_artifact(artifacts):
    libraries = {}
    for artifact in artifacts:
        identifier = _identifier_of_artifact(artifact)
        if identifier in libraries:
            fail(
                "Trying to link twice a library with the same identifier '{}',".format(identifier) +
                "files: {} and {}".format(
                    artifact.short_path,
                    libraries[identifier].short_path,
                ),
                attr = "srcs",
            )
        libraries[identifier] = artifact
    return libraries

def _identifier_of_artifact(artifact):
    name = artifact.short_path
    for pic_suffix in [".pic.a", ".nopic.a", ".pic.lo"]:
        if name.endswith(pic_suffix):
            return name[:len(name) - len(pic_suffix)]

    return name[:len(name) - len(artifact.extension) - 1]

def _identifier_of_library(library):
    if library.static_library != None:
        return _identifier_of_artifact(library.static_library)
    if library.pic_static_library != None:
        return _identifier_of_artifact(library.pic_static_library)
    if library.dynamic_library != None:
        return _identifier_of_artifact(library.dynamic_library)
    if library.interface_library != None:
        return _identifier_of_artifact(library.interface_libary)

    return None

def _create_libraries_to_link_list(current_library, precompiled_libraries):
    libraries = []
    libraries.extend(precompiled_libraries)
    if current_library != None:
        libraries.append(current_library)

    return libraries

def _filter_linker_scripts(files):
    linker_scripts = []
    for file in files:
        extension = "." + file.extension
        if extension in LINKER_SCRIPT:
            linker_scripts.append(file)
    return linker_scripts

def _check_if_link_outputs_colliding_with_precompiled_files(ctx, linking_outputs, precompiled_libraries):
    identifier = _identifier_of_library(linking_outputs.library_to_link)
    for precompiled_library in precompiled_libraries:
        precompiled_library_identifier = _identifier_of_library(precompiled_library)
        if precompiled_library_identifier == identifier:
            fail("Can't put library with identifier '{}' into the srcs of a cc_library with".format(identifier) +
                 " the same name ({}) which also contains other code or objects to link".format(
                     ctx.label.name,
                 ))

def _check_no_repeated_srcs(ctx):
    seen = {}
    for target in ctx.attr.srcs:
        if DefaultInfo in target:
            for file in target.files.to_list():
                extension = "." + file.extension
                if extension not in cc_helper.extensions.CC_HEADER:
                    if extension in cc_helper.extensions.CC_AND_OBJC:
                        if file.path in seen:
                            if seen[file.path] != target.label:
                                fail("Artifact '{}' is duplicated (through ".format(file.path) +
                                     "'{}' and '{}')".format(str(seen[file.path]), str(target.label)))
                        seen[file.path] = target.label

ALLOWED_SRC_FILES = []
ALLOWED_SRC_FILES.extend(cc_helper.extensions.CC_SOURCE)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.C_SOURCE)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.CC_HEADER)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.ASSESMBLER_WITH_C_PREPROCESSOR)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.ASSEMBLER)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.ARCHIVE)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.PIC_ARCHIVE)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.ALWAYSLINK_LIBRARY)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.ALWAYSLINK_PIC_LIBRARY)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.SHARED_LIBRARY)

SRCS_FOR_COMPILATION = []
SRCS_FOR_COMPILATION.extend(cc_helper.extensions.CC_SOURCE)
SRCS_FOR_COMPILATION.extend(cc_helper.extensions.C_SOURCE)
SRCS_FOR_COMPILATION.extend(cc_helper.extensions.ASSESMBLER_WITH_C_PREPROCESSOR)
SRCS_FOR_COMPILATION.extend(cc_helper.extensions.ASSEMBLER)

ALLOWED_SRC_FILES.extend(cc_helper.extensions.OBJECT_FILE)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.PIC_OBJECT_FILE)

LINKER_SCRIPT = [".ld", ".lds", ".ldscript"]
PREPROCESSED_C = [".i"]
DEPS_ALLOWED_RULES = [
    "genrule",
    "cc_library",
    "cc_inc_library",
    "cc_embed_data",
    "go_library",
    "objc_library",
    "cc_import",
    "cc_proto_library",
    "gentpl",
    "gentplvars",
    "genantlr",
    "sh_library",
    "cc_binary",
    "cc_test",
]

attrs = {
    "srcs": attr.label_list(
        allow_files = True,
        flags = ["DIRECT_COMPILE_TIME_INPUT"],
    ),
    "alwayslink": attr.bool(default = False),
    "linkstatic": attr.bool(default = False),
    "implementation_deps": attr.label_list(providers = [CcInfo], allow_files = False),
    "hdrs": attr.label_list(
        allow_files = True,
        flags = ["ORDER_INDEPENDENT", "DIRECT_COMPILE_TIME_INPUT"],
    ),
    "strip_include_prefix": attr.string(),
    "include_prefix": attr.string(),
    "textual_hdrs": attr.label_list(
        allow_files = True,
        flags = ["ORDER_INDEPENDENT", "DIRECT_COMPILE_TIME_INPUT"],
    ),
    "linkstamp": attr.label(allow_single_file = True),
    "linkopts": attr.string_list(),
    "includes": attr.string_list(),
    "defines": attr.string_list(),
    "copts": attr.string_list(),
    "hdrs_check": attr.string(default = cc_internal.default_hdrs_check_computed_default()),
    "local_defines": attr.string_list(),
    "deps": attr.label_list(
        providers = [CcInfo],
        flags = ["SKIP_ANALYSIS_TIME_FILETYPE_CHECK"],
        allow_files = LINKER_SCRIPT + PREPROCESSED_C,
        allow_rules = DEPS_ALLOWED_RULES,
    ),
    "data": attr.label_list(
        allow_files = True,
        flags = ["SKIP_CONSTRAINTS_OVERRIDE"],
    ),
    "win_def_file": attr.label(allow_single_file = [".def"]),
    # buildifier: disable=attr-license
    "licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
    "_stl": semantics.get_stl(),
    "_grep_includes": attr.label(
        allow_files = True,
        executable = True,
        cfg = "exec",
        default = Label("@" + semantics.get_repo() + "//tools/cpp:grep-includes"),
    ),
    "_def_parser": semantics.get_def_parser(),
    "_cc_toolchain": attr.label(default = "@" + semantics.get_repo() + "//tools/cpp:current_cc_toolchain"),
}
attrs.update(semantics.get_distribs_attr())
attrs.update(semantics.get_loose_mode_in_hdrs_check_allowed_attr())
attrs.update(semantics.get_implementation_deps_allowed_attr())
attrs.update(semantics.get_nocopts_attr())

cc_library = rule(
    implementation = _cc_library_impl,
    attrs = attrs,
    toolchains = cc_helper.use_cpp_toolchain() +
                 semantics.get_runtimes_toolchain(),
    fragments = ["cpp"] + semantics.additional_fragments(),
    incompatible_use_toolchain_transition = True,
    provides = [CcInfo],
    exec_groups = {
        "cpp_link": exec_group(copy_from_rule = True),
    },
    compile_one_filetype = [".cc", ".h", ".c"],
)
