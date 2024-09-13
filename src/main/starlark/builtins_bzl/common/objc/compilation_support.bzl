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

"""Utility methods used for creating objc_* rules actions"""

load(":common/cc/cc_common.bzl", "cc_common")
load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/cc/cc_info.bzl", "CcInfo")
load(":common/cc/semantics.bzl", cc_semantics = "semantics")
load(":common/objc/apple_env.bzl", "apple_host_system_env", "target_apple_env")
load(":common/objc/objc_common.bzl", "objc_common")
load(":common/objc/providers.bzl", "J2ObjcEntryClassInfo", "J2ObjcMappingFileInfo")
load(":common/xcode/providers.bzl", "XcodeVersionInfo")

objc_internal = _builtins.internal.objc_internal

def _build_variable_extensions(ctx, arc_enabled):
    extensions = {}
    if hasattr(ctx.attr, "pch") and ctx.attr.pch != None:
        extensions["pch_file"] = ctx.file.pch.path

    extensions["modules_cache_path"] = ctx.genfiles_dir.path + "/" + "_objc_module_cache"

    if arc_enabled:
        extensions["objc_arc"] = ""
    else:
        extensions["no_objc_arc"] = ""

    return extensions

def _build_common_variables(
        ctx,
        toolchain,
        use_pch = False,
        empty_compilation_artifacts = False,
        deps = [],
        implementation_deps = [],
        extra_disabled_features = [],
        extra_enabled_features = [],
        attr_linkopts = [],
        alwayslink = False,
        has_module_map = False,
        direct_cc_compilation_contexts = []):
    compilation_attributes = objc_internal.create_compilation_attributes(ctx = ctx)
    intermediate_artifacts = objc_internal.create_intermediate_artifacts(ctx = ctx)
    if empty_compilation_artifacts:
        compilation_artifacts = objc_internal.create_compilation_artifacts()
    else:
        compilation_artifacts = objc_internal.create_compilation_artifacts(ctx = ctx)

    (
        objc_provider,
        objc_compilation_context,
        objc_linking_context,
    ) = objc_common.create_context_and_provider(
        ctx = ctx,
        compilation_attributes = compilation_attributes,
        compilation_artifacts = compilation_artifacts,
        deps = deps,
        implementation_deps = implementation_deps,
        intermediate_artifacts = intermediate_artifacts,
        has_module_map = has_module_map,
        attr_linkopts = attr_linkopts,
        direct_cc_compilation_contexts = direct_cc_compilation_contexts,
        includes = cc_helper.system_include_dirs(ctx, {}) if hasattr(ctx.attr, "includes") else [],
    )

    return struct(
        ctx = ctx,
        intermediate_artifacts = intermediate_artifacts,
        compilation_attributes = compilation_attributes,
        compilation_artifacts = compilation_artifacts,
        extra_disabled_features = extra_disabled_features,
        extra_enabled_features = extra_enabled_features,
        objc_compilation_context = objc_compilation_context,
        objc_linking_context = objc_linking_context,
        toolchain = toolchain,
        alwayslink = alwayslink,
        use_pch = use_pch,
        objc_config = ctx.fragments.objc,
        objc_provider = objc_provider,
    )

def _build_feature_configuration(common_variables, for_swift_module_map, support_parse_headers):
    ctx = common_variables.ctx

    enabled_features = []
    enabled_features.extend(ctx.features)
    enabled_features.extend(common_variables.extra_enabled_features)

    disabled_features = []
    disabled_features.extend(ctx.disabled_features)
    disabled_features.extend(common_variables.extra_disabled_features)

    if not support_parse_headers:
        disabled_features.append("parse_headers")

    if for_swift_module_map:
        enabled_features.append("module_maps")
        enabled_features.append("compile_all_modules")
        enabled_features.append("only_doth_headers_in_module_maps")
        enabled_features.append("exclude_private_headers_in_module_maps")
        enabled_features.append("module_map_without_extern_module")
        disabled_features.append("generate_submodules")

    return cc_common.configure_features(
        ctx = common_variables.ctx,
        cc_toolchain = common_variables.toolchain,
        language = "objc",
        requested_features = enabled_features,
        unsupported_features = disabled_features,
    )

def _compile(
        common_variables,
        feature_configuration,
        extension,
        extra_compile_args,
        priority_headers,
        srcs,
        private_hdrs,
        public_hdrs,
        pch_hdr,
        module_map,
        purpose,
        generate_module_map):
    objc_compilation_context = common_variables.objc_compilation_context

    user_compile_flags = []
    user_compile_flags.extend(_get_compile_rule_copts(common_variables))
    user_compile_flags.extend(common_variables.objc_config.copts_for_current_compilation_mode)
    user_compile_flags.extend(extra_compile_args)
    user_compile_flags.extend(
        _paths_to_include_args(objc_compilation_context.strict_dependency_includes),
    )

    textual_hdrs = []
    textual_hdrs.extend(objc_compilation_context.public_textual_hdrs)
    if pch_hdr != None:
        textual_hdrs.append(pch_hdr)

    compilation_contexts = (
        objc_compilation_context.cc_compilation_contexts +
        cc_helper.get_compilation_contexts_from_deps(
            cc_semantics.get_cc_runtimes(common_variables.ctx, True),
        )
    )
    runtimes_copts = cc_semantics.get_cc_runtimes_copts(common_variables.ctx)

    return cc_common.compile(
        actions = common_variables.ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = common_variables.toolchain,
        name = common_variables.ctx.label.name,
        srcs = srcs,
        public_hdrs = public_hdrs,
        private_hdrs = private_hdrs,
        textual_hdrs = textual_hdrs,
        defines = objc_compilation_context.defines,
        includes = objc_compilation_context.includes,
        system_includes = objc_compilation_context.system_includes,
        quote_includes = objc_compilation_context.quote_includes,
        compilation_contexts = compilation_contexts,
        implementation_compilation_contexts = objc_compilation_context.implementation_cc_compilation_contexts,
        user_compile_flags = runtimes_copts + user_compile_flags,
        module_map = module_map,
        propagate_module_map_to_compile_action = True,
        variables_extension = extension,
        language = "objc",
        code_coverage_enabled = cc_helper.is_code_coverage_enabled(ctx = common_variables.ctx),
        hdrs_checking_mode = "strict",
        do_not_generate_module_map = not generate_module_map or module_map.file().is_source,
        purpose = purpose,
    )

def _validate_attributes(common_variables):
    for include in common_variables.compilation_attributes.includes.to_list():
        if include.startswith("/"):
            cc_helper.rule_error(
                "The path '{}' is absolute, but only relative paths are allowed.".format(include),
            )

    ctx = common_variables.ctx
    if hasattr(ctx.attr, "srcs"):
        srcs = {}
        for src in ctx.files.srcs:
            srcs[src.path] = True
        for src in ctx.files.non_arc_srcs:
            if src.path in srcs:
                cc_helper.attribute_error(
                    "srcs",
                    "File '{}' is present in both srcs and non_arc_srcs which is forbidden.".format(src.path),
                )

    if hasattr(ctx.attr, "module_name") and hasattr(ctx.attr, "module_map"):
        if ctx.attr.module_name != "" and ctx.attr.module_map != None:
            cc_helper.attribute_error(
                "module_name",
                "Specifying both module_name and module_map is invalid, please remove one of them.",
            )

def _get_compile_rule_copts(common_variables):
    attributes = common_variables.compilation_attributes
    copts = []
    copts.extend(attributes.copts)

    if attributes.enable_modules and common_variables.ctx.attr.module_map == None:
        copts.append("-fmodules")

    if "-fmodules" in copts:
        cache_path = common_variables.ctx.genfiles_dir.path + "/" + "_objc_module_cache"
        copts.append("-fmodules-cache-path=" + cache_path)

    return copts

def _paths_to_include_args(paths):
    new_paths = []
    for path in paths:
        new_paths.append("-I" + path)
    return new_paths

# TODO(bazel-team): This method can be deleted as soon as the native j2objc
#  rules are deleted. The native rules are deprecated and will be replaced by
#  better Starlark rules that are not a literal translation of the native
#  implementation and use a better approach. This is not done by the Bazel team
# but a separate team (tball@). This method is added so that common utility code
# in CompilationSupport can be deleted from Java.
def _register_compile_and_archive_actions_for_j2objc(
        ctx,
        toolchain,
        intermediate_artifacts,
        compilation_artifacts,
        objc_compilation_context,
        cc_linking_contexts,
        extra_compile_args):
    compilation_attributes = objc_internal.create_compilation_attributes(ctx = ctx)

    objc_linking_context = struct(
        cc_linking_contexts = cc_linking_contexts,
        linkopts = [],
    )

    common_variables = struct(
        ctx = ctx,
        intermediate_artifacts = intermediate_artifacts,
        compilation_attributes = compilation_attributes,
        compilation_artifacts = compilation_artifacts,
        extra_enabled_features = ["j2objc_transpiled"],
        extra_disabled_features = ["layering_check", "parse_headers"],
        objc_compilation_context = objc_compilation_context,
        objc_linking_context = objc_linking_context,
        toolchain = toolchain,
        alwayslink = True,
        use_pch = False,
        objc_config = ctx.fragments.objc,
        objc_provider = None,
    )

    return _cc_compile_and_link(
        compilation_artifacts.srcs,
        compilation_artifacts.non_arc_srcs,
        [],
        compilation_artifacts.additional_hdrs,
        common_variables,
        extra_compile_args,
        priority_headers = [],
        generate_module_map_for_swift = True,
    )

def _register_compile_and_archive_actions(
        common_variables,
        extra_compile_args = [],
        priority_headers = [],
        generate_module_map_for_swift = False):
    ctx = common_variables.ctx
    return _cc_compile_and_link(
        cc_helper.get_srcs(ctx),
        _get_non_arc_srcs(ctx),
        cc_helper.get_private_hdrs(ctx),
        cc_helper.get_public_hdrs(ctx),
        common_variables,
        extra_compile_args,
        priority_headers,
        generate_module_map_for_swift = generate_module_map_for_swift,
    )

# Returns a list of (Artifact, Label) tuples. Each tuple represents an input source
# file and the label of the rule that generates it (or the label of the source file itself if it
# is an input file).
def _get_non_arc_srcs(ctx):
    if not hasattr(ctx.attr, "non_arc_srcs"):
        return []
    artifact_label_map = {}
    for src in ctx.attr.non_arc_srcs:
        if DefaultInfo in src:
            for artifact in src[DefaultInfo].files.to_list():
                artifact_label_map[artifact] = src.label
    return _map_to_list(artifact_label_map)

def _map_to_list(m):
    result = []
    for k, v in m.items():
        result.append((k, v))
    return result

def _cc_compile_and_link(
        srcs,
        non_arc_srcs,
        private_hdrs,
        public_hdrs,
        common_variables,
        extra_compile_args,
        priority_headers,
        generate_module_map_for_swift):
    intermediate_artifacts = common_variables.intermediate_artifacts
    compilation_attributes = common_variables.compilation_attributes
    ctx = common_variables.ctx
    (objects, pic_objects) = _get_object_files(common_variables.ctx)

    pch_header = _get_pch_file(common_variables)
    feature_configuration = _build_feature_configuration(
        common_variables,
        for_swift_module_map = False,
        support_parse_headers = True,
    )

    generate_module_map = cc_common.is_enabled(
        feature_configuration = feature_configuration,
        feature_name = "module_maps",
    )
    module_map = None
    if generate_module_map:
        module_map = intermediate_artifacts.internal_module_map

    purpose = "{}_objc_arc".format(_get_purpose(common_variables))
    arc_primary_module_map_fc = feature_configuration
    arc_extensions = _build_variable_extensions(ctx, arc_enabled = True)
    (arc_compilation_context, arc_compilation_outputs) = _compile(
        common_variables,
        arc_primary_module_map_fc,
        arc_extensions,
        extra_compile_args,
        priority_headers,
        srcs,
        private_hdrs,
        public_hdrs,
        pch_header,
        module_map,
        purpose,
        generate_module_map,
    )

    purpose = "{}_non_objc_arc".format(_get_purpose(common_variables))
    non_arc_primary_module_map_fc = _build_feature_configuration(
        common_variables,
        for_swift_module_map = False,
        support_parse_headers = False,
    )
    non_arc_extensions = _build_variable_extensions(ctx, arc_enabled = False)
    (non_arc_compilation_context, non_arc_compilation_outputs) = _compile(
        common_variables,
        non_arc_primary_module_map_fc,
        non_arc_extensions,
        extra_compile_args,
        priority_headers,
        non_arc_srcs,
        private_hdrs,
        public_hdrs,
        pch_header,
        module_map,
        purpose,
        generate_module_map = False,
    )

    objc_compilation_context = common_variables.objc_compilation_context

    if generate_module_map_for_swift:
        _generate_extra_module_map(
            common_variables,
            intermediate_artifacts.swift_module_map,
            public_hdrs,
            private_hdrs,
            objc_compilation_context.public_textual_hdrs,
            pch_header,
            objc_compilation_context.cc_compilation_contexts,
            _build_feature_configuration(
                common_variables,
                for_swift_module_map = True,
                support_parse_headers = False,
            ),
        )

    compilation_context = cc_common.merge_compilation_contexts(
        compilation_contexts = [arc_compilation_context, non_arc_compilation_context],
    )

    precompiled_compilation_outputs = cc_common.create_compilation_outputs(
        pic_objects = depset(pic_objects),
        objects = depset(objects),
    )

    compilation_outputs = cc_common.merge_compilation_outputs(
        compilation_outputs = [
            precompiled_compilation_outputs,
            arc_compilation_outputs,
            non_arc_compilation_outputs,
        ],
    )

    objc_linking_context = common_variables.objc_linking_context
    if len(compilation_outputs.objects) != 0 or len(compilation_outputs.pic_objects) != 0:
        (linking_context, _) = cc_common.create_linking_context_from_compilation_outputs(
            actions = ctx.actions,
            feature_configuration = feature_configuration,
            cc_toolchain = common_variables.toolchain,
            compilation_outputs = compilation_outputs,
            user_link_flags = objc_linking_context.linkopts,
            linking_contexts = objc_linking_context.cc_linking_contexts,
            name = common_variables.ctx.label.name + intermediate_artifacts.archive_file_name_suffix,
            language = "c++",
            alwayslink = common_variables.alwayslink,
            disallow_dynamic_library = True,
            variables_extension = non_arc_extensions,
        )
    else:
        linker_input = cc_common.create_linker_input(
            owner = ctx.label,
            user_link_flags = objc_linking_context.linkopts,
        )
        cc_linking_context = cc_common.create_linking_context(
            linker_inputs = depset(direct = [linker_input]),
        )
        linking_context = cc_common.merge_linking_contexts(
            linking_contexts = [cc_linking_context] + objc_linking_context.cc_linking_contexts,
        )

    arc_output_groups = cc_helper.build_output_groups_for_emitting_compile_providers(
        arc_compilation_outputs,
        arc_compilation_context,
        ctx.fragments.cpp,
        common_variables.toolchain,
        feature_configuration,
        ctx,
        generate_hidden_top_level_group = True,
    )
    non_arc_output_groups = cc_helper.build_output_groups_for_emitting_compile_providers(
        non_arc_compilation_outputs,
        non_arc_compilation_context,
        ctx.fragments.cpp,
        common_variables.toolchain,
        feature_configuration,
        ctx,
        generate_hidden_top_level_group = True,
    )

    merged_output_groups = cc_helper.merge_output_groups(
        [arc_output_groups, non_arc_output_groups],
    )

    return (compilation_context, linking_context, compilation_outputs, merged_output_groups)

def _get_object_files(ctx):
    if not hasattr(ctx.attr, "srcs"):
        return ([], [])

    pic_objects = []
    for src in ctx.files.srcs:
        path = src.path
        if path.endswith(".pic.o") or path.endswith(".o") and not path.endswith(".nopic.o"):
            pic_objects.append(src)

    objects = []
    for src in ctx.files.srcs:
        path = src.path
        if path.endswith(".o") and not path.endswith(".pic.o"):
            objects.append(src)

    return (objects, pic_objects)

def _get_pch_file(common_variables):
    if not common_variables.use_pch:
        return None

    pch_hdr = None
    if hasattr(common_variables.ctx.attr, "pch"):
        pch_hdr = common_variables.ctx.file.pch

    return pch_hdr

def _get_purpose(common_variables):
    suffix = common_variables.intermediate_artifacts.archive_file_name_suffix
    config = common_variables.ctx.bin_dir.path.split("/")[1]
    return "Objc_build_arch_" + config + "_with_suffix_" + suffix

def _generate_extra_module_map(
        common_variables,
        module_map,
        public_hdrs,
        private_hdrs,
        textual_hdrs,
        pch_header,
        compilation_contexts,
        feature_configuration):
    purpose = "{}_extra_module_map".format(_get_purpose(common_variables))
    all_textual_hdrs = []
    all_textual_hdrs.extend(textual_hdrs)
    if pch_header != None:
        all_textual_hdrs.append(pch_header)
    cc_common.compile(
        actions = common_variables.ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = common_variables.toolchain,
        public_hdrs = public_hdrs,
        textual_hdrs = textual_hdrs,
        private_hdrs = private_hdrs,
        compilation_contexts = compilation_contexts,
        module_map = module_map,
        purpose = purpose,
        name = common_variables.ctx.label.name,
    )

def _build_fully_linked_variable_extensions(archive, libs):
    extensions = {}
    extensions["fully_linked_archive_path"] = archive.path
    extensions["objc_library_exec_paths"] = [lib.path for lib in libs]
    extensions["cc_library_exec_paths"] = []
    extensions["imported_library_exec_paths"] = []
    return extensions

def _get_static_library_for_linking(library_to_link):
    if library_to_link.static_library:
        return library_to_link.static_library
    elif library_to_link.pic_static_library:
        return library_to_link.pic_static_library
    else:
        return None

def _get_library_for_linking(library_to_link):
    if library_to_link.static_library:
        return library_to_link.static_library
    elif library_to_link.pic_static_library:
        return library_to_link.pic_static_library
    elif library_to_link.interface_library:
        return library_to_link.interface_library
    else:
        return library_to_link.dynamic_library

def _get_libraries_for_linking(libraries_to_link):
    libraries = []
    for library_to_link in libraries_to_link:
        libraries.append(_get_library_for_linking(library_to_link))
    return libraries

def _register_fully_link_action(name, common_variables, cc_linking_context):
    ctx = common_variables.ctx
    feature_configuration = _build_feature_configuration(common_variables, False, False)

    libraries_to_link = cc_helper.libraries_from_linking_context(cc_linking_context).to_list()
    libraries = _get_libraries_for_linking(libraries_to_link)

    output_archive = ctx.actions.declare_file(name + ".a")
    extensions = _build_fully_linked_variable_extensions(
        output_archive,
        libraries,
    )

    return cc_common.link(
        name = name,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = common_variables.toolchain,
        language = "objc",
        additional_inputs = libraries,
        output_type = "archive",
        variables_extension = extensions,
    )

def _register_j2objc_dead_code_removal_actions(common_variables, deps, build_config):
    """Registers actions to perform J2Objc dead code removal (when enabled)."""
    ctx = common_variables.ctx

    j2objc_entry_class_info = objc_common.j2objc_entry_class_info_union(
        [dep[J2ObjcEntryClassInfo] for dep in deps if J2ObjcEntryClassInfo in dep],
    )
    entry_classes = j2objc_entry_class_info.entry_classes

    # Only perform J2ObjC dead code stripping if flag --j2objc_dead_code_removal is specified and
    # users have specified entry classes.
    if (not ctx.fragments.j2objc.remove_dead_code() or not entry_classes or
        not common_variables.objc_provider.j2objc_library.to_list()):
        return {}

    j2objc_mapping_file_info = objc_common.j2objc_mapping_file_info_union(
        [dep[J2ObjcMappingFileInfo] for dep in deps if J2ObjcMappingFileInfo in dep],
    )
    j2objc_dependency_mapping_files = j2objc_mapping_file_info.dependency_mapping_files
    j2objc_header_mapping_files = j2objc_mapping_file_info.header_mapping_files
    j2objc_archive_source_mapping_files = j2objc_mapping_file_info.archive_source_mapping_files

    replace_libs = {}
    for j2objc_archive in common_variables.objc_provider.j2objc_library.to_list():
        pruned_j2objc_archive = ctx.actions.declare_shareable_artifact(
            ctx.label.package + "/_j2objc_pruned/" + ctx.label.name + "/" +
            j2objc_archive.short_path[:-len(j2objc_archive.extension)].strip(".") +
            "_pruned." + j2objc_archive.extension,
            build_config.bin_dir,
        )
        replace_libs[j2objc_archive] = pruned_j2objc_archive

        # Although _dummy_lib is always a label, weirdly when it has a cfg attached to it
        # Bazel makes it a list (even if the cfg is not split)
        _dummy_lib = ctx.attr._dummy_lib if type(ctx.attr._dummy_lib) == "Target" else ctx.attr._dummy_lib[0]
        [dummy_archive] = _get_libraries_for_linking(cc_helper.libraries_from_linking_context(
            _dummy_lib[CcInfo].linking_context,
        ).to_list())

        args = ctx.actions.args()
        args.add("--input_archive", j2objc_archive)
        args.add("--output_archive", pruned_j2objc_archive)
        args.add("--dummy_archive", dummy_archive)
        args.add_joined(
            "--dependency_mapping_files",
            j2objc_dependency_mapping_files,
            join_with = ",",
        )
        args.add_joined("--header_mapping_files", j2objc_header_mapping_files, join_with = ",")
        args.add_joined(
            "--archive_source_mapping_files",
            j2objc_archive_source_mapping_files,
            join_with = ",",
        )
        args.add("--entry_classes")
        args.add_joined(entry_classes, join_with = ",")
        args.set_param_file_format("multiline")
        args.use_param_file("@%s", use_always = True)
        ctx.actions.run(
            mnemonic = "DummyPruner",
            executable = ctx.executable._j2objc_dead_code_pruner,
            inputs = depset(
                [j2objc_archive, dummy_archive],
                transitive = [
                    j2objc_dependency_mapping_files,
                    j2objc_header_mapping_files,
                    j2objc_archive_source_mapping_files,
                ],
            ),
            arguments = [args],
            outputs = [pruned_j2objc_archive],
            exec_group = "j2objc",
        )
    return replace_libs

def _register_obj_filelist_action(ctx, build_config, obj_files):
    """
    Returns a File containing the given set of object files.

    This File is suitable to signal symbols to archive in a libtool archiving invocation.
    """
    obj_list = ctx.actions.declare_shareable_artifact(
        ctx.label.package + "/" + ctx.label.name + "-linker.objlist",
        build_config.bin_dir,
    )

    args = ctx.actions.args()
    args.add_all(obj_files)
    args.set_param_file_format("multiline")
    ctx.actions.write(obj_list, args)

    return obj_list

def _register_binary_strip_action(
        ctx,
        name,
        binary,
        feature_configuration,
        build_config,
        extra_link_args):
    """
    Registers an action that uses the 'strip' tool to perform binary stripping on the given binary.
    """

    strip_safe = ctx.fragments.objc.strip_executable_safely

    # For dylibs, loadable bundles, and kexts, must strip only local symbols.
    link_dylib = cc_common.is_enabled(
        feature_configuration = feature_configuration,
        feature_name = "link_dylib",
    )
    link_bundle = cc_common.is_enabled(
        feature_configuration = feature_configuration,
        feature_name = "link_bundle",
    )
    if ("-dynamiclib" in extra_link_args or link_dylib or
        "-bundle" in extra_link_args or link_bundle or "-kext" in extra_link_args):
        strip_safe = True

    stripped_binary = ctx.actions.declare_shareable_artifact(
        ctx.label.package + "/" + name,
        build_config.bin_dir,
    )
    args = ctx.actions.args()
    args.add("strip")
    if strip_safe:
        args.add("-x")
    args.add("-o", stripped_binary)
    args.add(binary)
    xcode_config = ctx.attr._xcode_config[XcodeVersionInfo]
    platform = _builtins.internal.objc_internal.get_target_platform(build_config = build_config)
    ctx.actions.run(
        mnemonic = "ObjcBinarySymbolStrip",
        executable = "/usr/bin/xcrun",
        arguments = [args],
        inputs = [binary],
        outputs = [stripped_binary],
        execution_requirements = ctx.attr._xcode_config[XcodeVersionInfo].execution_info(),
        env = apple_host_system_env(xcode_config) |
              target_apple_env(xcode_config, platform),
    )
    return stripped_binary

def _dedup_sdk_linkopts(linker_inputs):
    duplicates = {}
    final_linkopts = []

    for linker_input in linker_inputs.to_list():
        flags = linker_input.user_link_flags
        previous_arg = None
        for arg in flags:
            if previous_arg in ["-framework", "-weak_framework"]:
                framework = arg
                key = previous_arg[1] + framework
                if key not in duplicates:
                    final_linkopts.extend([previous_arg, framework])
                    duplicates[key] = None
                previous_arg = None
            elif arg in ["-framework", "-weak_framework"]:
                previous_arg = arg
            elif arg.startswith("-Wl,-framework,") or arg.startswith("-Wl,-weak_framework,"):
                framework = arg.split(",")[2]
                key = arg[5] + framework
                if key not in duplicates:
                    final_linkopts.extend([arg.split(",")[1], framework])
                    duplicates[key] = None
            elif arg.startswith("-l"):
                if arg not in duplicates:
                    final_linkopts.append(arg)
                    duplicates[arg] = None
            else:
                final_linkopts.append(arg)

    return final_linkopts

def _linkstamp_map(ctx, linkstamps, output, build_config):
    # create linkstamps_map - mapping from linkstamps to object files
    linkstamps_map = {}

    stamp_output_dir = ctx.label.package + "/_objs/" + output.basename + "/"
    for linkstamp in linkstamps.to_list():
        linkstamp_file = linkstamp.file()
        stamp_output_path = (
            stamp_output_dir +
            linkstamp_file.short_path[:-len(linkstamp_file.extension)].rstrip(".") + ".o"
        )
        stamp_output_file = ctx.actions.declare_shareable_artifact(
            stamp_output_path,
            build_config.bin_dir,
        )
        linkstamps_map[linkstamp_file] = stamp_output_file
    return linkstamps_map

def _classify_libraries(libraries_to_link):
    always_link_libraries = {
        lib: None
        for lib in _get_libraries_for_linking(
            [lib for lib in libraries_to_link if lib.alwayslink],
        )
    }
    as_needed_libraries = {
        lib: None
        for lib in _get_libraries_for_linking(
            [lib for lib in libraries_to_link if not lib.alwayslink],
        )
        if lib not in always_link_libraries
    }
    return always_link_libraries.keys(), as_needed_libraries.keys()

def _register_configuration_specific_link_actions(
        name,
        common_variables,
        cc_linking_context,
        build_config,
        extra_link_args,
        stamp,
        user_variable_extensions,
        additional_outputs,
        deps,
        extra_link_inputs,
        attr_linkopts):
    """
    Registers actions to link a single-platform/architecture Apple binary in a specific config.

    Registers any actions necessary to link this rule and its dependencies. Automatically infers
    the toolchain from the configuration.

    Returns:
        (File) the linked binary
    """
    ctx = common_variables.ctx
    feature_configuration = _build_feature_configuration(common_variables, False, False)

    # We need to split input libraries into those that require -force_load and those that don't.
    # Clang loads archives specified in filelists and also specified as -force_load twice,
    # resulting in duplicate symbol errors unless they are deduped.
    libraries_to_link = cc_helper.libraries_from_linking_context(cc_linking_context).to_list()
    always_link_libraries, as_needed_libraries = _classify_libraries(libraries_to_link)

    replace_libs = _register_j2objc_dead_code_removal_actions(common_variables, deps, build_config)

    # Substitutes both sets of unpruned J2ObjC libraries with pruned ones
    always_link_libraries = [replace_libs.get(lib, lib) for lib in always_link_libraries]
    as_needed_libraries = [replace_libs.get(lib, lib) for lib in as_needed_libraries]

    static_runtimes = common_variables.toolchain.static_runtime_lib(
        feature_configuration = feature_configuration,
    )

    # When compilation_mode=opt and objc_enable_binary_stripping are specified, the unstripped
    # binary containing debug symbols is generated by the linker, which also needs the debug
    # symbols for dead-code removal. The binary is also used to generate dSYM bundle if
    # --apple_generate_dsym is specified. A symbol strip action is later registered to strip
    # the symbol table from the unstripped binary.
    if (ctx.fragments.cpp.objc_enable_binary_stripping() and
        ctx.fragments.cpp.compilation_mode() == "opt"):
        binary = ctx.actions.declare_shareable_artifact(
            ctx.label.package + "/" + name + "_unstripped",
            build_config.bin_dir,
        )
    else:
        binary = ctx.actions.declare_shareable_artifact(
            ctx.label.package + "/" + name,
            build_config.bin_dir,
        )

    # Passing large numbers of inputs on the command line triggers a bug in Apple's Clang
    # (b/29094356), so we'll create an input list manually and pass -filelist path/to/input/list.

    # Populate the input file list with both the compiled object files and any linkstamp object
    # files.
    # There's some weirdness: cc_common.link compiles linkstamps and does the linking (without ever
    # returning linkstamp objects)
    # We replicate the linkstamp objects names (guess them) and generate input_file_list
    # which is input to linking action.
    linkstamp_map = _linkstamp_map(ctx, cc_linking_context.linkstamps(), binary, build_config)
    input_file_list = _register_obj_filelist_action(
        ctx,
        build_config,
        as_needed_libraries + static_runtimes.to_list() + linkstamp_map.values(),
    )

    extensions = user_variable_extensions | {
        "framework_paths": [],
        "framework_names": [],
        "weak_framework_names": [],
        "library_names": [],
        "filelist": input_file_list.path,
        "linked_binary": binary.path,
        # artifacts to be passed to the linker with `-force_load`
        "force_load_exec_paths": [lib.path for lib in always_link_libraries],
        # linkopts from dependency
        "dep_linkopts": _dedup_sdk_linkopts(cc_linking_context.linker_inputs),
        "attr_linkopts": attr_linkopts,  # linkopts arising from rule attributes
    }
    additional_inputs = [
        input
        for linker_input in cc_linking_context.linker_inputs.to_list()
        for input in linker_input.additional_inputs
    ]
    cc_common.link(
        name = name,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = common_variables.toolchain,
        language = "objc",
        additional_inputs = (
            as_needed_libraries + always_link_libraries + [input_file_list] + extra_link_inputs +
            additional_inputs +
            getattr(ctx.files, "additional_linker_inputs", [])
        ),
        linking_contexts = [cc_common.create_linking_context(linker_inputs = depset(
            [cc_common.create_linker_input(
                owner = ctx.label,
                linkstamps = cc_linking_context.linkstamps(),
            )],
        ))],
        output_type = "executable",
        build_config = build_config,
        user_link_flags = extra_link_args,
        stamp = stamp,
        variables_extension = extensions,
        additional_outputs = additional_outputs,
        main_output = binary,
    )

    if not (ctx.fragments.cpp.objc_enable_binary_stripping() and
            ctx.fragments.cpp.compilation_mode() == "opt"):
        return binary
    else:
        return _register_binary_strip_action(ctx, name, binary, feature_configuration, build_config, extra_link_args)

compilation_support = struct(
    register_compile_and_archive_actions = _register_compile_and_archive_actions,
    register_compile_and_archive_actions_for_j2objc = _register_compile_and_archive_actions_for_j2objc,
    build_common_variables = _build_common_variables,
    build_feature_configuration = _build_feature_configuration,
    get_library_for_linking = _get_library_for_linking,
    get_static_library_for_linking = _get_static_library_for_linking,
    validate_attributes = _validate_attributes,
    register_fully_link_action = _register_fully_link_action,
    register_configuration_specific_link_actions = _register_configuration_specific_link_actions,
)
