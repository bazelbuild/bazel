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

load("@_builtins//:common/cc/cc_helper.bzl", "cc_helper")
load("@_builtins//:common/objc/objc_common.bzl", "objc_common")

objc_internal = _builtins.internal.objc_internal
cc_common = _builtins.toplevel.cc_common

def _build_variable_extensions(
        common_variables,
        ctx,
        intermediate_artifacts,
        variable_categories,
        arc_enabled):
    extensions = {}
    if hasattr(ctx.attr, "pch") and ctx.attr.pch != None:
        extensions["pch_file"] = ctx.file.pch.path

    extensions["module_maps_dir"] = intermediate_artifacts.swift_module_map.file().path

    extensions["modules_cache_path"] = ctx.genfiles_dir.path + "/" + "_objc_module_cache"

    if "ARCHIVE_VARIABLE" in variable_categories:
        extensions["obj_list_path"] = intermediate_artifacts.archive_obj_list.path

    if arc_enabled:
        extensions["objc_arc"] = ""
    else:
        extensions["no_objc_arc"] = ""

    return extensions

def _build_common_variables(
        ctx,
        toolchain,
        use_pch,
        disable_layering_check,
        disable_parse_hdrs,
        empty_compilation_artifacts,
        deps,
        runtime_deps,
        extra_import_libraries,
        linkopts):
    compilation_attributes = objc_internal.create_compilation_attributes(ctx = ctx)
    intermediate_artifacts = objc_internal.create_intermediate_artifacts(ctx = ctx)
    if empty_compilation_artifacts:
        compilation_artifacts = objc_internal.create_compilation_artifacts()
    else:
        compilation_artifacts = objc_internal.create_compilation_artifacts(ctx = ctx)

    (objc_provider, objc_compilation_context) = objc_common.create_context_and_provider(
        purpose = "COMPILE_AND_LINK",
        ctx = ctx,
        compilation_attributes = compilation_attributes,
        compilation_artifacts = compilation_artifacts,
        deps = deps,
        runtime_deps = runtime_deps,
        intermediate_artifacts = intermediate_artifacts,
        alwayslink = ctx.attr.alwayslink,
        has_module_map = True,
        extra_import_libraries = extra_import_libraries,
        linkopts = linkopts,
    )

    return struct(
        ctx = ctx,
        intermediate_artifacts = intermediate_artifacts,
        compilation_attributes = compilation_attributes,
        compilation_artifacts = compilation_artifacts,
        objc_compilation_context = objc_compilation_context,
        toolchain = toolchain,
        use_pch = use_pch,
        disable_layering_check = disable_layering_check,
        disable_parse_headers = disable_parse_hdrs,
        objc_config = ctx.fragments.objc,
        objc_provider = objc_provider,
    )

def _build_feature_configuration(common_variables, for_swift_module_map, support_parse_headers):
    activated_crosstool_selectables = []
    ctx = common_variables.ctx
    OBJC_ACTIONS = [
        "objc-compile",
        "objc++-compile",
        "objc-archive",
        "objc-fully-link",
        "objc-executable",
        "objc++-executable",
    ]
    activated_crosstool_selectables.extend(ctx.features)
    activated_crosstool_selectables.extend(OBJC_ACTIONS)
    activated_crosstool_selectables.append("lang_objc")
    if common_variables.objc_config.should_strip_binary:
        activated_crosstool_selectables.append("dead_strip")

    if common_variables.objc_config.generate_linkmap:
        activated_crosstool_selectables.append("generate_linkmap")

    disabled_features = []
    disabled_features.extend(ctx.disabled_features)
    if common_variables.disable_parse_headers:
        disabled_features.append("parse_headers")

    if common_variables.disable_layering_check:
        disabled_features.append("layering_check")

    if not support_parse_headers:
        disabled_features.append("parse_headers")

    if for_swift_module_map:
        activated_crosstool_selectables.append("module_maps")
        activated_crosstool_selectables.append("compile_all_modules")
        activated_crosstool_selectables.append("only_doth_headers_in_module_maps")
        activated_crosstool_selectables.append("exclude_private_headers_in_module_maps")
        activated_crosstool_selectables.append("module_map_without_extern_module")
        disabled_features.append("generate_submodules")

    return cc_common.configure_features(
        ctx = common_variables.ctx,
        cc_toolchain = common_variables.toolchain,
        requested_features = activated_crosstool_selectables,
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
        generate_module_map,
        should_process_headers):
    objc_compilation_context = common_variables.objc_compilation_context
    includes = []
    includes.extend(priority_headers)
    includes.extend(objc_compilation_context.includes)

    user_compile_flags = []
    user_compile_flags.extend(_get_compile_rule_copts(common_variables))
    user_compile_flags.extend(common_variables.objc_config.copts_for_current_compilation_mode)
    user_compile_flags.extend(extra_compile_args)
    user_compile_flags.extend(_paths_to_include_args(objc_compilation_context.strict_dependency_includes))

    textual_hdrs = []
    textual_hdrs.extend(objc_compilation_context.public_textual_hdrs)
    if pch_hdr != None:
        textual_hdrs.append(pch_hdr)

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
        compilation_contexts = objc_compilation_context.cc_compilation_contexts,
        user_compile_flags = user_compile_flags,
        grep_includes = common_variables.ctx.executable._grep_includes,
        module_map = module_map,
        propagate_module_map_to_compile_action = True,
        variables_extension = extension,
        language = "objc",
        code_coverage_enabled = cc_helper.is_code_coverage_enabled(ctx = common_variables.ctx),
        hdrs_checking_mode = "strict",
        do_not_generate_module_map = module_map.file().is_source or not generate_module_map,
        purpose = purpose,
    )

def _validate_attributes(common_variables):
    for include in common_variables.compilation_attributes.includes.to_list():
        if include.startswith("/"):
            cc_helper.rule_error("The path '{}' is absolute, but only relative paths are allowed.".format(include))

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

    if ctx.attr.module_name != "" and ctx.attr.module_map != None:
        cc_helper.attribute_error("module_name", "Specifying both module_name and module_map is invalid, please remove one of them.")

def _get_compile_rule_copts(common_variables):
    attributes = common_variables.compilation_attributes
    copts = []
    copts.extend(common_variables.objc_config.copts)
    copts.extend(attributes.copts)

    if attributes.enable_modules and common_variables.ctx.attr.module_map == None:
        copts.append("-fmodules")

    if "-fmodules" in copts:
        cache_path = common_variables.ctx.genfiles_dir.path + "/" + "_objc_module_cache"
        copts.append("-fmodules-cache-path=" + cache_path)

    return copts

def _register_obj_file_list_action(common_variables, obj_files, obj_list):
    args = common_variables.ctx.actions.args()
    args.set_param_file_format("multiline")
    args.add_all(obj_files)
    common_variables.ctx.actions.write(obj_list, args)

def _paths_to_include_args(paths):
    new_paths = []
    for path in paths:
        new_paths.append("-I" + path)
    return new_paths

def _register_compile_and_archive_actions(
        common_variables,
        extra_compile_args,
        priority_headers):
    compilation_result = None

    if common_variables.compilation_artifacts.archive != None:
        obj_list = common_variables.intermediate_artifacts.archive_obj_list

        compilation_result = _cc_compile_and_link(
            common_variables,
            extra_compile_args,
            priority_headers,
            "OBJC_ARCHIVE",
            obj_list,
            ["ARCHIVE_VARIABLE"],
        )

        _register_obj_file_list_action(
            common_variables,
            compilation_result[1].objects,
            obj_list,
        )
    else:
        compilation_result = _cc_compile_and_link(
            common_variables,
            extra_compile_args,
            priority_headers,
            None,
            None,
            [],
        )

    return compilation_result

def _cc_compile_and_link(
        common_variables,
        extra_compile_args,
        priority_headers,
        link_type,
        link_action_input,
        variable_categories):
    compilation_artifacts = common_variables.compilation_artifacts
    intermediate_artifacts = common_variables.intermediate_artifacts
    compilation_attributes = common_variables.compilation_attributes
    ctx = common_variables.ctx
    (objects, pic_objects) = _get_object_files(common_variables.ctx)
    public_hdrs = []
    public_hdrs.extend(compilation_attributes.hdrs.to_list())
    public_hdrs.extend(compilation_artifacts.additional_hdrs.to_list())
    pch_header = _get_pch_file(common_variables)
    feature_configuration = _build_feature_configuration(common_variables, False, True)

    # Generate up to two module maps, while minimizing the number of actions created.  If
    # module_map feature is off, generate a swift module map.  If module_map feature is on,
    # generate a layering check and a swift module map.  In the latter case, the layering check
    # module map must be the primary one.
    #
    # TODO(waltl): Delete this logic when swift module map is migrated to swift_library.

    primary_module_map = None
    arc_primary_module_map_fc = None
    non_arc_primary_module_map_fc = None
    extra_module_map = None
    extra_module_map_fc = None
    fc_for_swift_module_map = _build_feature_configuration(
        common_variables,
        True,
        True,
    )
    if cc_common.is_enabled(feature_configuration = feature_configuration, feature_name = "module_maps"):
        primary_module_map = intermediate_artifacts.internal_module_map
        arc_primary_module_map_fc = feature_configuration
        non_arc_primary_module_map_fc = _build_feature_configuration(
            common_variables,
            True,
            False,
        )
        extra_module_map = intermediate_artifacts.swift_module_map
        extra_module_map_fc = fc_for_swift_module_map
    else:
        primary_module_map = intermediate_artifacts.swift_module_map
        arc_primary_module_map_fc = fc_for_swift_module_map
        non_arc_primary_module_map_fc = _build_feature_configuration(
            common_variables,
            True,
            False,
        )
        extra_module_map = None
        extra_module_map_fc = None

    purpose = "{}_objc_arc".format(_get_purpose(common_variables))
    arc_extensions = _build_variable_extensions(
        common_variables,
        ctx,
        intermediate_artifacts,
        variable_categories,
        True,
    )

    (arc_compilation_context, arc_compilation_outputs) = _compile(
        common_variables,
        arc_primary_module_map_fc,
        arc_extensions,
        extra_compile_args,
        priority_headers,
        compilation_artifacts.srcs,
        compilation_artifacts.private_hdrs,
        public_hdrs,
        pch_header,
        primary_module_map,
        purpose,
        True,
        True,
    )
    purpose = "{}_non_objc_arc".format(_get_purpose(common_variables))
    non_arc_extensions = _build_variable_extensions(
        common_variables,
        ctx,
        intermediate_artifacts,
        variable_categories,
        False,
    )
    (non_arc_compilation_context, non_arc_compilation_outputs) = _compile(
        common_variables,
        non_arc_primary_module_map_fc,
        non_arc_extensions,
        extra_compile_args,
        priority_headers,
        compilation_artifacts.non_arc_srcs,
        compilation_artifacts.private_hdrs,
        public_hdrs,
        pch_header,
        primary_module_map,
        purpose,
        False,
        False,
    )

    objc_compilation_context = common_variables.objc_compilation_context

    if extra_module_map != None and not extra_module_map.file().is_source:
        _generate_extra_module_map(
            common_variables,
            extra_module_map,
            public_hdrs,
            compilation_artifacts.private_hdrs,
            objc_compilation_context.public_textual_hdrs,
            pch_header,
            objc_compilation_context.cc_compilation_contexts,
            extra_module_map_fc,
        )

    if link_type == "OBJC_ARCHIVE":
        language = "objc"
    else:
        language = "c++"

    additional_inputs = []
    if link_action_input != None:
        additional_inputs.append(link_action_input)

    cc_compilation_context = cc_common.merge_compilation_contexts(
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

    cc_common.create_linking_context_from_compilation_outputs(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = common_variables.toolchain,
        compilation_outputs = compilation_outputs,
        linking_contexts = cc_helper.get_linking_contexts_from_deps(common_variables.ctx.attr.deps),
        name = common_variables.ctx.label.name + intermediate_artifacts.archive_file_name_suffix,
        language = language,
        disallow_dynamic_library = True,
        additional_inputs = additional_inputs,
        grep_includes = ctx.executable._grep_includes,
        variables_extension = non_arc_extensions,
    )

    arc_output_groups = cc_helper.build_output_groups_for_emitting_compile_providers(
        arc_compilation_outputs,
        arc_compilation_context,
        ctx.fragments.cpp,
        common_variables.toolchain,
        feature_configuration,
        ctx,
        True,
    )
    non_arc_output_groups = cc_helper.build_output_groups_for_emitting_compile_providers(
        non_arc_compilation_outputs,
        non_arc_compilation_context,
        ctx.fragments.cpp,
        common_variables.toolchain,
        feature_configuration,
        ctx,
        True,
    )

    merged_output_groups = cc_helper.merge_output_groups(
        [arc_output_groups, non_arc_output_groups],
    )

    return (cc_compilation_context, compilation_outputs, OutputGroupInfo(**merged_output_groups))

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
        grep_includes = common_variables.ctx.executable._grep_includes,
    )

compilation_support = struct(
    register_compile_and_archive_actions = _register_compile_and_archive_actions,
    build_common_variables = _build_common_variables,
    build_feature_configuration = _build_feature_configuration,
    validate_attributes = _validate_attributes,
)
