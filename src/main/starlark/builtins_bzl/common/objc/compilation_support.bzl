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
load(":common/cc/semantics.bzl", cc_semantics = "semantics")
load(":common/objc/compilation_artifacts_info.bzl", "CompilationArtifactsInfo")
load(":common/objc/intermediate_artifacts.bzl", "create_intermediate_artifacts")
load(":common/objc/objc_common.bzl", "objc_common")
load(":common/paths.bzl", "paths")

_cc_internal = _builtins.internal.cc_internal

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

def _create_compilation_attributes(ctx):
    disallow_sdk_frameworks = ctx.fragments.objc.disallow_sdk_frameworks_attributes
    sdk_frameworks = getattr(ctx.attr, "sdk_frameworks", [])
    weak_sdk_frameworks = getattr(ctx.attr, "weak_sdk_frameworks", [])
    if disallow_sdk_frameworks:
        if sdk_frameworks:
            fail("sdk_frameworks attribute is disallowed. Use explicit dependencies instead.")
        if weak_sdk_frameworks:
            fail("weak_sdk_frameworks attribute is disallowed. Use explicit dependencies instead.")

    return struct(
        hdrs = depset([artifact for artifact, _ in cc_helper.get_public_hdrs(ctx)]),
        textual_hdrs = depset(getattr(ctx.files, "textual_hdrs", [])),
        sdk_includes = depset(getattr(ctx.attr, "sdk_includes", [])),
        includes = depset(getattr(ctx.attr, "includes", [])),
        sdk_frameworks = depset(sdk_frameworks),
        weak_sdk_frameworks = depset(weak_sdk_frameworks),
        sdk_dylibs = depset(getattr(ctx.attr, "sdk_dylibs", [])),
        linkopts = _cc_internal.expand_and_tokenize(ctx = ctx, attr = "linkopts", flags = getattr(ctx.attr, "linkopts", [])),
        copts = _cc_internal.expand_and_tokenize(ctx = ctx, attr = "copts", flags = getattr(ctx.attr, "copts", [])),
        conlyopts = _cc_internal.expand_and_tokenize(ctx = ctx, attr = "conlyopts", flags = getattr(ctx.attr, "conlyopts", [])),
        cxxopts = _cc_internal.expand_and_tokenize(ctx = ctx, attr = "cxxopts", flags = getattr(ctx.attr, "cxxopts", [])),
        additional_linker_inputs = getattr(ctx.files, "additional_linker_inputs", []),
        defines = _cc_internal.expand_and_tokenize(ctx = ctx, attr = "defines", flags = getattr(ctx.attr, "defines", [])),
        enable_modules = getattr(ctx.attr, "enable_modules", False),
    )

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
    compilation_attributes = _create_compilation_attributes(ctx = ctx)
    intermediate_artifacts = create_intermediate_artifacts(ctx = ctx)
    if empty_compilation_artifacts:
        compilation_artifacts = CompilationArtifactsInfo()
    else:
        compilation_artifacts = CompilationArtifactsInfo(
            ctx = ctx,
            intermediate_artifacts = intermediate_artifacts,
        )

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
        includes = cc_helper.include_dirs(ctx, {}) if hasattr(ctx.attr, "includes") else [],
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
        cxx_flags = common_variables.compilation_attributes.cxxopts,
        conly_flags = common_variables.compilation_attributes.conlyopts,
        module_map = module_map,
        variables_extension = extension,
        language = "objc",
        code_coverage_enabled = cc_helper.is_code_coverage_enabled(ctx = common_variables.ctx),
        hdrs_checking_mode = "strict",
        do_not_generate_module_map = not generate_module_map or module_map.file.is_source,
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
        module_map = intermediate_artifacts.internal_module_map()

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
            intermediate_artifacts.swift_module_map(),
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

compilation_support = struct(
    register_compile_and_archive_actions = _register_compile_and_archive_actions,
    build_common_variables = _build_common_variables,
    validate_attributes = _validate_attributes,
)
