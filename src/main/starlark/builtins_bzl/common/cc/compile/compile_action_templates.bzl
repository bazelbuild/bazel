# Copyright 2025 The Bazel Authors. All rights reserved.
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
"""Tree artifact compilation actions"""

load(
    ":common/cc/cc_helper_internal.bzl",
    "CPP_SOURCE_TYPE_HEADER",
    "CPP_SOURCE_TYPE_SOURCE",
    artifact_category = "artifact_category_names",
)
load(":common/cc/compile/cc_compilation_helper.bzl", "dotd_files_enabled", "serialized_diagnostics_file_enabled")
load(":common/cc/compile/compile_build_variables.bzl", "get_copts", "get_specific_compile_build_variables")
load(":common/cc/semantics.bzl", cc_semantics = "semantics")
load(":common/paths.bzl", "paths")

_cc_internal = _builtins.internal.cc_internal

# buildifier: disable=function-docstring
def create_compile_action_templates(
        *,
        action_construction_context,
        cc_compilation_context,
        cc_toolchain,
        configuration,
        cpp_configuration,
        feature_configuration,
        language,
        common_compile_build_variables,
        source_dir,
        source_type,
        source_label,
        label,
        copts,
        conlyopts,
        cxxopts,
        copts_filter,
        generate_pic_action,
        generate_no_pic_action,
        additional_compilation_inputs,
        additional_include_scanning_roots,
        output_name,
        outputs):
    if source_type not in [CPP_SOURCE_TYPE_SOURCE, CPP_SOURCE_TYPE_HEADER]:
        fail("Encountered invalid source types when creating CppCompileActionTemplates: " + source_type)
    all_copts = get_copts(
        language = language,
        cpp_configuration = cpp_configuration,
        source_file = source_dir,
        conlyopts = conlyopts,
        copts = copts,
        cxxopts = cxxopts,
        label = source_label,
    )

    # Currently we do not generate minimized bitcode files for tree artifacts because of issues
    # with the indexing step.
    # If lto_index_tree_artifact is set to a tree artifact, the minimized bitcode files will be
    # properly generated and will be an input to the indexing step. However, the lto indexing step
    # fails. The indexing step finds the full bitcode file by replacing the suffix of the
    # minimized bitcode file, therefore they have to be in the same directory.
    # Since the files are in the same directory, the command line artifact expander expands the
    # tree artifact to both the minimized bitcode files and the full bitcode files, causing an
    # error that functions are defined twice.
    # TODO(b/289071777): support for minimized bitcode files.
    lto_index_tree_artifact = None

    if source_type == CPP_SOURCE_TYPE_HEADER:
        _create_compile_action_template(
            action_construction_context = action_construction_context,
            cc_compilation_context = cc_compilation_context,
            cc_toolchain = cc_toolchain,
            configuration = configuration,
            feature_configuration = feature_configuration,
            copts_filter = copts_filter,
            common_compile_build_variables = common_compile_build_variables,
            source_dir = source_dir,
            label = label,
            all_copts = all_copts,
            additional_compilation_inputs = additional_compilation_inputs,
            additional_include_scanning_roots = additional_include_scanning_roots,
            output_name = output_name,
            outputs = outputs,
            lto_output_enabled = False,
            use_pic = generate_pic_action,
            output_categories = [artifact_category.GENERATED_HEADER, artifact_category.PROCESSED_HEADER],
            outputs_key = "header_tokens",
            lto_index_tree_artifact = lto_index_tree_artifact,
            language = language,
        )
    else:  # CPP_SOURCE_TYPE_SOURCE
        lto_output_enabled = feature_configuration.is_enabled("thin_lto")
        if generate_no_pic_action:
            _create_compile_action_template(
                action_construction_context = action_construction_context,
                cc_compilation_context = cc_compilation_context,
                cc_toolchain = cc_toolchain,
                configuration = configuration,
                feature_configuration = feature_configuration,
                copts_filter = copts_filter,
                common_compile_build_variables = common_compile_build_variables,
                source_dir = source_dir,
                label = label,
                all_copts = all_copts,
                additional_compilation_inputs = additional_compilation_inputs,
                additional_include_scanning_roots = additional_include_scanning_roots,
                output_name = output_name,
                outputs = outputs,
                lto_output_enabled = lto_output_enabled,
                use_pic = False,
                output_categories = [artifact_category.OBJECT_FILE],
                outputs_key = "objects",
                lto_index_tree_artifact = lto_index_tree_artifact,
                language = language,
            )
        if generate_pic_action:
            _create_compile_action_template(
                action_construction_context = action_construction_context,
                cc_compilation_context = cc_compilation_context,
                cc_toolchain = cc_toolchain,
                configuration = configuration,
                feature_configuration = feature_configuration,
                copts_filter = copts_filter,
                common_compile_build_variables = common_compile_build_variables,
                source_dir = source_dir,
                label = label,
                all_copts = all_copts,
                additional_compilation_inputs = additional_compilation_inputs,
                additional_include_scanning_roots = additional_include_scanning_roots,
                output_name = output_name,
                outputs = outputs,
                lto_output_enabled = lto_output_enabled,
                use_pic = True,
                output_categories = [artifact_category.PIC_OBJECT_FILE],
                outputs_key = "pic_objects",
                lto_index_tree_artifact = lto_index_tree_artifact,
                language = language,
            )

def _create_compile_action_template(
        *,
        action_construction_context,
        cc_compilation_context,
        cc_toolchain,
        configuration,
        feature_configuration,
        copts_filter,
        common_compile_build_variables,
        source_dir,
        label,
        all_copts,
        additional_compilation_inputs,
        additional_include_scanning_roots,
        output_name,
        outputs,
        lto_output_enabled,
        use_pic,
        output_categories,
        outputs_key,
        lto_index_tree_artifact,
        language):
    output_dir = _declare_compile_output_tree_artifact(
        action_construction_context,
        label,
        output_name,
        use_pic = use_pic,
    )
    specific_compile_build_variables = get_specific_compile_build_variables(
        feature_configuration,
        use_pic = use_pic,
        source_file = source_dir,
        output_file = output_dir,
        cpp_module_map = cc_compilation_context._module_map,
        direct_module_maps = cc_compilation_context._direct_module_maps,
        user_compile_flags = all_copts,
    )
    dotd_tree_artifact = _maybe_declare_dotd_tree_artifact(
        action_construction_context,
        configuration,
        feature_configuration,
        language,
        label,
        output_name,
        use_pic = use_pic,
    )
    diagnostics_tree_artifact = _maybe_declare_diagnostics_tree_artifact(
        action_construction_context,
        feature_configuration,
        label,
        output_name,
        use_pic = use_pic,
    )
    if lto_output_enabled:
        outputs["lto_compilation_context"][output_dir] = (lto_index_tree_artifact, all_copts)
    _cc_internal.create_cc_compile_action_template(
        action_construction_context = action_construction_context,
        cc_compilation_context = cc_compilation_context,
        cc_toolchain = cc_toolchain,
        configuration = configuration,
        feature_configuration = feature_configuration,
        copts_filter = copts_filter,
        compile_build_variables = _cc_internal.combine_cc_toolchain_variables(
            common_compile_build_variables,
            specific_compile_build_variables,
        ),
        source = source_dir,
        additional_compilation_inputs = additional_compilation_inputs,
        additional_include_scanning_roots = additional_include_scanning_roots,
        use_pic = use_pic,
        output_categories = output_categories,
        output_files = output_dir,
        dotd_tree_artifact = dotd_tree_artifact,
        diagnostics_tree_artifact = diagnostics_tree_artifact,
        lto_indexing_tree_artifact = lto_index_tree_artifact,
        needs_include_validation = cc_semantics.needs_include_validation(language),
        toolchain_type = cc_semantics.toolchain,
    )
    outputs[outputs_key].append(output_dir)

def _declare_compile_output_tree_artifact(
        ctx,
        label,
        output_name,
        use_pic):
    return ctx.actions.declare_directory(paths.join(
        "_pic_objs" if use_pic else "_objs",
        label.name,
        output_name,
    ))

def _maybe_declare_dotd_tree_artifact(
        ctx,
        configuration,
        feature_configuration,
        language,
        label,
        output_name,
        use_pic):
    if not dotd_files_enabled(language, ctx.fragments.cpp, feature_configuration):
        return None
    return ctx.actions.declare_directory(paths.join(
        "_pic_dotd" if use_pic else "_dotd",
        label.name,
        output_name,
    ))

def _maybe_declare_diagnostics_tree_artifact(
        ctx,
        feature_configuration,
        label,
        output_name,
        use_pic):
    if not serialized_diagnostics_file_enabled(feature_configuration):
        return None
    return ctx.actions.declare_directory(paths.join(
        "_pic_dia" if use_pic else "_dia",
        label.name,
        output_name,
    ))
