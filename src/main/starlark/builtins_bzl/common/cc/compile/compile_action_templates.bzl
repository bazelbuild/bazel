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
        native_cc_semantics,
        language,
        common_compile_build_variables,
        cpp_source,
        source_artifact,
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
        outputs,
        bitcode_output):
    if cpp_source.type not in [CPP_SOURCE_TYPE_SOURCE, CPP_SOURCE_TYPE_HEADER]:
        fail("Encountered invalid source types when creating CppCompileActionTemplates: " + cpp_source.type)
    all_copts = get_copts(
        language = language,
        cpp_configuration = cpp_configuration,
        source_file = source_artifact,
        conlyopts = conlyopts,
        copts = copts,
        cxxopts = cxxopts,
        label = cpp_source.label,
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

    if cpp_source.type == CPP_SOURCE_TYPE_HEADER:
        header_token_file = _declare_compile_output_tree_artifact(
            action_construction_context,
            label,
            output_name,
            generate_pic_action,
        )
        specific_compile_build_variables = get_specific_compile_build_variables(
            feature_configuration,
            use_pic = generate_pic_action,
            source_file = source_artifact,
            output_file = header_token_file,
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
            generate_pic_action,
        )
        diagnostics_tree_artifact = _maybe_declare_diagnostics_tree_artifact(
            action_construction_context,
            feature_configuration,
            label,
            output_name,
            generate_pic_action,
        )
        if bitcode_output:
            outputs["lto_compilation_context"][header_token_file] = (lto_index_tree_artifact, all_copts)
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
            cc_semantics = native_cc_semantics,
            source = cpp_source.file,
            additional_compilation_inputs = additional_compilation_inputs,
            additional_include_scanning_roots = additional_include_scanning_roots,
            use_pic = generate_pic_action,
            output_categories = [artifact_category.GENERATED_HEADER, artifact_category.PROCESSED_HEADER],
            output_files = header_token_file,
            dotd_tree_artifact = dotd_tree_artifact,
            diagnostics_tree_artifact = diagnostics_tree_artifact,
            lto_indexing_tree_artifact = lto_index_tree_artifact,
        )
        outputs["header_tokens"].append(header_token_file)
    else:  # CPP_SOURCE_TYPE_SOURCE
        if generate_no_pic_action:
            object_file = _declare_compile_output_tree_artifact(
                action_construction_context,
                label,
                output_name,
                generate_pic_action = False,
            )
            specific_compile_build_variables = get_specific_compile_build_variables(
                feature_configuration,
                use_pic = False,
                source_file = source_artifact,
                output_file = object_file,
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
                generate_pic_action = False,
            )
            diagnostics_tree_artifact = _maybe_declare_diagnostics_tree_artifact(
                action_construction_context,
                feature_configuration,
                label,
                output_name,
                generate_pic_action,
            )
            if feature_configuration.is_enabled("thin_lto"):
                outputs["lto_compilation_context"][object_file] = (lto_index_tree_artifact, all_copts)
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
                cc_semantics = native_cc_semantics,
                source = cpp_source.file,
                additional_compilation_inputs = additional_compilation_inputs,
                additional_include_scanning_roots = additional_include_scanning_roots,
                use_pic = False,
                output_categories = [artifact_category.OBJECT_FILE],
                output_files = object_file,
                dotd_tree_artifact = dotd_tree_artifact,
                diagnostics_tree_artifact = diagnostics_tree_artifact,
                lto_indexing_tree_artifact = lto_index_tree_artifact,
            )
            outputs["objects"].append(object_file)
        if generate_pic_action:
            pic_object_file = _declare_compile_output_tree_artifact(
                action_construction_context,
                label,
                output_name,
                generate_pic_action = True,
            )
            specific_compile_build_variables = get_specific_compile_build_variables(
                feature_configuration,
                use_pic = generate_pic_action,
                source_file = source_artifact,
                output_file = pic_object_file,
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
                generate_pic_action = True,
            )
            diagnostics_tree_artifact = _maybe_declare_diagnostics_tree_artifact(
                action_construction_context,
                feature_configuration,
                label,
                output_name,
                generate_pic_action,
            )
            if feature_configuration.is_enabled("thin_lto"):
                outputs["lto_compilation_context"][pic_object_file] = (lto_index_tree_artifact, all_copts)
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
                cc_semantics = native_cc_semantics,
                source = cpp_source.file,
                additional_compilation_inputs = additional_compilation_inputs,
                additional_include_scanning_roots = additional_include_scanning_roots,
                use_pic = True,
                output_categories = [artifact_category.PIC_OBJECT_FILE],
                output_files = pic_object_file,
                dotd_tree_artifact = dotd_tree_artifact,
                diagnostics_tree_artifact = diagnostics_tree_artifact,
                lto_indexing_tree_artifact = lto_index_tree_artifact,
            )
            outputs["pic_objects"].append(pic_object_file)

def _declare_compile_output_tree_artifact(
        ctx,
        label,
        output_name,
        generate_pic_action):
    return ctx.actions.declare_directory(paths.join(
        "_pic_objs" if generate_pic_action else "_objs",
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
        generate_pic_action):
    if not dotd_files_enabled(language, ctx.fragments.cpp, feature_configuration):
        return None
    return ctx.actions.declare_directory(paths.join(
        "_pic_dotd" if generate_pic_action else "_dotd",
        label.name,
        output_name,
    ))

def _maybe_declare_diagnostics_tree_artifact(
        ctx,
        feature_configuration,
        label,
        output_name,
        generate_pic_action):
    if not serialized_diagnostics_file_enabled(feature_configuration):
        return None
    return ctx.actions.declare_directory(paths.join(
        "_pic_dia" if generate_pic_action else "_dia",
        label.name,
        output_name,
    ))
