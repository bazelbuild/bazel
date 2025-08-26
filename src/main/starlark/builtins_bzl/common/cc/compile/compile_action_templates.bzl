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
    "artifact_category",
)
load(":common/cc/compile/cc_compilation_helper.bzl", "dotd_files_enabled")
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
    if cpp_source.type == CPP_SOURCE_TYPE_HEADER:
        header_token_file = _declare_compile_output_tree_artifact(
            action_construction_context,
            label,
            output_name,
            generate_pic_action,
        )
        cpp_compile_action_builder = _cc_internal.create_cpp_compile_action_builder(
            action_construction_context = action_construction_context,
            cc_compilation_context = cc_compilation_context,
            cc_toolchain = cc_toolchain,
            configuration = configuration,
            copts_filter = copts_filter,
            feature_configuration = feature_configuration,
            semantics = native_cc_semantics,
            source_artifact = source_artifact,
            additional_compilation_inputs = additional_compilation_inputs,
            additional_include_scanning_roots = additional_include_scanning_roots,
            use_pic = generate_pic_action,
            output_file = header_token_file,
        )
        specific_compile_build_variables = get_specific_compile_build_variables(
            feature_configuration,
            use_pic = generate_pic_action,
            source_file = source_artifact,
            output_file = header_token_file,
            cpp_module_map = cc_compilation_context.module_map(),
            direct_module_maps = cc_compilation_context.direct_module_maps,
            user_compile_flags = get_copts(
                language = language,
                cpp_configuration = cpp_configuration,
                source_file = source_artifact,
                conlyopts = conlyopts,
                copts = copts,
                cxxopts = cxxopts,
                label = cpp_source.label,
            ),
        )
        dotd_tree_artifact = _maybe_declare_dotd_tree_artifact(
            action_construction_context,
            configuration,
            feature_configuration,
            native_cc_semantics,
            label,
            output_name,
            generate_pic_action,
        )
        _cc_internal.create_compile_action_template(
            action_construction_context = action_construction_context,
            cc_toolchain = cc_toolchain,
            configuration = configuration,
            conlyopts = conlyopts,
            copts = copts,
            cpp_configuration = cpp_configuration,
            cxxopts = cxxopts,
            feature_configuration = feature_configuration,
            label = label,
            compile_build_variables = _cc_internal.combine_cc_toolchain_variables(
                common_compile_build_variables,
                specific_compile_build_variables,
            ),
            cpp_semantics = native_cc_semantics,
            source = cpp_source,
            output_name = output_name,
            cpp_compile_action_builder = cpp_compile_action_builder,
            outputs = outputs,
            output_categories = [artifact_category.GENERATED_HEADER, artifact_category.PROCESSED_HEADER],
            use_pic = generate_pic_action,
            bitcode_output = bitcode_output,
            output_files = header_token_file,
            dotd_tree_artifact = dotd_tree_artifact,
        )
        outputs.add_header_token_file(header_token_file)
    else:  # CPP_SOURCE_TYPE_SOURCE
        if generate_no_pic_action:
            object_file = _declare_compile_output_tree_artifact(
                action_construction_context,
                label,
                output_name,
                generate_pic_action = False,
            )
            cpp_compile_action_builder = _cc_internal.create_cpp_compile_action_builder(
                action_construction_context = action_construction_context,
                cc_compilation_context = cc_compilation_context,
                cc_toolchain = cc_toolchain,
                configuration = configuration,
                copts_filter = copts_filter,
                feature_configuration = feature_configuration,
                semantics = native_cc_semantics,
                source_artifact = source_artifact,
                additional_compilation_inputs = additional_compilation_inputs,
                additional_include_scanning_roots = additional_include_scanning_roots,
                use_pic = False,
                output_file = object_file,
            )
            specific_compile_build_variables = get_specific_compile_build_variables(
                feature_configuration,
                use_pic = False,
                source_file = source_artifact,
                output_file = object_file,
                cpp_module_map = cc_compilation_context.module_map(),
                direct_module_maps = cc_compilation_context.direct_module_maps,
                user_compile_flags = get_copts(
                    language = language,
                    cpp_configuration = cpp_configuration,
                    source_file = source_artifact,
                    conlyopts = conlyopts,
                    copts = copts,
                    cxxopts = cxxopts,
                    label = cpp_source.label,
                ),
            )
            dotd_tree_artifact = _maybe_declare_dotd_tree_artifact(
                action_construction_context,
                configuration,
                feature_configuration,
                native_cc_semantics,
                label,
                output_name,
                generate_pic_action = False,
            )
            _cc_internal.create_compile_action_template(
                action_construction_context = action_construction_context,
                cc_toolchain = cc_toolchain,
                configuration = configuration,
                conlyopts = conlyopts,
                copts = copts,
                cpp_configuration = cpp_configuration,
                cxxopts = cxxopts,
                feature_configuration = feature_configuration,
                label = label,
                compile_build_variables = _cc_internal.combine_cc_toolchain_variables(
                    common_compile_build_variables,
                    specific_compile_build_variables,
                ),
                cpp_semantics = native_cc_semantics,
                source = cpp_source,
                output_name = output_name,
                cpp_compile_action_builder = cpp_compile_action_builder,
                outputs = outputs,
                output_categories = [artifact_category.OBJECT_FILE],
                use_pic = False,
                bitcode_output = feature_configuration.is_enabled("thin_lto"),
                output_files = object_file,
                dotd_tree_artifact = dotd_tree_artifact,
            )
            outputs.add_object_file(object_file)
        if generate_pic_action:
            pic_object_file = _declare_compile_output_tree_artifact(
                action_construction_context,
                label,
                output_name,
                generate_pic_action = True,
            )
            cpp_compile_action_builder = _cc_internal.create_cpp_compile_action_builder(
                action_construction_context = action_construction_context,
                cc_compilation_context = cc_compilation_context,
                cc_toolchain = cc_toolchain,
                configuration = configuration,
                copts_filter = copts_filter,
                feature_configuration = feature_configuration,
                semantics = native_cc_semantics,
                source_artifact = source_artifact,
                additional_compilation_inputs = additional_compilation_inputs,
                additional_include_scanning_roots = additional_include_scanning_roots,
                use_pic = True,
                output_file = pic_object_file,
            )
            specific_compile_build_variables = get_specific_compile_build_variables(
                feature_configuration,
                use_pic = generate_pic_action,
                source_file = source_artifact,
                output_file = pic_object_file,
                cpp_module_map = cc_compilation_context.module_map(),
                direct_module_maps = cc_compilation_context.direct_module_maps,
                user_compile_flags = get_copts(
                    language = language,
                    cpp_configuration = cpp_configuration,
                    source_file = source_artifact,
                    conlyopts = conlyopts,
                    copts = copts,
                    cxxopts = cxxopts,
                    label = cpp_source.label,
                ),
            )
            dotd_tree_artifact = _maybe_declare_dotd_tree_artifact(
                action_construction_context,
                configuration,
                feature_configuration,
                native_cc_semantics,
                label,
                output_name,
                generate_pic_action = True,
            )
            _cc_internal.create_compile_action_template(
                action_construction_context = action_construction_context,
                cc_toolchain = cc_toolchain,
                configuration = configuration,
                conlyopts = conlyopts,
                copts = copts,
                cpp_configuration = cpp_configuration,
                cxxopts = cxxopts,
                feature_configuration = feature_configuration,
                label = label,
                compile_build_variables = _cc_internal.combine_cc_toolchain_variables(
                    common_compile_build_variables,
                    specific_compile_build_variables,
                ),
                cpp_semantics = native_cc_semantics,
                source = cpp_source,
                output_name = output_name,
                cpp_compile_action_builder = cpp_compile_action_builder,
                outputs = outputs,
                output_categories = [artifact_category.PIC_OBJECT_FILE],
                use_pic = True,
                bitcode_output = feature_configuration.is_enabled("thin_lto"),
                output_files = pic_object_file,
                dotd_tree_artifact = dotd_tree_artifact,
            )
            outputs.add_pic_object_file(pic_object_file)

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
        native_cc_semantics,
        label,
        output_name,
        generate_pic_action):
    if not dotd_files_enabled(native_cc_semantics, configuration, feature_configuration):
        return None
    return ctx.actions.declare_directory(paths.join(
        "_pic_dotd" if generate_pic_action else "_dotd",
        label.name,
        output_name,
    ))
