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
        common_compile_build_variables,
        fdo_build_variables,
        cpp_source,
        source_artifact,
        label,
        copts,
        conlyopts,
        cxxopts,
        copts_filter,
        fdo_context,
        auxiliary_fdo_inputs,
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
        )
        header_token_file = _cc_internal.create_compile_action_template(
            action_construction_context = action_construction_context,
            cc_compilation_context = cc_compilation_context,
            cc_toolchain = cc_toolchain,
            configuration = configuration,
            conlyopts = conlyopts,
            copts = copts,
            cpp_configuration = cpp_configuration,
            cxxopts = cxxopts,
            fdo_context = fdo_context,
            auxiliary_fdo_inputs = auxiliary_fdo_inputs,
            feature_configuration = feature_configuration,
            label = label,
            common_compile_build_variables = common_compile_build_variables,
            fdo_build_variables = fdo_build_variables,
            cpp_semantics = native_cc_semantics,
            source = cpp_source,
            output_name = output_name,
            cpp_compile_action_builder = cpp_compile_action_builder,
            outputs = outputs,
            output_categories = [artifact_category.GENERATED_HEADER, artifact_category.PROCESSED_HEADER],
            use_pic = generate_pic_action,
            bitcode_output = bitcode_output,
        )
        outputs.add_header_token_file(header_token_file)
    else:  # CPP_SOURCE_TYPE_SOURCE
        if generate_no_pic_action:
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
            )
            object_file = _cc_internal.create_compile_action_template(
                action_construction_context = action_construction_context,
                cc_compilation_context = cc_compilation_context,
                cc_toolchain = cc_toolchain,
                configuration = configuration,
                conlyopts = conlyopts,
                copts = copts,
                cpp_configuration = cpp_configuration,
                cxxopts = cxxopts,
                fdo_context = fdo_context,
                auxiliary_fdo_inputs = auxiliary_fdo_inputs,
                feature_configuration = feature_configuration,
                label = label,
                common_compile_build_variables = common_compile_build_variables,
                fdo_build_variables = fdo_build_variables,
                cpp_semantics = native_cc_semantics,
                source = cpp_source,
                output_name = output_name,
                cpp_compile_action_builder = cpp_compile_action_builder,
                outputs = outputs,
                output_categories = [artifact_category.OBJECT_FILE],
                use_pic = False,
                bitcode_output = feature_configuration.is_enabled("thin_lto"),
            )
            outputs.add_object_file(object_file)
        if generate_pic_action:
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
            )
            pic_object_file = _cc_internal.create_compile_action_template(
                action_construction_context = action_construction_context,
                cc_compilation_context = cc_compilation_context,
                cc_toolchain = cc_toolchain,
                configuration = configuration,
                conlyopts = conlyopts,
                copts = copts,
                cpp_configuration = cpp_configuration,
                cxxopts = cxxopts,
                fdo_context = fdo_context,
                auxiliary_fdo_inputs = auxiliary_fdo_inputs,
                feature_configuration = feature_configuration,
                label = label,
                common_compile_build_variables = common_compile_build_variables,
                fdo_build_variables = fdo_build_variables,
                cpp_semantics = native_cc_semantics,
                source = cpp_source,
                output_name = output_name,
                cpp_compile_action_builder = cpp_compile_action_builder,
                outputs = outputs,
                output_categories = [artifact_category.PIC_OBJECT_FILE],
                use_pic = True,
                bitcode_output = feature_configuration.is_enabled("thin_lto"),
            )
            outputs.add_pic_object_file(pic_object_file)
