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

"""Utility functions for C++ rules."""

CcInfo = _builtins.toplevel.CcInfo
cc_common = _builtins.toplevel.cc_common

def _merge_cc_debug_contexts(compilation_outputs, dep_cc_infos):
    debug_context = cc_common.create_debug_context(compilation_outputs)
    debug_contexts = [debug_context]
    for dep_cc_info in dep_cc_infos:
        debug_contexts.append(dep_cc_info.debug_context())

    return cc_common.merge_debug_context(debug_contexts)

def _is_code_coverage_enabled(ctx):
    if ctx.coverage_instrumented():
        return True
    for dep in ctx.attr.deps:
        if CcInfo in dep:
            if ctx.coverage_instrumented(dep):
                return True
    return False

def _get_dynamic_libraries_for_runtime(cc_linking_context, linking_statically):
    libraries = []
    for linker_input in cc_linking_context.linker_inputs.to_list():
        libraries.extend(linker_input.libraries)

    dynamic_libraries_for_runtime = []
    for library in libraries:
        artifact = _get_dynamic_library_for_runtime_or_none(library, linking_statically)
        if artifact != None:
            dynamic_libraries_for_runtime.append(artifact)

    return dynamic_libraries_for_runtime

def _get_dynamic_library_for_runtime_or_none(library, linking_statically):
    if library.dynamic_library == None:
        return None

    if linking_statically and (library.static_library != None or library.pic_static_library != None):
        return None

    return library.dynamic_library

def _find_cpp_toolchain(ctx):
    """
    Finds the c++ toolchain.

    If the c++ toolchain is in use, returns it.  Otherwise, returns a c++
    toolchain derived from legacy toolchain selection.

    Args:
      ctx: The rule context for which to find a toolchain.

    Returns:
      A CcToolchainProvider.
    """

    # Check the incompatible flag for toolchain resolution.
    if hasattr(cc_common, "is_cc_toolchain_resolution_enabled_do_not_use") and cc_common.is_cc_toolchain_resolution_enabled_do_not_use(ctx = ctx):
        if not "@//tools/cpp:toolchain_type" in ctx.toolchains:
            fail("In order to use find_cpp_toolchain, you must include the '//tools/cpp:toolchain_type' in the toolchains argument to your rule.")
        toolchain_info = ctx.toolchains["@//tools/cpp:toolchain_type"]
        if hasattr(toolchain_info, "cc_provider_in_toolchain") and hasattr(toolchain_info, "cc"):
            return toolchain_info.cc
        return toolchain_info

    # Otherwise, fall back to the legacy attribute.
    if hasattr(ctx.attr, "_cc_toolchain"):
        return ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]

    # We didn't find anything.
    fail("In order to use find_cpp_toolchain, you must define the '_cc_toolchain' attribute on your rule or aspect.")

def _build_output_groups_for_emitting_compile_providers(
        compilation_outputs,
        compilation_context,
        cpp_configuration,
        cc_toolchain,
        feature_configuration,
        ctx,
        generate_hidden_top_level_group):
    output_groups_builder = {}
    process_hdrs = cpp_configuration.process_headers_in_dependencies()
    use_pic = cc_toolchain.needs_pic_for_dynamic_libraries(feature_configuration = feature_configuration)
    output_groups_builder["temp_files_INTERNAL_"] = compilation_outputs.temps()
    files_to_compile = compilation_outputs.files_to_compile(
        parse_headers = process_hdrs,
        use_pic = use_pic,
    )
    output_groups_builder["compilation_outputs"] = files_to_compile
    output_groups_builder["compilation_prerequisites_INTERNAL_"] = _collect_compilation_prerequisites(ctx, compilation_context)

    if generate_hidden_top_level_group:
        output_groups_builder["_hidden_top_level_INTERNAL_"] = _collect_library_hidden_top_level_artifacts(
            ctx,
            files_to_compile,
        )

    _create_save_feature_state_artifacts(
        output_groups_builder,
        cpp_configuration,
        feature_configuration,
        ctx,
    )

    return output_groups_builder

CC_SOURCE = [".cc", ".cpp", ".cxx", ".c++", ".C", ".cu", ".cl"]
C_SOURCE = [".c"]
OBJC_SOURCE = [".m"]
OBJCPP_SOURCE = [".mm"]
CLIF_INPUT_PROTO = [".ipb"]
CLIF_OUTPUT_PROTO = [".opb"]
CC_AND_OBJC_EXTENSIONS = []
CC_AND_OBJC_EXTENSIONS.extend(CC_SOURCE)
CC_AND_OBJC_EXTENSIONS.extend(C_SOURCE)
CC_AND_OBJC_EXTENSIONS.extend(OBJC_SOURCE)
CC_AND_OBJC_EXTENSIONS.extend(OBJCPP_SOURCE)
CC_AND_OBJC_EXTENSIONS.extend(CLIF_INPUT_PROTO)
CC_AND_OBJC_EXTENSIONS.extend(CLIF_OUTPUT_PROTO)

def _collect_compilation_prerequisites(ctx, compilation_context):
    # This doesn't go through the same code path as the native one. The native
    # method accesses other fields in compilation_context. So far this has not
    # been a problem for any migrated rule.
    direct = []
    for src in ctx.files.srcs:
        if src.extension in CC_AND_OBJC_EXTENSIONS:
            direct.append(src)
    return depset(direct = direct, transitive = [compilation_context.transitive_compilation_prerequisites()])

def _collect_header_tokens(
        ctx,
        cpp_configuration,
        compilation_outputs,
        process_hdrs,
        add_self_tokens):
    header_tokens_transitive = []
    for dep in ctx.attr.deps:
        if "_hidden_header_tokens_INTERNAL_" in dep[OutputGroupInfo]:
            header_tokens_transitive.append(dep[OutputGroupInfo]["_hidden_header_tokens_INTERNAL_"])
        else:
            header_tokens_transitive.append(depset([]))

    header_tokens_direct = []
    if add_self_tokens and process_hdrs:
        header_tokens_direct.extend(compilation_outputs.header_tokens())

    return depset(direct = header_tokens_direct, transitive = header_tokens_transitive)

def _collect_library_hidden_top_level_artifacts(
        ctx,
        files_to_compile):
    artifacts_to_force_builder = [files_to_compile]
    for dep in ctx.attr.deps:
        artifacts_to_force_builder.append(dep[OutputGroupInfo]["_hidden_top_level_INTERNAL_"])

    return depset(transitive = artifacts_to_force_builder)

def _create_save_feature_state_artifacts(
        output_groups_builder,
        cpp_configuration,
        feature_configuration,
        ctx):
    if cpp_configuration.save_feature_state():
        feature_state_file = ctx.actions.declare_file(ctx.label.name + "_feature_state.txt")

        ctx.actions.write(feature_state_file, str(feature_configuration))
        output_groups_builder["default"] = depset(direct = [feature_state_file])

def _merge_output_groups(output_groups):
    merged_output_groups_builder = {}
    for output_group in output_groups:
        for output_key, output_value in output_group.items():
            depset_list = merged_output_groups_builder.get(output_key, [])
            depset_list.append(output_value)
            merged_output_groups_builder[output_key] = depset_list

    merged_output_group = {}
    for k, v in merged_output_groups_builder.items():
        merged_output_group[k] = depset(transitive = v)

    return merged_output_group

def _rule_error(msg):
    fail(msg)

def _attribute_error(attr_name, msg):
    fail("in attribute '" + attr_name + "': " + msg)

def _get_linking_contexts_from_deps(deps):
    linking_contexts = []
    for dep in deps:
        if CcInfo in dep:
            linking_contexts.append(dep[CcInfo].linking_context)
    return linking_contexts

def _is_test_target(ctx):
    if hasattr(ctx.attr, "testonly"):
        return ctx.attr.testonly

    return False

cc_helper = struct(
    merge_cc_debug_contexts = _merge_cc_debug_contexts,
    is_code_coverage_enabled = _is_code_coverage_enabled,
    get_dynamic_libraries_for_runtime = _get_dynamic_libraries_for_runtime,
    get_dynamic_library_for_runtime_or_none = _get_dynamic_library_for_runtime_or_none,
    find_cpp_toolchain = _find_cpp_toolchain,
    build_output_groups_for_emitting_compile_providers = _build_output_groups_for_emitting_compile_providers,
    merge_output_groups = _merge_output_groups,
    rule_error = _rule_error,
    attribute_error = _attribute_error,
    get_linking_contexts_from_deps = _get_linking_contexts_from_deps,
    is_test_target = _is_test_target,
)
