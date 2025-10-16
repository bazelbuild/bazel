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
"""
The cc_common.register_linkstamp_compile_action function.

Used for C++ linkstamp compiling.
"""

load(":common/cc/action_names.bzl", "LINKSTAMP_COMPILE_ACTION_NAME")
load(
    ":common/cc/cc_helper_internal.bzl",
    "is_stamping_enabled",
)
load(
    ":common/cc/compile/compile_build_variables.bzl",
    "get_linkstamp_compile_variables",
)
load(":common/cc/semantics.bzl", cc_semantics = "semantics")

_cc_common_internal = _builtins.internal.cc_common
_cc_internal = _builtins.internal.cc_internal

def register_linkstamp_compile_action(
        *,
        actions,
        cc_toolchain,
        feature_configuration,
        source_file,
        output_file,
        compilation_inputs,
        inputs_for_validation,
        label_replacement,
        output_replacement,
        needs_pic = False,
        stamping = None,
        additional_linkstamp_defines = None):
    """Registers a C++ compile action for linkstamps.

    Args:
        actions: `actions` object.
        cc_toolchain: `CcToolchainInfo` provider to be used.
        feature_configuration: `feature_configuration` to be queried.
        source_file: The linkstamp source file to be compiled.
        output_file: The output object file.
        compilation_inputs: A depset of artifacts used for compilation.
        inputs_for_validation: A depset of artifacts to be used as inputs for invalidation for the action.
        label_replacement: String to replace ${LABEL} in linkstamp defines.
        output_replacement: String to replace ${OUTPUT_PATH} in linkstamp defines.
        needs_pic: Whether PIC compilation is needed.
        stamping: Whether stamping is enabled. If None, it's computed based on configuration.
        additional_linkstamp_defines: A list of additional defines for linkstamp compilation.
    """
    ctx = _cc_internal.actions2ctx_cheat(actions)

    if stamping == None:
        stamping_tri_state = is_stamping_enabled(ctx)
        stamping = False if ctx.configuration.is_tool_configuration() else (
            stamping_tri_state == 1 or (stamping_tri_state == -1 and ctx.configuration.stamp_binaries())
        )

    output_group_info = cc_toolchain._build_info_files
    if stamping:
        build_info_files = output_group_info.non_redacted_build_info_files.to_list()
    else:
        build_info_files = output_group_info.redacted_build_info_files.to_list()

    compile_build_variables = get_linkstamp_compile_variables(
        source_file = source_file,
        output_file = output_file,
        label_replacement = label_replacement,
        output_replacement = output_replacement,
        additional_linkstamp_defines = additional_linkstamp_defines,
        build_info_header_artifacts = build_info_files,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        needs_pic = needs_pic,
    )

    if stamping:
        # Makes the target depend on BUILD_INFO_KEY, which helps to discover stamped targets
        # See b/326620485 for more details.
        unused = ctx.version_file  # @unused

    # TODO(b/447325425): Consider if inputs_for_validation could (and should?) be passed in via
    # cc_compilation_context instead of via cache_key_inputs - a param that is used only here.
    _cc_internal.create_cc_compile_action(
        action_construction_context = ctx,
        cc_compilation_context = _cc_internal.empty_compilation_context(),
        cc_toolchain = cc_toolchain,
        configuration = ctx.configuration,
        copts_filter = _cc_internal.create_copts_filter(),
        feature_configuration = feature_configuration,
        source = source_file,
        additional_compilation_inputs_set = compilation_inputs,
        output_file = output_file,
        use_pic = needs_pic,
        compile_build_variables = compile_build_variables,
        cache_key_inputs = inputs_for_validation,
        build_info_header_files = build_info_files,
        action_name = LINKSTAMP_COMPILE_ACTION_NAME,
        should_scan_includes = False,
        shareable = True,
        needs_include_validation = cc_semantics.needs_include_validation(language = "c++"),
        toolchain_type = cc_semantics.toolchain,
    )
