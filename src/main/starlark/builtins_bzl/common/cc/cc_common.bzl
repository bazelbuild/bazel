# Copyright 2023 The Bazel Authors. All rights reserved.
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
# LINT.IfChange(forked_exports)
"""Utilities related to C++ support."""

load(":common/cc/cc_helper_internal.bzl", _CREATE_COMPILE_ACTION_API_ALLOWLISTED_PACKAGES = "CREATE_COMPILE_ACTION_API_ALLOWLISTED_PACKAGES", _PRIVATE_STARLARKIFICATION_ALLOWLIST = "PRIVATE_STARLARKIFICATION_ALLOWLIST")

_cc_common_internal = _builtins.internal.cc_common
_cc_internal = _builtins.internal.cc_internal

def _get_tool_for_action(*, feature_configuration, action_name):
    return _cc_common_internal.get_tool_for_action(feature_configuration = feature_configuration, action_name = action_name)

def _get_execution_requirements(*, feature_configuration, action_name):
    return _cc_common_internal.get_execution_requirements(feature_configuration = feature_configuration, action_name = action_name)

def _action_is_enabled(*, feature_configuration, action_name):
    return _cc_common_internal.action_is_enabled(feature_configuration = feature_configuration, action_name = action_name)

def _get_memory_inefficient_command_line(*, feature_configuration, action_name, variables):
    return _cc_common_internal.get_memory_inefficient_command_line(feature_configuration = feature_configuration, action_name = action_name, variables = variables)

def _get_environment_variables(*, feature_configuration, action_name, variables):
    return _cc_common_internal.get_environment_variables(feature_configuration = feature_configuration, action_name = action_name, variables = variables)

def _empty_variables():
    return _cc_common_internal.empty_variables()

def _legacy_cc_flags_make_variable_do_not_use(*, cc_toolchain):
    return _cc_common_internal.legacy_cc_flags_make_variable_do_not_use(cc_toolchain = cc_toolchain)

def _check_experimental_cc_shared_library():
    _cc_internal.check_private_api(allowlist = _PRIVATE_STARLARKIFICATION_ALLOWLIST)
    return _cc_common_internal.check_experimental_cc_shared_library()

def _incompatible_disable_objc_library_transition():
    _cc_internal.check_private_api(allowlist = _PRIVATE_STARLARKIFICATION_ALLOWLIST)
    return _cc_common_internal.incompatible_disable_objc_library_transition()

def _add_go_exec_groups_to_binary_rules():
    _cc_internal.check_private_api(allowlist = _PRIVATE_STARLARKIFICATION_ALLOWLIST)
    return _cc_common_internal.add_go_exec_groups_to_binary_rules()

def _get_tool_requirement_for_action(*, feature_configuration, action_name):
    _cc_internal.check_private_api(allowlist = _PRIVATE_STARLARKIFICATION_ALLOWLIST)
    return _cc_common_internal.get_tool_requirement_for_action(feature_configuration = feature_configuration, action_name = action_name)

def _create_compile_action(
        *,
        actions,
        cc_toolchain,
        feature_configuration,
        source_file,
        output_file,
        variables,
        action_name,
        compilation_context,
        additional_inputs = None,
        additional_outputs = []):
    _cc_internal.check_private_api(allowlist = _CREATE_COMPILE_ACTION_API_ALLOWLISTED_PACKAGES)
    return _cc_common_internal.create_compile_action(
        actions = actions,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        source_file = source_file,
        output_file = output_file,
        variables = variables,
        action_name = action_name,
        compilation_context = compilation_context,
        additional_inputs = additional_inputs,
        additional_outputs = additional_outputs,
    )

def _implementation_deps_allowed_by_allowlist(*, ctx):
    _cc_internal.check_private_api(allowlist = _PRIVATE_STARLARKIFICATION_ALLOWLIST)
    return _cc_common_internal.implementation_deps_allowed_by_allowlist(ctx = ctx)

def _register_swig_action(*args, **kwargs):
    _cc_internal.check_private_api(allowlist = _PRIVATE_STARLARKIFICATION_ALLOWLIST)
    return _cc_common_internal.register_swig_action(*args, **kwargs)

def _internal_exports():
    _cc_internal.check_private_api(allowlist = [
        ("", "third_party/bazel_rules/rules_cc"),
        ("rules_cc", ""),
        ("", "javatests/com/google/devtools/grok/kythe"),
    ])
    return _cc_internal

cc_common = struct(
    internal_DO_NOT_USE = _internal_exports,
    action_is_enabled = _action_is_enabled,
    add_go_exec_groups_to_binary_rules = _add_go_exec_groups_to_binary_rules,
    check_experimental_cc_shared_library = _check_experimental_cc_shared_library,
    create_compile_action = _create_compile_action,
    do_not_use_tools_cpp_compiler_present = _cc_common_internal.do_not_use_tools_cpp_compiler_present,
    empty_variables = _empty_variables,
    get_environment_variables = _get_environment_variables,
    get_execution_requirements = _get_execution_requirements,
    get_memory_inefficient_command_line = _get_memory_inefficient_command_line,
    get_tool_for_action = _get_tool_for_action,
    get_tool_requirement_for_action = _get_tool_requirement_for_action,
    implementation_deps_allowed_by_allowlist = _implementation_deps_allowed_by_allowlist,
    incompatible_disable_objc_library_transition = _incompatible_disable_objc_library_transition,
    legacy_cc_flags_make_variable_do_not_use = _legacy_cc_flags_make_variable_do_not_use,
    register_swig_action = _register_swig_action,
)
