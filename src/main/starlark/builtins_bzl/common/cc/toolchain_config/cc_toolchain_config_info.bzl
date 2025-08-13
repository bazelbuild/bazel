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

"""Information describing C++ toolchain derived from CROSSTOOL file."""

_cc_common_internal = _builtins.internal.cc_common

# buildifier: disable=function-docstring
def create_cc_toolchain_config_info(
        *,
        ctx,
        toolchain_identifier,
        compiler,
        features = [],
        action_configs = [],
        artifact_name_patterns = [],
        cxx_builtin_include_directories = [],
        host_system_name = None,
        target_system_name = None,
        target_cpu = None,
        target_libc = None,
        abi_version = None,
        abi_libc_version = None,
        tool_paths = [],
        make_variables = [],
        builtin_sysroot = None,
        # buildifier: disable=unused-variable
        cc_target_os = None):
    return _cc_common_internal.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = toolchain_identifier,
        target_system_name = target_system_name,
        target_cpu = target_cpu,
        target_libc = target_libc,
        compiler = compiler,
        features = features,
        action_configs = action_configs,
        artifact_name_patterns = artifact_name_patterns,
        cxx_builtin_include_directories = cxx_builtin_include_directories,
        host_system_name = host_system_name,
        abi_version = abi_version,
        abi_libc_version = abi_libc_version,
        tool_paths = tool_paths,
        make_variables = make_variables,
        builtin_sysroot = builtin_sysroot,
    )
