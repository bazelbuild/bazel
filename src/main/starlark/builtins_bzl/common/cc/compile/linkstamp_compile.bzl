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

cc_common_internal = _builtins.internal.cc_common

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
    compile_build_variables = cc_common_internal.get_linkstamp_compile_variables(
        actions = actions,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        source_file = source_file,
        output_file = output_file,
        label_replacement = label_replacement,
        output_replacement = output_replacement,
        needs_pic = needs_pic,
        stamping = stamping,
        additional_linkstamp_defines = additional_linkstamp_defines,
    )
    cc_common_internal.register_linkstamp_compile_action_internal(
        actions = actions,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        source_file = source_file,
        output_file = output_file,
        compilation_inputs = compilation_inputs,
        inputs_for_validation = inputs_for_validation,
        label_replacement = label_replacement,
        output_replacement = output_replacement,
        needs_pic = needs_pic,
        stamping = stamping,
        compile_build_variables = compile_build_variables,
    )
