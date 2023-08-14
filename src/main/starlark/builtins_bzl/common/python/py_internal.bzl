# Copyright 2022 The Bazel Authors. All rights reserved.
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
"""PYTHON RULE IMPLEMENTATION ONLY: Do not use outside of the rule implementations and their tests.

Various builtin Starlark defined objects exposed for non-builtin Starlark.

These may change at any time and are closely coupled to the rule implementation.
"""

_py_builtins = _builtins.internal.py_builtins

def _is_available_for(package_group_target, label):
    return package_group_target.isAvailableFor(label)

# This replaces the Java-defined name using exports.bzl toplevels mapping.
py_internal = struct(
    add_py_extra_pseudo_action = _py_builtins.add_py_extra_pseudo_action,
    are_action_listeners_enabled = _py_builtins.are_action_listeners_enabled,
    copy_without_caching = _py_builtins.copy_without_caching,
    create_repo_mapping_manifest = _py_builtins.create_repo_mapping_manifest,
    create_sources_only_manifest = _py_builtins.create_sources_only_manifest,
    declare_constant_metadata_file = _py_builtins.declare_constant_metadata_file,
    expand_location_and_make_variables = _py_builtins.expand_location_and_make_variables,
    get_current_os_name = _py_builtins.get_current_os_name,
    get_label_repo_runfiles_path = _py_builtins.get_label_repo_runfiles_path,
    get_legacy_exernal_runfiles = _py_builtins.get_legacy_external_runfiles,
    get_rule_name = _py_builtins.get_rule_name,
    is_available_for = _is_available_for,
    is_bzlmod_enabled = _py_builtins.is_bzlmod_enabled,
    is_singleton_depset = _py_builtins.is_singleton_depset,
    make_runfiles_respect_legacy_external_runfiles = _py_builtins.make_runfiles_respect_legacy_external_runfiles,
    merge_runfiles_with_generated_inits_empty_files_supplier = _py_builtins.merge_runfiles_with_generated_inits_empty_files_supplier,
)
