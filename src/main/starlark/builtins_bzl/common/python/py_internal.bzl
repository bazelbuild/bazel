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
PackageSpecificationInfo = _builtins.toplevel.PackageSpecificationInfo

# NOTE: Directly exporting the Java objects on _py_builtins results in some
# stricter serialization check failures within the Google implementation.
# Instead, they must be wrapped in a Starlark function.

def _add_py_extra_pseudo_action(*args, **kwargs):
    return _py_builtins.add_py_extra_pseudo_action(*args, **kwargs)

def _are_action_listeners_enabled(*args, **kwargs):
    return _py_builtins.are_action_listeners_enabled(*args, **kwargs)

def _copy_without_caching(*args, **kwargs):
    return _py_builtins.copy_without_caching(*args, **kwargs)

def _create_repo_mapping_manifest(*args, **kwargs):
    return _py_builtins.create_repo_mapping_manifest(*args, **kwargs)

def _create_sources_only_manifest(*args, **kwargs):
    return _py_builtins.create_sources_only_manifest(*args, **kwargs)

def _declare_constant_metadata_file(*args, **kwargs):
    return _py_builtins.declare_constant_metadata_file(*args, **kwargs)

def _declare_shareable_artifact(ctx, *args, **kwargs):
    return ctx.actions.declare_shareable_artifact(*args, **kwargs)

def _expand_location_and_make_variables(*args, **kwargs):
    return _py_builtins.expand_location_and_make_variables(*args, **kwargs)

def _get_current_os_name(*args, **kwargs):
    return _py_builtins.get_current_os_name(*args, **kwargs)

def _get_label_repo_runfiles_path(*args, **kwargs):
    return _py_builtins.get_label_repo_runfiles_path(*args, **kwargs)

def _get_legacy_external_runfiles(*args, **kwargs):
    return False

def _get_rule_name(*args, **kwargs):
    return _py_builtins.get_rule_name(*args, **kwargs)

def _is_available_for(package_group_target, label):
    return package_group_target[PackageSpecificationInfo].contains(label)

def _is_bzlmod_enabled(*args, **kwargs):
    return _py_builtins.is_bzlmod_enabled(*args, **kwargs)

def _is_singleton_depset(*args, **kwargs):
    return _py_builtins.is_singleton_depset(*args, **kwargs)

def _is_tool_configuration(ctx):
    return ctx.configuration.is_tool_configuration()

def _regex_match(*args, **kwargs):
    return _py_builtins.regex_match(*args, **kwargs)

def _runfiles_enabled(ctx):
    return ctx.configuration.runfiles_enabled()

def _linkstamp_file(linkstamp, *args, **kwargs):
    return linkstamp.file(*args, **kwargs)

# TODO(https://github.com/bazelbuild/bazel/issues/17415): Cleanup uses and remove
def _make_runfiles_respect_legacy_external_runfiles(_ctx, runfiles):
    return runfiles

def _merge_runfiles_with_generated_inits_empty_files_supplier(*args, **kwargs):
    return _py_builtins.merge_runfiles_with_generated_inits_empty_files_supplier(*args, **kwargs)

def _share_native_deps(ctx):
    return ctx.fragments.cpp.share_native_deps()

def _stamp_binaries(ctx):
    return ctx.configuration.stamp_binaries()

def _strip_opts(ctx):
    return ctx.fragments.cpp.strip_opts()

_SHARED_LIBRARY_EXTENSIONS = ["so", "dll", "dylib", "pyd", "wasm"]

def _is_valid_shared_library_artifact(shared_library):
    if (shared_library.extension in _SHARED_LIBRARY_EXTENSIONS):
        return True

    return _is_versioned_shared_library_extension_valid(shared_library.basename)

def _is_versioned_shared_library_extension_valid(shared_library_name):
    """Validates the name against the regex "^.+\\.((so)|(dylib))(\\.\\d\\w*)+$",

    Args:
        shared_library_name: (str) the name to validate

    Returns:
        (bool)
    """

    # must match VERSIONED_SHARED_LIBRARY.
    for ext in (".so.", ".dylib."):
        name, _, version = shared_library_name.rpartition(ext)
        if name and version:
            version_parts = version.split(".")
            for part in version_parts:
                if not part[0].isdigit():
                    return False
                for c in part[1:].elems():
                    if not (c.isalnum() or c == "_"):
                        return False
            return True
    return False

# This replaces the Java-defined name using exports.bzl toplevels mapping.
py_internal = struct(
    PackageSpecificationInfo = PackageSpecificationInfo,
    add_py_extra_pseudo_action = _add_py_extra_pseudo_action,
    are_action_listeners_enabled = _are_action_listeners_enabled,
    cc_helper = struct(
        is_valid_shared_library_artifact = _is_valid_shared_library_artifact,
    ),
    copy_without_caching = _copy_without_caching,
    create_repo_mapping_manifest = _create_repo_mapping_manifest,
    create_sources_only_manifest = _create_sources_only_manifest,
    declare_constant_metadata_file = _declare_constant_metadata_file,
    declare_shareable_artifact = _declare_shareable_artifact,
    expand_location_and_make_variables = _expand_location_and_make_variables,
    get_current_os_name = _get_current_os_name,
    get_label_repo_runfiles_path = _get_label_repo_runfiles_path,
    get_legacy_external_runfiles = _get_legacy_external_runfiles,
    get_rule_name = _get_rule_name,
    is_available_for = _is_available_for,
    is_bzlmod_enabled = _is_bzlmod_enabled,
    is_singleton_depset = _is_singleton_depset,
    is_tool_configuration = _is_tool_configuration,
    linkstamp_file = _linkstamp_file,
    make_runfiles_respect_legacy_external_runfiles = _make_runfiles_respect_legacy_external_runfiles,
    merge_runfiles_with_generated_inits_empty_files_supplier = _merge_runfiles_with_generated_inits_empty_files_supplier,
    regex_match = _regex_match,
    runfiles_enabled = _runfiles_enabled,
    share_native_deps = _share_native_deps,
    stamp_binaries = _stamp_binaries,
    strip_opts = _strip_opts,
)
