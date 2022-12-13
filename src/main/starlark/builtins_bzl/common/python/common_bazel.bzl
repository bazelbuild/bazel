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
"""Common functions that are specific to Bazel rule implementation"""

_py_builtins = _builtins.internal.py_builtins
_CcInfo = _builtins.toplevel.CcInfo
_cc_common = _builtins.toplevel.cc_common

def collect_cc_info(ctx, extra_deps = []):
    """Collect C++ information from dependencies for Bazel.

    Args:
        ctx: Rule ctx; must have `deps` attribute.
        extra_deps: list of Target to also collect C+ information from.

    Returns:
        CcInfo provider of merged information.
    """
    deps = ctx.attr.deps
    if extra_deps:
        deps = list(deps)
        deps.extend(extra_deps)
    cc_infos = []
    for dep in deps:
        if _CcInfo in dep:
            cc_infos.append(dep[_CcInfo])
        if PyCcLinkParamsProvider in dep:
            cc_infos.append(dep[PyCcLinkParamsProvider].cc_info)
    return _cc_common.merge_cc_infos(cc_infos = cc_infos)

def maybe_precompile(ctx, srcs):
    """Computes all the outputs (maybe precompiled) from the input srcs.

    See create_binary_semantics_struct for details about this function.

    Args:
        ctx: Rule ctx.
        srcs: List of Files; the inputs to maybe precompile.

    Returns:
        List of Files; the desired output files derived from the input sources.
    """

    # Precompilation isn't implemented yet, so just return srcs as-is
    return srcs

def get_imports(ctx):
    """Gets the imports from a rule's `imports` attribute.

    See create_binary_semantics_struct for details about this function.

    Args:
        ctx: Rule ctx.

    Returns:
        List of strings.
    """
    prefix = "{}/{}".format(
        ctx.workspace_name,
        _py_builtins.get_label_repo_runfiles_path(ctx.label),
    )
    result = []
    for import_str in ctx.attr.imports:
        import_str = ctx.expand_make_variables(import_str)
        if import_str.startswith("/"):
            continue
        import_path = "{}/{}".format(prefix, expanded)

        # TODO: Reject paths with uplevel references (see
        # PathFragment#containsUplevelReferences)
        result.append(import_path)
    return result
