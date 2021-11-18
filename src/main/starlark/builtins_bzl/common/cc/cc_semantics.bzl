# Copyright 2021 The Bazel Authors. All rights reserved.
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

"""Semantics for Google cc rules"""

load(":common/cc/experimental_cc_shared_library.bzl", "CcSharedLibraryInfo")

cc_common = _builtins.toplevel.cc_common
cc_internal = _builtins.internal.cc_internal
ProtoInfo = _builtins.toplevel.ProtoInfo

def _create_cc_launcher_info(cc_info, cc_compilation_outputs):
    return cc_common.create_cc_launcher_info(cc_info = cc_info, compilation_outputs = cc_compilation_outputs)

def _contains_up_level_references(includes_path):
    return includes_path.startswith("..") and (len(includes_path) == 2 or includes_path[2] == "/")

def _expand_make_var_list(ctx, var_list, var_name):
    return [ctx.expand_make_variables(var_name, var, {}) for var in var_list]

def _validate_deps(ctx):
    # TODO(b/38307368): remove this check after cc_proto_library migration is finished.
    for dep in ctx.attr.deps:
        if ProtoInfo in dep:
            fail("dependency on proto_library '" + dep.name + "' is not allowed. Depend on the " +
                 "corresponding cc_proto_library rule instead")

def _validate_attributes(ctx):
    package_path = ctx.label.package
    for includes_attr in _expand_make_var_list(ctx, ctx.attr.includes, "includes"):
        if includes_attr.startswith("/"):
            # Will be reported later.
            continue
        includes_path = package_path + includes_attr
        if _contains_up_level_references(includes_path):
            # Will be reported later.
            continue
        if ctx.label.workspace_name == "" and not package_path.startswith("third_party") and not package_path.startswith("experimental"):
            if hasattr(ctx.rule.attr, "strip_include_prefix") and ctx.attr.strip_include_prefix != None:
                fail("this attribute is only allowed under third_party", attr = "strip_include_prefix")
            if hasattr(ctx.rule.attr, "include_prefix") and ctx.attr.include_prefix != None:
                fail("this attribute is only allowed under third_party", attr = "include_prefix")

def _determine_headers_checking_mode(ctx):
    return "loose"

    # TODO(b/198254254): Enable the implementation once isAttributeValueExplicitlySpecified is implemented.
    headers_checking_mode = ctx.fragments.google_cpp.hdrs_check

    # Package default overrides command line options.
    if cc_internal.is_package_headers_checking_mode_set():
        headers_checking_mode = cc_internal.get_package_headers_checking_mode()

    # 'hdrs_check' attribute overrides package_default.
    if hasattr(ctx.rule.attr, "hdrs_check") and ctx.attr.hdrs_check != None:
        headers_checking_mode = ctx.attr.hdrs_check
    if headers_checking_mode == "loose" and cc_internal.loose_hdrs_check_forbidden_by_allowlist(ctx):
        fail(
            "C++ rules should use 'strict' hdrs_check (not 'loose' or 'warn').The" +
            " hdrs_check='strict' ensures that all #included files must be explicitly declared" +
            " somewhere in the hdrs or srcs of the rules providing the libraries or the rules" +
            " containing the including source. Please remove 'hdrs_check' attribute from" +
            " this rule, as well as the 'default_hdrs_check' attribute from the package, or" +
            " set them to 'strict'.",
        )
    return headers_checking_mode

def _get_cc_shared_library_info(dep):
    cc_shared_library_info = dep[CcSharedLibraryInfo]
    return cc_shared_library_info

semantics = struct(
    ALLOWED_RULES_IN_DEPS = [
        "proto_library",
        "genrule",
        "cc_library",
        "cc_inc_library",
        "cc_embde_data",
        "go_library",
        "objc_library",
        "cc_import",
        "cc_proto_library",
    ],
    ALLOWED_FILES_IN_DEPS = [
        ".ld",
        ".lds",
        ".ldscript",
        ".i",
    ],
    ALLOWED_RULES_WITH_WARNINGS_IN_DEPS = [
        "gentpl",
        "gentplvars",
        "genantlr",
        "sh_library",
        "cc_binary",
        "cc_test",
    ],
    validate_deps = _validate_deps,
    validate_attributes = _validate_attributes,
    determine_headers_checking_mode = _determine_headers_checking_mode,
    get_cc_shared_library_info = _get_cc_shared_library_info,
    create_cc_launcher_info = _create_cc_launcher_info,
)
