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

"""Semantics for Bazel cc rules"""

def _should_create_empty_archive():
    return False

def _validate_deps(ctx):
    pass

def _validate_attributes(ctx):
    pass

def _determine_headers_checking_mode(ctx):
    return "strict"

def _get_semantics():
    return _builtins.internal.bazel_cc_internal.semantics

def _get_stl():
    return attr.label()

def _get_repo():
    return "bazel_tools"

def _additional_fragments():
    return []

def _get_distribs_attr():
    return {}

def _get_licenses_attr():
    # TODO(b/182226065): Change to applicable_licenses
    return {}

def _get_loose_mode_in_hdrs_check_allowed_attr():
    return {}

def _get_def_parser():
    return attr.label(
        default = _builtins.internal.cc_internal.def_parser_computed_default(),
        allow_single_file = True,
        cfg = "exec",
    )

def _get_grep_includes():
    return attr.label()

def _get_interface_deps_allowed_attr():
    return {}

def _should_use_interface_deps_behavior(ctx):
    experimental_cc_interface_deps = ctx.fragments.cpp.experimental_cc_interface_deps()
    if (not experimental_cc_interface_deps and
        len(ctx.attr.interface_deps) > 0):
        fail("requires --experimental_cc_interface_deps", attr = "interface_deps")

    return experimental_cc_interface_deps

semantics = struct(
    ALLOWED_RULES_IN_DEPS = [
        "cc_library",
        "objc_library",
        "cc_proto_library",
        "cc_import",
    ],
    ALLOWED_FILES_IN_DEPS = [
        ".ld",
        ".lds",
        ".ldscript",
    ],
    ALLOWED_RULES_WITH_WARNINGS_IN_DEPS = [],
    validate_deps = _validate_deps,
    validate_attributes = _validate_attributes,
    determine_headers_checking_mode = _determine_headers_checking_mode,
    get_semantics = _get_semantics,
    get_repo = _get_repo,
    additional_fragments = _additional_fragments,
    get_distribs_attr = _get_distribs_attr,
    get_licenses_attr = _get_licenses_attr,
    get_loose_mode_in_hdrs_check_allowed_attr = _get_loose_mode_in_hdrs_check_allowed_attr,
    get_def_parser = _get_def_parser,
    get_stl = _get_stl,
    should_create_empty_archive = _should_create_empty_archive,
    get_grep_includes = _get_grep_includes,
    get_interface_deps_allowed_attr = _get_interface_deps_allowed_attr,
    should_use_interface_deps_behavior = _should_use_interface_deps_behavior,
)
