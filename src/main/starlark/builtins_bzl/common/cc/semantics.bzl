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
load(":common/cc/experimental_cc_shared_library.bzl", "CcSharedLibraryInfo")

def _create_cc_launcher_info(cc_info, cc_compilation_outputs):
    return None

def _validate_deps(ctx):
    pass

def _validate_attributes(ctx):
    pass

def _determine_headers_checking_mode(ctx):
    return "strict"

def _get_cc_shared_library_info(dep):
    return dep[CcSharedLibraryInfo]

def _get_semantics():
    return _builtins.internal.bazel_cc_internal.semantics

def _get_repo():
    return ""

def _additional_fragments():
    return []

def _get_licenses_attr():
    # TODO(b/182226065): Change to applicable_licenses
    return {}

def _get_loose_mode_in_hdrs_check_allowed_attr():
    return {}

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
    get_cc_shared_library_info = _get_cc_shared_library_info,
    create_cc_launcher_info = _create_cc_launcher_info,
    get_semantics = _get_semantics,
    get_repo = _get_repo,
    additional_fragments = _additional_fragments,
    get_licenses_attr = _get_licenses_attr,
    get_loose_mode_in_hdrs_check_allowed_attr = _get_loose_mode_in_hdrs_check_allowed_attr,
)
