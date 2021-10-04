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

"""Semantics for Google cc rules"""

def _get_semantics():
    return _builtins.internal.google_cc_internal.semantics

def _get_repo():
    return ""

def _additional_fragments():
    return ["google_cpp"]

def _get_licenses_attr():
    if hasattr(attr, "license"):
        # TODO(b/182226065): Change to applicable_licenses
        return {
            # buildifier: disable=attr-license
            "licenses": attr.license(),
        }
    else:
        return {
            "licenses": attr.string_list(),
        }

def _get_distribs_attr():
    return {
        "distribs": attr.string_list(),
    }

def _get_loose_mode_in_hdrs_check_allowed_attr():
    return {
        "_allowlist_loose_mode_in_hdrs_check_allowed": attr.label(
            default = "@//tools/build_defs/cc/whitelists/hdrs_check:loose_mode_in_hdrs_check_allowed",
            providers = [_builtins.internal.cc_internal.PackageGroupInfo],
        ),
    }

semantics = struct(
    get_semantics = _get_semantics,
    get_repo = _get_repo,
    additional_fragments = _additional_fragments,
    get_licenses_attr = _get_licenses_attr,
    get_distribs_attr = _get_distribs_attr,
    get_loose_mode_in_hdrs_check_allowed_attr = _get_loose_mode_in_hdrs_check_allowed_attr,
)
