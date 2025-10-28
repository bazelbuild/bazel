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

"""Exported builtins symbols that are specific to OSS Bazel."""

load(":common/cc/cc_common_bazel.bzl", "cc_common")
load(":common/java/java_common.bzl", "java_common_export_for_bazel")
load(":common/python/py_internal.bzl", "py_internal")

_REMOVED_RULES = [
    "cc_binary",
    "cc_import",
    "cc_library",
    "cc_shared_library",
    "cc_static_library",
    "cc_test",
    "cc_toolchain",
    "cc_toolchain_alias",
    "fdo_prefetch_hints",
    "fdo_profile",
    "memprof_profile",
    "propeller_optimize",
    "objc_import",
    "objc_library",
]

def _removed_rule_failure(**_kwargs):
    fail("""
         This rule has been removed from Bazel. Please add a `load()` statement for it.
         This can also be done automatically by running:
         buildifier --lint=fix <path-to-BUILD-or-bzl-file>
         """)

exported_toplevels = {
    "py_internal": py_internal,
    "java_common": java_common_export_for_bazel,
    "cc_common": cc_common,
}

exported_rules = {
    "cc_toolchain_suite": lambda name, **kwargs: _builtins.toplevel.native.filegroup(name = name),
} | {
    rule_name: _removed_rule_failure
    for rule_name in _REMOVED_RULES
}

exported_to_java = {}
