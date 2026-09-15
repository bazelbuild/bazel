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

"""Top-level exports file read by Bazel."""
# (See StarlarkBuiltinsFunction.java)

load(
    "@_builtins//:bazel/exports.bzl",
    bazel_exported_rules = "exported_rules",
    bazel_exported_to_java = "exported_to_java",
    bazel_exported_toplevels = "exported_toplevels",
)
load(
    "@_builtins//:common/exports.bzl",
    common_exported_rules = "exported_rules",
    common_exported_to_java = "exported_to_java",
    common_exported_toplevels = "exported_toplevels",
)
load("@_builtins//:common/util.bzl", "dict_union")

exported_rules = dict_union(common_exported_rules, bazel_exported_rules)
exported_toplevels = dict_union(common_exported_toplevels, bazel_exported_toplevels)
exported_to_java = dict_union(common_exported_to_java, bazel_exported_to_java)
