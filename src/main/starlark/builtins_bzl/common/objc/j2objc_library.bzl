# Copyright 2023 The Bazel Authors. All rights reserved.
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
Definition of j2objc_library rule.
"""

load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/objc/attrs.bzl", "common_attrs")
load(":common/objc/transitions.bzl", "apple_crosstool_transition")
load(":common/cc/semantics.bzl", "semantics")

def _jre_deps_aspect_impl(_, ctx):
    if "j2objc_jre_lib" not in ctx.rule.attr.tags:
        fail("in jre_deps attribute of j2objc_library rule: objc_library rule '%s' is misplaced here (Only J2ObjC JRE libraries are allowed)" %
             str(ctx.label).removeprefix("@"))

jre_deps_aspect = aspect(
    implementation = _jre_deps_aspect_impl,
)

def _j2objc_library_impl(ctx):
    # TODO(kotlaja): Continue with the implementation.
    return [ctx.label]

J2OBJC_ATTRS = {
    "deps": attr.label_list(
        allow_rules = ["j2objc_library", "java_library", "java_import", "java_proto_library"],
        # aspects = [j2objc_aspect],
    ),
    "entry_classes": attr.string_list(),
    "jre_deps": attr.label_list(
        allow_rules = ["objc_library"],
        aspects = [jre_deps_aspect],
    ),
}

j2objc_library = rule(
    _j2objc_library_impl,
    attrs = common_attrs.union(
        J2OBJC_ATTRS,
        common_attrs.CC_TOOLCHAIN_RULE,
    ),
    cfg = apple_crosstool_transition,
    fragments = ["apple", "cpp", "j2objc", "objc"] + semantics.additional_fragments(),
    toolchains = cc_helper.use_cpp_toolchain(),
)
