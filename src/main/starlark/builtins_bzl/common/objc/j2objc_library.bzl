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
load(":common/objc/providers.bzl", "J2ObjcEntryClassInfo", "J2ObjcMappingFileInfo")
load(":common/cc/cc_info.bzl", "CcInfo")
load(":common/cc/cc_common.bzl", "cc_common")
load(":common/objc/compilation_support.bzl", "compilation_support")
load(":common/objc/j2objc_aspect.bzl", "j2objc_aspect")

_MIGRATION_TAG = "__J2OBJC_LIBRARY_MIGRATION_DO_NOT_USE_WILL_BREAK__"

def _jre_deps_aspect_impl(_, ctx):
    if "j2objc_jre_lib" not in ctx.rule.attr.tags:
        fail("in jre_deps attribute of j2objc_library rule: objc_library rule '%s' is misplaced here (Only J2ObjC JRE libraries are allowed)" %
             str(ctx.label).removeprefix("@"))
    return []

jre_deps_aspect = aspect(
    implementation = _jre_deps_aspect_impl,
)

def _check_entry_classes(ctx):
    entry_classes = ctx.attr.entry_classes
    remove_dead_code = ctx.fragments.j2objc.remove_dead_code()
    if remove_dead_code and not entry_classes:
        fail("Entry classes must be specified when flag --compilation_mode=opt is on in order to perform J2ObjC dead code stripping.")

def _entry_class_provider(entry_classes, deps):
    transitive_entry_classes = [dep[J2ObjcEntryClassInfo].entry_classes for dep in deps if J2ObjcEntryClassInfo in dep]
    return J2ObjcEntryClassInfo(entry_classes = depset(entry_classes, transitive = transitive_entry_classes))

def _mapping_file_provider(deps):
    infos = [dep[J2ObjcMappingFileInfo] for dep in deps if J2ObjcMappingFileInfo in dep]
    transitive_header_mapping_files = [info.header_mapping_files for info in infos]
    transitive_class_mapping_files = [info.class_mapping_files for info in infos]
    transitive_dependency_mapping_files = [info.dependency_mapping_files for info in infos]
    transitive_archive_source_mapping_files = [info.archive_source_mapping_files for info in infos]

    return J2ObjcMappingFileInfo(
        header_mapping_files = depset([], transitive = transitive_header_mapping_files),
        class_mapping_files = depset([], transitive = transitive_class_mapping_files),
        dependency_mapping_files = depset([], transitive = transitive_dependency_mapping_files),
        archive_source_mapping_files = depset([], transitive = transitive_archive_source_mapping_files),
    )

def j2objc_library_lockdown(ctx):
    if not ctx.fragments.j2objc.j2objc_library_migration():
        return
    if _MIGRATION_TAG not in ctx.attr.tags:
        fail("j2objc_library is locked. Please do not use this rule since it will be deleted in the future.")

def _j2objc_library_impl(ctx):
    j2objc_library_lockdown(ctx)

    _check_entry_classes(ctx)

    common_variables = compilation_support.build_common_variables(
        ctx = ctx,
        toolchain = None,
        deps = ctx.attr.deps + ctx.attr.jre_deps,
        empty_compilation_artifacts = True,
        direct_cc_compilation_contexts = [dep[CcInfo].compilation_context for dep in ctx.attr.deps if CcInfo in dep],
    )

    return [
        _entry_class_provider(ctx.attr.entry_classes, ctx.attr.deps),
        _mapping_file_provider(ctx.attr.deps),
        common_variables.objc_provider,
        CcInfo(
            compilation_context = common_variables.objc_compilation_context.create_cc_compilation_context(),
            linking_context = cc_common.merge_linking_contexts(linking_contexts = common_variables.objc_linking_context.cc_linking_contexts),
        ),
    ]

J2OBJC_ATTRS = {
    "deps": attr.label_list(
        allow_rules = ["j2objc_library", "java_library", "java_import", "java_proto_library"],
        aspects = [j2objc_aspect],
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
    provides = [CcInfo, J2ObjcEntryClassInfo, J2ObjcMappingFileInfo],
)
