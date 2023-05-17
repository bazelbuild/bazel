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
Definition of j2objc_aspect.
"""

load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/cc/semantics.bzl", cc_semantics = "semantics")
load(":common/java/java_semantics.bzl", java_semantics = "semantics")
load(":common/proto/proto_info.bzl", "ProtoInfo")
load(":common/objc/providers.bzl", "J2ObjcMappingFileInfo")
load(
    ":common/proto/proto_common.bzl",
    "ProtoLangToolchainInfo",
    proto_common = "proto_common_do_not_use",
)

JavaInfo = _builtins.toplevel.JavaInfo
apple_common = _builtins.toplevel.apple_common

def _proto_j2objc_source(ctx, proto_info, proto_sources, objc_file_path):
    return struct(
        target = ctx.label,
        objc_srcs = [] if not proto_sources else proto_common.declare_generated_files(ctx.actions, proto_info, ".j2objc.pb.m"),
        objc_hdrs = [] if not proto_sources else proto_common.declare_generated_files(ctx.actions, proto_info, ".j2objc.pb.h"),
        objc_file_path = objc_file_path,
        source_type = "PROTO",
        header_search_paths = [objc_file_path],
        compile_with_arc = False,
    )

def _create_j2objc_proto_compile_actions(
        proto_info,
        proto_lang_toolchain_info,
        ctx,
        filtered_proto_sources_non_empty,
        j2objc_source,
        objc_file_path):
    output_header_mapping_files = []
    output_class_mapping_files = []
    if filtered_proto_sources_non_empty:
        output_header_mapping_files = proto_common.declare_generated_files(ctx.actions, proto_info, ".j2objc.mapping")
        output_class_mapping_files = proto_common.declare_generated_files(ctx.actions, proto_info, ".clsmap.properties")

    outputs = j2objc_source.objc_srcs + j2objc_source.objc_hdrs + output_header_mapping_files + output_class_mapping_files

    proto_common.compile(
        actions = ctx.actions,
        proto_info = proto_info,
        proto_lang_toolchain_info = proto_lang_toolchain_info,
        generated_files = outputs,
        plugin_output = objc_file_path,
    )

    return J2ObjcMappingFileInfo(
        header_mapping_files = depset(direct = output_header_mapping_files),
        class_mapping_files = depset(direct = output_class_mapping_files),
        dependency_mapping_files = depset([]),
        archive_source_mapping_files = depset([]),
    )

def _proto(target, ctx):
    proto_lang_toolchain_info = ctx.attr._j2objc_proto_toolchain[ProtoLangToolchainInfo]
    filtered_proto_sources, _ = proto_common.experimental_filter_sources(target[ProtoInfo], proto_lang_toolchain_info)
    objc_file_path = cc_helper.proto_output_root(
        proto_root = target[ProtoInfo].proto_source_root,
        genfiles_dir_path = ctx.genfiles_dir.path,
        bin_dir_path = ctx.bin_dir.path,
    )
    j2objc_source = _proto_j2objc_source(ctx, target[ProtoInfo], filtered_proto_sources, objc_file_path)

    direct_j2objc_mapping_file_provider = None
    if len(j2objc_source.objc_srcs) == 0:
        direct_j2objc_mapping_file_provider = J2ObjcMappingFileInfo(
            header_mapping_files = depset([]),
            class_mapping_files = depset([]),
            dependency_mapping_files = depset([]),
            archive_source_mapping_files = depset([]),
        )
    else:
        direct_j2objc_mapping_file_provider = _create_j2objc_proto_compile_actions(
            proto_info = target[ProtoInfo],
            proto_lang_toolchain_info = proto_lang_toolchain_info,
            ctx = ctx,
            filtered_proto_sources_non_empty = len(filtered_proto_sources) > 0,
            j2objc_source = j2objc_source,
            objc_file_path = objc_file_path,
        )

    return _build_aspect(
        target = target,
        ctx = ctx,
        j2objc_source = j2objc_source,
        direct_j2objc_mapping_file_provider = direct_j2objc_mapping_file_provider,
        dep_attributes = "deps",
        proto_toolchain_runtime = [proto_lang_toolchain_info.runtime],
    )

# buildifier: disable=unused-variable Implementation will be continued.
def _java(target, ctx):
    return []

# buildifier: disable=unused-variable Implementation will be continued.
def _build_aspect(target, ctx, j2objc_source, direct_j2objc_mapping_file_provider, dep_attributes, proto_toolchain_runtime):
    return []

def _j2objc_aspect_impl(target, ctx):
    if ProtoInfo in target:
        return _proto(target, ctx)
    return _java(target, ctx)

j2objc_aspect = aspect(
    implementation = _j2objc_aspect_impl,
    attr_aspects = ["deps", "exports", "runtime_deps"],
    attrs = {
        "_use_auto_exec_groups": attr.bool(default = True),
        "_j2objc_wrapper": attr.label(
            cfg = "exec",
            default = "@" + cc_semantics.get_repo() + "//tools/j2objc:j2objc_wrapper_binary",
        ),
        "_j2objc_header_map": attr.label(
            cfg = "exec",
            default = "@" + cc_semantics.get_repo() + "//tools/j2objc:j2objc_header_map_binary",
        ),
        "_jre_emul_jar": attr.label(
            allow_files = True,
            default = Label("@//third_party/java/j2objc:jre_emul.jar"),
        ),
        "_jre_emul_module": attr.label(
            allow_files = True,
            default = Label("@//third_party/java/j2objc:jre_emul_module"),
        ),
        "_dead_code_report": attr.label(
            default = configuration_field(
                name = "dead_code_report",
                fragment = "j2objc",
            ),
        ),
        "_jre_lib": attr.label(
            allow_files = True,
            default = Label("@//third_party/java/j2objc:jre_core_lib"),
        ),
        "_xcrunwrapper": attr.label(
            allow_files = True,
            cfg = "exec",
            default = "@" + cc_semantics.get_repo() + "//tools/objc:xcrunwrapper",
        ),
        "_xcode_config": attr.label(
            allow_rules = ["xcode_config"],
            default = configuration_field(
                fragment = "apple",
                name = "xcode_config_label",
            ),
            # TODO(kotlaja): Do we need "checkConstraints" here? Label doesn't have a flag attribute.
        ),
        "_zipper": attr.label(
            allow_files = True,
            cfg = "exec",
            default = "@" + cc_semantics.get_repo() + "//tools/zip:zipper",
        ),
        "_j2objc_proto_toolchain": attr.label(
            default = configuration_field(fragment = "proto", name = "proto_toolchain_for_j2objc"),
        ),
        "_java_toolchain_type": attr.label(default = java_semantics.JAVA_TOOLCHAIN_TYPE),
        "_cc_toolchain": attr.label(
            default = "@" + cc_semantics.get_repo() + "//tools/cpp:current_cc_toolchain",
        ),
    },
    required_providers = [[JavaInfo], [ProtoInfo]],
    provides = [apple_common.Objc],
    toolchains = cc_helper.use_cpp_toolchain(),
    fragments = ["apple", "cpp", "j2objc", "objc", "proto"],
)
