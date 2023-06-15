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

load(":common/paths.bzl", "paths")
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
load(":common/java/java_info.bzl", "JavaInfo")

apple_common = _builtins.toplevel.apple_common
objc_internal = _builtins.internal.objc_internal

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

def _get_output_objc_files(actions, srcs, objc_file_root_relative_path, suffix):
    objc_sources = []
    for src in srcs:
        src_path = src.short_path.removesuffix("." + src.extension)
        objc_source_path = paths.get_relative(objc_file_root_relative_path, src_path) + suffix
        objc_sources.append(actions.declare_file(objc_source_path))
    return objc_sources

def _get_header_base(experimental_shorter_header_path):
    return "_ios" if experimental_shorter_header_path else "_j2objc"

def _get_source_tree_artifact_rel_path(label_name):
    return "_j2objc/src_jar_files/" + label_name + "/source_files"

def _get_header_tree_artifact_rel_path(ctx):
    header_base = _get_header_base(ctx.fragments.j2objc.experimental_shorter_header_path())
    return header_base + "/src_jar_files/" + ctx.label.name + "/header_files"

def _java_j2objc_source(ctx, java_source_files, java_source_jars):
    header_base = _get_header_base(ctx.fragments.j2objc.experimental_shorter_header_path())
    objc_file_root_relative_path = header_base + "/" + ctx.label.name
    objc_file_root_exec_path = paths.get_relative(ctx.bin_dir.path, ctx.label.package + "/" + objc_file_root_relative_path)

    objc_srcs = _get_output_objc_files(
        ctx.actions,
        java_source_files,
        objc_file_root_relative_path,
        ".m",
    )
    objc_hdrs = _get_output_objc_files(
        ctx.actions,
        java_source_files,
        objc_file_root_relative_path,
        ".h",
    )
    header_search_paths = [objc_file_root_exec_path]

    if java_source_jars:
        source_tree_artifact_rel_path = _get_source_tree_artifact_rel_path(ctx.label.name)
        objc_srcs.append(ctx.actions.declare_directory(source_tree_artifact_rel_path))
        header_tree_artifact_rel_path = _get_header_tree_artifact_rel_path(ctx)
        translated_header = ctx.actions.declare_directory(header_tree_artifact_rel_path)
        objc_hdrs.append(translated_header)
        header_search_paths.append(translated_header.short_path)

    return struct(
        target = ctx.label,
        objc_srcs = objc_srcs,
        objc_hdrs = objc_hdrs,
        objc_file_path = objc_file_root_exec_path,
        source_type = "JAVA",
        header_search_paths = header_search_paths,
        compile_with_arc = "-use-arc" in ctx.fragments.j2objc.translation_flags,
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

def _empty_j2objc_mapping_file_info():
    return J2ObjcMappingFileInfo(
        header_mapping_files = depset([]),
        class_mapping_files = depset([]),
        dependency_mapping_files = depset([]),
        archive_source_mapping_files = depset([]),
    )

def _proto(target, ctx):
    proto_lang_toolchain_info = ctx.attr._j2objc_proto_toolchain[ProtoLangToolchainInfo]
    filtered_proto_sources, _ = proto_common.experimental_filter_sources(target[ProtoInfo], proto_lang_toolchain_info)
    objc_file_path = cc_helper.proto_output_root(
        proto_root = target[ProtoInfo].proto_source_root,
        bin_dir_path = ctx.bin_dir.path,
    )
    j2objc_source = _proto_j2objc_source(ctx, target[ProtoInfo], filtered_proto_sources, objc_file_path)

    direct_j2objc_mapping_file_provider = None
    if len(j2objc_source.objc_srcs) == 0:
        direct_j2objc_mapping_file_provider = _empty_j2objc_mapping_file_info()
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
        dep_attributes = ["deps"],
        proto_toolchain_runtime = [proto_lang_toolchain_info.runtime],
    )

def _dep_j2objc_mapping_file_provider(ctx):
    transitive_header_mapping_files = []
    transitive_class_mapping_files = []
    transitive_dependency_mapping_files = []
    transitive_archive_source_mapping_files = []

    deps = getattr(ctx.rule.attr, "deps", []) + getattr(ctx.rule.attr, "runtime_deps", []) + getattr(ctx.rule.attr, "exports", [])
    providers = [dep[J2ObjcMappingFileInfo] for dep in deps if J2ObjcMappingFileInfo in dep]
    for provider in providers:
        transitive_header_mapping_files.extend(provider.header_mapping_files)
        transitive_class_mapping_files.extend(provider.class_mapping_files)
        transitive_dependency_mapping_files.extend(provider.dependency_mapping_files)
        transitive_archive_source_mapping_files.extend(provider.archive_source_mapping_files)

    return J2ObjcMappingFileInfo(
        header_mapping_files = depset([], transitive = transitive_header_mapping_files),
        class_mapping_files = depset([], transitive = transitive_class_mapping_files),
        dependency_mapping_files = depset([], transitive = transitive_dependency_mapping_files),
        archive_source_mapping_files = depset([], transitive = transitive_archive_source_mapping_files),
    )

def _create_j2objc_transpilation_action(
        ctx,
        java_source_files,
        java_source_jars,
        dep_j2objc_mapping_file_provider,
        transitive_compile_time_jars,
        j2objc_source):
    java_runtime = java_semantics.find_java_runtime_toolchain(ctx)

    args = ctx.actions.args()
    args.use_param_file(param_file_arg = "@%s", use_always = True)
    args.set_param_file_format("multiline")

    args.add("--java", java_runtime.java_executable_exec_path)

    j2objc_deploy_jar = ctx.file._j2objc
    args.add("--j2objc", j2objc_deploy_jar)

    args.add("--main_class", "com.google.devtools.j2objc.J2ObjC")

    objc_file_path = j2objc_source.objc_file_path
    args.add("--objc_file_path", objc_file_path)

    output_dep_mapping_file = ctx.actions.declare_file(ctx.label.name + ".dependency_mapping.j2objc")
    args.add("--output_dependency_mapping_file", output_dep_mapping_file)

    if java_source_jars:
        args.add_joined("--src_jars", java_source_jars, join_with = ",")
        args.add("--output_gen_source_dir", ctx.bin_dir.path + "/" + j2objc_source.objc_srcs[0].short_path)
        args.add("--output_gen_header_dir", ctx.bin_dir.path + "/" + j2objc_source.objc_hdrs[0].short_path)

    args.add_all(ctx.fragments.j2objc.translation_flags)

    header_mapping_files = dep_j2objc_mapping_file_provider.header_mapping_files
    if header_mapping_files:
        args.add_joined("--header-mapping", header_mapping_files, join_with = ",")

    experimental_j2objc_header_map = ctx.fragments.j2objc.experimental_j2objc_header_map()
    output_header_mapping_file = ctx.actions.declare_file(ctx.label.name + ".mapping.j2objc")
    if not experimental_j2objc_header_map:
        args.add("--output-header-mapping", output_header_mapping_file)

    deps_class_mapping_files = dep_j2objc_mapping_file_provider.class_mapping_files
    if deps_class_mapping_files:
        args.add_joined("--mapping", deps_class_mapping_files, join_with = ",")

    archive_source_mapping_file = ctx.actions.declare_file(ctx.label.name + ".archive_source_mapping.j2objc")
    args.add("--output_archive_source_mapping_file", archive_source_mapping_file)

    compiled_library = objc_internal.j2objc_create_intermediate_artifacts(ctx = ctx)
    args.add("--compiled_archive_file_path", compiled_library)

    boothclasspath_jar = ctx.file._jre_emul_jar
    args.add("-Xbootclasspath:" + boothclasspath_jar.short_path)

    module_files = [m for target in getattr(ctx.rule.attr, "_jre_emul_module", []) for m in target.files.to_list()]
    for file in module_files:
        if file.basename == "release":
            args.add("--system", file.dirname)
            break

    dead_code_report = ctx.attr._dead_code_report
    if dead_code_report:
        args.add("--dead-code-report", dead_code_report)

    args.add("-d", objc_file_path)

    if transitive_compile_time_jars:
        args.add_joined("-classpath", transitive_compile_time_jars.to_list(), join_with = ":")

    args.add_all(java_source_files)

    ctx.actions.run(
        mnemonic = "TranspilingJ2objc",
        executable = ctx.executable._j2objc_wrapper,
        arguments = [args],
        inputs = depset(
            [
                j2objc_deploy_jar,
                boothclasspath_jar,
                dead_code_report,
            ] + module_files + java_source_files + java_source_jars +
            [output_header_mapping_file] if not experimental_j2objc_header_map else [],
            transitive = [
                transitive_compile_time_jars,
                java_runtime.files,
                header_mapping_files,
                deps_class_mapping_files,
            ],
        ),
        outputs = [output_dep_mapping_file, archive_source_mapping_file] +
                  j2objc_source.objc_srcs +
                  j2objc_source.objc_hdrs,
        toolchain = None,
    )

    if experimental_j2objc_header_map:
        args_header_map = ctx.actions.args()
        if java_source_files:
            args_header_map.add_joined("--source_files", java_source_files, join_with = ",")
        if java_source_jars:
            args_header_map.add_joined("--source_jars", java_source_jars, join_with = ",")
        args_header_map.add("--output_mapping_file", output_header_mapping_file)

        ctx.actions.run(
            mnemonic = "GenerateJ2objcHeaderMap",
            executable = ctx.executable._j2objc_header_map,
            arguments = [args_header_map],
            inputs = java_source_files + java_source_jars,
            outputs = [output_header_mapping_file],
            toolchain = None,
        )

    return J2ObjcMappingFileInfo(
        header_mapping_files = depset([output_header_mapping_file]),
        class_mapping_files = depset([]),
        dependency_mapping_files = depset([output_dep_mapping_file]),
        archive_source_mapping_files = depset([archive_source_mapping_file]),
    )

def _java(target, ctx):
    java_info = target[JavaInfo]
    transitive_compile_time_jars = java_info.transitive_compile_time_jars
    generated_source_jars = [
        output.generated_source_jar
        for output in java_info.java_outputs
        if output.generated_source_jar != None
    ]

    java_source_files = []
    java_source_jars = []
    if hasattr(ctx.rule.attr, "srcs"):
        for src in ctx.rule.files.srcs:
            src_path = src.path
            if src_path.endswith(".srcjar"):
                java_source_jars.append()(src)
            if src_path.endswith(".java"):
                java_source_files.append(src)

    src_jar_target = getattr(ctx.rule.attr, "srcjar", None)
    if src_jar_target:
        java_source_jars.extend(ctx.rule.files.srcjar)
    if generated_source_jars:
        java_source_jars.extend(generated_source_jars)

    j2objc_source = _java_j2objc_source(ctx, java_source_files, java_source_jars)

    dep_j2objc_mapping_file_provider = _dep_j2objc_mapping_file_provider(ctx)

    if len(j2objc_source.objc_srcs) == 0:
        direct_j2objc_mapping_file_provider = _empty_j2objc_mapping_file_info()
    else:
        direct_j2objc_mapping_file_provider = _create_j2objc_transpilation_action(
            ctx,
            java_source_files,
            java_source_jars,
            dep_j2objc_mapping_file_provider,
            transitive_compile_time_jars,
            j2objc_source,
        )
    return _build_aspect(
        target = target,
        ctx = ctx,
        j2objc_source = j2objc_source,
        direct_j2objc_mapping_file_provider = direct_j2objc_mapping_file_provider,
        dep_attributes = ["$jre_lib", "deps", "exports", "runtime_deps"],
        proto_toolchain_runtime = [],
    )

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
        "_j2objc": attr.label(
            cfg = "exec",
            allow_single_file = True,
            default = "@" + cc_semantics.get_repo() + "//tools/j2objc:j2objc_deploy.jar",
        ),
        "_j2objc_wrapper": attr.label(
            cfg = "exec",
            executable = True,
            default = "@" + cc_semantics.get_repo() + "//tools/j2objc:j2objc_wrapper_binary",
        ),
        "_j2objc_header_map": attr.label(
            cfg = "exec",
            executable = True,
            default = "@" + cc_semantics.get_repo() + "//tools/j2objc:j2objc_header_map_binary",
        ),
        "_jre_emul_jar": attr.label(
            allow_single_file = True,
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
    toolchains = [java_semantics.JAVA_TOOLCHAIN_TYPE, java_semantics.JAVA_RUNTIME_TOOLCHAIN_TYPE] + cc_helper.use_cpp_toolchain(),
    fragments = ["apple", "cpp", "j2objc", "objc", "proto"],
)
