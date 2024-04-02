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

load(":common/cc/cc_common.bzl", "cc_common")
load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/cc/cc_info.bzl", "CcInfo")
load(":common/cc/semantics.bzl", cc_semantics = "semantics")
load(":common/java/java_info.bzl", "JavaInfo")
load(":common/java/java_semantics.bzl", java_semantics = "semantics")
load(":common/objc/apple_common.bzl", "apple_common")
load(":common/objc/compilation_support.bzl", "compilation_support")
load(":common/objc/objc_common.bzl", "objc_common")
load(":common/objc/providers.bzl", "J2ObjcMappingFileInfo")
load(":common/paths.bzl", "paths")
load(
    ":common/proto/proto_common.bzl",
    "ProtoLangToolchainInfo",
    proto_common = "proto_common_do_not_use",
)
load(":common/proto/proto_info.bzl", "ProtoInfo")

objc_internal = _builtins.internal.objc_internal

def _j2objc_source_header_search_paths(genfiles_dir_path, bin_dir_path, objc_file_path, proto_sources):
    for source_to_translate in proto_sources:
        if not source_to_translate.is_source:
            genfiles_root_header_search_path = paths.get_relative(objc_file_path, genfiles_dir_path)
            bin_root_header_search_path = paths.get_relative(objc_file_path, bin_dir_path)
            return [objc_file_path, genfiles_root_header_search_path, bin_root_header_search_path]
    return [objc_file_path]

def _proto_j2objc_source(ctx, proto_info, proto_sources, objc_file_path):
    return struct(
        target = ctx.label,
        objc_srcs = [] if not proto_sources else proto_common.declare_generated_files(ctx.actions, proto_info, ".j2objc.pb.m"),
        objc_hdrs = [] if not proto_sources else proto_common.declare_generated_files(ctx.actions, proto_info, ".j2objc.pb.h"),
        objc_file_path = objc_file_path,
        source_type = "PROTO",
        header_search_paths = _j2objc_source_header_search_paths(
            ctx.genfiles_dir.path,
            ctx.bin_dir.path,
            objc_file_path,
            proto_sources,
        ),
        compile_with_arc = False,
    )

def _get_output_objc_files(actions, srcs, objc_file_root_relative_path, suffix):
    objc_sources = []
    for src in srcs:
        src_path = src.path.removesuffix("." + src.extension)
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
    header_search_paths = _j2objc_source_header_search_paths(
        ctx.genfiles_dir.path,
        ctx.bin_dir.path,
        objc_file_root_exec_path,
        java_source_files,
    )

    if java_source_jars:
        source_tree_artifact_rel_path = _get_source_tree_artifact_rel_path(ctx.label.name)
        objc_srcs.append(ctx.actions.declare_directory(source_tree_artifact_rel_path))
        header_tree_artifact_rel_path = _get_header_tree_artifact_rel_path(ctx)
        translated_header = ctx.actions.declare_directory(header_tree_artifact_rel_path)
        objc_hdrs.append(translated_header)
        header_search_paths.append(translated_header.path)

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
        transitive_header_mapping_files.append(provider.header_mapping_files)
        transitive_class_mapping_files.append(provider.class_mapping_files)
        transitive_dependency_mapping_files.append(provider.dependency_mapping_files)
        transitive_archive_source_mapping_files.append(provider.archive_source_mapping_files)

    return J2ObjcMappingFileInfo(
        header_mapping_files = depset([], transitive = transitive_header_mapping_files),
        class_mapping_files = depset([], transitive = transitive_class_mapping_files),
        dependency_mapping_files = depset([], transitive = transitive_dependency_mapping_files),
        archive_source_mapping_files = depset([], transitive = transitive_archive_source_mapping_files),
    )

def _exported_j2objc_mapping_file_provider(target, ctx, direct_j2objc_mapping_file_provider):
    dep_j2objc_mapping_file_provider = _dep_j2objc_mapping_file_provider(ctx)

    transitive_header_mapping_files = []
    transitive_class_mapping_files = []
    transitive_dependency_mapping_files = []
    transitive_archive_source_mapping_files = []

    transitive_header_mapping_files.append(direct_j2objc_mapping_file_provider.header_mapping_files)
    transitive_class_mapping_files.append(direct_j2objc_mapping_file_provider.class_mapping_files)
    transitive_dependency_mapping_files.append(direct_j2objc_mapping_file_provider.dependency_mapping_files)
    transitive_archive_source_mapping_files.append(direct_j2objc_mapping_file_provider.archive_source_mapping_files)

    experimental_j2objc_header_map = ctx.fragments.j2objc.experimental_j2objc_header_map()
    if ProtoInfo in target or len(transitive_header_mapping_files) == 0 or experimental_j2objc_header_map:
        transitive_header_mapping_files.append(dep_j2objc_mapping_file_provider.header_mapping_files)
    transitive_class_mapping_files.append(dep_j2objc_mapping_file_provider.class_mapping_files)
    transitive_dependency_mapping_files.append(dep_j2objc_mapping_file_provider.dependency_mapping_files)
    transitive_archive_source_mapping_files.append(dep_j2objc_mapping_file_provider.archive_source_mapping_files)

    return J2ObjcMappingFileInfo(
        header_mapping_files = depset([], transitive = transitive_header_mapping_files),
        class_mapping_files = depset([], transitive = transitive_class_mapping_files),
        dependency_mapping_files = depset([], transitive = transitive_dependency_mapping_files),
        archive_source_mapping_files = depset([], transitive = transitive_archive_source_mapping_files),
    )

def _get_file_path_with_suffix(objc_srcs, suffix):
    for src in objc_srcs:
        if src.path.endswith(suffix):
            return src.path
    fail("File with %s suffix must exist inside objc_sources.", suffix)

def _create_j2objc_transpilation_action(
        ctx,
        java_source_files,
        java_source_jars,
        dep_j2objc_mapping_file_provider,
        transitive_compile_time_jars,
        j2objc_source):
    java_runtime = ctx.toolchains[java_semantics.JAVA_TOOLCHAIN_TYPE].java.java_runtime

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
        args.add("--output_gen_source_dir", _get_file_path_with_suffix(j2objc_source.objc_srcs, "source_files"))
        args.add("--output_gen_header_dir", _get_file_path_with_suffix(j2objc_source.objc_hdrs, "header_files"))

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

    compiled_library = objc_internal.j2objc_create_intermediate_artifacts(ctx = ctx).archive()
    args.add("--compiled_archive_file_path", compiled_library)

    boothclasspath_jar = ctx.file._jre_emul_jar
    args.add("-Xbootclasspath:" + boothclasspath_jar.path)

    module_files = ctx.attr._jre_emul_module.files.to_list()
    for file in module_files:
        if file.basename == "release":
            args.add("--system", file.dirname)
            break

    dead_code_report = ctx.file._dead_code_report
    if dead_code_report:
        args.add("--dead-code-report", dead_code_report)

    args.add("-d", objc_file_path)

    if transitive_compile_time_jars:
        args.add_joined("-classpath", transitive_compile_time_jars.to_list(), join_with = ":")

    args.add_all(java_source_files)

    direct_files = [j2objc_deploy_jar, boothclasspath_jar]
    if dead_code_report != None:
        direct_files.append(dead_code_report)
    if not experimental_j2objc_header_map:
        direct_files.append(output_header_mapping_file)

    ctx.actions.run(
        mnemonic = "TranspilingJ2objc",
        executable = ctx.executable._j2objc_wrapper,
        arguments = [args],
        inputs = depset(
            direct_files + module_files + java_source_files + java_source_jars,
            transitive = [
                transitive_compile_time_jars,
                java_runtime.files,
                header_mapping_files,
                deps_class_mapping_files,
            ],
        ),
        outputs = j2objc_source.objc_srcs + j2objc_source.objc_hdrs +
                  [output_dep_mapping_file, archive_source_mapping_file],
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
                java_source_jars.append(src)
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
        dep_attributes = ["_jre_lib", "deps", "exports", "runtime_deps"],
        proto_toolchain_runtime = [],
    )

def _common(
        ctx,
        intermediate_artifacts,
        transpiled_sources,
        transpiled_headers,
        header_search_paths,
        dependent_attributes,
        other_deps,
        compile_with_arc):
    compilation_artifacts = None
    has_module_map = False
    if transpiled_sources or transpiled_headers:
        if compile_with_arc:
            compilation_artifacts = objc_internal.j2objc_create_compilation_artifacts(
                srcs = transpiled_sources,
                non_arc_srcs = [],
                hdrs = transpiled_headers,
                intermediate_artifacts = intermediate_artifacts,
            )
        else:
            compilation_artifacts = objc_internal.j2objc_create_compilation_artifacts(
                srcs = [],
                non_arc_srcs = transpiled_sources,
                hdrs = transpiled_headers,
                intermediate_artifacts = intermediate_artifacts,
            )
        has_module_map = True

    deps = []
    for dep_attr in dependent_attributes:
        if dep_attr == "_jre_lib":
            deps.append(ctx.attr._jre_lib)
        elif hasattr(ctx.rule.attr, dep_attr):
            deps.extend(getattr(ctx.rule.attr, dep_attr))

    (
        objc_provider,
        objc_compilation_context,
        objc_linking_context,
    ) = objc_common.create_context_and_provider(
        ctx = ctx,
        compilation_artifacts = compilation_artifacts,
        has_module_map = has_module_map,
        deps = deps + other_deps,
        intermediate_artifacts = intermediate_artifacts,
        includes = header_search_paths,
        compilation_attributes = None,
        implementation_deps = [],
        attr_linkopts = [],
        is_aspect = True,
    )

    return struct(
        compilation_artifacts = compilation_artifacts,
        objc_provider = objc_provider,
        objc_compilation_context = objc_compilation_context,
        objc_linking_context = objc_linking_context,
    )

def _build_aspect(
        target,
        ctx,
        j2objc_source,
        direct_j2objc_mapping_file_provider,
        dep_attributes,
        proto_toolchain_runtime):
    intermediate_artifacts = objc_internal.j2objc_create_intermediate_artifacts(ctx = ctx)
    if j2objc_source.objc_srcs:
        common = _common(
            ctx = ctx,
            intermediate_artifacts = intermediate_artifacts,
            transpiled_sources = j2objc_source.objc_srcs,
            transpiled_headers = j2objc_source.objc_hdrs,
            header_search_paths = j2objc_source.header_search_paths,
            dependent_attributes = dep_attributes,
            other_deps = proto_toolchain_runtime,
            compile_with_arc = j2objc_source.compile_with_arc,
        )

        cc_toolchain = cc_helper.find_cpp_toolchain(ctx)

        if j2objc_source.compile_with_arc:
            extra_compile_args = ["-fno-strict-overflow", "-fobjc-arc-exceptions"]
        else:
            extra_compile_args = ["-fno-strict-overflow", "-fobjc-weak"]

        compilation_result = compilation_support.register_compile_and_archive_actions_for_j2objc(
            ctx = ctx,
            toolchain = cc_toolchain,
            intermediate_artifacts = intermediate_artifacts,
            compilation_artifacts = common.compilation_artifacts,
            objc_compilation_context = common.objc_compilation_context,
            cc_linking_contexts = common.objc_linking_context.cc_linking_contexts,
            extra_compile_args = extra_compile_args,
        )
        cc_compilation_context = compilation_result[0]
        cc_linking_context = compilation_result[1]
    else:
        common = _common(
            ctx = ctx,
            intermediate_artifacts = intermediate_artifacts,
            transpiled_sources = [],
            transpiled_headers = [],
            header_search_paths = [],
            dependent_attributes = dep_attributes,
            other_deps = proto_toolchain_runtime,
            compile_with_arc = j2objc_source.compile_with_arc,
        )
        cc_compilation_context = common.objc_compilation_context.create_cc_compilation_context()
        cc_linking_context = cc_common.merge_linking_contexts(
            linking_contexts = common.objc_linking_context.cc_linking_contexts,
        )

    return [
        _exported_j2objc_mapping_file_provider(target, ctx, direct_j2objc_mapping_file_provider),
        common.objc_provider,
        CcInfo(
            compilation_context = cc_compilation_context,
            linking_context = cc_linking_context,
        ),
    ]

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
            cfg = "exec",
            allow_single_file = True,
            default = Label("@" + cc_semantics.get_repo() + "//third_party/java/j2objc:jre_emul.jar"),
        ),
        "_jre_emul_module": attr.label(
            cfg = "exec",
            allow_files = True,
            default = Label("@" + cc_semantics.get_repo() + "//third_party/java/j2objc:jre_emul_module"),
        ),
        "_dead_code_report": attr.label(
            allow_single_file = True,
            cfg = "exec",
            default = configuration_field(
                name = "dead_code_report",
                fragment = "j2objc",
            ),
        ),
        "_jre_lib": attr.label(
            allow_files = True,
            default = Label("@" + cc_semantics.get_repo() + "//third_party/java/j2objc:jre_core_lib"),
        ),
        "_j2objc_proto_toolchain": attr.label(
            default = configuration_field(fragment = "proto", name = "proto_toolchain_for_j2objc"),
        ),
    },
    required_providers = [[JavaInfo], [ProtoInfo]],
    provides = [apple_common.Objc, CcInfo],
    toolchains = [java_semantics.JAVA_TOOLCHAIN_TYPE] + cc_helper.use_cpp_toolchain(),
    fragments = ["apple", "cpp", "j2objc", "objc", "proto"],
)
