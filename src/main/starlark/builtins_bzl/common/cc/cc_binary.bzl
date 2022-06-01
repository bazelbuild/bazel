# Copyright 2021 The Bazel Authors. All rights reserved.
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

"""cc_binary Starlark implementation replacing native"""

load(":common/cc/semantics.bzl", "semantics")
load(":common/cc/experimental_cc_shared_library.bzl", "CcSharedLibraryInfo", "GraphNodeInfo", "build_exports_map_from_only_dynamic_deps", "build_link_once_static_libs_map", "merge_cc_shared_library_infos")
load(":common/cc/cc_helper.bzl", "cc_helper")

CcInfo = _builtins.toplevel.CcInfo
ProtoInfo = _builtins.toplevel.ProtoInfo
DebugPackageInfo = _builtins.toplevel.DebugPackageInfo
cc_common = _builtins.toplevel.cc_common
cc_internal = _builtins.internal.cc_internal
StaticallyLinkedMarkerInfo = _builtins.internal.StaticallyLinkedMarkerProvider

_EXECUTABLE = "executable"
_DYNAMIC_LIBRARY = "dynamic_library"

_LINKING_DYNAMIC = "dynamic_linking_mode"
_LINKING_STATIC = "static_linking_mode"

_IOS_SIMULATOR_TARGET_CPUS = ["ios_x86_64", "ios_i386", "ios_sim_arm64"]
_IOS_DEVICE_TARGET_CPUS = ["ios_armv6", "ios_arm64", "ios_armv7", "ios_armv7s", "ios_arm64e"]
_WATCHOS_SIMULATOR_TARGET_CPUS = ["watchos_i386", "watchos_x86_64", "watchos_arm64"]
_WATCHOS_DEVICE_TARGET_CPUS = ["watchos_armv7k", "watchos_arm64_32"]
_TVOS_SIMULATOR_TARGET_CPUS = ["tvos_x86_64", "tvos_sim_arm64"]
_TVOS_DEVICE_TARGET_CPUS = ["tvos_arm64"]
_CATALYST_TARGET_CPUS = ["catalyst_x86_64"]
_MACOS_TARGET_CPUS = ["darwin_x86_64", "darwin_arm64", "darwin_arm64e", "darwin"]

def _strip_extension(file):
    if file.extension == "":
        return file.basename
    return file.basename[:-(1 + len(file.extension))]

def _new_dwp_action(ctx, cc_toolchain, dwp_tools):
    return {
        "tools": dwp_tools,
        "executable": cc_toolchain.tool_path(tool = "DWP"),
        "arguments": ctx.actions.args(),
        "inputs": [],
        "outputs": [],
    }

def _get_intermediate_dwp_file(ctx, dwp_output, order_number):
    output_path = dwp_output.short_path

    # Since it is a dwp_output we can assume that it always
    # ends with .dwp suffix, because it is declared so in outputs
    # attribute.
    extension_stripped_output_path = output_path[0:len(output_path) - 4]
    intermediate_path = extension_stripped_output_path + "-" + str(order_number) + ".dwp"

    return ctx.actions.declare_file("_dwps/" + intermediate_path)

def _create_intermediate_dwp_packagers(ctx, dwp_output, cc_toolchain, dwp_files, dwo_files, intermediate_dwp_count):
    intermediate_outputs = dwo_files

    # This long loop is a substitution for recursion, which is not currently supported in Starlark.
    for i in range(2147483647):
        packagers = []
        current_packager = _new_dwp_action(ctx, cc_toolchain, dwp_files)
        inputs_for_current_packager = 0

        # Step 1: generate our batches. We currently break into arbitrary batches of fixed maximum
        # input counts, but we can always apply more intelligent heuristics if the need arises.
        for dwo_file in intermediate_outputs:
            if inputs_for_current_packager == 100:
                packagers.append(current_packager)
                current_packager = _new_dwp_action(ctx, cc_toolchain, dwp_files)
                inputs_for_current_packager = 0
            current_packager["inputs"].append(dwo_file)
            current_packager["arguments"].add(dwo_file)
            inputs_for_current_packager += 1

        packagers.append(current_packager)

        # Step 2: given the batches, create the actions.
        if len(packagers) > 1:
            # If we have multiple batches, make them all intermediate actions, then pipe their outputs
            # into an additional level.
            intermediate_outputs = []
            for packager in packagers:
                intermediate_output = _get_intermediate_dwp_file(ctx, dwp_output, intermediate_dwp_count)
                intermediate_dwp_count += 1
                packager["outputs"].append(intermediate_output)
                packager["arguments"].add("-o", intermediate_output)
                ctx.actions.run(
                    mnemonic = "CcGenerateIntermediateDwp",
                    tools = packager["tools"],
                    executable = packager["executable"],
                    arguments = [packager["arguments"]],
                    inputs = packager["inputs"],
                    outputs = packager["outputs"],
                )
                intermediate_outputs.append(intermediate_output)
        else:
            return packagers[0]

    # This is to fix buildifier errors, even though we should never reach this part of the code.
    return None

def _create_debug_packager_actions(ctx, cc_toolchain, dwp_output, dwo_files):
    # No inputs? Just generate a trivially empty .dwp.
    #
    # Note this condition automatically triggers for any build where fission is disabled.
    # Because rules referencing .dwp targets may be invoked with or without fission, we need
    # to support .dwp generation even when fission is disabled. Since no actual functionality
    # is expected then, an empty file is appropriate.
    dwo_files_list = dwo_files.to_list()
    if len(dwo_files_list) == 0:
        ctx.actions.write(dwp_output, "", False)
        return

    # We apply a hierarchical action structure to limit the maximum number of inputs to any
    # single action.
    #
    # While the dwp tool consumes .dwo files, it can also consume intermediate .dwp files,
    # allowing us to split a large input set into smaller batches of arbitrary size and order.
    # Aside from the parallelism performance benefits this offers, this also reduces input
    # size requirements: if a.dwo, b.dwo, c.dwo, and e.dwo are each 1 KB files, we can apply
    # two intermediate actions DWP(a.dwo, b.dwo) --> i1.dwp and DWP(c.dwo, e.dwo) --> i2.dwp.
    # When we then apply the final action DWP(i1.dwp, i2.dwp) --> finalOutput.dwp, the inputs
    # to this action will usually total far less than 4 KB.
    #
    # The actions form an n-ary tree with n == MAX_INPUTS_PER_DWP_ACTION. The tree is fuller
    # at the leaves than the root, but that both increases parallelism and reduces the final
    # action's input size.
    packager = _create_intermediate_dwp_packagers(ctx, dwp_output, cc_toolchain, cc_toolchain.dwp_files(), dwo_files_list, 1)
    packager["outputs"].append(dwp_output)
    packager["arguments"].add("-o", dwp_output)
    ctx.actions.run(
        mnemonic = "CcGenerateDwp",
        tools = packager["tools"],
        executable = packager["executable"],
        arguments = [packager["arguments"]],
        inputs = packager["inputs"],
        outputs = packager["outputs"],
    )

def _is_stamping_enabled(ctx):
    if ctx.configuration.is_tool_configuration():
        return 0
    stamp = 0
    if hasattr(ctx.attr, "stamp"):
        stamp = ctx.attr.stamp
    return stamp

def _get_non_data_deps(ctx):
    return ctx.attr.srcs + ctx.attr.deps

def _runfiles_function(ctx, dep, linking_statically):
    provider = None
    if CcInfo in dep:
        provider = dep[CcInfo]
    if provider == None:
        return depset()

    return depset(cc_helper.get_dynamic_libraries_for_runtime(provider.linking_context, linking_statically))

def _default_runfiles_function(ctx, dep):
    provider = None
    if DefaultInfo in dep:
        provider = dep[DefaultInfo].default_runfiles
    if provider == None:
        return ctx.runfiles()

    return provider

def _add(ctx, linking_statically):
    runfiles = []
    for dep in _get_non_data_deps(ctx):
        provider = None
        if CcInfo in dep:
            provider = dep[CcInfo]
        if provider != None:
            runfiles.extend(cc_helper.get_dynamic_libraries_for_runtime(provider.linking_context, linking_statically))
    return depset(runfiles)

def _get_file_content(objects):
    result = []
    for obj in objects:
        result.append(obj.short_path)
        result.append("\n")
    return "".join(result)

def _add_transitive_info_providers(ctx, cc_toolchain, cpp_config, feature_configuration, files_to_build, cc_compilation_outputs, compilation_context, libraries, runtime_objects_for_coverage, common):
    instrumented_object_files = cc_compilation_outputs.objects + cc_compilation_outputs.pic_objects
    additional_meta_data = []
    if len(runtime_objects_for_coverage) != 0 and cpp_config.generate_llvm_lcov():
        runtime_objects_list = ctx.actions.declare_file(ctx.label.name + "runtime_objects_list.txt")
        file_content = _get_file_content(runtime_objects_for_coverage)
        ctx.actions.write(output = runtime_objects_list, content = file_content, is_executable = False)
        additional_meta_data = [runtime_objects_list]

    instrumented_files_provider = common.instrumented_files_info_from_compilation_context(
        files = instrumented_object_files,
        with_base_line_coverage = not ctx.attr._is_test,
        compilation_context = compilation_context,
        additional_metadata = additional_meta_data,
    )
    output_groups = cc_helper.build_output_groups_for_emitting_compile_providers(
        cc_compilation_outputs,
        compilation_context,
        cpp_config,
        cc_toolchain,
        feature_configuration,
        ctx,
        False,  # generate_hidden_top_level_group
    )
    cc_info = CcInfo(
        compilation_context = compilation_context,
        cc_native_library_info = cc_helper.collect_native_cc_libraries(deps = ctx.attr.deps, libraries = libraries),
    )
    output_groups["_validation"] = compilation_context.validation_artifacts
    return (cc_info, instrumented_files_provider, output_groups)

def _collect_runfiles(ctx, feature_configuration, cc_toolchain, libraries, cc_library_linking_outputs, linking_mode, transitive_artifacts, link_compile_output_separately, cpp_config):
    # TODO(b/198254254): Add Legacyexternalrunfiles if necessary.
    runtime_objects_for_coverage = []
    builder_artifacts = []
    builder_transitive_artifacts = []

    builder = ctx.runfiles(transitive_files = _add(ctx, linking_mode != _LINKING_DYNAMIC), collect_default = True)
    coverage_runtime_objects_builder = ctx.runfiles(transitive_files = _add(ctx, linking_mode != _LINKING_DYNAMIC))

    runtime_objects_for_coverage.extend(coverage_runtime_objects_builder.files.to_list())
    dynamic_libraries_for_runtime = _get_dynamic_libraries_for_runtime(True, libraries)
    runtime_objects_for_coverage.extend(dynamic_libraries_for_runtime)

    builder_transitive_artifacts.extend(transitive_artifacts.to_list())
    builder_artifacts.extend(dynamic_libraries_for_runtime)

    runfiles_is_static = []
    runfiles_is_not_static = []
    for transitive_info_collection in ctx.attr.data:
        runfiles_is_static.append(ctx.runfiles(transitive_files = _runfiles_function(ctx, transitive_info_collection, True)))
        runfiles_is_not_static.append(ctx.runfiles(transitive_files = _runfiles_function(ctx, transitive_info_collection, False)))
        runtime_objects_for_coverage.extend(_runfiles_function(ctx, transitive_info_collection, True).to_list())
        runtime_objects_for_coverage.extend(_runfiles_function(ctx, transitive_info_collection, False).to_list())

    for dynamic_dep in ctx.attr.dynamic_deps:
        builder = builder.merge(dynamic_dep[DefaultInfo].default_runfiles)

    builder = builder.merge_all(runfiles_is_static + runfiles_is_not_static)
    if linking_mode == _LINKING_DYNAMIC:
        dynamic_runtime_lib = cc_toolchain.dynamic_runtime_lib(feature_configuration = feature_configuration)
        dynamic_runtime_lib_list = dynamic_runtime_lib.to_list()
        builder_transitive_artifacts.extend(dynamic_runtime_lib_list)
        runtime_objects_for_coverage.extend(dynamic_runtime_lib_list)

    if link_compile_output_separately:
        if cc_library_linking_outputs != None and cc_library_linking_outputs.library_to_link != None and cc_library_linking_outputs.library_to_link.dynamic_library != None:
            builder_artifacts.append(cc_library_linking_outputs.library_to_link.dynamic_library)
            runtime_objects_for_coverage.append(cc_library_linking_outputs.library_to_link.dynamic_library)

    # For cc_binary and cc_test rules, there is an implicit dependency on
    # the malloc library package, which is specified by the "malloc" attribute.
    # As the BUILD encyclopedia says, the "malloc" attribute should be ignored
    # if linkshared=1.
    link_shared = _is_link_shared(ctx)
    if not link_shared:
        malloc = _malloc_for_target(ctx, cpp_config)
        builder = builder.merge(_default_runfiles_function(ctx, malloc))
        builder = builder.merge(ctx.runfiles(transitive_files = _runfiles_function(ctx, malloc, linking_mode != _LINKING_DYNAMIC)))
    return (builder.merge(ctx.runfiles(files = builder_artifacts, transitive_files = depset(builder_transitive_artifacts))), runtime_objects_for_coverage)

def _get_target_sub_dir(target_name):
    last_separator = target_name.rfind("/")
    if last_separator == -1:
        return ""
    return target_name[0:last_separator]

def _create_dynamic_libraries_copy_actions(ctx, dynamic_libraries_for_runtime):
    result = []
    for lib in dynamic_libraries_for_runtime:
        # If the binary and the DLL don't belong to the same package or the DLL is a source file,
        # we should copy the DLL to the binary's directory.
        if ctx.label.package != lib.owner.package or ctx.label.workspace_name != lib.owner.workspace_name or lib.is_source:
            target_name = ctx.label.name
            target_sub_dir = _get_target_sub_dir(target_name)
            copy_file_path = lib.basename
            if target_sub_dir != "":
                copy_file_path = target_sub_dir + "/" + copy_file_path
            copy = ctx.actions.declare_file(copy_file_path)
            ctx.actions.symlink(output = copy, target_file = lib, progress_message = "Copying Execution Dynamic Library")
            result.append(copy)
        else:
            # If the library is already in the same directory as the binary, we don't need to copy it,
            # but we still add it to the result.
            result.append(lib)
    return depset(result)

def _get_dynamic_library_for_runtime_or_none(library_to_link, link_statically):
    if library_to_link.dynamic_library == None:
        return None
    if link_statically and (library_to_link.static_library != None or library_to_link.pic_static_library != None):
        return None
    return library_to_link.dynamic_library

def _get_dynamic_libraries_for_runtime(link_statically, libraries):
    dynamic_libraries_for_runtime = []
    for library_to_link in libraries:
        artifact = _get_dynamic_library_for_runtime_or_none(library_to_link, link_statically)
        if artifact != None:
            dynamic_libraries_for_runtime.append(artifact)
    return dynamic_libraries_for_runtime

def _get_providers(ctx, cpp_config):
    results = []
    malloc = _malloc_for_target(ctx, cpp_config)
    if CcInfo in malloc:
        results.append(malloc[CcInfo])
    for dep in ctx.attr.deps:
        if CcInfo in dep:
            results.append(dep[CcInfo])
    return results

def _collect_transitive_dwo_artifacts(cc_compilation_outputs, cc_debug_context, linking_mode, use_pic, lto_backend_artifacts):
    dwo_files = []
    transitive_dwo_files = depset()
    if use_pic:
        dwo_files.extend(cc_compilation_outputs.pic_dwo_files())
    else:
        dwo_files.extend(cc_compilation_outputs.dwo_files())

    if lto_backend_artifacts != None:
        for lto_backend_artifact in lto_backend_artifacts:
            if lto_backend_artifact.dwo_file() != None:
                dwo_files.append(lto_backend_artifact.dwo_file())

    if linking_mode != _LINKING_DYNAMIC:
        if use_pic:
            transitive_dwo_files = cc_debug_context.pic_files
        else:
            transitive_dwo_files = cc_debug_context.files
    return depset(dwo_files, transitive = [transitive_dwo_files])

def _separate_static_and_dynamic_link_libraries(direct_children, can_be_linked_dynamically):
    link_statically_labels = {}
    link_dynamically_labels = {}
    all_children = list(direct_children)
    seen_labels = {}

    # Some of the logic here is a duplicate from cc_shared_library.
    # But some parts are different hence rewriting.
    for i in range(2147483647):
        if i == len(all_children):
            break
        node = all_children[i]
        node_label = str(node.label)
        if node_label in seen_labels:
            continue
        seen_labels[node_label] = True
        if node_label in can_be_linked_dynamically:
            link_dynamically_labels[node_label] = True
        else:
            link_statically_labels[node_label] = True
            all_children.extend(node.children)
    return (link_statically_labels, link_dynamically_labels)

def _get_preloaded_deps_from_dynamic_deps(ctx):
    cc_infos = []
    for dep in ctx.attr.dynamic_deps:
        cc_shared_library_info = dep[CcSharedLibraryInfo]
        preloaded_deps_field = cc_shared_library_info.preloaded_deps
        if preloaded_deps_field != None:
            cc_infos.append(preloaded_deps_field)

    return cc_common.merge_cc_infos(direct_cc_infos = cc_infos, cc_infos = cc_infos).linking_context.linker_inputs.to_list()

def _get_canonical_form(label):
    repository = label.workspace_name
    if repository == "@":
        repository = ""
    return repository + "//" + label.package + ":" + label.name

def _filter_libraries_that_are_linked_dynamically(ctx, cc_linking_context, cpp_config):
    merged_cc_shared_library_infos = merge_cc_shared_library_infos(ctx)
    preloaded_deps = _get_preloaded_deps_from_dynamic_deps(ctx)
    link_once_static_libs_map = build_link_once_static_libs_map(merged_cc_shared_library_infos)
    exports_map = build_exports_map_from_only_dynamic_deps(merged_cc_shared_library_infos)
    static_linker_inputs = []
    graph_structure_aspect_nodes = []
    linker_inputs = cc_linking_context.linker_inputs.to_list()
    for dep in ctx.attr.deps:
        if GraphNodeInfo in dep:
            graph_structure_aspect_nodes.append(dep[GraphNodeInfo])
    malloc_for_target = _malloc_for_target(ctx, cpp_config)
    if GraphNodeInfo in malloc_for_target:
        graph_structure_aspect_nodes.append(malloc_for_target[GraphNodeInfo])

    can_be_linked_dynamically = {}
    for linker_input in linker_inputs:
        owner = str(linker_input.owner)
        if owner in exports_map:
            can_be_linked_dynamically[owner] = True

    (link_statically_labels, link_dynamically_labels) = _separate_static_and_dynamic_link_libraries(graph_structure_aspect_nodes, can_be_linked_dynamically)

    linker_inputs_seen = {}
    for linker_input in linker_inputs:
        stringified_linker_input = cc_helper.stringify_linker_input(linker_input)
        if stringified_linker_input in linker_inputs_seen:
            continue
        linker_inputs_seen[stringified_linker_input] = True
        owner = str(linker_input.owner)
        if owner not in link_dynamically_labels and (owner in link_statically_labels or _get_canonical_form(ctx.label) == owner):
            if owner in link_once_static_libs_map:
                fail(owner + " is already linked statically in " + link_once_static_libs_map[owner] + " but not exported.")
            else:
                static_linker_inputs.append(linker_input)

    rule_impl_debug_files = None
    if cpp_config.experimental_cc_shared_library_debug():
        debug_linker_inputs_file = ["Owner: " + str(ctx.label)]
        static_linker_inputs_and_preloaded_deps = static_linker_inputs + preloaded_deps
        for linker_input in static_linker_inputs_and_preloaded_deps:
            debug_linker_inputs_file.append(str(linker_input.owner))
        link_once_static_libs_debug_file = ctx.actions.declare_file(ctx.label.name + "_link_once_static_libs.txt")
        ctx.actions.write(link_once_static_libs_debug_file, "\n".join(debug_linker_inputs_file), False)
        transitive_debug_files_list = []
        for dep in ctx.attr.dynamic_deps:
            transitive_debug_files_list.append(dep[OutputGroupInfo].rule_impl_debug_files)
        rule_impl_debug_files = depset([link_once_static_libs_debug_file], transitive = transitive_debug_files_list)
    return (cc_common.create_linking_context(linker_inputs = depset(exports_map.values() + static_linker_inputs + preloaded_deps, order = "topological")), rule_impl_debug_files)

def _create_transitive_linking_actions(
        ctx,
        cc_toolchain,
        feature_configuration,
        cpp_config,
        common,
        precompiled_files,
        cc_compilation_outputs,
        additional_linker_inputs,
        cc_linking_outputs,
        compilation_context,
        binary,
        deps_cc_linking_context,
        extra_link_time_libraries_depset,
        link_compile_output_separately,
        linking_mode,
        link_target_type,
        pdb_file,
        win_def_file,
        additional_linkopts):
    cc_compilation_outputs_with_only_objects = cc_common.create_compilation_outputs(objects = None, pic_objects = None)
    deps_cc_info = CcInfo(linking_context = deps_cc_linking_context)
    libraries_for_current_cc_linking_context = []
    if link_compile_output_separately:
        if cc_linking_outputs != None and cc_linking_outputs.library_to_link != None:
            libraries_for_current_cc_linking_context.append(cc_linking_outputs.library_to_link)
    else:
        cc_compilation_outputs_with_only_objects = cc_common.create_compilation_outputs(
            objects = depset(cc_compilation_outputs.objects),
            pic_objects = depset(cc_compilation_outputs.pic_objects),
            lto_compilation_context = cc_compilation_outputs.lto_compilation_context(),
        )

    # Determine the libraries to link in.
    # First libraries from srcs. Shared library artifacts here are substituted with mangled symlink
    # artifacts generated by getDynamicLibraryLink(). This is done to minimize number of -rpath
    # entries during linking process.
    for libs in precompiled_files[:]:
        for artifact in libs:
            if _matches([".so", ".dylib", ".dll", ".ifso", ".tbd", ".lib", ".dll.a"], artifact.basename) or cc_helper.is_shared_library_extension_valid(artifact.basename):
                library_to_link = cc_common.create_library_to_link(
                    actions = ctx.actions,
                    feature_configuration = feature_configuration,
                    cc_toolchain = cc_toolchain,
                    dynamic_library = artifact,
                )
                libraries_for_current_cc_linking_context.append(library_to_link)
            elif _matches([".pic.lo", ".lo", ".lo.lib"], artifact.basename):
                library_to_link = cc_common.create_library_to_link(
                    actions = ctx.actions,
                    feature_configuration = feature_configuration,
                    cc_toolchain = cc_toolchain,
                    static_library = artifact,
                    alwayslink = True,
                )
                libraries_for_current_cc_linking_context.append(library_to_link)
            elif _matches([".a", ".lib", ".pic.a", ".rlib"], artifact.basename) and not _matches([".if.lib"], artifact.basename):
                library_to_link = cc_common.create_library_to_link(
                    actions = ctx.actions,
                    feature_configuration = feature_configuration,
                    cc_toolchain = cc_toolchain,
                    static_library = artifact,
                )
                libraries_for_current_cc_linking_context.append(library_to_link)

    linker_inputs = cc_common.create_linker_input(
        owner = ctx.label,
        libraries = depset(libraries_for_current_cc_linking_context),
        user_link_flags = common.linkopts + additional_linkopts,
        additional_inputs = depset(common.linker_scripts + compilation_context.transitive_compilation_prerequisites().to_list()),
    )
    current_cc_linking_context = cc_common.create_linking_context(linker_inputs = depset([linker_inputs]))

    cc_info_without_extra_link_time_libraries = cc_common.merge_cc_infos(cc_infos = [CcInfo(linking_context = current_cc_linking_context), deps_cc_info])
    extra_link_time_libraries_cc_info = CcInfo(linking_context = cc_common.create_linking_context(linker_inputs = extra_link_time_libraries_depset))
    cc_info = cc_common.merge_cc_infos(cc_infos = [cc_info_without_extra_link_time_libraries, extra_link_time_libraries_cc_info])
    cc_linking_context = cc_info.linking_context

    rule_impl_debug_files = None
    if len(ctx.attr.dynamic_deps) > 0:
        cc_linking_context, rule_impl_debug_files = _filter_libraries_that_are_linked_dynamically(ctx, cc_linking_context, cpp_config)
    link_deps_statically = True
    if linking_mode == _LINKING_DYNAMIC:
        link_deps_statically = False

    cc_linking_outputs = cc_common.link(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        compilation_outputs = cc_compilation_outputs_with_only_objects,
        grep_includes = cc_helper.grep_includes_executable(ctx.attr._grep_includes),
        stamp = _is_stamping_enabled(ctx),
        additional_inputs = additional_linker_inputs,
        linking_contexts = [cc_linking_context],
        name = ctx.label.name,
        use_test_only_flags = ctx.attr._is_test,
        # Note: Current Starlark API supports either dynamic or static linking modes,
        # even though there are more(LEGACY_FULL_STATIC, LEGACY_MOSTLY_STATIC_LIBRARIES) cc_binary
        # only uses dynamic or static modes. So instead of adding more native footprint
        # we can use what is already supported.
        # It is highly unlikely that cc_binary will start using legacy modes,
        # but if in case it does, code needs to be modified to support it.
        link_deps_statically = link_deps_statically,
        test_only_target = cc_helper.is_test_target(ctx) or ctx.attr._is_test,
        output_type = link_target_type,
        main_output = binary,
        never_link = True,
        pdb_file = pdb_file,
        win_def_file = win_def_file,
    )
    cc_launcher_info = cc_internal.create_cc_launcher_info(cc_info = cc_info_without_extra_link_time_libraries, compilation_outputs = cc_compilation_outputs_with_only_objects)
    return (cc_linking_outputs, cc_launcher_info, rule_impl_debug_files)

def _use_pic(ctx, cc_toolchain, cpp_config, feature_configuration):
    if _is_link_shared(ctx):
        return cc_toolchain.needs_pic_for_dynamic_libraries(feature_configuration = feature_configuration)
    return cpp_config.force_pic() or (cc_toolchain.needs_pic_for_dynamic_libraries(feature_configuration = feature_configuration) and ctx.var["COMPILATION_MODE"] != "opt")

def _collect_linking_context(ctx, cpp_config):
    cc_infos = []
    for dep in ctx.attr.deps:
        if CcInfo in dep:
            cc_infos.append(dep[CcInfo])

    if not _is_link_shared(ctx):
        cc_info = None
        malloc_for_target = _malloc_for_target(ctx, cpp_config)
        if CcInfo in malloc_for_target:
            cc_info = malloc_for_target[CcInfo]
        if cc_info != None:
            cc_infos.append(cc_info)

    return cc_common.merge_cc_infos(direct_cc_infos = cc_infos, cc_infos = cc_infos).linking_context

def _malloc_for_target(ctx, cpp_config):
    if cpp_config.custom_malloc != None:
        return ctx.attr._default_malloc
    return ctx.attr.malloc

def _get_link_staticness(ctx, cpp_config):
    linkstatic_attr = None
    if hasattr(ctx.attr, "_linkstatic_explicitly_set") and not ctx.attr._linkstatic_explicitly_set:
        # If we know that linkstatic is not explicitly set, use computed default:
        linkstatic_attr = semantics.get_linkstatic_default(ctx)
    else:
        linkstatic_attr = ctx.attr.linkstatic

    if cpp_config.dynamic_mode() == "FULLY":
        return _LINKING_DYNAMIC
    elif cpp_config.dynamic_mode() == "OFF" or linkstatic_attr:
        return _LINKING_STATIC
    else:
        return _LINKING_DYNAMIC

def _matches(extensions, target):
    for extension in extensions:
        if target.endswith(extension):
            return True
    return False

def _is_link_shared(ctx):
    return hasattr(ctx.attr, "linkshared") and ctx.attr.linkshared

def _report_invalid_options(ctx, cc_toolchain, cpp_config):
    if cpp_config.grte_top() != None and cc_toolchain.sysroot == None:
        fail("The selected toolchain does not support setting --grte_top (it doesn't specify builtin_sysroot).")

def _is_apple_platform(target_cpu):
    if target_cpu in _IOS_SIMULATOR_TARGET_CPUS or target_cpu in _IOS_DEVICE_TARGET_CPUS or target_cpu in _WATCHOS_SIMULATOR_TARGET_CPUS or target_cpu in _WATCHOS_DEVICE_TARGET_CPUS or target_cpu in _TVOS_SIMULATOR_TARGET_CPUS or target_cpu in _TVOS_DEVICE_TARGET_CPUS or target_cpu in _CATALYST_TARGET_CPUS or target_cpu in _MACOS_TARGET_CPUS:
        return True
    return False

def cc_binary_impl(ctx, additional_linkopts):
    """Implementation function of cc_binary rule.

    Do NOT import outside cc_test.

    Args:
      ctx: The Starlark rule context.
      additional_linkopts: Additional linkopts from an external source (e.g. toolchain)

    Returns:
      Appropriate providers for cc_binary/cc_test.
    """
    cc_helper.check_srcs_extensions(ctx, ALLOWED_SRC_FILES, "cc_binary", True)
    common = cc_internal.create_common(ctx = ctx)
    semantics.validate_deps(ctx)

    if len(ctx.attr.dynamic_deps) > 0:
        cc_common.check_experimental_cc_shared_library()
        # TODO(b/198254254): Add a check if linkshared value is explicitly specified.
        # if ctx.attr.linkshared:
        #     fail("Do not use 'linkshared' to build a shared library. Use cc_shared_library instead.")

    # TODO(b/198254254): Fill empty providers if needed.
    cc_toolchain = cc_helper.find_cpp_toolchain(ctx)
    cpp_config = ctx.fragments.cpp
    _report_invalid_options(ctx, cc_toolchain, cpp_config)

    precompiled_files = cc_helper.build_precompiled_files(ctx)
    link_target_type = _EXECUTABLE
    if _is_link_shared(ctx):
        link_target_type = _DYNAMIC_LIBRARY
    is_dynamic_link_type = True
    if link_target_type == _EXECUTABLE:
        is_dynamic_link_type = False
    semantics.validate_attributes(ctx)

    # TODO(b/198254254): Fill in empty providers if needed.
    # If cc_binary includes "linkshared=1" then gcc will be invoked with
    # linkopt "-shared", which causes the result of linking to be a shared library.
    # For linkshared=1 we used to force users to specify the file extension manually, as part of
    # the target name.
    # This is no longer necessary, the toolchain can figure out the correct file extensions.
    target_name = ctx.label.name
    has_legacy_link_shared_name = _is_link_shared(ctx) and (_matches([".so", ".dylib", ".dll"], target_name) or cc_helper.is_shared_library_extension_valid(target_name))
    binary = None
    if has_legacy_link_shared_name:
        binary = ctx.actions.declare_file(target_name)
    else:
        binary = cc_helper.get_linked_artifact(
            ctx = ctx,
            cc_toolchain = cc_toolchain,
            is_dynamic_link_type = is_dynamic_link_type,
        )
    linking_mode = _get_link_staticness(ctx, cpp_config)
    features = ctx.features
    features.append(linking_mode)
    disabled_features = ctx.disabled_features
    if ctx.attr._is_test and cpp_config.incompatible_enable_cc_test_feature:
        features.append("is_cc_test")
        disabled_features.append("legacy_is_cc_test")
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = features,
        unsupported_features = disabled_features,
    )
    compilation_context_deps = [dep[CcInfo].compilation_context for dep in ctx.attr.deps if CcInfo in dep]
    target_malloc = _malloc_for_target(ctx, cpp_config)
    malloc_dep = None
    if CcInfo in target_malloc:
        malloc_dep = target_malloc[CcInfo].compilation_context
    if malloc_dep != None:
        compilation_context_deps.append(malloc_dep)
    if ctx.attr._stl != None:
        compilation_context_deps.append(ctx.attr._stl[CcInfo].compilation_context)

    additional_make_variable_substitutions = cc_helper.get_toolchain_global_make_variables(cc_toolchain)
    additional_make_variable_substitutions.update(cc_helper.get_cc_flags_make_variable(ctx, common, cc_toolchain))

    (compilation_context, compilation_outputs) = cc_common.compile(
        name = ctx.label.name,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        user_compile_flags = cc_helper.get_copts(ctx, common, feature_configuration, additional_make_variable_substitutions),
        defines = common.defines,
        local_defines = common.local_defines,
        loose_includes = common.loose_include_dirs,
        system_includes = common.system_include_dirs,
        private_hdrs = common.private_hdrs,
        public_hdrs = common.public_hdrs,
        copts_filter = common.copts_filter,
        srcs = common.srcs,
        compilation_contexts = compilation_context_deps,
        grep_includes = cc_helper.grep_includes_executable(ctx.attr._grep_includes),
        code_coverage_enabled = cc_helper.is_code_coverage_enabled(ctx = ctx),
        hdrs_checking_mode = semantics.determine_headers_checking_mode(ctx),
    )
    precompiled_file_objects = cc_common.create_compilation_outputs(
        objects = depset(precompiled_files[0]),  # objects
        pic_objects = depset(precompiled_files[1]),  # pic_objects
    )
    cc_compilation_outputs = cc_common.merge_compilation_outputs(compilation_outputs = [compilation_outputs, precompiled_file_objects])
    additional_linker_inputs = ctx.files.additional_linker_inputs

    # Allows the dynamic library generated for code of test targets to be linked separately.
    link_compile_output_separately = ctx.attr._is_test and linking_mode == _LINKING_DYNAMIC and cpp_config.dynamic_mode() == "DEFAULT" and ("dynamic_link_test_srcs" in ctx.features)

    # When linking the object files directly into the resulting binary, we do not need
    # library-level link outputs; thus, we do not let CcCompilationHelper produce link outputs
    # (either shared object files or archives) for a non-library link type [*], and add
    # the object files explicitly in determineLinkerArguments.
    #
    # When linking the object files into their own library, we want CcCompilationHelper to
    # take care of creating the library link outputs for us, so we need to set the link
    # type to STATIC_LIBRARY.
    #
    # [*] The only library link type is STATIC_LIBRARY. EXECUTABLE specifies a normal
    # cc_binary output, while DYNAMIC_LIBRARY is a cc_binary rules that produces an
    # output matching a shared object, for example cc_binary(name="foo.so", ...) on linux.
    cc_linking_outputs = None
    if link_compile_output_separately and (len(cc_compilation_outputs.objects) != 0 or len(cc_compilation_outputs.pic_objects) != 0):
        (linking_context, cc_linking_outputs) = cc_common.create_linking_context_from_compilation_outputs(
            actions = ctx.actions,
            feature_configuration = feature_configuration,
            cc_toolchain = cc_toolchain,
            compilation_outputs = cc_compilation_outputs,
            name = ctx.label.name,
            grep_includes = cc_helper.grep_includes_executable(ctx.attr._grep_includes),
            linking_contexts = cc_helper.get_linking_contexts_from_deps([_malloc_for_target(ctx, cpp_config)]) + cc_helper.get_linking_contexts_from_deps(ctx.attr.deps),
            stamp = _is_stamping_enabled(ctx),
            alwayslink = True,
        )

    is_static_mode = linking_mode != _LINKING_DYNAMIC
    deps_cc_linking_context = _collect_linking_context(ctx, cpp_config)
    generated_def_file = None
    win_def_file = None
    if _is_link_shared(ctx):
        if cc_common.is_enabled(feature_configuration = feature_configuration, feature_name = "targets_windows"):
            # Make copy of a list, to avoid mutating frozen values.
            object_files = list(cc_compilation_outputs.objects)
            for linker_input in deps_cc_linking_context.linker_inputs.to_list():
                for library in linker_input.libraries:
                    if is_static_mode or (library.dynamic_library == None and library.interface_library == None):
                        if library.pic_static_library != None:
                            if library.pic_objects != None:
                                object_files.extend(library.pic_objects)
                        elif library.static_library != None:
                            if library.objects != None:
                                object_files.extend(library.objects)

            def_parser = ctx.file._def_parser
            if def_parser != None:
                generated_def_file = cc_helper.generate_def_file(ctx, def_parser, object_files, binary.basename)
            custom_win_def_file = ctx.file.win_def_file
            win_def_file = cc_helper.get_windows_def_file_for_linking(ctx, custom_win_def_file, generated_def_file, feature_configuration)

    use_pic = _use_pic(ctx, cc_toolchain, cpp_config, feature_configuration)

    # On Windows, if GENERATE_PDB_FILE feature is enabled
    # then a pdb file will be built along with the executable.
    pdb_file = None
    if cc_common.is_enabled(feature_configuration = feature_configuration, feature_name = "generate_pdb_file"):
        pdb_file = ctx.actions.declare_file(_strip_extension(binary) + ".pdb", sibling = binary)

    extra_link_time_libraries = deps_cc_linking_context.extra_link_time_libraries()
    linker_inputs_extra = depset()
    runtime_libraries_extra = depset()
    if extra_link_time_libraries != None:
        linker_inputs_extra, runtime_libraries_extra = extra_link_time_libraries.build_libraries(ctx = ctx, static_mode = linking_mode != _LINKING_DYNAMIC, for_dynamic_library = _is_link_shared(ctx))

    cc_linking_outputs_binary, cc_launcher_info, rule_impl_debug_files = _create_transitive_linking_actions(
        ctx,
        cc_toolchain,
        feature_configuration,
        cpp_config,
        common,
        precompiled_files,
        cc_compilation_outputs,
        additional_linker_inputs,
        cc_linking_outputs,
        compilation_context,
        binary,
        deps_cc_linking_context,
        linker_inputs_extra,
        link_compile_output_separately,
        linking_mode,
        link_target_type,
        pdb_file,
        win_def_file,
        additional_linkopts,
    )

    cc_linking_outputs_binary_library = cc_linking_outputs_binary.library_to_link
    libraries = []
    if _is_link_shared(ctx) and cc_linking_outputs_binary_library != None:
        libraries.append(cc_linking_outputs_binary_library)

    # Also add all shared libraries from srcs.
    for library in precompiled_files[6]:  #shared_libraries
        library_to_link = cc_common.create_library_to_link(
            actions = ctx.actions,
            feature_configuration = feature_configuration,
            cc_toolchain = cc_toolchain,
            dynamic_library = library,
            #dynamic_library_symlink_path = library.short_path,
        )
        libraries.append(library_to_link)

    files_to_build_list = [binary]

    # Create the stripped binary but don't add it to filesToBuild; it's only built when requested.
    stripped_file = ctx.outputs.stripped_binary
    cc_helper.create_strip_action(ctx, cc_toolchain, cpp_config, binary, stripped_file, feature_configuration)
    dwo_files = _collect_transitive_dwo_artifacts(
        cc_compilation_outputs,
        cc_helper.merge_cc_debug_contexts(cc_compilation_outputs, _get_providers(ctx, cpp_config)),
        linking_mode,
        use_pic,
        cc_linking_outputs_binary.all_lto_artifacts(),
    )
    dwp_file = ctx.outputs.dwp_file
    _create_debug_packager_actions(ctx, cc_toolchain, dwp_file, dwo_files)
    explicit_dwp_file = dwp_file
    if not cc_helper.should_create_per_object_debug_info(feature_configuration, cpp_config):
        explicit_dwp_file = None
    elif ctx.attr._is_test and linking_mode != _LINKING_DYNAMIC and cpp_config.build_test_dwp():
        files_to_build_list.append(dwp_file)

    # If the binary is linked dynamically and COPY_DYNAMIC_LIBRARIES_TO_BINARY is enabled, collect
    # all the dynamic libraries we need at runtime. Then copy these libraries next to the binary.
    copied_runtime_dynamic_libraries = None
    if cc_common.is_enabled(feature_configuration = feature_configuration, feature_name = "copy_dynamic_libraries_to_binary"):
        linker_inputs = deps_cc_linking_context.linker_inputs.to_list()
        libraries = []
        for linker_input in linker_inputs:
            libraries.extend(linker_input.libraries)
        copied_runtime_dynamic_libraries = _create_dynamic_libraries_copy_actions(ctx, _get_dynamic_libraries_for_runtime(is_static_mode, libraries))

    # TODO(b/198254254)(bazel-team): Do we need to put original shared libraries (along with
    # mangled symlinks) into the RunfilesSupport object? It does not seem
    # logical since all symlinked libraries will be linked anyway and would
    # not require manual loading but if we do, then we would need to collect
    # their names and use a different constructor below.

    files_to_build = depset(files_to_build_list)
    transitive_artifacts_list = [files_to_build, runtime_libraries_extra]
    if cc_common.is_enabled(feature_configuration = feature_configuration, feature_name = "copy_dynamic_libraries_to_binary"):
        transitive_artifacts_list.append(copied_runtime_dynamic_libraries)
    transitive_artifacts = depset(transitive = transitive_artifacts_list)

    runtime_objects_for_coverage = [binary]
    runfiles, new_runtime_objects_for_coverage = _collect_runfiles(
        ctx,
        feature_configuration,
        cc_toolchain,
        libraries,
        cc_linking_outputs,
        linking_mode,
        transitive_artifacts,
        link_compile_output_separately,
        cpp_config,
    )
    runtime_objects_for_coverage.extend(new_runtime_objects_for_coverage)
    (cc_info, instrumented_files_provider, output_groups) = _add_transitive_info_providers(
        ctx,
        cc_toolchain,
        cpp_config,
        feature_configuration,
        files_to_build,
        cc_compilation_outputs,
        compilation_context,
        libraries,
        runtime_objects_for_coverage,
        common,
    )
    if _is_apple_platform(cc_toolchain.cpu) and ctx.attr._is_test:
        # TODO(b/198254254): Add ExecutionInfo.
        execution_info = None

    # If PDB file is generated by the link action, we add it to pdb_file output group
    if pdb_file != None:
        output_groups["pdb_file"] = depset([pdb_file])
    if generated_def_file != None:
        output_groups["def_file"] = depset([generated_def_file])

    if rule_impl_debug_files != None:
        output_groups["rule_impl_debug_files"] = rule_impl_debug_files

    if cc_linking_outputs_binary_library != None:
        # For consistency and readability.
        library_to_link = cc_linking_outputs_binary_library
        dynamic_library_for_linking = None
        if library_to_link.interface_library != None:
            if library_to_link.resolved_symlink_interface_library != None:
                dynamic_library_for_linking = library_to_link.resolved_symlink_interface_library
            else:
                dynamic_library_for_linking = library_to_link.interface_library
        elif library_to_link.dynamic_library != None:
            if library_to_link.resolved_symlink_dynamic_library != None:
                dynamic_library_for_linking = library_to_link.resolved_symlink_dynamic_library
            else:
                dynamic_library_for_linking = library_to_link.dynamic_library
        if dynamic_library_for_linking != None:
            output_groups["interface_library"] = depset([dynamic_library_for_linking])

    if copied_runtime_dynamic_libraries != None:
        output_groups["runtime_dynamic_libraries"] = copied_runtime_dynamic_libraries

    # TODO(b/198254254): SetRunfilesSupport if needed.
    debug_package_info = DebugPackageInfo(
        target_label = ctx.label,
        stripped_file = stripped_file,
        unstripped_file = binary,
        dwp_file = explicit_dwp_file,
    )
    binary_info = struct(
        files = files_to_build,
        runfiles = runfiles,
        executable = binary,
    )
    result = [
        cc_info,
        instrumented_files_provider,
        debug_package_info,
        OutputGroupInfo(**output_groups),
    ]
    if "fully_static_link" in ctx.features:
        result.append(StaticallyLinkedMarkerInfo(is_linked_statically = True))
    if cc_launcher_info != None:
        result.append(cc_launcher_info)
    return binary_info, cc_info, result

ALLOWED_SRC_FILES = []
ALLOWED_SRC_FILES.extend(cc_helper.extensions.CC_SOURCE)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.C_SOURCE)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.CC_HEADER)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.ASSESMBLER_WITH_C_PREPROCESSOR)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.ASSEMBLER)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.ARCHIVE)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.PIC_ARCHIVE)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.ALWAYSLINK_LIBRARY)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.ALWAYSLINK_PIC_LIBRARY)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.SHARED_LIBRARY)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.OBJECT_FILE)
ALLOWED_SRC_FILES.extend(cc_helper.extensions.PIC_OBJECT_FILE)

def _impl(ctx):
    binary_info, cc_info, other_providers = cc_binary_impl(ctx, [])

    # We construct DefaultInfo here, as other cc_binary-like rules (cc_test) need
    # a different DefaultInfo.
    default_info = DefaultInfo(
        files = binary_info.files,
        runfiles = binary_info.runfiles,
        executable = binary_info.executable,
    )
    other_providers.append(default_info)
    if ctx.fragments.cpp.enable_legacy_cc_provider():
        # buildifier: disable=rule-impl-return
        return struct(
            cc = cc_internal.create_cc_provider(cc_info = cc_info),
            providers = other_providers,
        )
    else:
        return other_providers

def make_cc_binary(cc_binary_attrs, **kwargs):
    return rule(
        implementation = _impl,
        attrs = cc_binary_attrs,
        outputs = {
            "stripped_binary": "%{name}.stripped",
            "dwp_file": "%{name}.dwp",
        },
        fragments = ["google_cpp", "cpp"],
        exec_groups = {
            "cpp_link": exec_group(copy_from_rule = True),
        },
        toolchains = cc_helper.use_cpp_toolchain(),
        incompatible_use_toolchain_transition = True,
        executable = True,
        **kwargs
    )
