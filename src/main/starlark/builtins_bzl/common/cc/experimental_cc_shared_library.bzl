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

"""This is an experimental implementation of cc_shared_library.

We may change the implementation at any moment or even delete this file. Do not
rely on this. It requires bazel >1.2  and passing the flag
--experimental_cc_shared_library
"""

load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/cc/semantics.bzl", "semantics")
load(":common/proto/proto_info.bzl", "ProtoInfo")
load(":common/cc/cc_info.bzl", "CcInfo")

cc_common = _builtins.toplevel.cc_common

# TODO(#5200): Add export_define to library_to_link and cc_library

# Add this as a tag to any target that can be linked by more than one
# cc_shared_library because it doesn't have static initializers or anything
# else that may cause issues when being linked more than once. This should be
# used sparingly after making sure it's safe to use.
LINKABLE_MORE_THAN_ONCE = "LINKABLE_MORE_THAN_ONCE"

# Add this as a tag to any static lib target that doesn't export any symbols,
# thus can be statically linked more than once. This is useful in some cases,
# for example, a static lib has a constructor that needs to be run during
# loading time of the shared lib that has it linked into, which is how the
# code gets called by the OS. This static lib might need to be linked as a
# whole archive dep for multiple shared libs, otherwise this static lib will
# be dropped by the linker since there are no incoming symbol references.
NO_EXPORTING = "NO_EXPORTING"

GraphNodeInfo = provider(
    "Nodes in the graph of shared libraries.",
    fields = {
        "children": "Other GraphNodeInfo from dependencies of this target",
        "label": "Label of the target visited",
        "linkable_more_than_once": "Linkable into more than a single cc_shared_library",
        "no_exporting": "The static lib doesn't export any symbols so don't export it",
    },
)
CcSharedLibraryInfo = provider(
    "Information about a cc shared library.",
    fields = {
        "dynamic_deps": "All shared libraries depended on transitively",
        "exports": "cc_libraries that are linked statically and exported",
        "link_once_static_libs": "All libraries linked statically into this library that should " +
                                 "only be linked once, e.g. because they have static " +
                                 "initializers. If we try to link them more than once, " +
                                 "we will throw an error",
        "linker_input": "the resulting linker input artifact for the shared library",
    },
)

# For each target, find out whether it should be linked statically or
# dynamically.
def _separate_static_and_dynamic_link_libraries(
        direct_children,
        can_be_linked_dynamically):
    node = None
    all_children = list(direct_children)
    targets_to_be_linked_statically_map = {}
    targets_to_be_linked_dynamically_set = {}

    seen_labels = {}

    # Horrible I know. Perhaps Starlark team gives me a way to prune a tree.
    for i in range(2147483647):
        if i == len(all_children):
            break

        node = all_children[i]
        node_label = str(node.label)

        if node_label in seen_labels:
            continue
        seen_labels[node_label] = True

        if node_label in can_be_linked_dynamically:
            targets_to_be_linked_dynamically_set[node_label] = True
        else:
            targets_to_be_linked_statically_map[node_label] = node.linkable_more_than_once
            all_children.extend(node.children)

    return (targets_to_be_linked_statically_map, targets_to_be_linked_dynamically_set)

def _create_linker_context(ctx, linker_inputs):
    return cc_common.create_linking_context(
        linker_inputs = depset(linker_inputs, order = "topological"),
    )

def _merge_cc_shared_library_infos(ctx):
    dynamic_deps = []
    transitive_dynamic_deps = []
    for dep in ctx.attr.dynamic_deps:
        dynamic_dep_entry = (
            dep[CcSharedLibraryInfo].exports,
            dep[CcSharedLibraryInfo].linker_input,
            dep[CcSharedLibraryInfo].link_once_static_libs,
        )
        dynamic_deps.append(dynamic_dep_entry)
        transitive_dynamic_deps.append(dep[CcSharedLibraryInfo].dynamic_deps)

    return depset(direct = dynamic_deps, transitive = transitive_dynamic_deps)

def _build_exports_map_from_only_dynamic_deps(merged_shared_library_infos):
    exports_map = {}
    for entry in merged_shared_library_infos.to_list():
        exports = entry[0]
        linker_input = entry[1]
        for export in exports:
            if export in exports_map:
                fail("Two shared libraries in dependencies export the same symbols. Both " +
                     exports_map[export].libraries[0].dynamic_library.short_path +
                     " and " + linker_input.libraries[0].dynamic_library.short_path +
                     " export " + export)
            exports_map[export] = linker_input
    return exports_map

# The map points from the target that can only be linked once to the
# cc_shared_library target that already links it.
def _build_link_once_static_libs_map(merged_shared_library_infos):
    link_once_static_libs_map = {}
    for entry in merged_shared_library_infos.to_list():
        link_once_static_libs = entry[2]
        linker_input = entry[1]
        for static_lib in link_once_static_libs:
            if static_lib in link_once_static_libs_map:
                fail("Two shared libraries in dependencies link the same " +
                     " library statically. Both " + link_once_static_libs_map[static_lib] +
                     " and " + str(linker_input.owner) +
                     " link statically" + static_lib)
            link_once_static_libs_map[static_lib] = str(linker_input.owner)
    return link_once_static_libs_map

def _is_dynamic_only(library_to_link):
    if (library_to_link.static_library == None and
        library_to_link.pic_static_library == None and
        (library_to_link.objects == None or len(library_to_link.objects) == 0) and
        (library_to_link.pic_objects == None or len(library_to_link.pic_objects) == 0)):
        return True
    return False

def _wrap_static_library_with_alwayslink(ctx, feature_configuration, cc_toolchain, linker_input):
    new_libraries_to_link = []
    for old_library_to_link in linker_input.libraries:
        if _is_dynamic_only(old_library_to_link):
            new_libraries_to_link.append(old_library_to_link)
            continue
        new_library_to_link = cc_common.create_library_to_link(
            actions = ctx.actions,
            feature_configuration = feature_configuration,
            cc_toolchain = cc_toolchain,
            static_library = old_library_to_link.static_library,
            objects = old_library_to_link.objects,
            pic_static_library = old_library_to_link.pic_static_library,
            pic_objects = old_library_to_link.pic_objects,
            alwayslink = True,
        )
        new_libraries_to_link.append(new_library_to_link)

    return cc_common.create_linker_input(
        owner = linker_input.owner,
        libraries = depset(direct = new_libraries_to_link),
        user_link_flags = depset(direct = linker_input.user_link_flags),
        additional_inputs = depset(direct = linker_input.additional_inputs),
    )

def _check_if_target_under_path(value, pattern):
    if pattern.workspace_name != value.workspace_name:
        return False
    if pattern.name == "__pkg__":
        return pattern.package == value.package
    if pattern.name == "__subpackages__":
        return _same_package_or_above(pattern, value)

    return pattern.package == value.package and pattern.name == value.name

def _check_if_target_should_be_exported_without_filter(target, current_label):
    return _check_if_target_should_be_exported_with_filter(target, current_label, None)

def _check_if_target_should_be_exported_with_filter(target, current_label, exports_filter):
    should_be_exported = False
    if exports_filter == None:
        should_be_exported = True
    else:
        for export_filter in exports_filter:
            export_filter_label = current_label.relative(export_filter)
            if _check_if_target_under_path(target, export_filter_label):
                return True

    return False

def _filter_inputs(
        ctx,
        feature_configuration,
        cc_toolchain,
        deps,
        transitive_exports,
        link_once_static_libs_map):
    linker_inputs = []
    curr_link_once_static_libs_set = {}

    graph_structure_aspect_nodes = []
    dependency_linker_inputs = []
    direct_exports = {}
    for export in deps:
        direct_exports[str(export.label)] = True
        dependency_linker_inputs.extend(export[CcInfo].linking_context.linker_inputs.to_list())
        graph_structure_aspect_nodes.append(export[GraphNodeInfo])

    can_be_linked_dynamically = {}
    for linker_input in dependency_linker_inputs:
        owner = str(linker_input.owner)
        if owner in transitive_exports:
            can_be_linked_dynamically[owner] = True

    # The targets_to_be_linked_statically_map points to whether the target to
    # be linked statically can be linked more than once.
    (targets_to_be_linked_statically_map, targets_to_be_linked_dynamically_set) = _separate_static_and_dynamic_link_libraries(
        graph_structure_aspect_nodes,
        can_be_linked_dynamically,
    )

    # We keep track of precompiled_only_dynamic_libraries, so that we can add
    # them to runfiles.
    precompiled_only_dynamic_libraries = []
    exports = {}
    linker_inputs_seen = {}

    # We use this dictionary to give an error if a target containing only
    # precompiled dynamic libraries is placed directly in roots. If such a
    # precompiled dynamic library is needed it would be because a target in the
    # parallel cc_library graph actually needs it. Therefore the precompiled
    # dynamic library should be made a dependency of that cc_library instead.
    dynamic_only_roots = {}
    linked_statically_but_not_exported = {}
    for linker_input in dependency_linker_inputs:
        stringified_linker_input = cc_helper.stringify_linker_input(linker_input)
        if stringified_linker_input in linker_inputs_seen:
            continue
        linker_inputs_seen[stringified_linker_input] = True
        owner = str(linker_input.owner)
        if owner in targets_to_be_linked_dynamically_set:
            # Link the library in this iteration dynamically,
            # transitive_exports contains the artifacts produced by a
            # cc_shared_library
            linker_inputs.append(transitive_exports[owner])
        elif owner in targets_to_be_linked_statically_map:
            if owner in link_once_static_libs_map:
                # We are building a dictionary that will allow us to give
                # proper errors for libraries that have been linked multiple
                # times elsewhere but haven't been exported. The values in the
                # link_once_static_libs_map dictionary are the
                # cc_shared_library targets. In this iteration we know of at
                # least one target (i.e. owner) which is being linked
                # statically by the cc_shared_library
                # link_once_static_libs_map[owner] but is not being exported
                linked_statically_but_not_exported.setdefault(link_once_static_libs_map[owner], []).append(owner)

            is_direct_export = owner in direct_exports
            dynamic_only_libraries = []
            static_libraries = []
            for library in linker_input.libraries:
                if _is_dynamic_only(library):
                    dynamic_only_libraries.append(library)
                else:
                    static_libraries.append(library)

            if len(dynamic_only_libraries):
                precompiled_only_dynamic_libraries.extend(dynamic_only_libraries)
                if not len(static_libraries):
                    if is_direct_export:
                        dynamic_only_roots[owner] = True
                    linker_inputs.append(linker_input)
                    continue
            if len(static_libraries) and owner in dynamic_only_roots:
                dynamic_only_roots.pop(owner)

            linker_input_to_be_linked_statically = linker_input
            if is_direct_export:
                linker_input_to_be_linked_statically = _wrap_static_library_with_alwayslink(
                    ctx,
                    feature_configuration,
                    cc_toolchain,
                    linker_input,
                )
            elif _check_if_target_should_be_exported_with_filter(
                linker_input.owner,
                ctx.label,
                ctx.attr.exports_filter,
            ):
                exports[owner] = True

            linker_inputs.append(linker_input_to_be_linked_statically)

            if not targets_to_be_linked_statically_map[owner]:
                curr_link_once_static_libs_set[owner] = True

    if dynamic_only_roots:
        message = ("Do not place libraries which only contain a " +
                   "precompiled dynamic library in roots. The following " +
                   "libraries only have precompiled dynamic libraries:\n")
        for dynamic_only_root in dynamic_only_roots:
            message += dynamic_only_root + "\n"
        fail(message)

    _throw_linked_but_not_exported_errors(linked_statically_but_not_exported)
    return (exports, linker_inputs, curr_link_once_static_libs_set.keys(), precompiled_only_dynamic_libraries)

def _throw_linked_but_not_exported_errors(error_libs_dict):
    if not error_libs_dict:
        return

    error_builder = ["The following libraries were linked statically by different cc_shared_libraries but not exported:\n"]
    for cc_shared_library_target, error_libs in error_libs_dict.items():
        error_builder.append("cc_shared_library %s:\n" % str(cc_shared_library_target))
        for error_lib in error_libs:
            error_builder.append("  \"%s\",\n" % str(error_lib))

    error_builder.append("If you are sure that the previous libraries are exported by the cc_shared_libraries because:\n")
    error_builder.append("  1. You have visibility declarations in the source code\n")
    error_builder.append("  2. Or you are passing a visibility script to the linker to export symbols from them\n")
    error_builder.append("then add those libraries to roots or exports_filter for each cc_shared_library.\n")

    fail("".join(error_builder))

def _same_package_or_above(label_a, label_b):
    if label_a.workspace_name != label_b.workspace_name:
        return False
    package_a_tokenized = label_a.package.split("/")
    package_b_tokenized = label_b.package.split("/")
    if len(package_b_tokenized) < len(package_a_tokenized):
        return False

    if package_a_tokenized[0] != "":
        for i in range(len(package_a_tokenized)):
            if package_a_tokenized[i] != package_b_tokenized[i]:
                return False

    return True

def _get_deps(ctx):
    if len(ctx.attr.deps) and len(ctx.attr.roots):
        fail(
            "You are using the attribute 'roots' and 'deps'. 'deps' is the " +
            "new name for the attribute 'roots'. The attribute 'roots' will be" +
            "removed in the future",
            attr = "roots",
        )

    deps = ctx.attr.deps
    if not len(deps):
        deps = ctx.attr.roots

    if len(deps) == 0:
        fail(
            "'cc_shared_library' must have at least one dependency in 'deps' (or 'roots')",
            attr = "deps",
        )

    return deps

def _cc_shared_library_impl(ctx):
    semantics.check_experimental_cc_shared_library(ctx)

    if not cc_common.check_experimental_cc_shared_library():
        if len(ctx.attr.static_deps):
            fail(
                "This attribute is a no-op and its usage" +
                " is forbidden after cc_shared_library is no longer experimental. " +
                "Remove it from every cc_shared_library target",
                attr = "static_deps",
            )
        if len(ctx.attr.roots):
            fail(
                "This attribute has been renamed to 'deps'. Simply rename the" +
                " attribute on the target.",
                attr = "roots",
            )

    deps = _get_deps(ctx)

    cc_toolchain = cc_helper.find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    merged_cc_shared_library_info = _merge_cc_shared_library_infos(ctx)
    exports_map = _build_exports_map_from_only_dynamic_deps(merged_cc_shared_library_info)
    for export in deps:
        # Do not check for overlap between targets matched by the current
        # rule's exports_filter and what is in exports_map. A library in roots
        # will have to be linked in statically into the current rule with 100%
        # guarantee and it will also have to be exported. Therefore, we must
        # check it's not already exported by a different shared library. On the
        # other hand, a library in the transitive closure of the current rule
        # may be matched by the exports_filter but if it's already exported by
        # a dynamic_dep then it won't be linked statically (therefore not give
        # an error either) in the current target. The rule will intentionally
        # not throw an error in these cases.
        if str(export.label) in exports_map:
            fail("Trying to export a library already exported by a different shared library: " +
                 str(export.label))

        _check_if_target_should_be_exported_without_filter(export.label, ctx.label)

    link_once_static_libs_map = _build_link_once_static_libs_map(merged_cc_shared_library_info)

    (exports, linker_inputs, curr_link_once_static_libs_set, precompiled_only_dynamic_libraries) = _filter_inputs(
        ctx,
        feature_configuration,
        cc_toolchain,
        deps,
        exports_map,
        link_once_static_libs_map,
    )

    linking_context = _create_linker_context(ctx, linker_inputs)

    user_link_flags = []
    for user_link_flag in ctx.attr.user_link_flags:
        user_link_flags.append(ctx.expand_location(user_link_flag, targets = ctx.attr.additional_linker_inputs))

    main_output = None
    if ctx.attr.shared_lib_name:
        main_output = ctx.actions.declare_file(ctx.attr.shared_lib_name)

    win_def_file = None
    if cc_common.is_enabled(feature_configuration = feature_configuration, feature_name = "targets_windows"):
        object_files = []
        for linker_input in linking_context.linker_inputs.to_list():
            for library in linker_input.libraries:
                if library.pic_static_library != None:
                    if library.pic_objects != None:
                        object_files.extend(library.pic_objects)
                elif library.static_library != None:
                    if library.objects != None:
                        object_files.extend(library.objects)

        def_parser = ctx.file._def_parser

        generated_def_file = None
        if def_parser != None:
            generated_def_file = cc_helper.generate_def_file(ctx, def_parser, object_files, ctx.label.name)
        custom_win_def_file = ctx.file.win_def_file
        win_def_file = cc_helper.get_windows_def_file_for_linking(ctx, custom_win_def_file, generated_def_file, feature_configuration)

    additional_inputs = []
    additional_inputs.extend(ctx.files.additional_linker_inputs)
    linking_outputs = cc_common.link(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        linking_contexts = [linking_context],
        user_link_flags = user_link_flags,
        additional_inputs = additional_inputs,
        grep_includes = ctx.executable._grep_includes,
        name = ctx.label.name,
        output_type = "dynamic_library",
        main_output = main_output,
        win_def_file = win_def_file,
    )

    runfiles_files = []
    if linking_outputs.library_to_link.resolved_symlink_dynamic_library != None:
        runfiles_files.append(linking_outputs.library_to_link.resolved_symlink_dynamic_library)

    # This is different to cc_binary(linkshared=1). Bazel never handles the
    # linking implicitly for a cc_binary(linkshared=1) but it does so for a cc_shared_library,
    # for which it will use the symlink in the solib directory. If we don't add it, a dependent
    # linked against it would fail.
    runfiles_files.append(linking_outputs.library_to_link.dynamic_library)
    runfiles = ctx.runfiles(
        files = runfiles_files,
    )
    transitive_debug_files = []
    for dep in ctx.attr.dynamic_deps:
        runfiles = runfiles.merge(dep[DefaultInfo].data_runfiles)
        transitive_debug_files.append(dep[OutputGroupInfo].rule_impl_debug_files)

    precompiled_only_dynamic_libraries_runfiles = []
    for precompiled_dynamic_library in precompiled_only_dynamic_libraries:
        # precompiled_dynamic_library.dynamic_library could be None if the library to link just contains
        # an interface library which is valid if the actual library is obtained from the system.
        if precompiled_dynamic_library.dynamic_library != None:
            precompiled_only_dynamic_libraries_runfiles.append(precompiled_dynamic_library.dynamic_library)

    runfiles = runfiles.merge(ctx.runfiles(files = precompiled_only_dynamic_libraries_runfiles))

    for export in deps:
        export_label = str(export.label)
        if GraphNodeInfo in export and export[GraphNodeInfo].no_exporting:
            if export_label in curr_link_once_static_libs_set:
                curr_link_once_static_libs_set.remove(export_label)
            continue
        exports[export_label] = True

    debug_files = []
    exports_debug_file = ctx.actions.declare_file(ctx.label.name + "_exports.txt")
    ctx.actions.write(content = "\n".join(["Owner:" + str(ctx.label)] + exports.keys()), output = exports_debug_file)

    link_once_static_libs_debug_file = ctx.actions.declare_file(ctx.label.name + "_link_once_static_libs.txt")
    ctx.actions.write(content = "\n".join(["Owner:" + str(ctx.label)] + curr_link_once_static_libs_set), output = link_once_static_libs_debug_file)

    debug_files.append(exports_debug_file)
    debug_files.append(link_once_static_libs_debug_file)

    if not ctx.fragments.cpp.experimental_link_static_libraries_once():
        curr_link_once_static_libs_set = {}

    library = []
    if linking_outputs.library_to_link.resolved_symlink_dynamic_library != None:
        library.append(linking_outputs.library_to_link.resolved_symlink_dynamic_library)
    else:
        library.append(linking_outputs.library_to_link.dynamic_library)

    interface_library = []
    if linking_outputs.library_to_link.resolved_symlink_interface_library != None:
        interface_library.append(linking_outputs.library_to_link.resolved_symlink_interface_library)
    elif linking_outputs.library_to_link.interface_library != None:
        interface_library.append(linking_outputs.library_to_link.interface_library)
    else:
        interface_library = library

    return [
        DefaultInfo(
            files = depset(library),
            runfiles = runfiles,
        ),
        OutputGroupInfo(
            main_shared_library_output = depset(library),
            interface_library = depset(interface_library),
            rule_impl_debug_files = depset(direct = debug_files, transitive = transitive_debug_files),
        ),
        CcSharedLibraryInfo(
            dynamic_deps = merged_cc_shared_library_info,
            exports = exports.keys(),
            link_once_static_libs = curr_link_once_static_libs_set,
            linker_input = cc_common.create_linker_input(
                owner = ctx.label,
                libraries = depset([linking_outputs.library_to_link] + precompiled_only_dynamic_libraries),
            ),
        ),
    ]

def _graph_structure_aspect_impl(target, ctx):
    children = []

    # Collect graph structure info from any possible deplike attribute. The aspect
    # itself applies across every deplike attribute (attr_aspects is *), so enumerate
    # over all attributes and consume GraphNodeInfo if available.
    for fieldname in dir(ctx.rule.attr):
        deps = getattr(ctx.rule.attr, fieldname, None)
        if type(deps) == "list":
            for dep in deps:
                if type(dep) == "Target" and GraphNodeInfo in dep:
                    children.append(dep[GraphNodeInfo])
        elif type(deps) == "Target" and GraphNodeInfo in deps:
            children.append(deps[GraphNodeInfo])

    # TODO(bazel-team): Add flag to Bazel that can toggle the initialization of
    # linkable_more_than_once.
    linkable_more_than_once = False
    no_exporting = False
    if hasattr(ctx.rule.attr, "tags"):
        for tag in ctx.rule.attr.tags:
            if tag == LINKABLE_MORE_THAN_ONCE:
                linkable_more_than_once = True
            elif tag == NO_EXPORTING:
                no_exporting = True

    return [GraphNodeInfo(
        label = ctx.label,
        children = children,
        linkable_more_than_once = linkable_more_than_once,
        no_exporting = no_exporting,
    )]

graph_structure_aspect = aspect(
    attr_aspects = ["*"],
    required_providers = [[CcInfo], [ProtoInfo]],
    required_aspect_providers = [[CcInfo]],
    implementation = _graph_structure_aspect_impl,
)

cc_shared_library = rule(
    implementation = _cc_shared_library_impl,
    attrs = {
        "additional_linker_inputs": attr.label_list(allow_files = True),
        "shared_lib_name": attr.string(),
        "dynamic_deps": attr.label_list(providers = [CcSharedLibraryInfo]),
        "exports_filter": attr.string_list(),
        "win_def_file": attr.label(allow_single_file = [".def"]),
        "roots": attr.label_list(providers = [CcInfo], aspects = [graph_structure_aspect]),
        "deps": attr.label_list(providers = [CcInfo], aspects = [graph_structure_aspect]),
        "static_deps": attr.string_list(),
        "user_link_flags": attr.string_list(),
        "_def_parser": semantics.get_def_parser(),
        "_cc_toolchain": attr.label(default = "@" + semantics.get_repo() + "//tools/cpp:current_cc_toolchain"),
        "_grep_includes": attr.label(
            allow_files = True,
            executable = True,
            cfg = "exec",
            default = Label("@" + semantics.get_repo() + "//tools/cpp:grep-includes"),
        ),
    },
    toolchains = cc_helper.use_cpp_toolchain(),
    fragments = ["google_cpp", "cpp"],
    incompatible_use_toolchain_transition = True,
)

for_testing_dont_use_check_if_target_under_path = _check_if_target_under_path
merge_cc_shared_library_infos = _merge_cc_shared_library_infos
build_link_once_static_libs_map = _build_link_once_static_libs_map
build_exports_map_from_only_dynamic_deps = _build_exports_map_from_only_dynamic_deps
throw_linked_but_not_exported_errors = _throw_linked_but_not_exported_errors
