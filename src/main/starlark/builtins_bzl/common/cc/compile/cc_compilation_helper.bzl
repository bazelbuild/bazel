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

"""Compilation helper for C++ rules."""

load(
    ":common/cc/cc_helper_internal.bzl",
    "package_source_root",
    "repository_exec_path",
)
load(":common/cc/semantics.bzl", "USE_EXEC_ROOT_FOR_VIRTUAL_INCLUDES_SYMLINKS")
load(":common/paths.bzl", "paths")

cc_common_internal = _builtins.internal.cc_common
cc_internal = _builtins.internal.cc_internal

_VIRTUAL_INCLUDES_DIR = "_virtual_includes"

def _include_dir(directory, repo_path, sibling_repo_layout):
    if sibling_repo_layout:
        return directory
    else:
        return paths.get_relative(directory, repo_path)

def _repo_relative_path(artifact):
    relative_path = artifact.path
    if artifact.is_source:
        if artifact.owner.workspace_root:
            relative_path = "/".join(relative_path.split("/")[2:])
    else:
        relative_path = paths.relativize(relative_path, artifact.root.path)

    if (artifact.owner.workspace_root.startswith("external/") or artifact.owner.workspace_root.startswith("../")) and \
       relative_path.startswith("external"):
        relative_path = "/".join(relative_path.split("/")[2:])

    return relative_path

def _enabled(feature_configuration, feature_name):
    return feature_configuration.is_enabled(feature_name)

def _compute_public_headers(
        actions,
        config,
        public_headers_artifacts,
        include_prefix,
        strip_include_prefix,
        label,
        binfiles_dir,
        non_module_map_headers,
        is_sibling_repository_layout,
        shorten_virtual_includes):
    if include_prefix:
        if not paths.is_normalized(include_prefix, False):
            fail("include prefix should not contain uplevel references: " + include_prefix)
        if paths.is_absolute(include_prefix):
            fail("include prefix should be a relative path: " + include_prefix)

    if strip_include_prefix:
        if not paths.is_normalized(strip_include_prefix, False):
            fail("strip include prefix should not contain uplevel references: " + strip_include_prefix)
        strip_prefix = strip_include_prefix
        if strip_prefix.startswith("//"):
            strip_prefix = strip_prefix[1:]
        if paths.is_absolute(strip_prefix):
            # Very crude way of determining driver length, but the best one at the same time.
            strip_driver_length = 0
            if strip_prefix.startswith("/"):  # Unix
                strip_driver_length = 1
            elif len(strip_prefix) > 1 and strip_prefix[1] == ":":  # Windows
                strip_driver_length = 3
            strip_prefix = strip_prefix[strip_driver_length:]
        else:
            # paths.normalize differs from Java normalize call in a way
            # that if string to be normalized is "." Starlark version returns ".",
            # while Java version returns an empty string "".
            # Because of this if label.package is an empty string and strip_prefix
            # is "." paths.get_relative("", ".") returns "." instead of "".
            strip_prefix = paths.get_relative(label.package, strip_prefix)
            if strip_prefix == ".":
                strip_prefix = ""

    elif include_prefix:
        strip_prefix = label.package
    else:
        strip_prefix = None

    if strip_prefix and strip_prefix.startswith("/"):
        strip_prefix = strip_prefix[1:]

    if include_prefix and include_prefix.startswith("/"):
        include_prefix = include_prefix[1:]

    if strip_prefix == None and include_prefix == None:
        return struct(
            headers = public_headers_artifacts + non_module_map_headers,
            module_map_headers = public_headers_artifacts,
            virtual_include_path = None,
            virtual_to_original_headers = depset(),
        )

    module_map_headers = []
    virtual_to_original_headers_list = []
    source_package_path = package_source_root(label.workspace_name, label.package, is_sibling_repository_layout)
    if shorten_virtual_includes:
        virtual_include_dir = paths.join(_VIRTUAL_INCLUDES_DIR, "%x" % hash(paths.join(source_package_path, label.name)))
    else:
        virtual_include_dir = paths.join(source_package_path, _VIRTUAL_INCLUDES_DIR, label.name)
    for original_header in public_headers_artifacts:
        repo_relative_path = _repo_relative_path(original_header)
        if not repo_relative_path.startswith(strip_prefix):
            fail("header '{}' is not under the specified strip prefix '{}'".format(repo_relative_path, strip_prefix))
        include_path = paths.relativize(repo_relative_path, strip_prefix)
        if include_prefix != None:
            include_path = paths.get_relative(include_prefix, include_path)

        virtual_header = actions.declare_shareable_artifact(paths.join(virtual_include_dir, include_path))
        actions.symlink(
            output = virtual_header,
            target_file = original_header,
            progress_message = "Symlinking virtual headers for %{label}",
            use_exec_root_for_source = USE_EXEC_ROOT_FOR_VIRTUAL_INCLUDES_SYMLINKS,
        )
        module_map_headers.append(virtual_header)
        if config.coverage_enabled:
            virtual_to_original_headers_list.append((virtual_header.path, original_header.path))

        module_map_headers.append(original_header)

    virtual_headers = module_map_headers + non_module_map_headers
    return struct(
        headers = virtual_headers,
        module_map_headers = module_map_headers,
        virtual_include_path = paths.join(binfiles_dir, virtual_include_dir),
        virtual_to_original_headers = depset(virtual_to_original_headers_list),
    )

def _generates_header_module(feature_configuration, public_headers, private_headers, generate_action):
    return _enabled(feature_configuration, "header_modules") and \
           (public_headers or private_headers) and \
           generate_action

def _header_module_artifact(actions, label, is_sibling_repository_layout, suffix, extension):
    object_dir = paths.join(paths.join(package_source_root(label.workspace_name, label.package, is_sibling_repository_layout), "_objs"), label.name)
    base_name = label.name.split("/")[-1]
    output_path = paths.join(object_dir, base_name + suffix + extension)
    return actions.declare_shareable_artifact(output_path)

def _collect_module_maps(deps, cc_toolchain_compilation_context, additional_cpp_module_maps):
    # TODO(bazel-team): Here we use the implementationDeps to build the dependents of this rule's
    # module map. This is technically incorrect for the following reasons:
    #  - Clang will not issue a layering_check warning if headers from implementation deps are
    #    included from headers of this library.
    #  - If we were to ever build with modules, Clang might store this dependency inside the .pcm
    # It should be evaluated whether this is ok.  If this turned into a problem at some
    # point, we could probably just declare two different modules with different use-declarations
    # in the module map file.
    module_maps = []
    for cc_context in deps:
        if cc_context.module_map() != None:
            module_maps.append(cc_context.module_map())
        module_maps.extend(cc_context.exporting_module_maps())

    if cc_toolchain_compilation_context != None and cc_toolchain_compilation_context.module_map() != None:
        module_maps.append(cc_toolchain_compilation_context.module_map())

    for additional_cpp_module_map in additional_cpp_module_maps:
        module_maps.append(additional_cpp_module_map)

    return module_maps

def _module_map_struct_to_module_map_content(parameters, tree_expander):
    lines = []
    module_map = parameters.module_map
    lines.append("module \"%s\" {" % module_map.name())
    lines.append("  export *")

    def expanded(artifacts):
        expanded = []
        for artifact in artifacts:
            if artifact.is_directory:
                expanded.extend(tree_expander.expand(artifact))
            else:
                expanded.append(artifact)
        return expanded

    def add_header(path, visibility, can_compile):
        header_line = []
        if parameters.generate_submodules:
            lines.append("  module \"" + path + "\" {")
            lines.append("    export *")
            header_line.append("  ")
        header_line.append("  ")
        if visibility:
            header_line.append(visibility)
            header_line.append(" ")
        should_compile = parameters.compiled_module and not path.endswith(".inc")
        if not can_compile or not should_compile:
            header_line.append("textual ")
        header_line.append("header \"")
        header_line.append(parameters.leading_periods)
        header_line.append(path)
        header_line.append("\"")
        lines.append("".join(header_line))
        if parameters.generate_submodules:
            lines.append("  }")

    added_paths = set()
    for header in expanded(parameters.public_headers):
        if header.path in added_paths:
            continue
        add_header(path = header.path, visibility = "", can_compile = True)
        added_paths.add(header.path)

    for header in expanded(parameters.private_headers):
        if header.path in added_paths:
            continue
        add_header(path = header.path, visibility = "private", can_compile = True)
        added_paths.add(header.path)

    for header in parameters.separate_module_headers:
        if header.path in added_paths:
            continue
        add_header(path = header.path, visibility = "", can_compile = False)
        added_paths.add(header.path)

    for path in parameters.additional_exported_headers:
        if path in added_paths:
            continue
        add_header(path = path, visibility = "", can_compile = False)
        added_paths.add(path)

    for dep in parameters.dependency_module_maps:
        lines.append("  use \"" + dep.name() + "\"")

    if parameters.separate_module_headers:
        separate_name = module_map.name() + ".sep"
        lines.append("  use \"" + separate_name + "\"")
        lines.append("}")
        lines.append("module \"" + separate_name + "\" {")
        lines.append("  export *")

        added_paths = set()
        for header in parameters.separate_module_headers:
            if header.path in added_paths:
                continue
            add_header(path = header.path, visibility = "", can_compile = True)
            added_paths.add(header.path)

        for dep in parameters.dependency_module_maps:
            lines.append("  use \"" + dep.name() + "\"")

    lines.append("}")

    if parameters.extern_dependencies:
        for dep in parameters.dependency_module_maps:
            lines.append(
                "extern module \"" + dep.name() + "\" \"" +
                parameters.leading_periods + dep.file().path + "\"",
            )

    return lines

def _create_module_map_action(
        actions,
        module_map,
        private_headers,
        public_headers,
        dependency_module_maps,
        additional_exported_headers,
        separate_module_headers,
        compiled_module,
        module_map_home_is_cwd,
        generate_submodules,
        extern_dependencies):
    content = actions.args()
    content.set_param_file_format("multiline")
    segments_to_exec_path = module_map.file().path.count("/")
    leading_periods = "" if module_map_home_is_cwd else "../" * segments_to_exec_path
    data_struct = struct(
        module_map = module_map,
        public_headers = public_headers,
        private_headers = private_headers,
        dependency_module_maps = dependency_module_maps,
        additional_exported_headers = additional_exported_headers,
        separate_module_headers = separate_module_headers,
        compiled_module = compiled_module,
        generate_submodules = generate_submodules,
        extern_dependencies = extern_dependencies,
        leading_periods = leading_periods,
    )
    content.add_all([data_struct], map_each = _module_map_struct_to_module_map_content)

    # We need to add all tree artifacts to the args object directly so we they can be
    # expanded in the _module_map_struct_to_module_map_content callback function.
    # We don't want to do anything with them at this point, so the map_each callback should be a
    # simple null function.
    tree_artifacts = [h for h in private_headers if h.is_directory]
    tree_artifacts += [h for h in public_headers if h.is_directory]
    content.add_all(tree_artifacts, map_each = lambda x: None, allow_closure = True)

    actions.write(module_map.file(), content = content, is_executable = True, mnemonic = "CppModuleMap")

def _init_cc_compilation_context(
        # DO NOT use ctx, this is a temporary placeholder
        # to avoid adding a new field to CcCompilationHelper.
        # Once compile is in Starlark we can directly pass in actions here.
        ctx,
        binfiles_dir,
        genfiles_dir,
        label,
        config,
        quote_include_dirs,
        framework_include_dirs,
        system_include_dirs,
        include_dirs,
        feature_configuration,
        public_headers_artifacts,
        include_prefix,
        strip_include_prefix,
        non_module_map_headers,
        cc_toolchain_compilation_context,
        defines,
        local_defines,
        public_textual_headers,
        private_headers_artifacts,
        additional_inputs,
        separate_module_headers,
        generate_module_map,
        generate_pic_action,
        generate_no_pic_action,
        module_map,
        propagate_module_map_to_compile_action,
        additional_exported_headers,
        deps,
        purpose,
        implementation_deps,
        additional_cpp_module_maps):
    # Single usage of ctx.
    actions = ctx.actions

    # Setup the include path; local include directories come before those inherited from deps or
    # from the toolchain; in case of aliasing (same include file found on different entries),
    # prefer the local include rather than the inherited one.
    # Add in the roots for well-formed include names for source files and
    # generated files. It is important that the execRoot (EMPTY_FRAGMENT) comes
    # before the genfilesFragment to preferably pick up source files. Otherwise
    # we might pick up stale generated files.
    sibling_repo_layout = config.is_sibling_repository_layout()
    repo_name = label.workspace_name
    repo_path = repository_exec_path(repo_name, sibling_repo_layout)
    gen_include_dir = _include_dir(genfiles_dir, repo_path, sibling_repo_layout)
    bin_include_dir = _include_dir(binfiles_dir, repo_path, sibling_repo_layout)
    quote_include_dirs_for_context = [repo_path, gen_include_dir, bin_include_dir] + quote_include_dirs
    external = repo_name != "" and _enabled(feature_configuration, "external_include_paths")
    shorten_virtual_includes = _enabled(feature_configuration, "shorten_virtual_includes")
    external_include_dirs = []
    declared_include_srcs = []

    if not external:
        system_include_dirs_for_context = list(system_include_dirs)
        include_dirs_for_context = list(include_dirs)
    else:
        # Do not add system_include_dirs and include_dirs directly to compilation context.
        system_include_dirs_for_context = []
        include_dirs_for_context = []

        external_include_dirs.append(repo_path)
        external_include_dirs.append(gen_include_dir)
        external_include_dirs.append(bin_include_dir)
        external_include_dirs.extend(quote_include_dirs_for_context)
        external_include_dirs.extend(system_include_dirs)
        external_include_dirs.extend(include_dirs)

    public_headers = _compute_public_headers(
        actions,
        config,
        public_headers_artifacts,
        include_prefix,
        strip_include_prefix,
        label,
        binfiles_dir,
        non_module_map_headers,
        sibling_repo_layout,
        shorten_virtual_includes,
    )
    if public_headers.virtual_include_path:
        if external:
            external_include_dirs.append(public_headers.virtual_include_path)
        else:
            include_dirs_for_context.append(public_headers.virtual_include_path)

    if config.coverage_enabled:
        # Populate the map only when code coverage collection is enabled, to report the actual
        # source file name in the coverage output file.
        virtual_to_original_headers = public_headers.virtual_to_original_headers
    else:
        virtual_to_original_headers = depset()

    declared_include_srcs.extend(public_headers.headers)
    declared_include_srcs.extend(public_textual_headers)
    declared_include_srcs.extend(private_headers_artifacts)
    declared_include_srcs.extend(additional_inputs)

    generates_pic_header_module = _generates_header_module(feature_configuration, public_headers_artifacts, private_headers_artifacts, generate_pic_action)
    generates_no_pic_header_module = _generates_header_module(feature_configuration, public_headers_artifacts, private_headers_artifacts, generate_no_pic_action)
    if separate_module_headers:
        if not (_enabled(feature_configuration, "module_maps") and
                generate_module_map and
                (generates_pic_header_module or generates_no_pic_header_module)):
            fail("Should use separate headers only when building modules: " + label.name)

    separate_public_headers = _compute_public_headers(
        actions,
        config,
        separate_module_headers,
        include_prefix,
        strip_include_prefix,
        label,
        binfiles_dir,
        non_module_map_headers,
        sibling_repo_layout,
        shorten_virtual_includes,
    )

    separate_module = None
    separate_pic_module = None
    pic_header_module = None
    header_module = None
    if _enabled(feature_configuration, "module_maps"):
        if not module_map:
            module_map = cc_common_internal.create_module_map(
                file = actions.declare_file(label.name + ".cppmap"),
                name = label.workspace_name + "//" + label.package + ":" + label.name,
            )

        # There are different modes for module compilation:
        # 1. We create the module map and compile the module so that libraries depending on us can
        #    use the resulting module artifacts in their compilation (compiled is true).
        # 2. We create the module map so that libraries depending on us will include the headers
        #    textually (compiled is false).
        if generate_module_map:
            compiled = _enabled(feature_configuration, "header_modules") or \
                       _enabled(feature_configuration, "compile_all_modules")

            if _enabled(feature_configuration, "only_doth_headers_in_module_maps"):
                public_headers_for_module_map_action = [header for header in public_headers.module_map_headers if (header.is_directory or header.extension == "h")]
            else:
                public_headers_for_module_map_action = public_headers.module_map_headers

            private_headers_for_module_map_action = private_headers_artifacts
            if _enabled(feature_configuration, "exclude_private_headers_in_module_maps"):
                private_headers_for_module_map_action = []
            dependency_module_maps = _collect_module_maps(deps + implementation_deps, cc_toolchain_compilation_context, additional_cpp_module_maps)
            _create_module_map_action(
                actions = actions,
                module_map = module_map,
                public_headers = public_headers_for_module_map_action,
                separate_module_headers = separate_public_headers.module_map_headers,
                dependency_module_maps = dependency_module_maps,
                private_headers = private_headers_for_module_map_action,
                additional_exported_headers = additional_exported_headers,
                compiled_module = compiled,
                module_map_home_is_cwd = _enabled(feature_configuration, "module_map_home_cwd"),
                generate_submodules = _enabled(feature_configuration, "generate_submodules"),
                extern_dependencies = not _enabled(feature_configuration, "module_map_without_extern_module"),
            )

        if generates_pic_header_module:
            pic_header_module = _header_module_artifact(
                actions,
                label,
                sibling_repo_layout,
                "",
                ".pic.pcm",
            )
        if generates_no_pic_header_module:
            header_module = _header_module_artifact(
                actions,
                label,
                sibling_repo_layout,
                "",
                ".pcm",
            )
        if separate_module_headers:
            declared_include_srcs.extend(separate_public_headers.headers)
            if generates_no_pic_header_module:
                separate_module = _header_module_artifact(
                    actions,
                    label,
                    sibling_repo_layout,
                    ".sep",
                    ".pcm",
                )
            if generates_pic_header_module:
                separate_pic_module = _header_module_artifact(
                    actions,
                    label,
                    sibling_repo_layout,
                    ".sep",
                    ".pic.pcm",
                )

    else:
        # Do not set module map related attributes.
        module_map = None
        propagate_module_map_to_compile_action = True

    dependent_cc_compilation_contexts = []
    if cc_toolchain_compilation_context != None:
        dependent_cc_compilation_contexts.append(cc_toolchain_compilation_context)
    dependent_cc_compilation_contexts.extend(deps)

    main_context = cc_common_internal.create_compilation_context(
        actions = actions,
        label = label,
        quote_includes = depset(quote_include_dirs_for_context),
        framework_includes = depset(framework_include_dirs),
        external_includes = depset(external_include_dirs),
        system_includes = depset(system_include_dirs_for_context),
        includes = depset(include_dirs_for_context),
        virtual_to_original_headers = virtual_to_original_headers,
        dependent_cc_compilation_contexts = dependent_cc_compilation_contexts,
        non_code_inputs = additional_inputs,
        defines = depset(defines),
        local_defines = depset(local_defines),
        headers = depset(declared_include_srcs),
        direct_public_headers = public_headers.headers,
        direct_private_headers = private_headers_artifacts,
        direct_textual_headers = public_textual_headers,
        propagate_module_map_to_compile_action = propagate_module_map_to_compile_action,
        module_map = module_map,
        pic_header_module = pic_header_module,
        header_module = header_module,
        separate_module_headers = separate_public_headers.headers,
        separate_module = separate_module,
        separate_pic_module = separate_pic_module,
        purpose = purpose,
        add_public_headers_to_modular_headers = False,
        exported_dependent_cc_compilation_contexts = [],
        headers_checking_mode = "STRICT",
        loose_hdrs_dirs = [],
    )
    implementation_deps_context = None
    if implementation_deps:
        implementation_deps_context = cc_common_internal.create_compilation_context(
            actions = actions,
            label = label,
            quote_includes = depset(quote_include_dirs_for_context),
            framework_includes = depset(framework_include_dirs),
            external_includes = depset(external_include_dirs),
            system_includes = depset(system_include_dirs_for_context),
            includes = depset(include_dirs_for_context),
            virtual_to_original_headers = virtual_to_original_headers,
            dependent_cc_compilation_contexts = dependent_cc_compilation_contexts + implementation_deps,
            non_code_inputs = additional_inputs,
            defines = depset(defines),
            local_defines = depset(local_defines),
            headers = depset(declared_include_srcs),
            direct_public_headers = public_headers.headers,
            direct_private_headers = private_headers_artifacts,
            direct_textual_headers = public_textual_headers,
            propagate_module_map_to_compile_action = propagate_module_map_to_compile_action,
            module_map = module_map,
            pic_header_module = pic_header_module,
            header_module = header_module,
            separate_module_headers = separate_public_headers.headers,
            separate_module = separate_module,
            separate_pic_module = separate_pic_module,
            purpose = purpose + "_impl",
            add_public_headers_to_modular_headers = False,
            exported_dependent_cc_compilation_contexts = [],
            headers_checking_mode = "STRICT",
            loose_hdrs_dirs = [],
        )

    return main_context, implementation_deps_context

cc_compilation_helper = struct(
    init_cc_compilation_context = _init_cc_compilation_context,
)
