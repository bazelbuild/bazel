# Copyright 2025 The Bazel Authors. All rights reserved.
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
The cc_common.compile function.

Used for C++ compiling.
"""

load(
    ":common/cc/cc_helper_internal.bzl",
    "artifact_category",
    "extensions",
    "should_create_per_object_debug_info",
)
load(":common/cc/compile/cc_compilation_helper.bzl", "cc_compilation_helper")
load(":common/paths.bzl", "paths")

cc_common_internal = _builtins.internal.cc_common
cc_internal = _builtins.internal.cc_internal

CPP_SOURCE_TYPE_HEADER = "HEADER"
CPP_SOURCE_TYPE_SOURCE = "SOURCE"
CPP_SOURCE_TYPE_CLIF_INPUT_PROTO = "CLIF_INPUT_PROTO"

SOURCE_CATEGORY_CC = set(
    extensions.CC_SOURCE +
    extensions.CC_HEADER +
    extensions.C_SOURCE +
    extensions.ASSEMBLER +
    extensions.ASSESMBLER_WITH_C_PREPROCESSOR +
    extensions.CLIF_INPUT_PROTO,
)
SOURCE_CATEGORY_CC_AND_OBJC = set(
    extensions.CC_SOURCE +
    extensions.CC_HEADER +
    extensions.OBJC_SOURCE +
    extensions.OBJCPP_SOURCE +
    extensions.C_SOURCE +
    extensions.ASSEMBLER +
    extensions.ASSESMBLER_WITH_C_PREPROCESSOR,
)

# Filetypes that generate LLVM bitcode when -flto is specified.
LTO_SOURCE_EXTENSIONS = set(extensions.CC_SOURCE + extensions.C_SOURCE)

# LINT.IfChange(compile_api)
def compile(
        *,
        actions,
        feature_configuration,
        cc_toolchain,
        srcs = [],
        public_hdrs = [],
        private_hdrs = [],
        textual_hdrs = [],
        additional_exported_hdrs = [],
        includes = [],
        loose_includes = None,  # TODO(b/396122076): seems unused; double-check and remove
        quote_includes = [],
        system_includes = [],
        framework_includes = [],
        defines = [],
        local_defines = [],
        include_prefix = "",
        strip_include_prefix = "",
        user_compile_flags = [],
        conly_flags = [],
        cxx_flags = [],
        compilation_contexts = [],
        implementation_compilation_contexts = [],
        name,
        disallow_pic_outputs = False,
        disallow_nopic_outputs = False,
        additional_include_scanning_roots = [],
        additional_inputs = [],
        module_map = None,
        additional_module_maps = [],
        propagate_module_map_to_compile_action = True,
        do_not_generate_module_map = False,
        code_coverage_enabled = False,
        hdrs_checking_mode = None,  # TODO(b/396122076): seems unused; double-check and remove
        variables_extension = None,
        language = None,
        purpose = None,
        copts_filter = None,
        separate_module_headers = [],
        module_interfaces = [],
        non_compilation_additional_inputs = []):
    """Should be used for C++ compilation.

    Args:
        actions: <code>actions</code> object.
        feature_configuration: <code>feature_configuration</code> to be queried."),
        cc_toolchain: <code>CcToolchainInfo</code> provider to be used."),
        srcs: The list of source files to be compiled.",
        public_hdrs: List of headers needed for compilation of srcs and may be included by
            dependent rules transitively.
        private_hdrs: List of headers needed for compilation of srcs and NOT to be included by
            dependent rules. May be a list of File objects or a list of (File, Label) tuples.
            The latter option exists for the method collectPerFileCopts() which in turn supports
            the --per_file_copt flag.
        textual_hdrs: undocumented
        additional_exported_hdrs: undocumented
        includes: Search paths for header files referenced both by angle bracket and quotes.
            Usually passed with -I. Propagated to dependents transitively.
        loose_includes: undocumented
        quote_includes: Search paths for header files referenced by quotes,
            e.g. #include \"foo/bar/header.h\". They can be either relative to the exec
            root or absolute. Usually passed with -iquote. Propagated to dependents
            transitively.
        system_includes: Search paths for header files referenced by angle brackets, e.g. #include
            &lt;foo/bar/header.h&gt;. They can be either relative to the exec root or
            absolute. Usually passed with -isystem. Propagated to dependents
            transitively.
        framework_includes: Search paths for header files from Apple frameworks. They can be either
            relative to the exec root or absolute. Usually passed with -F. Propagated to
            dependents transitively.
        defines: Set of defines needed to compile this target. Each define is a string. Propagated
            to dependents transitively.
        local_defines: Set of defines needed to compile this target. Each define is a string. Not
            propagated to dependents transitively.
        include_prefix: The prefix to add to the paths of the headers of this rule. When set, the
            headers in the hdrs attribute of this rule are accessible at is the
            value of this attribute prepended to their repository-relative path.
            The prefix in the strip_include_prefix attribute is removed before this
            prefix is added.
        strip_include_prefix: The prefix to strip from the paths of the headers of this rule. When set, the
            headers in the hdrs attribute of this rule are accessible at their path
            with this prefix cut off. If it's a relative path, it's taken as a
            package-relative one. If it's an absolute one, it's understood as a
            repository-relative path. The prefix in the include_prefix attribute is
            added after this prefix is stripped.
        user_compile_flags: Additional list of compilation options.
        conly_flags: Additional list of compilation options for C compiles.
        cxx_flags: Additional list of compilation options for C++ compiles.
        compilation_contexts: Headers from dependencies used for compilation.
        implementation_compilation_contexts: undocumented
        name: This is used for naming the output artifacts of actions created by this
            method. See also the `main_output` arg.
        disallow_pic_outputs: Whether PIC outputs should be created.
        disallow_nopic_outputs: Whether NOPIC outputs should be created.
        additional_include_scanning_roots: undocumented
        additional_inputs: List of additional files needed for compilation of srcs
        module_map: undocumented
        additional_module_maps: undocumented
        propagate_module_map_to_compile_action: undocumented
        do_not_generate_module_map: undocumented
        code_coverage_enabled: undocumented
        hdrs_checking_mode: undocumented
        variables_extension: undocumented
        language: undocumented
        purpose: undocumented
        copts_filter: undocumented
        separate_module_headers: undocumented
        module_interfaces: The list of module interfaces source files to be compiled. Note: this is an
            experimental feature, only enabled with --experimental_cpp_modules
        non_compilation_additional_inputs: undocumented

    Returns:
        a tuple of  (<code>CompilationContext</code>, <code>CcCompilationOutputs</code>).
    """

    # LINT.ThenChange(//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/cpp/CcModuleApi.java:compile_api)
    # LINT.IfChange(compile)
    cc_common_internal.validate_starlark_compile_api_call(
        actions = actions,
        include_prefix = include_prefix,
        strip_include_prefix = strip_include_prefix,
        additional_include_scanning_roots = additional_include_scanning_roots,
    )

    cpp_configuration = cc_toolchain._cpp_configuration
    if additional_module_maps == None:
        additional_module_maps = []
    copts_filter_object = cc_internal.create_copts_filter(copts_filter)
    label = cc_internal.actions2ctx_cheat(actions).label.same_package_label(name)
    fdo_context = cc_toolchain._fdo_context

    use_pic_for_dynamic_libraries = cpp_configuration.force_pic() or feature_configuration.is_enabled("supports_pic")
    use_pic_for_binaries = cpp_configuration.force_pic() or (
        use_pic_for_dynamic_libraries and
        (cpp_configuration.compilation_mode() != "opt" or
         feature_configuration.is_enabled("prefer_pic_for_opt_binaries"))
    )
    generate_pic_action = use_pic_for_dynamic_libraries or use_pic_for_binaries
    generate_no_pic_action = not use_pic_for_dynamic_libraries or not use_pic_for_binaries
    if disallow_pic_outputs and disallow_nopic_outputs:
        fail("Either PIC or no PIC actions have to be created.")
    if disallow_nopic_outputs:
        generate_no_pic_action = False
    if disallow_pic_outputs:
        generate_pic_action = False
        generate_no_pic_action = True
    if not generate_pic_action and not generate_no_pic_action:
        fail("Either PIC or no PIC actions have to be created.")

    language_normalized = "c++" if language == None else language
    language_normalized = language_normalized.replace("+", "p").upper()
    source_category = SOURCE_CATEGORY_CC if language_normalized == "CPP" else SOURCE_CATEGORY_CC_AND_OBJC
    ctx = cc_internal.actions2ctx_cheat(actions)
    includes = includes.to_list() if type(includes) == "depset" else includes
    textual_hdrs_list = textual_hdrs.to_list() if type(textual_hdrs) == "depset" else textual_hdrs

    compilation_unit_sources = {}
    if (feature_configuration.is_enabled("parse_headers") and
        not feature_configuration.is_enabled("header_modules")):
        public_hdrs_with_labels = _to_file_label_tuple_list(public_hdrs, label)
        _add_suitable_headers_to_compilation_unit_sources(
            compilation_unit_sources,
            public_hdrs_with_labels,
        )
        private_hdrs_with_labels = _to_file_label_tuple_list(private_hdrs, label)
        _add_suitable_headers_to_compilation_unit_sources(
            compilation_unit_sources,
            private_hdrs_with_labels,
        )
    srcs_with_labels = _to_file_label_tuple_list(srcs, label)
    _add_suitable_srcs_to_compilation_unit_sources(
        compilation_unit_sources,
        srcs_with_labels,
        source_category,
    )

    public_hdrs_artifacts = _to_file_list(public_hdrs)
    private_hdrs_artifacts = _to_file_list(private_hdrs)

    public_compilation_context, implementation_deps_context = cc_compilation_helper.init_cc_compilation_context(
        ctx = ctx,
        binfiles_dir = ctx.bin_dir.path,
        genfiles_dir = ctx.genfiles_dir.path,
        label = label,
        config = ctx.configuration,
        quote_include_dirs = quote_includes,
        framework_include_dirs = framework_includes,
        system_include_dirs = system_includes,
        include_dirs = includes,
        feature_configuration = feature_configuration,
        public_headers_artifacts = public_hdrs_artifacts,
        include_prefix = include_prefix,
        strip_include_prefix = strip_include_prefix,
        non_module_map_headers = [],
        cc_toolchain_compilation_context = None if cc_toolchain == None else cc_toolchain._cc_info.compilation_context,
        defines = defines,
        local_defines = local_defines,
        public_textual_headers = textual_hdrs_list,
        private_headers_artifacts = private_hdrs_artifacts,
        additional_inputs = non_compilation_additional_inputs,
        separate_module_headers = separate_module_headers,
        generate_module_map = not do_not_generate_module_map,
        generate_pic_action = generate_pic_action,
        generate_no_pic_action = generate_no_pic_action,
        module_map = module_map,
        propagate_module_map_to_compile_action = propagate_module_map_to_compile_action,
        additional_exported_headers = additional_exported_hdrs + [h.path for h in textual_hdrs_list],
        deps = compilation_contexts,
        purpose = "unused",
        # init_cc_compilation_context() passes purpose to two calls to
        # createCcCompilationContext() where "purpose" is matched to param "unused3" after the
        # references to middleman were removed. See cl/696431863.
        # TODO(b/396122076): remove purpose from init_cc_compilation_context()
        implementation_deps = implementation_compilation_contexts,
        additional_cpp_module_maps = additional_module_maps,
    )

    if implementation_compilation_contexts and not implementation_deps_context:
        fail("Compilation context for implementation deps was not created")
    cc_compilation_context = implementation_deps_context if implementation_compilation_contexts else public_compilation_context

    if feature_configuration.is_enabled("header_modules") and not public_compilation_context.module_map:
        fail("All cc rules must support module maps.")

    common_compile_build_variables = cc_internal.setup_common_compile_build_variables(
        cc_compilation_context = cc_compilation_context,
        cc_toolchain = cc_toolchain,
        cpp_configuration = cpp_configuration,
        fdo_context = fdo_context,
        feature_configuration = feature_configuration,
        variables_extension = variables_extension,
    )
    auxiliary_fdo_inputs = cc_internal.get_auxiliary_fdo_inputs(
        cc_toolchain = cc_toolchain,
        fdo_context = fdo_context,
        feature_configuration = feature_configuration,
    )
    fdo_build_variables = cc_internal.setup_fdo_build_variables(
        cc_toolchain = cc_toolchain,
        fdo_context = fdo_context,
        auxiliary_fdo_inputs = auxiliary_fdo_inputs,
        feature_configuration = feature_configuration,
        fdo_instrument = cpp_configuration.fdo_instrument(),
        cs_fdo_instrument = cpp_configuration.cs_fdo_instrument(),
    )

    cc_outputs_builder = cc_internal.create_cc_compilation_outputs_builder()
    _create_cc_compile_actions(
        action_construction_context = ctx,
        additional_compilation_inputs = additional_inputs,
        additional_include_scanning_roots = additional_include_scanning_roots,
        cc_compilation_context = cc_compilation_context,
        cc_toolchain = cc_toolchain,
        compilation_unit_sources = compilation_unit_sources,
        configuration = ctx.configuration,
        conlyopts = conly_flags,
        copts = user_compile_flags,
        copts_filter = copts_filter_object,
        cpp_configuration = cpp_configuration,
        cxxopts = cxx_flags,
        fdo_context = fdo_context,
        feature_configuration = feature_configuration,
        generate_no_pic_action = generate_no_pic_action,
        generate_pic_action = generate_pic_action,
        is_code_coverage_enabled = code_coverage_enabled,
        label = label,
        private_headers = private_hdrs_artifacts,
        public_headers = public_hdrs_artifacts,
        purpose = purpose if purpose else "",
        separate_module_headers = separate_module_headers,
        language = language,
        outputs = cc_outputs_builder,
        common_compile_build_variables = common_compile_build_variables,
        auxiliary_fdo_inputs = auxiliary_fdo_inputs,
        fdo_build_variables = fdo_build_variables,
    )
    cc_outputs = cc_outputs_builder.build()

    if cpp_configuration.process_headers_in_dependencies():
        compilation_context = cc_internal.create_cc_compilation_context_with_extra_header_tokens(
            cc_compilation_context = public_compilation_context,
            extra_header_tokens = cc_outputs.header_tokens(),
        )
    else:
        compilation_context = public_compilation_context
    return (compilation_context, cc_outputs)

def _add_suitable_headers_to_compilation_unit_sources(
        compilation_unit_sources,
        headers_with_labels):
    """Adds headers and tree artifacts but not textual includes to compilation_unit_sources.

    Args:
        compilation_unit_sources: A dictionary of (File, Label) tuples to (CppSource, Label)
            tuples.
        headers_with_labels: A list of (File, Label) tuples of header files.
    """
    for header, label in headers_with_labels:
        is_header = "." + header.extension in extensions.CC_HEADER or cc_internal.is_tree_artifact(header)
        is_textual_include = "." + header.extension in extensions.CC_TEXTUAL_INCLUDE
        if is_header and not is_textual_include:
            compilation_unit_sources[header] = cc_internal.create_cpp_source(
                label = label,
                source = header,
                type = CPP_SOURCE_TYPE_HEADER,
            )

def _add_suitable_srcs_to_compilation_unit_sources(
        compilation_unit_sources,
        srcs_with_labels,
        source_category):
    """Adds sources matching source_category and tree artifacts to compilation_unit_sources."""
    for source, label in srcs_with_labels:
        if "." + source.extension in extensions.CC_HEADER:
            fail("Adding header as source: " + source.basename)

        # TODO(b/413333884): If it's a non-source file we ignore it. This is only the case for
        # precompiled files which should be forbidden in srcs of cc_library|binary and instead be
        # migrated to cc_import rules.
        if "." + source.extension in source_category or cc_internal.is_tree_artifact(source):
            compilation_unit_sources[source] = cc_internal.create_cpp_source(
                label = label,
                source = source,
                type = CPP_SOURCE_TYPE_CLIF_INPUT_PROTO if "." + source.extension in extensions.CLIF_INPUT_PROTO else CPP_SOURCE_TYPE_SOURCE,
            )

def _to_file_list(list_of_files_or_tuples):
    """Converts a list that may contain File objects or tuples into  a list of File objects.

    Args:
        list_of_files_or_tuples: A list of File objects or a list of (File, Label) tuples.
            All list elements are expected to be of the same type.
    Returns:
        A list of File objects.
    """
    if not list_of_files_or_tuples:
        return []
    if type(list_of_files_or_tuples[0]) == "File":
        return list_of_files_or_tuples
    if type(list_of_files_or_tuples[0]) == "tuple":
        return [h[0] for h in list_of_files_or_tuples]
    fail("Should be either tuple or File: " + type(list_of_files_or_tuples[0]))

def _to_file_label_tuple_list(list_of_files_or_tuples, label):
    """Converts a list that may contain File objects or tuples into  a list of (File, Label) tuples.

    Args:
        list_of_files_or_tuples: A list of File objects or a list of (File, Label) tuples.
            All list elements are expected to be of the same type.
        label: The Label to be used in the returned list of (File, Label) tuples if
            list_of_files_or_tuples consists of File objects.
    Returns:
        A list of (File, Label) tuples.
    """
    if not list_of_files_or_tuples:
        return []
    if type(list_of_files_or_tuples[0]) == "tuple":
        return list_of_files_or_tuples
    if type(list_of_files_or_tuples[0]) == "File":
        return [(h, label) for h in list_of_files_or_tuples]
    fail("Should be either tuple or File: " + type(list_of_files_or_tuples[0]))

def _should_provide_header_modules(
        feature_configuration,
        private_headers,
        public_headers):
    """Returns whether we want to provide header modules for the current target."""
    return (
        feature_configuration.is_enabled("header_modules") and
        (private_headers or public_headers)
    )

# LINT.ThenChange(//src/main/java/com/google/devtools/build/lib/rules/cpp/CcModule.java:compile)

def _create_cc_compile_actions(
        *,
        action_construction_context,
        additional_compilation_inputs,
        additional_include_scanning_roots,
        cc_compilation_context,
        cc_toolchain,
        compilation_unit_sources,
        configuration,
        conlyopts,
        copts,
        copts_filter,
        cpp_configuration,
        cxxopts,
        fdo_context,
        feature_configuration,
        generate_no_pic_action,
        generate_pic_action,
        is_code_coverage_enabled,
        label,
        private_headers,
        public_headers,
        purpose,
        separate_module_headers,
        language,
        outputs,
        common_compile_build_variables,
        auxiliary_fdo_inputs,
        fdo_build_variables):
    """Constructs the C++ compiler actions.

    It generally creates one action for every specified source
    file. It takes into account coverage, and PIC, in addition to using the settings specified on
    the current object. This method should only be called once.
    """
    if generate_pic_action and not feature_configuration.is_enabled("pic") and not feature_configuration.is_enabled("supports_pic"):
        fail("PIC compilation is requested but the toolchain does not support it " +
             "(feature named 'supports_pic' is not enabled)")

    cpp_semantics = cc_common_internal.get_cpp_semantics(language = language)

    if _should_provide_header_modules(feature_configuration, private_headers, public_headers):
        cpp_module_map = cc_compilation_context.module_map()
        module_map_label = Label(cpp_module_map.name())
        cpp_compile_action_builder = cc_internal.create_cpp_compile_action_builder(
            action_construction_context = action_construction_context,
            cc_toolchain = cc_toolchain,
            cc_compilation_context = cc_compilation_context,
            configuration = configuration,
            copts_filter = copts_filter,
            feature_configuration = feature_configuration,
            semantics = cpp_semantics,
            source_artifact = cpp_module_map.file(),
        )
        modules = cc_internal.create_module_action(
            action_construction_context = action_construction_context,
            cc_compilation_context = cc_compilation_context,
            cc_toolchain = cc_toolchain,
            configuration = configuration,
            conlyopts = conlyopts,
            copts = copts,
            copts_filter = copts_filter,
            cpp_configuration = cpp_configuration,
            cxxopts = cxxopts,
            fdo_context = fdo_context,
            auxiliary_fdo_inputs = auxiliary_fdo_inputs,
            feature_configuration = feature_configuration,
            generate_no_pic_action = generate_no_pic_action,
            generate_pic_action = generate_pic_action,
            label = label,
            common_toolchain_variables = common_compile_build_variables,
            fdo_build_variables = fdo_build_variables,
            cpp_semantics = cpp_semantics,
            outputs = outputs,
            cpp_module_map = cpp_module_map,
            cpp_compile_action_builder = cpp_compile_action_builder,
        )
        if separate_module_headers:
            separate_cpp_module_map = cpp_module_map.create_separate_module_map()
            cpp_compile_action_builder = cc_internal.create_cpp_compile_action_builder(
                action_construction_context = action_construction_context,
                cc_toolchain = cc_toolchain,
                cc_compilation_context = cc_compilation_context,
                configuration = configuration,
                copts_filter = copts_filter,
                feature_configuration = feature_configuration,
                semantics = cpp_semantics,
                source_artifact = separate_cpp_module_map.file(),
            )
            separate_modules = cc_internal.create_module_action(
                action_construction_context = action_construction_context,
                cc_compilation_context = cc_compilation_context,
                cc_toolchain = cc_toolchain,
                configuration = configuration,
                conlyopts = conlyopts,
                copts = copts,
                copts_filter = copts_filter,
                cpp_configuration = cpp_configuration,
                cxxopts = cxxopts,
                fdo_context = fdo_context,
                auxiliary_fdo_inputs = auxiliary_fdo_inputs,
                feature_configuration = feature_configuration,
                generate_no_pic_action = generate_no_pic_action,
                generate_pic_action = generate_pic_action,
                label = label,
                common_toolchain_variables = common_compile_build_variables,
                fdo_build_variables = fdo_build_variables,
                cpp_semantics = cpp_semantics,
                outputs = outputs,
                cpp_module_map = separate_cpp_module_map,
                cpp_compile_action_builder = cpp_compile_action_builder,
            )
            modules = modules + separate_modules
        if feature_configuration.is_enabled("header_module_codegen"):
            for module in modules:
                cpp_compile_action_builder = cc_internal.create_cpp_compile_action_builder(
                    action_construction_context = action_construction_context,
                    cc_toolchain = cc_toolchain,
                    cc_compilation_context = cc_compilation_context,
                    configuration = configuration,
                    copts_filter = copts_filter,
                    feature_configuration = feature_configuration,
                    semantics = cpp_semantics,
                    source_artifact = module,
                )
                cc_internal.create_module_codegen_action(
                    action_construction_context = action_construction_context,
                    cc_compilation_context = cc_compilation_context,
                    cc_toolchain = cc_toolchain,
                    configuration = configuration,
                    conlyopts = conlyopts,
                    copts = copts,
                    copts_filter = copts_filter,
                    cpp_configuration = cpp_configuration,
                    cxxopts = cxxopts,
                    fdo_context = fdo_context,
                    auxiliary_fdo_inputs = auxiliary_fdo_inputs,
                    feature_configuration = feature_configuration,
                    is_code_coverage_enabled = is_code_coverage_enabled,
                    label = label,
                    common_toolchain_variables = common_compile_build_variables,
                    fdo_build_variables = fdo_build_variables,
                    cpp_semantics = cpp_semantics,
                    outputs = outputs,
                    source_label = module_map_label,
                    module = module,
                    cpp_compile_action_builder = cpp_compile_action_builder,
                )

    output_name_prefix_dir = cc_internal.compute_output_name_prefix_dir(configuration = configuration, purpose = purpose)
    output_name_map = _calculate_output_name_map_by_type(compilation_unit_sources, output_name_prefix_dir)

    compiled_basenames = set()
    for cpp_source in compilation_unit_sources.values():
        source_artifact = cpp_source.file
        if not cc_internal.is_tree_artifact(source_artifact) and cpp_source.type == CPP_SOURCE_TYPE_HEADER:
            continue

        output_name = output_name_map[source_artifact]
        source_label = cpp_source.label
        bitcode_output = feature_configuration.is_enabled("thin_lto") and (("." + source_artifact.extension) in LTO_SOURCE_EXTENSIONS)

        cpp_compile_action_builder = cc_internal.create_cpp_compile_action_builder_with_inputs(
            action_construction_context = action_construction_context,
            cc_compilation_context = cc_compilation_context,
            cc_toolchain = cc_toolchain,
            configuration = configuration,
            copts_filter = copts_filter,
            feature_configuration = feature_configuration,
            semantics = cpp_semantics,
            source_artifact = source_artifact,
            additional_compilation_inputs = additional_compilation_inputs,
            additional_include_scanning_roots = additional_include_scanning_roots,
        )

        if not cc_internal.is_tree_artifact(source_artifact):
            compiled_basenames.add(_basename_without_extension(source_artifact))
            cc_internal.create_compile_source_action_from_builder(
                action_construction_context = action_construction_context,
                cc_compilation_context = cc_compilation_context,
                cc_toolchain = cc_toolchain,
                configuration = configuration,
                conlyopts = conlyopts,
                copts = copts,
                cpp_configuration = cpp_configuration,
                cxxopts = cxxopts,
                fdo_context = fdo_context,
                auxiliary_fdo_inputs = auxiliary_fdo_inputs,
                feature_configuration = feature_configuration,
                generate_no_pic_action = generate_no_pic_action,
                generate_pic_action = generate_pic_action,
                label = label,
                common_compile_build_variables = common_compile_build_variables,
                fdo_build_variables = fdo_build_variables,
                cpp_semantics = cpp_semantics,
                source_label = source_label,
                output_name = output_name,
                outputs = outputs,
                source_artifact = source_artifact,
                cpp_compile_action_builder = cpp_compile_action_builder,
                output_category = artifact_category.CLIF_OUTPUT_PROTO if cpp_source.type == CPP_SOURCE_TYPE_CLIF_INPUT_PROTO else artifact_category.OBJECT_FILE,
                cpp_module_map = cc_compilation_context.module_map(),
                add_object = True,
                enable_coverage = is_code_coverage_enabled,
                generate_dwo = should_create_per_object_debug_info(feature_configuration, cpp_configuration),
                bitcode_output = bitcode_output,
            )
        else:  # Tree artifact
            if cpp_source.type not in [CPP_SOURCE_TYPE_SOURCE, CPP_SOURCE_TYPE_HEADER]:
                fail("Encountered invalid source types when creating CppCompileActionTemplates: " + cpp_source.type)
            if cpp_source.type == CPP_SOURCE_TYPE_HEADER:
                header_token_file = cc_internal.create_compile_action_template(
                    action_construction_context = action_construction_context,
                    cc_compilation_context = cc_compilation_context,
                    cc_toolchain = cc_toolchain,
                    configuration = configuration,
                    conlyopts = conlyopts,
                    copts = copts,
                    cpp_configuration = cpp_configuration,
                    cxxopts = cxxopts,
                    fdo_context = fdo_context,
                    auxiliary_fdo_inputs = auxiliary_fdo_inputs,
                    feature_configuration = feature_configuration,
                    label = label,
                    common_compile_build_variables = common_compile_build_variables,
                    fdo_build_variables = fdo_build_variables,
                    cpp_semantics = cpp_semantics,
                    source = cpp_source,
                    output_name = output_name,
                    cpp_compile_action_builder = cpp_compile_action_builder,
                    outputs = outputs,
                    output_categories = [artifact_category.GENERATED_HEADER, artifact_category.PROCESSED_HEADER],
                    use_pic = generate_pic_action,
                    bitcode_output = bitcode_output,
                )
                outputs.add_header_token_file(header_token_file)
            else:  # CPP_SOURCE_TYPE_SOURCE
                if generate_no_pic_action:
                    object_file = cc_internal.create_compile_action_template(
                        action_construction_context = action_construction_context,
                        cc_compilation_context = cc_compilation_context,
                        cc_toolchain = cc_toolchain,
                        configuration = configuration,
                        conlyopts = conlyopts,
                        copts = copts,
                        cpp_configuration = cpp_configuration,
                        cxxopts = cxxopts,
                        fdo_context = fdo_context,
                        auxiliary_fdo_inputs = auxiliary_fdo_inputs,
                        feature_configuration = feature_configuration,
                        label = label,
                        common_compile_build_variables = common_compile_build_variables,
                        fdo_build_variables = fdo_build_variables,
                        cpp_semantics = cpp_semantics,
                        source = cpp_source,
                        output_name = output_name,
                        cpp_compile_action_builder = cpp_compile_action_builder,
                        outputs = outputs,
                        output_categories = [artifact_category.OBJECT_FILE],
                        use_pic = False,
                        bitcode_output = feature_configuration.is_enabled("thin_lto"),
                    )
                    outputs.add_object_file(object_file)
                if generate_pic_action:
                    pic_object_file = cc_internal.create_compile_action_template(
                        action_construction_context = action_construction_context,
                        cc_compilation_context = cc_compilation_context,
                        cc_toolchain = cc_toolchain,
                        configuration = configuration,
                        conlyopts = conlyopts,
                        copts = copts,
                        cpp_configuration = cpp_configuration,
                        cxxopts = cxxopts,
                        fdo_context = fdo_context,
                        auxiliary_fdo_inputs = auxiliary_fdo_inputs,
                        feature_configuration = feature_configuration,
                        label = label,
                        common_compile_build_variables = common_compile_build_variables,
                        fdo_build_variables = fdo_build_variables,
                        cpp_semantics = cpp_semantics,
                        source = cpp_source,
                        output_name = output_name,
                        cpp_compile_action_builder = cpp_compile_action_builder,
                        outputs = outputs,
                        output_categories = [artifact_category.PIC_OBJECT_FILE],
                        use_pic = True,
                        bitcode_output = feature_configuration.is_enabled("thin_lto"),
                    )
                    outputs.add_pic_object_file(pic_object_file)

    for cpp_source in compilation_unit_sources.values():
        source_artifact = cpp_source.file
        if cpp_source.type != CPP_SOURCE_TYPE_HEADER or cc_internal.is_tree_artifact(source_artifact):
            continue
        if (feature_configuration.is_enabled("validates_layering_check_in_textual_hdrs") and
            _basename_without_extension(source_artifact) in compiled_basenames):
            continue

        output_name = output_name_map[source_artifact]

        cpp_compile_action_builder = cc_internal.create_cpp_compile_action_builder_with_inputs(
            action_construction_context = action_construction_context,
            cc_compilation_context = cc_compilation_context,
            cc_toolchain = cc_toolchain,
            configuration = configuration,
            copts_filter = copts_filter,
            feature_configuration = feature_configuration,
            semantics = cpp_semantics,
            source_artifact = source_artifact,
            additional_compilation_inputs = additional_compilation_inputs,
            additional_include_scanning_roots = additional_include_scanning_roots,
        )
        cc_internal.create_parse_header_action(
            action_construction_context = action_construction_context,
            cc_compilation_context = cc_compilation_context,
            cc_toolchain = cc_toolchain,
            configuration = configuration,
            conlyopts = conlyopts,
            copts = copts,
            cpp_configuration = cpp_configuration,
            cxxopts = cxxopts,
            fdo_context = fdo_context,
            auxiliary_fdo_inputs = auxiliary_fdo_inputs,
            feature_configuration = feature_configuration,
            use_pic = generate_pic_action,
            label = label,
            common_compile_build_variables = common_compile_build_variables,
            fdo_build_variables = fdo_build_variables,
            cpp_semantics = cpp_semantics,
            source_label = cpp_source.label,
            output_name = output_name,
            outputs = outputs,
            cpp_compile_action_builder = cpp_compile_action_builder,
        )

def _calculate_output_name_map_by_type(sources, prefix_dir):
    return (
        _calculate_output_name_map(
            _get_source_artifacts_by_type(
                sources,
                CPP_SOURCE_TYPE_SOURCE,
            ),
            prefix_dir,
        ) |
        _calculate_output_name_map(
            _get_source_artifacts_by_type(
                sources,
                CPP_SOURCE_TYPE_HEADER,
            ),
            prefix_dir,
        ) |
        _calculate_output_name_map(
            _get_source_artifacts_by_type(
                sources,
                CPP_SOURCE_TYPE_CLIF_INPUT_PROTO,
            ),
            prefix_dir,
        )
    )

def _calculate_output_name_map(source_artifacts, prefix_dir):
    """Calculates the output names for object file paths from a set of source files."""
    output_name_map = {}
    count = {}
    number = {}
    for source_artifact in source_artifacts:
        output_name_lowercase = _basename_without_extension(source_artifact).lower()
        count[output_name_lowercase] = count.get(output_name_lowercase, 0) + 1

    for source_artifact in source_artifacts:
        output_name = _basename_without_extension(source_artifact)
        output_name_lowercase = output_name.lower()
        if count.get(output_name_lowercase, 0) >= 2:
            num = number.get(output_name_lowercase, 0)
            number[output_name_lowercase] = num + 1
            output_name = "%s/%s" % (num, output_name)
        if prefix_dir:
            output_name = "%s/%s" % (prefix_dir, output_name)
        output_name_map[source_artifact] = output_name
    return output_name_map

def _get_source_artifacts_by_type(sources, source_type):
    return [cpp_source.file for cpp_source in sources.values() if cpp_source.type == source_type]

def _basename_without_extension(filename):
    return paths.split_extension(filename.basename)[0]
