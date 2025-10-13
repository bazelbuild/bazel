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
    "CPP_SOURCE_TYPE_CLIF_INPUT_PROTO",
    "CPP_SOURCE_TYPE_HEADER",
    "CPP_SOURCE_TYPE_SOURCE",
    "extensions",
    "should_create_per_object_debug_info",
    artifact_category = "artifact_category_names",
)
load(":common/cc/compile/cc_compilation_helper.bzl", "cc_compilation_helper", "dotd_files_enabled", "serialized_diagnostics_file_enabled")
load(":common/cc/compile/cc_compilation_outputs.bzl", "create_compilation_outputs_internal")
load(":common/cc/compile/compile_action_templates.bzl", "create_compile_action_templates")
load(
    ":common/cc/compile/compile_build_variables.bzl",
    "get_copts",
    "get_fdo_variables_and_inputs",
    "get_specific_compile_build_variables",
    "setup_common_compile_build_variables",
)
load(":common/cc/compile/lto_compilation_context.bzl", "create_lto_compilation_context")
load(":common/cc/semantics.bzl", _starlark_cc_semantics = "semantics")
load(":common/paths.bzl", "paths")

cc_common_internal = _builtins.internal.cc_common
cc_internal = _builtins.internal.cc_internal

_VALID_CPP_SOURCE_TYPES = set([CPP_SOURCE_TYPE_SOURCE, CPP_SOURCE_TYPE_HEADER, CPP_SOURCE_TYPE_CLIF_INPUT_PROTO])

def _cpp_source_init(*, label, source, type):
    if type not in _VALID_CPP_SOURCE_TYPES:
        fail("invalid type of cpp source, got:", type, "expected one of:", _VALID_CPP_SOURCE_TYPES)
    return {
        "file": source,
        "label": label,
        "type": type,
    }

# buildifier: disable=unused-variable
_CppSourceInfo, __new_cpp_source_info = provider(
    "A source file that is an input to a c++ compilation.",
    fields = ["file", "label", "type"],
    init = _cpp_source_init,
)

SOURCE_CATEGORY_CC = set(
    extensions.CC_SOURCE +
    extensions.CC_HEADER +
    extensions.C_SOURCE +
    extensions.ASSEMBLER +
    extensions.ASSESMBLER_WITH_C_PREPROCESSOR +
    extensions.CLIF_INPUT_PROTO,
)

# LINT.IfChange(cc_and_objc_file_types)
SOURCE_CATEGORY_CC_AND_OBJC = set(
    extensions.CC_SOURCE +
    extensions.CC_HEADER +
    extensions.OBJC_SOURCE +
    extensions.OBJCPP_SOURCE +
    extensions.C_SOURCE +
    extensions.ASSEMBLER +
    extensions.ASSESMBLER_WITH_C_PREPROCESSOR,
)
# LINT.ThenChange(//src/main/java/com/google/devtools/build/lib/rules/cpp/CppCompileActionTemplate.java:cc_and_objc_file_types)

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
    ctx = cc_internal.actions2ctx_cheat(actions)
    _starlark_cc_semantics.validate_cc_compile_call(
        label = ctx.label,
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
    if type(includes) == "depset":
        includes = includes.to_list()
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
        additional_exported_headers =
            additional_exported_hdrs + [h.path for h in textual_hdrs_list] if textual_hdrs_list else additional_exported_hdrs,
        deps = compilation_contexts,
        implementation_deps = implementation_compilation_contexts,
        additional_cpp_module_maps = additional_module_maps,
    )

    if implementation_compilation_contexts and not implementation_deps_context:
        fail("Compilation context for implementation deps was not created")
    cc_compilation_context = implementation_deps_context if implementation_compilation_contexts else public_compilation_context

    if feature_configuration.is_enabled("header_modules") and not public_compilation_context._module_map:
        fail("All cc rules must support module maps.")

    common_compile_build_variables = setup_common_compile_build_variables(
        cc_compilation_context = cc_compilation_context,
        cc_toolchain = cc_toolchain,
        cpp_configuration = cpp_configuration,
        fdo_context = fdo_context,
        feature_configuration = feature_configuration,
        variables_extension = variables_extension,
    )

    fdo_build_variables, auxiliary_fdo_inputs = get_fdo_variables_and_inputs(
        cc_toolchain = cc_toolchain,
        fdo_context = fdo_context,
        feature_configuration = feature_configuration,
        cpp_configuration = cpp_configuration,
    )

    compilation_outputs_dict = {
        "objects": [],
        "pic_objects": [],
        "temps": [],
        "header_tokens": [],
        "module_files": [],
        "lto_compilation_context": {},
        "gcno_files": [],
        "pic_gcno_files": [],
        "dwo_files": [],
        "pic_dwo_files": [],
    }
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
        outputs = compilation_outputs_dict,
        common_compile_build_variables = common_compile_build_variables,
        auxiliary_fdo_inputs = auxiliary_fdo_inputs,
        fdo_build_variables = fdo_build_variables,
    )

    compilation_outputs_dict["lto_compilation_context"] = create_lto_compilation_context(
        objects = compilation_outputs_dict["lto_compilation_context"],
    )
    compilation_outputs = create_compilation_outputs_internal(**compilation_outputs_dict)

    if cpp_configuration.process_headers_in_dependencies():
        compilation_context = cc_internal.create_cc_compilation_context_with_extra_header_tokens(
            cc_compilation_context = public_compilation_context,
            extra_header_tokens = compilation_outputs._header_tokens,
        )
    else:
        compilation_context = public_compilation_context
    return (compilation_context, compilation_outputs)

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
            compilation_unit_sources[header] = _CppSourceInfo(
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
            compilation_unit_sources[source] = _CppSourceInfo(
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
        cpp_configuration,  # Note: this is from the cc_toolchain, and is NOT the same as ctx.fragments.cpp
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

    native_cc_semantics = cc_common_internal.get_cc_semantics(language = language)

    if _should_provide_header_modules(feature_configuration, private_headers, public_headers):
        cpp_module_map = cc_compilation_context._module_map
        module_map_label = Label(cpp_module_map.name())
        modules = _create_module_action(
            action_construction_context = action_construction_context,
            cc_compilation_context = cc_compilation_context,
            cc_toolchain = cc_toolchain,
            configuration = configuration,
            conlyopts = conlyopts,
            copts = copts,
            cpp_configuration = cpp_configuration,
            cxxopts = cxxopts,
            copts_filter = copts_filter,
            fdo_context = fdo_context,
            auxiliary_fdo_inputs = auxiliary_fdo_inputs,
            feature_configuration = feature_configuration,
            generate_no_pic_action = generate_no_pic_action,
            generate_pic_action = generate_pic_action,
            label = label,
            common_compile_build_variables = common_compile_build_variables,
            fdo_build_variables = fdo_build_variables,
            native_cc_semantics = native_cc_semantics,
            outputs = outputs,
            cpp_module_map = cpp_module_map,
            language = language,
            additional_compilation_inputs = [],
            additional_include_scanning_roots = [],
        )
        if separate_module_headers:
            separate_cpp_module_map = cpp_module_map.create_separate_module_map()
            separate_modules = _create_module_action(
                action_construction_context = action_construction_context,
                cc_compilation_context = cc_compilation_context,
                cc_toolchain = cc_toolchain,
                configuration = configuration,
                conlyopts = conlyopts,
                copts = copts,
                cpp_configuration = cpp_configuration,
                cxxopts = cxxopts,
                copts_filter = copts_filter,
                fdo_context = fdo_context,
                auxiliary_fdo_inputs = auxiliary_fdo_inputs,
                feature_configuration = feature_configuration,
                generate_no_pic_action = generate_no_pic_action,
                generate_pic_action = generate_pic_action,
                label = label,
                common_compile_build_variables = common_compile_build_variables,
                fdo_build_variables = fdo_build_variables,
                native_cc_semantics = native_cc_semantics,
                outputs = outputs,
                cpp_module_map = separate_cpp_module_map,
                language = language,
                additional_compilation_inputs = [],
                additional_include_scanning_roots = [],
            )
            modules = modules + separate_modules
        if feature_configuration.is_enabled("header_module_codegen"):
            for module in modules:
                _create_module_codegen_action(
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
                    cpp_semantics = native_cc_semantics,
                    language = language,
                    outputs = outputs,
                    source_label = module_map_label,
                    module = module,
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

        if not cc_internal.is_tree_artifact(source_artifact):
            compiled_basenames.add(_basename_without_extension(source_artifact))
            _create_pic_nopic_compile_source_actions(
                action_construction_context = action_construction_context,
                cc_compilation_context = cc_compilation_context,
                label = label,
                source_label = source_label,
                source_artifact = source_artifact,
                output_name = output_name,
                outputs = outputs,
                cc_toolchain = cc_toolchain,
                feature_configuration = feature_configuration,
                configuration = configuration,
                cpp_configuration = cpp_configuration,
                native_cc_semantics = native_cc_semantics,
                language = language,
                conlyopts = conlyopts,
                cxxopts = cxxopts,
                copts_filter = copts_filter,
                copts = copts,
                common_compile_variables = common_compile_build_variables,
                fdo_build_variables = fdo_build_variables,
                output_category = (artifact_category.CLIF_OUTPUT_PROTO if cpp_source.type == CPP_SOURCE_TYPE_CLIF_INPUT_PROTO else artifact_category.OBJECT_FILE),
                cpp_module_map = cc_compilation_context._module_map,
                add_object = True,
                enable_coverage = is_code_coverage_enabled,
                generate_dwo = should_create_per_object_debug_info(feature_configuration, cpp_configuration),
                bitcode_output = bitcode_output,
                fdo_context = fdo_context,
                auxiliary_fdo_inputs = auxiliary_fdo_inputs,
                additional_compilation_inputs = additional_compilation_inputs,
                additional_include_scanning_roots = additional_include_scanning_roots,
                generate_pic_action = generate_pic_action,
                generate_no_pic_action = generate_no_pic_action,
            )
        else:  # Tree artifact
            create_compile_action_templates(
                action_construction_context = action_construction_context,
                cc_compilation_context = cc_compilation_context,
                cc_toolchain = cc_toolchain,
                configuration = configuration,
                cpp_configuration = cpp_configuration,
                feature_configuration = feature_configuration,
                native_cc_semantics = native_cc_semantics,
                language = language,
                common_compile_build_variables = common_compile_build_variables,
                cpp_source = cpp_source,
                source_artifact = source_artifact,
                label = label,
                copts = copts,
                conlyopts = conlyopts,
                cxxopts = cxxopts,
                copts_filter = copts_filter,
                generate_pic_action = generate_pic_action,
                generate_no_pic_action = generate_no_pic_action,
                additional_compilation_inputs = additional_compilation_inputs,
                additional_include_scanning_roots = additional_include_scanning_roots,
                output_name = output_name,
                outputs = outputs,
                bitcode_output = bitcode_output,
            )
    for cpp_source in compilation_unit_sources.values():
        source_artifact = cpp_source.file
        if cpp_source.type != CPP_SOURCE_TYPE_HEADER or cc_internal.is_tree_artifact(source_artifact):
            continue
        if (feature_configuration.is_enabled("validates_layering_check_in_textual_hdrs") and
            _basename_without_extension(source_artifact) in compiled_basenames):
            continue

        output_name_base = cc_internal.get_artifact_name_for_category(
            cc_toolchain = cc_toolchain,
            category = artifact_category.GENERATED_HEADER,
            output_name = output_name_map[source_artifact],
        )
        output_file = _get_compile_output_file(
            action_construction_context,
            label,
            configuration = configuration,
            output_name = cc_internal.get_artifact_name_for_category(
                cc_toolchain = cc_toolchain,
                category = artifact_category.PROCESSED_HEADER,
                output_name = output_name_base,
            ),
        )
        dotd_file = _get_compile_output_file(
            action_construction_context,
            label,
            configuration = configuration,
            output_name = cc_internal.get_artifact_name_for_category(
                cc_toolchain = cc_toolchain,
                category = artifact_category.INCLUDED_FILE_LIST,
                output_name = output_name_base,
            ),
        ) if (
            dotd_files_enabled(native_cc_semantics, configuration, feature_configuration) and
            _use_dotd_file(feature_configuration, source_artifact)
        ) else None
        diagnostics_file = _get_compile_output_file(
            action_construction_context,
            label,
            configuration = configuration,
            output_name = cc_internal.get_artifact_name_for_category(
                cc_toolchain = cc_toolchain,
                category = artifact_category.SERIALIZED_DIAGNOSTICS_FILE,
                output_name = output_name_base,
            ),
        ) if serialized_diagnostics_file_enabled(feature_configuration) else None
        specific_compile_build_variables = get_specific_compile_build_variables(
            feature_configuration,
            use_pic = generate_pic_action,
            source_file = source_artifact,
            output_file = output_file,
            dotd_file = dotd_file,
            diagnostics_file = diagnostics_file,
            cpp_module_map = cc_compilation_context._module_map,
            direct_module_maps = cc_compilation_context._direct_module_maps_set,
            user_compile_flags = get_copts(
                language = language,
                cpp_configuration = cpp_configuration,
                source_file = source_artifact,
                conlyopts = conlyopts,
                copts = copts,
                cxxopts = cxxopts,
                label = cpp_source.label,
            ),
        )
        compile_variables = cc_internal.combine_cc_toolchain_variables(
            common_compile_build_variables,
            specific_compile_build_variables,
        )

        # This creates the action to parse a header file.
        # If we generate pic actions, we prefer the header actions to use the pic artifacts.
        cc_internal.create_cc_compile_action(
            action_construction_context = action_construction_context,
            cc_compilation_context = cc_compilation_context,
            cc_toolchain = cc_toolchain,
            configuration = configuration,
            copts_filter = copts_filter,
            feature_configuration = feature_configuration,
            cc_semantics = native_cc_semantics,
            source = source_artifact,
            additional_compilation_inputs = additional_compilation_inputs,
            additional_include_scanning_roots = additional_include_scanning_roots,
            output_file = output_file,
            dotd_file = dotd_file,
            diagnostics_file = diagnostics_file,
            use_pic = generate_pic_action,
            compile_build_variables = compile_variables,
        )
        outputs["header_tokens"].append(output_file)

def _create_pic_nopic_compile_source_actions(
        action_construction_context,
        cc_compilation_context,
        label,
        source_label,
        source_artifact,
        output_name,
        outputs,
        cc_toolchain,
        feature_configuration,
        configuration,
        cpp_configuration,
        native_cc_semantics,
        language,
        conlyopts,
        copts,
        cxxopts,
        copts_filter,
        common_compile_variables,
        fdo_build_variables,
        output_category,
        cpp_module_map,
        add_object,
        enable_coverage,
        generate_dwo,
        bitcode_output,
        fdo_context,
        auxiliary_fdo_inputs,
        additional_compilation_inputs,
        additional_include_scanning_roots,
        generate_pic_action,
        generate_no_pic_action):
    results = []
    if generate_pic_action:
        pic_object = _create_compile_source_action(
            action_construction_context = action_construction_context,
            cc_compilation_context = cc_compilation_context,
            label = label,
            source_label = source_label,
            source_artifact = source_artifact,
            output_name = output_name,
            outputs = outputs,
            cc_toolchain = cc_toolchain,
            feature_configuration = feature_configuration,
            configuration = configuration,
            cpp_configuration = cpp_configuration,
            native_cc_semantics = native_cc_semantics,
            language = language,
            conlyopts = conlyopts,
            copts = copts,
            cxxopts = cxxopts,
            copts_filter = copts_filter,
            common_compile_variables = common_compile_variables,
            fdo_build_variables = fdo_build_variables,
            output_category = output_category,
            cpp_module_map = cpp_module_map,
            add_object = add_object,
            enable_coverage = enable_coverage,
            generate_dwo = generate_dwo,
            bitcode_output = bitcode_output,
            fdo_context = fdo_context,
            auxiliary_fdo_inputs = auxiliary_fdo_inputs,
            additional_compilation_inputs = additional_compilation_inputs,
            additional_include_scanning_roots = additional_include_scanning_roots,
            use_pic = True,
        )
        results.append(pic_object)
        if output_category == artifact_category.CPP_MODULE:
            outputs["module_files"].append(pic_object)

    if generate_no_pic_action:
        nopic_object = _create_compile_source_action(
            action_construction_context = action_construction_context,
            cc_compilation_context = cc_compilation_context,
            label = label,
            source_label = source_label,
            source_artifact = source_artifact,
            output_name = output_name,
            outputs = outputs,
            cc_toolchain = cc_toolchain,
            feature_configuration = feature_configuration,
            configuration = configuration,
            cpp_configuration = cpp_configuration,
            native_cc_semantics = native_cc_semantics,
            language = language,
            conlyopts = conlyopts,
            copts = copts,
            cxxopts = cxxopts,
            copts_filter = copts_filter,
            common_compile_variables = common_compile_variables,
            fdo_build_variables = fdo_build_variables,
            output_category = output_category,
            cpp_module_map = cpp_module_map,
            add_object = add_object,
            enable_coverage = enable_coverage,
            generate_dwo = generate_dwo,
            bitcode_output = bitcode_output,
            fdo_context = fdo_context,
            auxiliary_fdo_inputs = auxiliary_fdo_inputs,
            additional_compilation_inputs = additional_compilation_inputs,
            additional_include_scanning_roots = additional_include_scanning_roots,
            use_pic = False,
        )
        results.append(nopic_object)
        if output_category == artifact_category.CPP_MODULE:
            outputs["module_files"].append(nopic_object)

    return results

def _create_compile_source_action(
        action_construction_context,
        cc_compilation_context,
        label,
        source_label,
        source_artifact,
        output_name,
        outputs,
        cc_toolchain,
        feature_configuration,
        configuration,
        cpp_configuration,
        native_cc_semantics,
        language,
        conlyopts,
        copts,
        cxxopts,
        copts_filter,
        common_compile_variables,
        fdo_build_variables,
        output_category,
        cpp_module_map,
        add_object,
        enable_coverage,
        generate_dwo,
        bitcode_output,
        fdo_context,
        auxiliary_fdo_inputs,
        additional_compilation_inputs,
        additional_include_scanning_roots,
        use_pic):
    output_pic_nopic_name = output_name
    if use_pic:
        output_pic_nopic_name = cc_internal.get_artifact_name_for_category(
            cc_toolchain = cc_toolchain,
            category = artifact_category.PIC_FILE,
            output_name = output_name,
        )
    object_file = _get_compile_output_file(
        ctx = action_construction_context,
        label = label,
        configuration = configuration,
        output_name = cc_internal.get_artifact_name_for_category(
            cc_toolchain = cc_toolchain,
            category = output_category,
            output_name = output_pic_nopic_name,
        ),
    )
    dotd_file = _maybe_declare_dotd_file(
        ctx = action_construction_context,
        label = label,
        source_artifact = source_artifact,
        category = output_category,
        output_name = output_pic_nopic_name,
        cc_toolchain = cc_toolchain,
        cpp_semantics = native_cc_semantics,
        configuration = configuration,
        feature_configuration = feature_configuration,
    )
    diagnostics_file = _maybe_declare_diagnostics_file(
        ctx = action_construction_context,
        label = label,
        category = output_category,
        output_name = output_pic_nopic_name,
        cc_toolchain = cc_toolchain,
        configuration = configuration,
        feature_configuration = feature_configuration,
    )
    gcno_file = _maybe_declare_gcno_file(
        ctx = action_construction_context,
        label = label,
        output_name = output_pic_nopic_name,
        cc_toolchain = cc_toolchain,
        cpp_configuration = cpp_configuration,
        configuration = configuration,
        enable_coverage = enable_coverage,
    )

    dwo_file = None
    if generate_dwo and not bitcode_output:
        dwo_file_name = paths.replace_extension(paths.basename(object_file.path), ".dwo")

        dwo_file = cc_internal.declare_other_output_file(
            ctx = action_construction_context,
            output_name = dwo_file_name,
            object_file = object_file,
        )

    lto_indexing_file = None
    if bitcode_output and not feature_configuration.is_enabled("no_use_lto_indexing_bitcode_file"):
        lto_indexing_file_name = paths.replace_extension(
            paths.basename(object_file.path),
            extensions.LTO_INDEXING_OBJECT_FILE[0],
        )
        lto_indexing_file = cc_internal.declare_other_output_file(
            ctx = action_construction_context,
            output_name = lto_indexing_file_name,
            object_file = object_file,
        )

    complete_copts = get_copts(
        language = language,
        cpp_configuration = cpp_configuration,
        source_file = source_artifact,
        copts = copts,
        conlyopts = conlyopts,
        cxxopts = cxxopts,
        label = source_label,
    )

    compile_variables = get_specific_compile_build_variables(
        source_file = source_artifact,
        output_file = object_file,
        code_coverage_enabled = enable_coverage,
        gcno_file = gcno_file,
        dwo_file = dwo_file,
        using_fission = generate_dwo,
        lto_indexing_file = lto_indexing_file,
        user_compile_flags = complete_copts,
        dotd_file = dotd_file,
        diagnostics_file = diagnostics_file,
        use_pic = use_pic,
        cpp_module_map = cpp_module_map,
        feature_configuration = feature_configuration,
        direct_module_maps = cc_compilation_context._direct_module_maps_set,
        fdo_build_variables = fdo_build_variables,
        additional_build_variables = {},
    )

    temp_action_outputs = _create_temps_action(
        action_construction_context = action_construction_context,
        cc_compilation_context = cc_compilation_context,
        label = label,
        source_artifact = source_artifact,
        output_name = output_pic_nopic_name,
        cc_toolchain = cc_toolchain,
        configuration = configuration,
        cpp_configuration = cpp_configuration,
        cpp_semantics = native_cc_semantics,
        copts = complete_copts,
        copts_filter = copts_filter,
        common_compile_variables = common_compile_variables,
        feature_configuration = feature_configuration,
        fdo_context = fdo_context,
        fdo_build_variables = fdo_build_variables,
        auxiliary_fdo_inputs = auxiliary_fdo_inputs,
        additional_compilation_inputs = additional_compilation_inputs,
        additional_include_scanning_roots = additional_include_scanning_roots,
        use_pic = use_pic,
    )

    # The fdo_context struct does not always have fields set, so we have to do this.
    # TODO(cmita): This is very error prone and should be fixed.
    fdo_context_has_artifacts = (getattr(fdo_context, "branch_fdo_profile", None) or
                                 getattr(fdo_context, "prefetch_hints_artifact", None) or
                                 getattr(fdo_context, "propeller_optimize_info", None) or
                                 getattr(fdo_context, "memprof_profile_artifact", None))
    additional_inputs = additional_compilation_inputs
    if add_object and fdo_context_has_artifacts:
        additional_inputs = additional_compilation_inputs + auxiliary_fdo_inputs.to_list()

    cc_internal.create_cc_compile_action(
        action_construction_context = action_construction_context,
        cc_compilation_context = cc_compilation_context,
        cc_toolchain = cc_toolchain,
        configuration = configuration,
        copts_filter = copts_filter,
        feature_configuration = feature_configuration,
        cc_semantics = native_cc_semantics,
        additional_compilation_inputs = additional_inputs,
        additional_include_scanning_roots = additional_include_scanning_roots,
        source = source_artifact,
        output_file = object_file,
        diagnostics_file = diagnostics_file,
        dotd_file = dotd_file,
        gcno_file = gcno_file,
        dwo_file = dwo_file,
        use_pic = use_pic,
        lto_indexing_file = lto_indexing_file,
        compile_build_variables = cc_internal.combine_cc_toolchain_variables(
            common_compile_variables,
            compile_variables,
        ),
    )

    outputs["temps"].extend(temp_action_outputs)
    if add_object:
        if use_pic:
            outputs["pic_objects"].append(object_file)
        else:
            outputs["objects"].append(object_file)
    if add_object and bitcode_output:
        outputs["lto_compilation_context"][object_file] = (lto_indexing_file, complete_copts)
    if dwo_file:
        if use_pic:
            outputs["pic_dwo_files"].append(dwo_file)
        else:
            outputs["dwo_files"].append(dwo_file)
    if gcno_file:
        if use_pic:
            outputs["pic_gcno_files"].append(gcno_file)
        else:
            outputs["gcno_files"].append(gcno_file)
    return object_file

def _create_temps_action(
        action_construction_context,
        cc_compilation_context,
        label,
        source_artifact,
        output_name,
        cc_toolchain,
        configuration,
        cpp_configuration,
        cpp_semantics,
        copts,
        copts_filter,
        common_compile_variables,
        feature_configuration,
        fdo_context,
        fdo_build_variables,
        auxiliary_fdo_inputs,
        additional_compilation_inputs,
        additional_include_scanning_roots,
        use_pic):
    if not cpp_configuration.save_temps():
        return []

    filename = paths.basename(source_artifact.path)
    extension = filename[filename.find("."):]
    c_source = extension in extensions.C_SOURCE
    cpp_source = extension in extensions.CC_SOURCE
    objc_source = extension in extensions.OBJC_SOURCE
    objcpp_source = extension in extensions.OBJCPP_SOURCE
    if not c_source and not cpp_source and not objc_source and not objcpp_source:
        return []

    category = artifact_category.PREPROCESSED_C_SOURCE if c_source else artifact_category.PREPROCESSED_CPP_SOURCE

    fdo_context_has_artifacts = (getattr(fdo_context, "branch_fdo_profile", None) or
                                 getattr(fdo_context, "prefetch_hints_artifact", None) or
                                 getattr(fdo_context, "propeller_optimize_info", None) or
                                 getattr(fdo_context, "memprof_profile_artifact", None))
    if fdo_context_has_artifacts:
        additional_compilation_inputs = additional_compilation_inputs + auxiliary_fdo_inputs.to_list()

    preprocess_object_file = _get_compile_output_file(
        ctx = action_construction_context,
        label = label,
        configuration = configuration,
        output_name = cc_internal.get_artifact_name_for_category(
            cc_toolchain = cc_toolchain,
            category = category,
            output_name = output_name,
        ),
    )
    assembly_object_file = _get_compile_output_file(
        ctx = action_construction_context,
        label = label,
        configuration = configuration,
        output_name = cc_internal.get_artifact_name_for_category(
            cc_toolchain = cc_toolchain,
            category = artifact_category.GENERATED_ASSEMBLY,
            output_name = output_name,
        ),
    )
    preprocess_dotd_file = _maybe_declare_dotd_file(
        ctx = action_construction_context,
        label = label,
        output_name = output_name,
        source_artifact = source_artifact,
        category = category,
        cc_toolchain = cc_toolchain,
        cpp_semantics = cpp_semantics,
        configuration = configuration,
        feature_configuration = feature_configuration,
    )
    assembly_dotd_file = _maybe_declare_dotd_file(
        ctx = action_construction_context,
        label = label,
        output_name = output_name,
        source_artifact = source_artifact,
        category = artifact_category.GENERATED_ASSEMBLY,
        cc_toolchain = cc_toolchain,
        cpp_semantics = cpp_semantics,
        configuration = configuration,
        feature_configuration = feature_configuration,
    )
    preprocess_diagnostics_file = _maybe_declare_diagnostics_file(
        ctx = action_construction_context,
        label = label,
        category = category,
        output_name = output_name,
        cc_toolchain = cc_toolchain,
        configuration = configuration,
        feature_configuration = feature_configuration,
    )
    assembly_diagnostics_file = _maybe_declare_diagnostics_file(
        ctx = action_construction_context,
        label = label,
        category = artifact_category.GENERATED_ASSEMBLY,
        output_name = output_name,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        configuration = configuration,
    )
    preprocess_compile_variables = get_specific_compile_build_variables(
        source_file = source_artifact,
        output_file = preprocess_object_file,
        code_coverage_enabled = False,
        gcno_file = None,
        dwo_file = None,
        using_fission = False,
        lto_indexing_file = None,
        user_compile_flags = copts,
        dotd_file = preprocess_dotd_file,
        diagnostics_file = preprocess_diagnostics_file,
        use_pic = use_pic,
        cpp_module_map = cc_compilation_context._module_map,
        direct_module_maps = cc_compilation_context._direct_module_maps_set,
        feature_configuration = feature_configuration,
        fdo_build_variables = fdo_build_variables,
        additional_build_variables = {"output_preprocess_file": preprocess_object_file.path},
    )

    assembly_compile_variables = get_specific_compile_build_variables(
        source_file = source_artifact,
        output_file = assembly_object_file,
        code_coverage_enabled = False,
        gcno_file = None,
        dwo_file = None,
        using_fission = False,
        lto_indexing_file = None,
        user_compile_flags = copts,
        dotd_file = assembly_dotd_file,
        diagnostics_file = assembly_diagnostics_file,
        use_pic = use_pic,
        cpp_module_map = cc_compilation_context._module_map,
        direct_module_maps = cc_compilation_context._direct_module_maps_set,
        feature_configuration = feature_configuration,
        fdo_build_variables = fdo_build_variables,
        additional_build_variables = {"output_assembly_file": assembly_object_file.path},
    )
    cc_internal.create_cc_compile_action(
        action_construction_context = action_construction_context,
        cc_compilation_context = cc_compilation_context,
        cc_toolchain = cc_toolchain,
        configuration = configuration,
        copts_filter = copts_filter,
        feature_configuration = feature_configuration,
        cc_semantics = cpp_semantics,
        source = source_artifact,
        additional_compilation_inputs = additional_compilation_inputs,
        additional_include_scanning_roots = additional_include_scanning_roots,
        output_file = preprocess_object_file,
        dotd_file = preprocess_dotd_file,
        diagnostics_file = preprocess_diagnostics_file,
        use_pic = use_pic,
        compile_build_variables = cc_internal.combine_cc_toolchain_variables(
            common_compile_variables,
            preprocess_compile_variables,
        ),
    )
    cc_internal.create_cc_compile_action(
        action_construction_context = action_construction_context,
        cc_compilation_context = cc_compilation_context,
        cc_toolchain = cc_toolchain,
        configuration = configuration,
        copts_filter = copts_filter,
        feature_configuration = feature_configuration,
        cc_semantics = cpp_semantics,
        source = source_artifact,
        additional_compilation_inputs = additional_compilation_inputs,
        additional_include_scanning_roots = additional_include_scanning_roots,
        output_file = assembly_object_file,
        dotd_file = assembly_dotd_file,
        diagnostics_file = assembly_diagnostics_file,
        use_pic = use_pic,
        compile_build_variables = cc_internal.combine_cc_toolchain_variables(
            common_compile_variables,
            assembly_compile_variables,
        ),
    )
    return [preprocess_object_file, assembly_object_file]

def _create_module_codegen_action(
        action_construction_context,
        cc_compilation_context,
        cc_toolchain,
        configuration,
        conlyopts,
        copts,
        copts_filter,
        cpp_configuration,
        cxxopts,
        fdo_context,
        auxiliary_fdo_inputs,
        feature_configuration,
        is_code_coverage_enabled,
        label,
        common_toolchain_variables,
        fdo_build_variables,
        cpp_semantics,
        language,
        source_label,
        module,
        outputs):
    use_pic = ".pic" in module.basename
    output_name = paths.basename(module.basename)

    gcno_file = None
    if is_code_coverage_enabled and not cpp_configuration.use_llvm_coverage_map_format():
        gcno_file = _get_compile_output_file(
            ctx = action_construction_context,
            label = label,
            configuration = configuration,
            output_name = cc_internal.get_artifact_name_for_category(
                cc_toolchain = cc_toolchain,
                category = artifact_category.COVERAGE_DATA_FILE,
                output_name = output_name,
            ),
        )

    bitcode_output = (feature_configuration.is_enabled("thin_lto") and
                      paths.split_extension(module.basename)[-1] in LTO_SOURCE_EXTENSIONS)

    # TODO(tejohnson): Add support for ThinLTO if needed.
    if bitcode_output:
        fail("bitcode output not currently supported for feature header_module_codegen")

    complete_copts = get_copts(
        language = language,
        cpp_configuration = cpp_configuration,
        source_file = module,
        conlyopts = conlyopts,
        copts = copts,
        cxxopts = cxxopts,
        label = source_label,
    )

    object_file = _get_compile_output_file(
        ctx = action_construction_context,
        label = label,
        configuration = configuration,
        output_name = cc_internal.get_artifact_name_for_category(
            cc_toolchain = cc_toolchain,
            category = artifact_category.OBJECT_FILE,
            output_name = output_name,
        ),
    )

    dotd_file = None
    if (dotd_files_enabled(cpp_semantics, configuration, feature_configuration) and
        _use_dotd_file(feature_configuration, module)):
        dotd_file = _get_compile_output_file(
            ctx = action_construction_context,
            label = label,
            configuration = configuration,
            output_name = cc_internal.get_artifact_name_for_category(
                cc_toolchain = cc_toolchain,
                category = artifact_category.INCLUDED_FILE_LIST,
                output_name = output_name,
            ),
        )

    diagnostics_file = None
    if feature_configuration.is_enabled("serialized_diagnostics_file"):
        diagnostics_file = _get_compile_output_file(
            ctx = action_construction_context,
            label = label,
            configuration = configuration,
            output_name = cc_internal.get_artifact_name_for_category(
                cc_toolchain = cc_toolchain,
                category = artifact_category.SERIALIZED_DIAGNOSTICS_FILE,
                output_name = output_name,
            ),
        )

    dwo_file = None
    generate_dwo = should_create_per_object_debug_info(feature_configuration, cpp_configuration)
    if generate_dwo:
        dwo_file_name = paths.replace_extension(paths.basename(object_file.path), ".dwo")
        dwo_file = action_construction_context.actions.declare_file(
            dwo_file_name,
            sibling = object_file,
        )

    specific_compile_build_variables = get_specific_compile_build_variables(
        source_file = module,
        output_file = object_file,
        code_coverage_enabled = is_code_coverage_enabled,
        gcno_file = gcno_file,
        dwo_file = dwo_file,
        using_fission = generate_dwo,
        lto_indexing_file = None,
        user_compile_flags = complete_copts,
        dotd_file = dotd_file,
        diagnostics_file = diagnostics_file,
        use_pic = use_pic,
        cpp_module_map = cc_compilation_context._module_map,
        feature_configuration = feature_configuration,
        direct_module_maps = cc_compilation_context._direct_module_maps_set,
        fdo_build_variables = fdo_build_variables,
        additional_build_variables = {},
    )
    compile_variables = cc_internal.combine_cc_toolchain_variables(
        common_toolchain_variables,
        specific_compile_build_variables,
    )

    additional_inputs = []

    # The fdo_context struct does not always have fields set, so we have to do this.
    # TODO(cmita): This is very error prone and should be fixed.
    fdo_context_has_artifacts = (getattr(fdo_context, "branch_fdo_profile", None) or
                                 getattr(fdo_context, "prefetch_hints_artifact", None) or
                                 getattr(fdo_context, "propeller_optimize_info", None) or
                                 getattr(fdo_context, "memprof_profile_artifact", None))

    # This flattening is cheap and is only necessary because get_auxiliary_fdo_inputs creates a
    # depset instead of returning a list, and and create_cc_compile_action() expects a
    # list instead of a depset
    # TODO(cmita): Fix this muddle
    if fdo_context_has_artifacts:
        additional_inputs = auxiliary_fdo_inputs.to_list()

    cc_internal.create_cc_compile_action(
        action_construction_context = action_construction_context,
        cc_compilation_context = cc_compilation_context,
        cc_toolchain = cc_toolchain,
        configuration = configuration,
        copts_filter = copts_filter,
        feature_configuration = feature_configuration,
        cc_semantics = cpp_semantics,
        source = module,
        additional_compilation_inputs = additional_inputs,
        output_file = object_file,
        dotd_file = dotd_file,
        diagnostics_file = diagnostics_file,
        gcno_file = gcno_file,
        dwo_file = dwo_file,
        compile_build_variables = compile_variables,
    )
    if use_pic:
        outputs["pic_objects"].append(object_file)
    else:
        outputs["objects"].append(object_file)

def _create_module_action(
        action_construction_context,
        cc_compilation_context,
        cc_toolchain,
        configuration,
        conlyopts,
        copts,
        cxxopts,
        copts_filter,
        cpp_configuration,
        fdo_context,
        auxiliary_fdo_inputs,
        feature_configuration,
        generate_no_pic_action,
        generate_pic_action,
        label,
        common_compile_build_variables,
        fdo_build_variables,
        native_cc_semantics,
        language,
        cpp_module_map,
        additional_compilation_inputs,
        additional_include_scanning_roots,
        outputs):
    module_map_label = Label(cpp_module_map.name())
    return _create_pic_nopic_compile_source_actions(
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
        common_compile_variables = common_compile_build_variables,
        fdo_build_variables = fdo_build_variables,
        native_cc_semantics = native_cc_semantics,
        source_label = module_map_label,
        output_name = paths.basename(module_map_label.name),
        outputs = outputs,
        source_artifact = cpp_module_map.file(),
        language = language,
        copts_filter = copts_filter,
        output_category = artifact_category.CPP_MODULE,
        cpp_module_map = cpp_module_map,
        add_object = False,
        enable_coverage = False,
        generate_dwo = False,
        bitcode_output = False,
        additional_compilation_inputs = additional_compilation_inputs,
        additional_include_scanning_roots = additional_include_scanning_roots,
    )

def _get_compile_output_file(ctx, label, *, output_name, configuration):
    file = cc_internal.declare_compile_output_file(
        ctx = ctx,
        label = label,
        output_name = output_name,
        configuration = configuration,
    )
    return file

def _use_dotd_file(feature_configuration, source_file):
    extension = "." + source_file.extension if source_file.extension else ""
    header_discover_required = extension not in (extensions.ASSEMBLER + extensions.CPP_MODULE)
    use_header_modules = (
        feature_configuration.is_enabled("use_header_modules") and
        extension in extensions.CC_SOURCE + extensions.CC_HEADER + extensions.CPP_MODULE_MAP
    )
    return header_discover_required and not use_header_modules

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

def _maybe_declare_dotd_file(
        ctx,
        label,
        source_artifact,
        category,
        output_name,
        cc_toolchain,
        cpp_semantics,
        configuration,
        feature_configuration):
    dotd_file = None
    if (dotd_files_enabled(cpp_semantics, configuration, feature_configuration) and
        _use_dotd_file(feature_configuration, source_artifact)):
        dotd_base_name = output_name
        if category != artifact_category.OBJECT_FILE and category != artifact_category.PROCESSED_HEADER:
            dotd_base_name = cc_internal.get_artifact_name_for_category(
                cc_toolchain = cc_toolchain,
                category = category,
                output_name = output_name,
            )
        dotd_name = cc_internal.get_artifact_name_for_category(
            cc_toolchain = cc_toolchain,
            category = artifact_category.INCLUDED_FILE_LIST,
            output_name = dotd_base_name,
        )
        dotd_file = _get_compile_output_file(
            ctx = ctx,
            label = label,
            configuration = configuration,
            output_name = dotd_name,
        )
    return dotd_file

def _maybe_declare_diagnostics_file(
        ctx,
        label,
        category,
        output_name,
        cc_toolchain,
        feature_configuration,
        configuration):
    diagnostics_file = None
    if feature_configuration.is_enabled("serialized_diagnostics_file"):
        base_name = output_name
        if category != artifact_category.OBJECT_FILE and category != artifact_category.PROCESSED_HEADER:
            base_name = cc_internal.get_artifact_name_for_category(
                cc_toolchain = cc_toolchain,
                category = category,
                output_name = output_name,
            )
        diagnostics_file_name = cc_internal.get_artifact_name_for_category(
            cc_toolchain = cc_toolchain,
            category = artifact_category.SERIALIZED_DIAGNOSTICS_FILE,
            output_name = base_name,
        )
        diagnostics_file = _get_compile_output_file(
            ctx = ctx,
            label = label,
            configuration = configuration,
            output_name = diagnostics_file_name,
        )
    return diagnostics_file

def _maybe_declare_gcno_file(
        ctx,
        label,
        output_name,
        cc_toolchain,
        cpp_configuration,
        configuration,
        enable_coverage):
    gcno_file = None
    if enable_coverage and not cpp_configuration.use_llvm_coverage_map_format():
        gcno_file = _get_compile_output_file(
            ctx = ctx,
            label = label,
            configuration = configuration,
            output_name = cc_internal.get_artifact_name_for_category(
                cc_toolchain = cc_toolchain,
                category = artifact_category.COVERAGE_DATA_FILE,
                output_name = output_name,
            ),
        )
    return gcno_file
