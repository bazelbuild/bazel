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

load(":common/cc/cc_helper_internal.bzl", "extensions")
load(":common/cc/compile/cc_compilation_helper.bzl", "cc_compilation_helper")

cc_common_internal = _builtins.internal.cc_common
cc_internal = _builtins.internal.cc_internal

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

    cc_outputs = cc_common_internal.create_cc_compile_actions(
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
        variables_extension = variables_extension,
        language = language,
    )

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
                type = "HEADER",
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
                type = "CLIF_INPUT_PROTO" if "." + source.extension in extensions.CLIF_INPUT_PROTO else "SOURCE",
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

# LINT.ThenChange(//src/main/java/com/google/devtools/build/lib/rules/cpp/CcModule.java:compile)
