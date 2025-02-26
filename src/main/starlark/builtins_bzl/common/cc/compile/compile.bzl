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

cc_common_internal = _builtins.internal.cc_common

EMPTY_DICT = {}
LANGUAGE_CPP = "c++"

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
        variables_extension = EMPTY_DICT,
        language = LANGUAGE_CPP,
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
        dependent rules.
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

    Returns tuple of  (<code>CompilationContext</code>, <code>CcCompilationOutputs</code>).
    """
    cc_common_internal.validate_starlark_compile_api_call(
        actions = actions,
        include_prefix = include_prefix,
        strip_include_prefix = strip_include_prefix,
        additional_include_scanning_roots = additional_include_scanning_roots,
    )
    return cc_common_internal.compile(
        actions = actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        srcs = srcs,
        public_hdrs = public_hdrs,
        private_hdrs = private_hdrs,
        textual_hdrs = textual_hdrs,
        additional_exported_hdrs = additional_exported_hdrs,
        includes = includes,
        loose_includes = loose_includes,
        quote_includes = quote_includes,
        system_includes = system_includes,
        framework_includes = framework_includes,
        defines = defines,
        local_defines = local_defines,
        include_prefix = include_prefix,
        strip_include_prefix = strip_include_prefix,
        user_compile_flags = user_compile_flags,
        conly_flags = conly_flags,
        cxx_flags = cxx_flags,
        compilation_contexts = compilation_contexts,
        implementation_compilation_contexts = implementation_compilation_contexts,
        name = name,
        disallow_pic_outputs = disallow_pic_outputs,
        disallow_nopic_outputs = disallow_nopic_outputs,
        additional_include_scanning_roots = additional_include_scanning_roots,
        additional_inputs = additional_inputs,
        module_map = module_map,
        additional_module_maps = additional_module_maps,
        propagate_module_map_to_compile_action = propagate_module_map_to_compile_action,
        do_not_generate_module_map = do_not_generate_module_map,
        code_coverage_enabled = code_coverage_enabled,
        hdrs_checking_mode = hdrs_checking_mode,
        variables_extension = variables_extension,
        language = language,
        purpose = purpose,
        copts_filter = copts_filter,
        separate_module_headers = separate_module_headers,
        module_interfaces = module_interfaces,
        non_compilation_additional_inputs = non_compilation_additional_inputs,
    )
