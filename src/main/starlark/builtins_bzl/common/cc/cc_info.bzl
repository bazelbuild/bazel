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
# limitations under the License
# LINT.IfChange(forked_exports)
"""
Definition of CcInfo provider.
"""

load(":common/cc/cc_helper_internal.bzl", "check_private_api")
load(":common/cc/link/create_extra_link_time_library.bzl", "create_extra_link_time_libraries", "merge_extra_link_time_libraries")
load(":common/paths.bzl", "paths")

_cc_internal = _builtins.internal.cc_internal

CcCompilationContextInfo = provider(
    "CcCompilationContext",
    fields = {
        # CommandLineCcCompilationContext fields:
        "includes": "Returns the set of search paths (as strings) for header files referenced " +
                    "both by angle bracket and quotes. Usually passed with -I.",
        "quote_includes": "Returns the set of search paths (as strings) for header files " +
                          "referenced by quotes, e.g. #include \"foo/bar/header.h\". They can be " +
                          "either relative to the exec root or absolute. Usually passed with -iquote.",
        "system_includes": "Returns the set of search paths (as strings) for header files " +
                           "referenced by angle brackets, e.g. #include &lt;foo/bar/header.h&gt;. " +
                           "They can be either relative to the exec root or absolute. Usually passed " +
                           "with -isystem.",
        "framework_includes": "Returns the set of search paths (as strings) for framework " +
                              "header files. Usually passed with -F.",
        "external_includes": "Returns the set of search paths (as strings) for external header " +
                             "files referenced by angle bracket. Usually passed with -isystem.",
        "defines": "Returns the set of defines needed to compile this target. Each define is a " +
                   "string. These values are propagated to the target's transitive dependents, " +
                   "that is, any rules that depend on this target.",
        "local_defines": "Returns the set of defines needed to compile this target. Each define " +
                         "is a string. These values are not propagated to the target's transitive " +
                         "dependents.",
        # CcCompilationContext fields:
        "headers": "Returns the set of headers needed to compile this target.",  # == declaredIncludeSrcs
        "validation_artifacts": "Returns the set of validation artifacts.",  # == headerTokens
        "_direct_module_maps": "Internal.",
        "_exporting_module_maps": "Internal",
        "_non_code_inputs": "Internal.",
        "_transitive_modules": "Internal",
        "_transitive_pic_modules": "Internal",
        "_module_map": "Internal",
        "_virtual_to_original_headers": "Internal",
        "_modules_info_files": "Internal",
        "_pic_modules_info_files": "Internal",
        "_module_files": "Internal",
        "_pic_module_files": "Internal",
        # duplicated HeaderInfo fields
        "direct_headers": "Returns the list of modular headers that are declared by this target. " +
                          "This includes both public headers (such as those listed in \"hdrs\") " +
                          " and private headers (such as those listed in \"srcs\").",
        "direct_public_headers": "Returns the list of modular public headers (those listed in " +
                                 "\"hdrs\") that are declared by this target.",
        "direct_private_headers": "Returns the list of modular private headers (those listed in " +
                                  "\"srcs\") that are declared by this target.",
        "direct_textual_headers": "Returns the list of textual headers that are declared by this target.",

        # HeaderInfo:
        "_header_info": "Internal",
    },
)

CcLinkingContextInfo = provider(
    "CcLinkingContextInfo",
    fields = {
        "linker_inputs": "A depset of linker inputs.",
        "_extra_link_time_libraries": "Extra link time libraries.",
    },
)

CcNativeLibraryInfo = provider(
    "CcNativeLibraryInfo",
    fields = ["libraries_to_link"],
)

CcDebugContextInfo = provider(
    doc = """
        C++ debug related objects, specifically when fission is used.
        Stores .dwo files which can be combined into a .dwp in the packaging step.

        <p>It is not expected for this to be used externally at this time. This API is experimental
        and subject to change, and its usage should be restricted to internal packages.
    """,
    fields = {
        "files": """(.dwo files) The .dwo files for non-PIC compilation.
            Returns the .dwo files that should be included in this target's .dwp packaging (if this
            target is linked) or passed through to a dependant's .dwp packaging (e.g. if this is a
            cc_library depended on by a statically linked cc_binary).
            Assumes the corresponding link consumes .o files (vs. .pic.o files).
            """,
        "pic_files": "(.dwo files) The .dwo files for PIC compilation.",
    },
)

EMPTY_COMPILATION_CONTEXT = CcCompilationContextInfo(
    defines = depset(),
    local_defines = depset(),
    headers = depset(),
    direct_headers = [],
    direct_public_headers = [],
    direct_private_headers = [],
    direct_textual_headers = [],
    includes = depset(),
    quote_includes = depset(),
    system_includes = depset(),
    framework_includes = depset(),
    external_includes = depset(),
    validation_artifacts = depset(),
    _virtual_to_original_headers = depset(),
    _module_map = None,
    _exporting_module_maps = [],
    _non_code_inputs = depset(),
    _transitive_modules = depset(),
    _transitive_pic_modules = depset(),
    _direct_module_maps = depset(),
    _header_info = _cc_internal.create_header_info(),
    _module_files = depset(),
    _pic_module_files = depset(),
    _modules_info_files = depset(),
    _pic_modules_info_files = depset(),
)

_EMPTY_LINKING_CONTEXT = CcLinkingContextInfo(
    linker_inputs = depset(),
    _extra_link_time_libraries = None,
)

_EMPTY_DEBUG_CONTEXT = CcDebugContextInfo(
    files = depset(),
    pic_files = depset(),
)

_ModuleMapInfo = provider(
    "ModuleMapInfo",
    fields = {
        "file": "The module map file.",
        "name": "The name of the module.",
    },
)

def create_module_map(*, file, name):
    """
    Creates a module map struct.

    Args:
        file: The module map file.
        name: The name of the module.
    Returns:
        A module map struct.
    """
    check_private_api()
    return _ModuleMapInfo(file = file, name = name)

def create_separate_module_map(module_map):
    """
    Creates a separate module map struct.

    Args:
        module_map: The module map struct.
    Returns:
        A module map struct.
    """
    return _ModuleMapInfo(file = module_map.file, name = module_map.name + ".sep")

def create_linking_context(
        *,
        linker_inputs,
        extra_link_time_library = None):
    """Creates a CcLinkingContextInfo provider.

    Args:
        linker_inputs: A depset of linker inputs.
        extra_link_time_library: An optional extra link time library.

    Returns:
        A CcLinkingContextInfo provider.
    """
    return CcLinkingContextInfo(
        linker_inputs = linker_inputs,
        _extra_link_time_libraries = create_extra_link_time_libraries(extra_link_time_library),
    )

def merge_linking_contexts(*, linking_contexts):
    """Merges a list of CcLinkingContextInfo providers.
    """
    linker_inputs = depset(transitive = [ctx.linker_inputs for ctx in linking_contexts], order = "topological")
    extra_link_time_libraries = merge_extra_link_time_libraries([ctx._extra_link_time_libraries for ctx in linking_contexts if ctx._extra_link_time_libraries != None])
    return CcLinkingContextInfo(
        linker_inputs = linker_inputs,
        _extra_link_time_libraries = extra_link_time_libraries,
    )

def create_debug_context(compilation_outputs):
    """Creates a CcDebugContextInfo from CcCompilationOutputs.

    Args:
        compilation_outputs: A CcCompilationOutputs object.

    Returns:
        A new CcDebugContextInfo object.
    """
    check_private_api()
    return CcDebugContextInfo(
        files = depset(compilation_outputs._dwo_files),
        pic_files = depset(compilation_outputs._pic_dwo_files),
    )

def merge_debug_context(debug_contexts = []):
    """Merge multiple CcDebugContextInfos into one.

    Args:
        debug_contexts: A list of CcDebugContextInfo objects.

    Returns:
        A new CcDebugContextInfo object.
    """
    check_private_api()
    if not debug_contexts:
        return _EMPTY_DEBUG_CONTEXT

    transitive_dwo_files = []
    transitive_pic_dwo_files = []

    for ctx in debug_contexts:
        transitive_dwo_files.append(ctx.files)
        transitive_pic_dwo_files.append(ctx.pic_files)

    return CcDebugContextInfo(
        files = depset(transitive = transitive_dwo_files),
        pic_files = depset(transitive = transitive_pic_dwo_files),
    )

def _create_cc_info(
        *,
        compilation_context = None,
        linking_context = None,
        debug_context = None,
        cc_native_library_info = None):
    return dict(
        compilation_context = compilation_context or EMPTY_COMPILATION_CONTEXT,
        linking_context = linking_context or _EMPTY_LINKING_CONTEXT,
        _debug_context = debug_context or _EMPTY_DEBUG_CONTEXT,
        _legacy_transitive_native_libraries = cc_native_library_info.libraries_to_link if cc_native_library_info else depset(),
    )

CcInfo, _ = provider(
    doc = "Provider for C++ compilation and linking information.",
    fields = {
        "compilation_context": "A `CcCompilationContext`.",
        "linking_context": "A `CcLinkingContext`.",
        "_debug_context": "A `CcDebugInfoContext`.",
        "_legacy_transitive_native_libraries": "A `CcNativeLibraryInfo`.",
    },
    init = _create_cc_info,
)

def _normalize_paths(paths_depset):
    if not paths_depset:
        return depset()
    return depset([paths.normalize(p) for p in paths_depset.to_list()])

def create_compilation_context(
        *,
        headers = None,
        includes = None,
        quote_includes = None,
        system_includes = None,
        framework_includes = None,
        external_includes = None,
        defines = None,
        local_defines = None,
        direct_textual_headers = [],
        direct_public_headers = [],
        direct_private_headers = [],
        dependent_cc_compilation_contexts = [],
        exported_dependent_cc_compilation_contexts = [],
        non_code_inputs = [],
        module_map = None,
        virtual_to_original_headers = None,
        pic_header_module = None,
        header_module = None,
        separate_module_headers = [],
        separate_module = None,
        separate_pic_module = None,
        add_public_headers_to_modular_headers = True):
    """
    Creates CcCompilationContextInfo provider.

    Args:
        headers: A depset of headers to compile.
        includes: A depset of include directories.
        quote_includes: A depset of quoted include directories.
        system_includes: A depset of system include directories.
        framework_includes: A depset of framework include directories.
        external_includes: A depset of external include directories.
        defines: A depset of defines.
        local_defines: A depset of local defines.
        direct_textual_headers: A list of textual headers.
        direct_public_headers: A list of modular public headers.
        direct_private_headers: A list of modular private headers.
        dependent_cc_compilation_contexts: A list of CcCompilationContextInfo providers for
            dependencies.
        exported_dependent_cc_compilation_contexts: A list of CcCompilationContextInfo providers for
            exported dependencies.
        non_code_inputs: A list of non-code inputs.
        module_map: The module map for this target.
        virtual_to_original_headers: A depset of virtual to original header mappings.
        pic_header_module: The PIC header module for this target.
        header_module: The header module for this target.
        separate_module_headers: A list of separate module headers.
        separate_module: The separate module for this target.
        separate_pic_module: The separate PIC module for this target.
        add_public_headers_to_modular_headers: Whether to add public headers to modular headers.
    Returns:
        A CcCompilationContextInfo provider.
    """
    if headers != None and type(headers) != type(depset()):
        fail("for headers, got list, want a depset of File")

    modular_public_hdrs_list = list(direct_public_headers)
    if add_public_headers_to_modular_headers and headers:
        modular_public_hdrs_list.extend(headers.to_list())

    header_info = _cc_internal.create_header_info(
        header_module = header_module,
        pic_header_module = pic_header_module,
        modular_public_headers = modular_public_hdrs_list,
        modular_private_headers = direct_private_headers,
        textual_headers = direct_textual_headers,
        separate_module_headers = separate_module_headers,
        separate_module = separate_module,
        separate_pic_module = separate_pic_module,
    )

    includes = _normalize_paths(includes)
    quote_includes = _normalize_paths(quote_includes)
    system_includes = _normalize_paths(system_includes)
    framework_includes = _normalize_paths(framework_includes)
    external_includes = _normalize_paths(external_includes)

    single_compilation_context = CcCompilationContextInfo(
        defines = defines if defines else depset(),
        local_defines = local_defines if local_defines else depset(),
        headers = headers if headers else depset(),
        # Duplication with HeaderInfo data:
        direct_headers = _cc_internal.freeze(header_info.modular_public_headers + header_info.modular_private_headers + header_info.separate_module_headers),
        direct_public_headers = header_info.modular_public_headers,
        direct_private_headers = header_info.modular_private_headers,
        direct_textual_headers = header_info.textual_headers,
        includes = includes,
        quote_includes = quote_includes,
        system_includes = system_includes,
        framework_includes = framework_includes,
        external_includes = external_includes,
        validation_artifacts = depset(),
        _virtual_to_original_headers = virtual_to_original_headers if virtual_to_original_headers else depset(),
        _module_map = module_map,
        _exporting_module_maps = [],
        _non_code_inputs = depset(non_code_inputs),
        _transitive_modules = depset(),
        _transitive_pic_modules = depset(),
        _direct_module_maps = depset(),
        _header_info = header_info,
        _module_files = depset(),
        _pic_module_files = depset(),
        _modules_info_files = depset(),
        _pic_modules_info_files = depset(),
    )

    return _merge_compilation_contexts(
        compilation_context = single_compilation_context,
        exported_deps = exported_dependent_cc_compilation_contexts,
        deps = dependent_cc_compilation_contexts,
    )

# This class helps create efficient flattened transitive sets across all transitive dependencies.
# For very sparsely populated items, this can be more efficient both in terms of CPU and in terms
# of memory than NestedSets. Merged transitive set will be returned as a flat list to be memory
# efficient. As a further optimization, if a single dependencies contains a superset of all other
# dependencies, its list is simply re-used.
def _flat_depset(*, transitive = []):
    largest_depset = depset()
    largest_depset_list = []
    for t in transitive:
        t_list = t.to_list()
        if len(t_list) > len(largest_depset_list):
            largest_depset_list = t_list
            largest_depset = t

    all = depset(transitive = transitive)
    if all.to_list() == largest_depset_list:
        return largest_depset
    return all

def _merge_compilation_contexts(*, compilation_context = EMPTY_COMPILATION_CONTEXT, exported_deps = [], deps = []):
    direct_module_maps = set()
    exporting_module_maps = set()

    for dep in exported_deps:
        if dep._module_map:
            direct_module_maps.add(dep._module_map.file)
            exporting_module_maps.add(dep._module_map)
        for module_map in dep._exporting_module_maps:
            direct_module_maps.add(module_map.file)
        exporting_module_maps.update(dep._exporting_module_maps)
    for dep in deps:
        if dep._module_map:
            direct_module_maps.add(dep._module_map.file)
        for module_map in dep._exporting_module_maps:
            direct_module_maps.add(module_map.file)

    all_deps = exported_deps + deps
    dep_header_infos = [dep._header_info for dep in all_deps]
    merged_header_infos = [dep._header_info for dep in exported_deps]

    compilation_context_header_info = compilation_context._header_info
    header_info = _cc_internal.create_header_info_with_deps(
        header_info = compilation_context_header_info,
        deps = dep_header_infos,
        merged_deps = merged_header_infos,
    )

    transitive_modules_artifacts = []
    transitive_pic_modules_artifacts = []
    for dep in all_deps:
        dep_header_info = dep._header_info
        if dep_header_info.header_module:
            transitive_modules_artifacts.append(dep_header_info.header_module)
        if dep_header_info.separate_module:
            transitive_modules_artifacts.append(dep_header_info.separate_module)
        if dep_header_info.pic_header_module:
            transitive_pic_modules_artifacts.append(dep_header_info.pic_header_module)
        if dep_header_info.separate_pic_module:
            transitive_pic_modules_artifacts.append(dep_header_info.separate_pic_module)

    return CcCompilationContextInfo(
        includes = _flat_depset(
            transitive = [compilation_context.includes] + [dep.includes for dep in all_deps],
        ),
        quote_includes = _flat_depset(
            transitive = [compilation_context.quote_includes] + [dep.quote_includes for dep in all_deps],
        ),
        system_includes = _flat_depset(
            transitive = [compilation_context.system_includes] + [dep.system_includes for dep in all_deps],
        ),
        framework_includes = _flat_depset(
            transitive = [compilation_context.framework_includes] + [dep.framework_includes for dep in all_deps],
        ),
        external_includes = _flat_depset(
            transitive = [compilation_context.external_includes] + [dep.external_includes for dep in all_deps],
        ),
        defines = _flat_depset(
            transitive = [dep.defines for dep in all_deps] + [compilation_context.defines],
        ),
        local_defines = compilation_context.local_defines,
        headers = depset(
            direct = compilation_context.headers.to_list(),
            transitive = [dep.headers for dep in all_deps],
        ),
        # Duplication with HeaderInfo data:
        direct_headers = _cc_internal.freeze(header_info.modular_public_headers + header_info.modular_private_headers + header_info.separate_module_headers),
        direct_public_headers = header_info.modular_public_headers,
        direct_private_headers = header_info.modular_private_headers,
        direct_textual_headers = header_info.textual_headers,
        _direct_module_maps = depset(list(direct_module_maps)),
        _module_map = compilation_context._module_map,
        _exporting_module_maps = _cc_internal.freeze(exporting_module_maps),
        _non_code_inputs = depset(
            direct = compilation_context._non_code_inputs.to_list(),
            transitive = [dep._non_code_inputs for dep in all_deps],
        ),
        _virtual_to_original_headers = depset(
            transitive = [compilation_context._virtual_to_original_headers] + [dep._virtual_to_original_headers for dep in all_deps],
        ),
        validation_artifacts = depset(
            transitive = [compilation_context.validation_artifacts] + [dep.validation_artifacts for dep in all_deps],
        ),
        _header_info = header_info,
        _transitive_modules = depset(
            transitive_modules_artifacts,
            transitive = [dep._transitive_modules for dep in all_deps],
        ),
        _transitive_pic_modules = depset(
            transitive_pic_modules_artifacts,
            transitive = [dep._transitive_pic_modules for dep in all_deps],
        ),
        _modules_info_files = depset(
            transitive = [compilation_context._modules_info_files] + [dep._modules_info_files for dep in all_deps],
        ),
        _pic_modules_info_files = depset(
            transitive = [compilation_context._pic_modules_info_files] + [dep._pic_modules_info_files for dep in all_deps],
        ),
        _module_files = depset(
            transitive = [compilation_context._module_files] + [dep._module_files for dep in all_deps],
        ),
        _pic_module_files = depset(
            transitive = [compilation_context._pic_module_files] + [dep._pic_module_files for dep in all_deps],
        ),
    )

def merge_compilation_contexts(*, compilation_contexts = []):
    """
    Merges multiple CcCompilationContextInfo providers into one.

    Args:
        compilation_contexts: List of CcCompilationContextInfo providers to be merged.

    Returns:
        Merged CcCompilationContextInfo provider.
    """
    return _merge_compilation_contexts(exported_deps = compilation_contexts)

def create_compilation_context_with_extra_header_tokens(
        *,
        cc_compilation_context,
        extra_header_tokens):
    """
    Creates a CcCompilationContextInfo provider with the given extra header tokens.

    Args:
        cc_compilation_context: The CcCompilationContextInfo provider to copy.
        extra_header_tokens: A list of extra header tokens to add.

    Returns:
        A CcCompilationContextInfo provider with the same data as the given one, but with the
        given extra header tokens.
    """
    return CcCompilationContextInfo(
        defines = cc_compilation_context.defines,
        local_defines = cc_compilation_context.local_defines,
        headers = cc_compilation_context.headers,
        direct_headers = cc_compilation_context.direct_headers,
        direct_public_headers = cc_compilation_context.direct_public_headers,
        direct_private_headers = cc_compilation_context.direct_private_headers,
        direct_textual_headers = cc_compilation_context.direct_textual_headers,
        includes = cc_compilation_context.includes,
        quote_includes = cc_compilation_context.quote_includes,
        system_includes = cc_compilation_context.system_includes,
        framework_includes = cc_compilation_context.framework_includes,
        external_includes = cc_compilation_context.external_includes,
        validation_artifacts = depset(extra_header_tokens, transitive = [cc_compilation_context.validation_artifacts]),
        _virtual_to_original_headers = cc_compilation_context._virtual_to_original_headers,
        _module_map = cc_compilation_context._module_map,
        _exporting_module_maps = cc_compilation_context._exporting_module_maps,
        _non_code_inputs = cc_compilation_context._non_code_inputs,
        _transitive_modules = cc_compilation_context._transitive_modules,
        _transitive_pic_modules = cc_compilation_context._transitive_pic_modules,
        _direct_module_maps = cc_compilation_context._direct_module_maps,
        _header_info = cc_compilation_context._header_info,
        _modules_info_files = cc_compilation_context._modules_info_files,
        _pic_modules_info_files = cc_compilation_context._pic_modules_info_files,
        _module_files = cc_compilation_context._module_files,
        _pic_module_files = cc_compilation_context._pic_module_files,
    )

def create_cc_compilation_context_with_cpp20_modules(
        *,
        cc_compilation_context,
        cpp_module_files,
        pic_cpp_module_files,
        cpp_modules_info_file,
        pic_cpp_modules_info_file):
    """
    Creates a CcCompilationContextInfo provider with C++20 Modules

    Args:
        cc_compilation_context: The CcCompilationContextInfo provider to copy.
        cpp_module_files: C++20 Module files.
        pic_cpp_module_files: PIC C++20 Module files.
        cpp_modules_info_file: C++20 Modules info file.
        pic_cpp_modules_info_file: PIC C++20 Modules info file.

    Returns:
        A CcCompilationContextInfo provider with the same data as the given one, but with the
        C++20 Modules info.
    """
    return CcCompilationContextInfo(
        defines = cc_compilation_context.defines,
        local_defines = cc_compilation_context.local_defines,
        headers = cc_compilation_context.headers,
        direct_headers = cc_compilation_context.direct_headers,
        direct_public_headers = cc_compilation_context.direct_public_headers,
        direct_private_headers = cc_compilation_context.direct_private_headers,
        direct_textual_headers = cc_compilation_context.direct_textual_headers,
        includes = cc_compilation_context.includes,
        quote_includes = cc_compilation_context.quote_includes,
        system_includes = cc_compilation_context.system_includes,
        framework_includes = cc_compilation_context.framework_includes,
        external_includes = cc_compilation_context.external_includes,
        validation_artifacts = cc_compilation_context.validation_artifacts,
        _virtual_to_original_headers = cc_compilation_context._virtual_to_original_headers,
        _module_map = cc_compilation_context._module_map,
        _exporting_module_maps = cc_compilation_context._exporting_module_maps,
        _non_code_inputs = cc_compilation_context._non_code_inputs,
        _transitive_modules = cc_compilation_context._transitive_modules,
        _transitive_pic_modules = cc_compilation_context._transitive_pic_modules,
        _direct_module_maps = cc_compilation_context._direct_module_maps,
        _header_info = cc_compilation_context._header_info,
        _module_files = depset(cpp_module_files, transitive = [cc_compilation_context._module_files]),
        _pic_module_files = depset(pic_cpp_module_files, transitive = [cc_compilation_context._pic_module_files]),
        _modules_info_files = depset([cpp_modules_info_file], transitive = [cc_compilation_context._modules_info_files]),
        _pic_modules_info_files = depset([pic_cpp_modules_info_file], transitive = [cc_compilation_context._pic_modules_info_files]),
    )

def merge_cc_infos(*, direct_cc_infos = [], cc_infos = []):
    """
    Merges multiple `CcInfo`s into one.

    Args:
      direct_cc_infos: List of `CcInfo`s to be merged, whose headers will be exported by
        the direct fields in the returned provider.
      cc_infos: List of `CcInfo`s to be merged, whose headers will not be exported
        by the direct fields in the returned provider.

    Returns:
      Merged CcInfo.
    """
    cc_linking_contexts = []
    cc_debug_info_contexts = []
    transitive_native_cc_libraries = []

    for cc_info in direct_cc_infos:
        cc_linking_contexts.append(cc_info.linking_context)
        cc_debug_info_contexts.append(cc_info._debug_context)
        transitive_native_cc_libraries.append(cc_info._legacy_transitive_native_libraries)

    for cc_info in cc_infos:
        cc_linking_contexts.append(cc_info.linking_context)
        cc_debug_info_contexts.append(cc_info._debug_context)
        transitive_native_cc_libraries.append(cc_info._legacy_transitive_native_libraries)

    return CcInfo(
        compilation_context = _merge_compilation_contexts(
            compilation_context = EMPTY_COMPILATION_CONTEXT,
            exported_deps = [cc_info.compilation_context for cc_info in direct_cc_infos],
            deps = [cc_info.compilation_context for cc_info in cc_infos],
        ),
        linking_context = merge_linking_contexts(linking_contexts = cc_linking_contexts),
        debug_context = merge_debug_context(cc_debug_info_contexts),
        cc_native_library_info = CcNativeLibraryInfo(libraries_to_link = depset(order = "topological", transitive = transitive_native_cc_libraries)),
    )

# LINT.ThenChange(@rules_cc//cc/private/cc_info.bzl:forked_exports)
