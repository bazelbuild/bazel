# Copyright 2020 The Bazel Authors. All rights reserved.
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

"""Common functionality for Objc rules."""

load(":common/cc/cc_info.bzl", "CcInfo")

objc_internal = _builtins.internal.objc_internal
apple_common = _builtins.toplevel.apple_common

CPP_SOURCES = [".cc", ".cpp", ".mm", ".cxx", ".C"]
NON_CPP_SOURCES = [".m", ".c"]
ASSEMBLY_SOURCES = [".s", ".S", ".asm"]
OBJECT_FILE_SOURCES = [".o"]
HEADERS = [".h", ".inc", ".hpp", ".hh"]

COMPILABLE_SRCS = CPP_SOURCES + NON_CPP_SOURCES + ASSEMBLY_SOURCES
SRCS = COMPILABLE_SRCS + OBJECT_FILE_SOURCES + HEADERS
NON_ARC_SRCS = [".m", ".mm"]

extensions = struct(
    CPP_SOURCES = CPP_SOURCES,
    NON_CPP_SOURCES = NON_CPP_SOURCES,
    ASSEMBLY_SOURCES = ASSEMBLY_SOURCES,
    HEADERS = HEADERS,
    SRCS = SRCS,
    NON_ARC_SRCS = NON_ARC_SRCS,
)

def _create_context_and_provider(
        ctx,
        compilation_attributes,
        compilation_artifacts,
        intermediate_artifacts,
        alwayslink,
        has_module_map,
        extra_import_libraries,
        deps,
        implementation_deps,
        attr_linkopts,
        direct_cc_compilation_contexts = []):
    objc_providers = []
    cc_compilation_contexts = []
    cc_linking_contexts = []

    # List of CcLinkingContext to be merged into ObjcProvider, to be done for
    # deps that don't have ObjcProviders.  TODO(waltl): remove after objc link
    # info migration.
    cc_linking_contexts_for_merging = []
    for dep in deps:
        if apple_common.Objc in dep:
            objc_providers.append(dep[apple_common.Objc])
        elif CcInfo in dep:
            # We only use CcInfo's linking info if there is no ObjcProvider.
            # This is required so that objc_library archives do not get treated
            # as if they are from cc targets.
            cc_linking_contexts_for_merging.append(dep[CcInfo].linking_context)

        if CcInfo in dep:
            cc_compilation_contexts.append(dep[CcInfo].compilation_context)
            cc_linking_contexts.append(dep[CcInfo].linking_context)

    implementation_cc_compilation_contexts = []
    for impl_dep in implementation_deps:
        if apple_common.Objc in impl_dep:
            # For implementation deps, we only need to propagate linker inputs
            # with Objc provider, but no compilation artifacts
            # (eg module_map, umbrella_header).
            implementation_dep_objc_provider_kwargs = {
                "force_load_library": impl_dep[apple_common.Objc].force_load_library,
                "imported_library": impl_dep[apple_common.Objc].imported_library,
                "library": impl_dep[apple_common.Objc].library,
                "linkopt": impl_dep[apple_common.Objc].linkopt,
                "sdk_dylib": impl_dep[apple_common.Objc].sdk_dylib,
                "sdk_framework": impl_dep[apple_common.Objc].sdk_framework,
                "source": impl_dep[apple_common.Objc].source,
                "weak_sdk_framework": impl_dep[apple_common.Objc].weak_sdk_framework,
            }
            objc_provider = apple_common.new_objc_provider(**implementation_dep_objc_provider_kwargs)
            objc_providers.append(objc_provider)
        elif CcInfo in impl_dep:
            cc_linking_contexts_for_merging.append(impl_dep[CcInfo].linking_context)

        if CcInfo in impl_dep:
            implementation_cc_compilation_contexts.append(impl_dep[CcInfo].compilation_context)
            cc_linking_contexts.append(impl_dep[CcInfo].linking_context)

    link_order_keys = [
        "imported_library",
        "cc_library",
        "library",
        "force_load_library",
        "linkopt",
    ]
    objc_provider_kwargs = {
        "imported_library": [depset(direct = extra_import_libraries, order = "topological")],
        "weak_sdk_framework": [],
        "sdk_dylib": [],
        "linkopt": [],
        "library": [],
        "providers": objc_providers,
        "cc_library": [],
        "sdk_framework": [],
        "force_load_library": [],
        "umbrella_header": [],
        "module_map": [],
        "source": [],
    }

    objc_compilation_context_kwargs = {
        "providers": objc_providers,
        "cc_compilation_contexts": cc_compilation_contexts,
        "implementation_cc_compilation_contexts": implementation_cc_compilation_contexts,
        "public_hdrs": [],
        "private_hdrs": [],
        "public_textual_hdrs": [],
        "defines": [],
        "includes": [],
        "direct_cc_compilation_contexts": direct_cc_compilation_contexts,
    }

    # Merge cc_linking_context's library and linkopt information into
    # objc_provider.
    all_non_sdk_linkopts = []
    for cc_linking_context in cc_linking_contexts_for_merging:
        if not ctx.fragments.objc.linking_info_migration:
            linkopts = []
            for linker_input in cc_linking_context.linker_inputs.to_list():
                linkopts.extend(linker_input.user_link_flags)
            non_sdk_linkopts = _add_linkopts(objc_provider_kwargs, linkopts)
            all_non_sdk_linkopts.extend(non_sdk_linkopts)

        libraries_to_link = []
        for linker_input in cc_linking_context.linker_inputs.to_list():
            libraries_to_link.extend(linker_input.libraries)
        objc_provider_kwargs["cc_library"].append(
            depset(direct = libraries_to_link, order = "topological"),
        )

    non_sdk_linkopts = _add_linkopts(
        objc_provider_kwargs,
        objc_internal.expand_toolchain_and_ctx_variables(ctx = ctx, flags = attr_linkopts),
    )
    all_non_sdk_linkopts.extend(non_sdk_linkopts)

    if compilation_attributes != None:
        sdk_dir = apple_common.apple_toolchain().sdk_dir()
        usr_include_dir = sdk_dir + "/usr/include/"
        sdk_includes = []

        for sdk_include in compilation_attributes.sdk_includes.to_list():
            sdk_includes.append(usr_include_dir + sdk_include)

        objc_provider_kwargs["sdk_framework"].extend(
            compilation_attributes.sdk_frameworks.to_list(),
        )
        objc_provider_kwargs["weak_sdk_framework"].extend(
            compilation_attributes.weak_sdk_frameworks.to_list(),
        )
        objc_provider_kwargs["sdk_dylib"].extend(compilation_attributes.sdk_dylibs.to_list())
        objc_compilation_context_kwargs["public_hdrs"].extend(compilation_attributes.hdrs.to_list())
        objc_compilation_context_kwargs["public_textual_hdrs"].extend(
            compilation_attributes.textual_hdrs.to_list(),
        )
        objc_compilation_context_kwargs["defines"].extend(compilation_attributes.defines)
        objc_compilation_context_kwargs["includes"].extend(
            compilation_attributes.header_search_paths(
                genfiles_dir = ctx.genfiles_dir.path,
            ).to_list(),
        )
        objc_compilation_context_kwargs["includes"].extend(sdk_includes)

    if compilation_artifacts != None:
        all_sources = _filter_out_by_extension(compilation_artifacts.srcs, OBJECT_FILE_SOURCES) + \
                      compilation_artifacts.non_arc_srcs

        if compilation_artifacts.archive != None:
            objc_provider_kwargs["library"] = [
                depset([compilation_artifacts.archive], order = "topological"),
            ]
        objc_provider_kwargs["source"].extend(all_sources)

        objc_compilation_context_kwargs["public_hdrs"].extend(
            compilation_artifacts.additional_hdrs,
        )
        objc_compilation_context_kwargs["private_hdrs"].extend(
            _filter_by_extension(compilation_artifacts.srcs, HEADERS),
        )

    if alwayslink:
        direct = []
        if compilation_artifacts != None:
            if compilation_artifacts.archive != None:
                direct.append(compilation_artifacts.archive)

        direct.extend(extra_import_libraries)

        objc_provider_kwargs["force_load_library"] = [
            depset(
                direct = direct,
                transitive = objc_provider_kwargs["force_load_library"],
                order = "topological",
            ),
        ]

    if has_module_map:
        module_map = intermediate_artifacts.swift_module_map
        umbrella_header = module_map.umbrella_header()
        if umbrella_header != None:
            objc_provider_kwargs["umbrella_header"].append(umbrella_header)

        objc_provider_kwargs["module_map"].append(module_map.file())

    objc_provider_kwargs_built = {}
    for k, v in objc_provider_kwargs.items():
        if k == "providers":
            objc_provider_kwargs_built[k] = v
        elif k in link_order_keys:
            objc_provider_kwargs_built[k] = depset(transitive = v, order = "topological")
        else:
            objc_provider_kwargs_built[k] = depset(v)

    objc_compilation_context = objc_internal.create_compilation_context(
        **objc_compilation_context_kwargs
    )

    # The non-straightfoward way we initialize the sdk related
    # information in linkopts (sdk_framework, weak_sdk_framework,
    # sdk_dylib):
    #
    # - Filter them out of cc_linking_contexts_for_merging and self's
    #   linkopts.  Add them to corresponding fields in
    #   objc_provider_kwargs.  This also has the side effect that it
    #   deduplicates those fields.
    #
    # - Use the sdk fields in objc_provider_kwargs to construct
    #   cc_linking_context's linkopts.
    all_linkopts = all_non_sdk_linkopts
    for sdk_framework in objc_provider_kwargs["sdk_framework"]:
        all_linkopts.append("-framework")
        all_linkopts.append(sdk_framework)

    for weak_sdk_framework in objc_provider_kwargs["weak_sdk_framework"]:
        all_linkopts.append("-weak_framework")
        all_linkopts.append(weak_sdk_framework)

    for sdk_dylib in objc_provider_kwargs["sdk_dylib"]:
        if sdk_dylib.startswith("lib"):
            sdk_dylib = sdk_dylib[3:]
        all_linkopts.append("-l%s" % sdk_dylib)

    objc_linking_context = struct(
        cc_linking_contexts = cc_linking_contexts,
        linkopts = all_linkopts,
    )

    return (
        apple_common.new_objc_provider(**objc_provider_kwargs_built),
        objc_compilation_context,
        objc_linking_context,
    )

def _filter_by_extension(file_list, extensions):
    return [file for file in file_list if "." + file.extension in extensions]

def _filter_out_by_extension(file_list, extensions):
    return [file for file in file_list if "." + file.extension not in extensions]

def _add_linkopts(objc_provider_kwargs, linkopts):
    non_sdk_linkopts = []
    sdk_frameworks = {}
    weak_sdk_frameworks = {}
    sdk_dylib = {}
    i = 0
    skip_next = False
    for arg in linkopts:
        if skip_next:
            skip_next = False
            i += 1
            continue
        if arg == "-framework" and i < len(linkopts) - 1:
            sdk_frameworks[linkopts[i + 1]] = True
            skip_next = True
        elif arg == "-weak_framework" and i < len(linkopts) - 1:
            weak_sdk_frameworks[linkopts[i + 1]] = True
            skip_next = True
        elif arg.startswith("-Wl,-framework,"):
            sdk_frameworks[arg[len("-Wl,-framework,"):]] = True
        elif arg.startswith("-Wl,-weak_framework,"):
            weak_sdk_frameworks[arg[len("-Wl,-weak_framework,"):]] = True
        elif arg.startswith("-l"):
            sdk_dylib[arg[2:]] = True
        else:
            non_sdk_linkopts.append(arg)
        i += 1

    objc_provider_kwargs["sdk_framework"].extend(sdk_frameworks.keys())
    objc_provider_kwargs["weak_sdk_framework"].extend(weak_sdk_frameworks.keys())
    objc_provider_kwargs["sdk_dylib"].extend(sdk_dylib.keys())
    objc_provider_kwargs["linkopt"].append(
        depset(
            direct = non_sdk_linkopts,
            order = "topological",
        ),
    )

    return non_sdk_linkopts

objc_common = struct(
    create_context_and_provider = _create_context_and_provider,
)
