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
load(":common/objc/apple_toolchain.bzl", "apple_toolchain")
load(":common/objc/objc_info.bzl", "ObjcInfo")
load(":common/objc/providers.bzl", "J2ObjcEntryClassInfo", "J2ObjcMappingFileInfo")

objc_internal = _builtins.internal.objc_internal
cc_internal = _builtins.internal.cc_internal

CPP_SOURCES = [".cc", ".cpp", ".mm", ".cxx", ".C"]
NON_CPP_SOURCES = [".m", ".c"]
ASSEMBLY_SOURCES = [".s", ".S", ".asm"]
OBJECT_FILE_SOURCES = [".o"]
HEADERS = [".h", ".inc", ".hpp", ".hh"]

COMPILABLE_SRCS = CPP_SOURCES + NON_CPP_SOURCES + ASSEMBLY_SOURCES
SRCS = COMPILABLE_SRCS + OBJECT_FILE_SOURCES + HEADERS
NON_ARC_SRCS = [".m", ".mm"]

ios_cpus = struct(
    IOS_SIMULATOR_TARGET_CPUS = ["ios_x86_64", "ios_i386", "ios_sim_arm64"],
    IOS_DEVICE_TARGET_CPUS = ["ios_armv6", "ios_arm64", "ios_armv7", "ios_armv7s", "ios_arm64e"],
    VISIONOS_SIMULATOR_TARGET_CPUS = ["visionos_sim_arm64"],
    VISIONOS_DEVICE_TARGET_CPUS = ["visionos_arm64"],
    WATCHOS_SIMULATOR_TARGET_CPUS = ["watchos_i386", "watchos_x86_64", "watchos_arm64"],
    WATCHOS_DEVICE_TARGET_CPUS = ["watchos_armv7k", "watchos_arm64_32", "watchos_device_arm64", "watchos_device_arm64e"],
    TVOS_SIMULATOR_TARGET_CPUS = ["tvos_x86_64", "tvos_sim_arm64"],
    TVOS_DEVICE_TARGET_CPUS = ["tvos_arm64"],
    CATALYST_TARGET_CPUS = ["catalyst_x86_64"],
    MACOS_TARGET_CPUS = ["darwin_x86_64", "darwin_arm64", "darwin_arm64e"],
)

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
        has_module_map,
        deps,
        implementation_deps,
        attr_linkopts,
        direct_cc_compilation_contexts = [],
        includes = [],
        is_aspect = False):
    objc_providers = []
    cc_compilation_contexts = []
    cc_linking_contexts = []

    for dep in deps:
        if ObjcInfo in dep:
            objc_providers.append(dep[ObjcInfo])

        if CcInfo in dep:
            cc_compilation_contexts.append(dep[CcInfo].compilation_context)
            cc_linking_contexts.append(dep[CcInfo].linking_context)

    implementation_cc_compilation_contexts = []
    for impl_dep in implementation_deps:
        implementation_cc_compilation_contexts.append(impl_dep[CcInfo].compilation_context)
        cc_linking_contexts.append(impl_dep[CcInfo].linking_context)

    sdk_linking_info = {
        "sdk_dylib": [],
        "sdk_framework": [],
        "weak_sdk_framework": [],
    }

    objc_provider_kwargs = {
        "providers": objc_providers,
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
        "includes": list(includes),
        "direct_cc_compilation_contexts": direct_cc_compilation_contexts,
    }

    all_non_sdk_linkopts = []
    non_sdk_linkopts = _add_linkopts(
        sdk_linking_info,
        objc_internal.expand_and_tokenize(ctx = ctx, attr = "linkopts", flags = attr_linkopts),
    )
    all_non_sdk_linkopts.extend(non_sdk_linkopts)

    if compilation_attributes != None:
        sdk_dir = apple_toolchain.sdk_dir()
        usr_include_dir = sdk_dir + "/usr/include/"
        sdk_includes = []

        for sdk_include in compilation_attributes.sdk_includes.to_list():
            sdk_includes.append(usr_include_dir + sdk_include)

        sdk_linking_info["sdk_framework"].extend(
            compilation_attributes.sdk_frameworks.to_list(),
        )
        sdk_linking_info["weak_sdk_framework"].extend(
            compilation_attributes.weak_sdk_frameworks.to_list(),
        )
        sdk_linking_info["sdk_dylib"].extend(compilation_attributes.sdk_dylibs.to_list())

        objc_compilation_context_kwargs["public_hdrs"].extend(compilation_attributes.hdrs.to_list())
        objc_compilation_context_kwargs["public_textual_hdrs"].extend(
            compilation_attributes.textual_hdrs.to_list(),
        )
        objc_compilation_context_kwargs["defines"].extend(compilation_attributes.defines)
        objc_compilation_context_kwargs["includes"].extend(sdk_includes)

    if compilation_artifacts != None:
        all_sources = _filter_out_by_extension(compilation_artifacts.srcs, OBJECT_FILE_SOURCES) + \
                      compilation_artifacts.non_arc_srcs

        if compilation_artifacts.archive != None:
            if is_aspect:
                if ctx.rule.kind in ["j2objc_library", "java_library", "java_import", "java_proto_library"]:
                    objc_provider_kwargs["j2objc_library"] = [compilation_artifacts.archive]

        objc_provider_kwargs["source"].extend(all_sources)

        objc_compilation_context_kwargs["public_hdrs"].extend(
            compilation_artifacts.additional_hdrs,
        )
        objc_compilation_context_kwargs["private_hdrs"].extend(
            _filter_by_extension(compilation_artifacts.srcs, HEADERS),
        )

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
        else:
            objc_provider_kwargs_built[k] = depset(v)

    objc_compilation_context = objc_internal.create_compilation_context(
        **objc_compilation_context_kwargs
    )

    all_linkopts = all_non_sdk_linkopts
    for sdk_framework in depset(sdk_linking_info["sdk_framework"]).to_list():
        all_linkopts.append("-framework")
        all_linkopts.append(sdk_framework)

    for weak_sdk_framework in depset(sdk_linking_info["weak_sdk_framework"]).to_list():
        all_linkopts.append("-weak_framework")
        all_linkopts.append(weak_sdk_framework)

    for sdk_dylib in depset(sdk_linking_info["sdk_dylib"]).to_list():
        if sdk_dylib.startswith("lib"):
            sdk_dylib = sdk_dylib[3:]
        all_linkopts.append("-l%s" % sdk_dylib)

    objc_linking_context = struct(
        cc_linking_contexts = cc_linking_contexts,
        linkopts = all_linkopts,
    )

    return (
        ObjcInfo(**objc_provider_kwargs_built),
        objc_compilation_context,
        objc_linking_context,
    )

def _filter_by_extension(file_list, extensions):
    return [file for file in file_list if "." + file.extension in extensions]

def _filter_out_by_extension(file_list, extensions):
    return [file for file in file_list if "." + file.extension not in extensions]

def _add_linkopts(sdk_linking_info, linkopts):
    non_sdk_linkopts = []
    i = 0
    skip_next = False
    for arg in linkopts:
        if skip_next:
            skip_next = False
            i += 1
            continue
        if arg == "-framework" and i < len(linkopts) - 1:
            sdk_linking_info["sdk_framework"].append(linkopts[i + 1])
            skip_next = True
        elif arg == "-weak_framework" and i < len(linkopts) - 1:
            sdk_linking_info["weak_sdk_framework"].append(linkopts[i + 1])
            skip_next = True
        elif arg.startswith("-Wl,-framework,"):
            sdk_linking_info["sdk_framework"].append(arg[len("-Wl,-framework,"):])
        elif arg.startswith("-Wl,-weak_framework,"):
            sdk_linking_info["weak_sdk_framework"].append(arg[len("-Wl,-weak_framework,"):])
        elif arg.startswith("-l"):
            sdk_linking_info["sdk_dylib"].append(arg[2:])
        else:
            non_sdk_linkopts.append(arg)
        i += 1

    return non_sdk_linkopts

def _is_apple_platform(cpu):
    return cpu in ios_cpus.IOS_SIMULATOR_TARGET_CPUS or \
           cpu in ios_cpus.IOS_DEVICE_TARGET_CPUS or \
           cpu in ios_cpus.VISIONOS_SIMULATOR_TARGET_CPUS or \
           cpu in ios_cpus.VISIONOS_DEVICE_TARGET_CPUS or \
           cpu in ios_cpus.WATCHOS_SIMULATOR_TARGET_CPUS or \
           cpu in ios_cpus.WATCHOS_DEVICE_TARGET_CPUS or \
           cpu in ios_cpus.TVOS_SIMULATOR_TARGET_CPUS or \
           cpu in ios_cpus.TVOS_DEVICE_TARGET_CPUS or \
           cpu in ios_cpus.CATALYST_TARGET_CPUS or \
           cpu in ios_cpus.MACOS_TARGET_CPUS

# TODO(bazel-team): Delete this function when MultiArchBinarySupport is starlarkified.
def _j2objc_mapping_file_info_union(providers):
    transitive_header_mapping_files = []
    transitive_class_mapping_files = []
    transitive_dependency_mapping_files = []
    transitive_archive_source_mapping_files = []

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

# TODO(bazel-team): Delete this function when MultiArchBinarySupport is starlarkified.
def _j2objc_entry_class_info_union(providers):
    transitive_entry_classes = []

    for provider in providers:
        transitive_entry_classes.append(provider.entry_classes)

    return J2ObjcEntryClassInfo(
        entry_classes = depset([], transitive = transitive_entry_classes),
    )

objc_common = struct(
    create_context_and_provider = _create_context_and_provider,
    is_apple_platform = _is_apple_platform,
    j2objc_mapping_file_info_union = _j2objc_mapping_file_info_union,
    j2objc_entry_class_info_union = _j2objc_entry_class_info_union,
)
