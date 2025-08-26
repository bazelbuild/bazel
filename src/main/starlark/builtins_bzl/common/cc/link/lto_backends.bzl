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

load(":common/cc/cc_helper_internal.bzl", "should_create_per_object_debug_info")

cc_common_internal = _builtins.internal.cc_common
cc_internal = _builtins.internal.cc_internal

def create_lto_backends(
        actions,
        lto_compilation_context,
        feature_configuration,
        cc_toolchain,
        use_pic,
        object_file_inputs,
        lto_output_root_prefix,
        lto_obj_root_prefix,
        static_libraries_to_link,
        allow_lto_indexing,
        include_link_static_in_lto_indexing,
        prefer_pic_libs):
    cpp_config = cc_toolchain._cpp_configuration
    debug = should_create_per_object_debug_info(feature_configuration, cpp_config)

    compiled = set()
    static_library_files = set()
    for lib in static_libraries_to_link:
        pic = (prefer_pic_libs and lib.pic_static_library != None) or \
              lib.static_library == None
        library_file = lib.pic_static_library if pic else lib.static_library
        if library_file in static_library_files:
            # Duplicated static libraries are linked just once and don't error out.
            # TODO(b/413333884): Clean up violations and error out
            continue
        static_library_files.add(library_file)
        context = lib._pic_lto_compilation_context if pic else lib._lto_compilation_context
        if context:
            compiled.update(context.lto_bitcode_inputs().keys())

    all_bitcode = []
    # Since this link includes object files from another library, we know that library must be
    # statically linked, so we need to look at includeLinkStaticInLtoIndexing to decide whether
    # to include its objects in the LTO indexing for this target.

    if include_link_static_in_lto_indexing:
        for lib in static_libraries_to_link:
            if not lib._contains_objects:
                continue
            pic = (prefer_pic_libs and lib.pic_static_library != None) or \
                  lib.static_library == None
            objects = lib.pic_objects if pic else lib.objects
            for obj in objects:
                if obj in compiled:
                    all_bitcode.append(obj)

    for obj in object_file_inputs:
        if obj in lto_compilation_context.lto_bitcode_inputs():
            all_bitcode.append(obj)

    if lto_output_root_prefix == lto_obj_root_prefix:
        for file in all_bitcode:
            if file.is_directory:
                fail("Thinlto with tree artifacts requires feature use_lto_native_object_directory.")

    # Make this a NestedSet to return from LtoBackendAction.getAllowedDerivedInputs. For M binaries
    # and N .o files, this is O(M*N). If we had nested sets of bitcode files, it would be O(M + N).
    all_bitcode_depset = depset(all_bitcode)
    lto_outputs = []
    for lib in static_libraries_to_link:
        if not lib._contains_objects:
            continue
        pic = (prefer_pic_libs and lib.pic_static_library != None) or \
              lib.static_library == None
        objects = lib.pic_objects if pic else lib.objects
        lib_lto_compilation_context = lib._pic_lto_compilation_context if pic else lib._lto_compilation_context
        shared_lto_backends = lib._pic_shared_non_lto_backends if pic else lib._shared_non_lto_backends

        for obj in objects:
            if obj not in compiled:
                continue
            if include_link_static_in_lto_indexing:
                backend_user_compile_flags = _backend_user_compile_flags(cpp_config, obj, lib_lto_compilation_context)
                lto_outputs.append(cc_common_internal.create_lto_backend_artifacts(
                    actions = actions,
                    lto_output_root_prefix = lto_output_root_prefix,
                    lto_obj_root_prefix = lto_obj_root_prefix,
                    bitcode_file = obj,
                    all_bitcode_files = all_bitcode_depset,
                    feature_configuration = feature_configuration,
                    cc_toolchain = cc_toolchain,
                    fdo_context = cc_toolchain._fdo_context,  #TODO: remove
                    use_pic = use_pic,
                    should_create_per_object_debug_info = debug,
                    argv = backend_user_compile_flags,
                ))
            else:
                if not shared_lto_backends:
                    fail(("Statically linked test target requires non-LTO backends for its library inputs," +
                          " but library input %s does not specify shared_non_lto_backends") % lib)
                lto_outputs.append(shared_lto_backends[obj])

    for obj in object_file_inputs:
        if obj not in lto_compilation_context.lto_bitcode_inputs():
            continue
        backend_user_compile_flags = _backend_user_compile_flags(cpp_config, obj, lto_compilation_context)
        lto_outputs.append(cc_common_internal.create_lto_backend_artifacts(
            actions = actions,
            lto_output_root_prefix = lto_output_root_prefix,
            lto_obj_root_prefix = lto_obj_root_prefix,
            bitcode_file = obj,
            all_bitcode_files = all_bitcode_depset if allow_lto_indexing else None,
            feature_configuration = feature_configuration,
            cc_toolchain = cc_toolchain,
            fdo_context = cc_toolchain._fdo_context,  #TODO: remove
            use_pic = use_pic,
            should_create_per_object_debug_info = debug,
            create_shared_non_lto = not allow_lto_indexing,
            argv = backend_user_compile_flags,
        ))
    return lto_outputs

def _backend_user_compile_flags(cpp_config, obj, context):
    argv = []
    lto_bitcode_files = context.lto_bitcode_inputs()
    if obj in lto_bitcode_files:
        argv.extend(lto_bitcode_files[obj].copts)
    argv.extend(cpp_config.lto_backend_options)
    argv.extend(cc_internal.collect_per_file_lto_backend_opts(cpp_config, obj))
    return argv

def create_shared_non_lto_artifacts(
        actions,
        lto_compilation_context,
        is_linker,
        feature_configuration,
        cc_toolchain,
        use_pic,
        object_file_inputs):
    # Only create the shared LTO artifacts for a statically linked library that has bitcode files.
    if not lto_compilation_context or is_linker:
        return {}

    lto_output_root_prefix = "shared.nonlto"
    if feature_configuration.is_enabled("use_lto_native_object_directory"):
        lto_obj_root_prefix = "shared.nonlto-obj"
    else:
        lto_obj_root_prefix = "shared.nonlto"
    cpp_config = cc_toolchain._cpp_configuration
    debug = should_create_per_object_debug_info(feature_configuration, cpp_config)

    shared_non_lto_backends = {}
    for obj in object_file_inputs:
        if obj not in lto_compilation_context.lto_bitcode_inputs():
            continue

        backend_user_compile_flags = _backend_user_compile_flags(cpp_config, obj, lto_compilation_context)
        shared_non_lto_backends[obj] = cc_common_internal.create_lto_backend_artifacts(
            actions = actions,
            lto_output_root_prefix = lto_output_root_prefix,
            lto_obj_root_prefix = lto_obj_root_prefix,
            bitcode_file = obj,
            all_bitcode_files = None,
            feature_configuration = feature_configuration,
            cc_toolchain = cc_toolchain,
            fdo_context = cc_toolchain._fdo_context,  #TODO: remove
            use_pic = use_pic,
            should_create_per_object_debug_info = debug,
            create_shared_non_lto = True,
            argv = backend_user_compile_flags,
        )
    return shared_non_lto_backends
