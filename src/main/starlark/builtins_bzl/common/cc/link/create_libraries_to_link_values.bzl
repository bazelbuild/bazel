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
"""Goes over LibraryToLinks and produces LibraryToLinkValue-s."""

load(":common/cc/cc_helper_internal.bzl", "is_shared_library", "is_versioned_shared_library")

cc_internal = _builtins.internal.cc_internal

# Types of LibraryToLinkValues
_TYPE = struct(
    STATIC_LIBRARY = "static_library",
    DYNAMIC_LIBRARY = "dynamic_library",
    INTERFACE_LIBRARY = "interface_library",
    OBJECT_FILE = "object_file",
    OBJECT_FILE_GROUP = "object_file_group",
    VERSIONED_DYNAMIC_LIBRARY = "versioned_dynamic_library",
)

# Structures exposed to cc_toolchain configuration, representing LibraryToLinkValues.
_NamedLibraryInfo = provider(
    """
    NamedLibraryInfo represents following types of libraries: "static_library", "dynamic_library",
    "interface_library", and "object_file".
    """,
    fields = ["type", "name", "is_whole_archive"],
)
_ObjectFileGroupInfo = provider(
    """ObjectFileGroupInfo. The type is always "object_file_group".""",
    fields = ["type", "object_files", "is_whole_archive"],
)
_VersionedLibraryInfo = provider(
    """VersionedLibraryInfo. The type is always "versioned_dynamic_library".""",
    fields = ["type", "name", "path", "is_whole_archive"],
)

def add_object_files_to_link(object_files, libraries_to_link_values):
    """Adds object files to libraries_to_link_values.

    Object files are repacked into two types of LibraryToLinkValues:
        - "object_file"
        - "object_file_group" (handling tree artifacts)

    Args:
        object_files: (list[File]) Object files (.o, .pic.o)
        libraries_to_link_values: (list[LibraryToLinkValue]) Output collecting libraries to link.
    """
    for object_file in object_files:
        if object_file.is_directory:
            libraries_to_link_values.append(
                _ObjectFileGroupInfo(
                    type = _TYPE.OBJECT_FILE_GROUP,
                    object_files = [object_file],
                    is_whole_archive = False,
                ),
            )
        else:
            libraries_to_link_values.append(
                _NamedLibraryInfo(
                    type = _TYPE.OBJECT_FILE,
                    name = object_file.path,
                    is_whole_archive = False,
                ),
            )

def add_libraries_to_link(
        libraries,
        prefer_static_libs,
        # For static libs
        prefer_pic_libs,
        use_start_end_lib,
        need_whole_archive,
        # For LTO in static libs:
        lto_map,
        allow_lto_indexing,
        shared_non_lto_obj_root_prefix,
        # For dynamic libs
        feature_configuration,
        # Outputs:
        expanded_linker_inputs,
        libraries_to_link_values):
    """Converts libraries from LibraryToLink to LibraryToLinkValue.

    Static library use `prefer_pic_libs` for selection.

    When start-end library is used, static libraries are unpacked into following
    types of LibraryToLinkValues:
    - "object_file"
    - "object_file_group"

    When start-end library isn't used, static libraries are converted to "static_library"
    LibraryToLinkValue.

    Either whole library or library's object files are expanded and added to
    expanded_linker_inputs.

    When library is expanded, object files are processed for LTO.
    See also `process_objects_for_lto`

    For dynamic libraries one of three types of LibraryToLinkValue are appended
    to libraries_to_link_values:
    - "dynamic_library"
    - "versioned_dynamic_library"
    - "interface_library"

    Args:
        libraries: (list[LibraryToLink]) Libraries to Link (all of them).
        prefer_static_libs: (bool) Prefer static libraries.
            Used to select dynamic libraries from the whole set.
        prefer_pic_libs: (bool) Use pic / no-pic library.
        use_start_end_lib: (bool) Whether to use start end lib.
        need_whole_archive: (bool) Whether we need to use whole-archive for the link.
        lto_map: (dict[File, File]) Map from bitcode files to object files.
            Used to replace all linker inputs.
        allow_lto_indexing: (bool) Is LTO indexing being done.
        shared_non_lto_obj_root_prefix: (str) the root prefix of where the shared non lto obj are
            stored
        feature_configuration: Feature configuration to be queried.
        expanded_linker_inputs: (list[File]) Output collecting expanded linker inputs.
        libraries_to_link_values: (list[LibraryToLinkValue]) Output collecting libraries to link.

    Returns:
        None
    """

    # For static libraries:
    static_library_files = set()

    # For dynamic libraries
    windows_with_interface_shared_libraries = (
        feature_configuration.is_enabled("targets_windows") and
        feature_configuration.is_enabled("supports_interface_shared_libraries")
    )

    for library in libraries:
        static_lib = (prefer_static_libs and
                      (library.static_library != None or library.pic_static_library != None) or
                      (library.interface_library == None and library.dynamic_library == None))
        if static_lib:
            pic = (prefer_pic_libs and library.pic_static_library != None) or \
                  library.static_library == None
            library_file = library.pic_static_library if pic else library.static_library
            if library_file in static_library_files:
                # Duplicated static libraries are linked just once and don't error out.
                # TODO(b/413333884): Clean up cc_library.src -> cc_library and error out
                continue
            static_library_files.add(library_file)
            _add_static_library_to_link(
                library,
                prefer_pic_libs,
                use_start_end_lib,
                need_whole_archive,
                # For LTO in static libs:
                lto_map,
                allow_lto_indexing,
                shared_non_lto_obj_root_prefix,
                # Outputs:
                expanded_linker_inputs,
                libraries_to_link_values,
            )
        else:
            _add_dynamic_library_to_link(
                library,
                windows_with_interface_shared_libraries,
                # Outputs:
                expanded_linker_inputs,
                libraries_to_link_values,
            )

def _add_static_library_to_link(
        library,
        prefer_pic_libs,
        use_start_end_lib,
        need_whole_archive,
        # For LTO in static libs:
        lto_map,
        allow_lto_indexing,
        shared_non_lto_obj_root_prefix,
        # Outputs:
        expanded_linker_inputs,
        libraries_to_link_values):
    # input.disable_whole_archive should only be true for libstdc++/libc++ etc.
    input_is_whole_archive = not library._disable_whole_archive and (
        library.alwayslink or need_whole_archive
    )

    pic = (prefer_pic_libs and library.pic_static_library != None) or \
          library.static_library == None
    library_file = library.pic_static_library if pic else library.static_library
    objects = library.pic_objects if pic else library.objects

    # start-lib/end-lib library: adds its input object files.
    # TODO(bazel-team): Figure out if PicArchives are actually used. For it to be used, both
    # linkingStatically and linkShared must me true, we must be in opt mode and cpu has to be k8
    if use_start_end_lib and library._contains_objects:
        # If we had any LTO artifacts, lto_map whould be non-null. In that case,
        # we should have created a thinlto_param_file which the LTO indexing
        # step will populate with the exec paths that correspond to the LTO
        # artifacts that the linker decided to include based on symbol resolution.
        # Those files will be included directly in the link (and not wrapped
        # in --start-lib/--end-lib) to ensure consistency between the two link
        # steps.
        objects = process_objects_for_lto(
            objects,
            lto_map,
            allow_lto_indexing,
            shared_non_lto_obj_root_prefix,
            expanded_linker_inputs,
        )

        if input_is_whole_archive:
            for object in objects:
                if object.is_directory:
                    # TODO(b/78189629): This object filegroup is expanded at action time but
                    # wrapped with --start/--end-lib. There's currently no way to force these
                    # objects to be linked in.
                    libraries_to_link_values.append(
                        _ObjectFileGroupInfo(
                            type = _TYPE.OBJECT_FILE_GROUP,
                            object_files = [object],
                            is_whole_archive = True,
                        ),
                    )
                else:
                    # TODO(b/78189629): These each need to be their own LibraryToLinkValue so
                    # they're not wrapped in --start/--end-lib (which lets the linker leave out
                    # objects with unreferenced code).
                    libraries_to_link_values.append(
                        _NamedLibraryInfo(
                            type = _TYPE.OBJECT_FILE,
                            name = object.path,
                            is_whole_archive = True,
                        ),
                    )
        elif objects:
            libraries_to_link_values.append(
                _ObjectFileGroupInfo(
                    type = _TYPE.OBJECT_FILE_GROUP,
                    object_files = objects,
                    is_whole_archive = False,
                ),
            )
    else:
        libraries_to_link_values.append(
            _NamedLibraryInfo(
                type = _TYPE.STATIC_LIBRARY,
                name = library_file.path,
                is_whole_archive = input_is_whole_archive,
            ),
        )
        expanded_linker_inputs.append(library_file)

def _add_dynamic_library_to_link(
        library,
        windows_with_interface_shared_libraries,
        # Outputs:
        expanded_linker_inputs,
        libraries_to_link_values):
    # Dynamic library
    input_file = library.interface_library or library.dynamic_library

    expanded_linker_inputs.append(input_file)

    shared_library = is_shared_library(input_file)
    if windows_with_interface_shared_libraries and shared_library:
        # On Windows, dynamic library (dll) cannot be linked directly when using toolchains
        # that support interface library (eg. MSVC). If the user is doing so, it is only to be
        # referenced in other places (such as copy_dynamic_libraries_to_binary); skip adding it
        return

    name = input_file.basename

    # Use the normal shared library resolution rules if possible, otherwise treat as a versioned
    # library that must use the exact name. e.g.:
    # -lfoo -> libfoo.so
    # -l:foo -> foo.so
    # -l:libfoo.so.1 -> libfoo.so.1
    has_compatible_name = (
        name.startswith("lib") or
        (not name.endswith(".so") and not name.endswith(".dylib") and not name.endswith(".dll"))
    )
    if shared_library and has_compatible_name:
        lib_name = name.removeprefix("lib").removesuffix(".so").removesuffix(".dylib") \
            .removesuffix(".dll")
        libraries_to_link_values.append(
            _NamedLibraryInfo(
                type = _TYPE.DYNAMIC_LIBRARY,
                name = lib_name,
                is_whole_archive = False,
            ),
        )
    elif shared_library or is_versioned_shared_library(input_file):
        libraries_to_link_values.append(
            _VersionedLibraryInfo(
                type = _TYPE.VERSIONED_DYNAMIC_LIBRARY,
                name = name,
                path = input_file.path,
                is_whole_archive = False,
            ),
        )
    else:
        # Interface shared objects have a non-standard extension
        # that the linker won't be able to find.  So use the
        # filename directly rather than a -l option.  Since the
        #  library has an SONAME attribute, this will work fine.
        libraries_to_link_values.append(
            _NamedLibraryInfo(
                type = _TYPE.INTERFACE_LIBRARY,
                name = input_file.path,
                is_whole_archive = False,
            ),
        )

def process_objects_for_lto(
        object_files,
        lto_map,
        allow_lto_indexing,
        shared_non_lto_obj_root_prefix,
        expanded_linker_artifacts):
    """Processes and returns the subset of object files not handled by LTO.

    If object is produced from a bitcode file that will be input to the LTO indexing step,
    it is removed. In that case that step will add it to the generated thinlto_param_file for
    inclusion in the final link step if the linker decides to include it.

    All object files are mapped using lto_map and added to expanded_linker_artifacts.
    The objects are removed from the `lto_map`, to keep tract that all objects were mapped.

    Args:
      object_files: (list[File]) list of object files
      lto_map: (dict[File, File]) Map from bitcode files to object files.
          Used to replace all linker inputs.
      allow_lto_indexing: (bool) Is LTO indexing being done.
      shared_non_lto_obj_root_prefix: (str) the root prefix of where the shared non lto obj are
          stored
      expanded_linker_artifacts: (list[File]) are all the files that will be consumed by the linker.
    Returns:
      (list[File]) Object files not handled by LTO
    """
    if allow_lto_indexing:
        mapped_object_files = []
        remaining_object_files = []
        for orig_object in object_files:
            object = lto_map.pop(orig_object, orig_object)
            mapped_object_files.append(object)
            if object == orig_object or object.short_path.startswith(shared_non_lto_obj_root_prefix):
                remaining_object_files.append(object)
    else:
        mapped_object_files = [lto_map.pop(obj, obj) for obj in object_files] if lto_map else object_files
        remaining_object_files = mapped_object_files

    expanded_linker_artifacts.extend(mapped_object_files)

    return remaining_object_files
