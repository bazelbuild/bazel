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

cc_internal = _builtins.internal.cc_internal

def add_object_files_to_link(object_files, libraries_to_link_values):
    """Adds object files to libraries_to_link_values.

    Object files are repacked into two flavours of LibraryToLinkValues:
        - for_object_file
        - for_object_file_group (handling tree artifacts)

    Args:
        object_files: (list[File]) Object files (.o, .pic.o)
        libraries_to_link_values: (list[LibraryToLinkValue]) Output collecting libraries to link.
    """
    for object_file in object_files:
        if object_file.is_directory:
            libraries_to_link_values.append(
                cc_internal.for_object_file_group([object_file], False),
            )
        else:
            libraries_to_link_values.append(
                cc_internal.for_object_file(object_file.path, False),
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
