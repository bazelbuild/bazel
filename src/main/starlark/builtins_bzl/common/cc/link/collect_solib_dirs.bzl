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
"""Goes over LegacyLinkerInputs and produces LibraryToLinkValue-s."""

load(":common/cc/cc_helper_internal.bzl", "is_shared_library")
load(":common/cc/link/target_types.bzl", "LINKING_MODE", "LINK_TARGET_TYPE", "is_dynamic_library")
load(":common/paths.bzl", "paths")

# TODO(b/338618120): Refine the signature of collect_solib_dirs. Large objects are passed in
# just to determine a single property, for example link_type and linking_mode are passed in, just to
# determine need_toolchain_libraries_rpath. Refining the signature will increase readability.
def collect_solib_dirs(
        libraries,
        cc_toolchain,
        feature_configuration,
        prefer_static_libs,
        output,
        dynamic_library_solib_symlink_output,
        link_type,
        linking_mode,
        is_native_deps,
        solib_dir,
        toolchain_libraries_solib_dir,
        workspace_name):
    """Goes over LegacyLinkerInputs and produces LibraryToLinkValue-s and rpaths.

    LibraryToLinkValues are consumed by link_build_variables.bzl.

    When linking a shared library fully or mostly static then we need to link in *all* dependent
    files, not just what the shared library needs for its own code. This is done by wrapping all
    objects/libraries with -Wl,-whole-archive and -Wl,-no-whole-archive. For this case the
    globalNeedWholeArchive parameter must be set to true. Otherwise only library objects (.lo) need
    to be wrapped with -Wl,-whole-archive and -Wl,-no-whole-archive.

    Args:
      libraries: (list[LibraryToLink]) Libraries to link in.
      cc_toolchain: cc_toolchain providing some extra information in the conversion.
      feature_configuration: Feature configuration to be queried.
      prefer_static_libs: (bool) Prefer static libraries.
          Used to select dynamic libraries from the whole set.
      output: (File) The linker's output.
      dynamic_library_solib_symlink_output: (None|File) Symlink to the dynamic library being created.
      link_type: (LINK_TARGET_TYPE) The type of ELF file to be created (.a, .so, .lo, executable).
      linking_mode: ("static"|"dynamic") Linking mode.
      is_native_deps: (bool) Is this link action is used for a native dependency library.
      need_whole_archive: (bool) Whether we need to use whole-archive for the link.
      solib_dir: (str) solib directory.
      toolchain_libraries_solib_dir: (str) Directory where toolchain stores language-runtime libraries (libstdc++, libc++ ...).
      allow_lto_indexing: (bool) Is LTO indexing being done.
      lto_mapping: (dict[File, File]) Map from bitcode files to object files. Used to replace all linker inputs.
      workspace_name: (str) Workspace name. To support legacy code.

    Returns:
      ({libraries_to_link: list[LibraryToLinkValue],
        expanded_linker_inputs: list[File],
        library_search_directories: depset[str],
        all_runtime_library_search_directories: depset[str]})

      - Returned libraries_to_link are passed in this form to link_build_variables.
      - expanded_linker_inputs are all the files that will be consumed by the linker (if
        start-end library is used, then it contains object files, otherwise an archive)
      - library_search_directories are absolute directories of all dynamic libraries
      - all_runtime_library_search_directories are directories of all dynamic libraries

      Both depsets of directories are exposed to the link_build_variables.

    """
    need_toolchain_libraries_rpath = (
        toolchain_libraries_solib_dir and
        (is_dynamic_library(link_type) or
         (link_type == LINK_TARGET_TYPE.EXECUTABLE and linking_mode == LINKING_MODE.DYNAMIC))
    )

    # Collect LibrariesToLink
    library_search_directories = []
    rpath_roots_for_explicit_so_deps = {}

    # Calculate the correct relative value for the "-rpath" link option (which sets
    # the search path for finding shared libraries).
    solib_dir_path_string = cc_toolchain._solib_dir
    if is_native_deps and cc_toolchain._cpp_configuration.share_native_deps():
        # For shared native libraries, special symlinking is applied to ensure C++
        # toolchain libraries are available under $ORIGIN/_solib_[arch]. So we set the RPATH to find
        # them.

        # Note that we have to do this because $ORIGIN points to different paths for
        # different targets. In other words, blaze-bin/d1/d2/d3/a_shareddeps.so and
        # blaze-bin/d4/b_shareddeps.so have different path depths. The first could
        # reference a standard blaze-bin/_solib_[arch] via $ORIGIN/../../../_solib[arch],
        # and the second could use $ORIGIN/../_solib_[arch]. But since this is a shared
        # artifact, both are symlinks to the same place, so
        # there's no *one* RPATH setting that fits all targets involved in the sharing.
        potential_solib_parents = []
        rpath_roots = [solib_dir_path_string + "/"]
    else:
        potential_solib_parents = _find_potential_solib_parents(output, dynamic_library_solib_symlink_output, workspace_name)
        rpath_roots = [exec_root + solib_dir_path_string + "/" for exec_root in potential_solib_parents]

    # include_solib_dir: bool, include_toolchain_libraries_solib_dir: bool
    # TODO(b/338618120): instead of returning include_solib_dir, and the paths inside _add_linker_inputs
    include_solib_dir, include_toolchain_libraries_solib_dir = _collect_solib_dirs_from_libraries(
        libraries,
        prefer_static_libs,
        cc_toolchain,
        feature_configuration,
        solib_dir,
        toolchain_libraries_solib_dir,
        rpath_roots,
        # Outputs:
        library_search_directories,
        rpath_roots_for_explicit_so_deps,
    )

    # Remove repetitions
    rpath_roots_for_explicit_so_deps = rpath_roots_for_explicit_so_deps.keys()

    # rpath ordering matters for performance; first add the one where most libraries are found.
    direct_runtime_library_search_directories = []
    if include_solib_dir:
        direct_runtime_library_search_directories.extend(rpath_roots)
    direct_runtime_library_search_directories.extend(rpath_roots_for_explicit_so_deps)

    transitive_runtime_library_search_directories = []
    if include_toolchain_libraries_solib_dir:
        transitive_runtime_library_search_directories = [
            _collect_toolchain_runtime_library_search_directories(
                cc_toolchain,
                output,
                potential_solib_parents,
                need_toolchain_libraries_rpath,
                toolchain_libraries_solib_dir,
                is_native_deps,
                workspace_name,
            ),
        ]
    all_runtime_library_search_directories = depset(
        order = "topological",
        direct = direct_runtime_library_search_directories,
        transitive = transitive_runtime_library_search_directories,
    )

    return depset(library_search_directories), all_runtime_library_search_directories

def _collect_solib_dirs_from_libraries(
        libraries,
        prefer_static_libs,
        cc_toolchain,
        feature_configuration,
        solib_dir,
        toolchain_libraries_solib_dir,
        rpath_roots,
        # Outputs:
        library_search_directories,
        rpath_roots_for_explicit_so_deps):
    """
    Goes over all linker_inputs transforming them and collecting rpath_roots.

    Args:
        libraries: (list[LegacyLinkerInput]) Linker inputs
        prefer_static_libs: (bool) Prefer static libraries.
        cc_toolchain: cc_toolchain providing some extra information in the conversion.
        feature_configuration: Feature configuration to be queried.
        solib_dir: (str) solib directory.
        toolchain_libraries_solib_dir: (str) Directory where toolchain stores language-runtime libraries (libstdc++, libc++ ...).
        rpath_roots: (list[str]) rpath roots (for example solib_dir)
        library_search_directories: (list[str]) Output collecting library search directories.
        rpath_roots_for_explicit_so_deps: (dict[str, None]) Output collecting rpaths.

    Returns:
      (include_solib_dir: bool, include_toolchain_libraries_solib_dir: bool)
    """

    include_solib_dir, include_toolchain_libraries_solib_dir = False, False
    linked_libraries_paths = {}  # :dict[str, str]

    dont_copy_dynamic_libraries_to_binary = not feature_configuration.is_enabled("copy_dynamic_libraries_to_binary")

    # On Windows, dynamic library (dll) cannot be linked directly when using toolchains that
    # support interface library (eg. MSVC). If the user is doing so, it is only to be referenced
    # in other places (such as copy_dynamic_libraries_to_binary); skip adding it.
    windows_shared_libraries = (feature_configuration.is_enabled("targets_windows") and
                                feature_configuration.is_enabled("supports_interface_shared_libraries"))
    solib_dir_split = reversed(solib_dir.split("/"))

    for library in libraries:
        static_lib = (prefer_static_libs and
                      (library.static_library != None or library.pic_static_library != None) or
                      (library.interface_library == None and library.dynamic_library == None))
        if static_lib:
            continue
        if library.interface_library:
            input_file = library.interface_library
            original_file = library.resolved_symlink_interface_library or input_file
        else:
            input_file = library.dynamic_library
            original_file = library.resolved_symlink_dynamic_library or input_file

        original_lib_dir = original_file.dirname
        library_identifier = library._library_identifier
        previous_lib_dir = linked_libraries_paths.setdefault(library_identifier, original_lib_dir)

        if previous_lib_dir != original_lib_dir:
            fail(("You are trying to link the same dynamic library %s built in a different" +
                  " configuration. Previously registered instance had path %s, current one" +
                  " has path %s") %
                 (library_identifier, previous_lib_dir, original_lib_dir))

        lib_dir = input_file.dirname

        # When COPY_DYNAMIC_LIBRARIES_TO_BINARY is enabled, dynamic libraries are not symlinked
        # under solib_dir, so don't check it and don't include solib_dir.
        if dont_copy_dynamic_libraries_to_binary:
            # The first fragment is bazel-out, and the second may contain a configuration mnemonic.
            # We should always add the default solib dir because that's where libraries will be found
            # e.g., in remote execution, so we ignore the first two fragments.
            if not include_solib_dir and lib_dir.split("/")[2:] == solib_dir.split("/")[2:]:
                include_solib_dir = True
            if lib_dir == toolchain_libraries_solib_dir:
                include_toolchain_libraries_solib_dir = True

        if windows_shared_libraries:
            if is_shared_library(input_file):
                continue

        lib_dir = input_file.dirname
        if lib_dir != solib_dir and (not toolchain_libraries_solib_dir or toolchain_libraries_solib_dir != lib_dir):
            # TODO(b/338618120): the code should be optimized to first get unique library_search_directories and
            # then compute relative paths, i.e. rpath_roots_for_explicit_so_deps
            # TODO(b/331164666): this is a duplication of _get_relative function implemented below
            dotdots = ""
            common_parent = solib_dir
            for seg in solib_dir_split:
                if lib_dir.startswith(common_parent + "/"):
                    break
                dotdots += "../"
                common_parent = common_parent[:-len(seg) - 1]

            #  When all dynamic deps are built in transitioned configurations, the default solib dir is
            #  not created. While resolving paths, the dynamic linker stops at the first directory that
            #  does not exist, even when followed by "../". We thus have to normalize the relative path.
            for rpath_root in rpath_roots:
                normalized_path_to_root = paths.normalize(rpath_root + dotdots + paths.relativize(lib_dir, common_parent))
                rpath_roots_for_explicit_so_deps[normalized_path_to_root] = None

            # Unless running locally, libraries will be available under the root relative path, so we
            # should add that to the rpath as well.
            if input_file.short_path.startswith("_solib_"):
                artifact_path_under_solib = "/".join(input_file.short_path.split("/")[1:-1])
                for rpath_root in rpath_roots:
                    rpath_roots_for_explicit_so_deps[rpath_root + artifact_path_under_solib] = None

        library_search_directories.append(lib_dir)

    return include_solib_dir, include_toolchain_libraries_solib_dir

def _collect_toolchain_runtime_library_search_directories(
        cc_toolchain,
        output,
        potential_solib_parents,
        need_toolchain_libraries_rpath,
        toolchain_libraries_solib_dir,
        is_native_deps,
        workspace_name):
    if not need_toolchain_libraries_rpath:
        return depset()

    runtime_library_search_directories = []
    toolchain_libraries_solib_name = paths.basename(toolchain_libraries_solib_dir)
    if not (is_native_deps and cc_toolchain.cc_configuration.share_native_deps()):
        for potential_exec_root in _find_toolchain_solib_parents(cc_toolchain, output, potential_solib_parents, toolchain_libraries_solib_dir, workspace_name):
            runtime_library_search_directories.append(potential_exec_root + toolchain_libraries_solib_name + "/")

    if is_native_deps:
        runtime_library_search_directories.append("../" + toolchain_libraries_solib_name + "/")
        runtime_library_search_directories.append(".")

    runtime_library_search_directories.append(toolchain_libraries_solib_name + "/")

    return depset(runtime_library_search_directories, order = "topological")

# TODO(b/338618120): converge back together _find_toolchain_solib_parents and _find_potential_solib_parents

def _find_potential_solib_parents(output, dynamic_library_solib_symlink_output, workspace_name):
    solib_parents = []
    outputs = [output]
    if dynamic_library_solib_symlink_output:
        outputs.append(dynamic_library_solib_symlink_output)

    for output in outputs:
        # The runtime location of the solib directory relative to the binary depends on four factors:
        #
        # * whether the binary is contained in the main repository or an external repository;
        # * whether the binary is executed directly or from a runfiles tree;
        # * whether the binary is staged as a symlink (sandboxed execution; local execution if the
        #   binary is in the runfiles of another target) or a regular file (remote execution) - the
        #   dynamic linker follows sandbox and runfiles symlinks into its location under the
        #   unsandboxed execroot, which thus becomes the effective $ORIGIN;
        # * whether --experimental_sibling_repository_layout is enabled or not.
        #
        # The rpaths emitted into the binary thus have to cover the following cases (assuming that
        # the binary target is located in the pkg `pkg` and has name `file`) for the directory used
        # as $ORIGIN by the dynamic linker and the directory containing the solib directories:
        #
        # 1. main, direct, symlink:
        #    $ORIGIN:    $EXECROOT/pkg
        #    solib root: $EXECROOT
        # 2. main, direct, regular file:
        #    $ORIGIN:    $EXECROOT/pkg
        #    solib root: $EXECROOT/pkg/file.runfiles/main_repo
        # 3. main, runfiles, symlink:
        #    $ORIGIN:    $EXECROOT/pkg
        #    solib root: $EXECROOT
        # 4. main, runfiles, regular file:
        #    $ORIGIN:    other_target.runfiles/main_repo/pkg
        #    solib root: other_target.runfiles/main_repo
        # 5a. external, direct, symlink:
        #    $ORIGIN:    $EXECROOT/external/other_repo/pkg
        #    solib root: $EXECROOT
        # 5b. external, direct, symlink, with --experimental_sibling_repository_layout:
        #    $ORIGIN:    $EXECROOT/../other_repo/pkg
        #    solib root: $EXECROOT/../other_repo
        # 6a. external, direct, regular file:
        #    $ORIGIN:    $EXECROOT/external/other_repo/pkg
        #    solib root: $EXECROOT/external/other_repo/pkg/file.runfiles/main_repo
        # 6b. external, direct, regular file, with --experimental_sibling_repository_layout:
        #    $ORIGIN:    $EXECROOT/../other_repo/pkg
        #    solib root: $EXECROOT/../other_repo/pkg/file.runfiles/other_repo
        # 7a. external, runfiles, symlink:
        #    $ORIGIN:    $EXECROOT/external/other_repo/pkg
        #    solib root: $EXECROOT
        # 7b. external, runfiles, symlink, with --experimental_sibling_repository_layout:
        #    $ORIGIN:    $EXECROOT/../other_repo/pkg
        #    solib root: $EXECROOT/../other_repo
        # 8a. external, runfiles, regular file:
        #    $ORIGIN:    other_target.runfiles/some_repo/pkg
        #    solib root: other_target.runfiles/main_repo
        # 8b. external, runfiles, regular file, with --experimental_sibling_repository_layout:
        #    $ORIGIN:    other_target.runfiles/some_repo/pkg
        #    solib root: other_target.runfiles/some_repo
        #
        # Cases 1, 3, 4, 5, 7, and 8b are covered by an rpath that walks up the root relative path.
        # Cases 2 and 6 covered by walking into file.runfiles/main_repo.
        # Case 8a is covered by walking up some_repo/pkg and then into main_repo.
        path = output.short_path
        is_external = path.startswith("../")
        uses_legacy_repository_layout = is_external and output.path.split("/")[3] == "external"

        # Handles cases 1, 3, 4, 5, and 7.
        solib_parents.append("../" * (len(path.split("/")) - 1 - (2 if is_external and not uses_legacy_repository_layout else 0)))

        # Handle cases 2 and 6.
        if is_external and not uses_legacy_repository_layout:
            # Case 6b
            solib_repository_name = path.split("/")[1]
        else:
            # Cases 2 and 6a
            solib_repository_name = workspace_name

        solib_parents.append(output.basename + ".runfiles/" + solib_repository_name + "/")
        if is_external and uses_legacy_repository_layout:
            # Handles case 8a. The runfiles path is of the form ../some_repo/pkg/file and we need to
            # walk up some_repo/pkg and then down into main_repo.
            solib_parents.append(
                "../" * (len(output.root.path.split("/")) - 1) + workspace_name + "/",
            )

    return solib_parents

def _find_toolchain_solib_parents(cc_toolchain, output, potential_solib_parents, toolchain_libraries_solib_dir, workspace_name):
    uses_legacy_repository_layout = not cc_toolchain._is_sibling_repository_layout

    # When -experimental_sibling_repository_layout is not enabled, the toolchain solib sits next to
    # the solib_<cpu> directory - so that it shares the same parents.
    if uses_legacy_repository_layout:
        return potential_solib_parents

    # When -experimental_sibling_repository_layout is enabled, the toolchain solib is located in
    # these 2 places:
    # 1. The `bin` directory of the repository where the toolchain target is declared (this is the
    # parent directory of `toolchainLibrariesSolibDir`).
    # 2. In `target.runfiles/<toolchain repo>`
    #
    # And the following factors affect what $ORIGIN is resolved to:
    # * whether the binary is contained in the main repository or an external repository;
    # * whether the binary is executed directly or from a runfiles tree;
    # * whether the binary is staged as a symlink (sandboxed execution; local execution if the
    #   binary is in the runfiles of another target) or a regular file (remote execution) - the
    #   dynamic linker follows sandbox and runfiles symlinks into its location under the
    #   unsandboxed execroot, which thus becomes the effective $ORIGIN;
    #
    # The rpaths emitted into the binary thus have to cover the following cases (assuming that
    # the binary target is located in the pkg `pkg` and has name `file`) for the directory used
    # as $ORIGIN by the dynamic linker and the directory containing the solib directories:
    # 1. main, direct, symlink:
    #    $ORIGIN:    $EXECROOT/pkg
    #    solib root: <toolchain repo bin>
    # 2. main, direct, regular file:
    #    $ORIGIN:    $EXECROOT/pkg
    #    solib root: $EXECROOT/pkg/file.runfiles/<toolchain repo>
    # 3. main, runfiles, symlink:
    #    $ORIGIN:    $EXECROOT/pkg
    #    solib root: <toolchain repo bin>
    # 4. main, runfiles, regular file:
    #    $ORIGIN:    other_target.runfiles/main_repo/pkg
    #    solib root: other_target.runfiles/<toolchain repo>
    # 5. external, direct, symlink:
    #    $ORIGIN:    $EXECROOT/../other_repo/pkg
    #    solib root: <toolchain repo bin>
    # 6. external, direct, regular file:
    #    $ORIGIN:    $EXECROOT/../other_repo/pkg
    #    solib root: $EXECROOT/../other_repo/pkg/file.runfiles/<toolchain repo>
    # 7. external, runfiles, symlink:
    #    $ORIGIN:    $EXECROOT/../other_repo/pkg
    #    solib root: <toolchain repo bin>
    # 8. external, runfiles, regular file:
    #    $ORIGIN:    other_target.runfiles/some_repo/pkg
    #    solib root: other_target.runfiles/<toolchain repo>
    #
    # For cases 1, 3, 5, 7, we need to compute the relative path from the output artifact to
    # toolchain repo's bin directory. For 2 and 6, we walk down into `file.runfiles/<toolchain
    # repo>`. For 4 and 8, we need to compute the relative path from the output runfile to
    # <toolchain repo> under runfiles.
    solib_parents = []

    # Cases 1, 3, 5, 7
    toolchain_bin_exec_path = paths.dirname(toolchain_libraries_solib_dir)
    binary_origin_exec_path = output.dirname
    solib_parents.append(
        _get_relative(binary_origin_exec_path, toolchain_bin_exec_path) + "/",
    )

    # Cases 2 and 6
    toolchain_runfiles_repo_name = _get_runfiles_repo_name(cc_toolchain._toolchain_label.repo_name, workspace_name)
    solib_parents.append(
        output.basename + ".runfiles/" + toolchain_runfiles_repo_name + "/",
    )

    # Cases 4 and 8
    binary_repo_name = _get_runfiles_repo_name(output.owner.repo_name, workspace_name)
    toolchain_bin_runfiles_path = toolchain_runfiles_repo_name
    binary_origin_runfiles_path = binary_repo_name + "/" + output.dirname[len(output.root.path):].removeprefix("/")
    solib_parents.append(
        _get_relative(binary_origin_runfiles_path, toolchain_bin_runfiles_path) + "/",
    )

    return solib_parents

def _get_runfiles_repo_name(repo_name, workspace_name):
    # TODO(b/331164666): inline
    return repo_name or workspace_name

def _get_relative(start, to):
    """
    Returns the relative path from "from" to "to".

    Example 1:
    `_get_relative("foo", "foo/bar/wiz") -> returns "bar/wiz"`

    Example 2:
    `_get_relative("foo/bar/wiz", "foo/wiz") -> returns "../../wiz"`

    The following requirements / assumptions are made: 1) paths must be both relative; 2) they
    are assumed to be relative to the same location; 3) when the `from` path starts with
    `..` prefixes, the prefix length must not exceed `..` prefixes of the `to` path.
    """
    common_parent = start
    dotdots = ""
    for seg in reversed(common_parent.split("/")):
        if paths.starts_with(to, common_parent):
            break
        dotdots += "../"
        common_parent = common_parent[:-len(seg) - 1]

    return dotdots + paths.relativize(to, common_parent)
