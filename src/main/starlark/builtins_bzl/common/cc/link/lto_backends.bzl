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
# LINT.IfChange(forked_exports)
"""
ThinLTO expands the traditional 2 step compile (N x compile .cc, 1x link (N .o files) into a 4
step process:


1. Bitcode generation (N times). This is produces intermediate LLVM bitcode from a source
   file. For this product, it reuses the .o extension.
2. Indexing (once on N files). This takes all bitcode .o files, and for each .o file, it
   decides from which other .o files symbols can be inlined. In addition, it generates an
   index for looking up these symbols, and an imports file for identifying new input files for
   each step 3 {@link LtoBackendAction}.
3. Backend compile (N times). This is the traditional compilation, and uses the same
   command line as the Bitcode generation in 1). Since the compiler has many bit code files
   available, it can inline functions and propagate constants across .o files. This step is
   costly, as it will do traditional optimization. The result is a .lto.o file, a traditional
   ELF object file.
4. Backend link (once). This is the traditional link, and produces the final executable.
"""

load(":common/cc/cc_helper_internal.bzl", "should_create_per_object_debug_info")
load(":common/paths.bzl", "paths")

_cc_common_internal = _builtins.internal.cc_common
_cc_internal = _builtins.internal.cc_internal

LtoBackendArtifactsInfo = provider(
    doc = "LtoBackendArtifacts represents a set of artifacts for a single ThinLTO backend compile.",
    fields = {
        "index": "(None|File) A file containing mapping of symbol => bitcode file containing the " +
                 " symbol. It will be None when this is a shared non-lto backend.",
        "imports": "(None|File) A file containing a list of bitcode files necessary to run " +
                   "the backend step. It will be None when this is a shared non-lto backend.",
        "_bitcode_file": "(File) The bitcode file which is the input of the compile.",
        "_object_file": "(File) The result of executing the above command line, an ELF object " +
                        " file.",
        "_dwo_file": "(File) The corresponding dwo file if fission is used.",
    },
)

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
    """Create the LTO backends for a link.

    Args:
      actions: (actions) The actions object.
      lto_compilation_context: (LtoCompilationContext) The LTO compilation context.
      feature_configuration: (feature_configuration) The feature configuration.
      cc_toolchain: (CcToolchainInfo) The C++ toolchain.
      use_pic: (bool) Whether to use PIC.
      object_file_inputs: (depset[File]) The object file inputs.
      lto_output_root_prefix: (str) The root prefix for the LTO output files.
      lto_obj_root_prefix: (str) The root prefix for the LTO object files.
      static_libraries_to_link: (list[LibraryToLink]) The static libraries to link.
      allow_lto_indexing: (bool) Whether LTO indexing is allowed.
      include_link_static_in_lto_indexing: (bool) Whether to include the static libraries in the
        LTO indexing.
      prefer_pic_libs: (bool) Whether to prefer PIC static libraries.
    Returns:
      (list[LtoBackendArtifactsInfo]) The LTO backends.
    """
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
            compiled.update(context.lto_bitcode_inputs.keys())

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
        if obj in lto_compilation_context.lto_bitcode_inputs:
            all_bitcode.append(obj)

    if lto_output_root_prefix == lto_obj_root_prefix:
        for file in all_bitcode:
            if file.is_directory:
                fail("Thinlto with tree artifacts requires feature use_lto_native_object_directory.")

    build_variables, additional_inputs = setup_common_lto_variables(cc_toolchain, feature_configuration)

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
                lto_outputs.append(create_lto_backend_artifacts(
                    actions = actions,
                    lto_output_root_prefix = lto_output_root_prefix,
                    lto_obj_root_prefix = lto_obj_root_prefix,
                    bitcode_file = obj,
                    all_bitcode_files = all_bitcode_depset,
                    feature_configuration = feature_configuration,
                    cc_toolchain = cc_toolchain,
                    use_pic = use_pic,
                    should_create_per_object_debug_info = debug,
                    build_variables = build_variables,
                    additional_inputs = additional_inputs,
                    argv = backend_user_compile_flags,
                ))
            else:
                if not shared_lto_backends:
                    fail(("Statically linked test target requires non-LTO backends for its library inputs," +
                          " but library input %s does not specify shared_non_lto_backends") % lib)
                lto_outputs.append(shared_lto_backends[obj])

    for obj in object_file_inputs:
        if obj not in lto_compilation_context.lto_bitcode_inputs:
            continue
        backend_user_compile_flags = _backend_user_compile_flags(cpp_config, obj, lto_compilation_context)
        if not allow_lto_indexing:
            # Depending on whether LTO indexing is allowed, generate an LTO backend
            # that will be fed the results of the indexing step, or a dummy LTO backend
            # that simply compiles the bitcode into native code without any index-based
            # cross module optimization.
            actions = _cc_internal.wrap_link_actions(actions, None, True)
        lto_outputs.append(create_lto_backend_artifacts(
            actions = actions,
            lto_output_root_prefix = lto_output_root_prefix,
            lto_obj_root_prefix = lto_obj_root_prefix,
            bitcode_file = obj,
            all_bitcode_files = all_bitcode_depset if allow_lto_indexing else None,
            feature_configuration = feature_configuration,
            cc_toolchain = cc_toolchain,
            use_pic = use_pic,
            should_create_per_object_debug_info = debug,
            build_variables = build_variables,
            additional_inputs = additional_inputs,
            argv = backend_user_compile_flags,
        ))
    return lto_outputs

def _backend_user_compile_flags(cpp_config, obj, context):
    argv = []
    lto_bitcode_files = context.lto_bitcode_inputs
    if obj in lto_bitcode_files:
        argv.extend(lto_bitcode_files[obj].copts)
    argv.extend(cpp_config.lto_backend_options)
    argv.extend(_cc_internal.collect_per_file_lto_backend_opts(cpp_config, obj))
    return argv

def create_shared_non_lto_artifacts(
        actions,
        lto_compilation_context,
        is_linker,
        feature_configuration,
        cc_toolchain,
        use_pic,
        object_file_inputs):
    """Create the shared non-LTO artifacts for a statically linked library.

    Args:
      actions: (actions) The actions object.
      lto_compilation_context: (LtoCompilationContext) The LTO compilation context.
      is_linker: (bool) Whether the link is a linker.
      feature_configuration: (feature_configuration) The feature configuration.
      cc_toolchain: (CcToolchainInfo) The C++ toolchain.
      use_pic: (bool) Whether to use PIC.
      object_file_inputs: (depset[File]) The object file inputs.
    Returns:
      (dict[File, LtoBackendArtifactsInfo]) The shared non-LTO artifacts.
    """

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

    build_variables, additional_inputs = setup_common_lto_variables(cc_toolchain, feature_configuration)

    shared_non_lto_backends = {}
    for obj in object_file_inputs:
        if obj not in lto_compilation_context.lto_bitcode_inputs:
            continue

        backend_user_compile_flags = _backend_user_compile_flags(cpp_config, obj, lto_compilation_context)
        shared_non_lto_backends[obj] = create_lto_backend_artifacts(
            actions = _cc_internal.wrap_link_actions(actions, None, True),
            lto_output_root_prefix = lto_output_root_prefix,
            lto_obj_root_prefix = lto_obj_root_prefix,
            bitcode_file = obj,
            all_bitcode_files = None,
            feature_configuration = feature_configuration,
            cc_toolchain = cc_toolchain,
            use_pic = use_pic,
            should_create_per_object_debug_info = debug,
            build_variables = build_variables,
            additional_inputs = additional_inputs,
            argv = backend_user_compile_flags,
        )
    return shared_non_lto_backends

def setup_common_lto_variables(
        cc_toolchain,
        feature_configuration):
    """
    Populates build_variables and additional_inputs with data that is independent of what file is the input to the action.

    Args:
      cc_toolchain: (CcToolchainInfo) The C++ toolchain.
      feature_configuration: (feature_configuration) The feature configuration.

    Returns:
      A CcToolchainVariables provider and a list[File] of additional inputs.
    """

    build_variables = {}
    additional_inputs = []

    _add_profile_for_lto_backend(
        additional_inputs,
        cc_toolchain._fdo_context,
        feature_configuration,
        build_variables,
    )

    # Add the context sensitive instrument path to the backend.
    if feature_configuration.is_enabled("cs_fdo_instrument"):
        build_variables["cs_fdo_instrument_path"] = cc_toolchain._cpp_configuration.cs_fdo_instrument()

    build_variables = _cc_internal.combine_cc_toolchain_variables(
        cc_toolchain._build_variables,
        _cc_internal.cc_toolchain_variables(vars = build_variables),
    )

    return build_variables, additional_inputs

def create_lto_backend_artifacts(
        *,
        actions,
        lto_output_root_prefix,
        lto_obj_root_prefix,
        bitcode_file,
        all_bitcode_files = None,
        feature_configuration,
        cc_toolchain,
        use_pic,
        should_create_per_object_debug_info,
        build_variables,
        additional_inputs,
        argv):
    """Create an LTO backend.

    It uses the appropriate constructor depending on whether the associated
    ThinLTO link will utilize LTO indexing (therefore unique LTO backend actions), or not (and
    therefore the library being linked will create a set of shared LTO backends).

    TODO(b/128341904): Do cross module optimization once there is Starlark support.

    If all_bitcode_files is null, create an LTO backend that does not perform any cross-module
    optimization, by not generating import and index files.

    Args:
      actions: (actions) The actions object.
      lto_output_root_prefix: (str) The root prefix for the LTO output files.
      lto_obj_root_prefix: (str) The root prefix for the LTO object files.
      bitcode_file: (File) The bitcode file to create the LTO backend for.
      all_bitcode_files: (None|depset[File]) The set of all bitcode files to be indexed.
      feature_configuration: (feature_configuration) The feature configuration.
      cc_toolchain: (CcToolchainInfo) The C++ toolchain.
      use_pic: (bool) Whether to use PIC.
      should_create_per_object_debug_info: (bool) Whether to create per-object debug info.
      build_variables: (CcToolchainVariables) Toolchain variables to use for argument expansion.
      additional_inputs: list[File] Additional file inputs required for generated actions.
      argv: (list[str]) The command line arguments to pass to the LTO backend.

    Returns:
      An LtoBackendArtifactsInfo provider.
    """

    if not _cc_common_internal.action_is_enabled(feature_configuration = feature_configuration, action_name = "lto-backend"):
        fail("Thinlto build is requested, but the C++ toolchain doesn't define an action_config for 'lto-backend' action.")

    create_shared_non_lto = all_bitcode_files == None

    build_variables = _cc_internal.combine_cc_toolchain_variables(
        build_variables,
        _cc_internal.cc_toolchain_variables(vars = {
            "user_compile_flags": _cc_internal.intern_string_sequence_variable_value(argv),
        }),
    )

    env = _cc_common_internal.get_environment_variables(
        feature_configuration = feature_configuration,
        action_name = "lto-backend",
        variables = build_variables,
    )

    obj = lto_obj_root_prefix + "/" + bitcode_file.path

    # index_obj is an object that does not exist but helps us find where to store the index and
    # imports files
    index_obj = lto_output_root_prefix + "/" + bitcode_file.path

    additional_inputs = depset(additional_inputs, transitive = [cc_toolchain._compiler_files])

    imports, index, dwo_file = None, None, None
    if bitcode_file.is_directory:
        # declare_shareable_directory is needed to create TreeArtifact in a different configuration (for Android split configurations)
        object_file = actions.declare_shareable_directory(obj)
        if not create_shared_non_lto:
            imports = actions.declare_shareable_directory(index_obj)
            index = imports

        # No support for dwo files for tree artifacts at the moment. This should not throw an
        # irrecoverable exception because we can still generate dwo files for the other artifacts.
        # TODO(b/289089713): Add support for dwo files for tree artifacts.

        _cc_internal.create_lto_backend_action_template(
            actions = actions,
            feature_configuration = feature_configuration,
            additional_inputs = additional_inputs,
            env = env,
            build_variables = build_variables,
            use_pic = use_pic,
            all_bitcode_files = all_bitcode_files,
            index = index,
            bitcode_file = bitcode_file,
            object_file = object_file,
            dwo_file = dwo_file,
        )
    else:
        object_file = actions.declare_shareable_artifact(obj)
        if not create_shared_non_lto:
            imports = actions.declare_shareable_artifact(index_obj + ".imports")
            index = actions.declare_shareable_artifact(index_obj + ".thinlto.bc")
        if should_create_per_object_debug_info:
            dwo_file = actions.declare_shareable_artifact(paths.replace_extension(obj, ".dwo"))

        _create_lto_backend_action(
            actions,
            additional_inputs,
            env,
            build_variables,
            feature_configuration,
            index,
            imports,
            bitcode_file,
            object_file,
            all_bitcode_files,
            dwo_file,
            use_pic,
            None,  # bitcode_file_path
        )

    return LtoBackendArtifactsInfo(
        index = index,
        imports = imports,
        _bitcode_file = bitcode_file,
        _object_file = object_file,
        _dwo_file = dwo_file,
    )

def _add_profile_for_lto_backend(additional_inputs, fdo_context, feature_configuration, build_variables):
    prefetch = getattr(fdo_context, "prefetch_hints_artifact", None)
    if prefetch != None:
        build_variables["fdo_prefetch_hints_path"] = prefetch.path
        additional_inputs.append(fdo_context.prefetch_hints_artifact)
    propeller_optimize_info = getattr(fdo_context, "propeller_optimize_info", None)
    if propeller_optimize_info != None and propeller_optimize_info.cc_profile != None:
        build_variables["propeller_optimize_cc_path"] = propeller_optimize_info.cc_profile
        additional_inputs.append(propeller_optimize_info.cc_profile)
    if propeller_optimize_info != None and propeller_optimize_info.ld_profile != None:
        build_variables["propeller_optimize_ld_path"] = propeller_optimize_info.ld_profile
        additional_inputs.append(propeller_optimize_info.ld_profile)
    if not feature_configuration.is_enabled("autofdo") and \
       not feature_configuration.is_enabled("cs_fdo_optimize") and \
       not feature_configuration.is_enabled("xbinaryfdo"):
        return

    branch_fdo_profile = getattr(fdo_context, "branch_fdo_profile", None)
    if branch_fdo_profile == None:
        fail("Branch FDO profile is None")
    profile = branch_fdo_profile.profile_artifact
    build_variables["fdo_profile_path"] = profile
    additional_inputs.append(profile)

# LINT.IfChange(lto_backends)
def _create_lto_backend_action(
        actions,
        additional_inputs,
        env,
        build_variables,
        feature_configuration,
        index,
        imports,
        bitcode_artifact,
        object_file,
        bitcode_files,
        dwo_file,
        use_pic,
        bitcode_file_path):
    if index != None and index.is_directory:
        fail("index cannot be a TreeArtifact")
    if imports != None and imports.is_directory:
        fail("imports cannot be a TreeArtifact")
    if dwo_file != None and dwo_file.is_directory:
        fail("dwo_file cannot be a TreeArtifact")
    if object_file.is_directory:
        fail("object_file cannot be a TreeArtifact")

    if bitcode_artifact.is_directory and bitcode_file_path == None:
        fail("If bitcode file is a tree artifact, the bitcode file path must contain the path.")
    if not bitcode_artifact.is_directory and bitcode_file_path != None:
        fail("If bitcode file is not a tree artifact, then bitcode file path should be null to not override the path.")

    inputs = _get_lto_backend_action_inputs(index, imports, bitcode_artifact, additional_inputs)
    outputs = _get_lto_backend_action_outputs(object_file, dwo_file)

    _path_variables = _paths_build_variables(
        index,
        object_file,
        dwo_file,
        bitcode_file_path if bitcode_file_path != None else bitcode_artifact,
    )
    _path_variables = _cc_internal.cc_toolchain_variables(vars = _path_variables)
    build_variables = _cc_internal.combine_cc_toolchain_variables(build_variables, _path_variables)

    _cc_internal.create_lto_backend_action(
        actions = actions,
        feature_configuration = feature_configuration,
        build_variables = build_variables,
        use_pic = use_pic,
        inputs = inputs,
        all_bitcode_files = bitcode_files,
        imports = imports,
        outputs = outputs,
        env = env,
    )

def _paths_build_variables(index, object_file, dwo_file, bitcode_file):
    build_variables = {}

    # Ideally, those strings would come directly from the execPath of the Artifacts of
    # the LtoBackendAction.Builder; however, in order to support tree artifacts, we need
    # the bitcode_file_path to be different from the bitcode_tree_artifact execPath.
    # The former is a file path and the latter is the directory path.
    # Therefore we accept strings as inputs rather than artifacts.
    if index != None:
        build_variables["thinlto_index"] = index
    else:
        # An empty input indicates not to perform cross-module optimization.
        build_variables["thinlto_index"] = "/dev/null"

    # The output from the LTO backend step is a native object file.
    build_variables["thinlto_output_object_file"] = object_file

    # The input to the LTO backend step is the bitcode file.
    build_variables["thinlto_input_bitcode_file"] = bitcode_file

    if dwo_file != None:
        build_variables["per_object_debug_info_file"] = dwo_file
        build_variables["is_using_fission"] = ""
    return build_variables

def _get_lto_backend_action_inputs(index, imports, bitcode_file, additional_inputs):
    inputs = [bitcode_file]
    if imports != None:
        # Although the imports file is not used by the LTOBackendAction while the action is
        # executing, it is needed during the input discovery phase, and we must list it as an input
        # to the action in order for it to be preserved under --discard_orphaned_artifacts.
        inputs.append(imports)
    if index != None:
        inputs.append(index)
    return depset(inputs, transitive = [additional_inputs])

def _get_lto_backend_action_outputs(object_file, dwo_file):
    outputs = [object_file]
    if dwo_file != None:
        outputs.append(dwo_file)
    return outputs

# LINT.ThenChange(//src/main/java/com/google/devtools/build/lib/rules/cpp/LtoBackendArtifacts.java:lto_backends)
# LINT.ThenChange(@rules_cc//cc/private/link/lto_backends.bzl:forked_exports)
