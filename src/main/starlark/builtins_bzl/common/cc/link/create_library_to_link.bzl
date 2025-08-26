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
The cc_common.create_library_to_link function.
"""

load(":common/cc/cc_helper_internal.bzl", "is_versioned_shared_library")
load(":common/cc/link/lto_backends.bzl", "create_shared_non_lto_artifacts")
load(":common/paths.bzl", "paths")

cc_common_internal = _builtins.internal.cc_common
cc_internal = _builtins.internal.cc_internal
_EMPTY_LTO = cc_common_internal.create_lto_compilation_context()

_warning = """ Don't use this field. It's intended for internal use and will be changed or removed
    without warning."""

# A library the user can link to. This is different from a simple linker input in that it also has
# a library identifier.
LibraryToLinkInfo = provider(
    "A library the user can link against.",
    fields = {
        "static_library": "Artifact of static library to be linked.",
        "pic_static_library": "Artifact of pic static library to be linked.",
        "dynamic_library": """Artifact of dynamic library to be linked. Always used for runtime
            and used for linking if `interface_library` is not passed.""",
        "interface_library": "Artifact of interface library to be linked.",
        "resolved_symlink_dynamic_library": """The resolved `Artifact` of the dynamic library
            to be linked if `dynamic_library` is a symlink, otherwise this is None.""",
        "resolved_symlink_interface_library": """The resolved `Artifact` of the interface library
            to be linked if `interface_library` is a symlink, otherwise this is None.""",
        "objects": "List of object files in the library.",
        "pic_objects": "List of pic object files in the library.",
        "alwayslink": "Whether to link the static library/objects in the --whole_archive block.",
        "lto_bitcode_files": "List of LTO bitcode files in the library.",
        "pic_lto_bitcode_files": "List of pic LTO bitcode files in the library.",
        "_library_identifier": _warning,
        "_contains_objects": """
This is essential for start-end library functionality. _contains_objects is False when calling
cc_common.create_library_to_link with empty object files. This signifies to start-end that an
archive needs to be used. On the other hand cc_common.link will set object files to exactly
what's in the archive. Start-end library functionality may correctly expand the object files.
In case they are empty, this means also the archive is empty.""" + _warning,
        "_disable_whole_archive": _warning,
        "_must_keep_debug": """
TODO(b/338618120): This is just needed for Go, do not expose to Starlark and try to remove it.
This was introduced to let a linker input declare that it needs debug info in the executable.
Specifically, this was introduced for linking Go into a C++ binary when using the gccgo compiler.
        """ + _warning,
        "_lto_compilation_context": _warning,
        "_pic_lto_compilation_context": _warning,
        "_shared_non_lto_backends": _warning,
        "_pic_shared_non_lto_backends": _warning,
    },
)

# buildifier: disable=function-docstring-args
# buildifier: disable=function-docstring-return
def make_library_to_link(
        *,
        static_library = None,
        pic_static_library = None,
        dynamic_library = None,
        interface_library = None,
        resolved_symlink_dynamic_library = None,
        resolved_symlink_interface_library = None,
        objects = [],
        pic_objects = [],
        alwayslink = False,
        _library_identifier,
        _contains_objects = False,
        _disable_whole_archive = False,
        _must_keep_debug = False,
        _lto_compilation_context = None,
        _pic_lto_compilation_context = None,
        _shared_non_lto_backends = {},
        _pic_shared_non_lto_backends = {}):
    """Constructs a frozen LibraryToLink."""
    if not (static_library or pic_static_library or dynamic_library or interface_library):
        fail("At least on library artifact must be non null")
    if resolved_symlink_dynamic_library and not dynamic_library:
        fail("resolved_symlink_dynamic_library without dynamic_library")
    if resolved_symlink_interface_library and not interface_library:
        fail("resolved_symlink_interface_library without interface_library")
    return LibraryToLinkInfo(
        static_library = static_library,
        pic_static_library = pic_static_library,
        interface_library = interface_library,
        dynamic_library = dynamic_library,
        resolved_symlink_dynamic_library = resolved_symlink_dynamic_library,
        resolved_symlink_interface_library = resolved_symlink_interface_library,
        objects = cc_internal.freeze(objects),
        pic_objects = cc_internal.freeze(pic_objects),
        alwayslink = alwayslink,
        # LTO data duplication is forced by public APIs
        lto_bitcode_files = cc_internal.freeze(_lto_compilation_context.lto_bitcode_inputs().keys() if _lto_compilation_context else []),
        pic_lto_bitcode_files = cc_internal.freeze(_pic_lto_compilation_context.lto_bitcode_inputs().keys() if _pic_lto_compilation_context else []),
        _library_identifier = _library_identifier,
        _contains_objects = _contains_objects,
        _must_keep_debug = _must_keep_debug,
        _disable_whole_archive = _disable_whole_archive,
        _lto_compilation_context = _lto_compilation_context,
        _pic_lto_compilation_context = _pic_lto_compilation_context,
        _shared_non_lto_backends = cc_internal.freeze(_shared_non_lto_backends),
        _pic_shared_non_lto_backends = cc_internal.freeze(_pic_shared_non_lto_backends),
    )

def create_library_to_link(
        *,
        actions,
        feature_configuration = None,
        cc_toolchain = None,
        static_library = None,
        pic_static_library = None,
        dynamic_library = None,
        interface_library = None,
        pic_objects = None,
        objects = None,
        lto_compilation_context = None,
        alwayslink = False,
        dynamic_library_symlink_path = "",
        interface_library_symlink_path = "",
        must_keep_debug = False):
    """Creates a `LibraryToLink` struct with information for linking a single library.

    Validates the names of all the passed in `File`s.

    Creates a symlink for `dynamic_library` and `interface_library` in `_solib_` directory.

    Args:
        actions: (Actions) `actions` object.
        feature_configuration: (FeatureConfiguration) `feature_configuration` to be queried.
        cc_toolchain: (CcToolchainInfo) CcToolchainInfo provider to be used.
        static_library: (File|None) Static library to be linked.
        pic_static_library: (File|None) PIC static library to be linked.
        dynamic_library: (File|None) Dynamic library to be linked. Always used for runtime and used for
            linking if `interface_library` is not passed.
        interface_library: (File|None) Interface library to be linked.
        pic_objects: (list[File]|None) Experimental, do not use.
        objects: (list[File]|None) Experimental, do not use.
        lto_compilation_context: (LtoCompilationContext) Experimental, do not use.
        alwayslink: (bool) Whether to link the static library/objects in the --whole_archive block.
        dynamic_library_symlink_path: (str) Override the default path of the dynamic library link
            in the solib directory. Empty string to use the default.
        interface_library_symlink_path: (str) Override the default path of the interface library
            link in the solib directory. Empty string to use the default.
        must_keep_debug: (bool) Experimental, do not use.

    Returns:
        (LibraryToLink)
    """
    errors = []
    _fail = lambda error: errors.append(error)
    _validate_ext = lambda *args, **kwargs: _validate_extension(fail = _fail, *args, **kwargs)

    if dynamic_library_symlink_path:
        _validate_symlink_path("dynamic_library_symlink_path", dynamic_library_symlink_path)
        _validate_ext(
            dynamic_library_symlink_path,
            [".so", ".dylib", ".dll", ".pyd", ".wasm", ".tgt", ".vpi"],
            is_versioned_shared_library,
            empty_ext = True,
        )

    if interface_library_symlink_path:
        _validate_symlink_path("interface_library_symlink_path", interface_library_symlink_path)
        _validate_ext(interface_library_symlink_path, [".ifso", ".tbd", ".lib", ".dll.a"])

    if static_library:
        if alwayslink:
            _validate_ext(static_library, [".a", ".lib", ".rlib"] + [".lo"], not_ext = [".pic.lo", ".if.lib"], empty_ext = True)
        else:
            _validate_ext(static_library, [".a", ".lib", ".rlib"], not_ext = [".lo.lib", ".if.lib"], empty_ext = True)

    if pic_static_library:
        if alwayslink:
            # Ideally we'd allow only `.pic.lo` instead of `.lo`, `.pic.a` instead of `.a`, `.lo.lib` instead of `.lib`
            # but in reality pic libs are often called same as no-pic.
            _validate_ext(pic_static_library, [".a", ".lib", ".rlib"] + [".lo"], not_ext = [".if.lib"], empty_ext = True)
        else:
            _validate_ext(pic_static_library, [".a", ".lib", ".rlib"], not_ext = [".lo.lib", ".if.lib"], empty_ext = True)

    resolved_symlink_dynamic_library = None
    if dynamic_library:
        _validate_ext(dynamic_library, [".so", ".dylib", ".dll", ".pyd", ".wasm", ".tgt", ".vpi"], is_versioned_shared_library, empty_ext = True)
        if not cc_toolchain:
            fail("If you pass 'dynamic_library', you must also pass a 'cc_toolchain'")
        if not feature_configuration:
            fail("If you pass 'dynamic_library', you must also pass a 'feature_configuration'")
        if not feature_configuration.is_enabled("targets_windows"):
            resolved_symlink_dynamic_library = dynamic_library
            if dynamic_library_symlink_path:
                if dynamic_library.short_path.startswith("_solib_"):
                    fail("dynamic_library must not be a symbolic link in the solib directory. Got '%s'" % dynamic_library.short_path)
                dynamic_library = cc_internal.dynamic_library_symlink2(actions, dynamic_library, cc_toolchain._solib_dir, dynamic_library_symlink_path)
            else:
                dynamic_library = cc_internal.dynamic_library_symlink(actions, dynamic_library, cc_toolchain._solib_dir, True, True)

    resolved_symlink_interface_library = None
    if interface_library:
        _validate_ext(interface_library, [".ifso", ".tbd", ".lib", ".dll.a", ".so", ".dylib"])
        if not cc_toolchain:
            fail("If you pass 'interface_library', you must also pass a 'cc_toolchain'")
        if not feature_configuration:
            fail("If you pass 'interface_library', you must also pass a 'feature_configuration'")
        if not feature_configuration.is_enabled("targets_windows"):
            resolved_symlink_interface_library = interface_library
            if interface_library_symlink_path:
                if interface_library.short_path.startswith("_solib_"):
                    fail("dynamic_library must not be a symbolic link in the solib directory. Got '%s'" % dynamic_library.short_path)
                interface_library = cc_internal.dynamic_library_symlink2(actions, interface_library, cc_toolchain._solib_dir, interface_library_symlink_path)
            else:
                interface_library = cc_internal.dynamic_library_symlink(actions, interface_library, cc_toolchain._solib_dir, True, True)

    if errors:
        fail("\n".join(errors))

    identifier = static_library or pic_static_library or dynamic_library or interface_library
    if not identifier:
        fail("Must pass at least one of the following parameters: static_library, pic_static_library, " +
             "dynamic_library and interface_library.")

    library_identifier = identifier.short_path.removesuffix(".pic.a").removesuffix(".nopic.a").removesuffix(".pic.lo")
    library_identifier = paths.replace_extension(library_identifier, "")

    objects = objects or []
    pic_objects = pic_objects or []

    if lto_compilation_context and static_library and objects != None:
        shared_non_lto_backends = create_shared_non_lto_artifacts(
            actions,
            lto_compilation_context,
            False,
            feature_configuration,
            cc_toolchain,
            False,
            objects,
        )
    else:
        shared_non_lto_backends = None
    lto_compilation_context = lto_compilation_context or _EMPTY_LTO

    return make_library_to_link(
        static_library = static_library,
        pic_static_library = pic_static_library,
        objects = objects,
        pic_objects = pic_objects,
        dynamic_library = dynamic_library,
        resolved_symlink_dynamic_library = resolved_symlink_dynamic_library,
        interface_library = interface_library,
        resolved_symlink_interface_library = resolved_symlink_interface_library,
        alwayslink = alwayslink,
        _library_identifier = library_identifier,
        _must_keep_debug = must_keep_debug,
        _lto_compilation_context = lto_compilation_context,
        _shared_non_lto_backends = shared_non_lto_backends,
        _contains_objects = bool(objects) or bool(pic_objects),
    )

def _validate_symlink_path(attr, path):
    if not path or paths.is_absolute(path) or paths.contains_up_level_references(path):
        fail("%s must be a relative file path. Got '%s" % (attr, path))

def _validate_extension(path, extensions, func = None, not_ext = [], fail = fail, empty_ext = False):
    path = getattr(path, "basename", path)  # Handle str|File
    for ext in not_ext:
        if path.endswith(ext):
            fail("'%s' does not have any of the allowed extensions %s" % (path, ", ".join(extensions)))
    for ext in extensions:
        if path.endswith(ext):
            return
    if empty_ext:
        _, actual_ext = paths.split_extension(path)
        if actual_ext == "":
            return
    if func and func(struct(basename = path)):
        return
    fail("'%s' does not have any of the allowed extensions %s" % (path, ", ".join(extensions)))
