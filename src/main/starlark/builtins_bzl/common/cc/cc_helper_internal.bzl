# Copyright 2024 The Bazel Authors. All rights reserved.
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
Utility functions for C++ rules that don't depend on cc_common.

Only use those within C++ implementation. The others need to go through cc_common.
"""

load(":common/paths.bzl", "paths")

cc_common_internal = _builtins.internal.cc_common

CREATE_COMPILE_ACTION_API_ALLOWLISTED_PACKAGES = [("", "devtools/rust/cc_interop"), ("", "third_party/crubit"), ("", "tools/build_defs/clif")]

PRIVATE_STARLARKIFICATION_ALLOWLIST = [
    ("_builtins", ""),
    # Android rules
    ("", "tools/build_defs/android"),
    ("", "third_party/bazel_rules/rules_android"),
    ("build_bazel_rules_android", ""),
    ("rules_android", ""),
    # Apple rules
    ("", "third_party/bazel_rules/rules_apple"),
    ("apple_support", ""),
    ("rules_apple", ""),
    # C++ rules
    ("", "bazel_internal/test_rules/cc"),
    ("", "third_party/bazel_rules/rules_cc"),
    ("", "tools/build_defs/cc"),
    ("rules_cc", ""),
    # CUDA rules
    ("", "third_party/gpus/cuda"),
    # Go rules
    ("", "tools/build_defs/go"),
    # Java rules
    ("", "third_party/bazel_rules/rules_java"),
    ("rules_java", ""),
    # Objc rules
    ("", "tools/build_defs/objc"),
    # Protobuf rules
    ("", "third_party/protobuf"),
    ("protobuf", ""),
    ("com_google_protobuf", ""),
    # Rust rules
    ("", "rust/private"),
    ("rules_rust", "rust/private"),
] + CREATE_COMPILE_ACTION_API_ALLOWLISTED_PACKAGES

def check_private_api():
    cc_common_internal.check_private_api(allowlist = PRIVATE_STARLARKIFICATION_ALLOWLIST, depth = 2)

def wrap_with_check_private_api(symbol):
    """
    Protects the symbol so it can only be used internally.

    Returns:
      A function. When the function is invoked (without any params), the check
      is done and if it passes the symbol is returned.
    """

    def callback():
        cc_common_internal.check_private_api(allowlist = PRIVATE_STARLARKIFICATION_ALLOWLIST)
        return symbol

    return callback

CPP_SOURCE_TYPE_HEADER = "HEADER"
CPP_SOURCE_TYPE_SOURCE = "SOURCE"
CPP_SOURCE_TYPE_CLIF_INPUT_PROTO = "CLIF_INPUT_PROTO"

# LINT.IfChange(forked_exports)

_CC_SOURCE = [".cc", ".cpp", ".cxx", ".c++", ".C", ".cu", ".cl"]
_C_SOURCE = [".c"]
_OBJC_SOURCE = [".m"]
_OBJCPP_SOURCE = [".mm"]
_CLIF_INPUT_PROTO = [".ipb"]
_CLIF_OUTPUT_PROTO = [".opb"]
_CC_HEADER = [".h", ".hh", ".hpp", ".ipp", ".hxx", ".h++", ".inc", ".inl", ".tlh", ".tli", ".H", ".tcc"]
_CC_TEXTUAL_INCLUDE = [".inc"]
_ASSEMBLER_WITH_C_PREPROCESSOR = [".S"]
_ASSEMBLER = [".s", ".asm"]
_ARCHIVE = [".a", ".lib"]
_PIC_ARCHIVE = [".pic.a"]
_ALWAYSLINK_LIBRARY = [".lo"]
_ALWAYSLINK_PIC_LIBRARY = [".pic.lo"]
_SHARED_LIBRARY = [".so", ".dylib", ".dll", ".wasm"]
_INTERFACE_SHARED_LIBRARY = [".ifso", ".tbd", ".lib", ".dll.a"]
_OBJECT_FILE = [".o", ".obj"]
_PIC_OBJECT_FILE = [".pic.o"]
_CPP_MODULE = [".pcm", ".gcm", ".ifc"]
_CPP_MODULE_MAP = [".cppmap"]
_LTO_INDEXING_OBJECT_FILE = [".indexing.o"]

_CC_AND_OBJC = []
_CC_AND_OBJC.extend(_CC_SOURCE)
_CC_AND_OBJC.extend(_C_SOURCE)
_CC_AND_OBJC.extend(_OBJC_SOURCE)
_CC_AND_OBJC.extend(_OBJCPP_SOURCE)
_CC_AND_OBJC.extend(_CC_HEADER)
_CC_AND_OBJC.extend(_ASSEMBLER)
_CC_AND_OBJC.extend(_ASSEMBLER_WITH_C_PREPROCESSOR)

_DISALLOWED_HDRS_FILES = []
_DISALLOWED_HDRS_FILES.extend(_ARCHIVE)
_DISALLOWED_HDRS_FILES.extend(_PIC_ARCHIVE)
_DISALLOWED_HDRS_FILES.extend(_ALWAYSLINK_LIBRARY)
_DISALLOWED_HDRS_FILES.extend(_ALWAYSLINK_PIC_LIBRARY)
_DISALLOWED_HDRS_FILES.extend(_SHARED_LIBRARY)
_DISALLOWED_HDRS_FILES.extend(_INTERFACE_SHARED_LIBRARY)
_DISALLOWED_HDRS_FILES.extend(_OBJECT_FILE)
_DISALLOWED_HDRS_FILES.extend(_PIC_OBJECT_FILE)

extensions = struct(
    CC_SOURCE = _CC_SOURCE,
    C_SOURCE = _C_SOURCE,
    OBJC_SOURCE = _OBJC_SOURCE,
    OBJCPP_SOURCE = _OBJCPP_SOURCE,
    CC_HEADER = _CC_HEADER,
    CC_TEXTUAL_INCLUDE = _CC_TEXTUAL_INCLUDE,
    ASSEMBLER_WITH_C_PREPROCESSOR = _ASSEMBLER_WITH_C_PREPROCESSOR,
    # TODO(b/345158656): Remove ASSESMBLER_WITH_C_PREPROCESSOR after next blaze release
    ASSESMBLER_WITH_C_PREPROCESSOR = _ASSEMBLER_WITH_C_PREPROCESSOR,
    ASSEMBLER = _ASSEMBLER,
    CLIF_INPUT_PROTO = _CLIF_INPUT_PROTO,
    CLIF_OUTPUT_PROTO = _CLIF_OUTPUT_PROTO,
    ARCHIVE = _ARCHIVE,
    PIC_ARCHIVE = _PIC_ARCHIVE,
    ALWAYSLINK_LIBRARY = _ALWAYSLINK_LIBRARY,
    ALWAYSLINK_PIC_LIBRARY = _ALWAYSLINK_PIC_LIBRARY,
    SHARED_LIBRARY = _SHARED_LIBRARY,
    OBJECT_FILE = _OBJECT_FILE,
    PIC_OBJECT_FILE = _PIC_OBJECT_FILE,
    CC_AND_OBJC = _CC_AND_OBJC,
    DISALLOWED_HDRS_FILES = _DISALLOWED_HDRS_FILES,  # Also includes VERSIONED_SHARED_LIBRARY files.
    CPP_MODULE = _CPP_MODULE,
    CPP_MODULE_MAP = _CPP_MODULE_MAP,
    LTO_INDEXING_OBJECT_FILE = _LTO_INDEXING_OBJECT_FILE,
)

def _artifact_category_info_init(name, default_prefix, *extensions):
    return {
        "name": name,
        "default_prefix": default_prefix,
        "default_extension": extensions[0],
        "allowed_extensions": extensions,
    }

# buildifier: disable=unused-variable
_ArtifactCategoryInfo, _unused_new_aci = provider(
    """A category of artifacts that are candidate input/output to an action, for
     which the toolchain can select a single artifact.""",
    fields = ["name", "default_prefix", "default_extension", "allowed_extensions"],
    init = _artifact_category_info_init,
)

# TODO: b/433485282 - remove duplicated extensions lists with above constants
_artifact_categories = [
    _ArtifactCategoryInfo("STATIC_LIBRARY", "lib", ".a", ".lib"),
    _ArtifactCategoryInfo("ALWAYSLINK_STATIC_LIBRARY", "lib", ".lo", ".lo.lib"),
    _ArtifactCategoryInfo("DYNAMIC_LIBRARY", "lib", ".so", ".dylib", ".dll", ".wasm"),
    _ArtifactCategoryInfo("EXECUTABLE", "", "", ".exe", ".wasm"),
    _ArtifactCategoryInfo("INTERFACE_LIBRARY", "lib", ".ifso", ".tbd", ".if.lib", ".lib"),
    _ArtifactCategoryInfo("PIC_FILE", "", ".pic"),
    _ArtifactCategoryInfo("INCLUDED_FILE_LIST", "", ".d"),
    _ArtifactCategoryInfo("SERIALIZED_DIAGNOSTICS_FILE", "", ".dia"),
    _ArtifactCategoryInfo("OBJECT_FILE", "", ".o", ".obj"),
    _ArtifactCategoryInfo("PIC_OBJECT_FILE", "", ".pic.o"),
    _ArtifactCategoryInfo("CPP_MODULE", "", ".pcm"),
    _ArtifactCategoryInfo("CPP_MODULE_GCM", "", ".gcm"),
    _ArtifactCategoryInfo("CPP_MODULE_IFC", "", ".ifc"),
    _ArtifactCategoryInfo("CPP_MODULES_INFO", "", ".CXXModules.json"),
    _ArtifactCategoryInfo("CPP_MODULES_DDI", "", ".ddi"),
    _ArtifactCategoryInfo("CPP_MODULES_MODMAP", "", ".modmap"),
    _ArtifactCategoryInfo("CPP_MODULES_MODMAP_INPUT", "", ".modmap.input"),
    _ArtifactCategoryInfo("GENERATED_ASSEMBLY", "", ".s", ".asm"),
    _ArtifactCategoryInfo("PROCESSED_HEADER", "", ".processed"),
    _ArtifactCategoryInfo("GENERATED_HEADER", "", ".h"),
    _ArtifactCategoryInfo("PREPROCESSED_C_SOURCE", "", ".i"),
    _ArtifactCategoryInfo("PREPROCESSED_CPP_SOURCE", "", ".ii"),
    _ArtifactCategoryInfo("COVERAGE_DATA_FILE", "", ".gcno"),
    # A matched-clif protobuf. Typically in binary format, but could be text
    # depending on the options passed to the clif_matcher.
    _ArtifactCategoryInfo("CLIF_OUTPUT_PROTO", "", ".opb"),
]

artifact_category_names = struct(**{ac.name: ac.name for ac in _artifact_categories})

output_subdirectories = struct(
    OBJS = "_objs",
    PIC_OBJS = "_pic_objs",
    DOTD_FILES = "_dotd",
    PIC_DOTD_FILES = "_pic_dotd",
    DIA_FILES = "_dia",
    PIC_DIA_FILES = "_pic_dia",
)

def should_create_per_object_debug_info(feature_configuration, cpp_configuration):
    return cpp_configuration.fission_active_for_current_compilation_mode() and \
           feature_configuration.is_enabled("per_object_debug_info")

def is_versioned_shared_library_extension_valid(shared_library_name):
    """Validates the name against the regex "^.+\\.((so)|(dylib))(\\.\\d\\w*)+$",

    Args:
        shared_library_name: (str) the name to validate

    Returns:
        (bool)
    """

    # must match VERSIONED_SHARED_LIBRARY.
    for ext in (".so.", ".dylib."):
        name, _, version = shared_library_name.rpartition(ext)
        if name and version:
            version_parts = version.split(".")
            for part in version_parts:
                if not part[0].isdigit():
                    return False
                for c in part[1:].elems():
                    if not (c.isalnum() or c == "_"):
                        return False
            return True
    return False

def _is_repository_main(repository):
    return repository == ""

def package_source_root(repository, package, sibling_repository_layout):
    """
    Determines the source root for a given repository and package.

    Args:
      repository: The repository to get the source root for.
      package: The package to get the source root for.
      sibling_repository_layout: Whether the repository layout is a sibling repository layout.

    Returns:
      The source root for the given repository and package.
    """
    if _is_repository_main(repository) or sibling_repository_layout:
        return package
    if repository.startswith("@"):
        repository = repository[1:]
    return paths.get_relative(paths.get_relative("external", repository), package)

def repository_exec_path(repository, sibling_repository_layout):
    """
    Determines the exec path for a given repository.

    Args:
      repository: The repository to get the exec path for.
      sibling_repository_layout: Whether the repository layout is a sibling repository layout.

    Returns:
      The exec path for the given repository.
    """
    if _is_repository_main(repository):
        return ""
    prefix = "external"
    if sibling_repository_layout:
        prefix = ".."
    if repository.startswith("@"):
        repository = repository[1:]
    return paths.get_relative(prefix, repository)

def is_stamping_enabled(ctx):
    """Returns whether to encode build information into the binary.

    Args:
        ctx: The rule context.

    Returns:
    (int): 1: Always stamp the build information into the binary, even in [--nostamp][stamp] builds.
        This setting should be avoided, since it potentially kills remote caching for the binary and
        any downstream actions that depend on it.
        0: Always replace build information by constant values. This gives good build result caching.
        -1: Embedding of build information is controlled by the [--[no]stamp][stamp] flag.
    """
    if ctx.configuration.is_tool_configuration():
        return 0
    stamp = 0
    if hasattr(ctx.attr, "stamp"):
        stamp = ctx.attr.stamp
    return stamp

# LINT.ThenChange(@rules_cc//cc/common/cc_helper_internal.bzl:forked_exports)

def is_shared_library(file):
    return file.extension in ["so", "dylib", "dll", "pyd", "wasm", "tgt", "vpi"]

def is_versioned_shared_library(file):
    # Because regex matching can be slow, we first do a quick check for ".so." and ".dylib."
    # substring before risking the full-on regex match. This should eliminate the performance
    # hit on practically every non-qualifying file type.
    if ".so." not in file.basename and ".dylib." not in file.basename:
        return False
    return is_versioned_shared_library_extension_valid(file.basename)

def use_pic_for_binaries(cpp_config, feature_configuration):
    """
    Returns whether binaries must be compiled with position independent code.
    """
    return cpp_config.force_pic() or (
        feature_configuration.is_enabled("supports_pic") and
        (cpp_config.compilation_mode() != "opt" or feature_configuration.is_enabled("prefer_pic_for_opt_binaries"))
    )

def use_pic_for_dynamic_libs(cpp_config, feature_configuration):
    """Determines if we should apply -fPIC for this rule's C++ compilations.

    This determination is
    generally made by the global C++ configuration settings "needsPic" and "usePicForBinaries".
    However, an individual rule may override these settings by applying -fPIC" to its "nocopts"
    attribute. This allows incompatible rules to "opt out" of global PIC settings (see bug:
    "Provide a way to turn off -fPIC for targets that can't be built that way").

    Returns:
       true if this rule's compilations should apply -fPIC, false otherwise
    """
    return (cpp_config.force_pic() or
            feature_configuration.is_enabled("supports_pic"))
