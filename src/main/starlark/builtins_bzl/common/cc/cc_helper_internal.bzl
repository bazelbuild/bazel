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

artifact_category = struct(
    STATIC_LIBRARY = "STATIC_LIBRARY",
    ALWAYSLINK_STATIC_LIBRARY = "ALWAYSLINK_STATIC_LIBRARY",
    DYNAMIC_LIBRARY = "DYNAMIC_LIBRARY",
    EXECUTABLE = "EXECUTABLE",
    INTERFACE_LIBRARY = "INTERFACE_LIBRARY",
    PIC_FILE = "PIC_FILE",
    INCLUDED_FILE_LIST = "INCLUDED_FILE_LIST",
    SERIALIZED_DIAGNOSTICS_FILE = "SERIALIZED_DIAGNOSTICS_FILE",
    OBJECT_FILE = "OBJECT_FILE",
    PIC_OBJECT_FILE = "PIC_OBJECT_FILE",
    CPP_MODULE = "CPP_MODULE",
    CPP_MODULE_GCM = "CPP_MODULE_GCM",
    CPP_MODULE_IFC = "CPP_MODULE_IFC",
    CPP_MODULES_INFO = "CPP_MODULES_INFO",
    CPP_MODULES_DDI = "CPP_MODULES_DDI",
    CPP_MODULES_MODMAP = "CPP_MODULES_MODMAP",
    CPP_MODULES_MODMAP_INPUT = "CPP_MODULES_MODMAP_INPUT",
    GENERATED_ASSEMBLY = "GENERATED_ASSEMBLY",
    PROCESSED_HEADER = "PROCESSED_HEADER",
    GENERATED_HEADER = "GENERATED_HEADER",
    PREPROCESSED_C_SOURCE = "PREPROCESSED_C_SOURCE",
    PREPROCESSED_CPP_SOURCE = "PREPROCESSED_CPP_SOURCE",
    COVERAGE_DATA_FILE = "COVERAGE_DATA_FILE",
    CLIF_OUTPUT_PROTO = "CLIF_OUTPUT_PROTO",
)

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
)

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

def should_create_per_object_debug_info(feature_configuration, cpp_configuration):
    return cpp_configuration.fission_active_for_current_compilation_mode() and \
           feature_configuration.is_enabled("per_object_debug_info")

def is_versioned_shared_library_extension_valid(shared_library_name):
    # validate against the regex "^.+\\.((so)|(dylib))(\\.\\d\\w*)+$",
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

def is_shared_library(file):
    return file.extension in ["so", "dylib", "dll", "pyd", "wasm", "tgt", "vpi"]

def is_versioned_shared_library(file):
    # Because regex matching can be slow, we first do a quick check for ".so." and ".dylib."
    # substring before risking the full-on regex match. This should eliminate the performance
    # hit on practically every non-qualifying file type.
    if ".so." not in file.basename and ".dylib." not in file.basename:
        return False
    return is_versioned_shared_library_extension_valid(file.basename)

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
