// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.annotations.VisibleForTesting;

/** Rule class definitions for C++ rules. */
public class CppRuleClasses {
  /** A string constant for the Objective-C language feature. */
  public static final String LANG_OBJC = "lang_objc";

  /** Name of the feature that will be exempt from flag filtering when nocopts are used */
  public static final String UNFILTERED_COMPILE_FLAGS_FEATURE_NAME = "unfiltered_compile_flags";

  /** A string constant for the parse_headers feature. */
  public static final String PARSE_HEADERS = "parse_headers";

  /**
   * A string constant for the module_maps feature; this is a precondition to the layering_check and
   * header_modules features.
   */
  public static final String MODULE_MAPS = "module_maps";

  /** A string constant for the cpp_modules feature. */
  public static final String CPP_MODULES = "cpp_modules";

  /**
   * A string constant for the serialized_diagnostics_file feature. This feature generates the .dia
   * file.
   */
  public static final String SERIALIZED_DIAGNOSTICS_FILE = "serialized_diagnostics_file";

  /** A string constant for the module_map_home_cwd feature. */
  public static final String MODULE_MAP_HOME_CWD = "module_map_home_cwd";

  /**
   * A string constant for the module_map_without_extern_module feature.
   *
   * <p>This features is a transitional feature; enabling it means that generated module maps will
   * not have "extern module" declarations inside them; instead, the module maps need to be passed
   * via the dependent_module_map_files build variable.
   *
   * <p>This variable is phrased negatively to aid the roll-out: currently, the default is that
   * "extern module" declarations are generated.
   */
  public static final String MODULE_MAP_WITHOUT_EXTERN_MODULE = "module_map_without_extern_module";

  /** A string constant for the layering_check feature. */
  public static final String LAYERING_CHECK = "layering_check";

  /**
   * A string constant for the use_header_modules feature.
   *
   * <p>This feature is only used during rollout; we expect to default enable this once we have
   * verified that module-enabled compilation is stable enough.
   */
  public static final String USE_HEADER_MODULES = "use_header_modules";

  /**
   * A string constant for the generate_submodules feature.
   *
   * <p>This feature is only used temporarily to make the switch to using submodules easier. With
   * submodules, each header of a cc_library is placed into a submodule of the module generated for
   * the appropriate target. As this influences the layering_check semantics and needs to be synced
   * with a clang release, we want to be able to switch back and forth easily.
   */
  public static final String GENERATE_SUBMODULES = "generate_submodules";


  /**
   * A string constant for the no_legacy_features feature.
   *
   * <p>If this feature is enabled, Bazel will not extend the crosstool configuration with the
   * default legacy feature set.
   */
  public static final String NO_LEGACY_FEATURES = "no_legacy_features";

  /** A string constant for the feature that makes us build per-object debug info files. */
  public static final String PER_OBJECT_DEBUG_INFO = "per_object_debug_info";

  /**
   * A string constant for the PIC feature.
   *
   * <p>If this feature is active (currently it cannot be switched off) and PIC compilation is
   * requested, the "pic" build variable will be defined with an empty string as its value.
   */
  public static final String PIC = "pic";

  /** A string constant for a feature that indicates that the toolchain can produce PIC objects. */
  public static final String SUPPORTS_PIC = "supports_pic";

  /**
   * A string constant for a feature that indicates that PIC compiles are preferred for binaries
   * even in optimized builds. For configurations that use dynamic linking for tests, this provides
   * increases sharing of artifacts between tests and binaries at the cost of performance overhead.
   */
  public static final String PREFER_PIC_FOR_OPT_BINARIES = "prefer_pic_for_opt_binaries";

  /** A string constant for the feature the represents preprocessor defines. */
  public static final String PREPROCESSOR_DEFINES = "preprocessor_defines";

  /** A string constant for the includes feature. */
  public static final String INCLUDES = "includes";

  /** A string constant for the include_paths feature. */
  public static final String INCLUDE_PATHS = "include_paths";

  /** A string constant for the external_include_paths feature. */
  public static final String EXTERNAL_INCLUDE_PATHS = "external_include_paths";

  /** A string constant for the feature signalling static linking mode. */
  public static final String STATIC_LINKING_MODE = "static_linking_mode";

  /** A string constant for the feature signalling dynamic linking mode. */
  public static final String DYNAMIC_LINKING_MODE = "dynamic_linking_mode";

  /** A string constant for the ThinLTO feature. */
  public static final String THIN_LTO = "thin_lto";

  /** A string constant for the LTO indexing bitcode feature. */
  public static final String NO_USE_LTO_INDEXING_BITCODE_FILE = "no_use_lto_indexing_bitcode_file";

  /** A string constant for the LTO separate native object directory feature. */
  public static final String USE_LTO_NATIVE_OBJECT_DIRECTORY = "use_lto_native_object_directory";

  /*
   * A string constant for allowing implicit ThinLTO enablement for AFDO.
   */
  public static final String AUTOFDO_IMPLICIT_THINLTO = "autofdo_implicit_thinlto";

  /*
   * A string constant for enabling ThinLTO for AFDO implicitly.
   */
  public static final String ENABLE_AFDO_THINLTO = "enable_afdo_thinlto";

  /*
   * A string constant for enabling ThinLTO for FDO implicitly.
   */
  public static final String ENABLE_FDO_THINLTO = "enable_fdo_thinlto";

  /*
   * A string constant for enabling ThinLTO for XFDO implicitly.
   */
  public static final String ENABLE_XFDO_THINLTO = "enable_xbinaryfdo_thinlto";

  /** A string constant for the split functions feature. */
  public static final String SPLIT_FUNCTIONS = "split_functions";

  /** A string constant for enabling split functions for FDO implicitly. */
  public static final String ENABLE_FDO_SPLIT_FUNCTIONS = "enable_fdo_split_functions";

  /** A string constant for the fsafdo feature. */
  public static final String FSAFDO = "fsafdo";

  /** A string constant for enabling fsafdo for AutoFDO implicitly. */
  public static final String ENABLE_FSAFDO = "enable_fsafdo";

  /** A string constant for enabling memprof_optimize for AutoFDO implicitly. */
  public static final String ENABLE_AUTOFDO_MEMPROF_OPTIMIZE = "enable_autofdo_memprof_optimize";

  /** A string constant for allowing memprof_optimize for AutoFDO implicitly. */
  public static final String AUTOFDO_IMPLICIT_MEMPROF_OPTIMIZE =
      "autofdo_implicit_memprof_optimize";

  /**
   * A string constant for allowing use of shared LTO backend actions for linkstatic tests building
   * with ThinLTO.
   */
  public static final String THIN_LTO_LINKSTATIC_TESTS_USE_SHARED_NONLTO_BACKENDS =
      "thin_lto_linkstatic_tests_use_shared_nonlto_backends";

  /**
   * A string constant for allowing use of shared LTO backend actions for all linkstatic links
   * building with ThinLTO.
   */
  public static final String THIN_LTO_ALL_LINKSTATIC_USE_SHARED_NONLTO_BACKENDS =
      "thin_lto_all_linkstatic_use_shared_nonlto_backends";

  /** A string constant for native deps links. */
  public static final String NATIVE_DEPS_LINK = "native_deps_link";

  /** A string constant for java launcher links. */
  public static final String JAVA_LAUNCHER_LINK = "java_launcher_link";

  /** A string constant for python launcher links. */
  public static final String PY_LAUNCHER_LINK = "py_launcher_link";

  /**
   * A string constant for the PDB file generation feature, should only be used for toolchains
   * targeting Windows that include a linker producing PDB files
   */
  public static final String GENERATE_PDB_FILE = "generate_pdb_file";

  /** A string constant for a feature to copy dynamic libraries to the binary's directory. */
  public static final String COPY_DYNAMIC_LIBRARIES_TO_BINARY = "copy_dynamic_libraries_to_binary";

  /** A string constant for a feature to statically link the C++ runtimes. */
  public static final String STATIC_LINK_CPP_RUNTIMES = "static_link_cpp_runtimes";

  /**
   * A string constant for a feature that indicates we are using a toolchain building for Windows.
   */
  public static final String TARGETS_WINDOWS = "targets_windows";

  /**
   * A string constant for a feature that indicates we are using a toolchain building for Windows.
   */
  public static final String SUPPORTS_INTERFACE_SHARED_LIBRARIES =
      "supports_interface_shared_libraries";

  /** A string constant for /showIncludes parsing feature, should only be used for MSVC toolchain */
  public static final String PARSE_SHOWINCLUDES = "parse_showincludes";

  /** A string constant for a feature that, if enabled, disables .d file handling. */
  public static final String NO_DOTD_FILE = "no_dotd_file";

  /**
   * A string constant for a feature that, if enabled, shortens the virtual include paths via
   * hashing.
   */
  public static final String SHORTEN_VIRTUAL_INCLUDES = "shorten_virtual_includes";

  /*
   * A string constant for the fdo_instrument feature.
   */
  public static final String FDO_INSTRUMENT = "fdo_instrument";

  /** A string constant for the cs_fdo_instrument feature. */
  public static final String CS_FDO_INSTRUMENT = "cs_fdo_instrument";

  /** A string constant for the fdo_optimize feature. */
  public static final String FDO_OPTIMIZE = "fdo_optimize";

  /** A string constant for the cs_fdo_optimize feature. */
  public static final String CS_FDO_OPTIMIZE = "cs_fdo_optimize";

  /** A string constant for the cache prefetch hints feature. */
  public static final String FDO_PREFETCH_HINTS = "fdo_prefetch_hints";

  /** A string constant for the propeller optimize feature. */
  public static final String PROPELLER_OPTIMIZE = "propeller_optimize";

  /** A string constant for the memprof profile optimization feature. */
  public static final String MEMPROF_OPTIMIZE = "memprof_optimize";

  /** A string constant for the autofdo feature. */
  public static final String AUTOFDO = "autofdo";


  /** A string constant for the xbinaryfdo feature. */
  public static final String XBINARYFDO = "xbinaryfdo";

  /** A string constant for the coverage feature. */
  public static final String COVERAGE = "coverage";

  /** Produce artifacts for coverage in llvm coverage mapping format. */
  public static final String LLVM_COVERAGE_MAP_FORMAT = "llvm_coverage_map_format";

  /** Produce artifacts for coverage in gcc coverage mapping format. */
  public static final String GCC_COVERAGE_MAP_FORMAT = "gcc_coverage_map_format";

  /** A feature marking that the toolchain can use --start-lib/--end-lib flags */
  public static final String SUPPORTS_START_END_LIB = "supports_start_end_lib";

  /**
   * A feature marking that the toolchain can produce binaries that load shared libraries at
   * runtime.
   */
  public static final String SUPPORTS_DYNAMIC_LINKER = "supports_dynamic_linker";


  public static final String COMPILER_PARAM_FILE = "compiler_param_file";

  /**
   * A feature to control whether to use param files for archiving commands. This can be applied to
   * individual targets.
   */
  public static final String ARCHIVE_PARAM_FILE = "archive_param_file";

  /** A feature to use gcc quoting for linking param files. */
  public static final String GCC_QUOTING_FOR_PARAM_FILES = "gcc_quoting_for_param_files";

  /**
   * A feature to indicate that this target generates debug symbols for a dSYM file. For Apple
   * platform only.
   */
  public static final String GENERATE_DSYM_FILE_FEATURE_NAME = "generate_dsym_file";

  /**
   * A feature to indicate that this target does not generate debug symbols. For Apple platform
   * only.
   *
   * <p>Note that the crosstool does not support feature negation in FlagSet.with_feature, which is
   * the mechanism used to condition linker arguments here. Therefore, we expose
   * "no_generate_debug_symbols" in addition to "generate_dsym_file"
   */
  public static final String NO_GENERATE_DEBUG_SYMBOLS_FEATURE_NAME = "no_generate_debug_symbols";

  /** A feature to indicate whether to generate linkmap. */
  public static final String GENERATE_LINKMAP_FEATURE_NAME = "generate_linkmap";

  /** A feature to indicate whether to do linker deadstrip. For Apple platform only. */
  public static final String DEAD_STRIP_FEATURE_NAME = "dead_strip";

  /** Name of the exec group that Cpp link actions run under */
  @VisibleForTesting public static final String CPP_LINK_EXEC_GROUP = "cpp_link";
}
