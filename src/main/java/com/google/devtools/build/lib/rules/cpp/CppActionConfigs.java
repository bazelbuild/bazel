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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;

/**
 * A helper class for creating action_configs for the c++ actions.
 *
 * <p>TODO(b/30109612): Replace this with action_configs in the crosstool instead of putting it in
 * legacy features.
 */
public class CppActionConfigs {

  /** A platform for C++ tool invocations. */
  public enum CppPlatform {
    LINUX,
    MAC
  }

  /** A string constant for the macOS target libc value. */
  public static final String MACOS_TARGET_LIBC = "macosx";

  // Note: these features won't be added to the crosstools that defines no_legacy_features feature
  // (e.g. ndk, apple, enclave crosstools). Those need to be modified separately.
  public static ImmutableList<CToolchain.Feature> getLegacyFeatures(
      CppPlatform platform,
      ImmutableSet<String> existingFeatureNames,
      String cppLinkDynamicLibraryToolPath,
      boolean supportsEmbeddedRuntimes,
      boolean supportsInterfaceSharedLibraries,
      boolean doNotSplitLinkingCmdline) {

    ImmutableList.Builder<CToolchain.Feature> featureBuilder = ImmutableList.builder();
    try {
      if (!existingFeatureNames.contains(CppRuleClasses.STATIC_LINK_CPP_RUNTIMES)
          && supportsEmbeddedRuntimes) {
        featureBuilder.add(getFeature("name: 'static_link_cpp_runtimes' enabled: true"));
      }
      if (!existingFeatureNames.contains(CppRuleClasses.SUPPORTS_INTERFACE_SHARED_LIBRARIES)
          && supportsInterfaceSharedLibraries) {
        featureBuilder.add(getFeature("name: 'supports_interface_shared_libraries' enabled: true"));
      }
      if (!existingFeatureNames.contains(CppRuleClasses.LEGACY_COMPILE_FLAGS)) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'legacy_compile_flags'",
                        "  enabled: true",
                        "  flag_set {",
                        "    action: 'assemble'",
                        "    action: 'preprocess-assemble'",
                        "    action: 'linkstamp-compile'",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'c++-header-parsing'",
                        "    action: 'c++-module-compile'",
                        "    action: 'c++-module-codegen'",
                        "    action: 'lto-backend'",
                        "    action: 'clif-match'",
                        "    flag_group {",
                        "      expand_if_all_available: 'legacy_compile_flags'",
                        "      iterate_over: 'legacy_compile_flags'",
                        "      flag: '%{legacy_compile_flags}'",
                        "    }",
                        "  }")));
      }
      // Gcc options:
      //  -MD turns on .d file output as a side-effect (doesn't imply -E)
      //  -MM[D] enables user includes only, not system includes
      //  -MF <name> specifies the dotd file name
      // Issues:
      //  -M[M] alone subverts actual .o output (implies -E)
      //  -M[M]D alone breaks some of the .d naming assumptions
      // This combination gets user and system includes with specified name:
      //  -MD -MF <name>
      if (!existingFeatureNames.contains(CppRuleClasses.DEPENDENCY_FILE)) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'dependency_file'",
                        "  enabled: true",
                        "  flag_set {",
                        "    action: 'assemble'",
                        "    action: 'preprocess-assemble'",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'c++-module-compile'",
                        "    action: 'objc-compile'",
                        "    action: 'objc++-compile'",
                        "    action: 'c++-header-parsing'",
                        "    action: 'clif-match'",
                        "    flag_group {",
                        "      expand_if_all_available: 'dependency_file'",
                        "      flag: '-MD'",
                        "      flag: '-MF'",
                        "      flag: '%{dependency_file}'",
                        "    }",
                        "  }")));
      }
      // GCC and Clang give randomized names to symbols which are defined in
      // an anonymous namespace but have external linkage.  To make
      // computation of these deterministic, we want to override the
      // default seed for the random number generator.  It's safe to use
      // any value which differs for all translation units; we use the
      // path to the object file.
      if (!existingFeatureNames.contains(CppRuleClasses.RANDOM_SEED)) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'random_seed'",
                        "  enabled: true",
                        "  flag_set {",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'c++-module-codegen'",
                        "    action: 'c++-module-compile'",
                        "    flag_group {",
                        "      expand_if_all_available: 'output_file'",
                        "      flag: '-frandom-seed=%{output_file}'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains(CppRuleClasses.PIC)) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'pic'",
                        "  enabled: true",
                        "  flag_set {",
                        "    action: 'assemble'",
                        "    action: 'preprocess-assemble'",
                        "    action: 'linkstamp-compile'",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'c++-module-codegen'",
                        "    action: 'c++-module-compile'",
                        "    flag_group {",
                        "      expand_if_all_available: 'pic'",
                        "      flag: '-fPIC'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains(CppRuleClasses.PER_OBJECT_DEBUG_INFO)) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'per_object_debug_info'",
                        "  flag_set {",
                        "    action: 'assemble'",
                        "    action: 'preprocess-assemble'",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'c++-module-codegen'",
                        "    flag_group {",
                        "      expand_if_all_available: 'per_object_debug_info_file'",
                        "      flag: '-gsplit-dwarf'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains(CppRuleClasses.PREPROCESSOR_DEFINES)) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'preprocessor_defines'",
                        "  enabled: true",
                        "  flag_set {",
                        "    action: 'preprocess-assemble'",
                        "    action: 'linkstamp-compile'",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'c++-header-parsing'",
                        "    action: 'c++-module-compile'",
                        "    action: 'clif-match'",
                        "    flag_group {",
                        "      iterate_over: 'preprocessor_defines'",
                        "      flag: '-D%{preprocessor_defines}'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains("includes")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'includes'",
                        "  enabled: true",
                        "  flag_set {",
                        "    action: 'preprocess-assemble'",
                        "    action: 'linkstamp-compile'",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'c++-header-parsing'",
                        "    action: 'c++-module-compile'",
                        "    action: 'clif-match'",
                        "    action: 'objc-compile'",
                        "    action: 'objc++-compile'",
                        "    flag_group {",
                        "      expand_if_all_available: 'includes'",
                        "      iterate_over: 'includes'",
                        "      flag: '-include'",
                        "      flag: '%{includes}'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains(CppRuleClasses.INCLUDE_PATHS)) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'include_paths'",
                        "  enabled: true",
                        "  flag_set {",
                        "    action: 'preprocess-assemble'",
                        "    action: 'linkstamp-compile'",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'c++-header-parsing'",
                        "    action: 'c++-module-compile'",
                        "    action: 'clif-match'",
                        "    action: 'objc-compile'",
                        "    action: 'objc++-compile'",
                        "    flag_group {",
                        "      iterate_over: 'quote_include_paths'",
                        "      flag: '-iquote'",
                        "      flag: '%{quote_include_paths}'",
                        "    }",
                        "    flag_group {",
                        "      iterate_over: 'include_paths'",
                        "      flag: '-I%{include_paths}'",
                        "    }",
                        "    flag_group {",
                        "      iterate_over: 'system_include_paths'",
                        "      flag: '-isystem'",
                        "      flag: '%{system_include_paths}'",
                        "    }",
                        "    flag_group {",
                        "      iterate_over: 'framework_include_paths'",
                        "      flag: '-F%{framework_include_paths}'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains(CppRuleClasses.FDO_INSTRUMENT)) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'fdo_instrument'",
                        "  provides: 'profile'",
                        "  flag_set {",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'c++-link-executable'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    flag_group {",
                        "      expand_if_all_available: 'fdo_instrument_path'",
                        "      flag: '-fprofile-generate=%{fdo_instrument_path}'",
                        "      flag: '-fno-data-sections'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains(CppRuleClasses.FDO_OPTIMIZE)) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'fdo_optimize'",
                        "  provides: 'profile'",
                        "  flag_set {",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    flag_group {",
                        "      expand_if_all_available: 'fdo_profile_path'",
                        "      flag: '-fprofile-use=%{fdo_profile_path}'",
                        "      flag: '-Wno-profile-instr-unprofiled'",
                        "      flag: '-Wno-profile-instr-out-of-date'",
                        "      flag: '-fprofile-correction'",
                        "    }",
                        "  }")));
      }

      if (!existingFeatureNames.contains(CppRuleClasses.CS_FDO_INSTRUMENT)) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'cs_fdo_instrument'",
                        "  provides: 'csprofile'",
                        "  flag_set {",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'lto-backend'",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'c++-link-executable'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    flag_group {",
                        "      expand_if_all_available: 'cs_fdo_instrument_path'",
                        "      flag: '-fcs-profile-generate=%{cs_fdo_instrument_path}'",
                        "    }",
                        "  }")));
      }

      if (!existingFeatureNames.contains(CppRuleClasses.CS_FDO_OPTIMIZE)) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'cs_fdo_optimize'",
                        "  provides: 'csprofile'",
                        "  flag_set {",
                        "    action: 'lto-backend'",
                        "    flag_group {",
                        "      expand_if_all_available: 'fdo_profile_path'",
                        "      flag: '-fprofile-use=%{fdo_profile_path}'",
                        "      flag: '-Wno-profile-instr-unprofiled'",
                        "      flag: '-Wno-profile-instr-out-of-date'",
                        "      flag: '-fprofile-correction'",
                        "    }",
                        "  }")));
      }

      if (!existingFeatureNames.contains(CppRuleClasses.FDO_PREFETCH_HINTS)) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'fdo_prefetch_hints'",
                        "  flag_set {",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'lto-backend'",
                        "    flag_group {",
                        "      expand_if_all_available: 'fdo_prefetch_hints_path'",
                        "      flag: '-mllvm'",
                        "      flag: '-prefetch-hints-file=" + "%{fdo_prefetch_hints_path}'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains(CppRuleClasses.AUTOFDO)) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'autofdo'",
                        "  provides: 'profile'",
                        "  flag_set {",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    flag_group {",
                        "      expand_if_all_available: 'fdo_profile_path'",
                        "      flag: '-fauto-profile=%{fdo_profile_path}'",
                        "      flag: '-fprofile-correction'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains(CppRuleClasses.PROPELLER_OPTIMIZE)) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'propeller_optimize'",
                        "  flag_set {",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'lto-backend'",
                        "    flag_group {",
                        "      expand_if_all_available: 'propeller_optimize_cc_path'",
                        "      flag: '-fbasic-block-sections=list=%{propeller_optimize_cc_path}'",
                        "    }",
                        "  }",
                        "  flag_set {",
                        "    action: 'c++-link-executable'",
                        "    flag_group {",
                        "      expand_if_true: 'propeller_optimize_ld_path'",
                        "      flag: '-Wl,--symbol-ordering-file=%{propeller_optimize_ld_path}'",
                        "    }",
                        "  }")));
      }

      if (!existingFeatureNames.contains(CppRuleClasses.BUILD_INTERFACE_LIBRARIES)) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'build_interface_libraries'",
                        "  flag_set {",
                        "    with_feature { feature: 'supports_interface_shared_libraries' }",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    flag_group {",
                        "      expand_if_all_available: 'generate_interface_library'",
                        "      flag: '%{generate_interface_library}'",
                        "      flag: '%{interface_library_builder_path}'",
                        "      flag: '%{interface_library_input_path}'",
                        "      flag: '%{interface_library_output_path}'",
                        "    }",
                        "  }")));
      }

      // Order of feature declaration matters, cppDynamicLibraryLinkerTool has to
      // follow right after build_interface_libraries.
      if (!existingFeatureNames.contains("dynamic_library_linker_tool")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'dynamic_library_linker_tool'",
                        "  flag_set {",
                        "    with_feature { feature: 'supports_interface_shared_libraries' }",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    flag_group {",
                        "      expand_if_all_available: 'generate_interface_library'",
                        "      flag: '" + cppLinkDynamicLibraryToolPath + "'",
                        "    }",
                        "  }")));
      }

      if (!existingFeatureNames.contains("symbol_counts")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'symbol_counts'",
                        "  flag_set {",
                        "    action: 'c++-link-executable'",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    flag_group {",
                        "      expand_if_all_available: 'symbol_counts_output'",
                        "      flag: '-Wl,--print-symbol-counts=%{symbol_counts_output}'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains("shared_flag")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'shared_flag'",
                        "  flag_set {",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    flag_group {",
                        "      flag: '-shared'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains("linkstamps")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'linkstamps'",
                        "  flag_set {",
                        "    action: 'c++-link-executable'",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    flag_group {",
                        "      expand_if_all_available: 'linkstamp_paths'",
                        "      iterate_over: 'linkstamp_paths'",
                        "      flag: '%{linkstamp_paths}'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains("output_execpath_flags")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'output_execpath_flags'",
                        "  flag_set {",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'c++-link-executable'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    flag_group {",
                        "      expand_if_all_available: 'output_execpath'",
                        "      flag: '-o'",
                        "      flag: '%{output_execpath}'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains("runtime_library_search_directories")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'runtime_library_search_directories',",
                        "  flag_set {",
                        "    with_feature { feature: 'static_link_cpp_runtimes' }",
                        "    action: 'c++-link-executable'",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    flag_group {",
                        "      expand_if_all_available: 'runtime_library_search_directories'",
                        "      iterate_over: 'runtime_library_search_directories'",
                        "      flag_group {",
                        "        expand_if_true: 'is_cc_test'",
                        // TODO(b/27153401): This should probably be @loader_path on osx.
                        "        flag: ",
                        "          '-Wl,-rpath,$EXEC_ORIGIN/%{runtime_library_search_directories}'",
                        "      }",
                        "      flag_group {",
                        "        expand_if_false: 'is_cc_test'",
                        ifLinux(
                            platform,
                            "        flag: '-Wl,-rpath,$ORIGIN/"
                                + "%{runtime_library_search_directories}'"),
                        ifMac(
                            platform,
                            "        flag: '-Wl,-rpath,@loader_path/"
                                + "%{runtime_library_search_directories}'"),
                        "      }",
                        "    }",
                        "  }",
                        "  flag_set {",
                        "    with_feature { not_feature: 'static_link_cpp_runtimes' }",
                        "    action: 'c++-link-executable'",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    flag_group {",
                        "      expand_if_all_available: 'runtime_library_search_directories'",
                        "      iterate_over: 'runtime_library_search_directories'",
                        "      flag_group {",
                        ifLinux(
                            platform,
                            "        flag: '-Wl,-rpath,$ORIGIN/"
                                + "%{runtime_library_search_directories}'"),
                        ifMac(
                            platform,
                            "        flag: '-Wl,-rpath,@loader_path/"
                                + "%{runtime_library_search_directories}'"),
                        "    }",
                        "  }",
                        "}")));
      }
      if (!existingFeatureNames.contains("library_search_directories")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'library_search_directories'",
                        "  flag_set {",
                        "    action: 'c++-link-executable'",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    flag_group {",
                        "      expand_if_all_available: 'library_search_directories'",
                        "      iterate_over: 'library_search_directories'",
                        "      flag: '-L%{library_search_directories}'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains("archiver_flags")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'archiver_flags'",
                        "  flag_set {",
                        "    action: 'c++-link-static-library'",
                        "    flag_group {",
                        ifLinux(platform, "flag: 'rcsD'"),
                        ifMac(platform, "flag: '-static'", "flag: '-s'"),
                        "    }",
                        "    flag_group {",
                        "      expand_if_all_available: 'output_execpath'",
                        ifLinux(platform, "flag: '%{output_execpath}'"),
                        ifMac(platform, "flag: '-o'", "flag: '%{output_execpath}'"),
                        "    }",
                        "  }",
                        "  flag_set { ",
                        "    action: 'c++-link-static-library'",
                        "    flag_group {",
                        "      expand_if_all_available: 'libraries_to_link'",
                        "      iterate_over: 'libraries_to_link'",
                        "      flag_group {",
                        "        expand_if_equal {",
                        "          variable: 'libraries_to_link.type'",
                        "          value: 'object_file'",
                        "        }",
                        "        flag: '%{libraries_to_link.name}'",
                        "      }",
                        "      flag_group {",
                        "        expand_if_equal {",
                        "          variable: 'libraries_to_link.type'",
                        "          value: 'object_file_group'",
                        "        }",
                        "        iterate_over: 'libraries_to_link.object_files'",
                        "        flag: '%{libraries_to_link.object_files}'",
                        "      }",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains("libraries_to_link")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'libraries_to_link'",
                        "  flag_set {",
                        "    action: 'c++-link-executable'",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        ifTrue(
                            doNotSplitLinkingCmdline,
                            "    flag_group {",
                            "      expand_if_true: 'thinlto_param_file'",
                            "      flag: '-Wl,@%{thinlto_param_file}'",
                            "    }"),
                        "    flag_group {",
                        "      expand_if_all_available: 'libraries_to_link'",
                        "      iterate_over: 'libraries_to_link'",
                        "      flag_group {",
                        "        expand_if_equal: {",
                        "          variable: 'libraries_to_link.type'",
                        "          value: 'object_file_group'",
                        "        }",
                        "        expand_if_false: 'libraries_to_link.is_whole_archive'",
                        "        flag: '-Wl,--start-lib'",
                        "      }",
                        ifLinux(
                            platform,
                            "  flag_group {",
                            "    expand_if_true: 'libraries_to_link.is_whole_archive'",
                            "    expand_if_equal: {",
                            "        variable: 'libraries_to_link.type'",
                            "        value: 'static_library'",
                            "    }",
                            "    flag: '-Wl,-whole-archive'",
                            "  }",
                            "  flag_group {",
                            "    expand_if_equal: {",
                            "        variable: 'libraries_to_link.type'",
                            "        value: 'object_file_group'",
                            "    }",
                            "    iterate_over: 'libraries_to_link.object_files'",
                            "    flag: '%{libraries_to_link.object_files}'",
                            "  }",
                            "  flag_group {",
                            "    expand_if_equal: {",
                            "      variable: 'libraries_to_link.type'",
                            "      value: 'object_file'",
                            "    }",
                            "    flag: '%{libraries_to_link.name}'",
                            "  }",
                            "  flag_group {",
                            "    expand_if_equal: {",
                            "      variable: 'libraries_to_link.type'",
                            "      value: 'interface_library'",
                            "    }",
                            "    flag: '%{libraries_to_link.name}'",
                            "  }",
                            "  flag_group {",
                            "    expand_if_equal: {",
                            "      variable: 'libraries_to_link.type'",
                            "      value: 'static_library'",
                            "    }",
                            "    flag: '%{libraries_to_link.name}'",
                            "  }",
                            "  flag_group {",
                            "    expand_if_equal: {",
                            "      variable: 'libraries_to_link.type'",
                            "      value: 'dynamic_library'",
                            "    }",
                            "    flag: '-l%{libraries_to_link.name}'",
                            "  }",
                            "  flag_group {",
                            "    expand_if_equal: {",
                            "      variable: 'libraries_to_link.type'",
                            "      value: 'versioned_dynamic_library'",
                            "    }",
                            "    flag: '-l:%{libraries_to_link.name}'",
                            "  }",
                            "  flag_group {",
                            "    expand_if_true: 'libraries_to_link.is_whole_archive'",
                            "    expand_if_equal: {",
                            "        variable: 'libraries_to_link.type'",
                            "        value: 'static_library'",
                            "    }",
                            "    flag: '-Wl,-no-whole-archive'",
                            "  }"),
                        ifMac(
                            platform,
                            "  flag_group {",
                            "    expand_if_equal: {",
                            "      variable: 'libraries_to_link.type'",
                            "      value: 'object_file_group'",
                            "    }",
                            "    iterate_over: 'libraries_to_link.object_files'",
                            "    flag_group {",
                            "      expand_if_false: 'libraries_to_link.is_whole_archive'",
                            "      flag: '%{libraries_to_link.object_files}'",
                            "    }",
                            "    flag_group {",
                            "      expand_if_true: 'libraries_to_link.is_whole_archive'",
                            "      flag: '-Wl,-force_load,%{libraries_to_link.object_files}'",
                            "    }",
                            "  }",
                            "  flag_group {",
                            "    expand_if_equal: {",
                            "      variable: 'libraries_to_link.type'",
                            "      value: 'object_file'",
                            "    }",
                            "    flag_group {",
                            "      expand_if_false: 'libraries_to_link.is_whole_archive'",
                            "      flag: '%{libraries_to_link.name}'",
                            "    }",
                            "    flag_group {",
                            "      expand_if_true: 'libraries_to_link.is_whole_archive'",
                            "      flag: '-Wl,-force_load,%{libraries_to_link.name}'",
                            "    }",
                            "  }",
                            "  flag_group {",
                            "    expand_if_equal: {",
                            "      variable: 'libraries_to_link.type'",
                            "      value: 'interface_library'",
                            "    }",
                            "    flag_group {",
                            "      expand_if_false: 'libraries_to_link.is_whole_archive'",
                            "      flag: '%{libraries_to_link.name}'",
                            "    }",
                            "    flag_group {",
                            "      expand_if_true: 'libraries_to_link.is_whole_archive'",
                            "      flag: '-Wl,-force_load,%{libraries_to_link.name}'",
                            "    }",
                            "  }",
                            "  flag_group {",
                            "    expand_if_equal: {",
                            "      variable: 'libraries_to_link.type'",
                            "      value: 'static_library'",
                            "    }",
                            "    flag_group {",
                            "      expand_if_false: 'libraries_to_link.is_whole_archive'",
                            "      flag: '%{libraries_to_link.name}'",
                            "    }",
                            "    flag_group {",
                            "      expand_if_true: 'libraries_to_link.is_whole_archive'",
                            "      flag: '-Wl,-force_load,%{libraries_to_link.name}'",
                            "    }",
                            "  }",
                            "  flag_group {",
                            "    expand_if_equal: {",
                            "      variable: 'libraries_to_link.type'",
                            "      value: 'dynamic_library'",
                            "    }",
                            "    flag: '-l%{libraries_to_link.name}'",
                            "  }",
                            "  flag_group {",
                            "    expand_if_equal: {",
                            "      variable: 'libraries_to_link.type'",
                            "      value: 'versioned_dynamic_library'",
                            "    }",
                            "    flag: '-l:%{libraries_to_link.name}'",
                            "  }"),
                        "      flag_group {",
                        "        expand_if_equal: {",
                        "          variable: 'libraries_to_link.type'",
                        "          value: 'object_file_group'",
                        "        }",
                        "        expand_if_false: 'libraries_to_link.is_whole_archive'",
                        "        flag: '-Wl,--end-lib'",
                        "      }",
                        "    }",
                        ifTrue(
                            !doNotSplitLinkingCmdline,
                            "    flag_group {",
                            "      expand_if_true: 'thinlto_param_file'",
                            "      flag: '-Wl,@%{thinlto_param_file}'",
                            "    }"),
                        "  }")));
      }
      if (!existingFeatureNames.contains("force_pic_flags")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'force_pic_flags'",
                        "  flag_set {",
                        "    action: 'c++-link-executable'",
                        "    action: 'lto-index-for-executable'",
                        "    flag_group {",
                        "      expand_if_all_available: 'force_pic'",
                        ifLinux(platform, "flag: '-pie'"),
                        ifMac(platform, "flag: '-Wl,-pie'"),
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains("user_link_flags")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'user_link_flags'",
                        "  flag_set {",
                        "    action: 'c++-link-executable'",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    flag_group {",
                        "      expand_if_all_available: 'user_link_flags'",
                        "      iterate_over: 'user_link_flags'",
                        "      flag: '%{user_link_flags}'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains("legacy_link_flags")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'legacy_link_flags'",
                        "  flag_set {",
                        "    action: 'c++-link-executable'",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    flag_group {",
                        "      expand_if_all_available: 'legacy_link_flags'",
                        "      iterate_over: 'legacy_link_flags'",
                        "      flag: '%{legacy_link_flags}'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains("static_libgcc")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'static_libgcc'",
                        "  enabled: true",
                        "  flag_set {",
                        "    action: 'c++-link-executable'",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    with_feature { feature: 'static_link_cpp_runtimes' }",
                        "    flag_group {",
                        "      flag: '-static-libgcc'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains("fission_support")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'fission_support'",
                        "  flag_set {",
                        "    action: 'c++-link-executable'",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    flag_group {",
                        "      expand_if_all_available: 'is_using_fission'",
                        "      flag: '-Wl,--gdb-index'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains("strip_debug_symbols")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'strip_debug_symbols'",
                        "  flag_set {",
                        "    action: 'c++-link-executable'",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    flag_group {",
                        "      expand_if_all_available: 'strip_debug_symbols'",
                        "      flag: '-Wl,-S'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains(CppRuleClasses.COVERAGE)) {
        featureBuilder.add(
            getFeature("  name: 'coverage'"),
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'llvm_coverage_map_format'",
                        "  provides: 'profile'",
                        "  flag_set {",
                        "    action: 'preprocess-assemble'",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'c++-module-compile'",
                        "    action: 'objc-compile'",
                        "    action: 'objc++-compile'",
                        "    flag_group {",
                        "      flag: '-fprofile-instr-generate'",
                        "      flag: '-fcoverage-mapping'",
                        "    }",
                        "  }",
                        "  flag_set {",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'c++-link-executable'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    action: 'objc-executable'",
                        "    action: 'objc++-executable'",
                        "    flag_group {",
                        "      flag: '-fprofile-instr-generate'",
                        "    }",
                        "  }",
                        "  requires {",
                        "    feature: 'coverage'",
                        "  }")),
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'gcc_coverage_map_format'",
                        "  provides: 'profile'",
                        "  flag_set {",
                        "    action: 'preprocess-assemble'",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'c++-module-compile'",
                        "    action: 'objc-compile'",
                        "    action: 'objc++-compile'",
                        "    action: 'objc-executable'",
                        "    action: 'objc++-executable'",
                        "    flag_group {",
                        "      expand_if_all_available: 'gcov_gcno_file'",
                        "      flag: '-fprofile-arcs'",
                        "      flag: '-ftest-coverage'",
                        "    }",
                        "  }",
                        "  flag_set {",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'c++-link-executable'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    flag_group {",
                        "      flag: '--coverage'",
                        "    }",
                        "  }",
                        "  requires {",
                        "    feature: 'coverage'",
                        "  }")));
      }
    } catch (ParseException e) {
      // Can only happen if we change the proto definition without changing our
      // configuration above.
      throw new IllegalStateException(e);
    }
    return featureBuilder.build();
  }

  // Note:  these configs won't be added to the crosstools that defines no_legacy_features feature
  // (e.g. ndk, apple, enclave crosstools). Those need to be modified separately.
  public static ImmutableList<CToolchain.ActionConfig> getLegacyActionConfigs(
      CppPlatform platform,
      String gccToolPath,
      String arToolPath,
      String stripToolPath,
      boolean supportsInterfaceSharedLibraries,
      ImmutableSet<String> existingActionConfigNames) {
    try {
      ImmutableList.Builder<CToolchain.ActionConfig> actionConfigBuilder = ImmutableList.builder();
      if (!existingActionConfigNames.contains(CppActionNames.ASSEMBLE)) {
        actionConfigBuilder.add(
            getActionConfig(
                Joiner.on("\n")
                    .join(
                        "  config_name: 'assemble'",
                        "  action_name: 'assemble'",
                        "  tool {",
                        "    tool_path: '" + gccToolPath + "'",
                        "  }",
                        "  implies: 'legacy_compile_flags'",
                        "  implies: 'user_compile_flags'",
                        "  implies: 'sysroot'",
                        "  implies: 'unfiltered_compile_flags'",
                        "  implies: 'compiler_input_flags'",
                        "  implies: 'compiler_output_flags'")));
      }
      if (!existingActionConfigNames.contains(CppActionNames.PREPROCESS_ASSEMBLE)) {
        actionConfigBuilder.add(
            getActionConfig(
                Joiner.on("\n")
                    .join(
                        "  config_name: 'preprocess-assemble'",
                        "  action_name: 'preprocess-assemble'",
                        "  tool {",
                        "    tool_path: '" + gccToolPath + "'",
                        "  }",
                        "  implies: 'legacy_compile_flags'",
                        "  implies: 'user_compile_flags'",
                        "  implies: 'sysroot'",
                        "  implies: 'unfiltered_compile_flags'",
                        "  implies: 'compiler_input_flags'",
                        "  implies: 'compiler_output_flags'")));
      }
      if (!existingActionConfigNames.contains(CppActionNames.LINKSTAMP_COMPILE)) {
        actionConfigBuilder.add(
            getActionConfig(
                Joiner.on("\n")
                    .join(
                        "  config_name: 'linkstamp-compile'",
                        "  action_name: 'linkstamp-compile'",
                        "  tool {",
                        "    tool_path: '" + gccToolPath + "'",
                        "  }",
                        "  implies: 'legacy_compile_flags'",
                        "  implies: 'user_compile_flags'",
                        "  implies: 'sysroot'",
                        "  implies: 'unfiltered_compile_flags'",
                        "  implies: 'compiler_input_flags'",
                        "  implies: 'compiler_output_flags'")));
      }
      if (!existingActionConfigNames.contains(CppActionNames.LTO_BACKEND)) {
        actionConfigBuilder.add(
            getActionConfig(
                Joiner.on("\n")
                    .join(
                        "  config_name: 'lto-backend'",
                        "  action_name: 'lto-backend'",
                        "  tool {",
                        "    tool_path: '" + gccToolPath + "'",
                        "  }",
                        "  implies: 'legacy_compile_flags'",
                        "  implies: 'user_compile_flags'",
                        "  implies: 'sysroot'",
                        "  implies: 'unfiltered_compile_flags'",
                        "  implies: 'compiler_input_flags'",
                        "  implies: 'compiler_output_flags'")));
      }
      if (!existingActionConfigNames.contains(CppActionNames.C_COMPILE)) {
        actionConfigBuilder.add(
            getActionConfig(
                Joiner.on("\n")
                    .join(
                        "  config_name: 'c-compile'",
                        "  action_name: 'c-compile'",
                        "  tool {",
                        "    tool_path: '" + gccToolPath + "'",
                        "  }",
                        "  implies: 'legacy_compile_flags'",
                        "  implies: 'user_compile_flags'",
                        "  implies: 'sysroot'",
                        "  implies: 'unfiltered_compile_flags'",
                        "  implies: 'compiler_input_flags'",
                        "  implies: 'compiler_output_flags'")));
      }
      if (!existingActionConfigNames.contains(CppActionNames.CPP_COMPILE)) {
        actionConfigBuilder.add(
            getActionConfig(
                Joiner.on("\n")
                    .join(
                        "  config_name: 'c++-compile'",
                        "  action_name: 'c++-compile'",
                        "  tool {",
                        "    tool_path: '" + gccToolPath + "'",
                        "  }",
                        "  implies: 'legacy_compile_flags'",
                        "  implies: 'user_compile_flags'",
                        "  implies: 'sysroot'",
                        "  implies: 'unfiltered_compile_flags'",
                        "  implies: 'compiler_input_flags'",
                        "  implies: 'compiler_output_flags'")));
      }
      if (!existingActionConfigNames.contains(CppActionNames.CPP_HEADER_PARSING)) {
        actionConfigBuilder.add(
            getActionConfig(
                Joiner.on("\n")
                    .join(
                        "  config_name: 'c++-header-parsing'",
                        "  action_name: 'c++-header-parsing'",
                        "  tool {",
                        "    tool_path: '" + gccToolPath + "'",
                        "  }",
                        "  implies: 'legacy_compile_flags'",
                        "  implies: 'user_compile_flags'",
                        "  implies: 'sysroot'",
                        "  implies: 'unfiltered_compile_flags'",
                        "  implies: 'compiler_input_flags'",
                        "  implies: 'compiler_output_flags'")));
      }
      if (!existingActionConfigNames.contains(CppActionNames.CPP_MODULE_COMPILE)) {
        actionConfigBuilder.add(
            getActionConfig(
                Joiner.on("\n")
                    .join(
                        "  config_name: 'c++-module-compile'",
                        "  action_name: 'c++-module-compile'",
                        "  tool {",
                        "    tool_path: '" + gccToolPath + "'",
                        "  }",
                        "  implies: 'legacy_compile_flags'",
                        "  implies: 'user_compile_flags'",
                        "  implies: 'sysroot'",
                        "  implies: 'unfiltered_compile_flags'",
                        "  implies: 'compiler_input_flags'",
                        "  implies: 'compiler_output_flags'")));
      }
      if (!existingActionConfigNames.contains(CppActionNames.CPP_MODULE_CODEGEN)) {
        actionConfigBuilder.add(
            getActionConfig(
                Joiner.on("\n")
                    .join(
                        "  config_name: 'c++-module-codegen'",
                        "  action_name: 'c++-module-codegen'",
                        "  tool {",
                        "    tool_path: '" + gccToolPath + "'",
                        "  }",
                        "  implies: 'legacy_compile_flags'",
                        "  implies: 'user_compile_flags'",
                        "  implies: 'sysroot'",
                        "  implies: 'unfiltered_compile_flags'",
                        "  implies: 'compiler_input_flags'",
                        "  implies: 'compiler_output_flags'")));
      }
      if (!existingActionConfigNames.contains(CppActionNames.CPP_LINK_EXECUTABLE)) {
        actionConfigBuilder.add(
            getActionConfig(
                Joiner.on("\n")
                    .join(
                        "  config_name: 'c++-link-executable'",
                        "  action_name: 'c++-link-executable'",
                        "  tool {",
                        "    tool_path: '" + gccToolPath + "'",
                        "  }",
                        "  implies: 'symbol_counts'",
                        "  implies: 'strip_debug_symbols'",
                        "  implies: 'linkstamps'",
                        "  implies: 'output_execpath_flags'",
                        "  implies: 'runtime_library_search_directories'",
                        "  implies: 'library_search_directories'",
                        "  implies: 'libraries_to_link'",
                        "  implies: 'force_pic_flags'",
                        "  implies: 'user_link_flags'",
                        "  implies: 'legacy_link_flags'",
                        "  implies: 'linker_param_file'",
                        "  implies: 'fission_support'",
                        "  implies: 'sysroot'")));
      }
      if (!existingActionConfigNames.contains(CppActionNames.LTO_INDEX_EXECUTABLE)) {
        actionConfigBuilder.add(
            getActionConfig(
                Joiner.on("\n")
                    .join(
                        "  config_name: 'lto-index-for-executable'",
                        "  action_name: 'lto-index-for-executable'",
                        "  tool {",
                        "    tool_path: '" + gccToolPath + "'",
                        "  }",
                        "  implies: 'symbol_counts'",
                        "  implies: 'strip_debug_symbols'",
                        "  implies: 'linkstamps'",
                        "  implies: 'output_execpath_flags'",
                        "  implies: 'runtime_library_search_directories'",
                        "  implies: 'library_search_directories'",
                        "  implies: 'libraries_to_link'",
                        "  implies: 'force_pic_flags'",
                        "  implies: 'user_link_flags'",
                        "  implies: 'legacy_link_flags'",
                        "  implies: 'linker_param_file'",
                        "  implies: 'fission_support'",
                        "  implies: 'sysroot'")));
      }
      if (!existingActionConfigNames.contains(CppActionNames.CPP_LINK_NODEPS_DYNAMIC_LIBRARY)) {
        actionConfigBuilder.add(
            getActionConfig(
                Joiner.on("\n")
                    .join(
                        "  config_name: 'c++-link-nodeps-dynamic-library'",
                        "  action_name: 'c++-link-nodeps-dynamic-library'",
                        "  tool {",
                        "    tool_path: '" + gccToolPath + "'",
                        "  }",
                        "  implies: 'build_interface_libraries'",
                        "  implies: 'dynamic_library_linker_tool'",
                        "  implies: 'symbol_counts'",
                        "  implies: 'strip_debug_symbols'",
                        "  implies: 'shared_flag'",
                        "  implies: 'linkstamps'",
                        "  implies: 'output_execpath_flags'",
                        "  implies: 'runtime_library_search_directories'",
                        "  implies: 'library_search_directories'",
                        "  implies: 'libraries_to_link'",
                        "  implies: 'user_link_flags'",
                        "  implies: 'legacy_link_flags'",
                        "  implies: 'linker_param_file'",
                        "  implies: 'fission_support'",
                        "  implies: 'sysroot'")));
      }
      if (!existingActionConfigNames.contains(CppActionNames.LTO_INDEX_NODEPS_DYNAMIC_LIBRARY)) {
        actionConfigBuilder.add(
            getActionConfig(
                Joiner.on("\n")
                    .join(
                        "  config_name: 'lto-index-for-nodeps-dynamic-library'",
                        "  action_name: 'lto-index-for-nodeps-dynamic-library'",
                        "  tool {",
                        "    tool_path: '" + gccToolPath + "'",
                        "  }",
                        "  implies: 'build_interface_libraries'",
                        "  implies: 'dynamic_library_linker_tool'",
                        "  implies: 'symbol_counts'",
                        "  implies: 'strip_debug_symbols'",
                        "  implies: 'shared_flag'",
                        "  implies: 'linkstamps'",
                        "  implies: 'output_execpath_flags'",
                        "  implies: 'runtime_library_search_directories'",
                        "  implies: 'library_search_directories'",
                        "  implies: 'libraries_to_link'",
                        "  implies: 'user_link_flags'",
                        "  implies: 'legacy_link_flags'",
                        "  implies: 'linker_param_file'",
                        "  implies: 'fission_support'",
                        "  implies: 'sysroot'")));
      }
      if (!existingActionConfigNames.contains(CppActionNames.CPP_LINK_DYNAMIC_LIBRARY)) {
        actionConfigBuilder.add(
            getActionConfig(
                Joiner.on("\n")
                    .join(
                        "  config_name: 'c++-link-dynamic-library'",
                        "  action_name: 'c++-link-dynamic-library'",
                        "  tool {",
                        "    tool_path: '" + gccToolPath + "'",
                        "  }",
                        "  implies: 'build_interface_libraries'",
                        "  implies: 'dynamic_library_linker_tool'",
                        "  implies: 'symbol_counts'",
                        "  implies: 'strip_debug_symbols'",
                        "  implies: 'shared_flag'",
                        "  implies: 'linkstamps'",
                        "  implies: 'output_execpath_flags'",
                        "  implies: 'runtime_library_search_directories'",
                        "  implies: 'library_search_directories'",
                        "  implies: 'libraries_to_link'",
                        "  implies: 'user_link_flags'",
                        "  implies: 'legacy_link_flags'",
                        "  implies: 'linker_param_file'",
                        "  implies: 'fission_support'",
                        "  implies: 'sysroot'")));
      }
      if (!existingActionConfigNames.contains(CppActionNames.LTO_INDEX_DYNAMIC_LIBRARY)) {
        actionConfigBuilder.add(
            getActionConfig(
                Joiner.on("\n")
                    .join(
                        "  config_name: 'lto-index-for-dynamic-library'",
                        "  action_name: 'lto-index-for-dynamic-library'",
                        "  tool {",
                        "    tool_path: '" + gccToolPath + "'",
                        "  }",
                        "  implies: 'build_interface_libraries'",
                        "  implies: 'dynamic_library_linker_tool'",
                        "  implies: 'symbol_counts'",
                        "  implies: 'strip_debug_symbols'",
                        "  implies: 'shared_flag'",
                        "  implies: 'linkstamps'",
                        "  implies: 'output_execpath_flags'",
                        "  implies: 'runtime_library_search_directories'",
                        "  implies: 'library_search_directories'",
                        "  implies: 'libraries_to_link'",
                        "  implies: 'user_link_flags'",
                        "  implies: 'legacy_link_flags'",
                        "  implies: 'linker_param_file'",
                        "  implies: 'fission_support'",
                        "  implies: 'sysroot'")));
      }
      if (!existingActionConfigNames.contains(CppActionNames.CPP_LINK_STATIC_LIBRARY)) {
        actionConfigBuilder.add(
            getActionConfig(
                Joiner.on("\n")
                    .join(
                        "  config_name: 'c++-link-static-library'",
                        "  action_name: 'c++-link-static-library'",
                        "  tool {",
                        "    tool_path: '" + arToolPath + "'",
                        "  }",
                        "  implies: 'archiver_flags'",
                        "  implies: 'linker_param_file'")));
      }
      if (!existingActionConfigNames.contains(CppActionNames.STRIP)) {
        actionConfigBuilder.add(
            getActionConfig(
                Joiner.on("\n")
                    .join(
                        "  config_name: 'strip'",
                        "  action_name: 'strip'",
                        "  tool {",
                        "    tool_path: '" + stripToolPath + "'",
                        "  }",
                        "  flag_set {",
                        "    flag_group {",
                        "      flag: '-S'",
                        ifLinux(platform, "flag: '-p'"),
                        "      flag: '-o'",
                        "      flag: '%{output_file}'",
                        "    }",
                        "    flag_group {",
                        "      iterate_over: 'stripopts'",
                        "      flag: '%{stripopts}'",
                        "    }",
                        "    flag_group {",
                        "      flag: '%{input_file}'",
                        "    }",
                        "  }")));
      }
      return actionConfigBuilder.build();
    } catch (ParseException e) {
      // Can only happen if we change the proto definition without changing our
      // configuration above.
      throw new IllegalStateException(e);
    }
  }

  // Note:  these feaures won't be added to the crosstools that defines no_legacy_features feature
  // (e.g. ndk, apple, enclave crosstools). Those need to be modified separately.
  public static ImmutableList<CToolchain.Feature> getFeaturesToAppearLastInFeaturesList(
      ImmutableSet<String> existingFeatureNames, boolean doNotSplitLinkingCmdline) {
    ImmutableList.Builder<CToolchain.Feature> featureBuilder = ImmutableList.builder();
    try {
      if (!existingFeatureNames.contains("fully_static_link")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'fully_static_link'",
                        "  flag_set {",
                        "    action: 'c++-link-executable'",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    flag_group {",
                        "      flag: '-static'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains("user_compile_flags")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'user_compile_flags'",
                        "  enabled: true",
                        "  flag_set {",
                        "    action: 'assemble'",
                        "    action: 'preprocess-assemble'",
                        "    action: 'linkstamp-compile'",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'c++-header-parsing'",
                        "    action: 'c++-module-compile'",
                        "    action: 'c++-module-codegen'",
                        "    action: 'lto-backend'",
                        "    action: 'clif-match'",
                        "    flag_group {",
                        "      expand_if_all_available: 'user_compile_flags'",
                        "      iterate_over: 'user_compile_flags'",
                        "      flag: '%{user_compile_flags}'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains("sysroot")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'sysroot'",
                        "  enabled: true",
                        "  flag_set {",
                        "    action: 'preprocess-assemble'",
                        "    action: 'linkstamp-compile'",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'c++-header-parsing'",
                        "    action: 'c++-module-compile'",
                        "    action: 'c++-link-executable'",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    action: 'clif-match'",
                        "    action: 'lto-backend'",
                        "    flag_group {",
                        "      expand_if_all_available: 'sysroot'",
                        "      flag: '--sysroot=%{sysroot}'",
                        "    }",
                        "  }")));
      }
      // unfiltered_compile_flags contain system include paths. These must be added
      // after the user provided options (present in legacy_compile_flags build
      // variable above), otherwise users adding include paths will not pick up their own
      // include paths first.
      if (!existingFeatureNames.contains("unfiltered_compile_flags")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'unfiltered_compile_flags'",
                        "  enabled: true",
                        "  flag_set {",
                        "    action: 'assemble'",
                        "    action: 'preprocess-assemble'",
                        "    action: 'linkstamp-compile'",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'c++-header-parsing'",
                        "    action: 'c++-module-compile'",
                        "    action: 'c++-module-codegen'",
                        "    action: 'lto-backend'",
                        "    action: 'clif-match'",
                        "    flag_group {",
                        "      expand_if_all_available: 'unfiltered_compile_flags'",
                        "      iterate_over: 'unfiltered_compile_flags'",
                        "      flag: '%{unfiltered_compile_flags}'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains("linker_param_file")) {
        String dynamicLibraryParamFile = "      flag: '-Wl,@%{linker_param_file}'";
        if (doNotSplitLinkingCmdline
            || existingFeatureNames.contains(CppRuleClasses.DO_NOT_SPLIT_LINKING_CMDLINE)) {
          dynamicLibraryParamFile = "      flag: '@%{linker_param_file}'";
        }
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'linker_param_file'",
                        "  flag_set {",
                        "    action: 'c++-link-executable'",
                        "    action: 'c++-link-dynamic-library'",
                        "    action: 'c++-link-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-dynamic-library'",
                        "    action: 'lto-index-for-nodeps-dynamic-library'",
                        "    action: 'lto-index-for-executable'",
                        "    flag_group {",
                        "      expand_if_all_available: 'linker_param_file'",
                        dynamicLibraryParamFile,
                        "    }",
                        "  }",
                        "  flag_set {",
                        "    action: 'c++-link-static-library'",
                        "    flag_group {",
                        "      expand_if_all_available: 'linker_param_file'",
                        "      flag: '@%{linker_param_file}'",
                        "    }",
                        "  }")));
      }
      if (!existingFeatureNames.contains("compiler_input_flags")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'compiler_input_flags'",
                        "  enabled: true",
                        "  flag_set {",
                        "    action: 'assemble'",
                        "    action: 'preprocess-assemble'",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'linkstamp-compile'",
                        "    action: 'c++-module-compile'",
                        "    action: 'c++-module-codegen'",
                        "    action: 'objc-compile'",
                        "    action: 'objc++-compile'",
                        "    action: 'c++-header-parsing'",
                        "    action: 'lto-backend'",
                        "    flag_group {",
                        "      expand_if_all_available: 'source_file'",
                        "      flag: '-c'",
                        "      flag: '%{source_file}'",
                        "    }",
                        "  }")));
      }

      if (!existingFeatureNames.contains("compiler_output_flags")) {
        featureBuilder.add(
            getFeature(
                Joiner.on("\n")
                    .join(
                        "  name: 'compiler_output_flags'",
                        "  enabled: true",
                        "  flag_set {",
                        "    action: 'assemble'",
                        "    action: 'preprocess-assemble'",
                        "    action: 'c-compile'",
                        "    action: 'c++-compile'",
                        "    action: 'linkstamp-compile'",
                        "    action: 'c++-module-compile'",
                        "    action: 'c++-module-codegen'",
                        "    action: 'objc-compile'",
                        "    action: 'objc++-compile'",
                        "    action: 'c++-header-parsing'",
                        "    action: 'lto-backend'",
                        "    flag_group {",
                        "      expand_if_all_available: 'output_assembly_file'",
                        "      flag: '-S'",
                        "    }",
                        "    flag_group {",
                        "      expand_if_all_available: 'output_preprocess_file'",
                        "      flag: '-E'",
                        "    }",
                        "    flag_group {",
                        "      expand_if_all_available: 'output_file'",
                        "      flag: '-o'",
                        "      flag: '%{output_file}'",
                        "    }",
                        "  }")));
      }
    } catch (ParseException e) {
      // Can only happen if we change the proto definition without changing our
      // configuration above.
      throw new IllegalStateException(e);
    }
    return featureBuilder.build();
  }

  private static String ifLinux(CppPlatform platform, String... lines) {
    // Platform `LINUX` also includes FreeBSD and OpenBSD.
    return ifTrue(platform == CppPlatform.LINUX, lines);
  }

  private static String ifMac(CppPlatform platform, String... lines) {
    return ifTrue(platform == CppPlatform.MAC, lines);
  }

  private static String ifTrue(boolean condition, String... lines) {
    if (condition) {
      return Joiner.on("\n").join(lines);
    } else {
      return "";
    }
  }

  private static CToolchain.Feature getFeature(String protoText) throws ParseException {
    CToolchain.Feature.Builder feature = CToolchain.Feature.newBuilder();
    TextFormat.merge(protoText, feature);
    return feature.build();
  }

  private static CToolchain.ActionConfig getActionConfig(String protoText) throws ParseException {
    CToolchain.ActionConfig.Builder actionConfig = CToolchain.ActionConfig.newBuilder();
    TextFormat.merge(protoText, actionConfig);
    return actionConfig.build();
  }
}
