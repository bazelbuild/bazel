// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages.util;

import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.rules.cpp.CppActionNames;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.protobuf.TextFormat;
import java.io.IOException;

/**
 * Creates mock BUILD files required for the C/C++ rules.
 */
public abstract class MockCcSupport {

  /** Filter to remove implicit crosstool artifact and module map inputs of C/C++ rules. */
  public static final Predicate<Artifact> CC_ARTIFACT_FILTER =
      new Predicate<Artifact>() {
        @Override
        public boolean apply(Artifact artifact) {
          String basename = artifact.getExecPath().getBaseName();
          String pathString = artifact.getExecPathString();
          return !pathString.startsWith("third_party/crosstool/")
              && !pathString.startsWith("tools/cpp/link_dynamic_library")
              && !pathString.startsWith("tools/cpp/build_interface_so")
              && !(pathString.contains("/internal/_middlemen") && basename.contains("crosstool"))
              && !pathString.startsWith("_bin/build_interface_so")
              && !pathString.endsWith(".cppmap")
              && !pathString.startsWith("tools/cpp/grep-includes");
        }
      };

  /** This feature will prevent bazel from patching the crosstool. */
  public static final String NO_LEGACY_FEATURES_FEATURE = "feature { name: 'no_legacy_features' }";

  public static final String DYNAMIC_LINKING_MODE_FEATURE =
      "feature { name: '" + CppRuleClasses.DYNAMIC_LINKING_MODE + "'}";

  public static final String DO_NOT_SPLIT_LINKING_CMDLINE_FEATURE =
      "feature { name: '" + CppRuleClasses.DO_NOT_SPLIT_LINKING_CMDLINE + "' enabled: true}";

  public static final String SUPPORTS_DYNAMIC_LINKER_FEATURE =
      "feature { name: '" + CppRuleClasses.SUPPORTS_DYNAMIC_LINKER + "' enabled: true}";

  public static final String SUPPORTS_INTERFACE_SHARED_LIBRARIES_FEATURE =
      "feature { name: '" + CppRuleClasses.SUPPORTS_INTERFACE_SHARED_LIBRARIES + "' enabled: true}";


  /** Feature expected by the C++ rules when pic build is requested */
  public static final String PIC_FEATURE =
      ""
          + "feature {"
          + "  name: 'pic'"
          + "  enabled: true"
          + "  flag_set {"
          + "    action: 'assemble'"
          + "    action: 'preprocess-assemble'"
          + "    action: 'linkstamp-compile'"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-module-codegen'"
          + "    action: 'c++-module-compile'"
          + "    flag_group {"
          + "      expand_if_all_available: 'pic'"
          + "      flag: '-fPIC'"
          + "    }"
          + "  }"
          + "}";

  /** A feature configuration snippet useful for testing header processing. */
  public static final String PARSE_HEADERS_FEATURE_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'parse_headers'"
          + "  flag_set {"
          + "    action: 'c++-header-parsing'"
          + "    flag_group {"
          + "      flag: '<c++-header-parsing>'"
          + "    }"
          + "  }"
          + "}";

  /** A feature configuration snippet useful for testing the layering check. */
  public static final String LAYERING_CHECK_FEATURE_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'layering_check'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-module-compile'"
          + "    flag_group {"
          + "      iterate_over: 'dependent_module_map_files'"
          + "      flag: 'dependent_module_map_file:%{dependent_module_map_files}'"
          + "    }"
          + "  }"
          + "}";


  /** A feature configuration snippet useful for testing header modules. */
  public static final String HEADER_MODULES_FEATURE_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'header_modules'"
          + "  implies: 'use_header_modules'"
          + "  implies: 'header_module_compile'"
          + "}"
          + "feature {"
          + "  name: 'header_module_compile'"
          + "  enabled: true"
          + "  implies: 'module_maps'"
          + "  flag_set {"
          + "    action: 'c++-module-compile'"
          + "    flag_group {"
          + "      flag: '--woohoo_modules'"
          + "    }"
          + "  }"
          + "  flag_set {"
          + "    action: 'c++-module-codegen'"
          + "    flag_group {"
          + "      flag: '--this_is_modules_codegen'"
          + "    }"
          + "  }"
          + "}"
          + "feature {"
          + "  name: 'header_module_codegen'"
          + "  implies: 'header_modules'"
          + "}"
          + "feature {"
          + "  name: 'module_maps'"
          + "  enabled: true"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-module-compile'"
          + "    flag_group {"
          + "      flag: 'module_name:%{module_name}'"
          + "      flag: 'module_map_file:%{module_map_file}'"
          + "    }"
          + "  }"
          + "}"
          + "feature {"
          + "  name: 'use_header_modules'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-modules-compile'"
          + "    flag_group {"
          + "      iterate_over: 'module_files'"
          + "      flag: 'module_file:%{module_files}'"
          + "    }"
          + "  }"
          + "}";

  public static final String MODULE_MAP_HOME_CWD_FEATURE =
      ""
          + "feature {"
          + "  name: 'module_map_home_cwd'"
          + "  enabled: true"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-module-compile'"
          + "    action: 'preprocess-assemble'"
          + "    flag_group {"
          + "      flag: '<flag>'"
          + "    }"
          + "  }"
          + "}";

  /** A feature configuration snippet useful for testing environment variables. */
  public static final String ENV_VAR_FEATURE_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'env_feature'"
          + "  implies: 'static_env_feature'"
          + "  implies: 'module_maps'"
          + "}"
          + "feature {"
          + "  name: 'static_env_feature'"
          + "  env_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-module-compile'"
          + "    env_entry {"
          + "      key: 'cat'"
          + "      value: 'meow'"
          + "    }"
          + "  }"
          + "}"
          + "feature {"
          + "  name: 'module_maps'"
          + "  enabled: true"
          + "  env_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-module-compile'"
          + "    env_entry {"
          + "      key: 'module'"
          + "      value: 'module_name:%{module_name}'"
          + "    }"
          + "  }"
          + "}";

  public static final String HOST_AND_NONHOST_CONFIGURATION =
      ""
          + "feature { "
          + "  name: 'host'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    flag_group {"
          + "      flag: '-host'"
          + "    }"
          + "  }"
          + "}"
          + "feature { "
          + "  name: 'nonhost'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    flag_group {"
          + "      flag: '-nonhost'"
          + "    }"
          + "  }"
          + "}";

  public static final String USER_COMPILE_FLAGS_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'user_compile_flags'"
          + "  enabled: true"
          + "  flag_set {"
          + "    action: 'assemble'"
          + "    action: 'preprocess-assemble'"
          + "    action: 'linkstamp-compile'"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-module-compile'"
          + "    action: 'c++-module-codegen'"
          + "    action: 'lto-backend'"
          + "    action: 'clif-match'"
          + "    flag_group {"
          + "      flag: '%{user_compile_flags}'"
          + "      iterate_over: 'user_compile_flags'"
          + "      expand_if_all_available: 'user_compile_flags'"
          + "    }"
          + "  }"
          + "}";

  public static final String LEGACY_COMPILE_FLAGS_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'legacy_compile_flags'"
          + "  enabled: true"
          + "  flag_set {"
          + "    action: 'assemble'"
          + "    action: 'preprocess-assemble'"
          + "    action: 'linkstamp-compile'"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-module-compile'"
          + "    action: 'c++-module-codegen'"
          + "    action: 'lto-backend'"
          + "    action: 'clif-match'"
          + "    flag_group {"
          + "      flag: '%{legacy_compile_flags}'"
          + "      iterate_over: 'legacy_compile_flags'"
          + "      expand_if_all_available: 'legacy_compile_flags'"
          + "    }"
          + "  }"
          + "}"
          + "compiler_flag: 'legacy_compile_flag'";

  public static final String THIN_LTO_CONFIGURATION =
      ""
          + "feature { "
          + "  name: 'thin_lto'"
          + "  requires { feature: 'nonhost' }"
          + "  flag_set {"
          + "    action: 'c++-link-executable'"
          + "    action: 'c++-link-dynamic-library'"
          + "    action: 'c++-link-nodeps-dynamic-library'"
          + "    action: 'c++-link-static-library'"
          + "    flag_group {"
          + "      expand_if_all_available: 'thinlto_param_file'"
          + "      flag: 'thinlto_param_file=%{thinlto_param_file}'"
          + "    }"
          + "  }"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    flag_group {"
          + "      flag: '-flto=thin'"
          + "    }"
          + "    flag_group {"
          + "      expand_if_all_available: 'lto_indexing_bitcode_file'"
          + "      flag: 'lto_indexing_bitcode=%{lto_indexing_bitcode_file}'"
          + "    }"
          + "  }"
          + "  flag_set {"
          + "    action: 'lto-indexing'"
          + "    flag_group {"
          + "      flag: 'param_file=%{thinlto_indexing_param_file}'"
          + "      flag: 'prefix_replace=%{thinlto_prefix_replace}'"
          + "    }"
          + "    flag_group {"
          + "      expand_if_all_available: 'thinlto_object_suffix_replace'"
          + "      flag: 'object_suffix_replace=%{thinlto_object_suffix_replace}'"
          + "    }"
          + "    flag_group {"
          + "      expand_if_all_available: 'thinlto_merged_object_file'"
          + "      flag: 'thinlto_merged_object_file=%{thinlto_merged_object_file}'"
          + "    }"
          + "  }"
          + "  flag_set {"
          + "    action: 'lto-backend'"
          + "    flag_group {"
          + "      flag: 'thinlto_index=%{thinlto_index}'"
          + "      flag: 'thinlto_output_object_file=%{thinlto_output_object_file}'"
          + "      flag: 'thinlto_input_bitcode_file=%{thinlto_input_bitcode_file}'"
          + "    }"
          + "  }"
          + "}";

  public static final String THIN_LTO_LINKSTATIC_TESTS_USE_SHARED_NONLTO_BACKENDS_CONFIGURATION =
      "" + "feature {  name: 'thin_lto_linkstatic_tests_use_shared_nonlto_backends'}";

  public static final String THIN_LTO_ALL_LINKSTATIC_USE_SHARED_NONLTO_BACKENDS_CONFIGURATION =
      "" + "feature {  name: 'thin_lto_all_linkstatic_use_shared_nonlto_backends'}";

  public static final String ENABLE_AFDO_THINLTO_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'enable_afdo_thinlto'"
          + "  requires { feature: 'autofdo_implicit_thinlto' }"
          + "  implies: 'thin_lto'"
          + "}";

  public static final String AUTOFDO_IMPLICIT_THINLTO_CONFIGURATION =
      "" + "feature {  name: 'autofdo_implicit_thinlto'}";

  public static final String ENABLE_FDO_THINLTO_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'enable_fdo_thinlto'"
          + "  requires { feature: 'fdo_implicit_thinlto' }"
          + "  implies: 'thin_lto'"
          + "}";

  public static final String FDO_IMPLICIT_THINLTO_CONFIGURATION =
      "" + "feature {  name: 'fdo_implicit_thinlto'}";

  public static final String ENABLE_XFDO_THINLTO_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'enable_xbinaryfdo_thinlto'"
          + "  requires { feature: 'xbinaryfdo_implicit_thinlto' }"
          + "  implies: 'thin_lto'"
          + "}";

  public static final String XFDO_IMPLICIT_THINLTO_CONFIGURATION =
      "" + "feature {  name: 'xbinaryfdo_implicit_thinlto'}";

  public static final String AUTO_FDO_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'autofdo'"
          + "  provides: 'profile'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'lto-backend'"
          + "    flag_group {"
          + "      expand_if_all_available: 'fdo_profile_path'"
          + "      flag: '-fauto-profile=%{fdo_profile_path}'"
          + "      flag: '-fprofile-correction'"
          + "    }"
          + "  }"
          + "}";

  public static final String IS_CC_FAKE_BINARY_CONFIGURATION =
      "feature { name: 'is_cc_fake_binary' }";

  public static final String XBINARY_FDO_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'xbinaryfdo'"
          + "  provides: 'profile'"
          + "  flag_set {"
          + "    with_feature { not_feature: 'is_cc_fake_binary' }"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'lto-backend'"
          + "    flag_group {"
          + "      expand_if_all_available: 'fdo_profile_path'"
          + "      flag: '-fauto-profile=%{fdo_profile_path}'"
          + "      flag: '-fprofile-correction'"
          + "    }"
          + "  }"
          + "}";

  public static final String FDO_OPTIMIZE_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'fdo_optimize'"
          + "  provides: 'profile'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    flag_group {"
          + "      expand_if_all_available: 'fdo_profile_path'"
          + "      flag: '-fprofile-use=%{fdo_profile_path}'"
          + "      flag: '-Xclang-only=-Wno-profile-instr-unprofiled'"
          + "      flag: '-Xclang-only=-Wno-profile-instr-out-of-date'"
          + "      flag: '-Xclang-only=-Wno-backend-plugin'"
          + "      flag: '-fprofile-correction'"
          + "    }"
          + "  }"
          + "}";

  public static final String FDO_INSTRUMENT_CONFIGURATION =
      ""
          + "feature { "
          + "  name: 'fdo_instrument'"
          + "  provides: 'profile'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-link-dynamic-library'"
          + "    action: 'c++-link-nodeps-dynamic-library'"
          + "    action: 'c++-link-executable'"
          + "    flag_group {"
          + "      flag: 'fdo_instrument_option'"
          + "      flag: 'path=%{fdo_instrument_path}'"
          + "    }"
          + "  }"
          + "}";

  public static final String PER_OBJECT_DEBUG_INFO_CONFIGURATION =
      ""
          + "feature { "
          + "  name: 'per_object_debug_info'"
          + "  enabled: true"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'assemble'"
          + "    action: 'preprocess-assemble'"
          + "    action: 'c++-module-codegen'"
          + "    action: 'lto-backend'"
          + "    flag_group {"
          + "      expand_if_all_available: 'per_object_debug_info_file'"
          + "      flag: 'per_object_debug_info_option'"
          + "    }"
          + "  }"
          + "}";

  public static final String COPY_DYNAMIC_LIBRARIES_TO_BINARY_CONFIGURATION =
      "" + "feature { name: 'copy_dynamic_libraries_to_binary' }";

  public static final String SUPPORTS_START_END_LIB_FEATURE =
      "" + "feature { name: 'supports_start_end_lib' enabled: true }";

  public static final String SUPPORTS_PIC_FEATURE =
      "" + "feature { name: 'supports_pic' enabled: true }";

  public static final String TARGETS_WINDOWS_CONFIGURATION =
      ""
          + "feature {"
          + "   name: 'targets_windows'"
          + "   implies: 'copy_dynamic_libraries_to_binary'"
          + "   enabled: true"
          + "}";

  public static final String STATIC_LINK_TWEAKED_CONFIGURATION =
      ""
          + "artifact_name_pattern {"
          + "   category_name: 'static_library'"
          + "   prefix: 'lib'"
          + "   extension: '.lib'"
          + "}";

  public static final String STATIC_LINK_AS_DOT_A_CONFIGURATION =
      ""
          + "artifact_name_pattern {"
          + "   category_name: 'static_library'"
          + "   prefix: 'lib'"
          + "   extension: '.a'"
          + "}";

  public static final String MODULE_MAPS_FEATURE =
      ""
          + "feature {"
          + "  name: 'module_maps'"
          + "  enabled: true"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-module-compile'"
          + "    flag_group {"
          + "      flag: 'module_name:%{module_name}'"
          + "      flag: 'module_map_file:%{module_map_file}'"
          + "    }"
          + "  }"
          + "}";

  public static final String EMPTY_COMPILE_ACTION_CONFIG =
      emptyActionConfigFor(CppActionNames.CPP_COMPILE);

  public static final String EMPTY_MODULE_CODEGEN_ACTION_CONFIG =
      emptyActionConfigFor(CppActionNames.CPP_MODULE_CODEGEN);

  public static final String EMPTY_MODULE_COMPILE_ACTION_CONFIG =
      emptyActionConfigFor(CppActionNames.CPP_MODULE_COMPILE);

  public static final String EMPTY_EXECUTABLE_ACTION_CONFIG =
      emptyActionConfigFor(LinkTargetType.EXECUTABLE.getActionName());

  public static final String EMPTY_DYNAMIC_LIBRARY_ACTION_CONFIG =
      emptyActionConfigFor(LinkTargetType.NODEPS_DYNAMIC_LIBRARY.getActionName());

  public static final String EMPTY_TRANSITIVE_DYNAMIC_LIBRARY_ACTION_CONFIG =
      emptyActionConfigFor(LinkTargetType.DYNAMIC_LIBRARY.getActionName());

  public static final String EMPTY_STATIC_LIBRARY_ACTION_CONFIG =
      emptyActionConfigFor(LinkTargetType.STATIC_LIBRARY.getActionName());

  public static final String EMPTY_CLIF_MATCH_ACTION_CONFIG =
      emptyActionConfigFor(CppActionNames.CLIF_MATCH);

  public static final String EMPTY_STRIP_ACTION_CONFIG = emptyActionConfigFor(CppActionNames.STRIP);

  public static final String STATIC_LINK_CPP_RUNTIMES_FEATURE =
      "feature { name: 'static_link_cpp_runtimes' enabled: true }";

  public static final String EMPTY_CROSSTOOL =
      "major_version: 'foo'\nminor_version:' foo'\n" + emptyToolchainForCpu("k8");

  public static final String SIMPLE_COMPILE_FEATURE = "simple_compile_feature";

  public static String emptyToolchainForCpu(String cpu, String... append) {
    return Joiner.on("\n")
        .join(
            ImmutableList.builder()
                .add(
                    "toolchain {",
                    "  toolchain_identifier: 'mock-llvm-toolchain-" + cpu + "'",
                    "  host_system_name: 'mock-system-name-for-" + cpu + "'",
                    "  target_system_name: 'mock-target-system-name-for-" + cpu + "'",
                    "  target_cpu: '" + cpu + "'",
                    "  target_libc: 'mock-libc-for-" + cpu + "'",
                    "  compiler: 'mock-compiler-for-" + cpu + "'",
                    "  abi_version: 'mock-abi-version-for-" + cpu + "'",
                    "  abi_libc_version: 'mock-abi-libc-for-" + cpu + "'")
                .addAll(ImmutableList.copyOf(append))
                .add("}")
                .build());
  }

  /**
   * Creates action_config for {@code actionName} action using DUMMY_TOOL that doesn't imply any
   * features.
   */
  private static String emptyActionConfigFor(String actionName) {
    return String.format(
        "action_config {"
            + "  config_name: '%s'"
            + "  action_name: '%s'"
            + "  tool {"
            + "    tool_path: 'DUMMY_TOOL'"
            + "  }"
            + "}",
        actionName, actionName);
  }

  /** Filter to remove implicit dependencies of C/C++ rules. */
  private final Predicate<Label> ccLabelFilter =
      new Predicate<Label>() {
        @Override
        public boolean apply(Label label) {
          return labelNameFilter().apply("//" + label.getPackageName());
        }
      };

  public static String mergeCrosstoolConfig(String original, CToolchain toolchain)
      throws TextFormat.ParseException {
    CrosstoolConfig.CrosstoolRelease.Builder builder =
        CrosstoolConfig.CrosstoolRelease.newBuilder();
    TextFormat.merge(original, builder);
    for (CToolchain.Builder toolchainBuilder : builder.getToolchainBuilderList()) {
      toolchainBuilder.mergeFrom(toolchain);
    }
    return TextFormat.printToString(builder.build());
  }

  public abstract Predicate<String> labelNameFilter();

  /**
   * Setup the support for building C/C++.
   */
  public abstract void setup(MockToolsConfig config) throws IOException;

  /**
   * Creates a crosstool package by merging {@code toolchain} with the default mock CROSSTOOL file.
   *
   * @param partialToolchain A string representation of a CToolchain protocol buffer; note that
   *        this is allowed to be a partial buffer (required fields may be omitted).
   */
  public void setupCrosstool(MockToolsConfig config, String... partialToolchain)
      throws IOException {
    setupCrosstool(config, /* appendToCurrentToolchain= */ true, partialToolchain);
  }

  public void setupCrosstool(
      MockToolsConfig config, boolean appendToCurrentToolchain, String... partialToolchain)
      throws IOException {
    String toolchainString = Joiner.on("\n").join(partialToolchain);
    String crosstoolFile;
    if (appendToCurrentToolchain) {
      CToolchain.Builder toolchainBuilder = CToolchain.newBuilder();
      TextFormat.merge(toolchainString, toolchainBuilder);
      crosstoolFile = mergeCrosstoolConfig(readCrosstoolFile(), toolchainBuilder.buildPartial());
    } else {
      crosstoolFile = readCrosstoolFile() + toolchainString;
    }
    createCrosstoolPackage(
        config,
        crosstoolFile);
  }

  public void setupCcToolchainConfigForCpu(MockToolsConfig config, String... cpus)
      throws IOException {
    String crosstoolTop = getCrosstoolTopPathForConfig(config);
    ImmutableList.Builder<CcToolchainConfig> toolchainConfigBuilder = ImmutableList.builder();
    toolchainConfigBuilder.add(CcToolchainConfig.getDefaultCcToolchainConfig());
    for (String cpu : cpus) {
      toolchainConfigBuilder.add(CcToolchainConfig.getCcToolchainConfigForCpu(cpu));
    }
    new Crosstool(config, crosstoolTop, /* disableCrosstool= */ true)
        .setSupportedArchs(getCrosstoolArchs())
        .setToolchainConfigs(toolchainConfigBuilder.build())
        .setSupportsHeaderParsing(true)
        .write();
  }

  public void setupCcToolchainConfig(
      MockToolsConfig config, CcToolchainConfig.Builder ccToolchainConfig) throws IOException {
    String crosstoolTop = getCrosstoolTopPathForConfig(config);
    new Crosstool(config, crosstoolTop, /* disableCrosstool= */ true)
        .setSupportedArchs(getCrosstoolArchs())
        .setToolchainConfigs(ImmutableList.of(ccToolchainConfig.build()))
        .setSupportsHeaderParsing(true)
        .write();
  }

  protected void createCrosstoolPackage(
      MockToolsConfig config,
      String crosstoolFile)
      throws IOException {
    String crosstoolTop = getCrosstoolTopPathForConfig(config);
    if (config.isRealFileSystem()) {
      config.linkTools(getRealFilesystemTools(crosstoolTop));
    } else {
      new Crosstool(config, crosstoolTop, /* disableCrosstool= */ false)
          .setCrosstoolFile(getMockCrosstoolVersion(), crosstoolFile)
          .setSupportedArchs(getCrosstoolArchs())
          .setSupportsHeaderParsing(true)
          .write();
    }
  }

  protected String getCrosstoolTopPathForConfig(MockToolsConfig config) {
    if (config.isRealFileSystem()) {
      return getRealFilesystemCrosstoolTopPath();
    } else {
      return getMockCrosstoolPath();
    }
  }

  public abstract String getMockCrosstoolPath();

  public static PackageIdentifier getMockCrosstoolsTop() {
    try {
      return PackageIdentifier.create(
          RepositoryName.create(TestConstants.TOOLS_REPOSITORY),
          PathFragment.create(TestConstants.MOCK_CC_CROSSTOOL_PATH));
    } catch (LabelSyntaxException e) {
      Verify.verify(false);
      throw new AssertionError(e);
    }
  }

  protected String readCrosstoolFile() throws IOException {
    return ResourceLoader.readFromResources(
        "com/google/devtools/build/lib/analysis/mock/MOCK_CROSSTOOL");
  }

  public abstract String getMockCrosstoolVersion();

  public abstract Label getMockCrosstoolLabel();

  protected abstract ImmutableList<String> getCrosstoolArchs();

  protected abstract String[] getRealFilesystemTools(String crosstoolTop);

  protected abstract String getRealFilesystemCrosstoolTopPath();

  public final Predicate<Label> labelFilter() {
    return ccLabelFilter;
  }
}
