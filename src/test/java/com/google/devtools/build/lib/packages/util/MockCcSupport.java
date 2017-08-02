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
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
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

  /**
   * Builder to build crosstool files for unit testing.
   */
  public static final class CrosstoolBuilder {
    public static final String DEFAULT_TARGET_CPU = "k8";
    public static final String MAJOR_VERSION = "13";
    public static final String MINOR_VERSION = "0";

    private final CrosstoolConfig.CrosstoolRelease.Builder builder =
        CrosstoolConfig.CrosstoolRelease.newBuilder()
            .setMajorVersion(MAJOR_VERSION)
            .setMinorVersion(MINOR_VERSION)
            .setDefaultTargetCpu(DEFAULT_TARGET_CPU);

    /** Adds a default toolchain in the order of the calls. */
    public CrosstoolBuilder addDefaultToolchain(String cpu, String toolchainIdentifier) {
      builder.addDefaultToolchainBuilder().setCpu(cpu).setToolchainIdentifier(toolchainIdentifier);
      return this;
    }

    /** Adds a default toolchain in the order of the calls. */
    public CrosstoolBuilder addDefaultToolchain(
        String cpu, String toolchainIdentifier, boolean supportsLipo) {
      builder
          .addDefaultToolchainBuilder()
          .setCpu(cpu)
          .setToolchainIdentifier(toolchainIdentifier)
          .setSupportsLipo(supportsLipo);
      return this;
    }

    /** Adds a toolchain where all required fields are set. */
    public CrosstoolBuilder addToolchain(String cpu, String toolchainIdentifier, String compiler) {
      builder
          .addToolchainBuilder()
          .setTargetCpu(cpu)
          .setToolchainIdentifier(toolchainIdentifier)
          .setHostSystemName("host-system-name")
          .setTargetSystemName("target-system-name")
          .setTargetLibc("target-libc")
          .setCompiler(compiler)
          .setAbiVersion("abi-version")
          .setAbiLibcVersion("abi-libc-version")
          // TODO(klimek): Move to features.
          .setSupportsStartEndLib(true);
      return this;
    }

    /** Appends the snippet {@code configuration} to all previously added toolchains. */
    public CrosstoolBuilder appendToAllToolchains(CToolchain configuration) {
      for (CToolchain.Builder toolchainBuilder : builder.getToolchainBuilderList()) {
        toolchainBuilder.mergeFrom(configuration);
      }
      return this;
    }

    /** Appends the snippet {@code configuration} to all previously added toolchains. */
    public CrosstoolBuilder appendToAllToolchains(String... configuration) throws IOException {
      CToolchain.Builder toolchainBuilder = CToolchain.newBuilder();
      TextFormat.merge(Joiner.on("\n").join(configuration), toolchainBuilder);
      appendToAllToolchains(toolchainBuilder.buildPartial());
      return this;
    }

    /** Returns the text proto containing the crosstool. */
    public String build() {
      return TextFormat.printToString(builder.build());
    }
  }

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
              && !pathString.endsWith(".cppmap");
        }
      };

  /** This feature will prevent bazel from patching the crosstool. */
  public static final String NO_LEGACY_FEATURES_FEATURE = "feature { name: 'no_legacy_features' }";

  /** Feature expected by the C++ rules when pic build is requested */
  public static final String PIC_FEATURE =
      ""
          + "feature {"
          + "  name: 'pic'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-module-codegen'"
          + "    action: 'c++-module-compile'"
          + "    action: 'preprocess-assemble'"
          + "    expand_if_all_available: 'pic'"
          + "    flag_group {"
          + "      flag: '-fPIC'"
          + "    }"
          + "  }"
          + "}";

  /**
   * A feature configuration snippet useful for testing header processing.
   */
  public static final String HEADER_PROCESSING_FEATURE_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'parse_headers'"
          + "  flag_set {"
          + "    action: 'c++-header-parsing'"
          + "    flag_group {"
          + "      flag: '<c++-header-parsing>'"
          + "    }"
          + "  }"
          + "}"
          + "feature {"
          + "  name: 'preprocess_headers'"
          + "  flag_set {"
          + "    action: 'c++-header-preprocessing'"
          + "    flag_group {"
          + "      flag: '<c++-header-preprocessing>'"
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
          + "    action: 'c++-header-preprocessing'"
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
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-header-preprocessing'"
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
          + "    action: 'c++-header-preprocessing'"
          + "    action: 'c++-modules-compile'"
          + "    flag_group {"
          + "      iterate_over: 'module_files'"
          + "      flag: 'module_file:%{module_files}'"
          + "    }"
          + "  }"
          + "}";

  /**
   * A feature configuration snippet useful for testing environment variables.
   */
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
          + "    action: 'c++-header-preprocessing'"
          + "    action: 'c++-module-compile'"
          + "    env_entry {"
          + "      key: 'cat'"
          + "      value: 'meow'"
          + "    }"
          + "  }"
          + "}"
          + "feature {"
          + "  name: 'module_maps'"
          + "  env_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-header-preprocessing'"
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

  public static final String THIN_LTO_CONFIGURATION =
      ""
          + "feature { "
          + "  name: 'thin_lto'"
          + "  requires { feature: 'nonhost' }"
          + "  flag_set {"
          + "    expand_if_all_available: 'thinlto_param_file'"
          + "    action: 'c++-link-executable'"
          + "    action: 'c++-link-dynamic-library'"
          + "    action: 'c++-link-static-library'"
          + "    action: 'c++-link-alwayslink-static-library'"
          + "    action: 'c++-link-pic-static-library'"
          + "    action: 'c++-link-alwayslink-pic-static-library'"
          + "    flag_group {"
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

  public static final String AUTO_FDO_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'autofdo'"
          + "  provides: 'profile'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'lto-backend'"
          + "    expand_if_all_available: 'fdo_profile_path'"
          + "    flag_group {"
          + "      flag: '-fauto-profile=%{fdo_profile_path}'"
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
          + "    action: 'c++-link-interface-dynamic-library'"
          + "    action: 'c++-link-dynamic-library'"
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
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'assemble'"
          + "    action: 'preprocess-assemble'"
          + "    action: 'c++-module-codegen'"
          + "    action: 'lto-backend'"
          + "    expand_if_all_available: 'per_object_debug_info_file'"
          + "    flag_group {"
          + "      flag: 'per_object_debug_info_option'"
          + "    }"
          + "  }"
          + "}";

  public static final String STATIC_LINK_TWEAKED_CONFIGURATION =
      ""
          + "artifact_name_pattern {"
          + "   category_name: 'static_library'"
          + "   pattern: 'lib%{base_name}.tweaked.a'"
          + "}";

  public static final String STATIC_LINK_AS_DOT_A_CONFIGURATION =
      ""
          + "artifact_name_pattern {"
          + "   category_name: 'static_library'"
          + "   pattern: 'lib%{base_name}.a'"
          + "}";

  public static final String STATIC_LINK_BAD_TEMPLATE_CONFIGURATION =
      ""
          + "artifact_name_pattern {"
          + "   category_name: 'static_library'"
          + "   pattern: 'foo%{bad_variable}bar'"
          + "}";

  public static final String EMPTY_COMPILE_ACTION_CONFIG =
      emptyActionConfigFor(CppCompileAction.CPP_COMPILE);

  public static final String EMPTY_MODULE_CODEGEN_ACTION_CONFIG =
      emptyActionConfigFor(CppCompileAction.CPP_MODULE_CODEGEN);

  public static final String EMPTY_MODULE_COMPILE_ACTION_CONFIG =
      emptyActionConfigFor(CppCompileAction.CPP_MODULE_COMPILE);

  public static final String EMPTY_EXECUTABLE_ACTION_CONFIG =
      emptyActionConfigFor(LinkTargetType.EXECUTABLE.getActionName());

  public static final String EMPTY_DYNAMIC_LIBRARY_ACTION_CONFIG =
      emptyActionConfigFor(LinkTargetType.DYNAMIC_LIBRARY.getActionName());

  public static final String EMPTY_STATIC_LIBRARY_ACTION_CONFIG =
      emptyActionConfigFor(LinkTargetType.STATIC_LIBRARY.getActionName());

  public static final String EMPTY_CLIF_MATCH_ACTION_CONFIG =
      emptyActionConfigFor(CppCompileAction.CLIF_MATCH);

  /**
   * Creates action_config for {@code actionName} action using DUMMY_TOOL that doesn't imply any
   * features.
   */
  private static String emptyActionConfigFor(String actionName) {
    return String.format(
        ""
            + "action_config {"
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

  public static String addOptionalDefaultCoptsToCrosstool(String original)
      throws TextFormat.ParseException {
    CrosstoolConfig.CrosstoolRelease.Builder builder =
        CrosstoolConfig.CrosstoolRelease.newBuilder();
    TextFormat.merge(original, builder);
    for (CrosstoolConfig.CToolchain.Builder toolchain : builder.getToolchainBuilderList()) {
      CrosstoolConfig.CToolchain.OptionalFlag.Builder defaultTrue =
          CrosstoolConfig.CToolchain.OptionalFlag.newBuilder();
      defaultTrue.setDefaultSettingName("crosstool_default_true");
      defaultTrue.addFlag("-DDEFAULT_TRUE");
      toolchain.addOptionalCompilerFlag(defaultTrue.build());
      CrosstoolConfig.CToolchain.OptionalFlag.Builder defaultFalse =
          CrosstoolConfig.CToolchain.OptionalFlag.newBuilder();
      defaultFalse.setDefaultSettingName("crosstool_default_false");
      defaultFalse.addFlag("-DDEFAULT_FALSE");
      toolchain.addOptionalCompilerFlag(defaultFalse.build());
    }

    CrosstoolConfig.CrosstoolRelease.DefaultSetting.Builder defaultTrue =
        CrosstoolConfig.CrosstoolRelease.DefaultSetting.newBuilder();
    defaultTrue.setName("crosstool_default_true");
    defaultTrue.setDefaultValue(true);
    builder.addDefaultSetting(defaultTrue.build());
    CrosstoolConfig.CrosstoolRelease.DefaultSetting.Builder defaultFalse =
        CrosstoolConfig.CrosstoolRelease.DefaultSetting.newBuilder();
    defaultFalse.setName("crosstool_default_false");
    defaultFalse.setDefaultValue(false);
    builder.addDefaultSetting(defaultFalse.build());

    return TextFormat.printToString(builder.build());
  }

  public static String addLibcLabelToCrosstool(String original, String label)
      throws TextFormat.ParseException {
    CrosstoolConfig.CrosstoolRelease.Builder builder =
        CrosstoolConfig.CrosstoolRelease.newBuilder();
    TextFormat.merge(original, builder);
    for (CrosstoolConfig.CToolchain.Builder toolchain : builder.getToolchainBuilderList()) {
      toolchain.setDefaultGrteTop(label);
    }
    return TextFormat.printToString(builder.build());
  }

  public abstract Predicate<String> labelNameFilter();

  /**
   * Setup the support for building C/C++.
   */
  public abstract void setup(MockToolsConfig config) throws IOException;

  public void setupCrosstoolWithEmbeddedRuntimes(MockToolsConfig config) throws IOException {
    createCrosstoolPackage(config, true);
  }

  /**
   * Creates a crosstool package by merging {@code toolchain} with the default mock CROSSTOOL file.
   *
   * @param partialToolchain A string representation of a CToolchain protocol buffer; note that
   *        this is allowed to be a partial buffer (required fields may be omitted).
   */
  public void setupCrosstool(MockToolsConfig config, String... partialToolchain)
      throws IOException {
    CToolchain.Builder toolchainBuilder = CToolchain.newBuilder();
    TextFormat.merge(Joiner.on("\n").join(partialToolchain), toolchainBuilder);
    setupCrosstool(config, toolchainBuilder.buildPartial());
  }

  public void setupCrosstoolWithRelease(MockToolsConfig config, String crosstool)
      throws IOException {
    createCrosstoolPackage(config, false, true, null, null, crosstool);
  }

  /**
   * Creates a crosstool package by merging {@code toolchain} with the default mock CROSSTOOL file.
   */
  public void setupCrosstool(MockToolsConfig config, CToolchain toolchain) throws IOException {
    createCrosstoolPackage(
        config, /* addEmbeddedRuntimes= */ false, /* addModuleMap= */ true, null, null, toolchain);
  }

  /**
   * Create a new crosstool configuration specified by {@code builder}.
   *
   * <p>If used from a {@code BuildViewTestCase}, make sure to call {@code
   * BuildViewTestCase.invalidatePackages} after calling this method to force re-loading of the
   * crosstool's BUILD files.
   */
  public void setupCrosstoolFromScratch(MockToolsConfig config, CrosstoolBuilder builder)
      throws IOException {
    if (config.isRealFileSystem()) {
      throw new RuntimeException(
          "Cannot use setupCrosstoolFromScratch when config.isRealFileSystem() is true; "
              + "use this function only for unit tests.");
    }
    // TODO(klimek): Get the version information from the crosstool builder and set up the
    // crosstool with that information.
    createCrosstoolPackage(
        config,
        /*addEmbeddedRuntimes=*/ false,
        /*addModuleMap=*/ true,
        null,
        null,
        builder.build());
  }

  /**
   * Create a crosstool package. For integration tests, it actually links in a working crosstool,
   * for all other tests, it only creates a dummy package, with a working CROSSTOOL file.
   *
   * <p>If <code>addEmbeddedRuntimes</code> is true, it also adds filegroups for the embedded
   * runtimes.
   */
  public void setupCrosstool(
      MockToolsConfig config,
      boolean addEmbeddedRuntimes,
      boolean addModuleMap,
      String staticRuntimesLabel,
      String dynamicRuntimesLabel,
      CToolchain toolchain) throws IOException {
    createCrosstoolPackage(
        config,
        addEmbeddedRuntimes,
        addModuleMap,
        staticRuntimesLabel,
        dynamicRuntimesLabel,
        toolchain);
  }

  public void setupCrosstool(
      MockToolsConfig config,
      boolean addEmbeddedRuntimes,
      boolean addModuleMap,
      String staticRuntimesLabel,
      String dynamicRuntimesLabel,
      String crosstool)
      throws IOException {
    createCrosstoolPackage(
        config,
        addEmbeddedRuntimes,
        addModuleMap,
        staticRuntimesLabel,
        dynamicRuntimesLabel,
        crosstool);
  }

  protected static void createToolsCppPackage(MockToolsConfig config) throws IOException {
    config.create(
        "tools/cpp/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "toolchain_type(name = 'toolchain_type')",
        "cc_library(name = 'stl')",
        "alias(name='toolchain', actual='//third_party/crosstool')",
        "cc_library(name = 'malloc')",
        "filegroup(",
        "    name = 'interface_library_builder',",
        "    srcs = ['build_interface_so'],",
        ")",
        "filegroup(",
        "    name = 'link_dynamic_library',",
        "    srcs = ['link_dynamic_library.sh'],",
        ")");
    if (config.isRealFileSystem()) {
      config.linkTool("tools/cpp/link_dynamic_library.sh");
      config.linkTool("tools/cpp/build_interface_so");
    } else {
      config.create("tools/cpp/link_dynamic_library.sh", "");
      config.create("tools/cpp/build_interface_so", "");
    }
  }

  protected void createCrosstoolPackage(MockToolsConfig config, boolean addEmbeddedRuntimes)
      throws IOException {
    createCrosstoolPackage(config, addEmbeddedRuntimes, /*addModuleMap=*/ true, null, null);
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
          PathFragment.create(TestConstants.TOOLS_REPOSITORY_PATH));
    } catch (LabelSyntaxException e) {
      Verify.verify(false);
      throw new AssertionError();
    }
  }

  private void createCrosstoolPackage(
      MockToolsConfig config,
      boolean addEmbeddedRuntimes,
      boolean addModuleMap,
      String staticRuntimesLabel,
      String dynamicRuntimesLabel)
      throws IOException {
    createCrosstoolPackage(config, addEmbeddedRuntimes, addModuleMap, staticRuntimesLabel,
        dynamicRuntimesLabel, readCrosstoolFile());
  }

  private void createCrosstoolPackage(
      MockToolsConfig config,
      boolean addEmbeddedRuntimes,
      boolean addModuleMap,
      String staticRuntimesLabel,
      String dynamicRuntimesLabel,
      CToolchain toolchain)
      throws IOException {
    String crosstoolFile = mergeCrosstoolConfig(readCrosstoolFile(), toolchain);
    createCrosstoolPackage(config, addEmbeddedRuntimes, addModuleMap, staticRuntimesLabel,
        dynamicRuntimesLabel, crosstoolFile);
  }

  protected void createCrosstoolPackage(
      MockToolsConfig config,
      boolean addEmbeddedRuntimes,
      boolean addModuleMap,
      String staticRuntimesLabel,
      String dynamicRuntimesLabel,
      String crosstoolFile)
      throws IOException {
    String crosstoolTop = getCrosstoolTopPathForConfig(config);
    if (config.isRealFileSystem()) {
      config.linkTools(getRealFilesystemTools(crosstoolTop));
    } else {
      new Crosstool(config, crosstoolTop)
          .setEmbeddedRuntimes(addEmbeddedRuntimes, staticRuntimesLabel, dynamicRuntimesLabel)
          .setCrosstoolFile(getMockCrosstoolVersion(), crosstoolFile)
          .setSupportedArchs(getCrosstoolArchs())
          .setAddModuleMap(addModuleMap)
          .setSupportsHeaderParsing(true)
          .write();
    }
  }

  public abstract String getMockCrosstoolVersion();

  public abstract Label getMockCrosstoolLabel();

  public abstract String readCrosstoolFile() throws IOException;

  public abstract String getMockLibcPath();

  protected abstract ImmutableList<String> getCrosstoolArchs();

  protected abstract String[] getRealFilesystemTools(String crosstoolTop);

  protected abstract String getRealFilesystemCrosstoolTopPath();

  public final Predicate<Label> labelFilter() {
    return ccLabelFilter;
  }
}
