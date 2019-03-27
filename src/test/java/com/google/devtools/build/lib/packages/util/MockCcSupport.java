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

  public static final String SUPPORTS_DYNAMIC_LINKER_FEATURE =
      "feature { name: '" + CppRuleClasses.SUPPORTS_DYNAMIC_LINKER + "' enabled: true}";

  public static final String SUPPORTS_INTERFACE_SHARED_LIBRARIES_FEATURE =
      "feature { name: '" + CppRuleClasses.SUPPORTS_INTERFACE_SHARED_LIBRARIES + "' enabled: true}";

  public static final String SIMPLE_LAYERING_CHECK_FEATURE_CONFIGURATION = "simple_layering_check";

  public static final String HEADER_MODULES_FEATURES = "header_modules_feature_configuration";

  /** A feature configuration snippet useful for testing environment variables. */
  public static final String ENV_VAR_FEATURES = "env_var_feature_configuration";

  public static final String HOST_AND_NONHOST_CONFIGURATION_FEATURES =
      "host_and_nonhost_configuration";

  public static final String USER_COMPILE_FLAGS = "user_compile_flags";

  public static final String AUTOFDO_IMPLICIT_THINLTO = "autofdo_implicit_thinlto";

  public static final String FDO_IMPLICIT_THINLTO = "fdo_implicit_thinlto";

  public static final String XFDO_IMPLICIT_THINLTO = "xbinaryfdo_implicit_thinlto";

  public static final ImmutableList<String> STATIC_LINK_TWEAKED_ARTIFACT_NAME_PATTERN =
      ImmutableList.of("static_library", "lib", ".lib");

  public static final ImmutableList<String> STATIC_LINK_AS_DOT_A_ARTIFACT_NAME_PATTERN =
      ImmutableList.of("static_library", "lib", ".a");

  public static final String EMPTY_EXECUTABLE_ACTION_CONFIG =
      emptyActionConfigFor(LinkTargetType.EXECUTABLE.getActionName());

  public static final String STATIC_LINK_CPP_RUNTIMES_FEATURE =
      "feature { name: 'static_link_cpp_runtimes' enabled: true }";

  public static final String EMPTY_CC_TOOLCHAIN =
      Joiner.on("\n")
          .join(
              "def _impl(ctx):",
              "    return cc_common.create_cc_toolchain_config_info(",
              "                ctx = ctx,",
              "                toolchain_identifier = 'mock-llvm-toolchain-k8',",
              "                host_system_name = 'mock-system-name-for-k8',",
              "                target_system_name = 'mock-target-system-name-for-k8',",
              "                target_cpu = 'k8',",
              "                target_libc = 'mock-libc-for-k8',",
              "                compiler = 'mock-compiler-for-k8',",
              "                abi_libc_version = 'mock-abi-libc-for-k8',",
              "                abi_version = 'mock-abi-version-for-k8')",
              "cc_toolchain_config = rule(",
              "    implementation = _impl,",
              "    attrs = {},",
              "    provides = [CcToolchainConfigInfo],",
              ")");

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
   * @param partialToolchain A string representation of a CToolchain protocol buffer; note that this
   *     is allowed to be a partial buffer (required fields may be omitted).
   */
  public void setupCrosstool(MockToolsConfig config, String... partialToolchain)
      throws IOException {
    String toolchainString = Joiner.on("\n").join(partialToolchain);

    CToolchain.Builder toolchainBuilder = CToolchain.newBuilder();
    TextFormat.merge(toolchainString, toolchainBuilder);
    String crosstoolFile =
        mergeCrosstoolConfig(readCrosstoolFile(), toolchainBuilder.buildPartial());
    createCrosstoolPackage(
        config,
        crosstoolFile);
  }

  public void setupCcToolchainConfigForCpu(MockToolsConfig config, String... cpus)
      throws IOException {
    String crosstoolTop = getCrosstoolTopPathForConfig(config);
    if (config.isRealFileSystem()) {
      config.linkTools(getRealFilesystemTools(crosstoolTop));
    } else {
      ImmutableList.Builder<CcToolchainConfig> toolchainConfigBuilder = ImmutableList.builder();
      toolchainConfigBuilder.add(CcToolchainConfig.getDefaultCcToolchainConfig());
      for (String cpu : cpus) {
        toolchainConfigBuilder.add(CcToolchainConfig.getCcToolchainConfigForCpu(cpu));
      }
      new Crosstool(config, crosstoolTop, /* disableCrosstool= */ true)
          .setCcToolchainFile(readCcToolchainConfigFile())
          .setSupportedArchs(getCrosstoolArchs())
          .setToolchainConfigs(toolchainConfigBuilder.build())
          .setSupportsHeaderParsing(true)
          .write();
    }
  }

  public void setupCcToolchainConfig(MockToolsConfig config) throws IOException {
    setupCcToolchainConfig(config, CcToolchainConfig.builder());
  }

  public void setupCcToolchainConfig(
      MockToolsConfig config, CcToolchainConfig.Builder ccToolchainConfig) throws IOException {
    String crosstoolTop = getCrosstoolTopPathForConfig(config);
    if (config.isRealFileSystem()) {
      config.linkTools(getRealFilesystemTools(crosstoolTop));
    } else {
      new Crosstool(config, crosstoolTop, /* disableCrosstool= */ true)
          .setCcToolchainFile(readCcToolchainConfigFile())
          .setSupportedArchs(getCrosstoolArchs())
          .setToolchainConfigs(ImmutableList.of(ccToolchainConfig.build()))
          .setSupportsHeaderParsing(true)
          .write();
    }
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

  protected String readCcToolchainConfigFile() throws IOException {
    return ResourceLoader.readFromResources(
        "com/google/devtools/build/lib/analysis/mock/cc_toolchain_config.bzl");
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
