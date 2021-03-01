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

import static java.util.stream.Collectors.joining;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ExpansionException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.LinkBuildVariables;
import com.google.devtools.build.lib.rules.cpp.LinkCommandLine;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

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

  public static final String FDO_SPLIT_FUNCTIONS = "fdo_split_functions";

  public static final String SPLIT_FUNCTIONS = "split_functions";

  public static final ImmutableList<String> STATIC_LINK_TWEAKED_ARTIFACT_NAME_PATTERN =
      ImmutableList.of("static_library", "lib", ".lib");

  public static final ImmutableList<String> STATIC_LINK_AS_DOT_A_ARTIFACT_NAME_PATTERN =
      ImmutableList.of("static_library", "lib", ".a");

  public static final String EMPTY_EXECUTABLE_ACTION_CONFIG =
      emptyActionConfigFor(LinkTargetType.EXECUTABLE.getActionName());

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
  public static final String CPP_COMPILE_ACTION_WITH_REQUIREMENTS = "cpp_compile_with_requirements";

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

  /** Returns the additional linker options for this link. */
  public static ImmutableList<String> getLinkopts(LinkCommandLine linkCommandLine)
      throws ExpansionException {
    if (linkCommandLine
        .getBuildVariables()
        .isAvailable(LinkBuildVariables.USER_LINK_FLAGS.getVariableName())) {
      return CcToolchainVariables.toStringList(
          linkCommandLine.getBuildVariables(),
          LinkBuildVariables.USER_LINK_FLAGS.getVariableName());
    } else {
      return ImmutableList.of();
    }
  }

  public abstract Predicate<String> labelNameFilter();

  /**
   * Setup the support for building C/C++.
   */
  public abstract void setup(MockToolsConfig config) throws IOException;

  public void setupCcToolchainConfigForCpu(MockToolsConfig config, String... cpus)
      throws IOException {
    if (config.isRealFileSystem()) {
      String crosstoolTopPath = getRealFilesystemCrosstoolTopPath();
      config.linkTools(getRealFilesystemTools(crosstoolTopPath));
      writeToolchainsForRealFilesystemTools(config, crosstoolTopPath);
    } else {
      ImmutableList.Builder<CcToolchainConfig> toolchainConfigBuilder = ImmutableList.builder();
      toolchainConfigBuilder.add(CcToolchainConfig.getDefaultCcToolchainConfig());
      for (String cpu : cpus) {
        toolchainConfigBuilder.add(CcToolchainConfig.getCcToolchainConfigForCpu(cpu));
      }
      new Crosstool(config, getMockCrosstoolPath(), getMockCrosstoolLabel())
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
    if (config.isRealFileSystem()) {
      String crosstoolTopPath = getRealFilesystemCrosstoolTopPath();
      config.linkTools(getRealFilesystemTools(crosstoolTopPath));
      writeToolchainsForRealFilesystemTools(config, crosstoolTopPath);
    } else {
      new Crosstool(config, getMockCrosstoolPath(), getMockCrosstoolLabel())
          .setCcToolchainFile(readCcToolchainConfigFile())
          .setSupportedArchs(getCrosstoolArchs())
          .setToolchainConfigs(ImmutableList.of(ccToolchainConfig.build()))
          .setSupportsHeaderParsing(true)
          .write();
    }
  }

  /** Writes a basic toolchain definition to keep the CC tests working. */
  // TODO(cc-rules): Remove this when crosstool provides its own toolchain definitions.
  private void writeToolchainsForRealFilesystemTools(
      MockToolsConfig config, String crosstoolTopPath) throws IOException {
    config.create(
        "toolchains/BUILD",
        "toolchain(",
        "    name = 'k8-toolchain',",
        "    toolchain = '//" + crosstoolTopPath + ":cc-compiler-k8-llvm',",
        "    toolchain_type = '" + TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_type',",
        "    target_compatible_with = [",
        "        '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:x86_64',",
        "        '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:linux',",
        "    ],",
        ")",
        "toolchain(",
        "    name = 'arm-toolchain',",
        "    toolchain = '//" + crosstoolTopPath + ":cc-compiler-arm-llvm',",
        "    toolchain_type = '" + TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_type',",
        "    target_compatible_with = [",
        "        '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:arm',",
        "        '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:android',",
        "    ],",
        ")");
    config.append("WORKSPACE", "register_toolchains('//toolchains:all')");
  }

  protected void setupRulesCc(MockToolsConfig config) throws IOException {
    for (String path :
        ImmutableList.of(
            "cc/BUILD",
            "cc/defs.bzl",
            "cc/action_names.bzl",
            "cc/cc_toolchain_config_lib.bzl",
            "cc/find_cc_toolchain.bzl",
            "cc/toolchain_utils.bzl",
            "cc/private/rules_impl/BUILD")) {
      try {
        config.create(
            TestConstants.RULES_CC_REPOSITORY_SCRATCH + path,
            ResourceLoader.readFromResources(TestConstants.RULES_CC_REPOSITORY_EXECROOT + path));
      } catch (Exception e) {
        throw new RuntimeException("Couldn't read rules_cc file from " + path, e);
      }
    }

    config.overwrite(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/cpp/cc_toolchain_config_lib.bzl",
        ResourceLoader.readFromResources(
            TestConstants.RULES_CC_REPOSITORY_EXECROOT + "cc/cc_toolchain_config_lib.bzl"));
    config.overwrite(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/build_defs/cc/action_names.bzl",
        ResourceLoader.readFromResources(
            TestConstants.RULES_CC_REPOSITORY_EXECROOT + "cc/action_names.bzl"));
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/build_defs/cc/BUILD");
    config.append(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/cpp/BUILD", "");
  }

  protected static void createParseHeadersAndLayeringCheckWhitelist(MockToolsConfig config)
      throws IOException {
    config.create(
        TestConstants.TOOLS_REPOSITORY_SCRATCH
            + "tools/build_defs/cc/whitelists/parse_headers_and_layering_check/BUILD",
        "package_group(",
        "    name = 'disabling_parse_headers_and_layering_check_allowed',",
        "    packages = ['//...']",
        ")");
  }

  public static void createStarlarkLooseHeadersWhitelist(MockToolsConfig config, String... packages)
      throws IOException {
    String joinedPackages = Arrays.stream(packages).map(s -> "'" + s + "'").collect(joining(","));
    config.overwrite(
        TestConstants.TOOLS_REPOSITORY_SCRATCH
            + "tools/build_defs/cc/whitelists/starlark_hdrs_check/BUILD",
        "package_group(",
        "    name = 'loose_header_check_allowed_in_toolchain',",
        "    packages = [" + joinedPackages + "]",
        ")");
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

  protected String readCcToolchainConfigFile() throws IOException {
    return ResourceLoader.readFromResources(
        "com/google/devtools/build/lib/analysis/mock/cc_toolchain_config.bzl");
  }

  public abstract Label getMockCrosstoolLabel();

  protected abstract ImmutableList<String> getCrosstoolArchs();

  protected abstract String[] getRealFilesystemTools(String crosstoolTop);

  protected abstract String getRealFilesystemCrosstoolTopPath();

  public final Predicate<Label> labelFilter() {
    return ccLabelFilter;
  }

  public void writeMacroFile(MockToolsConfig config) throws IOException {
    List<String> ruleNames =
        ImmutableList.of(
            "cc_library",
            "cc_binary",
            "cc_test",
            "cc_import",
            "objc_import",
            "objc_library",
            "cc_toolchain",
            "cc_toolchain_suite",
            "fdo_profile",
            "fdo_prefetch_hints",
            "cc_proto_library");
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "third_party/cc_rules/macros/BUILD", "");

    StringBuilder macros = new StringBuilder();
    for (String ruleName : ruleNames) {
      Joiner.on("\n")
          .appendTo(
              macros,
              "def " + ruleName + "(**attrs):",
              "    if 'tags' in attrs and attrs['tags'] != None:",
              "        attrs['tags'] = attrs['tags'] +"
                  + " ['__CC_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__']",
              "    else:",
              "        attrs['tags'] = ['__CC_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__']",
              "    native." + ruleName + "(**attrs)");
      macros.append("\n");
    }
    config.create(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + "third_party/cc_rules/macros/defs.bzl",
        macros.toString());
  }

  public String getMacroLoadStatement(boolean loadMacro, String... ruleNames) {
    if (!loadMacro) {
      return "";
    }
    Preconditions.checkState(ruleNames.length > 0);
    StringBuilder loadStatement =
        new StringBuilder()
            .append("load('")
            .append(TestConstants.TOOLS_REPOSITORY)
            .append("//third_party/cc_rules/macros:defs.bzl', ");
    ImmutableList.Builder<String> quotedRuleNames = ImmutableList.builder();
    for (String ruleName : ruleNames) {
      quotedRuleNames.add(String.format("'%s'", ruleName));
    }
    Joiner.on(",").appendTo(loadStatement, quotedRuleNames.build());
    loadStatement.append(")");
    return loadStatement.toString();
  }
}
