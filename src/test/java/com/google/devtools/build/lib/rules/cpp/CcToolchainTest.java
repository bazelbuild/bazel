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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertDoesNotContainSublist;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.packages.util.ResourceLoader;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.DynamicMode;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for toolchain features.
 */
@RunWith(JUnit4.class)
public class CcToolchainTest extends BuildViewTestCase {
  private static final String CPP_TOOLCHAIN_TYPE =
      TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_type";

  @Test
  public void testFilesToBuild() throws Exception {
    scratch.file("a/BUILD", "cc_toolchain_alias(name = 'b')");
    ConfiguredTarget b = getConfiguredTarget("//a:b");
    assertThat(ActionsTestUtil.baseArtifactNames(getFilesToBuild(b))).isNotNull();
  }

  @Test
  public void testInterfaceSharedObjects() throws Exception {
    scratch.file("a/BUILD", "cc_toolchain_alias(name = 'b')");
    getAnalysisMock()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            CrosstoolConfig.CToolchain.newBuilder()
                .setSupportsInterfaceSharedObjects(false)
                .buildPartial());
    useConfiguration();
    invalidatePackages();

    ConfiguredTarget target = getConfiguredTarget("//a:b");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(
            CppHelper.useInterfaceSharedObjects(
                getConfiguration(target).getFragment(CppConfiguration.class), toolchainProvider))
        .isFalse();

    useConfiguration("--interface_shared_objects");
    invalidatePackages();
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(
            CppHelper.useInterfaceSharedObjects(
                getConfiguration(target).getFragment(CppConfiguration.class), toolchainProvider))
        .isFalse();

    getAnalysisMock()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            CrosstoolConfig.CToolchain.newBuilder()
                .setSupportsInterfaceSharedObjects(true)
                .buildPartial());
    useConfiguration();
    invalidatePackages();

    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(
            CppHelper.useInterfaceSharedObjects(
                getConfiguration(target).getFragment(CppConfiguration.class), toolchainProvider))
        .isTrue();

    useConfiguration("--nointerface_shared_objects");
    invalidatePackages();
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(
            CppHelper.useInterfaceSharedObjects(
                getConfiguration(target).getFragment(CppConfiguration.class), toolchainProvider))
        .isFalse();
  }

  @Test
  public void testFission() throws Exception {
    scratch.file("a/BUILD", "cc_toolchain_alias(name = 'b')");

    // Default configuration: disabled.
    getAnalysisMock()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            CrosstoolConfig.CToolchain.newBuilder().setSupportsFission(true).buildPartial());
    useConfiguration();
    ConfiguredTarget target = getConfiguredTarget("//a:b");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);

    assertThat(toolchainProvider.useFission()).isFalse();

    // Mode-specific settings.
    useConfiguration("-c", "dbg", "--fission=dbg");
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(toolchainProvider.useFission()).isTrue();

    useConfiguration("-c", "dbg", "--fission=opt");
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(toolchainProvider.useFission()).isFalse();

    useConfiguration("-c", "dbg", "--fission=opt,dbg");
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(toolchainProvider.useFission()).isTrue();

    useConfiguration("-c", "fastbuild", "--fission=opt,dbg");
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(toolchainProvider.useFission()).isFalse();

    useConfiguration("-c", "fastbuild", "--fission=opt,dbg");
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(toolchainProvider.useFission()).isFalse();

    // Universally enabled
    useConfiguration("-c", "dbg", "--fission=yes");
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(toolchainProvider.useFission()).isTrue();

    useConfiguration("-c", "opt", "--fission=yes");
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(toolchainProvider.useFission()).isTrue();

    useConfiguration("-c", "fastbuild", "--fission=yes");
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(toolchainProvider.useFission()).isTrue();

    // Universally disabled
    useConfiguration("-c", "dbg", "--fission=no");
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(toolchainProvider.useFission()).isFalse();

    useConfiguration("-c", "opt", "--fission=no");
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(toolchainProvider.useFission()).isFalse();

    useConfiguration("-c", "fastbuild", "--fission=no");
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(toolchainProvider.useFission()).isFalse();
  }

  @Test
  public void testPic() throws Exception {
    scratch.file("a/BUILD", "cc_toolchain_alias(name = 'b')");

    assertThat(usePicForBinariesWithConfiguration("--cpu=piii")).isFalse();
    assertThat(usePicForBinariesWithConfiguration("--cpu=piii", "-c", "opt")).isFalse();
    assertThat(usePicForBinariesWithConfiguration("--cpu=k8")).isTrue();
    assertThat(usePicForBinariesWithConfiguration("--cpu=k8", "-c", "opt")).isFalse();
  }

  private boolean usePicForBinariesWithConfiguration(String... configuration) throws Exception {
    useConfiguration(configuration);
    ConfiguredTarget target = getConfiguredTarget("//a:b");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    RuleContext ruleContext = getRuleContext(target);
    return CppHelper.usePicForBinaries(ruleContext, toolchainProvider);
  }

  @Test
  public void testBadDynamicRuntimeLib() throws Exception {
    scratch.file(
        "a/BUILD",
        "filegroup(name='dynamic', srcs=['not-an-so', 'so.so'])",
        "filegroup(name='static', srcs=['not-an-a', 'a.a'])",
        "cc_toolchain(",
        "    name = 'a',",
        "    module_map = 'map',",
        "    ar_files = 'ar-a',",
        "    as_files = 'as-a',",
        "    cpu = 'cherry',",
        "    compiler_files = 'compile-a',",
        "    dwp_files = 'dwp-a',",
        "    coverage_files = 'gcov-a',",
        "    linker_files = 'link-a',",
        "    strip_files = 'strip-a',",
        "    objcopy_files = 'objcopy-a',",
        "    all_files = 'all-a',",
        "    dynamic_runtime_libs = [':dynamic'],",
        "    static_runtime_libs = [':static'])");

    getAnalysisMock()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            CrosstoolConfig.CToolchain.newBuilder()
                .setSupportsEmbeddedRuntimes(true)
                .buildPartial());

    useConfiguration();

    getConfiguredTarget("//a:b");
  }

  @Test
  public void testDynamicMode() throws Exception {
    scratch.file(
        "a/BUILD",
        "filegroup(",
        "   name='empty')",
        "filegroup(",
        "    name = 'banana',",
        "    srcs = ['banana1', 'banana2'])",
        "cc_toolchain(",
        "    name = 'b',",
        "    toolchain_identifier = 'toolchain-identifier-k8',",
        "    cpu = 'banana',",
        "    all_files = ':banana',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    dynamic_runtime_libs = [':empty'],",
        "    static_runtime_libs = [':empty'])");
    scratch.file("a/CROSSTOOL", AnalysisMock.get().ccSupport().readCrosstoolFile());

    // Check defaults.
    useConfiguration();
    ConfiguredTarget target = getConfiguredTarget("//a:b");
    CppConfiguration cppConfiguration =
        getConfiguration(target).getFragment(CppConfiguration.class);

    assertThat(cppConfiguration.getDynamicModeFlag()).isEqualTo(DynamicMode.DEFAULT);

    // Test "off"
    useConfiguration("--dynamic_mode=off");
    target = getConfiguredTarget("//a:b");
    cppConfiguration = getConfiguration(target).getFragment(CppConfiguration.class);

    assertThat(cppConfiguration.getDynamicModeFlag()).isEqualTo(DynamicMode.OFF);

    // Test "fully"
    useConfiguration("--dynamic_mode=fully");
    target = getConfiguredTarget("//a:b");
    cppConfiguration = getConfiguration(target).getFragment(CppConfiguration.class);

    assertThat(cppConfiguration.getDynamicModeFlag()).isEqualTo(DynamicMode.FULLY);

    // Check an invalid value for disable_dynamic.
    try {
      useConfiguration("--dynamic_mode=very");
      fail("OptionsParsingException not thrown."); // COV_NF_LINE
    } catch (OptionsParsingException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo(
              "While parsing option --dynamic_mode=very: Not a valid dynamic mode: 'very' "
                  + "(should be off, default or fully)");
    }
  }

  // Regression test for bug 2088255:
  // "StringIndexOutOfBoundsException in BuildConfiguration.<init>()"
  @Test
  public void testShortLibcVersion() throws Exception {
    scratch.file("a/BUILD", "cc_toolchain_alias(name = 'b')");

    getAnalysisMock()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            CrosstoolConfig.CToolchain.newBuilder().setTargetLibc("2.3.6").buildPartial());

    useConfiguration();

    ConfiguredTarget target = getConfiguredTarget("//a:b");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);

    assertThat(toolchainProvider.getTargetLibc()).isEqualTo("2.3.6");
  }

  @Test
  public void testParamDfDoubleQueueThresholdFactor() throws Exception {
    scratch.file("a/BUILD", "cc_toolchain_alias(name = 'b')");
    useConfiguration();

    scratch.file("lib/BUILD", "cc_library(", "   name = 'lib',", "   srcs = ['a.cc'],", ")");

    ConfiguredTarget lib = getConfiguredTarget("//lib");
    CcToolchainProvider toolchain =
        CppHelper.getToolchainUsingDefaultCcToolchainAttribute(getRuleContext(lib));

    assertDoesNotContainSublist(
        toolchain.getLegacyCompileOptionsWithCopts(),
        "--param",
        "df-double-quote-threshold-factor=0");
  }

  @Test
  public void testMergesDefaultCoptsWithUserProvidedOnes() throws Exception {
    scratch.file("a/BUILD", "cc_toolchain_alias(name = 'b')");
    scratch.file("lib/BUILD", "cc_library(name = 'lib', srcs = ['a.cc'])");

    ConfiguredTarget lib = getConfiguredTarget("//lib");
    CcToolchainProvider toolchain =
        CppHelper.getToolchainUsingDefaultCcToolchainAttribute(getRuleContext(lib));

    List<String> expected = new ArrayList<>();
    expected.addAll(toolchain.getLegacyCompileOptionsWithCopts());
    expected.add("-Dfoo");

    useConfiguration("--copt", "-Dfoo");
    lib = getConfiguredTarget("//lib");
    toolchain = CppHelper.getToolchainUsingDefaultCcToolchainAttribute(getRuleContext(lib));
    assertThat(ImmutableList.copyOf(toolchain.getLegacyCompileOptionsWithCopts()))
        .isEqualTo(ImmutableList.copyOf(expected));
  }

  public void assertInvalidIncludeDirectoryMessage(String entry, String messageRegex)
      throws Exception {
    try {
      scratch.overwriteFile("a/BUILD", "cc_toolchain_alias(name = 'b')");
      getAnalysisMock()
          .ccSupport()
          .setupCrosstool(
              mockToolsConfig,
              CrosstoolConfig.CToolchain.newBuilder()
                  .addCxxBuiltinIncludeDirectory(entry)
                  .buildPartial());

      useConfiguration();
      invalidatePackages();

      ConfiguredTarget target = getConfiguredTarget("//a:b");
      CcToolchainProvider toolchainProvider =
          (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
      // Must call this function to actually see if there's an error with the directories.
      toolchainProvider.getBuiltInIncludeDirectories();

      fail("C++ configuration creation succeeded unexpectedly");
    } catch (AssertionError e) {
      assertThat(e).hasMessageThat().containsMatch(messageRegex);
    }
  }

  @Test
  public void testInvalidIncludeDirectory() throws Exception {
    assertInvalidIncludeDirectoryMessage("%package(//a", "has an unrecognized %prefix%");
    assertInvalidIncludeDirectoryMessage(
        "%package(//a:@@a)%", "The package '//a:@@a' is not valid");
    assertInvalidIncludeDirectoryMessage(
        "%package(//a)%foo", "The path in the package.*is not valid");
    assertInvalidIncludeDirectoryMessage(
        "%package(//a)%/../bar", "The include path.*is not normalized");
  }

  @Test
  public void testModuleMapAttribute() throws Exception {
    scratchConfiguredTarget(
        "modules/map",
        "c",
        "cc_toolchain(",
        "    name = 'c',",
        "    toolchain_identifier = 'toolchain-identifier-k8',",
        "    module_map = 'map',",
        "    cpu = 'cherry',",
        "    ar_files = 'ar-cherry',",
        "    as_files = 'as-cherry',",
        "    compiler_files = 'compile-cherry',",
        "    dwp_files = 'dwp-cherry',",
        "    coverage_files = 'gcov-cherry',",
        "    linker_files = 'link-cherry',",
        "    strip_files = ':every-file',",
        "    objcopy_files = 'objcopy-cherry',",
        "    all_files = ':every-file',",
        "    dynamic_runtime_libs = ['dynamic-runtime-libs-cherry'],",
        "    static_runtime_libs = ['static-runtime-libs-cherry'])");
  }

  @Test
  public void testModuleMapAttributeOptional() throws Exception {
    scratchConfiguredTarget(
        "modules/map",
        "c",
        "cc_toolchain(",
        "    name = 'c',",
        "    toolchain_identifier = 'toolchain-identifier-k8',",
        "    cpu = 'cherry',",
        "    ar_files = 'ar-cherry',",
        "    as_files = 'as-cherry',",
        "    compiler_files = 'compile-cherry',",
        "    dwp_files = 'dwp-cherry',",
        "    linker_files = 'link-cherry',",
        "    strip_files = ':every-file',",
        "    objcopy_files = 'objcopy-cherry',",
        "    all_files = ':every-file',",
        "    dynamic_runtime_libs = ['dynamic-runtime-libs-cherry'],",
        "    static_runtime_libs = ['static-runtime-libs-cherry'])");
  }

  @Test
  public void testFailWithMultipleModuleMaps() throws Exception {
    checkError(
        "modules/multiple",
        "c",
        "expected a single artifact",
        "filegroup(name = 'multiple-maps', srcs = ['a.cppmap', 'b.cppmap'])",
        "cc_toolchain(",
        "    name = 'c',",
        "    toolchain_identifier = 'toolchain-identifier-k8',",
        "    module_map = ':multiple-maps',",
        "    cpu = 'cherry',",
        "    ar_files = 'ar-cherry',",
        "    as_files = 'as-cherry',",
        "    compiler_files = 'compile-cherry',",
        "    dwp_files = 'dwp-cherry',",
        "    coverage_files = 'gcov-cherry',",
        "    linker_files = 'link-cherry',",
        "    strip_files = ':every-file',",
        "    objcopy_files = 'objcopy-cherry',",
        "    all_files = ':every-file',",
        "    dynamic_runtime_libs = ['dynamic-runtime-libs-cherry'],",
        "    static_runtime_libs = ['static-runtime-libs-cherry'])");
  }

  @Test
  public void testToolchainAlias() throws Exception {
    ConfiguredTarget reference = scratchConfiguredTarget("a", "ref",
        "cc_toolchain_alias(name='ref')");
    assertThat(reference.get(ToolchainInfo.PROVIDER.getKey())).isNotNull();
  }

  @Test
  public void testFdoOptimizeInvalidUseGeneratedArtifact() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "a/BUILD",
        "cc_toolchain_alias(name = 'b')",
        "genrule(",
        "    name ='gen_artifact',",
        "    outs=['profile.profdata'],",
        "    cmd='touch $@')");
    useConfiguration("-c", "opt", "--fdo_optimize=//a:gen_artifact");
    assertThat(getConfiguredTarget("//a:b")).isNull();
    assertContainsEvent("--fdo_optimize points to a target that is not an input file");
  }

  @Test
  public void testFdoOptimizeUnexpectedExtension() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "a/BUILD", "cc_toolchain_alias(name = 'b')", "exports_files(['profile.unexpected'])");
    scratch.file("a/profile.unexpected", "");
    useConfiguration("-c", "opt", "--fdo_optimize=//a:profile.unexpected");
    assertThat(getConfiguredTarget("//a:b")).isNull();
    assertContainsEvent("invalid extension for FDO profile file");
  }

  @Test
  public void testFdoOptimizeNotInputFile() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "a/BUILD",
        "cc_toolchain_alias(name = 'b')",
        "filegroup(",
        "    name ='profile',",
        "    srcs=['my_profile.afdo'])");
    scratch.file("my_profile.afdo", "");
    useConfiguration("-c", "opt", "--fdo_optimize=//a:profile");
    assertThat(getConfiguredTarget("//a:b")).isNull();
    assertContainsEvent("--fdo_optimize points to a target that is not an input file");
  }

  @Test
  public void testFdoOptimizeNotCompatibleWithCoverage() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("a/BUILD", "cc_toolchain_alias(name = 'b')", "exports_files(['profile.afdo'])");
    scratch.file("a/profile.afdo", "");
    useConfiguration("-c", "opt", "--fdo_optimize=//a:profile.afdo", "--collect_code_coverage");
    assertThat(getConfiguredTarget("//a:b")).isNull();
    assertContainsEvent("coverage mode is not compatible with FDO optimization");
  }

  @Test
  public void testXFdoOptimizeNotProvider() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "a/BUILD",
        "cc_toolchain_alias(name = 'b')",
        "fdo_profile(name='out.xfdo', profile='profile.xfdo')");
    useConfiguration("-c", "opt", "--xbinary_fdo=//a:profile.xfdo");
    assertThat(getConfiguredTarget("//a:b")).isNull();
    assertContainsEvent("--fdo_profile/--xbinary_fdo input needs to be an fdo_profile rule");
  }

  @Test
  public void testXFdoOptimizeRejectAFdoInput() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "a/BUILD",
        "cc_toolchain_alias(name = 'b')",
        "fdo_profile(name='out.afdo', profile='profile.afdo')");
    useConfiguration("-c", "opt", "--xbinary_fdo=//a:out.afdo");
    assertThat(getConfiguredTarget("//a:b")).isNull();
    assertContainsEvent("--xbinary_fdo cannot accept profile input other than *.xfdo");
  }

  @Test
  public void testZipperInclusionDependsOnFdoOptimization() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "a/BUILD",
        "filegroup(",
        "   name='empty')",
        "cc_toolchain(",
        "    name = 'b',",
        "    toolchain_identifier = 'toolchain-identifier-k8',",
        "    cpu = 'banana',",
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    dynamic_runtime_libs = [':empty'],",
        "    static_runtime_libs = [':empty'])");
    scratch.file("fdo/my_profile.afdo", "");
    scratch.file(
        "fdo/BUILD",
        "exports_files(['my_profile.afdo'])",
        "fdo_profile(name = 'fdo', profile = ':my_profile.profdata')");
    scratch.file("a/CROSSTOOL", AnalysisMock.get().ccSupport().readCrosstoolFile());

    useConfiguration();
    assertThat(getPrerequisites(getConfiguredTarget("//a:b"), ":zipper")).isEmpty();

    useConfiguration("-c", "opt", "--fdo_optimize=//fdo:my_profile.afdo");
    assertThat(getPrerequisites(getConfiguredTarget("//a:b"), ":zipper")).isNotEmpty();

    useConfiguration("-c", "opt", "--fdo_profile=//fdo:fdo");
    assertThat(getPrerequisites(getConfiguredTarget("//a:b"), ":zipper")).isNotEmpty();
  }

  @Test
  public void testInlineCtoolchain_withToolchainResolution() throws Exception {
    scratch.file(
        "a/BUILD",
        "filegroup(",
        "   name='empty')",
        "cc_toolchain(",
        "    name = 'b',",
        "    cpu = 'banana',",
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    dynamic_runtime_libs = [':empty'],",
        "    static_runtime_libs = [':empty'],",
        "    proto=\"\"\"",
        "      toolchain_identifier: \"banana\"",
        "      abi_version: \"banana\"",
        "      abi_libc_version: \"banana\"",
        "      compiler: \"banana\"",
        "      host_system_name: \"banana\"",
        "      target_system_name: \"banana\"",
        "      target_cpu: \"banana\"",
        "      target_libc: \"banana\"",
        "    \"\"\")");

    getAnalysisMock()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, CrosstoolConfig.CToolchain.newBuilder()
            .setAbiVersion("orange")
            .buildPartial());

    useConfiguration("--enabled_toolchain_types=" + CPP_TOOLCHAIN_TYPE);

    ConfiguredTarget target = getConfiguredTarget("//a:b");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(toolchainProvider.getAbi()).isEqualTo("banana");
  }

  private void loadCcToolchainConfigLib() throws IOException {
    scratch.appendFile("tools/cpp/BUILD", "");
    scratch.file(
        "tools/cpp/cc_toolchain_config_lib.bzl",
        ResourceLoader.readFromResources(
            TestConstants.BAZEL_REPO_PATH + "tools/cpp/cc_toolchain_config_lib.bzl"));
  }

  @Test
  public void testToolchainFromStarlarkRule() throws Exception {
    loadCcToolchainConfigLib();
    writeStarlarkRule();

    getAnalysisMock()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            CrosstoolConfig.CToolchain.newBuilder().setAbiVersion("orange").buildPartial());

    useConfiguration("--cpu=k8", "--experimental_enable_cc_toolchain_config_info");

    ConfiguredTarget target = getConfiguredTarget("//a:a");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);

    assertThat(toolchainProvider.getAbi()).isEqualTo("banana");
    assertThat(toolchainProvider.getFeatures().getActivatableNames())
        .containsExactly("simple_action", "simple_feature", "no_legacy_features");
  }

  @Test
  public void testToolPathsInToolchainFromStarlarkRule() throws Exception {
    loadCcToolchainConfigLib();
    writeStarlarkRule();

    getAnalysisMock().ccSupport();

    useConfiguration("--cpu=k8", "--experimental_enable_cc_toolchain_config_info");

    ConfiguredTarget target = getConfiguredTarget("//a:a");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(toolchainProvider.getToolPathFragment(Tool.AR).toString())
        .isEqualTo("/absolute/path");
    assertThat(toolchainProvider.getToolPathFragment(Tool.CPP).toString())
        .isEqualTo("a/relative/path");
  }

  private void writeStarlarkRule() throws IOException {
    scratch.file(
        "a/BUILD",
        "load(':crosstool_rule.bzl', 'cc_toolchain_config_rule')",
        "cc_toolchain_config_rule(name = 'toolchain_config')",
        "filegroup(",
        "   name='empty')",
        "cc_toolchain_suite(",
        "    name = 'a',",
        "    toolchains = { 'k8': ':b' },",
        ")",
        "cc_toolchain(",
        "    name = 'b',",
        "    cpu = 'banana',",
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    dynamic_runtime_libs = [':empty'],",
        "    static_runtime_libs = [':empty'],",
        "    toolchain_config = ':toolchain_config')");
    scratch.file("a/CROSSTOOL", AnalysisMock.get().ccSupport().readCrosstoolFile());

    scratch.file(
        "a/crosstool_rule.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl',",
        "        'feature',",
        "        'action_config',",
        "        'artifact_name_pattern',",
        "        'env_entry',",
        "        'variable_with_value',",
        "        'make_variable',",
        "        'feature_set',",
        "        'with_feature_set',",
        "        'env_set',",
        "        'flag_group',",
        "        'flag_set',",
        "        'tool_path',",
        "        'tool')",
        "",
        "def _impl(ctx):",
        "    return cc_common.create_cc_toolchain_config_info(",
        "                ctx = ctx,",
        "                features = [feature(name = 'simple_feature'), ",
        "                            feature(name = 'no_legacy_features')],",
        "                action_configs = [",
        "                   action_config(action_name = 'simple_action', enabled=True)",
        "                ],",
        "                artifact_name_patterns = [artifact_name_pattern(",
        "                   category_name = 'static_library',",
        "                   prefix = 'prefix',",
        "                   extension = '.a')],",
        "                cxx_builtin_include_directories = ['dir1', 'dir2', 'dir3'],",
        "                toolchain_identifier = 'toolchain',",
        "                host_system_name = 'host',",
        "                target_system_name = 'target',",
        "                target_cpu = 'cpu',",
        "                target_libc = 'libc',",
        "                default_libc_top = 'libc_top',",
        "                compiler = 'compiler',",
        "                abi_libc_version = 'abi_libc',",
        "                abi_version = 'banana',",
        "                supports_gold_linker = True,",
        "                supports_start_end_lib = True,",
        "                tool_paths = [",
        "                     tool_path(name = 'ar', path = '/absolute/path'),",
        "                     tool_path(name = 'cpp', path = 'relative/path'),",
        "                     tool_path(name = 'gcc', path = '/some/path'),",
        "                     tool_path(name = 'gcov', path = '/some/path'),",
        "                     tool_path(name = 'gcovtool', path = '/some/path'),",
        "                     tool_path(name = 'ld', path = '/some/path'),",
        "                     tool_path(name = 'nm', path = '/some/path'),",
        "                     tool_path(name = 'objcopy', path = '/some/path'),",
        "                     tool_path(name = 'objdump', path = '/some/path'),",
        "                     tool_path(name = 'strip', path = '/some/path'),",
        "                     tool_path(name = 'dwp', path = '/some/path'),",
        "                     tool_path(name = 'llvm_profdata', path = '/some/path'),",
        "                ],",
        "                cc_target_os = 'os',",
        "                compiler_flags = ['flag1', 'flag2', 'flag3'],",
        "                linker_flags = ['flag1'],",
        "                compilation_mode_compiler_flags = {",
        "                    'OPT' : ['flagopt'], 'FASTBUILD' : ['flagfast'] ",
        "                },",
        "                objcopy_embed_flags = ['flag1'],",
        "                needs_pic = True,",
        "                builtin_sysroot = 'sysroot')",
        "cc_toolchain_config_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {},",
        "    provides = [CcToolchainConfigInfo],",
        "    fragments = ['cpp']",
        ")");
  }

  @Test
  public void testSupportsDynamicLinkerCheckFeatures() throws Exception {
    scratch.file("a/BUILD", "cc_toolchain_alias(name = 'b')");

    getAnalysisMock()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.DYNAMIC_LINKING_MODE_FEATURE);

    // To make sure the toolchain doesn't define linking_mode_flags { mode: DYNAMIC } as that would
    // also result in supportsDynamicLinker returning true
    useConfiguration("--compiler=compiler_no_dyn_linker");

    ConfiguredTarget target = getConfiguredTarget("//a:b");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);

    assertThat(toolchainProvider.supportsDynamicLinker()).isTrue();
  }

  // Tests CcCommon::enableStaticLinkCppRuntimesFeature when supports_embedded_runtimes is not
  // present at all in the toolchain.
  @Test
  public void testStaticLinkCppRuntimesSetViaSupportsEmbeddedRuntimesUnset() throws Exception {
    scratch.file("a/BUILD", "cc_toolchain_alias(name = 'b')");
    getAnalysisMock().ccSupport().setupCrosstool(mockToolsConfig);
    useConfiguration();
    ConfiguredTarget target = getConfiguredTarget("//a:b");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrReportRuleError(getRuleContext(target), toolchainProvider);
    assertThat(toolchainProvider.supportsEmbeddedRuntimes())
        .isEqualTo(featureConfiguration.isEnabled(CppRuleClasses.STATIC_LINK_CPP_RUNTIMES));
  }

  // Tests CcCommon::enableStaticLinkCppRuntimesFeature when supports_embedded_runtimes is false
  // in the toolchain.
  @Test
  public void testStaticLinkCppRuntimesSetViaSupportsEmbeddedRuntimesFalse() throws Exception {
    scratch.file("a/BUILD", "cc_toolchain_alias(name = 'b')");
    getAnalysisMock().ccSupport().setupCrosstoolWithEmbeddedRuntimes(mockToolsConfig);
    useConfiguration();
    ConfiguredTarget target = getConfiguredTarget("//a:b");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrReportRuleError(getRuleContext(target), toolchainProvider);
    assertThat(toolchainProvider.supportsEmbeddedRuntimes())
        .isEqualTo(featureConfiguration.isEnabled(CppRuleClasses.STATIC_LINK_CPP_RUNTIMES));
  }

  private FeatureConfiguration configureFeaturesForStaticLinkCppRuntimesTest(
      String partialToolchain, String configurationToUse) throws Exception {
    scratch.file("a/BUILD", "cc_toolchain_alias(name = 'b')");
    CToolchain.Builder toolchainBuilder = CToolchain.newBuilder();
    TextFormat.merge(partialToolchain, toolchainBuilder);
    getAnalysisMock()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            /* addEmbeddedRuntimes= */ true,
            /* addModuleMap= */ false,
            /* staticRuntimesLabel= */ null,
            /* dynamicRuntimesLabel= */ null,
            toolchainBuilder.buildPartial());
    useConfiguration(configurationToUse);
    ConfiguredTarget target = getConfiguredTarget("//a:b");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    return CcCommon.configureFeaturesOrReportRuleError(getRuleContext(target), toolchainProvider);
  }

  // Tests CcCommon::enableStaticLinkCppRuntimesFeature when supports_embedded_runtimes is true in
  // the toolchain and the feature is not present at all.
  @Test
  public void testSupportsEmbeddedRuntimesNoFeatureAtAll() throws Exception {
    FeatureConfiguration featureConfiguration =
        configureFeaturesForStaticLinkCppRuntimesTest("supports_embedded_runtimes: true", "");
    assertThat(featureConfiguration.isEnabled(CppRuleClasses.STATIC_LINK_CPP_RUNTIMES)).isTrue();
  }

  // Tests CcCommon::enableStaticLinkCppRuntimesFeature when supports_embedded_runtimes is true in
  // the toolchain and the feature is enabled.
  @Test
  public void testSupportsEmbeddedRuntimesFeatureEnabled() throws Exception {
    FeatureConfiguration featureConfiguration =
        configureFeaturesForStaticLinkCppRuntimesTest(
            "supports_embedded_runtimes: true", "--features=static_link_cpp_runtimes");
    assertThat(featureConfiguration.isEnabled(CppRuleClasses.STATIC_LINK_CPP_RUNTIMES)).isTrue();
  }

  // Tests CcCommon::enableStaticLinkCppRuntimesFeature when supports_embedded_runtimes is true in
  // the toolchain and the feature is disabled.
  @Test
  public void testStaticLinkCppRuntimesOverridesSupportsEmbeddedRuntimes() throws Exception {
    FeatureConfiguration featureConfiguration =
        configureFeaturesForStaticLinkCppRuntimesTest(
            "supports_embedded_runtimes: true feature { name: 'static_link_cpp_runtimes' }",
            "--features=-static_link_cpp_runtimes");
    assertThat(featureConfiguration.isEnabled(CppRuleClasses.STATIC_LINK_CPP_RUNTIMES)).isFalse();
  }

  @Test
  public void testSysroot_fromCrosstool_unset() throws Exception {
    scratch.file("a/BUILD", "cc_toolchain_alias(name = 'b')");
    scratch.file("libc1/BUILD", "filegroup(name = 'everything', srcs = ['header1.h'])");
    scratch.file("libc1/header1.h", "#define FOO 1");

    getAnalysisMock()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            CrosstoolConfig.CToolchain.newBuilder().clearDefaultGrteTop().buildPartial());
    useConfiguration();
    ConfiguredTarget target = getConfiguredTarget("//a:b");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);

    assertThat(toolchainProvider.getSysroot()).isEqualTo("/usr/grte/v1");
  }

  @Test
  public void testSysroot_fromCcToolchain() throws Exception {
    scratch.file(
        "a/BUILD",
        "filegroup(",
        "    name='empty')",
        "cc_toolchain_suite(",
        "    name = 'a',",
        "    toolchains = { 'k8': ':b' },",
        ")",
        "cc_toolchain(",
        "    name = 'b',",
        "    cpu = 'banana',",
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    dynamic_runtime_libs = [':empty'],",
        "    static_runtime_libs = [':empty'],",
        "    proto = \"\"\"",
        "      toolchain_identifier: \"a\"",
        "      host_system_name: \"a\"",
        "      target_system_name: \"a\"",
        "      target_cpu: \"a\"",
        "      target_libc: \"a\"",
        "      compiler: \"a\"",
        "      abi_version: \"a\"",
        "      abi_libc_version: \"a\"",
        "\"\"\",",
        "    libc_top = '//libc2:everything')");
    scratch.file("libc1/BUILD", "filegroup(name = 'everything', srcs = ['header1.h'])");
    scratch.file("libc1/header1.h", "#define FOO 1");
    scratch.file("libc2/BUILD", "filegroup(name = 'everything', srcs = ['header2.h'])");
    scratch.file("libc2/header2.h", "#define FOO 2");

    getAnalysisMock()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            CrosstoolConfig.CToolchain.newBuilder().setDefaultGrteTop("//libc1").buildPartial());
    useConfiguration("--cpu=k8");
    ConfiguredTarget target = getConfiguredTarget("//a:a");
    CcToolchainProvider ccToolchainProvider =
        (CcToolchainProvider) target.get(CcToolchainProvider.PROVIDER);

    assertThat(ccToolchainProvider.getSysroot()).isEqualTo("libc2");
  }

  @Test
  public void testSysroot_fromFlag() throws Exception {
    scratch.file(
        "a/BUILD",
        "filegroup(",
        "    name='empty')",
        "cc_toolchain_suite(",
        "    name = 'a',",
        "    toolchains = { 'k8': ':b' },",
        ")",
        "cc_toolchain(",
        "    name = 'b',",
        "    cpu = 'banana',",
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    dynamic_runtime_libs = [':empty'],",
        "    static_runtime_libs = [':empty'],",
        "    proto = \"\"\"",
        "      toolchain_identifier: \"a\"",
        "      host_system_name: \"a\"",
        "      target_system_name: \"a\"",
        "      target_cpu: \"a\"",
        "      target_libc: \"a\"",
        "      compiler: \"a\"",
        "      abi_version: \"a\"",
        "      abi_libc_version: \"a\"",
        "      builtin_sysroot: \":empty\"",
        "\"\"\",",
        "    libc_top = '//libc2:everything')");
    scratch.file("libc1/BUILD", "filegroup(name = 'everything', srcs = ['header.h'])");
    scratch.file("libc1/header.h", "#define FOO 1");
    scratch.file("libc2/BUILD", "filegroup(name = 'everything', srcs = ['header.h'])");
    scratch.file("libc2/header.h", "#define FOO 2");
    scratch.file("libc3/BUILD", "filegroup(name = 'everything', srcs = ['header.h'])");
    scratch.file("libc3/header.h", "#define FOO 3");

    getAnalysisMock()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            CrosstoolConfig.CToolchain.newBuilder().setDefaultGrteTop("//libc1").buildPartial());
    useConfiguration("--cpu=k8", "--grte_top=//libc3");
    ConfiguredTarget target = getConfiguredTarget("//a:a");
    CcToolchainProvider ccToolchainProvider =
        (CcToolchainProvider) target.get(CcToolchainProvider.PROVIDER);

    assertThat(ccToolchainProvider.getSysroot()).isEqualTo("libc3");
  }
}
