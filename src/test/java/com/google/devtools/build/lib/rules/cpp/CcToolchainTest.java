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
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.MoreCollectors;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.packages.util.ResourceLoader;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.DynamicMode;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.IOException;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for toolchain features.
 */
@RunWith(JUnit4.class)
public class CcToolchainTest extends BuildViewTestCase {

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
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(CppRuleClasses.SUPPORTS_INTERFACE_SHARED_LIBRARIES));
    useConfiguration("--features=-supports_interface_shared_libraries");
    invalidatePackages();

    ConfiguredTarget target = getConfiguredTarget("//a:b");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(
            CppHelper.useInterfaceSharedLibraries(
                getConfiguration(target).getFragment(CppConfiguration.class),
                toolchainProvider,
                FeatureConfiguration.EMPTY))
        .isFalse();

    useConfiguration();
    invalidatePackages();
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(
            CppHelper.useInterfaceSharedLibraries(
                getConfiguration(target).getFragment(CppConfiguration.class),
                toolchainProvider,
                FeatureConfiguration.EMPTY))
        .isFalse();

    useConfiguration("--nointerface_shared_objects");
    invalidatePackages();
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(
            CppHelper.useInterfaceSharedLibraries(
                getConfiguration(target).getFragment(CppConfiguration.class),
                toolchainProvider,
                FeatureConfiguration.EMPTY))
        .isFalse();
  }

  @Test
  public void testFission() throws Exception {
    scratch.file("a/BUILD", "cc_library(name = 'a', srcs = ['a.cc'])");

    // Default configuration: disabled.
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.PER_OBJECT_DEBUG_INFO));
    useConfiguration();

    assertThat(getCppCompileOutputs()).doesNotContain("yolo");

    // Mode-specific settings.
    useConfiguration("-c", "dbg", "--fission=dbg");
    assertThat(getCppCompileOutputs()).contains("a.dwo");

    useConfiguration("-c", "dbg", "--fission=opt");
    assertThat(getCppCompileOutputs()).doesNotContain("a.dwo");

    useConfiguration("-c", "dbg", "--fission=opt,dbg");
    assertThat(getCppCompileOutputs()).contains("a.dwo");

    useConfiguration("-c", "fastbuild", "--fission=opt,dbg");
    assertThat(getCppCompileOutputs()).doesNotContain("a.dwo");

    useConfiguration("-c", "fastbuild", "--fission=opt,dbg");
    assertThat(getCppCompileOutputs()).doesNotContain("a.dwo");

    // Universally enabled
    useConfiguration("-c", "dbg", "--fission=yes");
    assertThat(getCppCompileOutputs()).contains("a.dwo");

    useConfiguration("-c", "opt", "--fission=yes");
    assertThat(getCppCompileOutputs()).contains("a.dwo");

    useConfiguration("-c", "fastbuild", "--fission=yes");
    assertThat(getCppCompileOutputs()).contains("a.dwo");

    // Universally disabled
    useConfiguration("-c", "dbg", "--fission=no");
    assertThat(getCppCompileOutputs()).doesNotContain("a.dwo");

    useConfiguration("-c", "opt", "--fission=no");
    assertThat(getCppCompileOutputs()).doesNotContain("a.dwo");

    useConfiguration("-c", "fastbuild", "--fission=no");
    assertThat(getCppCompileOutputs()).doesNotContain("a.dwo");
  }

  private ImmutableList<String> getCppCompileOutputs() throws Exception {
    RuleConfiguredTarget target = (RuleConfiguredTarget) getConfiguredTarget("//a:a");
    return target.getActions().stream()
        .filter(a -> a.getMnemonic().equals("CppCompile"))
        .findFirst()
        .get()
        .getOutputs()
        .stream()
        .map(a -> a.getFilename())
        .collect(ImmutableList.toImmutableList());
  }

  @Test
  public void testPic() throws Exception {
    scratch.file("a/BUILD", "cc_toolchain_alias(name = 'b')");

    assertThat(usePicForBinariesWithConfiguration("--cpu=k8")).isFalse();
    assertThat(usePicForBinariesWithConfiguration("--cpu=k8", "-c", "opt")).isFalse();
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_PIC));
    invalidatePackages();
    assertThat(usePicForBinariesWithConfiguration("--cpu=k8")).isTrue();
    assertThat(usePicForBinariesWithConfiguration("--cpu=k8", "-c", "opt")).isFalse();
  }

  private boolean usePicForBinariesWithConfiguration(String... configuration) throws Exception {
    useConfiguration(configuration);
    ConfiguredTarget target = getConfiguredTarget("//a:b");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    CppConfiguration cppConfiguration = getRuleContext(target).getFragment(CppConfiguration.class);
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrThrowEvalException(
            /* requestedFeatures= */ ImmutableSet.of(),
            /* unsupportedFeatures= */ ImmutableSet.of(),
            toolchainProvider,
            cppConfiguration);
    return CppHelper.usePicForBinaries(toolchainProvider, cppConfiguration, featureConfiguration);
  }

  @Test
  public void testBadDynamicRuntimeLib() throws Exception {
    scratch.file("a/cc_toolchain_config.bzl", MockCcSupport.EMPTY_CC_TOOLCHAIN);
    scratch.file(
        "a/BUILD",
        "load(':cc_toolchain_config.bzl', 'cc_toolchain_config')",
        "filegroup(name='dynamic', srcs=['not-an-so', 'so.so'])",
        "filegroup(name='static', srcs=['not-an-a', 'a.a'])",
        "cc_toolchain(",
        "    name = 'a',",
        "    toolchain_config = ':toolchain_config',",
        "    module_map = 'map',",
        "    ar_files = 'ar-a',",
        "    as_files = 'as-a',",
        "    compiler_files = 'compile-a',",
        "    dwp_files = 'dwp-a',",
        "    coverage_files = 'gcov-a',",
        "    linker_files = 'link-a',",
        "    strip_files = 'strip-a',",
        "    objcopy_files = 'objcopy-a',",
        "    all_files = 'all-a',",
        "    dynamic_runtime_lib = ':dynamic',",
        "    static_runtime_lib = ':static')",
        "cc_toolchain_config(name='toolchain_config')");

    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.STATIC_LINK_CPP_RUNTIMES));

    useConfiguration();

    getConfiguredTarget("//a:a");
  }

  @Test
  public void testDynamicMode() throws Exception {
    scratch.file(
        "a/BUILD",
        "load(':cc_toolchain_config.bzl', 'cc_toolchain_config')",
        "filegroup(",
        "   name='empty')",
        "filegroup(",
        "    name = 'banana',",
        "    srcs = ['banana1', 'banana2'])",
        "cc_toolchain(",
        "    name = 'b',",
        "    toolchain_identifier = 'toolchain-identifier-k8',",
        "    toolchain_config = ':toolchain_config',",
        "    all_files = ':banana',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    dynamic_runtime_lib = ':empty',",
        "    static_runtime_lib = ':empty')",
        "cc_toolchain_config(name='toolchain_config')");
    scratch.file("a/cc_toolchain_config.bzl", MockCcSupport.EMPTY_CC_TOOLCHAIN);

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
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> useConfiguration("--dynamic_mode=very"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "While parsing option --dynamic_mode=very: Not a valid dynamic mode: 'very' "
                + "(should be off, default or fully)");
  }

  public void assertInvalidIncludeDirectoryMessage(String entry, String messageRegex)
      throws Exception {
    scratch.overwriteFile("a/BUILD", "cc_toolchain_alias(name = 'b')");
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withCxxBuiltinIncludeDirectories(entry));

    useConfiguration();
    invalidatePackages();

    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//a:b"));
    assertThat(e).hasMessageThat().containsMatch(messageRegex);
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
    scratch.file("modules/map/cc_toolchain_config.bzl", MockCcSupport.EMPTY_CC_TOOLCHAIN);
    scratchConfiguredTarget(
        "modules/map",
        "c",
        "load(':cc_toolchain_config.bzl', 'cc_toolchain_config')",
        "cc_toolchain(",
        "    name = 'c',",
        "    toolchain_identifier = 'toolchain-identifier-k8',",
        "    toolchain_config = ':toolchain_config',",
        "    module_map = 'map',",
        "    ar_files = 'ar-cherry',",
        "    as_files = 'as-cherry',",
        "    compiler_files = 'compile-cherry',",
        "    dwp_files = 'dwp-cherry',",
        "    coverage_files = 'gcov-cherry',",
        "    linker_files = 'link-cherry',",
        "    strip_files = ':every-file',",
        "    objcopy_files = 'objcopy-cherry',",
        "    all_files = ':every-file',",
        "    dynamic_runtime_lib = 'dynamic-runtime-libs-cherry',",
        "    static_runtime_lib = 'static-runtime-libs-cherry')",
        "cc_toolchain_config(name = 'toolchain_config')");
  }

  @Test
  public void testModuleMapAttributeOptional() throws Exception {
    scratch.file("modules/map/cc_toolchain_config.bzl", MockCcSupport.EMPTY_CC_TOOLCHAIN);
    scratchConfiguredTarget(
        "modules/map",
        "c",
        "load(':cc_toolchain_config.bzl', 'cc_toolchain_config')",
        "cc_toolchain(",
        "    name = 'c',",
        "    toolchain_identifier = 'toolchain-identifier-k8',",
        "    toolchain_config = ':toolchain_config',",
        "    ar_files = 'ar-cherry',",
        "    as_files = 'as-cherry',",
        "    compiler_files = 'compile-cherry',",
        "    dwp_files = 'dwp-cherry',",
        "    linker_files = 'link-cherry',",
        "    strip_files = ':every-file',",
        "    objcopy_files = 'objcopy-cherry',",
        "    all_files = ':every-file',",
        "    dynamic_runtime_lib = 'dynamic-runtime-libs-cherry',",
        "    static_runtime_lib = 'static-runtime-libs-cherry')",
        "cc_toolchain_config(name = 'toolchain_config')");
  }

  @Test
  public void testFailWithMultipleModuleMaps() throws Exception {
    scratch.file("modules/multiple/cc_toolchain_config.bzl", MockCcSupport.EMPTY_CC_TOOLCHAIN);
    checkError(
        "modules/multiple",
        "c",
        "expected a single artifact",
        "load(':cc_toolchain_config.bzl', 'cc_toolchain_config')",
        "filegroup(name = 'multiple-maps', srcs = ['a.cppmap', 'b.cppmap'])",
        "cc_toolchain(",
        "    name = 'c',",
        "    toolchain_identifier = 'toolchain-identifier-k8',",
        "    toolchain_config = ':toolchain_config',",
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
        "    dynamic_runtime_lib = 'dynamic-runtime-libs-cherry',",
        "    static_runtime_lib = 'static-runtime-libs-cherry')",
        "cc_toolchain_config(name = 'toolchain_config')");
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
  public void testCSFdoRejectRelativePath() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("a/BUILD", "cc_toolchain_alias(name = 'b')");
    scratch.file("a/profile.profdata", "");
    scratch.file("a/csprofile.profdata", "");
    Exception e =
        assertThrows(
            Exception.class,
            () ->
                useConfiguration(
                    "-c",
                    "opt",
                    "--fdo_optimize=a/profile.profdata",
                    "--cs_fdo_absolute_path=a/csprofile.profdata"));
    assertThat(e).hasMessageThat().contains("in --cs_fdo_absolute_path is not an absolute path");
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
        "load(':cc_toolchain_config.bzl', 'cc_toolchain_config')",
        "filegroup(",
        "   name='empty')",
        "cc_toolchain(",
        "    name = 'b',",
        "    toolchain_identifier = 'toolchain-identifier-k8',",
        "    toolchain_config = ':toolchain_config',",
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty')",
        "cc_toolchain_config(name = 'toolchain_config')");
    scratch.file("fdo/my_profile.afdo", "");
    scratch.file(
        "fdo/BUILD",
        "exports_files(['my_profile.afdo'])",
        "fdo_profile(name = 'fdo', profile = ':my_profile.profdata')");
    scratch.file("a/cc_toolchain_config.bzl", MockCcSupport.EMPTY_CC_TOOLCHAIN);

    useConfiguration();
    assertThat(getPrerequisites(getConfiguredTarget("//a:b"), ":zipper")).isEmpty();

    useConfiguration("-c", "opt", "--fdo_optimize=//fdo:my_profile.afdo");
    assertThat(getPrerequisites(getConfiguredTarget("//a:b"), ":zipper")).isNotEmpty();

    useConfiguration("-c", "opt", "--fdo_profile=//fdo:fdo");
    assertThat(getPrerequisites(getConfiguredTarget("//a:b"), ":zipper")).isNotEmpty();
  }

  private void loadCcToolchainConfigLib() throws IOException {
    scratch.appendFile("tools/cpp/BUILD", "");
    scratch.overwriteFile(
        "tools/cpp/cc_toolchain_config_lib.bzl",
        ResourceLoader.readFromResources(
            TestConstants.RULES_CC_REPOSITORY_EXECROOT + "cc/cc_toolchain_config_lib.bzl"));
  }

  @Test
  public void testToolPathsInToolchainFromStarlarkRule() throws Exception {
    loadCcToolchainConfigLib();
    writeStarlarkRule();

    useConfiguration("--cpu=k8");

    ConfiguredTarget target = getConfiguredTarget("//a:a");
    RuleContext ruleContext = getRuleContext(target);
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(toolchainProvider.getToolPathFragment(Tool.AR, ruleContext).toString())
        .isEqualTo("/absolute/path");
    assertThat(toolchainProvider.getToolPathFragment(Tool.CPP, ruleContext).toString())
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
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    toolchain_config = ':toolchain_config')");

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
        "                compiler = 'compiler',",
        "                abi_libc_version = 'abi_libc',",
        "                abi_version = 'banana',",
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
        "                builtin_sysroot = 'sysroot')",
        "cc_toolchain_config_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {},",
        "    provides = [CcToolchainConfigInfo],",
        "    fragments = ['cpp']",
        ")");
  }

  @Test
  public void testSupportsDynamicLinkerIsFalseWhenFeatureNotSet() throws Exception {
    scratch.file("a/BUILD", "cc_toolchain_alias(name = 'b')");

    ConfiguredTarget target = getConfiguredTarget("//a:b");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);

    assertThat(toolchainProvider.supportsDynamicLinker(FeatureConfiguration.EMPTY)).isFalse();
  }

  @Test
  public void testSysroot_fromCrosstool_unset() throws Exception {
    scratch.file("a/BUILD", "cc_toolchain_alias(name = 'b')");
    scratch.file("libc1/BUILD", "filegroup(name = 'everything', srcs = ['header1.h'])");
    scratch.file("libc1/header1.h", "#define FOO 1");
    useConfiguration();
    ConfiguredTarget target = getConfiguredTarget("//a:b");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);

    assertThat(toolchainProvider.getSysroot()).isEqualTo("/usr/grte/v1");
  }

  @Test
  public void correctToolFilesUsed() throws Exception {
    scratch.file(
        "a/BUILD",
        "cc_toolchain_alias(name = 'a')",
        "cc_library(name = 'l', srcs = ['l.c'])",
        "cc_library(name = 'asm', srcs = ['a.s'])",
        "cc_library(name = 'preprocessed-asm', srcs = ['a.S'])");
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_DYNAMIC_LINKER));
    useConfiguration("--incompatible_use_specific_tool_files");
    ConfiguredTarget target = getConfiguredTarget("//a:a");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    // Check that the mock toolchain tool file sets are an antichain, so that our subset assertions
    // below are meaningful.
    ImmutableList<Set<Artifact>> fileGroups =
        ImmutableList.of(
            toolchainProvider.getArFiles().toSet(),
            toolchainProvider.getLinkerFiles().toSet(),
            toolchainProvider.getCompilerFiles().toSet(),
            toolchainProvider.getAsFiles().toSet(),
            toolchainProvider.getAllFiles().toSet());
    for (int i = 0; i < fileGroups.size(); i++) {
      assertThat(fileGroups.get(i)).isNotEmpty();
      for (int j = 0; j < fileGroups.size(); j++) {
        if (i == j) {
          continue;
        }
        Set<Artifact> one = fileGroups.get(i);
        Set<Artifact> two = fileGroups.get(j);
        assertWithMessage(String.format("%s should not contain %s", one, two))
            .that(one.containsAll(two))
            .isFalse();
      }
    }
    assertThat(
            Sets.difference(
                toolchainProvider.getArFiles().toSet(), toolchainProvider.getLinkerFiles().toSet()))
        .isNotEmpty();
    assertThat(
            Sets.difference(
                toolchainProvider.getLinkerFiles().toSet(), toolchainProvider.getArFiles().toSet()))
        .isNotEmpty();

    RuleConfiguredTarget libTarget = (RuleConfiguredTarget) getConfiguredTarget("//a:l");
    Artifact staticLib =
        getOutputGroup(libTarget, "archive").toList().stream()
            .collect(MoreCollectors.onlyElement());
    ActionAnalysisMetadata staticAction = getGeneratingAction(staticLib);
    assertThat(staticAction.getInputs().toList())
        .containsAtLeastElementsIn(toolchainProvider.getArFiles().toList());
    Artifact dynamicLib =
        getOutputGroup(libTarget, "dynamic_library").toList().stream()
            .collect(MoreCollectors.onlyElement());
    ActionAnalysisMetadata dynamicAction = getGeneratingAction(dynamicLib);
    assertThat(dynamicAction.getInputs().toList())
        .containsAtLeastElementsIn(toolchainProvider.getLinkerFiles().toList());
    ActionAnalysisMetadata cCompileAction =
        libTarget.getActions().stream()
            .filter((a) -> a.getMnemonic().equals("CppCompile"))
            .collect(MoreCollectors.onlyElement());
    assertThat(cCompileAction.getInputs().toList())
        .containsAtLeastElementsIn(toolchainProvider.getCompilerFiles().toList());
    ActionAnalysisMetadata asmAction =
        ((RuleConfiguredTarget) getConfiguredTarget("//a:asm"))
            .getActions().stream()
                .filter((a) -> a.getMnemonic().equals("CppCompile"))
                .collect(MoreCollectors.onlyElement());
    assertThat(asmAction.getInputs().toList())
        .containsAtLeastElementsIn(toolchainProvider.getAsFiles().toList());
    ActionAnalysisMetadata preprocessedAsmAction =
        ((RuleConfiguredTarget) getConfiguredTarget("//a:preprocessed-asm"))
            .getActions().stream()
                .filter((a) -> a.getMnemonic().equals("CppCompile"))
                .collect(MoreCollectors.onlyElement());
    assertThat(preprocessedAsmAction.getInputs().toList())
        .containsAtLeastElementsIn(toolchainProvider.getCompilerFiles().toList());
  }

  @Test
  public void testCcToolchainLoadedThroughMacro() throws Exception {
    setupTestCcToolchainLoadedThroughMacro(/* loadMacro= */ true);
    assertThat(getConfiguredTarget("//a:a")).isNotNull();
    assertNoEvents();
  }

  @Test
  public void testCcToolchainNotLoadedThroughMacro() throws Exception {
    setupTestCcToolchainLoadedThroughMacro(/* loadMacro= */ false);
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:a");
    assertContainsEvent("rules are deprecated");
  }

  private void setupTestCcToolchainLoadedThroughMacro(boolean loadMacro) throws Exception {
    useConfiguration("--incompatible_load_cc_rules_from_bzl");
    scratch.file("a/cc_toolchain_config.bzl", MockCcSupport.EMPTY_CC_TOOLCHAIN);
    scratch.file(
        "a/BUILD",
        "load(':cc_toolchain_config.bzl', 'cc_toolchain_config')",
        getAnalysisMock().ccSupport().getMacroLoadStatement(loadMacro, "cc_toolchain"),
        getToolchainRule("a"));
  }

  @Test
  public void setupTestCcToolchainSuiteLoadedThroughMacro() throws Exception {
    setupTestCcToolchainSuiteLoadedThroughMacro(/* loadMacro= */ true);
    assertThat(getConfiguredTarget("//a:a")).isNotNull();
    assertNoEvents();
  }

  private void setupTestCcToolchainSuiteLoadedThroughMacro(boolean loadMacro) throws Exception {
    useConfiguration("--incompatible_load_cc_rules_from_bzl");
    scratch.file("a/cc_toolchain_config.bzl", MockCcSupport.EMPTY_CC_TOOLCHAIN);
    scratch.file(
        "a/BUILD",
        "load(':cc_toolchain_config.bzl', 'cc_toolchain_config')",
        getAnalysisMock()
            .ccSupport()
            .getMacroLoadStatement(loadMacro, "cc_toolchain", "cc_toolchain_suite"),
        "cc_toolchain_suite(",
        "    name = 'a',",
        "    toolchains = { 'k8': ':b' },",
        ")",
        getToolchainRule("b"));
  }

  @Test
  public void testCcToolchainSuiteNotLoadedThroughMacro() throws Exception {
    setupTestCcToolchainSuiteLoadedThroughMacro(/* loadMacro= */ false);
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:a");
    assertContainsEvent("rules are deprecated");
  }

  private static String getToolchainRule(String targetName) {
    return Joiner.on("\n")
        .join(
            "cc_toolchain(",
            "    name = '" + targetName + "',",
            "    toolchain_identifier = 'toolchain-identifier-k8',",
            "    toolchain_config = ':toolchain_config',",
            "    all_files = ':banana',",
            "    ar_files = ':empty',",
            "    as_files = ':empty',",
            "    compiler_files = ':empty',",
            "    dwp_files = ':empty',",
            "    linker_files = ':empty',",
            "    strip_files = ':empty',",
            "    objcopy_files = ':empty',",
            "    dynamic_runtime_lib = ':empty',",
            "    static_runtime_lib = ':empty')",
            "filegroup(",
            "   name='empty')",
            "filegroup(",
            "    name = 'banana',",
            "    srcs = ['banana1', 'banana2'])",
            "cc_toolchain_config(name='toolchain_config')");
  }
}
