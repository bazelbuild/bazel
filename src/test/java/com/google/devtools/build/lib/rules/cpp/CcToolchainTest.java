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
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.DynamicMode;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.devtools.common.options.OptionsParsingException;
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
    scratch.file("a/BUILD",
        "filegroup(",
        "   name='empty')",
        "filegroup(",
        "    name = 'banana',",
        "    srcs = ['banana1', 'banana2'])",
        "cc_toolchain(",
        "    name = 'b',",
        "    cpu = 'banana',",
        "    all_files = ':banana',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    dynamic_runtime_libs = [':empty'],",
        "    static_runtime_libs = [':empty'])");

    ConfiguredTarget b = getConfiguredTarget("//a:b");
    assertThat(ActionsTestUtil.baseArtifactNames(getFilesToBuild(b)))
        .containsExactly("banana1", "banana2");
  }

  @Test
  public void testBadDynamicRuntimeLib() throws Exception {
    scratch.file("a/BUILD",
        "filegroup(name='dynamic', srcs=['not-an-so', 'so.so'])",
        "filegroup(name='static', srcs=['not-an-a', 'a.a'])",
        "cc_toolchain(",
        "    name = 'a',",
        "    module_map = 'map',",
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

    getAnalysisMock().ccSupport().setupCrosstool(mockToolsConfig,
        CrosstoolConfig.CToolchain.newBuilder()
            .setSupportsEmbeddedRuntimes(true)
            .buildPartial());

    useConfiguration();

    getConfiguredTarget("//a:a");
  }

  @Test
  public void testTurnOffDynamicLinkWhenLipoBinary() throws Exception {
    scratch.file(
        "a/BUILD",
        "filegroup(",
        "   name='empty')",
        "filegroup(",
        "    name = 'banana',",
        "    srcs = ['banana1', 'banana2'])",
        "cc_toolchain(",
        "    name = 'b',",
        "    cpu = 'banana',",
        "    all_files = ':banana',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    dynamic_runtime_libs = [':empty'],",
        "    static_runtime_libs = [':empty'])");
    scratch.file("foo/BUILD", "cc_binary(name='foo')");

    useConfiguration("--lipo=binary", "--lipo_context=//foo", "--compilation_mode=opt");
    ConfiguredTarget target = getConfiguredTarget("//a:b");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(
            CppHelper.getDynamicMode(
                target.getConfiguration().getFragment(CppConfiguration.class), toolchainProvider))
        .isEqualTo(DynamicMode.OFF);

    useConfiguration("--lipo=off", "--lipo_context=//foo");
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);
    assertThat(
            CppHelper.getDynamicMode(
                target.getConfiguration().getFragment(CppConfiguration.class), toolchainProvider))
        .isEqualTo(DynamicMode.DEFAULT);
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
        "    cpu = 'banana',",
        "    all_files = ':banana',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    dynamic_runtime_libs = [':empty'],",
        "    static_runtime_libs = [':empty'])");

    // Check defaults.
    useConfiguration();
    ConfiguredTarget target = getConfiguredTarget("//a:b");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);

    assertThat(
            CppHelper.getDynamicMode(
                target.getConfiguration().getFragment(CppConfiguration.class), toolchainProvider))
        .isEqualTo(DynamicMode.DEFAULT);

    // Test "off"
    useConfiguration("--dynamic_mode=off");
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);

    assertThat(
            CppHelper.getDynamicMode(
                target.getConfiguration().getFragment(CppConfiguration.class), toolchainProvider))
        .isEqualTo(DynamicMode.OFF);

    // Test "fully"
    useConfiguration("--dynamic_mode=fully");
    target = getConfiguredTarget("//a:b");
    toolchainProvider = (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);

    assertThat(
            CppHelper.getDynamicMode(
                target.getConfiguration().getFragment(CppConfiguration.class), toolchainProvider))
        .isEqualTo(DynamicMode.FULLY);

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

  public void assertInvalidIncludeDirectoryMessage(String entry, String messageRegex)
      throws Exception {
    try {
      scratch.overwriteFile(
          "a/BUILD",
          "filegroup(",
          "   name='empty')",
          "cc_toolchain(",
          "    name = 'b',",
          "    cpu = 'k8',",
          "    all_files = ':banana',",
          "    compiler_files = ':empty',",
          "    dwp_files = ':empty',",
          "    linker_files = ':empty',",
          "    strip_files = ':empty',",
          "    objcopy_files = ':empty',",
          "    dynamic_runtime_libs = [':empty'],",
          "    static_runtime_libs = [':empty'])");

      getAnalysisMock()
          .ccSupport()
          .setupCrosstool(
              mockToolsConfig,
              CrosstoolConfig.CToolchain.newBuilder()
                  .addCxxBuiltinIncludeDirectory(entry)
                  .buildPartial());

      useConfiguration();

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
    assertInvalidIncludeDirectoryMessage("%package(//a@@a)%", "The package '//a@@a' is not valid");
    assertInvalidIncludeDirectoryMessage(
        "%package(//a)%foo", "The path in the package.*is not valid");
    assertInvalidIncludeDirectoryMessage(
        "%package(//a)%/../bar", "The include path.*is not normalized");
  }

  @Test
  public void testModuleMapAttribute() throws Exception {
    scratchConfiguredTarget("modules/map", "c",
        "cc_toolchain(",
        "    name = 'c',",
        "    module_map = 'map',",
        "    cpu = 'cherry',",
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
    scratchConfiguredTarget("modules/map", "c",
        "cc_toolchain(",
        "    name = 'c',",
        "    cpu = 'cherry',",
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
    checkError("modules/multiple", "c", "expected a single artifact",
        "filegroup(name = 'multiple-maps', srcs = ['a.cppmap', 'b.cppmap'])",
        "cc_toolchain(",
        "    name = 'c',",
        "    module_map = ':multiple-maps',",
        "    cpu = 'cherry',",
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
}
