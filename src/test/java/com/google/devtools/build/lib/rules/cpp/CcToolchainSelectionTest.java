// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ToolchainContext.ResolvedToolchainProviders;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.ScratchAttributeWriter;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.util.MockPlatformSupport;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for platform-based toolchain selection in the c++ rules. */
@RunWith(JUnit4.class)
public class CcToolchainSelectionTest extends BuildViewTestCase {

  @Before
  public void setup() throws Exception {
    MockPlatformSupport.addMockPiiiPlatform(
        mockToolsConfig, analysisMock.ccSupport().getMockCrosstoolLabel());
  }

  private CppCompileAction getCppCompileAction(String label) throws Exception {
    ConfiguredTarget target = getConfiguredTarget(label);
    List<CppCompileAction> compilationSteps =
        actionsTestUtil()
            .findTransitivePrerequisitesOf(
                getFilesToBuild(target).iterator().next(), CppCompileAction.class);
    return compilationSteps.get(0);
  }

  private static final String CPP_TOOLCHAIN_TYPE =
      TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_category";

  @Test
  public void testResolvedCcToolchain() throws Exception {
    useConfiguration(
        "--experimental_platforms=//mock_platform:mock-piii-platform",
        "--extra_toolchains=//mock_platform:toolchain_cc-compiler-piii");
    ConfiguredTarget target =
        ScratchAttributeWriter.fromLabelString(this, "cc_library", "//lib")
            .setList("srcs", "a.cc")
            .write();
    ResolvedToolchainProviders providers =
        (ResolvedToolchainProviders)
            getRuleContext(target).getToolchainContext().getResolvedToolchainProviders();
    CcToolchainProvider toolchain =
        (CcToolchainProvider)
            providers.getForToolchainType(Label.parseAbsolute(CPP_TOOLCHAIN_TYPE));
    assertThat(Iterables.getOnlyElement(toolchain.getCompile()).getExecPathString())
        .endsWith("piii");
  }

  @Test
  public void testToolchainSelectionWithPlatforms() throws Exception {
    useConfiguration(
        "--enabled_toolchain_types=" + CPP_TOOLCHAIN_TYPE,
        "--experimental_platforms=//mock_platform:mock-piii-platform",
        "--extra_toolchains=//mock_platform:toolchain_cc-compiler-piii");
    ScratchAttributeWriter.fromLabelString(this, "cc_library", "//lib")
        .setList("srcs", "a.cc")
        .write();
    CppCompileAction compileAction = getCppCompileAction("//lib");
    boolean isPiii =
        ImmutableList.copyOf(compileAction.getInputs())
            .stream()
            .anyMatch(artifact -> artifact.getExecPathString().endsWith("piii"));
    assertThat(isPiii).isTrue();
  }

  @Test
  public void testToolchainSelectionWithoutPlatforms() throws Exception {
    useConfiguration("--experimental_platforms=//mock_platform:mock-piii-platform");
    ConfiguredTarget target =
        ScratchAttributeWriter.fromLabelString(this, "cc_library", "//lib")
            .setList("srcs", "a.cc")
            .write();
    ResolvedToolchainProviders providers =
        (ResolvedToolchainProviders)
            getRuleContext(target).getToolchainContext().getResolvedToolchainProviders();
    ToolchainInfo toolchain =
        providers.getForToolchainType(Label.parseAbsolute(CPP_TOOLCHAIN_TYPE));
    assertThat(toolchain.getKeys()).isEmpty();
  }

  @Test
  public void testCToolchainSelectionFromCcToolchainAttrs() throws Exception {
    useConfiguration(
        "--enabled_toolchain_types=" + CPP_TOOLCHAIN_TYPE,
        "--experimental_platforms=//mock_platform:mock-piii-platform",
        "--extra_toolchains=//mock_platform:toolchain_cc-compiler-piii");
    ConfiguredTarget target =
        ScratchAttributeWriter.fromLabelString(this, "cc_library", "//lib")
            .setList("srcs", "a.cc")
            .write();
    ResolvedToolchainProviders providers =
        (ResolvedToolchainProviders)
            getRuleContext(target).getToolchainContext().getResolvedToolchainProviders();
    CcToolchainProvider toolchain =
        (CcToolchainProvider)
            providers.getForToolchainType(Label.parseAbsolute(CPP_TOOLCHAIN_TYPE));
    assertThat(toolchain.getToolchain().getTargetCpu()).isEqualTo("piii");
  }

  @Test
  public void testErrorForIncompleteCcToolchain() throws Exception {
    mockToolsConfig.create(
        "incomplete_toolchain/BUILD",
        "toolchain(",
        "   name = 'incomplete_toolchain_cc-compiler-piii',",
        "   toolchain_type = '" + CPP_TOOLCHAIN_TYPE + "',",
        "   toolchain = ':incomplete_cc-compiler-piii',",
        "   target_compatible_with = ['//mock_platform:mock_value']",
        ")",
        "cc_toolchain(",
        "   name = 'incomplete_cc-compiler-piii',",
        "   cpu = 'piii',",
        "   compiler_files = 'compile-piii',",
        "   dwp_files = 'dwp-piii',",
        "   linker_files = 'link-piii',",
        "   strip_files = ':dummy_filegroup',",
        "   objcopy_files = 'objcopy-piii',",
        "   all_files = ':dummy_filegroup',",
        "   static_runtime_libs = ['static-runtime-libs-piii'],",
        "   dynamic_runtime_libs = ['dynamic-runtime-libs-piii'],",
        ")",
        "filegroup(name = 'dummy_filegroup')");

    useConfiguration(
        "--enabled_toolchain_types=" + CPP_TOOLCHAIN_TYPE,
        "--experimental_platforms=//mock_platform:mock-piii-platform",
        "--extra_toolchains=//incomplete_toolchain:incomplete_toolchain_cc-compiler-piii");

    AssertionError thrown =
        MoreAsserts.expectThrows(
            AssertionError.class,
            () ->
                ScratchAttributeWriter.fromLabelString(this, "cc_library", "//lib")
                    .setList("srcs", "a.cc")
                    .write());

    assertThat(thrown)
        .hasMessageThat()
        .contains("Using cc_toolchain target requires the attribute 'compiler' to be present");
  }
}
