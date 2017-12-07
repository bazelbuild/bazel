// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.analysis.ToolchainContext.ResolvedToolchainProviders;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.ScratchAttributeWriter;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.util.MockPlatformSupport;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for the {@code toolchain_type} rule. */
@RunWith(JUnit4.class)
public class ToolchainTypeTest extends BuildViewTestCase {

  @Test
  public void testSmoke() throws Exception {
    ConfiguredTarget cc =
        getConfiguredTarget(TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_type");
    assertThat(cc.get(TemplateVariableInfo.PROVIDER).getVariables())
        .containsKey("TARGET_CPU");
  }

  @Test
  public void testCcToolchainDoesNotProvideJavaMakeVariables() throws Exception {
    ConfiguredTarget cc =
        getConfiguredTarget(TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_type");
    assertThat(cc.get(TemplateVariableInfo.PROVIDER).getVariables()).doesNotContainKey("JAVABASE");
  }

  @Test
  public void testMakeVariablesFromToolchain() throws Exception {
    MockPlatformSupport.addMockPiiiPlatform(
        mockToolsConfig, analysisMock.ccSupport().getMockCrosstoolLabel());
    useConfiguration(
        "--enabled_toolchain_types="
            + TestConstants.TOOLS_REPOSITORY
            + "//tools/cpp:toolchain_type",
        "--experimental_platforms=//mock_platform:mock-piii-platform",
        "--extra_toolchains=//mock_platform:toolchain_cc-compiler-piii",
        "--make_variables_source=toolchain");
    ConfiguredTarget cc =
        getConfiguredTarget(TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_type");
    assertThat(cc.get(TemplateVariableInfo.PROVIDER).getVariables())
        .containsEntry("TARGET_CPU", "piii");
  }

  @Test
  public void testGlibcVersionSetInEnv() throws Exception {
    MockPlatformSupport.addMockPiiiPlatform(
        mockToolsConfig, analysisMock.ccSupport().getMockCrosstoolLabel());
    useConfiguration(
        "--enabled_toolchain_types="
            + TestConstants.TOOLS_REPOSITORY
            + "//tools/cpp:toolchain_type",
        "--experimental_platforms=//mock_platform:mock-piii-platform",
        "--extra_toolchains=//mock_platform:toolchain_cc-compiler-piii",
        "--make_variables_source=toolchain");
    ConfiguredTarget toolchainType =
        getConfiguredTarget(TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_type");
    Map<String, String> makeVariables =
        toolchainType.get(TemplateVariableInfo.PROVIDER).getVariables();

    ConfiguredTarget target =
        ScratchAttributeWriter.fromLabelString(this, "cc_library", "//lib")
            .setList("srcs", "a.cc")
            .write();

    ResolvedToolchainProviders providers =
        (ResolvedToolchainProviders)
            getRuleContext(target).getToolchainContext().getResolvedToolchainProviders();
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider)
            providers.getForToolchainType(
                Label.parseAbsolute(TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_type"));

    String targetLibc = toolchainProvider.getTargetLibc();
    String glibcVersion = makeVariables.get("GLIBC_VERSION");
    assertThat(glibcVersion).isNotNull();
    if (targetLibc.startsWith("glibc-")) {
      assertThat(glibcVersion).isEqualTo(targetLibc.substring("glibc-".length()));
    } else {
      assertThat(glibcVersion).isEqualTo(targetLibc);
    }
  }

  @Test
  public void testCcTargetsDependOnCcToolchainAutomatically() throws Exception {
    MockPlatformSupport.addMockPiiiPlatform(
        mockToolsConfig, analysisMock.ccSupport().getMockCrosstoolLabel());
    useConfiguration(
        "--enabled_toolchain_types="
            + TestConstants.TOOLS_REPOSITORY
            + "//tools/cpp:toolchain_type",
        "--experimental_platforms=//mock_platform:mock-piii-platform",
        "--extra_toolchains=//mock_platform:toolchain_cc-compiler-piii",
        "--make_variables_source=toolchain");

    // for cc_library, cc_binary, and cc_test, we check that $(TARGET_CPU) is a valid Make variable
    ConfiguredTarget cclibrary =
        ScratchAttributeWriter.fromLabelString(this, "cc_library", "//cclib")
            .setList("srcs", "a.cc")
            .setList("copts", "foobar$(TARGET_CPU)")
            .write();
    CppCompileAction compileAction =
        (CppCompileAction) getGeneratingAction(getBinArtifact("_objs/cclib/cclib/a.o", cclibrary));
    assertThat(compileAction.getArgv()).contains("foobarpiii");

    ConfiguredTarget ccbinary =
        ScratchAttributeWriter.fromLabelString(this, "cc_binary", "//ccbin")
            .setList("srcs", "a.cc")
            .setList("copts", "foobar$(TARGET_CPU)")
            .write();
    compileAction =
        (CppCompileAction) getGeneratingAction(getBinArtifact("_objs/ccbin/ccbin/a.o", ccbinary));
    assertThat(compileAction.getArgv()).contains("foobarpiii");

    ConfiguredTarget cctest =
        ScratchAttributeWriter.fromLabelString(this, "cc_test", "//cctest")
            .setList("srcs", "a.cc")
            .setList("copts", "foobar$(TARGET_CPU)")
            .write();
    compileAction =
        (CppCompileAction) getGeneratingAction(getBinArtifact("_objs/cctest/cctest/a.o", cctest));
    assertThat(compileAction.getArgv()).contains("foobarpiii");
  }
}
