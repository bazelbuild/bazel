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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.cpp.AppleCcToolchain;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.cpp.Link;
import com.google.devtools.build.lib.rules.cpp.LinkBuildVariablesTestCase;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests that {@code CppLinkAction} is populated with the correct build variables for objective C
 * builds.
 */
@RunWith(JUnit4.class)
public class ObjcBuildVariablesTest extends LinkBuildVariablesTestCase {

  @Before
  public void createFooFooCcLibraryForRuleContext() throws IOException {
    scratch.file("foo/BUILD", "cc_library(name = 'foo')");
  }

  private RuleContext getRuleContext() throws Exception {
    return getRuleContext(getConfiguredTarget("//foo:foo"));
  }

  @Override
  public void initializeMockClient() throws IOException {
    super.initializeMockClient();
    MockObjcSupport.setup(mockToolsConfig);
  }

  @Override
  protected void useConfiguration(String... args) throws Exception {
    ImmutableList<String> extraArgs = ImmutableList.<String>builder()
        .add("--xcode_version_config=" + MockObjcSupport.XCODE_VERSION_CONFIG)
        .add("--apple_crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL)
        .add("--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL)
        .addAll(ImmutableList.copyOf(args))
        .build();

    super.useConfiguration(extraArgs.toArray(new String[extraArgs.size()]));
  }

  @Test
  public void testAppleBuildVariablesIos() throws Exception {
    MockObjcSupport.setup(mockToolsConfig);
    useConfiguration(
        "--crosstool_top=//tools/osx/crosstool", "--xcode_version=5.8",
        "--ios_minimum_os=12.345", "--watchos_minimum_os=11.111",
        "--cpu=ios_x86_64", "--apple_platform_type=ios");
     scratch.file(
        "x/BUILD",
        "cc_binary(",
        "   name = 'bin',",
        "   srcs = ['a.cc'],",
        ")");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:bin");
    CcToolchainVariables variables = getLinkBuildVariables(target, Link.LinkTargetType.EXECUTABLE);
    assertThat(
            getVariableValue(
                getRuleContext(), variables, AppleCcToolchain.XCODE_VERSION_OVERRIDE_VALUE_KEY))
        .contains("5.8");
    assertThat(
            getVariableValue(
                getRuleContext(), variables, AppleCcToolchain.APPLE_SDK_VERSION_OVERRIDE_VALUE_KEY))
        .contains("8.4");
    assertThat(
            getVariableValue(
                getRuleContext(), variables, AppleCcToolchain.APPLE_SDK_PLATFORM_VALUE_KEY))
        .contains("iPhoneSimulator");
    assertThat(getVariableValue(getRuleContext(), variables, AppleCcToolchain.VERSION_MIN_KEY))
        .contains("12.345");
  }

  @Test
  public void testAppleBuildVariablesWatchos() throws Exception {
    String dummyMinimumOsValue = "11.111";
    useConfiguration(
        "--crosstool_top=//tools/osx/crosstool", "--xcode_version=5.8",
        "--ios_minimum_os=12.345", "--watchos_minimum_os=" + dummyMinimumOsValue,
        "--watchos_cpus=armv7k");

    scratch.file(
        "x/BUILD",
        "apple_binary(",
        "   name = 'bin',",
        "   deps = [':a'],",
        "   platform_type = 'watchos',",
        ")",
        "cc_library(",
        "   name = 'a',",
        "   srcs = ['a.cc'],",
        ")");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:bin");
    // In order to get the set of variables that apply to the c++
    // actions, follow the chain of actions starting at the lipobin
    // creation.
    Artifact lipoBin =
        getBinArtifact(
            Label.parseAbsolute("//x:bin", ImmutableMap.of()).getName() + "_lipobin", target);
    Action lipoAction = getGeneratingAction(lipoBin);
    Artifact bin = ActionsTestUtil.getFirstArtifactEndingWith(lipoAction.getInputs(), "_bin");
    CommandAction appleBinLinkAction = (CommandAction) getGeneratingAction(bin);
    Artifact archive =
        ActionsTestUtil.getFirstArtifactEndingWith(appleBinLinkAction.getInputs(), "liba.a");
    CppLinkAction ccArchiveAction = (CppLinkAction) getGeneratingAction(archive);

    CcToolchainVariables variables = ccArchiveAction.getLinkCommandLine().getBuildVariables();
    assertThat(
            getVariableValue(
                getRuleContext(), variables, AppleCcToolchain.XCODE_VERSION_OVERRIDE_VALUE_KEY))
        .contains("5.8");
    assertThat(
            getVariableValue(
                getRuleContext(), variables, AppleCcToolchain.APPLE_SDK_VERSION_OVERRIDE_VALUE_KEY))
        .contains("2.0");
    assertThat(
            getVariableValue(
                getRuleContext(), variables, AppleCcToolchain.APPLE_SDK_PLATFORM_VALUE_KEY))
        .contains("WatchOS");
    assertThat(getVariableValue(getRuleContext(), variables, AppleCcToolchain.VERSION_MIN_KEY))
        .contains(dummyMinimumOsValue);
  }

  @Test
  public void testAppleBuildVariablesMacos() throws Exception {
    MockObjcSupport.setup(mockToolsConfig);
    String dummyMinimumOsValue = "13.579";
    useConfiguration(
        "--crosstool_top=//tools/osx/crosstool",
        "--cpu=darwin_x86_64",
        "--macos_minimum_os=" + dummyMinimumOsValue);
    scratch.file(
        "x/BUILD",
        "apple_binary(",
        "   name = 'bin',",
        "   deps = [':a'],",
        "   platform_type = 'macos',",
        ")",
        "cc_library(",
        "   name = 'a',",
        "   srcs = ['a.cc'],",
        ")");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:bin");
    // In order to get the set of variables that apply to the c++ actions, follow the chain of
    // actions starting at the lipobin creation.
    Artifact lipoBin =
        getBinArtifact(
            Label.parseAbsolute("//x:bin", ImmutableMap.of()).getName() + "_lipobin", target);
    Action lipoAction = getGeneratingAction(lipoBin);
    Artifact bin = ActionsTestUtil.getFirstArtifactEndingWith(lipoAction.getInputs(), "_bin");
    CommandAction appleBinLinkAction = (CommandAction) getGeneratingAction(bin);
    Artifact archive =
        ActionsTestUtil.getFirstArtifactEndingWith(appleBinLinkAction.getInputs(), "liba.a");
    CppLinkAction ccArchiveAction = (CppLinkAction) getGeneratingAction(archive);

    CcToolchainVariables variables = ccArchiveAction.getLinkCommandLine().getBuildVariables();
    assertThat(getVariableValue(getRuleContext(), variables, AppleCcToolchain.VERSION_MIN_KEY))
        .contains(dummyMinimumOsValue);
  }

  @Test
  public void testDefaultBuildVariablesIos() throws Exception {
     MockObjcSupport.setup(mockToolsConfig);
    useConfiguration(
        "--apple_platform_type=ios", "--crosstool_top=//tools/osx/crosstool", "--cpu=ios_x86_64");
     scratch.file(
        "x/BUILD",
        "cc_binary(",
        "   name = 'bin',",
        "   srcs = ['a.cc'],",
        ")");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:bin");
    CcToolchainVariables variables = getLinkBuildVariables(target, Link.LinkTargetType.EXECUTABLE);
    assertThat(
            getVariableValue(
                getRuleContext(), variables, AppleCcToolchain.XCODE_VERSION_OVERRIDE_VALUE_KEY))
        .contains(MockObjcSupport.DEFAULT_XCODE_VERSION);
    assertThat(
            getVariableValue(
                getRuleContext(), variables, AppleCcToolchain.APPLE_SDK_VERSION_OVERRIDE_VALUE_KEY))
        .contains(MockObjcSupport.DEFAULT_IOS_SDK_VERSION);
    assertThat(getVariableValue(getRuleContext(), variables, AppleCcToolchain.VERSION_MIN_KEY))
        .contains(AppleCommandLineOptions.DEFAULT_IOS_SDK_VERSION);
  }

  @Test
  public void testMinimumOsAttributeBuildVariable() throws Exception {
    MockObjcSupport.setup(mockToolsConfig);
    String dummyMinimumOsValue = "13.579";
    useConfiguration(
        "--crosstool_top=//tools/osx/crosstool",
        "--cpu=darwin_x86_64");
    scratch.file(
        "x/BUILD",
        "apple_binary(",
        "   name = 'bin',",
        "   deps = [':a'],",
        "   platform_type = 'ios',",
        "   minimum_os_version = '" + dummyMinimumOsValue + "',",
        ")",
        "cc_library(",
        "   name = 'a',",
        "   srcs = ['a.cc'],",
        ")");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:bin");
    // In order to get the set of variables that apply to the c++ actions, follow the chain of
    // actions starting at the lipobin creation.
    Artifact lipoBin =
        getBinArtifact(
            Label.parseAbsolute("//x:bin", ImmutableMap.of()).getName() + "_lipobin", target);
    Action lipoAction = getGeneratingAction(lipoBin);
    Artifact bin = ActionsTestUtil.getFirstArtifactEndingWith(lipoAction.getInputs(), "_bin");
    CommandAction appleBinLinkAction = (CommandAction) getGeneratingAction(bin);
    Artifact archive =
        ActionsTestUtil.getFirstArtifactEndingWith(appleBinLinkAction.getInputs(), "liba.a");
    CppLinkAction ccArchiveAction = (CppLinkAction) getGeneratingAction(archive);

    CcToolchainVariables variables = ccArchiveAction.getLinkCommandLine().getBuildVariables();
    assertThat(getVariableValue(getRuleContext(), variables, AppleCcToolchain.VERSION_MIN_KEY))
        .contains(dummyMinimumOsValue);
  }
}
