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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactEndingWith;

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.util.ScratchAttributeWriter;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions.ObjcCrosstoolMode;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for the use of the OSX crosstool. */
@RunWith(JUnit4.class)
public class AppleToolchainSelectionTest extends ObjcRuleTestCase {

  @Override
  protected void useConfiguration(String... args) throws Exception {
   useConfiguration(ObjcCrosstoolMode.LIBRARY, args);
  }

  /**
   * Returns the given target in the configuration that it would be given by this
   * {@code BuildViewTestCase}'s {@code Transitions}, were the target a top-level target.
   */
  private ConfiguredTarget getTopLevelConfiguredTarget(ConfiguredTarget target)
      throws InterruptedException {
    BuildConfiguration topLevelConfig = getAppleCrosstoolConfiguration();
    return getConfiguredTarget(target.getLabel(), topLevelConfig);
  }

  /**
   * Returns the action that produces the artifact with the given label and suffix, in a output
   * directory consistent with that action being registered by a top-level target.
   */
  private CommandAction actionProducingArtifactForTopLevelTarget(String targetLabel,
      String artifactSuffix) throws Exception {
    ConfiguredTarget libraryTarget = getConfiguredTarget(targetLabel);
    ConfiguredTarget topLevelLibraryTarget = getTopLevelConfiguredTarget(libraryTarget);
    Label parsedLabel = Label.parseAbsolute(targetLabel);
    Artifact linkedLibrary = getBinArtifact(
        parsedLabel.getName() + artifactSuffix,
        topLevelLibraryTarget);
    return (CommandAction) getGeneratingAction(linkedLibrary);
  }
  
  @Test
  public void testToolchainSelectionDefault() throws Exception {
    createLibraryTargetWriter("//a:lib").write();
    CppConfiguration cppConfig =
        getAppleCrosstoolConfiguration().getFragment(CppConfiguration.class);
    
    assertThat(cppConfig.getCrosstoolTopPathFragment().toString())
        .isEqualTo("tools/osx/crosstool");
    assertThat(cppConfig.getToolchainIdentifier())
        .isEqualTo("ios_x86_64");
  }

  @Test
  public void testToolchainSelectionIosDevice() throws Exception {
    useConfiguration("--cpu=ios_armv7");
    createLibraryTargetWriter("//a:lib").write();
    CppConfiguration cppConfig =
        getAppleCrosstoolConfiguration().getFragment(CppConfiguration.class);
    
    assertThat(cppConfig.getCrosstoolTopPathFragment().toString())
        .isEqualTo("tools/osx/crosstool");
    assertThat(cppConfig.getToolchainIdentifier())
        .isEqualTo("ios_armv7");
  }
  
  @Test
  public void testToolchainSelectionCcDepDefault() throws Exception {
    useConfiguration("--experimental_disable_jvm");
    ScratchAttributeWriter
        .fromLabelString(this, "cc_library", "//b:lib")
        .setList("srcs", "b.cc")
        .write();
    createBinaryTargetWriter("//a:bin")
        .setList("srcs", "a.m")
        .setList("deps", "//b:lib")
        .write();

    CommandAction linkAction = actionProducingArtifactForTopLevelTarget("//a:bin", "_bin");
    Artifact ccArchive = getFirstArtifactEndingWith(linkAction.getInputs(), "liblib.a");
    CommandAction ccArchiveAction = (CommandAction) getGeneratingAction(ccArchive);
    Artifact ccObjectFile = getFirstArtifactEndingWith(ccArchiveAction.getInputs(), ".o");
    CommandAction ccCompileAction = (CommandAction) getGeneratingAction(ccObjectFile);
    assertThat(ccCompileAction.getArguments()).contains("tools/osx/crosstool/iossim/wrapped_clang");
  }
  
  @Test
  public void testToolchainSelectionCcDepDevice() throws Exception {
    useConfiguration("--cpu=ios_armv7");
    ScratchAttributeWriter
        .fromLabelString(this, "cc_library", "//b:lib")
        .setList("srcs", "b.cc")
        .write();
    createBinaryTargetWriter("//a:bin")
        .setList("srcs", "a.m")
        .setList("deps", "//b:lib")
        .write();
    CommandAction linkAction = actionProducingArtifactForTopLevelTarget("//a:bin", "_bin");
    Artifact ccArchive = getFirstArtifactEndingWith(linkAction.getInputs(), "liblib.a");
    CommandAction ccArchiveAction = (CommandAction) getGeneratingAction(ccArchive);
    Artifact ccObjectFile = getFirstArtifactEndingWith(ccArchiveAction.getInputs(), ".o");
    CommandAction ccCompileAction = (CommandAction) getGeneratingAction(ccObjectFile);
    assertThat(ccCompileAction.getArguments()).contains("tools/osx/crosstool/ios/wrapped_clang");
  }

  @Test
  public void testToolchainSelectionMultiArchIos() throws Exception {
    useConfiguration(
        "--experimental_disable_jvm",
        "--ios_multi_cpus=armv7,arm64");
    ScratchAttributeWriter
        .fromLabelString(this, "cc_library", "//b:lib")
        .setList("srcs", "a.cc")
        .write(getAppleCrosstoolConfiguration());
    ScratchAttributeWriter
        .fromLabelString(this, "apple_binary", "//a:bin")
        .set("platform_type", "'ios'")
        .setList("deps", "//b:lib")
        .write();
    Action lipoAction = actionProducingArtifact("//a:bin", "_lipobin");
    String armv64Bin =
        configurationBin("arm64", ConfigurationDistinguisher.APPLEBIN_IOS,
            DEFAULT_IOS_SDK_VERSION)
        + "a/bin_bin";
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), armv64Bin);
    CommandAction linkAction = getGeneratingSpawnAction(binArtifact);
    CppLinkAction objcLibArchiveAction = (CppLinkAction) getGeneratingAction(
        getFirstArtifactEndingWith(linkAction.getInputs(), "liblib.a"));
    assertThat(Joiner.on(" ").join(objcLibArchiveAction.getArguments())).contains("ios_arm64");
  }
  
  @Test
  public void testToolchainSelectionMultiArchWatchos() throws Exception {
    useConfiguration(
        "--experimental_disable_jvm",
        "--ios_multi_cpus=armv7,arm64",
        "--watchos_cpus=armv7k");
    ScratchAttributeWriter
        .fromLabelString(this, "cc_library", "//b:lib")
        .setList("srcs", "a.cc")
        .write(getAppleCrosstoolConfiguration());
    ScratchAttributeWriter
        .fromLabelString(this, "apple_binary", "//a:bin")
        .setList("deps", "//b:lib")
        .set("platform_type", "\"watchos\"")
        .write();

    CommandAction linkAction = linkAction("//a:bin");
    CppLinkAction objcLibCompileAction = (CppLinkAction) getGeneratingAction(
        getFirstArtifactEndingWith(linkAction.getInputs(), "liblib.a"));
    assertThat(Joiner.on(" ").join(objcLibCompileAction.getArguments())).contains("watchos_armv7k");
  }  
}
