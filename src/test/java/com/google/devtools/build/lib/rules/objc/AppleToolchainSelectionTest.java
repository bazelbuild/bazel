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
import com.google.devtools.build.lib.analysis.util.ScratchAttributeWriter;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for the use of the OSX crosstool. */
@RunWith(JUnit4.class)
public class AppleToolchainSelectionTest extends ObjcRuleTestCase {

  @Test
  public void testToolchainSelectionDefault() throws Exception {
    createLibraryTargetWriter("//a:lib").write();
    CppConfiguration cppConfig =
        getAppleCrosstoolConfiguration().getFragment(CppConfiguration.class);

    assertThat(cppConfig.getRuleProvidingCcToolchainProvider().toString())
        .isEqualTo("//tools/osx/crosstool:crosstool");
    assertThat(cppConfig.getTransformedCpuFromOptions()).isEqualTo("darwin_x86_64");
  }

  @Test
  public void testToolchainSelectionIosDevice() throws Exception {
    useConfiguration("--apple_platform_type=ios", "--cpu=ios_armv7");
    createLibraryTargetWriter("//a:lib").write();
    CppConfiguration cppConfig =
        getAppleCrosstoolConfiguration().getFragment(CppConfiguration.class);

    assertThat(cppConfig.getRuleProvidingCcToolchainProvider().toString())
        .isEqualTo("//tools/osx/crosstool:crosstool");
    assertThat(cppConfig.getTransformedCpuFromOptions()).isEqualTo("ios_armv7");
  }

  @Test
  public void testToolchainSelectionCcDepDefault() throws Exception {
    ScratchAttributeWriter
        .fromLabelString(this, "cc_library", "//b:lib")
        .setList("srcs", "b.cc")
        .write();
    createBinaryTargetWriter("//a:bin")
        .setList("deps", "//b:lib")
        .write();

    Action lipoAction = actionProducingArtifact("//a:bin", "_lipobin");
    String x8664Bin =
        configurationBin("x86_64", ConfigurationDistinguisher.APPLEBIN_IOS) + "a/bin_bin";
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), x8664Bin);
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    CppLinkAction ccArchiveAction =
        (CppLinkAction)
            getGeneratingAction(getFirstArtifactEndingWith(linkAction.getInputs(), "liblib.a"));
    Artifact ccObjectFile = getFirstArtifactEndingWith(ccArchiveAction.getInputs(), ".o");
    CommandAction ccCompileAction = (CommandAction) getGeneratingAction(ccObjectFile);
    assertThat(ccCompileAction.getArguments()).contains("tools/osx/crosstool/iossim/wrapped_clang");
  }

  @Test
  public void testToolchainSelectionCcDepDevice() throws Exception {
    useConfiguration("--apple_platform_type=ios", "--cpu=ios_armv7");
    ScratchAttributeWriter
        .fromLabelString(this, "cc_library", "//b:lib")
        .setList("srcs", "b.cc")
        .write();
    createBinaryTargetWriter("//a:bin")
        .setList("deps", "//b:lib")
        .write();
    Action lipoAction = actionProducingArtifact("//a:bin", "_lipobin");
    String armv7Bin =
        configurationBin("armv7", ConfigurationDistinguisher.APPLEBIN_IOS) + "a/bin_bin";
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), armv7Bin);
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    CppLinkAction ccArchiveAction =
        (CppLinkAction)
            getGeneratingAction(getFirstArtifactEndingWith(linkAction.getInputs(), "liblib.a"));
    Artifact ccObjectFile = getFirstArtifactEndingWith(ccArchiveAction.getInputs(), ".o");
    CommandAction ccCompileAction = (CommandAction) getGeneratingAction(ccObjectFile);
    assertThat(ccCompileAction.getArguments()).contains("tools/osx/crosstool/ios/wrapped_clang");
  }

  @Test
  public void testToolchainSelectionMultiArchIos() throws Exception {
    useConfiguration("--ios_multi_cpus=armv7,arm64");
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
        configurationBin("arm64", ConfigurationDistinguisher.APPLEBIN_IOS) + "a/bin_bin";
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), armv64Bin);
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    CppLinkAction objcLibArchiveAction = (CppLinkAction) getGeneratingAction(
        getFirstArtifactEndingWith(linkAction.getInputs(), "liblib.a"));
    assertThat(Joiner.on(" ").join(objcLibArchiveAction.getArguments())).contains("ios_arm64");
  }

  @Test
  public void testToolchainSelectionMultiArchWatchos() throws Exception {
    useConfiguration(
        "--ios_multi_cpus=armv7,arm64",
        "--watchos_cpus=armv7k,arm64_32");
    ScratchAttributeWriter
        .fromLabelString(this, "cc_library", "//b:lib")
        .setList("srcs", "a.cc")
        .write(getAppleCrosstoolConfiguration());
    ScratchAttributeWriter
        .fromLabelString(this, "apple_binary", "//a:bin")
        .setList("deps", "//b:lib")
        .set("platform_type", "\"watchos\"")
        .write();

    Action lipoAction = actionProducingArtifact("//a:bin", "_lipobin");
	{
	    String arm64_32Bin =
	        configurationBin("arm64_32", ConfigurationDistinguisher.APPLEBIN_WATCHOS) + "a/bin_bin";
	    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), arm64_32Bin);
	    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
	    CppLinkAction objcLibArchiveAction = (CppLinkAction) getGeneratingAction(
	        getFirstArtifactEndingWith(linkAction.getInputs(), "liblib.a"));
	    assertThat(Joiner.on(" ").join(objcLibArchiveAction.getArguments())).contains("watchos_arm64_32");
	}
	{
	    String armv7kBin =
	        configurationBin("armv7k", ConfigurationDistinguisher.APPLEBIN_WATCHOS) + "a/bin_bin";
	    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), armv7kBin);
	    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
	    CppLinkAction objcLibArchiveAction = (CppLinkAction) getGeneratingAction(
	        getFirstArtifactEndingWith(linkAction.getInputs(), "liblib.a"));
	    assertThat(Joiner.on(" ").join(objcLibArchiveAction.getArguments())).contains("watchos_armv7k");
	}
  }
}
