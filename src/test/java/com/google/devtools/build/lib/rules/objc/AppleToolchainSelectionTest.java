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
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for the use of the OSX crosstool. */
@RunWith(JUnit4.class)
public class AppleToolchainSelectionTest extends ObjcRuleTestCase {

  @Test
  public void testToolchainSelectionCcDepDefault() throws Exception {
    ScratchAttributeWriter
        .fromLabelString(this, "cc_library", "//b:lib")
        .setList("srcs", "b.cc")
        .write();
    addAppleBinaryStarlarkRule(scratch);
    scratch.file(
        "a/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    platform_type = 'ios',",
        "    deps = ['//b:lib'],",
        ")");
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
    addAppleBinaryStarlarkRule(scratch);
    scratch.file(
        "a/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    platform_type = 'ios',",
        "    deps = ['//b:lib'],",
        ")");
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
    ScratchAttributeWriter.fromLabelString(this, "cc_library", "//b:lib")
        .setList("srcs", "a.cc")
        .write();
    addAppleBinaryStarlarkRule(scratch);
    scratch.file(
        "a/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    platform_type = 'ios',",
        "    deps = ['//b:lib'],",
        ")");
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
        "--watchos_cpus=armv7k");
    ScratchAttributeWriter.fromLabelString(this, "cc_library", "//b:lib")
        .setList("srcs", "a.cc")
        .write();
    addAppleBinaryStarlarkRule(scratch);
    scratch.file(
        "a/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    platform_type = 'watchos',",
        "    deps = ['//b:lib'],",
        ")");

    CommandAction linkAction = linkAction("//a:bin");
    CppLinkAction objcLibCompileAction = (CppLinkAction) getGeneratingAction(
        getFirstArtifactEndingWith(linkAction.getInputs(), "liblib.a"));
    assertThat(Joiner.on(" ").join(objcLibCompileAction.getArguments())).contains("watchos_armv7k");
  }
}
