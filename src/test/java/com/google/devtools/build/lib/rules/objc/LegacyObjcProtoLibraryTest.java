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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions.ObjcCrosstoolMode;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Legacy test: These tests test --experimental_objc_crosstool=off. See README.
 */
@RunWith(JUnit4.class)
public class LegacyObjcProtoLibraryTest extends ObjcProtoLibraryTest {
  @Override
  protected ObjcCrosstoolMode getObjcCrosstoolMode() {
    return ObjcCrosstoolMode.OFF;
  }

  @Test
  public void testCompilationAction() throws Exception {
    useConfiguration("--cpu=ios_i386");
    ApplePlatform platform = ApplePlatform.IOS_SIMULATOR;

    ConfiguredTarget target = getConfiguredTarget("//package:opl");
    CommandAction linkAction =
        (CommandAction) getGeneratingAction(getBinArtifact("libopl.a", target));

    CommandAction compileAction =
        (CommandAction)
            getGeneratingAction(
                ActionsTestUtil.getFirstArtifactEndingWith(linkAction.getInputs(), "/FileA.pb.o"));

    Artifact sourceFile =
        ActionsTestUtil.getFirstArtifactEndingWith(
            getFilesToBuild(getConfiguredTarget("//package:opl")), "/FileA.pb.m");

    Artifact objectFile =
        ActionsTestUtil.getFirstArtifactEndingWith(compileAction.getOutputs(), ".o");
    Artifact dotdFile =
        ActionsTestUtil.getFirstArtifactEndingWith(compileAction.getOutputs(), ".d");
    assertThat(compileAction.getArguments())
        .containsExactlyElementsIn(
            new ImmutableList.Builder<String>()
                .add(MOCK_XCRUNWRAPPER_PATH)
                .add(ObjcRuleClasses.CLANG)
                .addAll(AppleToolchain.DEFAULT_WARNINGS.values())
                .add("-fexceptions")
                .add("-fasm-blocks")
                .add("-fobjc-abi-version=2")
                .add("-fobjc-legacy-dispatch")
                .addAll(CompilationSupport.DEFAULT_COMPILER_FLAGS)
                .add("-mios-simulator-version-min=" + DEFAULT_IOS_SDK_VERSION)
                .add("-arch", "i386")
                .add("-isysroot", AppleToolchain.sdkDir())
                .add("-F", AppleToolchain.sdkDir() + AppleToolchain.DEVELOPER_FRAMEWORK_PATH)
                .add("-F", frameworkDir(platform))
                .addAll(FASTBUILD_COPTS)
                .addAll(
                    LegacyObjcLibraryTest.iquoteArgs(
                        target.getProvider(ObjcProvider.class), getTargetConfiguration()))
                .add("-I")
                .add(sourceFile.getExecPath().getParentDirectory().getParentDirectory().toString())
                .add("-fno-objc-arc")
                .add("-c", sourceFile.getExecPathString())
                .add("-o")
                .add(objectFile.getExecPathString())
                .add("-MD")
                .add("-MF")
                .add(dotdFile.getExecPathString())
                .build())
        .inOrder();
    assertRequiresDarwin(compileAction);
    assertThat(Artifact.toRootRelativePaths(compileAction.getInputs())).containsAllOf(
        "package/_generated_protos/opl/package/FileA.pb.m",
        "package/_generated_protos/opl/package/FileA.pb.h",
        "package/_generated_protos/opl/package/dir/FileB.pb.h",
        "package/_generated_protos/opl/dep/File.pb.h"
    );
  }

  @Test
  public void testLibraryLinkAction() throws Exception {
    useConfiguration("--cpu=ios_armv7");
    Artifact libFile =
        ActionsTestUtil.getFirstArtifactEndingWith(
            getFilesToBuild(getConfiguredTarget("//package:opl")), "/libopl.a");
    CommandAction action = (CommandAction) getGeneratingAction(libFile);
    assertThat(action.getArguments())
        .isEqualTo(
            ImmutableList.of(
                MOCK_LIBTOOL_PATH,
                "-static",
                "-filelist",
                getBinArtifact("opl-archive.objlist", "//package:opl").getExecPathString(),
                "-arch_only",
                "armv7",
                "-syslibroot",
                AppleToolchain.sdkDir(),
                "-o",
                libFile.getExecPathString()));
    assertRequiresDarwin(action);
  }
}
