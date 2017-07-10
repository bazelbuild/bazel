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
import static com.google.devtools.build.lib.rules.objc.LegacyCompilationSupport.AUTOMATIC_SDK_FRAMEWORKS;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.ScratchAttributeWriter;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions.ObjcCrosstoolMode;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for linking targets with the OSX crosstool. */
@RunWith(JUnit4.class)
public class ExperimentalObjcBinaryTest extends ObjcRuleTestCase {
  static final RuleType RULE_TYPE = new BinaryRuleType("objc_binary");

  private static final String WRAPPED_CLANG = "wrapped_clang";
  private static final String WRAPPED_CLANGPLUSPLUS = "wrapped_clang++";

  private ConfiguredTarget addMockBinAndLibs(List<String> srcs) throws Exception {
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();

    return createBinaryTargetWriter("//bin:bin")
        .setAndCreateFiles("srcs", srcs)
        .setList("deps", "//lib1:lib1", "//lib2:lib2")
        .write();
  }

  private ImmutableList<String> automaticSdkFrameworks() {
    ImmutableList.Builder<String> result = ImmutableList.<String>builder();
    for (SdkFramework framework : AUTOMATIC_SDK_FRAMEWORKS) {
      result.add("-framework " + framework.getName());
    }
    return result.build();
  }

  @Test
  public void testDeviceBuild() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--experimental_objc_crosstool=all",
        "--experimental_disable_go",
        "--cpu=ios_armv7",
        "--ios_cpu=armv7");
    ApplePlatform platform = ApplePlatform.IOS_DEVICE;

    addMockBinAndLibs(ImmutableList.of("a.m"));
    CommandAction action = linkAction("//bin:bin");
    assertRequiresDarwin(action);
    assertThat(Artifact.toRootRelativePaths(action.getInputs()))
        .containsExactly(
            "bin/libbin.a",
            "lib1/liblib1.a",
            "lib2/liblib2.a",
            "bin/bin-linker.objlist");
    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly("bin/bin_bin");
    verifyObjlist(
        action,
        "bin-linker.objlist",
        execPathEndingWith(action.getInputs(), "libbin.a"),
        execPathEndingWith(action.getInputs(), "liblib1.a"),
        execPathEndingWith(action.getInputs(), "liblib2.a"));
    assertThat(action.getArguments())
        .containsExactlyElementsIn(
            new ImmutableList.Builder<String>()
                .add("tools/osx/crosstool/ios/" + WRAPPED_CLANG)
                .add("-F" + AppleToolchain.sdkDir() + AppleToolchain.DEVELOPER_FRAMEWORK_PATH)
                .add("-F" + frameworkDir(platform))
                .add("-isysroot")
                .add(AppleToolchain.sdkDir())
                // TODO(b/35853671): Factor out "-lc++"
                .add("-lc++")
                .add("-target", "armv7-apple-ios")
                .add("-miphoneos-version-min=" + DEFAULT_IOS_SDK_VERSION)
                .addAll(automaticSdkFrameworks())
                .add("-arch armv7")
                .add("-Xlinker", "-objc_abi_version", "-Xlinker", "2")
                .add("-Xlinker", "-rpath", "-Xlinker", "@executable_path/Frameworks")
                .add("-fobjc-link-runtime")
                .add("-ObjC")
                .add("-filelist " + execPathEndingWith(action.getInputs(), "bin-linker.objlist"))
                .add("-o " + Iterables.getOnlyElement(Artifact.toExecPaths(action.getOutputs())))
                .build())
        .inOrder();
  }

  @Test
  public void testSimulatorBuild() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--experimental_objc_crosstool=all",
        "--experimental_disable_go",
        "--cpu=ios_x86_64",
        "--ios_cpu=x86_64");
    ApplePlatform platform = ApplePlatform.IOS_SIMULATOR;

    addMockBinAndLibs(ImmutableList.of("a.m"));
    CommandAction action = linkAction("//bin:bin");
    assertThat(action.getArguments())
        .containsAllOf(
            "tools/osx/crosstool/iossim/" + WRAPPED_CLANG,
            "-arch x86_64",
            "-mios-simulator-version-min=" + DEFAULT_IOS_SDK_VERSION,
            "-F" + frameworkDir(platform));
  }

  @Test
  public void testAlwaysLinkCcDependenciesAreForceLoaded() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--experimental_objc_crosstool=all",
        "--experimental_disable_go",
        "--cpu=ios_armv7",
        "--ios_cpu=armv7");

    scratch.file(
        "bin/BUILD",
        "cc_library(",
        "    name = 'cclib1',",
        "    srcs = ['dep1.c'],",
        "    alwayslink = 1,",
        ")",
        "cc_library(",
        "    name = 'cclib2',",
        "    srcs = ['dep2.c'],",
        "    deps = [':cclib1'],",
        ")",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        "    deps = [':cclib2'],",
        ")");

    // cclib1 is force loaded.
    assertThat(Joiner.on(" ").join(linkAction("//bin").getArguments()))
        .containsMatch(Pattern.compile(" -force_load [^\\s]+/libcclib1.lo\\b"));
  }

  @Test
  public void testObjcPlusPlusLinkAction() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--experimental_objc_crosstool=all",
        "--experimental_disable_go",
        "--cpu=ios_armv7",
        "--ios_cpu=armv7");

    addMockBinAndLibs(ImmutableList.of("a.mm"));

    CommandAction action = linkAction("//bin:bin");
    assertThat(action.getArguments())
        .containsAllOf(
            "tools/osx/crosstool/ios/" + WRAPPED_CLANGPLUSPLUS,
            "-stdlib=libc++",
            "-std=gnu++11");
  }

  @Test
  public void testUnstrippedArtifactGeneratedForBinaryStripping() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--experimental_objc_crosstool=all",
        "--experimental_disable_go",
        "--objc_enable_binary_stripping",
        "--compilation_mode=opt");
    addMockBinAndLibs(ImmutableList.of("a.m"));
    Action linkAction = actionProducingArtifact("//bin:bin", "_bin_unstripped");
    Action stripAction = actionProducingArtifact("//bin:bin", "_bin");
    assertThat(linkAction).isNotNull();
    assertThat(stripAction).isNotNull();
  }


  @Test
  public void testDeadStripLinkArguments() throws Exception {
     useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--experimental_objc_crosstool=all",
        "--experimental_disable_go",
        "--cpu=ios_armv7",
        "--ios_cpu=armv7",
        "--objc_enable_binary_stripping",
        "--compilation_mode=opt");
    addMockBinAndLibs(ImmutableList.of("a.mm"));
    CommandAction linkAction =
        (CommandAction) actionProducingArtifact("//bin:bin", "_bin_unstripped");
    assertThat(linkAction.getArguments())
        .containsAllOf("-dead_strip", "-no_dead_strip_inits_and_terms");
  }

  @Test
  public void testDeadStripLinkArgumentsNotPresentIfStrippingNotEnabled() throws Exception {
     useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--experimental_objc_crosstool=all",
        "--experimental_disable_go",
        "--cpu=ios_armv7",
        "--ios_cpu=armv7",
        "--compilation_mode=opt");
    addMockBinAndLibs(ImmutableList.of("a.mm"));
    CommandAction linkAction =
        (CommandAction) actionProducingArtifact("//bin:bin", "_bin");
    assertThat(linkAction.getArguments())
        .containsNoneOf("--dead_strip", "--no_dead_strip_inits_and_terms");
  }

  @Test
  public void testDeadStripLinkArgumentsNotPresentIfCompilationModeFastbuild() throws Exception {
     useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--experimental_objc_crosstool=all",
        "--experimental_disable_go",
        "--cpu=ios_armv7",
        "--ios_cpu=armv7",
        "--objc_enable_binary_stripping",
        "--compilation_mode=fastbuild");
    addMockBinAndLibs(ImmutableList.of("a.mm"));
    CommandAction linkAction =
        (CommandAction) actionProducingArtifact("//bin:bin", "_bin");
    assertThat(linkAction.getArguments())
        .containsNoneOf("--dead_strip", "--no_dead_strip_inits_and_terms");
  }

  @Test
  public void testCompileEnv() throws Exception {
    MockObjcSupport.createCrosstoolPackage(mockToolsConfig);
    useConfiguration(
        ObjcCrosstoolMode.LIBRARY,
        "--experimental_disable_go",
        "--experimental_disable_jvm",
        "--ios_sdk_version=2.9",
        "--xcode_version=5.0",
        "--cpu=ios_x86_64");
    ScratchAttributeWriter.fromLabelString(this, "objc_library", "//main:lib")
        .setList("srcs", "a.m")
        .write();

    CppCompileAction compileAction = (CppCompileAction) compileAction("//main:lib", "a.o");

    Map<String, String> environment = compileAction.getEnvironment();
    assertThat(environment).containsEntry("XCODE_VERSION_OVERRIDE", "5.0");
    assertThat(environment).containsEntry("APPLE_SDK_VERSION_OVERRIDE", "2.9");
  }

  @Test
  public void testArchiveEnv() throws Exception {
    MockObjcSupport.createCrosstoolPackage(mockToolsConfig);
    useConfiguration(
        ObjcCrosstoolMode.LIBRARY,
        "--experimental_disable_go",
        "--experimental_disable_jvm",
        "--ios_sdk_version=2.9",
        "--xcode_version=5.0",
        "--cpu=ios_x86_64");
    ScratchAttributeWriter.fromLabelString(this, "objc_library", "//main:lib")
        .setList("srcs", "a.m")
        .write();

    CppLinkAction archiveAction = (CppLinkAction) archiveAction("//main:lib");

    Map<String, String> environment = archiveAction.getEnvironment();
    assertThat(environment).containsEntry("XCODE_VERSION_OVERRIDE", "5.0");
    assertThat(environment).containsEntry("APPLE_SDK_VERSION_OVERRIDE", "2.9");
  }
}
