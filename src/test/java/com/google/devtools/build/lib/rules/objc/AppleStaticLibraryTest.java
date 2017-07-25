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
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.LIPO;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions.ObjcCrosstoolMode;
import com.google.devtools.build.lib.testutil.Scratch;
import java.io.IOException;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for apple_static_library. */
@RunWith(JUnit4.class)
public class AppleStaticLibraryTest extends ObjcRuleTestCase {
  static final RuleType RULE_TYPE = new RuleType("apple_static_library") {
    @Override
    Iterable<String> requiredAttributes(Scratch scratch, String packageDir,
        Set<String> alreadyAdded) throws IOException {
      ImmutableList.Builder<String> attributes = new ImmutableList.Builder<>();
      if (!alreadyAdded.contains("srcs") && !alreadyAdded.contains("non_arc_srcs")) {
        scratch.file(packageDir + "/a.m");
        scratch.file(packageDir + "/private.h");
        attributes.add("srcs = ['a.m', 'private.h']");
      }
      if (!alreadyAdded.contains("platform_type")) {
        attributes.add("platform_type = 'ios'");
      }
      return attributes.build();
    }
  };

  @Test
  public void testMandatoryMinimumOsVersionUnset() throws Exception {
    RULE_TYPE.scratchTarget(scratch,
        "srcs", "['a.m']",
        "platform_type", "'watchos'");
    useConfiguration("--experimental_apple_mandatory_minimum_version");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//x:x");
    assertContainsEvent("must be explicitly specified");
  }

  @Test
  public void testMandatoryMinimumOsVersionSet() throws Exception {
    RULE_TYPE.scratchTarget(scratch,
        "minimum_os_version", "'8.0'",
        "srcs", "['a.m']",
        "platform_type", "'watchos'");
    useConfiguration("--experimental_apple_mandatory_minimum_version");
    getConfiguredTarget("//x:x");
  }

  @Test
  public void testUnknownPlatformType() throws Exception {
    checkError(
        "package",
        "test",
        String.format(MultiArchSplitTransitionProvider.UNSUPPORTED_PLATFORM_TYPE_ERROR_FORMAT,
            "meow_meow_os"),
        "apple_static_library(name = 'test', srcs = [ 'a.m' ], platform_type = 'meow_meow_os')");
  }

  @Test
  public void testCanUseCrosstool() throws Exception {
    useConfiguration(ObjcCrosstoolMode.ALL);
    RULE_TYPE.scratchTarget(scratch, "srcs", "['a.m']");

    // If the target is indeed using the c++ backend, then its archive action should be a
    // CppLinkAction.
    Action lipoLibAction = lipoLibAction("//x:x");
    Artifact archive = getFirstArtifactEndingWith(lipoLibAction.getInputs(), ".a");
    Action archiveAction = getGeneratingAction(archive);
    assertThat(archiveAction).isInstanceOf(CppLinkAction.class);
  }

  @Test
  public void testCanUseCrosstool_multiArch() throws Exception {
    useConfiguration(ObjcCrosstoolMode.ALL, "--ios_multi_cpus=i386,x86_64");
    RULE_TYPE.scratchTarget(scratch, "srcs", "['a.m']");

    // If the target is indeed using the c++ backend, then its archive action should be a
    // CppLinkAction.
    Action lipoLibAction = lipoLibAction("//x:x");
    Artifact archive = getFirstArtifactEndingWith(lipoLibAction.getInputs(), ".a");
    Action archiveAction = getGeneratingAction(archive);
    assertThat(archiveAction).isInstanceOf(CppLinkAction.class);
  }

  @Test
  public void testSymlinkInsteadOfLipoSingleArch() throws Exception {
    RULE_TYPE.scratchTarget(scratch,
        "srcs", "['a.m']");

    SymlinkAction action = (SymlinkAction) lipoLibAction("//x:x");
    CommandAction linkAction = linkLibAction("//x:x");

    assertThat(action.getInputs())
        .containsExactly(Iterables.getOnlyElement(linkAction.getOutputs()));
  }

  @Test
  public void testAvoidDepsProviders() throws Exception {
    scratch.file(
        "package/BUILD",
        "apple_static_library(",
        "    name = 'test',",
        "    deps = [':objcLib'],",
        "    platform_type = 'ios',",
        "    avoid_deps = [':avoidLib'],",
        ")",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ], deps = [':avoidLib', ':baseLib'])",
        "objc_library(",
        "    name = 'baseLib',",
        "    srcs = [ 'base.m' ],",
        "    sdk_frameworks = ['BaseSDK'],",
        "    resources = [':base.png']",
        ")",
        "objc_library(",
        "    name = 'avoidLib',",
        "    srcs = [ 'c.m' ],",
        "    sdk_frameworks = ['AvoidSDK'],",
        "    resources = [':avoid.png']",
        ")");

    ObjcProvider provider = getConfiguredTarget("//package:test")
        .get(AppleStaticLibraryProvider.SKYLARK_CONSTRUCTOR)
        .getDepsObjcProvider();
    // Do not remove SDK_FRAMEWORK or GENERAL_RESOURCE_FILE values in avoid_deps.
    assertThat(provider.get(ObjcProvider.SDK_FRAMEWORK))
        .containsAllOf(new SdkFramework("AvoidSDK"), new SdkFramework("BaseSDK"));
    assertThat(Artifact.toRootRelativePaths(provider.get(ObjcProvider.GENERAL_RESOURCE_FILE)))
        .containsExactly("package/base.png");
  }

  @Test
  public void testNoSrcs() throws Exception {
    scratch.file("package/BUILD",
        "apple_static_library(",
        "    name = 'test',",
        "    deps = [':objcLib'],",
        "    platform_type = 'ios',",
        ")",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");
    useConfiguration("--xcode_version=5.8");

    CommandAction action = linkLibAction("//package:test");
    assertThat(Artifact.toRootRelativePaths(action.getInputs())).contains("package/libobjcLib.a");
  }

  @Test
  public void testLipoAction() throws Exception {
    scratch.file("package/BUILD",
        "apple_static_library(name = 'test',",
        "    platform_type = 'ios',",
        "    srcs = [ 'a.m' ])");

    useConfiguration("--ios_multi_cpus=i386,x86_64");

    CommandAction action = (CommandAction) lipoLibAction("//package:test");
    String i386Lib =
        configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_IOS) + "package/test-fl.a";
    String x8664Lib =
        configurationBin("x86_64", ConfigurationDistinguisher.APPLEBIN_IOS) + "package/test-fl.a";

    assertThat(Artifact.toExecPaths(action.getInputs()))
        .containsExactly(i386Lib, x8664Lib, MOCK_XCRUNWRAPPER_PATH);

    assertThat(action.getArguments())
        .containsExactly(
            MOCK_XCRUNWRAPPER_PATH,
            LIPO,
            "-create",
            i386Lib,
            x8664Lib,
            "-o",
            execPathEndingWith(action.getOutputs(), "test_lipo.a"))
        .inOrder();

    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("package/test_lipo.a");
    assertRequiresDarwin(action);
  }

  @Test
  public void testWatchSimulatorDepCompile() throws Exception {
    scratch.file(
        "package/BUILD",
        "apple_static_library(",
        "    name = 'test',",
        "    srcs = ['a.m'],",
        "    deps = [':objcLib'],",
        "    platform_type = 'watchos'",
        ")",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");

    Action lipoAction = lipoLibAction("//package:test");

    String i386Bin =
        configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_WATCHOS) + "package/test-fl.a";
    Artifact libArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), i386Bin);
    CommandAction linkAction = (CommandAction) getGeneratingAction(libArtifact);
    CommandAction objcLibCompileAction =
        (CommandAction)
            getGeneratingAction(getFirstArtifactEndingWith(linkAction.getInputs(), "libobjcLib.a"));

    assertAppleSdkPlatformEnv(objcLibCompileAction, "WatchSimulator");
    assertThat(objcLibCompileAction.getArguments()).containsAllOf("-arch_only", "i386").inOrder();
  }

  @Test
  public void testMultiarchCcDep() throws Exception {
    scratch.file(
        "package/BUILD",
        "apple_static_library(name = 'test',",
        "    srcs = [ 'a.m' ],",
        "    deps = [ ':cclib' ],",
        "    platform_type = 'ios')",
        "cc_library(name = 'cclib', srcs = ['dep.c'])");

    useConfiguration(
        "--ios_multi_cpus=i386,x86_64",
        "--experimental_disable_go",
        "--experimental_disable_jvm",
        "--crosstool_top=//tools/osx/crosstool:crosstool");

    CommandAction action = (CommandAction) lipoLibAction("//package:test");
    String i386Prefix =
        configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_IOS,
            DEFAULT_IOS_SDK_VERSION);
    String x8664Prefix =
        configurationBin("x86_64", ConfigurationDistinguisher.APPLEBIN_IOS,
            DEFAULT_IOS_SDK_VERSION);

    CommandAction i386BinAction =
        (CommandAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(action.getInputs(), i386Prefix + "package/test-fl.a"));

    CommandAction x8664BinAction =
        (CommandAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(action.getInputs(), x8664Prefix + "package/test-fl.a"));

    assertThat(Artifact.toExecPaths(i386BinAction.getInputs()))
        .containsAllOf(i386Prefix + "package/libtest.a", i386Prefix + "package/libcclib.a");
    assertThat(Artifact.toExecPaths(x8664BinAction.getInputs()))
        .containsAllOf(x8664Prefix + "package/libtest.a", x8664Prefix + "package/libcclib.a");
  }

  @Test
  public void testWatchSimulatorLipoAction() throws Exception {
    scratch.file("package/BUILD",
        "apple_static_library(name = 'test', srcs = [ 'a.m' ], platform_type = 'watchos')");

    // Tests that ios_multi_cpus and ios_cpu are completely ignored.
    useConfiguration("--ios_multi_cpus=x86_64", "--ios_cpu=x86_64", "--watchos_cpus=i386,armv7k");

    CommandAction action = (CommandAction) lipoLibAction("//package:test");
    String i386Bin = configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_WATCHOS)
        + "package/test-fl.a";
    String armv7kBin = configurationBin("armv7k", ConfigurationDistinguisher.APPLEBIN_WATCHOS)
        + "package/test-fl.a";

    assertThat(Artifact.toExecPaths(action.getInputs()))
        .containsExactly(i386Bin, armv7kBin, MOCK_XCRUNWRAPPER_PATH);

    assertContainsSublist(action.getArguments(), ImmutableList.of(
        MOCK_XCRUNWRAPPER_PATH, LIPO, "-create"));
    assertThat(action.getArguments()).containsAllOf(armv7kBin, i386Bin);
    assertContainsSublist(action.getArguments(), ImmutableList.of(
        "-o", execPathEndingWith(action.getOutputs(), "test_lipo.a")));

    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("package/test_lipo.a");
    assertAppleSdkPlatformEnv(action, "WatchOS");
    assertRequiresDarwin(action);
  }

  @Test
  public void testProtoDeps() throws Exception {
    scratch.file(
        "protos/BUILD",
        "proto_library(",
        "    name = 'protos_main',",
        "    srcs = ['data_a.proto', 'data_b.proto'],",
        ")",
        "proto_library(",
        "    name = 'protos_low_level',",
        "    srcs = ['data_b.proto'],",
        ")",
        "objc_proto_library(",
        "    name = 'objc_protos_main',",
        "    portable_proto_filters = ['filter_a.pbascii'],",
        "    deps = [':protos_main'],",
        ")",
        "objc_proto_library(",
        "    name = 'objc_protos_low_level',",
        "    portable_proto_filters = ['filter_b.pbascii'],",
        "    deps = [':protos_low_level'],",
        ")");
    scratch.file(
        "libs/BUILD",
        "objc_library(",
        "    name = 'main_lib',",
        "    srcs = ['a.m'],",
        "    deps = ['//protos:objc_protos_main',]",
        ")",
        "objc_library(",
        "    name = 'apple_low_level_lib',",
        "    srcs = ['a.m'],",
        "    deps = ['//protos:objc_protos_low_level',]",
        ")");

    RULE_TYPE.scratchTarget(
        scratch,
        "deps", "['//libs:main_lib']",
        "avoid_deps", "['//libs:apple_low_level_lib']");

    CommandAction linkAction = linkLibAction("//x:x");
    Iterable<Artifact> linkActionInputs = linkAction.getInputs();

    ImmutableList.Builder<Artifact> objects = ImmutableList.builder();
    for (Artifact binActionArtifact : linkActionInputs) {
      if (binActionArtifact.getRootRelativePath().getPathString().endsWith(".a")) {
        CommandAction subLinkAction = (CommandAction) getGeneratingAction(binActionArtifact);
        for (Artifact linkActionArtifact : subLinkAction.getInputs()) {
          if (linkActionArtifact.getRootRelativePath().getPathString().endsWith(".o")) {
            objects.add(linkActionArtifact);
          }
        }
      }
    }

    ImmutableList<Artifact> objectFiles = objects.build();
    assertThat(getFirstArtifactEndingWith(objectFiles, "DataA.pbobjc.o")).isNotNull();
    assertThat(getFirstArtifactEndingWith(objectFiles, "DataB.pbobjc.o")).isNull();
  }

  @Test
  public void testMinimumOs() throws Exception {
    RULE_TYPE.scratchTarget(scratch,
        "deps", "['//package:objcLib']",
        "minimum_os_version", "'5.4'");
    scratch.file("package/BUILD",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");
    useConfiguration("--xcode_version=5.8");

    CommandAction linkAction = linkLibAction("//x:x");
    CommandAction objcLibArchiveAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(linkAction.getInputs(), "libobjcLib.a"));
    CommandAction objcLibCompileAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(objcLibArchiveAction.getInputs(), "b.o"));

    String compileArgs = Joiner.on(" ").join(objcLibCompileAction.getArguments());
    assertThat(compileArgs).contains("-mios-simulator-version-min=5.4");
  }

  @Test
  public void testMinimumOs_watchos() throws Exception {
    RULE_TYPE.scratchTarget(scratch,
        "deps", "['//package:objcLib']",
        "platform_type", "'watchos'",
        "minimum_os_version", "'5.4'");
    scratch.file("package/BUILD",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");
    useConfiguration("--xcode_version=5.8");

    CommandAction linkAction = linkLibAction("//x:x");
    CommandAction objcLibArchiveAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(linkAction.getInputs(), "libobjcLib.a"));
    CommandAction objcLibCompileAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(objcLibArchiveAction.getInputs(), "b.o"));

    String compileArgs = Joiner.on(" ").join(objcLibCompileAction.getArguments());
    assertThat(compileArgs).contains("-mwatchos-simulator-version-min=5.4");
  }

  @Test
  public void testMinimumOs_invalid_nonVersion() throws Exception {
    checkMinimumOs_invalid_nonVersion(RULE_TYPE);
  }

  @Test
  public void testMinimumOs_invalid_containsAlphabetic() throws Exception {
    checkMinimumOs_invalid_containsAlphabetic(RULE_TYPE);
  }

  @Test
  public void testMinimumOs_invalid_tooManyComponents() throws Exception {
    checkMinimumOs_invalid_tooManyComponents(RULE_TYPE);
  }

  @Test
  public void testFilesToCompileOutputGroup() throws Exception {
    checkFilesToCompileOutputGroup(RULE_TYPE);
  }

  @Test
  public void testAppleSdkVersionEnv() throws Exception {
    scratch.file("package/BUILD",
        "apple_static_library(name = 'test', srcs = [ 'a.m' ], platform_type = 'ios')");

    CommandAction action = linkLibAction("//package:test");

    assertAppleSdkVersionEnv(action);
  }

  @Test
  public void testNonDefaultAppleSdkVersionEnv() throws Exception {
    scratch.file("package/BUILD",
        "apple_static_library(name = 'test', srcs = [ 'a.m' ], platform_type = 'ios')");
    useConfiguration("--ios_sdk_version=8.1");

    CommandAction action = linkLibAction("//package:test");

    assertAppleSdkVersionEnv(action, "8.1");
  }

  @Test
  public void testAppleSdkDefaultPlatformEnv() throws Exception {
    scratch.file("package/BUILD",
        "apple_static_library(name = 'test', srcs = [ 'a.m' ], platform_type = 'ios')");
    CommandAction action = linkLibAction("//package:test");

    assertAppleSdkPlatformEnv(action, "iPhoneSimulator");
  }

  @Test
  public void testAppleSdkIphoneosPlatformEnv() throws Exception {
    scratch.file("package/BUILD",
        "apple_static_library(name = 'test', srcs = [ 'a.m' ], platform_type = 'ios')");
    useConfiguration("--cpu=ios_arm64");

    CommandAction action = linkLibAction("//package:test");

    assertAppleSdkPlatformEnv(action, "iPhoneOS");
  }

  @Test
  public void testAppleSdkWatchsimulatorPlatformEnv() throws Exception {
    scratch.file(
        "package/BUILD",
        "apple_static_library(name = 'test', srcs = [ 'a.m' ], platform_type = 'watchos')");
    useConfiguration("--watchos_cpus=i386");

    Action lipoAction = lipoLibAction("//package:test");

    String i386Lib =
        configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_WATCHOS) + "package/test-fl.a";
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), i386Lib);
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);

    assertAppleSdkPlatformEnv(linkAction, "WatchSimulator");
  }

  @Test
  public void testAppleSdkWatchosPlatformEnv() throws Exception {
    scratch.file(
        "package/BUILD",
        "apple_static_library(name = 'test', srcs = [ 'a.m' ], platform_type = 'watchos')");
    useConfiguration("--watchos_cpus=armv7k");

    Action lipoAction = lipoLibAction("//package:test");

    String armv7kLib =
        configurationBin("armv7k", ConfigurationDistinguisher.APPLEBIN_WATCHOS)
            + "package/test-fl.a";
    Artifact libArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), armv7kLib);
    CommandAction linkAction = (CommandAction) getGeneratingAction(libArtifact);

    assertAppleSdkPlatformEnv(linkAction, "WatchOS");
  }

  @Test
  public void testXcodeVersionEnv() throws Exception {
    scratch.file("package/BUILD",
        "apple_static_library(name = 'test', srcs = [ 'a.m' ], platform_type = 'ios')");
    useConfiguration("--xcode_version=5.8");

    CommandAction action = linkLibAction("//package:test");

    assertXcodeVersionEnv(action, "5.8");
  }

  @Test
  public void testWatchSimulatorLinkAction() throws Exception {
    scratch.file(
        "package/BUILD",
        "apple_static_library(",
        "    name = 'test',",
        "    srcs = ['a.m'],",
        "    deps = [':objcLib'],",
        "    platform_type = 'watchos'",
        ")",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");

    // Tests that ios_multi_cpus and ios_cpu are completely ignored.
    useConfiguration("--ios_multi_cpus=x86_64", "--ios_cpu=x86_64", "--watchos_cpus=i386");

    Action lipoAction = lipoLibAction("//package:test");

    String i386Bin =
        configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_WATCHOS) + "package/test-fl.a";
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), i386Bin);
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);

    assertAppleSdkPlatformEnv(linkAction, "WatchSimulator");
    assertThat(normalizeBashArgs(linkAction.getArguments()))
        .containsAllOf("-arch_only", "i386")
        .inOrder();
  }

  @Test
  public void testAppleStaticLibraryProvider() throws Exception {
    scratch.file(
        "lib/BUILD",
        "apple_static_library(",
        "    name = 'applelib',",
        "    srcs = ['a.m'],",
        "    platform_type = 'ios',",
        "    hdrs = ['a.h'],",
        ")");
    ConfiguredTarget binTarget = getConfiguredTarget("//lib:applelib");
    AppleStaticLibraryProvider provider =
        binTarget.get(AppleStaticLibraryProvider.SKYLARK_CONSTRUCTOR);
    assertThat(provider).isNotNull();
    assertThat(provider.getMultiArchArchive()).isNotNull();
    assertThat(provider.getDepsObjcProvider()).isNotNull();
    assertThat(provider.getMultiArchArchive()).isEqualTo(
        Iterables.getOnlyElement(
            provider.getDepsObjcProvider().get(ObjcProvider.MULTI_ARCH_LINKED_ARCHIVES)));
  }

  @Test
  public void testMinimumOsDifferentTargets() throws Exception {
    checkMinimumOsDifferentTargets(RULE_TYPE, "_lipo.a", "-fl.a");
  }
}
