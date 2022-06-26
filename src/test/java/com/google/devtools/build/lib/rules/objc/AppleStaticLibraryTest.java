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
import static org.junit.Assert.assertThrows;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.DefaultInfo;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
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
      if (!alreadyAdded.contains("deps")) {
        String depPackageDir = packageDir + "_defaultDep";
        scratch.file(depPackageDir + "/a.m");
        scratch.file(depPackageDir + "/private.h");
        scratch.file(depPackageDir + "/BUILD",
            "objc_library(name = 'lib_dep', srcs = ['a.m', 'private.h'])");
        attributes.add("deps = ['//" + depPackageDir + ":" + "lib_dep']");
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
        "platform_type", "'watchos'");
    useConfiguration("--experimental_apple_mandatory_minimum_version");
    getConfiguredTarget("//x:x");
  }

  @Test
  public void testMandatoryMinimumOsVersionTrailingZeros() throws Exception {
    RULE_TYPE.scratchTarget(scratch, "minimum_os_version", "'8.0.0'", "platform_type", "'watchos'");
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
        "apple_static_library(name = 'test', platform_type = 'meow_meow_os')");
  }

  @Test
  public void testSymlinkInsteadOfLipoSingleArch() throws Exception {
    RULE_TYPE.scratchTarget(scratch);

    SymlinkAction action = (SymlinkAction) lipoLibAction("//x:x");
    CommandAction linkAction = linkLibAction("//x:x");

    assertThat(action.getInputs().toList())
        .containsExactly(Iterables.getOnlyElement(linkAction.getOutputs()));
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
    RULE_TYPE.scratchTarget(scratch, "platform_type", "'ios'");

    useConfiguration("--ios_multi_cpus=i386,x86_64");

    CommandAction action = (CommandAction) lipoLibAction("//x:x");
    String i386Lib = "x/x-i386-apple-ios-fl.a";
    String x8664Lib = "x/x-x86_64-apple-ios-fl.a";

    assertThat(Artifact.toRootRelativePaths(action.getInputs())).containsAtLeast(i386Lib, x8664Lib);

    assertContainsSublist(
        removeConfigFragment(action.getArguments()),
        ImmutableList.of(removeConfigFragment(MOCK_XCRUNWRAPPER_EXECUTABLE_PATH), LIPO, "-create"));
    String binFragment =
        removeConfigFragment(targetConfig.getBinFragment(RepositoryName.MAIN) + "/");
    assertThat(removeConfigFragment(action.getArguments()))
        .containsAtLeast(binFragment + i386Lib, binFragment + x8664Lib);
    assertContainsSublist(
        action.getArguments(),
        ImmutableList.of("-o", execPathEndingWith(action.getOutputs(), "x_lipo.a")));

    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("x/x_lipo.a");
    assertRequiresDarwin(action);
  }

  @Test
  public void testWatchSimulatorDepCompile() throws Exception {
    scratch.file(
        "package/BUILD",
        "apple_static_library(",
        "    name = 'test',",
        "    deps = [':objcLib'],",
        "    platform_type = 'watchos'",
        ")",
        "objc_library(name = 'objcLib', srcs = [ 'a.m' ])");

    Action lipoAction = lipoLibAction("//package:test");

    String i386Bin = "i386-apple-watchos-fl.a";
    Artifact libArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), i386Bin);
    CommandAction linkAction = (CommandAction) getGeneratingAction(libArtifact);
    CommandAction objcLibCompileAction =
        (CommandAction)
            getGeneratingAction(getFirstArtifactEndingWith(linkAction.getInputs(), "libobjcLib.a"));

    assertAppleSdkPlatformEnv(objcLibCompileAction, "WatchSimulator");
    assertThat(objcLibCompileAction.getArguments()).containsAtLeast("-arch_only", "i386").inOrder();
  }

  @Test
  public void testMultiarchCcDep() throws Exception {
    scratch.file(
        "package/BUILD",
        "apple_static_library(name = 'test',",
        "    deps = [ ':cclib' ],",
        "    platform_type = 'ios')",
        "cc_library(name = 'cclib', srcs = ['dep.c'])");

    useConfiguration(
        "--ios_multi_cpus=i386,x86_64",
        "--crosstool_top=//tools/osx/crosstool:crosstool");

    CommandAction action = (CommandAction) lipoLibAction("//package:test");
    String i386Prefix = configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_IOS);
    String x8664Prefix = configurationBin("x86_64", ConfigurationDistinguisher.APPLEBIN_IOS);

    CommandAction i386BinAction =
        (CommandAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(action.getInputs(), "i386-apple-ios-fl.a"));

    CommandAction x8664BinAction =
        (CommandAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(action.getInputs(), "x86_64-apple-ios-fl.a"));

    assertThat(removeConfigFragment(Artifact.asExecPaths(i386BinAction.getInputs())))
        .contains(removeConfigFragment(i386Prefix + "package/libcclib.a"));
    assertThat(removeConfigFragment(Artifact.asExecPaths(x8664BinAction.getInputs())))
        .contains(removeConfigFragment(x8664Prefix + "package/libcclib.a"));
  }

  @Test
  public void testWatchSimulatorLipoAction() throws Exception {
    RULE_TYPE.scratchTarget(scratch, "platform_type", "'watchos'");

    // Tests that ios_multi_cpus and ios_cpu are completely ignored.
    useConfiguration("--ios_multi_cpus=x86_64", "--ios_cpu=x86_64", "--watchos_cpus=i386,armv7k");

    CommandAction action = (CommandAction) lipoLibAction("//x:x");
    String i386Bin = "x/x-i386-apple-watchos-fl.a";
    String armv7kBin = "x/x-armv7k-apple-watchos-fl.a";

    assertThat(Artifact.toRootRelativePaths(action.getInputs()))
        .containsAtLeast(i386Bin, armv7kBin);

    assertContainsSublist(
        removeConfigFragment(action.getArguments()),
        ImmutableList.of(removeConfigFragment(MOCK_XCRUNWRAPPER_EXECUTABLE_PATH), LIPO, "-create"));
    String binFragment =
        removeConfigFragment(targetConfig.getBinFragment(RepositoryName.MAIN) + "/");
    assertThat(removeConfigFragment(action.getArguments()))
        .containsAtLeast(binFragment + armv7kBin, binFragment + i386Bin);
    assertContainsSublist(action.getArguments(), ImmutableList.of(
        "-o", execPathEndingWith(action.getOutputs(), "x_lipo.a")));

    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("x/x_lipo.a");
    assertAppleSdkPlatformEnv(action, "WatchOS");
    assertRequiresDarwin(action);
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
  public void testAppleSdkVersionEnv() throws Exception {
    RULE_TYPE.scratchTarget(scratch, "platform_type", "'ios'");

    CommandAction action = linkLibAction("//x:x");

    assertAppleSdkVersionEnv(action);
  }

  @Test
  public void testNonDefaultAppleSdkVersionEnv() throws Exception {
    RULE_TYPE.scratchTarget(scratch, "platform_type", "'ios'");
    useConfiguration("--ios_sdk_version=8.1");

    CommandAction action = linkLibAction("//x:x");

    assertAppleSdkVersionEnv(action, "8.1");
  }

  @Test
  public void testAppleSdkDefaultPlatformEnv() throws Exception {
    RULE_TYPE.scratchTarget(scratch, "platform_type", "'ios'");
    CommandAction action = linkLibAction("//x:x");

    assertAppleSdkPlatformEnv(action, "iPhoneSimulator");
  }

  @Test
  public void testAppleSdkIphoneosPlatformEnv() throws Exception {
    RULE_TYPE.scratchTarget(scratch, "platform_type", "'ios'");
    useConfiguration("--apple_platform_type=ios", "--cpu=ios_arm64");

    CommandAction action = linkLibAction("//x:x");

    assertAppleSdkPlatformEnv(action, "iPhoneOS");
  }

  @Test
  public void testAppleSdkWatchsimulatorPlatformEnv() throws Exception {
    RULE_TYPE.scratchTarget(scratch, "platform_type", "'watchos'");
    useConfiguration("--watchos_cpus=i386");

    Action lipoAction = lipoLibAction("//x:x");

    String i386Lib = "i386-apple-watchos-fl.a";
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), i386Lib);
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);

    assertAppleSdkPlatformEnv(linkAction, "WatchSimulator");
  }

  @Test
  public void testAppleSdkWatchosPlatformEnv() throws Exception {
    RULE_TYPE.scratchTarget(scratch, "platform_type", "'watchos'");
    useConfiguration("--watchos_cpus=armv7k");

    Action lipoAction = lipoLibAction("//x:x");

    String armv7kLib = "armv7k-apple-watchos-fl.a";
    Artifact libArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), armv7kLib);
    CommandAction linkAction = (CommandAction) getGeneratingAction(libArtifact);

    assertAppleSdkPlatformEnv(linkAction, "WatchOS");
  }

  @Test
  public void testXcodeVersionEnv() throws Exception {
    RULE_TYPE.scratchTarget(scratch, "platform_type", "'ios'");
    useConfiguration("--xcode_version=5.8");

    CommandAction action = linkLibAction("//x:x");

    assertXcodeVersionEnv(action, "5.8");
  }

  @Test
  public void testWatchSimulatorLinkAction() throws Exception {
    scratch.file(
        "package/BUILD",
        "apple_static_library(",
        "    name = 'test',",
        "    deps = [':objcLib'],",
        "    platform_type = 'watchos'",
        ")",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");

    // Tests that ios_multi_cpus and ios_cpu are completely ignored.
    useConfiguration("--ios_multi_cpus=x86_64", "--ios_cpu=x86_64", "--watchos_cpus=i386");

    Action lipoAction = lipoLibAction("//package:test");

    String i386Bin = "i386-apple-watchos-fl.a";
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), i386Bin);
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);

    assertAppleSdkPlatformEnv(linkAction, "WatchSimulator");
    assertThat(normalizeBashArgs(linkAction.getArguments()))
        .containsAtLeast("-arch_only", "i386")
        .inOrder();
  }

  @Test
  public void testAppleStaticLibraryInfo() throws Exception {
    RULE_TYPE.scratchTarget(scratch, "platform_type", "'ios'");
    ConfiguredTarget binTarget = getConfiguredTarget("//x:x");
    SymlinkAction lipoAction = (SymlinkAction) lipoLibAction("//x:x");

    AppleStaticLibraryInfo provider = binTarget.get(AppleStaticLibraryInfo.STARLARK_CONSTRUCTOR);
    assertThat(provider).isNotNull();
    assertThat(provider.getMultiArchArchive())
        .isEqualTo(Iterables.getOnlyElement(lipoAction.getOutputs()));
    assertThat(provider.getDepsObjcProvider()).isNotNull();
  }

  @Test
  public void testMinimumOsDifferentTargets() throws Exception {
    checkMinimumOsDifferentTargets(RULE_TYPE, "_lipo.a", "-fl.a");
  }

  @Test
  public void testAvoidDepsObjects() throws Exception {
    scratch.file("package/BUILD",
        "apple_static_library(",
        "    name = 'test',",
        "    deps = [':objcLib'],",
        "    avoid_deps = [':avoidLib'],",
        "    platform_type = 'ios',",
        ")",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ], deps = [':avoidLib', ':baseLib'])",
        "objc_library(name = 'baseLib', srcs = [ 'base.m' ])",
        "objc_library(name = 'avoidLib', srcs = [ 'c.m' ])");

    CommandAction action = linkLibAction("//package:test");
    assertThat(Artifact.toRootRelativePaths(action.getInputs()))
        .containsAtLeast("package/libobjcLib.a", "package/libbaseLib.a");
    assertThat(Artifact.toRootRelativePaths(action.getInputs())).doesNotContain(
        "package/libavoidLib.a");
  }

  @Test
  // Tests that if there is a cc_library in avoid_deps, all of its dependencies are
  // transitively avoided, even if it is not present in deps.
  public void testAvoidDepsObjects_avoidViaCcLibrary() throws Exception {
    scratch.file("package/BUILD",
        "apple_static_library(",
        "    name = 'test',",
        "    deps = [':objcLib'],",
        "    avoid_deps = [':avoidCclib'],",
        "    platform_type = 'ios',",
        ")",
        "cc_library(name = 'avoidCclib', srcs = ['cclib.c'], deps = [':avoidLib'])",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ], deps = [':avoidLib'])",
        "objc_library(name = 'avoidLib', srcs = [ 'c.m' ])");

    CommandAction action = linkLibAction("//package:test");
    assertThat(Artifact.toRootRelativePaths(action.getInputs())).contains(
        "package/libobjcLib.a");
    assertThat(Artifact.toRootRelativePaths(action.getInputs())).doesNotContain(
        "package/libavoidCcLib.a");
  }

  @Test
  public void testRepeatedDepsViaObjcLibraryAreNotInCommandLine() throws Exception {
    scratch.file(
        "package/BUILD",
        "apple_static_library(",
        "    name = 'test',",
        "    deps = [':cclib', ':objcLib2'],",
        "    platform_type = 'ios',",
        ")",
        "objc_library(name = 'objcLib2', srcs = [ 'b2.m' ], deps = [':objcLib'])",
        "cc_library(name = 'cclib', srcs = ['cclib.cc'], deps = [':objcLib'])",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");

    CommandAction action = linkLibAction("//package:test");
    assertThat(action.getArguments()).containsNoDuplicates();
  }

  @Test
  // Tests that if there is a cc_library in avoid_deps, and it is present in deps, it will
  // be avoided, as well as its transitive dependencies.
  public void testAvoidDepsObjects_avoidCcLibrary() throws Exception {
    scratch.file("package/BUILD",
        "apple_static_library(",
        "    name = 'test',",
        "    deps = [':objcLib', ':avoidCclib'],",
        "    avoid_deps = [':avoidCclib'],",
        "    platform_type = 'ios',",
        ")",
        "cc_library(name = 'avoidCclib', srcs = ['cclib.c'], deps = [':avoidLib'])",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ])",
        "objc_library(name = 'avoidLib', srcs = [ 'c.m' ])");

    CommandAction action = linkLibAction("//package:test");
    assertThat(Artifact.toRootRelativePaths(action.getInputs())).contains(
        "package/libobjcLib.a");
    assertThat(Artifact.toRootRelativePaths(action.getInputs())).doesNotContain(
        "package/libavoidCcLib.a");
  }

  @Test
  public void testFeatureFlags_offByDefault() throws Exception {
    useConfiguration("--enforce_transitive_configs_for_config_feature_flag");
    scratchFeatureFlagTestLib();
    scratch.file(
        "test/BUILD",
        "apple_static_library(",
        "    name = 'static_lib',",
        "    deps = ['//lib:objcLib'],",
        "    platform_type = 'ios',",
        "    transitive_configs = ['//lib:flag1', '//lib:flag2'],",
        ")");

    CommandAction linkAction = linkLibAction("//test:static_lib");
    CommandAction objcLibArchiveAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(linkAction.getInputs(), "libobjcLib.a"));

    CommandAction flag1offCompileAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(objcLibArchiveAction.getInputs(), "flag1off.o"));
    CommandAction flag2offCompileAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(objcLibArchiveAction.getInputs(), "flag2off.o"));

    String compileArgs1 = Joiner.on(" ").join(flag1offCompileAction.getArguments());
    String compileArgs2 = Joiner.on(" ").join(flag2offCompileAction.getArguments());
    assertThat(compileArgs1).contains("FLAG_1_OFF");
    assertThat(compileArgs1).contains("FLAG_2_OFF");
    assertThat(compileArgs2).contains("FLAG_1_OFF");
    assertThat(compileArgs2).contains("FLAG_2_OFF");
  }

  @Test
  public void testAppleStaticLibraryProvider() throws Exception {
    RULE_TYPE.scratchTarget(scratch, "platform_type", "'ios'");
    scratch.file("examples/rule/BUILD");
    scratch.file(
        "examples/rule/apple_rules.bzl",
        "def starlark_static_lib_impl(ctx):",
        "   dep_provider = ctx.attr.proxy[apple_common.AppleStaticLibrary]",
        "   my_provider = apple_common.AppleStaticLibrary(archive = dep_provider.archive,",
        "       objc = dep_provider.objc)",
        "   return [my_provider]",
        "",
        "starlark_static_lib = rule(",
        "  implementation = starlark_static_lib_impl,",
        "  attrs = {'proxy': attr.label()},",
        ")");

    scratch.file(
        "examples/apple_starlark/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "load('//examples/rule:apple_rules.bzl', 'starlark_static_lib')",
        "starlark_static_lib(",
        "   name='my_target',",
        "   proxy='//x:x'",
        ")");

    ConfiguredTarget binTarget = getConfiguredTarget("//examples/apple_starlark:my_target");
    SymlinkAction lipoAction = (SymlinkAction) lipoLibAction("//x:x");

    AppleStaticLibraryInfo provider = binTarget.get(AppleStaticLibraryInfo.STARLARK_CONSTRUCTOR);
    assertThat(provider).isNotNull();
    assertThat(provider.getMultiArchArchive())
        .isEqualTo(Iterables.getOnlyElement(lipoAction.getOutputs()));
    assertThat(provider.getDepsObjcProvider()).isNotNull();
  }

  @Test
  public void testAppleStaticLibraryProvider_invalidArgs() throws Exception {
    RULE_TYPE.scratchTarget(scratch, "platform_type", "'ios'");
    scratch.file("examples/rule/BUILD");
    scratch.file(
        "examples/rule/apple_rules.bzl",
        "def starlark_static_lib_impl(ctx):",
        "   dep_provider = ctx.attr.proxy[apple_common.AppleStaticLibrary]",
        "   my_provider = apple_common.AppleStaticLibrary(archive = dep_provider.archive,",
        "       objc = dep_provider.objc, foo = 'bar')",
        "   return [my_provider]",
        "",
        "starlark_static_lib = rule(",
        "  implementation = starlark_static_lib_impl,",
        "  attrs = {'proxy': attr.label()},",
        ")");

    scratch.file(
        "examples/apple_starlark/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "load('//examples/rule:apple_rules.bzl', 'starlark_static_lib')",
        "starlark_static_lib(",
        "   name='my_target',",
        "   proxy='//x:x'",
        ")");

    AssertionError expected =
        assertThrows(
            AssertionError.class, () -> getConfiguredTarget("//examples/apple_starlark:my_target"));
    assertThat(expected).hasMessageThat().contains("unexpected keyword argument 'foo'");
  }

  @Test
  public void testProcessHeadersInDependencies() throws Exception {
    MockObjcSupport.setupCcToolchainConfig(
        mockToolsConfig, MockObjcSupport.darwinX86_64().withFeatures(CppRuleClasses.PARSE_HEADERS));
    useConfiguration("--features=parse_headers", "--process_headers_in_dependencies");
    ConfiguredTarget x =
        scratchConfiguredTarget(
            "foo",
            "x",
            "apple_static_library(name = 'x', platform_type = 'macos', deps = [':y', ':z'])",
            "cc_library(name = 'y', hdrs = ['y.h'])",
            "objc_library(name = 'z', hdrs = ['z.h'])");
    String validation = ActionsTestUtil.baseNamesOf(getOutputGroup(x, OutputGroupInfo.VALIDATION));
    assertThat(validation).contains("y.h.processed");
    assertThat(validation).contains("z.h.processed");
  }

  @Test
  public void testRunfiles() throws Exception {
    scratch.file(
        "package/BUILD",
        "apple_static_library(",
        "    name = 'test',",
        "    deps = [':objcLib'],",
        "    platform_type = 'ios',",
        ")",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");
    useConfiguration("--xcode_version=5.8");

    ConfiguredTarget target = getConfiguredTarget("//package:test");
    assertThat(
            artifactsToStrings(target.get(DefaultInfo.PROVIDER).getDataRunfiles().getArtifacts()))
        .contains("/ package/test_lipo.a");
  }

  @Test
  public void testLinkingActionsWithCpus() throws Exception {
    RepositoryName toolsRepo = TestConstants.TOOLS_REPOSITORY;
    scratch.file(
        "package/starlark_apple_multi_library.bzl",
        "def _starlark_apple_multi_library_impl(ctx):",
        "  link_result = apple_common.link_multi_arch_static_library(",
        "      ctx = ctx,",
        "  )",
        "  return [",
        "      DefaultInfo(files = depset([o.library for o in link_result.outputs])),",
        "      link_result.output_groups,",
        "  ]",
        "starlark_apple_multi_library = rule(",
        "    attrs = {",
        "        '_child_configuration_dummy': attr.label(",
        "            cfg = apple_common.multi_arch_split,",
        "            default = Label('" + toolsRepo + "//tools/cpp:current_cc_toolchain'),",
        "        ),",
        "        'deps': attr.label_list(",
        "            cfg = apple_common.multi_arch_split,",
        "            providers = [apple_common.Objc],",
        "            flags = ['DIRECT_COMPILE_TIME_INPUT'],",
        "            allow_rules = ['cc_library', 'cc_inc_library'],",
        "        ),",
        "        'avoid_deps': attr.label_list(",
        "            cfg = apple_common.multi_arch_split,",
        "            providers = [apple_common.Objc],",
        "            flags = ['DIRECT_COMPILE_TIME_INPUT'],",
        "            allow_rules = ['cc_library', 'cc_inc_library'],",
        "        ),",
        "        # test attr to assert built targets",
        "        # attrs for apple_common.multi_arch_split",
        "        'platform_type': attr.string(),",
        "        'minimum_os_version': attr.string(),",
        "    },",
        "    fragments = ['apple', 'objc', 'cpp'],",
        "    implementation = _starlark_apple_multi_library_impl,",
        ")");
    scratch.file(
        "package/BUILD",
        "load('//package:starlark_apple_multi_library.bzl', 'starlark_apple_multi_library')",
        "starlark_apple_multi_library(",
        "    name = 'main_library',",
        "    deps = [':main_lib'],",
        "    # apple_common.multi_arch_split",
        "    platform_type = 'ios',",
        "    minimum_os_version = '11.0',",
        ")",
        "objc_library(",
        "    name = 'main_lib',",
        "    srcs = ['main.m'],",
        ")");
    scratch.file("package/main.m", "int main(void) {", "  return 0;", "}");
    useConfiguration("--ios_multi_cpus=arm64,armv7,x86_64");

    ImmutableList<String> cpus = ImmutableList.of("arm64", "x86_64");
    for (String cpu : cpus) {
      CommandAction action =
          (CommandAction)
              actionProducingArtifact(
                  "//package:main_library", String.format("-%s-apple-ios-fl.a", cpu));
      assertThat(action.getArguments()).containsAtLeast("-arch_only", cpu).inOrder();
    }
  }
}
