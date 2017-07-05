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
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.APP_ICON_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.LAUNCH_IMAGE_ATTR;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMultiset;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multiset;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.BundleFile;
import java.io.IOException;
import java.util.List;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for ios_extension. */
@RunWith(JUnit4.class)
public class IosExtensionTest extends ObjcRuleTestCase {
  protected static final RuleType RULE_TYPE =
      new RuleType("ios_extension") {
        @Override
        Iterable<String> requiredAttributes(
            Scratch scratch, String packageDir, Set<String> alreadyAdded) throws IOException {
          ImmutableList.Builder<String> attributes = new ImmutableList.Builder<>();
          if (!alreadyAdded.contains("binary")) {
            scratch.file(packageDir + "/extension_binary/a.m");
            scratch.file(
                packageDir + "/extension_binary/BUILD",
                "ios_extension_binary(",
                "    name = 'extension_binary',",
                "    srcs = ['a.m'],",
                ")");
            attributes.add(String.format("binary = '//%s/extension_binary'", packageDir));
          }
          return attributes.build();
        }
      };

  protected static final BinaryRuleTypePair RULE_TYPE_PAIR =
      new BinaryRuleTypePair(
          IosExtensionBinaryTest.RULE_TYPE,
          RULE_TYPE,
          ReleaseBundlingSupport.EXTENSION_BUNDLE_DIR_FORMAT);

  private ConfiguredTarget addMockExtensionAndLibs(String... extraExtAttributes)
      throws Exception {
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    scratch.file("x/a.m");
    scratch.file("x/BUILD",
        "ios_extension_binary(",
        "    name = 'bin',",
        "    srcs = ['a.m'],",
        "    deps = ['//lib1:lib1', '//lib2:lib2'],",
        ")",
        "",
        "ios_extension(",
        "    name = 'x',",
        "    binary = ':bin',",
        Joiner.on(',').join(extraExtAttributes),
        ")");
    return getConfiguredTarget("//x:x");
  }

  @Test
  public void testSigningAction() throws Exception {
    checkDeviceSigningAction(RULE_TYPE);
  }

  @Test
  public void testSigningWithCertName() throws Exception {
    checkSigningWithCertName(RULE_TYPE);
  }

  @Test
  public void testPostProcessingAction() throws Exception {
    checkPostProcessingAction(RULE_TYPE);
  }

  @Test
  public void testSigningAndPostProcessing() throws Exception {
    checkSigningAndPostProcessing(RULE_TYPE);
  }

  @Test
  public void testSigning_simulatorBuild() throws Exception {
    checkSigningSimulatorBuild(RULE_TYPE_PAIR, false);
  }

  @Test
  public void testSigning_simulatorBuild_multiCpu() throws Exception {
    checkSigningSimulatorBuild(RULE_TYPE_PAIR, true);
  }

  @Test
  public void testProvisioningProfile_deviceBuild() throws Exception {
    checkProvisioningProfileDeviceBuild(RULE_TYPE_PAIR, false);
  }

  @Test
  public void testProvisioningProfile_deviceBuild_multiCpu() throws Exception {
    checkProvisioningProfileDeviceBuild(RULE_TYPE_PAIR, true);
  }

  @Test
  public void testUserSpecifiedProvisioningProfile_deviceBuild() throws Exception {
    checkProvisioningProfileUserSpecified(RULE_TYPE_PAIR, false);
  }

  @Test
  public void testUserSpecifiedProvisioningProfile_deviceBuild_multiCpu() throws Exception {
    checkProvisioningProfileUserSpecified(RULE_TYPE_PAIR, true);
  }

  @Test
  public void testMergeControlAction() throws Exception {
    addMockExtensionAndLibs("infoplist = 'Info.plist'");
    Action mergeAction = bundleMergeAction("//x:x");
    Action action = bundleMergeControlAction("//x:x");
    assertThat(action.getInputs()).isEmpty();
    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly(
        "x/x.ipa-control");
    assertThat(bundleMergeControl("//x:x"))
        .isEqualTo(
            BundleMergeProtos.Control.newBuilder()
                .addBundleFile(
                    BundleFile.newBuilder()
                        .setSourceFile(execPathEndingWith(mergeAction.getInputs(), "x_lipobin"))
                        .setBundlePath("x")
                        .setExternalFileAttribute(BundleableFile.EXECUTABLE_EXTERNAL_FILE_ATTRIBUTE)
                        .build())
                .setBundleRoot("PlugIns/x.appex")
                .setBundleInfoPlistFile(
                    getMergedInfoPlist(getConfiguredTarget("//x:x")).getExecPathString())
                .setOutFile(execPathEndingWith(mergeAction.getOutputs(), "x.unprocessed.ipa"))
                .setMinimumOsVersion(DEFAULT_IOS_SDK_VERSION.toString())
                .setSdkVersion(DEFAULT_IOS_SDK_VERSION.toString())
                .setPlatform("IOS_SIMULATOR")
                .setFallbackBundleIdentifier("example.x")
                .build());
  }

  @Test
  public void testMergeBundleAction() throws Exception {
    checkMergeBundleAction(RULE_TYPE_PAIR);
  }

  protected List<BuildConfiguration> getExtensionConfigurations() throws InterruptedException {
    return getSplitConfigurations(getTargetConfiguration(),
        IosExtension.MINIMUM_OS_AND_SPLIT_ARCH_TRANSITION);
  }

  @Test
  public void testErrorForLaunchImageGivenWithNoAssetCatalog() throws Exception {
    checkAssetCatalogAttributeError(RULE_TYPE, LAUNCH_IMAGE_ATTR);
  }

  @Test
  public void testErrorForAppIconGivenWithNoAssetCatalog() throws Exception {
    checkAssetCatalogAttributeError(RULE_TYPE, APP_ICON_ATTR);
  }

  @Test
  public void testCollectsAssetCatalogsTransitively() throws Exception {
    checkCollectsAssetCatalogsTransitively(RULE_TYPE_PAIR);
  }

  @Test
  public void testSpecifyAppIconAndLaunchImageUsingXcassetsOfDependency() throws Exception {
    checkSpecifyAppIconAndLaunchImageUsingXcassetsOfDependency(
        RULE_TYPE_PAIR, DEFAULT_IOS_SDK_VERSION);
  }

  private void addTargetWithAssetCatalogs() throws IOException {
    scratch.file("x/foo.xcassets/foo");
    scratch.file("x/foo.xcassets/bar");
    scratch.file("x/a.m");
    scratch.file("x/BUILD",
        "ios_extension_binary(",
        "    name = 'bin',",
        "    asset_catalogs = ['foo.xcassets/foo', 'bar.xcassets/bar'],",
        "    srcs = ['a.m'],",
        ")",
        "",
        "ios_extension(",
        "    name = 'x',",
        "    binary = ':bin',",
        ")");
  }

  @Test
  public void testActoolActionCorrectness() throws Exception {
    addTargetWithAssetCatalogs();
    checkActoolActionCorrectness(DEFAULT_IOS_SDK_VERSION);
  }

  @Test
  public void testPassesFamiliesToActool() throws Exception {
    checkPassesFamiliesToActool(RULE_TYPE_PAIR);
  }

  @Test
  public void testPassesFamiliesToIbtool() throws Exception {
    checkPassesFamiliesToIbtool(RULE_TYPE_PAIR);
  }

  @Test
  public void testReportsErrorsForInvalidFamiliesAttribute() throws Exception {
    checkReportsErrorsForInvalidFamiliesAttribute(RULE_TYPE);
  }

  @Test
  public void testMergeActionsWithAssetCatalog() throws Exception {
    addTargetWithAssetCatalogs();
    checkMergeActionsWithAssetCatalog(RULE_TYPE_PAIR);
  }

  private void addBinAndLibWithRawResources() throws Exception {
    addBinAndLibWithResources(
        "resources", "resource1.txt", "ja.lproj/resource2.txt", "ios_extension_binary");
    scratch.file("x/BUILD",
        "ios_extension(",
        "    name = 'x',",
        "    binary = '//bin:bin',",
        ")");
  }

  private void addBinAndLibWithStrings() throws Exception {
    addBinAndLibWithResources(
        "strings", "foo.strings", "ja.lproj/bar.strings", "ios_extension_binary");
    scratch.file("x/BUILD",
        "ios_extension(",
        "    name = 'x',",
        "    binary = '//bin:bin',",
        ")");
  }

  @Test
  public void testCollectsRawResourceFilesTransitively() throws Exception {
    addBinAndLibWithRawResources();
    checkCollectsResourceFilesTransitively(
        "//x:x",
        ImmutableList.of("lib/resource1.txt", "bin/ja.lproj/resource2.txt"),
        ImmutableList.of("lib/resource1.txt"),
        ImmutableSetMultimap.<String, Multiset<String>>of(
            "bin_bin", ImmutableMultiset.of("bin/ja.lproj/resource2.txt", "lib/resource1.txt"),
            "x_x", ImmutableMultiset.of("bin/ja.lproj/resource2.txt", "lib/resource1.txt"),
            "lib_lib", ImmutableMultiset.of("lib/resource1.txt")));
  }

  @Test
  public void testCollectsStringsFilesTransitively() throws Exception {
    addBinAndLibWithStrings();
    checkCollectsResourceFilesTransitively(
        "//x:x",
        ImmutableList.of("x/lib/foo.strings.binary", "x/bin/ja.lproj/bar.strings.binary"),
        ImmutableList.of("lib/foo.strings.binary"),
        ImmutableSetMultimap.<String, Multiset<String>>of(
            "bin_bin", ImmutableMultiset.of("bin/ja.lproj/bar.strings", "lib/foo.strings"),
            "x_x", ImmutableMultiset.of("bin/ja.lproj/bar.strings", "lib/foo.strings"),
            "lib_lib", ImmutableMultiset.of("lib/foo.strings")));
  }

  @Test
  public void testResourceFilesMergedInBundle() throws Exception {
    addBinAndLibWithRawResources();
    checkBundleablesAreMerged("//x:x",
        ImmutableListMultimap.of(
            "resource1.txt", "resource1.txt",
            "ja.lproj/resource2.txt", "ja.lproj/resource2.txt"));
  }

  @Test
  public void testStringsFilesMergedInBundle() throws Exception {
    addBinAndLibWithStrings();
    checkBundleablesAreMerged("//x:x",
        ImmutableListMultimap.of(
            "foo.strings.binary", "foo.strings",
            "ja.lproj/bar.strings.binary", "ja.lproj/bar.strings"));
  }

  @Test
  public void testMergesXcdatamodelZips() throws Exception {
    checkMergesXcdatamodelZips(RULE_TYPE_PAIR);
  }

  @Test
  public void testPlistRequiresDotInName() throws Exception {
    checkError("x", "x",
        "'//x:Infoplist' does not produce any ios_extension infoplist files (expected .plist)",
        "ios_extension_binary(",
        "    name = 'bin',",
        "    srcs = ['a.m'],",
        ")",
        "",
        "ios_extension(",
        "    name = 'x',",
        "    infoplist = 'Infoplist',",
        "    binary = ':bin',",
        ")");
  }

  @Test
  public void testMergesPartialInfoplists() throws Exception {
    checkMergesPartialInfoplists(RULE_TYPE_PAIR);
  }

  @Test
  public void testNibZipsMergedIntoBundle() throws Exception {
    checkNibZipsMergedIntoBundle(RULE_TYPE_PAIR);
  }

  @Test
  public void testNoEntitlementsDefined() throws Exception {
    checkNoEntitlementsDefined(RULE_TYPE);
  }

  @Test
  public void testEntitlementsDefined() throws Exception {
    checkEntitlementsDefined(RULE_TYPE);
  }

  @Test
  public void testExtraEntitlements() throws Exception {
    checkExtraEntitlements(RULE_TYPE);
  }

  @Test
  public void testDebugEntitlements() throws Exception {
    checkDebugEntitlements(RULE_TYPE);
  }

  @Test
  public void testFastbuildDebugEntitlements() throws Exception {
    checkFastbuildDebugEntitlements(RULE_TYPE);
  }

  @Test
  public void testOptNoDebugEntitlements() throws Exception {
    checkOptNoDebugEntitlements(RULE_TYPE);
  }

  @Test
  public void testExplicitNoDebugEntitlements() throws Exception {
    checkExplicitNoDebugEntitlements(RULE_TYPE);
  }

  @Test
  public void testPassesFallbackBundleIdToBundleMerging() throws Exception {
    checkBundleIdPassedAsFallbackId(RULE_TYPE);
  }

  @Test
  public void testPassesPrimaryBundleIdToBundleMerging() throws Exception {
    checkBundleIdPassedAsPrimaryId(RULE_TYPE);
  }

  @Test
  public void testMultiPlatformBuild_fails() throws Exception {
    checkBinaryActionMultiPlatform_fails(RULE_TYPE_PAIR);
  }

  @Test
  public void testMultiArchitectureResources() throws Exception {
    checkMultiCpuResourceInheritance(RULE_TYPE_PAIR);
  }

  @Test
  public void testMultiCpuCompiledResources() throws Exception {
    checkMultiCpuCompiledResources(RULE_TYPE_PAIR);
  }

  @Test
  public void testMomczipActions() throws Exception {
    checkMomczipActions(RULE_TYPE_PAIR, DEFAULT_IOS_SDK_VERSION);
  }

  @Test
  public void testConvertStringsActions() throws Exception {
    checkConvertStringsAction(RULE_TYPE_PAIR);
  }

  @Test
  public void testCompileXibActions() throws Exception {
    checkCompileXibActions(RULE_TYPE_PAIR, DEFAULT_IOS_SDK_VERSION, "iphone");
  }

  @Test
  public void testRegistersStoryboardCompileActions() throws Exception {
    checkRegistersStoryboardCompileActions(
        RULE_TYPE_PAIR, DEFAULT_IOS_SDK_VERSION, "iphone");
  }

  @Test
  public void testMultiCpuCompiledResourcesFromGenrule() throws Exception {
    checkMultiCpuCompiledResourcesFromGenrule(RULE_TYPE_PAIR);
  }

  @Test
  public void testMultiCpuGeneratedResourcesFromGenrule() throws Exception {
    checkMultiCpuGeneratedResourcesFromGenrule(RULE_TYPE_PAIR);
  }

  @Test
  public void testTwoStringsOneBundlePath() throws Exception {
    checkTwoStringsOneBundlePath(RULE_TYPE_PAIR, "x");
  }

  @Test
  public void testTwoResourcesOneBundlePath() throws Exception {
    checkTwoResourcesOneBundlePath(RULE_TYPE_PAIR, "x");
  }

  @Test
  public void testSameStringsTwice() throws Exception {
    checkSameStringsTwice(RULE_TYPE_PAIR, "x");
  }

  @Test
  public void testExtensionReplacesMinimumOsInBundleMerge() throws Exception {
    useConfiguration("--ios_minimum_os=7.1");
    addMockExtensionAndLibs("infoplist = 'Info.plist'");

    assertThat(bundleMergeControl("//x:x").getMinimumOsVersion())
        .isEqualTo(IosExtension.EXTENSION_MINIMUM_OS_VERSION.toString());
  }

  @Test
  public void testExtensionReplacesMinimumOsVersionInBundleMergeAtMost80() throws Exception {
    useConfiguration("--ios_minimum_os=8.1");
    addMockExtensionAndLibs("infoplist = 'Info.plist'");

    assertThat(bundleMergeControl("//x:x").getMinimumOsVersion())
        .isEqualTo("8.1");
  }

  @Test
  public void testCheckPrimaryBundleIdInMergedPlist() throws Exception {
    checkPrimaryBundleIdInMergedPlist(RULE_TYPE_PAIR);
  }

  @Test
  public void testCheckFallbackBundleIdInMergedPlist() throws Exception {
    checkFallbackBundleIdInMergedPlist(RULE_TYPE_PAIR);
  }

  protected void checkExtensionReplacesMinimumOsInCompilation() throws Exception {
    addMockExtensionAndLibs("infoplist = 'Info.plist'");

    Action lipoAction = lipoBinAction("//x:x");

    for (Artifact bin : lipoAction.getInputs()) {
      CommandAction action = (CommandAction) getGeneratingAction(bin);
      if (action == null) {
        continue;
      }
      assertThat(generatingArgumentsToString(action))
          .contains("-mios-simulator-version-min=" + IosExtension.EXTENSION_MINIMUM_OS_VERSION);
      assertThat(generatingArgumentsToString(action))
          .doesNotContain("-mios-simulator-version-min=7.1");
    }
  }

  private String generatingArgumentsToString(CommandAction generatingAction) {
    return Joiner.on(' ').join(generatingAction.getArguments());
  }

  protected void checkExtensionDoesNotReplaceMinimumOsInCompilation() throws Exception {
    addMockExtensionAndLibs("infoplist = 'Info.plist'");

    Action lipoAction = lipoBinAction("//x:x");

    for (Artifact bin : lipoAction.getInputs()) {
      CommandAction action = (CommandAction) getGeneratingAction(bin);
      if (action == null) {
        continue;
      }
      assertThat(generatingArgumentsToString(action)).contains("-mios-simulator-version-min=8.1");
      assertThat(generatingArgumentsToString(action))
          .doesNotContain("-mios-simulator-version-min=" + DEFAULT_IOS_SDK_VERSION);
    }
  }

  @Test
  public void testExtensionReplacesMinimumOsVersionInMomcZipAtMost80() throws Exception {
    useConfiguration("--ios_minimum_os=8.1");
    checkMomczipActions(RULE_TYPE_PAIR, DottedVersion.fromString("8.1"));
  }

  @Test
  public void testGenruleWithoutJavaCcDeps() throws Exception {
    checkGenruleWithoutJavaCcDependency(RULE_TYPE_PAIR);
  }

  @Test
  public void testCcDependencyWithProtoDependencyMultiArch() throws Exception {
    checkCcDependencyWithProtoDependencyMultiArch(
        RULE_TYPE_PAIR, ConfigurationDistinguisher.IOS_EXTENSION);
  }

  @Test
  public void testLaunchStoryboardIncluded() throws Exception {
    checkLaunchStoryboardIncluded(RULE_TYPE_PAIR);
  }

  @Test
  public void testLaunchStoryboardXibIncluded() throws Exception {
    checkLaunchStoryboardXib(RULE_TYPE_PAIR);
  }

  @Test
  public void testLaunchStoryboardLproj() throws Exception {
    checkLaunchStoryboardLproj(RULE_TYPE_PAIR);
  }
  
  @Test
  public void testAutomaticPlistEntries() throws Exception {
    checkAutomaticPlistEntries(RULE_TYPE);
  }

  @Test
  public void testBundleMergeInputContainsPlMergeOutput() throws Exception {
    checkBundleMergeInputContainsPlMergeOutput(RULE_TYPE);
  }

  @Test
  public void testMergeBundleActionsWithNestedBundle() throws Exception {
    BuildConfiguration extensionConfiguration =
        Iterables.getOnlyElement(getExtensionConfigurations());
    checkMergeBundleActionsWithNestedBundle(RULE_TYPE_PAIR, extensionConfiguration);
  }

  @Test
  public void testIncludesStoryboardOutputZipsAsMergeZips() throws Exception {
    BuildConfiguration extensionConfiguration =
        Iterables.getOnlyElement(getExtensionConfigurations());
    checkIncludesStoryboardOutputZipsAsMergeZips(RULE_TYPE_PAIR, extensionConfiguration);
  }

  @Test
  public void testCcDependency() throws Exception {
    checkCcDependency(RULE_TYPE_PAIR, ConfigurationDistinguisher.IOS_EXTENSION);
  }

  @Test
  public void testCcDependencyMultiArch() throws Exception {
    checkCcDependencyMultiArch(RULE_TYPE_PAIR, ConfigurationDistinguisher.IOS_EXTENSION);
  }

  @Test
  public void testCcDependencyWithProtoDependency() throws Exception {
    checkCcDependencyWithProtoDependency(RULE_TYPE_PAIR, ConfigurationDistinguisher.IOS_EXTENSION);
  }

  @Test
  public void testCcDependencyAndJ2objcDependency() throws Exception {
    checkCcDependencyAndJ2objcDependency(RULE_TYPE_PAIR, ConfigurationDistinguisher.IOS_EXTENSION);
  }

  @Test
  public void testMultiArchitectureFanOut() throws Exception {
    checkBinaryLipoActionMultiCpu(RULE_TYPE_PAIR, ConfigurationDistinguisher.IOS_EXTENSION);
  }

  @Test
  public void testTargetHasCpuSpecificDsymFiles() throws Exception {
    checkTargetHasCpuSpecificDsymFiles(RULE_TYPE);
  }

  @Test
  public void testTargetHasDsymPlist() throws Exception {
    checkTargetHasDsymPlist(RULE_TYPE);
  }

  @Test
  public void testGenruleDependencyMultiArch() throws Exception {
    checkGenruleDependencyMultiArch(RULE_TYPE_PAIR, ConfigurationDistinguisher.IOS_EXTENSION);
  }

  @Test
  public void testExtensionReplacesMinimumOsInCompilation() throws Exception {
    useConfiguration("--ios_minimum_os=7.1");
    checkExtensionReplacesMinimumOsInCompilation();
  }

  @Test
  public void testExtensionReplacesMinimumOsInCompilationAtMost80() throws Exception {
    useConfiguration("--ios_minimum_os=8.1");
    checkExtensionDoesNotReplaceMinimumOsInCompilation();
  }

  @Test
  public void testExtensionReplacesMinimumOsInCompilationMultiArch() throws Exception {
    useConfiguration("--ios_minimum_os=7.1", "--ios_multi_cpus=i386,x86_64");
    checkExtensionReplacesMinimumOsInCompilation();
  }

  @Test
  public void testExtensionReplacesMinimumOsInCompilationAtMost80MultiArch() throws Exception {
    useConfiguration("--ios_minimum_os=8.1", "--ios_multi_cpus=i386,x86_64");
    checkExtensionDoesNotReplaceMinimumOsInCompilation();
  }
}
