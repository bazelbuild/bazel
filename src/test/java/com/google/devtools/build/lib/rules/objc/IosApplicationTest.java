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

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMultiset;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.BundleFile;
import com.google.devtools.build.xcode.plmerge.proto.PlMergeProtos;
import java.io.IOException;
import java.util.List;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for ios_application. */
@RunWith(JUnit4.class)
public class IosApplicationTest extends ObjcRuleTestCase {
  protected static final RuleType RULE_TYPE =
      new RuleType("ios_application") {
        @Override
        Iterable<String> requiredAttributes(
            Scratch scratch, String packageDir, Set<String> alreadyAdded) throws IOException {
          ImmutableList.Builder<String> attributes = new ImmutableList.Builder<>();
          if (!alreadyAdded.contains("binary")) {
            scratch.file(packageDir + "/bin/a.m");
            scratch.file(packageDir + "/bin/BUILD", "objc_binary(name = 'bin', srcs = ['a.m'])");
            attributes.add("binary = '//" + packageDir + "/bin:bin'");
          }
          return attributes.build();
        }
      };

  protected static final BinaryRuleTypePair RULE_TYPE_PAIR =
      new BinaryRuleTypePair(
          ObjcBinaryTest.RULE_TYPE, RULE_TYPE, ReleaseBundlingSupport.APP_BUNDLE_DIR_FORMAT);

  private ConfiguredTarget addMockAppAndLibs(String... extraAppAttributes)
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
    scratch.file("x/x-Info.plist");
    scratch.file("x/BUILD",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['a.m'],",
        "    deps = ['//lib1:lib1', '//lib2:lib2'],",
        ")",
        "",
        "ios_application(",
        "    name = 'x',",
        "    binary = ':bin',",
        Joiner.on(',').join(extraAppAttributes),
        ")");
    return getConfiguredTarget("//x:x");
  }

  @Test
  public void testSplitConfigurationProviders() throws Exception {
    useConfiguration("--ios_multi_cpus=i386,x86_64");
    scratch.file("x/BUILD",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['a.m'],",
        ")",
        "",
        "ios_application(",
        "    name = 'x',",
        "    infoplist = 'Info.plist',",
        "    binary = ':bin',",
        ")");
    RuleContext ruleContext = getRuleContext(getConfiguredTarget("//x:x"));
    ImmutableListMultimap<BuildConfiguration, ObjcProvider> prereqByConfig =
        ruleContext.getPrerequisitesByConfiguration(
            "binary", Mode.SPLIT, ObjcProvider.SKYLARK_CONSTRUCTOR);
    List<String> childCpus = Lists.transform(prereqByConfig.keySet().asList(),
        new Function<BuildConfiguration, String>() {
          @Override
          public String apply(BuildConfiguration config) {
            return config.getFragment(AppleConfiguration.class).getIosCpu();
          }
        });
    assertThat(childCpus).containsExactly("i386", "x86_64");
  }

  @Test
  public void testRunfiles() throws Exception {
    ConfiguredTarget application = addMockAppAndLibs();
    RunfilesProvider runfiles = application.getProvider(RunfilesProvider.class);
    assertThat(runfiles.getDefaultRunfiles().getArtifacts()).isEmpty();
    assertThat(Artifact.toRootRelativePaths(runfiles.getDataRunfiles().getArtifacts()))
        .containsExactly("x/x.ipa");
  }

  @Test
  public void testFilesToRun() throws Exception {
    checkFilesToRun(RULE_TYPE);
  }

  @Test
  public void testNoRunfilesSupportForDevice() throws Exception {
    checkNoRunfilesSupportForDevice(RULE_TYPE);
  }

  @Test
  public void testGenerateRunnerScriptAction() throws Exception {
    checkGenerateRunnerScriptAction(RULE_TYPE);
  }

  @Test
  public void testGenerateRunnerScriptAction_escaped() throws Exception {
    checkGenerateRunnerScriptAction_escaped(RULE_TYPE);
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
    addMockAppAndLibs("infoplist = 'Info.plist'");
    Action mergeAction = bundleMergeAction("//x:x");
    Action action = bundleMergeControlAction("//x:x");
    assertThat(action.getInputs()).isEmpty();
    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("x/x.ipa-control");
    assertThat(bundleMergeControl("//x:x"))
        .isEqualTo(
            BundleMergeProtos.Control.newBuilder()
                .addBundleFile(
                    BundleFile.newBuilder()
                        .setSourceFile(execPathEndingWith(mergeAction.getInputs(), "x_lipobin"))
                        .setBundlePath("x")
                        .setExternalFileAttribute(BundleableFile.EXECUTABLE_EXTERNAL_FILE_ATTRIBUTE)
                        .build())
                .setBundleRoot(String.format(ReleaseBundlingSupport.APP_BUNDLE_DIR_FORMAT, "x"))
                .setBundleInfoPlistFile(execPathEndingWith(mergeAction.getInputs(), "Info.plist"))
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

  @Test
  public void testCheckPrimaryBundleIdInMergedPlist() throws Exception {
    checkPrimaryBundleIdInMergedPlist(RULE_TYPE_PAIR);
  }

  @Test
  public void testCheckFallbackBundleIdInMergedPlist() throws Exception {
    checkFallbackBundleIdInMergedPlist(RULE_TYPE_PAIR);
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
        "objc_binary(",
        "    name = 'bin',",
        "    asset_catalogs = ['foo.xcassets/foo', 'bar.xcassets/bar'],",
        "    srcs = ['a.m'],",
        ")",
        "",
        "ios_application(",
        "    name = 'x',",
        "    binary = ':bin',",
        ")");
  }

  @Test
  public void testActoolAction() throws Exception {
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
    // TODO(matvore): add this test to IosTestTest.java.
    addTargetWithAssetCatalogs();
    checkMergeActionsWithAssetCatalog(RULE_TYPE_PAIR);
  }

  private void addBinAndLibWithRawResources() throws Exception {
    addBinAndLibWithResources(
        "resources", "resource1.txt", "ja.lproj/resource2.txt", "objc_binary");
    scratch.file("x/BUILD",
        "ios_application(",
        "    name = 'x',",
        "    binary = '//bin:bin',",
        ")");
  }

  private void addBinAndLibWithStrings() throws Exception {
    addBinAndLibWithResources(
        "strings", "foo.strings", "ja.lproj/bar.strings", "objc_binary");
    scratch.file("x/BUILD",
        "ios_application(",
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
            "bin_bin",
            ImmutableMultiset.of("bin/ja.lproj/resource2.txt", "lib/resource1.txt"),
            "bin_static_lib_bin",
            ImmutableMultiset.of("bin/ja.lproj/resource2.txt", "lib/resource1.txt"),
            "x_x",
            ImmutableMultiset.of("bin/ja.lproj/resource2.txt", "lib/resource1.txt"),
            "lib_lib",
            ImmutableMultiset.of("lib/resource1.txt")));
  }

  @Test
  public void testCollectsStringsFilesTransitively() throws Exception {
    addBinAndLibWithStrings();
    checkCollectsResourceFilesTransitively(
        "//x:x",
        ImmutableList.of("x/lib/foo.strings.binary", "x/bin/ja.lproj/bar.strings.binary"),
        ImmutableList.of("x/lib/foo.strings.binary"),
        ImmutableSetMultimap.<String, Multiset<String>>of(
            "bin_bin",
            ImmutableMultiset.of("bin/ja.lproj/bar.strings", "lib/foo.strings"),
            "bin_static_lib_bin",
            ImmutableMultiset.of("bin/ja.lproj/bar.strings", "lib/foo.strings"),
            "x_x",
            ImmutableMultiset.of("bin/ja.lproj/bar.strings", "lib/foo.strings"),
            "lib_lib",
            ImmutableMultiset.of("lib/foo.strings")));
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
  public void testResourceFlattenedInBundle() throws Exception {
    addBinAndLibWithResources(
        "resources", "libres/resource1.txt", "binres/resource2.txt", "objc_binary");
    scratch.file("x/BUILD",
        "ios_application(",
        "    name = 'x',",
        "    binary = '//bin:bin',",
        ")");
    checkBundleablesAreMerged("//x:x",
        ImmutableListMultimap.of(
            "libres/resource1.txt", "resource1.txt",
            "binres/resource2.txt", "resource2.txt"));
  }

  @Test
  public void testStructuredResourceFilesMergedInBundle() throws Exception {
    addBinAndLibWithResources(
        "structured_resources", "libres/resource1.txt", "binres/resource2.txt", "objc_binary");
    scratch.file("x/BUILD",
        "ios_application(",
        "    name = 'x',",
        "    binary = '//bin:bin',",
        ")");
    checkBundleablesAreMerged("//x:x",
        ImmutableListMultimap.of(
            "libres/resource1.txt", "libres/resource1.txt",
            "binres/resource2.txt", "binres/resource2.txt"));
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
        "'//x:Infoplist' does not produce any ios_application infoplist files (expected .plist)",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['a.m'],",
        ")",
        "",
        "ios_application(",
        "    name = 'x',",
        "    infoplist = 'Infoplist',",
        "    binary = ':bin',",
        ")");
  }

  @Test
  public void testPopulatesBundling() throws Exception {
    scratch.file("x/x-Info.plist");
    scratch.file("x/a.m");
    scratch.file("x/assets.xcassets/1");
    scratch.file("x/BUILD",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['a.m'],",
        "    asset_catalogs = ['assets.xcassets/1']",
        ")",
        "ios_application(",
        "    name = 'x',",
        "    binary = ':bin',",
        "    infoplist = 'x-Info.plist',",
        ")");

    PlMergeProtos.Control control = plMergeControl("//x:x");
    assertThat(control.getSourceFileList())
        .contains(getSourceArtifact("x/x-Info.plist").getExecPathString());

    Artifact actoolzipOutput = getBinArtifact("x.actool.zip", "//x:x");
    assertThat(getGeneratingAction(actoolzipOutput).getInputs())
        .contains(getSourceArtifact("x/assets.xcassets/1"));
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
  public void testMultiPlatformBuild_fails() throws Exception {
    checkBinaryActionMultiPlatform_fails(RULE_TYPE_PAIR);
  }

  @Test
  public void testMultiArchitectureResources() throws Exception {
    checkMultiCpuResourceInheritance(RULE_TYPE_PAIR);
  }

  /**
   * Regression test for b/27946171. Verifies that nodistinct_host_configuration functions in
   * builds with more than one split transition. (In this case, both ios_application and
   * ios_extension split into two child configurations.)
   */
  @Test
  public void testNoDistinctHostConfiguration() throws Exception {
    useConfiguration("--ios_multi_cpus=i386,x86_64", "--nodistinct_host_configuration");
    scratch.file("x/BUILD",
        "ios_extension_binary(",
        "    name = 'ext_bin',",
        "    srcs = ['ebin.m'],",
        ")",
        "",
        "ios_extension(",
        "    name = 'ext',",
        "    binary = ':ext_bin',",
        ")",
        "",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        ")",
        "",
        "ios_application(",
        "    name = 'app',",
        "    binary = ':bin',",
        "    extensions = [':ext'],",
        ")");

    getConfiguredTarget("//x:app");

    // Assert that only the deprecation warnings are emitted, but no other events.
    assertContainsEventWithFrequency(
        "This rule is deprecated. Please use the new Apple build rules "
            + "(https://github.com/bazelbuild/rules_apple) to build Apple targets.",
        4);
  }

  @Test
  public void testApplicationExtensionSharedDependencyResourceActions() throws Exception {
    useConfiguration("--ios_multi_cpus=i386,x86_64");
    scratch.file("x/BUILD",
        "objc_library(",
        "    name = 'res',",
        "    xibs = ['interface.xib'],",
        "    storyboards = ['story.storyboard'],",
        "    datamodels = ['data.xcdatamodel/1'],",
        "    asset_catalogs = ['assets.xcassets/foo'],",
        ")",
        "",
        "ios_extension_binary(",
        "    name = 'ext_bin',",
        "    srcs = ['ebin.m'],",
        "    deps = [':res'],",
        ")",
        "",
        "ios_extension(",
        "    name = 'ext',",
        "    binary = ':ext_bin',",
        ")",
        "",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        "    deps = [':res'],",
        ")",
        "",
        "ios_application(",
        "    name = 'app',",
        "    binary = ':bin',",
        "    extensions = [':ext'],",
        ")");

    Action appIpaAction = bundleMergeAction("//x:app");

    Action extIpaAction = bundleMergeAction("//x:ext");

    Artifact appNibZip = Iterables.getOnlyElement(inputsEndingWith(appIpaAction, "nib.zip"));
    Artifact extNibZip = Iterables.getOnlyElement(inputsEndingWith(extIpaAction, "nib.zip"));
    assertThat(appNibZip.getExecPath()).isNotEqualTo(extNibZip.getExecPath());

    Artifact appStoryboardZip =
        Iterables.getOnlyElement(inputsEndingWith(appIpaAction, "story.storyboard.zip"));
    Artifact extStoryboardZip =
        Iterables.getOnlyElement(inputsEndingWith(extIpaAction, "story.storyboard.zip"));
    assertThat(appStoryboardZip.getExecPath()).isNotEqualTo(extStoryboardZip.getExecPath());

    Artifact appDatamodelZip = Iterables.getOnlyElement(inputsEndingWith(appIpaAction, "data.zip"));
    Artifact extDatamodelZip = Iterables.getOnlyElement(inputsEndingWith(extIpaAction, "data.zip"));
    assertThat(appDatamodelZip.getExecPath()).isNotEqualTo(extDatamodelZip.getExecPath());

    Artifact appAssetZip = Iterables.getOnlyElement(inputsEndingWith(appIpaAction, "actool.zip"));
    Artifact extAssetZip = Iterables.getOnlyElement(inputsEndingWith(extIpaAction, "actool.zip"));
    assertThat(appAssetZip.getExecPath()).isNotEqualTo(extAssetZip.getExecPath());
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
    checkRegistersStoryboardCompileActions(RULE_TYPE_PAIR, DEFAULT_IOS_SDK_VERSION, "iphone");
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
    //TODO(bazel-team): This error should be on //x:x but shows up on :bin right now until that
    // doesn't support bundling anymore.
    checkTwoStringsOneBundlePath(RULE_TYPE_PAIR, "bin");
  }

  @Test
  public void testTwoResourcesOneBundlePath() throws Exception {
    //TODO(bazel-team): This error should be on //x:x but shows up on :bin right now until that
    // doesn't support bundling anymore.
    checkTwoResourcesOneBundlePath(RULE_TYPE_PAIR, "bin");
  }

  @Test
  public void testSameStringsTwice() throws Exception {
    //TODO(bazel-team): This error should be on //x:x but shows up on :bin right now until that
    // doesn't support bundling anymore.
    checkSameStringsTwice(RULE_TYPE_PAIR, "bin");
  }

  @Test
  public void testGenruleWithoutJavaCcDeps() throws Exception {
    checkGenruleWithoutJavaCcDependency(RULE_TYPE_PAIR);
  }

  @Test
  public void testCcDependencyWithProtoDependencyMultiArch() throws Exception {
    checkCcDependencyWithProtoDependencyMultiArch(
        RULE_TYPE_PAIR, ConfigurationDistinguisher.IOS_APPLICATION);
  }

  @Test
  public void testAppleSdkVersionEnv() throws Exception {
    RULE_TYPE_PAIR.scratchTargets(scratch);

    useConfiguration("--ios_multi_cpus=x86_64,i386");
    SpawnAction action = (SpawnAction) getGeneratingAction(
        getBinArtifact("x_lipobin", getConfiguredTarget("//x:x")));

    assertAppleSdkVersionEnv(action);
  }

  @Test
  public void testNonDefaultAppleSdkVersionEnv() throws Exception {
    RULE_TYPE_PAIR.scratchTargets(scratch);

    useConfiguration("--ios_sdk_version=8.1", "--ios_multi_cpus=x86_64,i386");
    SpawnAction action = (SpawnAction) getGeneratingAction(
        getBinArtifact("x_lipobin", getConfiguredTarget("//x:x")));

    assertAppleSdkVersionEnv(action, "8.1");
  }

  @Test
  public void testAppleSdkDefaultPlatformEnv() throws Exception {
    RULE_TYPE_PAIR.scratchTargets(scratch);

    useConfiguration("--ios_multi_cpus=x86_64,i386");
    SpawnAction action = (SpawnAction) getGeneratingAction(
        getBinArtifact("x_lipobin", getConfiguredTarget("//x:x")));

    assertAppleSdkPlatformEnv(action, "iPhoneSimulator");
  }

  @Test
  public void testAppleSdkDevicePlatformEnv() throws Exception {
    RULE_TYPE_PAIR.scratchTargets(scratch);

    useConfiguration("--ios_multi_cpus=arm64,armv7");
    SpawnAction action = (SpawnAction) getGeneratingAction(
        getBinArtifact("x_lipobin", getConfiguredTarget("//x:x")));

    assertAppleSdkPlatformEnv(action, "iPhoneOS");
  }

  @Test
  public void testXcodeVersionEnv() throws Exception {
    RULE_TYPE_PAIR.scratchTargets(scratch);

    useConfiguration("--xcode_version=5.8", "--ios_multi_cpus=x86_64,i386");
    SpawnAction action = (SpawnAction) getGeneratingAction(
        getBinArtifact("x_lipobin", getConfiguredTarget("//x:x")));

    assertXcodeVersionEnv(action, "5.8");
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
  public void testMultipleInfoPlists() throws Exception {
    checkMultipleInfoPlists(RULE_TYPE);
  }

  @Test
  public void testInfoplistAndInfoplistsTogether() throws Exception {
    checkInfoplistAndInfoplistsTogether(RULE_TYPE);
  }

  @Test
  public void testLateLoadedObjcFrameworkInFinalBundle() throws Exception {
    scratch.file("x/Foo.framework/Foo");
    scratch.file("x/Foo.framework/Info.plist");
    scratch.file("x/Foo.framework/Headers/Foo.h");
    scratch.file("x/Foo.framework/Resources/bar.png");
    scratch.file(
        "x/BUILD",
        "objc_framework(",
        "    name = 'foo_framework',",
        "    framework_imports = glob(['Foo.framework/**']),",
        "    is_dynamic = 1,",
        ")",
        "",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = [ 'a.m' ],",
        "    runtime_deps = [ ':foo_framework' ],",
        ")",
        "",
        "ios_application(",
        "    name = 'x',",
        "    binary = ':bin',",
        ")");

    BundleMergeProtos.Control mergeControl = bundleMergeControl("//x:x");

    assertThat(mergeControl.getBundleFileList())
        .containsAllOf(
            BundleFile.newBuilder()
                .setBundlePath("Frameworks/Foo.framework/Foo")
                .setSourceFile(getSourceArtifact("x/Foo.framework/Foo").getExecPathString())
                .setExternalFileAttribute(BundleableFile.EXECUTABLE_EXTERNAL_FILE_ATTRIBUTE)
                .build(),
            BundleFile.newBuilder()
                .setBundlePath("Frameworks/Foo.framework/Info.plist")
                .setSourceFile(getSourceArtifact("x/Foo.framework/Info.plist").getExecPathString())
                .setExternalFileAttribute(BundleableFile.EXECUTABLE_EXTERNAL_FILE_ATTRIBUTE)
                .build(),
            BundleFile.newBuilder()
                .setBundlePath("Frameworks/Foo.framework/Resources/bar.png")
                .setSourceFile(
                    getSourceArtifact("x/Foo.framework/Resources/bar.png").getExecPathString())
                .setExternalFileAttribute(BundleableFile.DEFAULT_EXTERNAL_FILE_ATTRIBUTE)
                .build());

    assertThat(mergeControl.getBundleFileList())
        .doesNotContain(
            BundleFile.newBuilder()
                .setBundlePath("Frameworks/Foo.framework/Headers/Foo.h")
                .setSourceFile(
                    getSourceArtifact("x/Foo.framework/Headers/Foo.h").getExecPathString())
                .setExternalFileAttribute(BundleableFile.DEFAULT_EXTERNAL_FILE_ATTRIBUTE)
                .build());
  }

  @Test
  public void testLateloadedObjcFrameworkSigned() throws Exception {
    useConfiguration("--cpu=ios_arm64");

    scratch.file("x/Foo.framework/Foo");
    scratch.file("x/Foo.framework/Info.plist");
    scratch.file("x/Foo.framework/Headers/Foo.h");
    scratch.file("x/Foo.framework/Resources/bar.png");
    scratch.file(
        "x/BUILD",
        "objc_framework(",
        "    name = 'foo_framework',",
        "    framework_imports = glob(['Foo.framework/**']),",
        "    is_dynamic = 1,",
        ")",
        "",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = [ 'a.m' ],",
        "    runtime_deps = [ ':foo_framework' ],",
        ")",
        "",
        "ios_application(",
        "    name = 'x',",
        "    binary = ':bin',",
        ")");

    SpawnAction signingAction = (SpawnAction) ipaGeneratingAction();

    assertThat(normalizeBashArgs(signingAction.getArguments()))
        .containsAllOf("--sign", "${t}/Payload/x.app/Frameworks/*", "--sign", "${t}/Payload/x.app")
        .inOrder();
  }

  @Test
  public void aspectOnSplitAttributeRegressionTest() throws Exception {
    useConfiguration("--ios_multi_cpus=armv7,arm64");
    scratch.file("x/a.m");
    scratch.file("x/x-Info.plist");
    scratch.file(
        "x/extension.bzl",
        "def _my_aspect_impl(target, ctx):",
        "  if type(ctx.rule.attr.binary) != 'list':",
        "      fail('Expected a list for split')",
        "  if len(ctx.rule.attr.binary) != 2:",
        "      fail('Expected 2 items in split')",
        "  return struct()",
        "my_aspect = aspect(_my_aspect_impl)",
        "def _my_rule_impl(ctx):",
        "  pass",
        "my_rule = rule(_my_rule_impl, attrs = { 'deps' : attr.label_list(aspects = [my_aspect]) })"
    );
    scratch.file("x/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "objc_binary(name = 'bin', srcs = ['a.m'], )",
        "ios_application(name = 'x', binary = ':bin',)",
        "my_rule(name = 'y', deps = [ ':x' ])"
    );
    getConfiguredTarget("//x:y");
  }

  @Test
  public void aspectOnSplitAttributeNoSplitRegressionTest() throws Exception {
    useConfiguration("--ios_multi_cpus=arm64");
    scratch.file("x/a.m");
    scratch.file("x/x-Info.plist");
    scratch.file(
        "x/extension.bzl",
        "def _my_aspect_impl(target, ctx):",
        "  if type(ctx.rule.attr.binary) != 'list':",
        "      fail('Expected a list for split')",
        "  if len(ctx.rule.attr.binary) != 1:",
        "      fail('Expected 1 items in split')",
        "  return struct()",
        "my_aspect = aspect(_my_aspect_impl)",
        "def _my_rule_impl(ctx):",
        "  pass",
        "my_rule = rule(_my_rule_impl, attrs = { 'deps' : attr.label_list(aspects = [my_aspect]) })"
    );
    scratch.file("x/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "objc_binary(name = 'bin', srcs = ['a.m'], )",
        "ios_application(name = 'x', binary = ':bin',)",
        "my_rule(name = 'y', deps = [ ':x' ])"
    );
    getConfiguredTarget("//x:y");
  }

  @Test
  public void testMergeBundleActionsWithNestedBundle() throws Exception {
    checkMergeBundleActionsWithNestedBundle(RULE_TYPE_PAIR, targetConfig);
  }

  @Test
  public void testIncludesStoryboardOutputZipsAsMergeZips() throws Exception {
    checkIncludesStoryboardOutputZipsAsMergeZips(RULE_TYPE_PAIR, targetConfig);
  }

  @Test
  public void testCcDependency() throws Exception {
    checkCcDependency(RULE_TYPE_PAIR, ConfigurationDistinguisher.IOS_APPLICATION);
  }

  @Test
  public void testCcDependencyMultiArch() throws Exception {
    checkCcDependencyMultiArch(RULE_TYPE_PAIR, ConfigurationDistinguisher.IOS_APPLICATION);
  }

  @Test
  public void testCCDependencyWithProtoDependency() throws Exception {
    checkCcDependencyWithProtoDependency(
        RULE_TYPE_PAIR, ConfigurationDistinguisher.IOS_APPLICATION);
  }

  @Test
  public void testCcDependencyAndJ2objcDependency() throws Exception {
    checkCcDependencyAndJ2objcDependency(
        RULE_TYPE_PAIR, ConfigurationDistinguisher.IOS_APPLICATION);
  }

  @Test
  public void testApplicationExtension() throws Exception {
    // Including minimum OS version to trigger a special code path in extension split transitions
    // which have a higher chance of conflicting with application transitions. See flag
    // --DO_NOT_USE_configuration_distinguisher for details.
    useConfiguration("--ios_multi_cpus=i386,x86_64", "--ios_minimum_os=8.1");
    DottedVersion minOsString = DottedVersion.fromString("8.1");
    scratch.file(
        "x/BUILD",
        "ios_extension_binary(",
        "    name = 'ext_bin',",
        "    srcs = ['ebin.m'],",
        ")",
        "",
        "ios_extension(",
        "    name = 'ext',",
        "    binary = ':ext_bin',",
        ")",
        "",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        ")",
        "",
        "ios_application(",
        "    name = 'app',",
        "    binary = ':bin',",
        "    extensions = [':ext'],",
        ")");

    SpawnAction appLipoAction =
        (SpawnAction)
            getGeneratingAction(
                getBinArtifact(
                    "app_lipobin", getConfiguredTarget("//x:app", getTargetConfiguration())));

    assertThat(Artifact.toExecPaths(appLipoAction.getInputs()))
        .containsExactly(
            configurationBin("i386", ConfigurationDistinguisher.IOS_APPLICATION, minOsString)
                + "x/bin_bin",
            configurationBin("x86_64", ConfigurationDistinguisher.IOS_APPLICATION, minOsString)
                + "x/bin_bin",
            MOCK_XCRUNWRAPPER_PATH);

    SpawnAction extLipoAction =
        (SpawnAction)
            getGeneratingAction(
                getBinArtifact(
                    "ext_lipobin", getConfiguredTarget("//x:ext", getTargetConfiguration())));

    assertThat(Artifact.toExecPaths(extLipoAction.getInputs()))
        .containsExactly(
            configurationBin("i386", ConfigurationDistinguisher.IOS_EXTENSION, minOsString)
                + "x/ext_bin_bin",
            configurationBin("x86_64", ConfigurationDistinguisher.IOS_EXTENSION, minOsString)
                + "x/ext_bin_bin", MOCK_XCRUNWRAPPER_PATH);
  }

  @Test
  public void testGenruleDependency() throws Exception {
    checkGenruleDependency(RULE_TYPE_PAIR);
  }

  @Test
  public void testGenruleDependencyMultiArch() throws Exception {
    checkGenruleDependencyMultiArch(RULE_TYPE_PAIR, ConfigurationDistinguisher.IOS_APPLICATION);
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
  public void testMultiArchitectureFanOut() throws Exception {
    checkBinaryLipoActionMultiCpu(RULE_TYPE_PAIR, ConfigurationDistinguisher.IOS_APPLICATION);
  }

  @Test
  public void testMultiArchitectureWithConfigurableAttribute() throws Exception {
    useConfiguration("--ios_multi_cpus=armv7,arm64", "--cpu=ios_i386");
    scratch.file(
        "x/BUILD",
        "config_setting(",
        "    name = 'i386',",
        "    values = {'cpu': 'ios_i386'},",
        ")",
        "",
        "config_setting(",
        "    name = 'armv7',",
        "    values = {'cpu': 'ios_armv7'},",
        ")",
        "",
        "objc_library(",
        "    name = 'libi386',",
        "    srcs = ['i386.m'],",
        ")",
        "",
        "objc_library(",
        "    name = 'libarmv7',",
        "    srcs = ['armv7.m'],",
        ")",
        "",
        "objc_library(",
        "    name = 'libdefault',",
        "    srcs = ['default.m'],",
        ")",
        "",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        "    deps = select({",
        "        ':i386': [':libi386'],",
        "        ':armv7': [':libarmv7'],",
        "        '//conditions:default': [':libdefault'],",
        "    }),",
        ")",
        "",
        "ios_application(",
        "    name = 'app',",
        "    binary = ':bin',",
        ")");

    CommandAction appLipoAction =
        (CommandAction)
            getGeneratingAction(
                getBinArtifact(
                    "app_lipobin", getConfiguredTarget("//x:app", getTargetConfiguration())));

    assertThat(Artifact.toExecPaths(appLipoAction.getInputs()))
        .containsExactly(
            configurationBin("armv7", ConfigurationDistinguisher.IOS_APPLICATION) + "x/bin_bin",
            configurationBin("arm64", ConfigurationDistinguisher.IOS_APPLICATION) + "x/bin_bin",
            MOCK_XCRUNWRAPPER_PATH);

    ImmutableSet.Builder<Artifact> binInputs = ImmutableSet.builder();
    for (Artifact bin : appLipoAction.getInputs()) {
      CommandAction binAction = (CommandAction) getGeneratingAction(bin);
      if (binAction != null) {
        binInputs.addAll(binAction.getInputs());
      }
    }

    assertThat(Artifact.toExecPaths(binInputs.build()))
        .containsAllOf(
            configurationBin("armv7", ConfigurationDistinguisher.IOS_APPLICATION)
                + "x/liblibarmv7.a",
            configurationBin("arm64", ConfigurationDistinguisher.IOS_APPLICATION)
                + "x/liblibdefault.a");

    assertThat(Artifact.toExecPaths(binInputs.build()))
        .doesNotContain(
            configurationBin("i386", ConfigurationDistinguisher.IOS_APPLICATION)
                + "x/liblibi386.a");
  }
}
