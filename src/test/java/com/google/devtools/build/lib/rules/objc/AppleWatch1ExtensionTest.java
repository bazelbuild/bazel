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
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_BUNDLE_ID_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_ICON_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_INFOPLISTS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_PROVISIONING_PROFILE_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchExtensionBundleRule.WATCH_EXT_BUNDLE_ID_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchExtensionBundleRule.WATCH_EXT_INFOPLISTS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchExtensionBundleRule.WATCH_EXT_PROVISIONING_PROFILE_ATTR;

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultiset;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multiset;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.BundleFile;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.Control;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.MergeZip;
import com.google.devtools.build.xcode.plmerge.proto.PlMergeProtos;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for apple_watch1_extension. */
@RunWith(JUnit4.class)
public class AppleWatch1ExtensionTest extends ObjcRuleTestCase {
  private static final RuleType RULE_TYPE = new RuleType("apple_watch1_extension") {
    @Override
    Iterable<String> requiredAttributes(
        Scratch scratch, String packageDir, Set<String> alreadyAdded) throws IOException {
      ImmutableList.Builder<String> attributes = new ImmutableList.Builder<>();
      if (!alreadyAdded.contains("binary")) {
        scratch.file(packageDir + "/extension_binary/a.m");
        scratch.file(packageDir + "/extension_binary/BUILD",
            "apple_watch_extension_binary(",
            "    name = 'extension_binary',",
            "    srcs = ['a.m'],",
            ")");
        attributes.add(String.format("binary = '//%s/extension_binary'", packageDir));
      }
      if (!alreadyAdded.contains("app_name")) {
        attributes.add("app_name = 'y'");
      }
      return attributes.build();
    }
  };

  protected static final BinaryRuleTypePair RULE_TYPE_PAIR =
      new BinaryRuleTypePair(
          AppleWatchExtensionBinaryTest.RULE_TYPE,
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
        "apple_watch_extension_binary(",
        "    name = 'bin',",
        "    srcs = ['a.m'],",
        "    deps = ['//lib1:lib1', '//lib2:lib2'],",
        ")",
        "",
        "apple_watch1_extension(",
        "    name = 'x',",
        "    app_name = 'y',",
        "    binary = ':bin',",
        Joiner.on(',').join(extraExtAttributes),
        ")");
    return getConfiguredTarget("//x:x");
  }

  private void addEntitlements() throws Exception {
    scratch.file("x/ext_entitlements.entitlements");
    scratch.file("x/app_entitlements.entitlements");
    addMockExtensionAndLibs(
        "ext_entitlements = 'ext_entitlements.entitlements'",
        "app_entitlements = 'app_entitlements.entitlements'");
  }

  private Action watchApplicationIpaGeneratingAction() throws Exception {
    return getGeneratingAction(getBinArtifact("_watch/x/y.ipa",
        "//x:x"));
  }

  @Test
  public void testExtensionSigningAction() throws Exception {
    useConfiguration("--cpu=ios_armv7");
    addEntitlements();
    SpawnAction action = (SpawnAction) getGeneratingActionForLabel("//x:x.ipa");
    assertRequiresDarwin(action);
    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs()))
        .containsExactly("x.entitlements", "foo.mobileprovision", "x.unprocessed.ipa");
    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly("x/x.ipa");
  }

  @Test
  public void testApplicationSigningAction() throws Exception {
    useConfiguration("--cpu=ios_armv7");
    addEntitlements();
    SpawnAction action = (SpawnAction) watchApplicationIpaGeneratingAction();
    assertRequiresDarwin(action);
    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs()))
        .containsExactly("x-y.entitlements", "foo.mobileprovision", "x-y.unprocessed.ipa");
    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("x/_watch/x/y.ipa");
  }

  @Test
  public void testExtensionSigningWithCertName() throws Exception {
    useConfiguration("--cpu=ios_armv7", "--ios_signing_cert_name=Foo Bar");
    addEntitlements();
    SpawnAction action = (SpawnAction) getGeneratingActionForLabel("//x:x.ipa");
    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs()))
        .containsExactly("x.entitlements", "foo.mobileprovision", "x.unprocessed.ipa");
    assertThat(Joiner.on(' ').join(action.getArguments())).contains("--sign \"Foo Bar\"");
    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly("x/x.ipa");
  }

  @Test
  public void testApplicationSigningWithCertName() throws Exception {
    useConfiguration("--cpu=ios_armv7", "--ios_signing_cert_name=Foo Bar");
    addEntitlements();
    SpawnAction action = (SpawnAction) watchApplicationIpaGeneratingAction();
    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs()))
        .containsExactly("x-y.entitlements", "foo.mobileprovision", "x-y.unprocessed.ipa");
    assertThat(Joiner.on(' ').join(action.getArguments())).contains("--sign \"Foo Bar\"");

    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("x/_watch/x/y.ipa");
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
  public void testUserSpecifiedExtensionProvisioningProfile_deviceBuild() throws Exception {
    checkProvisioningProfileUserSpecified(RULE_TYPE_PAIR, false,
        WATCH_EXT_PROVISIONING_PROFILE_ATTR);
  }

  @Test
  public void testUserSpecifiedApplicationProvisioningProfile_deviceBuild() throws Exception {
    checkSpecifiedApplicationProvisioningProfile(false);
  }

  @Test
  public void testUserSpecifiedExtensionProvisioningProfile_deviceBuild_multiCpu()
      throws Exception {
    checkProvisioningProfileUserSpecified(RULE_TYPE_PAIR, true,
        WATCH_EXT_PROVISIONING_PROFILE_ATTR);
  }

  @Test
  public void testUserSpecifiedApplicationProvisioningProfile_deviceBuild_multiCpu()
      throws Exception {
    checkSpecifiedApplicationProvisioningProfile(true);
  }

  private void checkSpecifiedApplicationProvisioningProfile(boolean useMultiCpu) throws Exception {
    setArtifactPrefix("y");
    if (useMultiCpu) {
      useConfiguration("--ios_multi_cpus=armv7,arm64", "--cpu=ios_i386");
    } else {
      useConfiguration("--cpu=ios_armv7");
    }

    addCustomProvisioningProfile(RULE_TYPE_PAIR, WATCH_APP_PROVISIONING_PROFILE_ATTR);
    getConfiguredTarget("//x:x");

    Artifact defaultProvisioningProfile =
        getFileConfiguredTarget("//tools/objc:foo.mobileprovision").getArtifact();
    Artifact customProvisioningProfile =
        getFileConfiguredTarget("//custom:pp.mobileprovision").getArtifact();
    Action signingAction = watchApplicationIpaGeneratingAction();
    assertThat(signingAction.getInputs()).contains(customProvisioningProfile);
    assertThat(signingAction.getInputs()).doesNotContain(defaultProvisioningProfile);

    BundleMergeProtos.Control control = bundleMergeControl("//x:x");
    Map<String, String> profiles = mobileProvisionProfiles(control);
    Map<String, String> expectedProfiles = ImmutableMap.of(
        customProvisioningProfile.getExecPathString(),
        ReleaseBundlingSupport.PROVISIONING_PROFILE_BUNDLE_FILE);
    assertThat(profiles).isEqualTo(expectedProfiles);
  }

  @Test
  public void testExtensionMergeControlAction() throws Exception {
    addMockExtensionAndLibs("ext_infoplists = ['Info.plist']");
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
                .addMergeZip(
                    MergeZip.newBuilder()
                        .setEntryNamePrefix("PlugIns/x.appex/")
                        .setSourcePath(
                            execPathEndingWith(mergeAction.getInputs(), "_watch/x/y.zip"))
                        .build())
                .setBundleInfoPlistFile(
                    getMergedInfoPlist(getConfiguredTarget("//x:x")).getExecPathString())
                .setOutFile(execPathEndingWith(mergeAction.getOutputs(), "x.unprocessed.ipa"))
                .setMinimumOsVersion(DEFAULT_IOS_SDK_VERSION.toString())
                .setSdkVersion(DEFAULT_IOS_SDK_VERSION.toString())
                .setPlatform("IOS_SIMULATOR")
                .setFallbackBundleIdentifier("example.ext.x")
                .build());
  }

  @Test
  public void testApplicationMergeControlAction() throws Exception {
    setArtifactPrefix("y");
    addMockExtensionAndLibs("app_infoplists = ['Info.plist']");
    Action mergeAction = bundleMergeAction("//x:x");
    Action action = bundleMergeControlAction("//x:x");
    assertThat(action.getInputs()).isEmpty();
    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly(
        "x/x-y.ipa-control");
    assertThat(bundleMergeControl("//x:x"))
        .isEqualTo(
            BundleMergeProtos.Control.newBuilder()
                .setBundleRoot("Payload/y.app")
                .addMergeZip(
                    MergeZip.newBuilder()
                        .setEntryNamePrefix("Payload/y.app/")
                        .setSourcePath(
                            getBinArtifact("_watch/x/WatchKitStub.zip", "//x:x")
                                .getExecPathString())
                        .build())
                .setBundleInfoPlistFile(
                    getMergedInfoPlist(getConfiguredTarget("//x:x")).getExecPathString())
                .setOutFile(execPathEndingWith(mergeAction.getOutputs(), "x-y.unprocessed.ipa"))
                .setMinimumOsVersion(DEFAULT_IOS_SDK_VERSION.toString())
                .setSdkVersion(DEFAULT_IOS_SDK_VERSION.toString())
                .setPlatform("IOS_SIMULATOR")
                .setFallbackBundleIdentifier("example.app.y")
                .build());
  }

  @Test
  public void testMergeExtensionBundleAction() throws Exception {
    RULE_TYPE_PAIR.scratchTargets(scratch,
        "ext_infoplists", "['Info.plist']");
    SpawnAction action = bundleMergeAction("//x:x");
    assertThat(Artifact.toRootRelativePaths(action.getInputs()))
        .containsExactly(
            MOCK_BUNDLEMERGE_PATH,
            "x/x_lipobin",
            "x/x.ipa-control",
            "x/x-MergedInfo.plist",
            "x/_watch/x/y.zip");
    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("x/x.unprocessed.ipa");
    assertNotRequiresDarwin(action);
    assertThat(action.getEnvironment()).isEmpty();
    assertThat(action.getArguments())
        .containsExactly(
            MOCK_BUNDLEMERGE_PATH,
            execPathEndingWith(action.getInputs(), "x.ipa-control"))
        .inOrder();
  }

  @Test
  public void testMergeApplicationBundleAction() throws Exception {
    RULE_TYPE_PAIR.scratchTargets(scratch,
        "app_infoplists", "['Info.plist']");
    setArtifactPrefix("y");
    SpawnAction action = bundleMergeAction("//x:x");
    assertThat(Artifact.toRootRelativePaths(action.getInputs()))
        .containsExactly(
            MOCK_BUNDLEMERGE_PATH,
            "x/x-y.ipa-control",
            "x/x-y-MergedInfo.plist",
            "x/_watch/x/WatchKitStub.zip");
    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("x/x-y.unprocessed.ipa");
    assertNotRequiresDarwin(action);
    assertThat(action.getEnvironment()).isEmpty();
    assertThat(action.getArguments())
        .containsExactly(
            MOCK_BUNDLEMERGE_PATH,
            execPathEndingWith(action.getInputs(), "x-y.ipa-control"))
        .inOrder();
  }

  protected List<BuildConfiguration> getExtensionConfigurations() throws InterruptedException {
    return getSplitConfigurations(getTargetConfiguration(),
       AppleWatch1Extension.MINIMUM_OS_AND_SPLIT_ARCH_TRANSITION);
  }

  @Test
  public void testErrorForAppIconGivenWithNoAssetCatalog() throws Exception {
    checkAssetCatalogAttributeError(RULE_TYPE, WATCH_APP_ICON_ATTR, WATCH_EXT_INFOPLISTS_ATTR,
        "['pl.plist']");
  }

  @Override
  protected void checkCollectsAssetCatalogsTransitively(BinaryRuleTypePair ruleTypePair)
      throws Exception {
    scratch.file("lib/ac.xcassets/foo");
    scratch.file("lib/ac.xcassets/bar");
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .set("asset_catalogs", "glob(['ac.xcassets/**'])")
        .write();

    scratch.file("x/ac.xcassets/baz");
    scratch.file("x/ac.xcassets/42");
    ruleTypePair.scratchTargets(scratch,
        "deps", "['//lib:lib']",
        "app_asset_catalogs", "glob(['ac.xcassets/**'])");

    // Test that the actoolzip Action for extension has arguments and inputs obtained from
    // dependencies.
    SpawnAction extensionActoolZipAction = actoolZipActionForIpa("//x:x");
    assertThat(Artifact.toExecPaths(extensionActoolZipAction.getInputs())).containsExactly(
        "lib/ac.xcassets/foo", "lib/ac.xcassets/bar",
        MOCK_ACTOOLWRAPPER_PATH);
    assertContainsSublist(extensionActoolZipAction.getArguments(),
        ImmutableList.of("lib/ac.xcassets"));

    // Test that the actoolzip Action for application has arguments and inputs obtained from
    // dependencies.
    SpawnAction applicationActoolZipAction = (SpawnAction) getGeneratingAction(
        getBinArtifact("x-y.actool.zip", "//x:x"));
    assertThat(Artifact.toExecPaths(applicationActoolZipAction.getInputs())).containsExactly(
        "x/ac.xcassets/baz", "x/ac.xcassets/42",
        MOCK_ACTOOLWRAPPER_PATH);
    assertContainsSublist(applicationActoolZipAction.getArguments(),
        ImmutableList.of("x/ac.xcassets"));
  }

  @Test
  public void testCollectsAssetCatalogsTransitively() throws Exception {
    checkCollectsAssetCatalogsTransitively(RULE_TYPE_PAIR);
  }

  private void addTargetWithAssetCatalogs() throws IOException {
    scratch.file("x/foo.xcassets/foo");
    scratch.file("x/foo.xcassets/bar");
    scratch.file("x/a.m");
    scratch.file("x/BUILD",
        "apple_watch_extension_binary(",
        "    name = 'bin',",
        "    srcs = ['a.m'],",
        ")",
        "",
        "apple_watch1_extension(",
        "    name = 'x',",
        "    app_name = 'y',",
        "    app_asset_catalogs = ['foo.xcassets/foo', 'bar.xcassets/bar'],",
        "    binary = ':bin',",
        ")");
  }

  @Test
  public void testActoolActionCorrectness() throws Exception {
    addTargetWithAssetCatalogs();
    setArtifactPrefix("y");
    checkActoolActionCorrectness(DEFAULT_IOS_SDK_VERSION,
        TargetDeviceFamily.WATCH.getNameInRule().toLowerCase(), "iphonesimulator");
  }

  @Test
  public void testPassesFamiliesToActool() throws Exception {
    checkPassesFamiliesToActool(RULE_TYPE_PAIR, AppleWatch1ExtensionRule.WATCH_EXT_FAMILIES_ATTR);
  }

  @Test
  public void testPassesFamiliesToIbtool() throws Exception {
    checkPassesFamiliesToIbtool(RULE_TYPE_PAIR, AppleWatch1ExtensionRule.WATCH_EXT_FAMILIES_ATTR);
  }

  @Test
  public void testReportsErrorsForInvalidFamiliesAttribute() throws Exception {
    checkReportsErrorsForInvalidFamiliesAttribute(RULE_TYPE,
        AppleWatch1ExtensionRule.WATCH_EXT_FAMILIES_ATTR);
  }

  @Test
  public void testMergeActionsWithAssetCatalog() throws Exception {
    addTargetWithAssetCatalogs();
    setArtifactPrefix("y");
    Artifact actoolZipOut = getBinArtifact("x-y.actool.zip", "//x:x");
    assertThat(bundleMergeAction("//x:x").getInputs()).contains(actoolZipOut);

    BundleMergeProtos.Control mergeControl = bundleMergeControl("//x:x");
    assertThat(mergeControl.getMergeZipList())
        .containsExactly(
            MergeZip.newBuilder()
              .setEntryNamePrefix("Payload/y.app/")
              .setSourcePath(actoolZipOut.getExecPathString())
              .build(),
            MergeZip.newBuilder()
              .setEntryNamePrefix("Payload/y.app/")
              .setSourcePath(getBinArtifact("_watch/x/WatchKitStub.zip", "//x:x")
                  .getExecPathString())
              .build());
  }

  private void addBinAndLibWithRawResources() throws Exception {
    addBinAndLibWithResources(
        "resources", "resource1.txt", "ja.lproj/resource2.txt", "apple_watch_extension_binary");
    scratch.file("app_resource.txt");
    scratch.file("ext_resource.txt");
    scratch.file("x/BUILD",
        "apple_watch1_extension(",
        "    name = 'x',",
        "    app_name = 'y',",
        "    binary = '//bin:bin',",
        "    app_resources = ['app_resource.txt'],",
        "    ext_resources = ['ext_resource.txt'],",
        ")");
  }

  private void addBinAndLibWithStrings() throws Exception {
    addBinAndLibWithResources(
        "strings", "foo.strings", "ja.lproj/bar.strings", "apple_watch_extension_binary");
    scratch.file("app.strings");
    scratch.file("x/BUILD",
        "apple_watch1_extension(",
        "    name = 'x',",
        "    app_name = 'y',",
        "    binary = '//bin:bin',",
        "    app_strings = ['app.strings'],",
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
            "x_x", ImmutableMultiset.of("bin/ja.lproj/resource2.txt", "lib/resource1.txt",
                "x/ext_resource.txt"),
            "lib_lib", ImmutableMultiset.of("lib/resource1.txt"),
            "y__x", ImmutableMultiset.of("x/app_resource.txt")));
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
            "lib_lib", ImmutableMultiset.of("lib/foo.strings"),
            "y__x", ImmutableMultiset.of("x/app.strings")));
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
  public void testPlistRequiresDotInName() throws Exception {
    String errorMessage = "'//x:Infoplist' does not produce any apple_watch1_extension "
        + "ext_infoplists files (expected .plist)";
    checkError("x", "x",
        errorMessage,
        "apple_watch_extension_binary(",
        "    name = 'bin',",
        "    srcs = ['a.m'],",
        ")",
        "",
        "apple_watch1_extension(",
        "    name = 'x',",
        "    app_name = 'y',",
        "    ext_infoplists = ['Infoplist'],",
        "    binary = ':bin',",
        ")");
  }

  @Test
  public void testIncludesStoryboardOutputZipsAsMergeZipsForApplication() throws Exception {
    addStoryboards();
    setArtifactPrefix("y");
    Artifact libsbOutputZip = getBinArtifact("x-y/libsb.storyboard.zip", "//x:x");

    Control mergeControl = bundleMergeControl("//x:x");
    assertThat(mergeControl.getMergeZipList()).containsExactly(
        MergeZip.newBuilder()
            .setEntryNamePrefix("Payload/y.app/")
            .setSourcePath(libsbOutputZip.getExecPathString())
            .build(),
        MergeZip.newBuilder()
            .setEntryNamePrefix("Payload/y.app/")
            .setSourcePath(getBinArtifact("_watch/x/WatchKitStub.zip", "//x:x")
                .getExecPathString())
            .build());
  }

  protected void addStoryboards() throws Exception {
    scratch.file("lib/libsb.storyboard");
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("storyboards", "libsb.storyboard")
        .write();

    scratch.file("bndl/bndlsb.storyboard");
    scratch.file("bndl/BUILD",
        "objc_bundle_library(",
        "    name = 'bndl',",
        "    storyboards = ['ext_bndlsb.storyboard'],",
        ")");

    scratch.file("x/xsb.storyboard");
    RULE_TYPE_PAIR.scratchTargets(scratch,
        "storyboards", "['ext.storyboard']",
        "app_deps", "['//lib:lib']",
        "bundles", "['//bndl:bndl']");
    getConfiguredTarget("//x:x");
  }

  @Test
  public void testMergesPartialInfoplists() throws Exception {
    scratch.file("x/primary-Info.plist");
    setArtifactPrefix("y");
    RULE_TYPE_PAIR.scratchTargets(scratch,
        "app_asset_catalogs", "['foo.xcassets/bar']",
        "app_infoplists", "['primary-Info.plist']");

    String targetName = "//x:x";
    ConfiguredTarget target = getConfiguredTarget(targetName);
    PlMergeProtos.Control control = plMergeControl(targetName);

    Artifact merged = getBinArtifact("x-y-MergedInfo.plist", target);
    Artifact actoolPartial = getBinArtifact("x-y.actool-PartialInfo.plist", "//x:x");

    Artifact versionInfoplist = getBinArtifact("plists/x-y-version.plist", target);
    Artifact environmentInfoplist = getBinArtifact("plists/x-y-environment.plist", target);
    Artifact automaticInfoplist = getBinArtifact("plists/x-y-automatic.plist", target);

    assertPlistMergeControlUsesSourceFiles(
        control,
        ImmutableList.<String>of(
            "x/primary-Info.plist",
            versionInfoplist.getExecPathString(),
            environmentInfoplist.getExecPathString(),
            automaticInfoplist.getExecPathString(),
            actoolPartial.getExecPathString()));
    assertThat(control.getOutFile()).isEqualTo(merged.getExecPathString());
    assertThat(control.getVariableSubstitutionMapMap())
        .containsExactlyEntriesIn(variableSubstitutionsForWatchApplication());
    assertThat(control.getFallbackBundleId()).isEqualTo("example.app.y");
  }

  @Test
  public void testNibZipsMergedIntoBundle() throws Exception {
    checkNibZipsMergedIntoBundle(RULE_TYPE_PAIR);
  }

  @Test
  public void testPassesExtensionFallbackBundleIdToBundleMerging() throws Exception {
    scratch.file("bin/a.m");
    scratch.file("bin/Ext-Info.plist");

    RULE_TYPE.scratchTarget(scratch,
        WATCH_EXT_INFOPLISTS_ATTR, "['Ext-Info.plist']");

    BundleMergeProtos.Control control = bundleMergeControl("//x:x");

    assertThat(control.hasPrimaryBundleIdentifier()).isFalse();
    assertThat(control.getFallbackBundleIdentifier()).isEqualTo("example.ext.x");
  }

  @Test
  public void testPassesApplicationFallbackBundleIdToBundleMerging() throws Exception {
    setArtifactPrefix("y");
    scratch.file("bin/a.m");
    scratch.file("bin/App-Info.plist");

    RULE_TYPE.scratchTarget(scratch,
        WATCH_APP_INFOPLISTS_ATTR, "['App-Info.plist']");

    BundleMergeProtos.Control control = bundleMergeControl("//x:x");

    assertThat(control.hasPrimaryBundleIdentifier()).isFalse();
    assertThat(control.getFallbackBundleIdentifier()).isEqualTo("example.app.y");
  }

  @Test
  public void testPassesExtensionPrimaryBundleIdToBundleMerging() throws Exception {
    scratch.file("bin/a.m");
    scratch.file("bin/Ext-Info.plist");

    RULE_TYPE.scratchTarget(scratch,
        WATCH_EXT_INFOPLISTS_ATTR, "['Ext-Info.plist']",
        WATCH_EXT_BUNDLE_ID_ATTR, "'com.bundle.ext.id'");

    BundleMergeProtos.Control control = bundleMergeControl("//x:x");

    assertThat(control.getPrimaryBundleIdentifier()).isEqualTo("com.bundle.ext.id");
    assertThat(control.hasFallbackBundleIdentifier()).isFalse();
  }

  @Test
  public void testPassesApplicationPrimaryBundleIdToBundleMerging() throws Exception {
    setArtifactPrefix("y");
    scratch.file("bin/a.m");
    scratch.file("bin/App-Info.plist");

    RULE_TYPE.scratchTarget(scratch,
        WATCH_APP_INFOPLISTS_ATTR, "['App-Info.plist']",
        WATCH_APP_BUNDLE_ID_ATTR, "'com.bundle.app.id'");

    BundleMergeProtos.Control control = bundleMergeControl("//x:x");

    assertThat(control.getPrimaryBundleIdentifier()).isEqualTo("com.bundle.app.id");
    assertThat(control.hasFallbackBundleIdentifier()).isFalse();
  }

  @Test
  public void testMultiPlatformBuild_fails() throws Exception {
    checkBinaryActionMultiPlatform_fails(RULE_TYPE_PAIR);
  }

  @Test
  public void testMultiArchitectureResources() throws Exception {
    useConfiguration("--ios_multi_cpus=i386,x86_64");
    RULE_TYPE_PAIR.scratchTargets(scratch, "resources", "['foo.png']");

    assertThat(Artifact.toRootRelativePaths(bundleMergeAction("//x:x").getInputs()))
        .containsExactly(
            "x/foo.png",
            "x/x_lipobin",
            "tools/objc/bundlemerge",
            "x/x.ipa-control",
            "x/x-MergedInfo.plist",
            "x/_watch/x/y.zip");
  }

  @Override
  protected void addCommonResources(BinaryRuleTypePair ruleTypePair) throws Exception {
    ruleTypePair.scratchTargets(scratch,
        "strings", "['foo.strings']",
        "storyboards", "['baz.storyboard']");
  }

  @Test
  public void testMultiCpuCompiledResources() throws Exception {
    checkMultiCpuCompiledResources(RULE_TYPE_PAIR);
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
    addMockExtensionAndLibs("ext_infoplists = ['Info.plist']");

    assertThat(bundleMergeControl("//x:x").getMinimumOsVersion())
        .isEqualTo(WatchUtils.MINIMUM_OS_VERSION.toString());
  }

  @Test
  public void testExtensionReplacesMinimumOsVersionInBundleMergeAtMost82() throws Exception {
    useConfiguration("--ios_minimum_os=8.3");
    addMockExtensionAndLibs("ext_infoplists = ['Info.plist']");

    assertThat(bundleMergeControl("//x:x").getMinimumOsVersion())
        .isEqualTo("8.3");
  }

  @Test
  public void testCheckExtensionPrimaryBundleIdInMergedPlist() throws Exception {
    RULE_TYPE_PAIR.scratchTargets(scratch,
        WATCH_EXT_INFOPLISTS_ATTR, "['Info.plist']",
        WATCH_EXT_BUNDLE_ID_ATTR, "'com.ext.bundle.id'");
    scratch.file("ext/Info.plist");

    checkBundleIdFlagsInPlistMergeAction(
        Optional.of("com.ext.bundle.id"),
        getVariableSubstitutionArguments(RULE_TYPE_PAIR),
        "example.ext.x");
  }

  @Test
  public void testCheckApplicationPrimaryBundleIdInMergedPlist() throws Exception {
    setArtifactPrefix("y");
    RULE_TYPE_PAIR.scratchTargets(scratch,
        WATCH_APP_INFOPLISTS_ATTR, "['Info.plist']",
        WATCH_APP_BUNDLE_ID_ATTR, "'com.app.bundle.id'");
    scratch.file("app/Info.plist");

    checkBundleIdFlagsInPlistMergeAction(
        Optional.of("com.app.bundle.id"),
        variableSubstitutionsForWatchApplication(),
        "example.app.y");
  }

  @Test
  public void testCheckExtensionFallbackBundleIdInMergedPlist() throws Exception {
    RULE_TYPE_PAIR.scratchTargets(scratch,
        WATCH_EXT_INFOPLISTS_ATTR, "['Info.plist']");
    scratch.file("ext/Info.plist");

    checkBundleIdFlagsInPlistMergeAction(
        Optional.<String>absent(),
        getVariableSubstitutionArguments(RULE_TYPE_PAIR),
        "example.ext.x");
  }

  @Test
  public void testCheckApplicationFallbackBundleIdInMergedPlist() throws Exception {
    setArtifactPrefix("y");
    RULE_TYPE_PAIR.scratchTargets(scratch,
        WATCH_APP_INFOPLISTS_ATTR, "['Info.plist']");
    scratch.file("app/Info.plist");

    checkBundleIdFlagsInPlistMergeAction(
        Optional.<String>absent(),
        variableSubstitutionsForWatchApplication(),
        "example.app.y");
  }

  private ImmutableMap<String, String> variableSubstitutionsForWatchApplication() {
    return new ImmutableMap.Builder<String, String>()
      .put("EXECUTABLE_NAME", "y")
      .put("BUNDLE_NAME", "y.app")
      .put("PRODUCT_NAME", "y")
      .build();
  }

  protected void checkExtensionReplacesMinimumOsInCompilation() throws Exception {
    addMockExtensionAndLibs("ext_infoplists = ['Info.plist']");

    Action lipoAction = lipoBinAction("//x:x");

    for (Artifact bin : lipoAction.getInputs()) {
      CommandAction action = (CommandAction) getGeneratingAction(bin);
      if (action == null) {
        continue;
      }
      assertThat(generatingArgumentsToString(action))
          .contains("-mios-simulator-version-min=" + WatchUtils.MINIMUM_OS_VERSION);
      assertThat(generatingArgumentsToString(action))
          .doesNotContain("-mios-simulator-version-min=7.1");
    }
  }

  private String generatingArgumentsToString(CommandAction generatingAction) {
    return Joiner.on(' ').join(generatingAction.getArguments());
  }

  protected void checkExtensionDoesNotReplaceMinimumOsInCompilation() throws Exception {
    addMockExtensionAndLibs("ext_infoplists = ['Info.plist']");

    Action lipoAction = lipoBinAction("//x:x");

    for (Artifact bin : lipoAction.getInputs()) {
      CommandAction action = (CommandAction) getGeneratingAction(bin);
      if (action == null) {
        continue;
      }
      assertThat(generatingArgumentsToString(action)).contains("-mios-simulator-version-min=8.3");
      assertThat(generatingArgumentsToString(action))
          .doesNotContain("-mios-simulator-version-min=" + DEFAULT_IOS_SDK_VERSION);
    }
  }

  @Test
  public void testGenruleWithoutJavaCcDeps() throws Exception {
    checkGenruleWithoutJavaCcDependency(RULE_TYPE_PAIR);
  }

  @Test
  public void testCcDependencyWithProtoDependencyMultiArch() throws Exception {
    checkCcDependencyWithProtoDependencyMultiArch(
        RULE_TYPE_PAIR, ConfigurationDistinguisher.WATCH_OS1_EXTENSION);
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
  public void testIncludesStoryboardOutputZipsAsMergeZipsForExtension() throws Exception {
    BuildConfiguration configuration = Iterables.getOnlyElement(getExtensionConfigurations());
    addStoryboards();

    Artifact extBndlsbOutputZip =
        getBinArtifact(
            "bndl/ext_bndlsb.storyboard.zip", getConfiguredTarget("//bndl:bndl", configuration));
    Artifact extsbOutputZip = getBinArtifact("x/ext.storyboard.zip", "//x:x");

    String bundleDir = RULE_TYPE_PAIR.getBundleDir();
    Control mergeControl = bundleMergeControl("//x:x");
    assertThat(mergeControl.getMergeZipList())
        .containsExactly(
            MergeZip.newBuilder()
                .setEntryNamePrefix(bundleDir + "/")
                .setSourcePath(extsbOutputZip.getExecPathString())
                .build(),
            MergeZip.newBuilder()
                .setEntryNamePrefix(bundleDir + "/")
                .setSourcePath(getBinArtifact("_watch/x/y.zip", "//x:x").getExecPathString())
                .build());

    Control nestedMergeControl = Iterables.getOnlyElement(mergeControl.getNestedBundleList());
    assertThat(nestedMergeControl.getMergeZipList())
        .containsExactly(
            MergeZip.newBuilder()
                .setEntryNamePrefix(bundleDir + "/bndl.bundle/")
                .setSourcePath(extBndlsbOutputZip.getExecPathString())
                .build());
  }

  @Test
  public void testCcDependency() throws Exception {
    checkCcDependency(RULE_TYPE_PAIR, ConfigurationDistinguisher.WATCH_OS1_EXTENSION);
  }

  @Test
  public void testCcDependencyMultiArch() throws Exception {
    checkCcDependencyMultiArch(RULE_TYPE_PAIR, ConfigurationDistinguisher.WATCH_OS1_EXTENSION);
  }

  @Test
  public void testCcDependencyWithProtoDependency() throws Exception {
    checkCcDependencyWithProtoDependency(
        RULE_TYPE_PAIR, ConfigurationDistinguisher.WATCH_OS1_EXTENSION);
  }

  @Test
  public void testCcDependencyAndJ2objcDependency() throws Exception {
    checkCcDependencyAndJ2objcDependency(
        RULE_TYPE_PAIR, ConfigurationDistinguisher.WATCH_OS1_EXTENSION);
  }

  @Test
  public void testMultiArchitectureFanOut() throws Exception {
    checkBinaryLipoActionMultiCpu(RULE_TYPE_PAIR, ConfigurationDistinguisher.WATCH_OS1_EXTENSION);
  }

  @Test
  public void testGenruleDependencyMultiArch() throws Exception {
    checkGenruleDependencyMultiArch(RULE_TYPE_PAIR, ConfigurationDistinguisher.WATCH_OS1_EXTENSION);
  }

  @Test
  public void testExtensionReplacesMinimumOsInCompilation() throws Exception {
    useConfiguration("--ios_minimum_os=7.1");
    checkExtensionReplacesMinimumOsInCompilation();
  }

  @Test
  public void testExtensionReplacesMinimumOsInCompilationAtMost82() throws Exception {
    useConfiguration("--ios_minimum_os=8.3");
    checkExtensionDoesNotReplaceMinimumOsInCompilation();
  }

  @Test
  public void testExtensionReplacesMinimumOsInCompilationMultiArch() throws Exception {
    useConfiguration("--ios_minimum_os=7.1", "--ios_multi_cpus=i386,x86_64");
    checkExtensionReplacesMinimumOsInCompilation();
  }

  @Test
  public void testExtensionReplacesMinimumOsInCompilationAtMost82MultiArch() throws Exception {
    useConfiguration("--ios_minimum_os=8.3", "--ios_multi_cpus=i386,x86_64");
    checkExtensionDoesNotReplaceMinimumOsInCompilation();
  }
}
