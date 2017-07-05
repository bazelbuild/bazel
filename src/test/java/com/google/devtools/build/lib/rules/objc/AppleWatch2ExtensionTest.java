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
import com.google.common.collect.Iterables;
import com.google.common.collect.ObjectArrays;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.BundleFile;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.Control;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.MergeZip;
import com.google.devtools.build.xcode.plmerge.proto.PlMergeProtos;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for apple_watch2_extension. */
@RunWith(JUnit4.class)
public class AppleWatch2ExtensionTest extends ObjcRuleTestCase {
  private static final RuleType RULE_TYPE =
      new RuleType("apple_watch2_extension") {
        @Override
        Iterable<String> requiredAttributes(
            Scratch scratch, String packageDir, Set<String> alreadyAdded) throws IOException {
          ImmutableList.Builder<String> attributes = new ImmutableList.Builder<>();
          if (!alreadyAdded.contains("binary")) {
            scratch.file(packageDir + "/extension_binary/a.m");
            scratch.file(
                packageDir + "/extension_binary/BUILD",
                "apple_binary(",
                "    name = 'extension_binary',",
                "    srcs = ['a.m'],",
                "    platform_type = 'watchos'",
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
          AppleBinaryTest.RULE_TYPE, RULE_TYPE, ReleaseBundlingSupport.EXTENSION_BUNDLE_DIR_FORMAT);

  private ConfiguredTarget addMockExtensionAndLibs(String... extraExtAttributes) throws Exception {
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();

    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();

    scratch.file("x/a.m");
    scratch.file(
        "x/BUILD",
        "apple_binary(",
        "    name = 'bin',",
        "    srcs = ['a.m'],",
        "    deps = ['//lib1:lib1', '//lib2:lib2'],",
        "    platform_type = 'watchos'",
        ")",
        "",
        "apple_watch2_extension(",
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
    return getGeneratingAction(getBinArtifact("y.ipa", "//x:x"));
  }

  @Test
  public void testExtensionSigningAction() throws Exception {
    useConfiguration("--cpu=ios_armv7", "--watchos_cpus=armv7k");
    addEntitlements();
    SpawnAction action = (SpawnAction) ipaGeneratingAction();
    assertRequiresDarwin(action);
    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs()))
        .containsExactly("x.entitlements", "foo.mobileprovision", "x.unprocessed.ipa");
    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly("x/x.ipa");
  }

  @Test
  public void testApplicationSigningAction() throws Exception {
    useConfiguration("--cpu=ios_armv7", "--watchos_cpus=armv7k");
    addEntitlements();
    SpawnAction action = (SpawnAction) watchApplicationIpaGeneratingAction();
    assertRequiresDarwin(action);
    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs()))
        .containsExactly("x-y.entitlements", "foo.mobileprovision", "x-y.unprocessed.ipa");
    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly("x/y.ipa");
  }

  @Test
  public void testExtensionSigningWithCertName() throws Exception {
    useConfiguration("--cpu=ios_armv7", "--ios_signing_cert_name=Foo Bar", "--watchos_cpus=armv7k");
    addEntitlements();
    SpawnAction action = (SpawnAction) ipaGeneratingAction();
    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs()))
        .containsExactly("x.entitlements", "foo.mobileprovision", "x.unprocessed.ipa");
    assertThat(Joiner.on(' ').join(action.getArguments())).contains("--sign \"Foo Bar\"");
    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly("x/x.ipa");
  }

  @Test
  public void testApplicationSigningWithCertName() throws Exception {
    useConfiguration("--cpu=ios_armv7", "--ios_signing_cert_name=Foo Bar", "--watchos_cpus=armv7k");
    addEntitlements();
    SpawnAction action = (SpawnAction) watchApplicationIpaGeneratingAction();
    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs()))
        .containsExactly("x-y.entitlements", "foo.mobileprovision", "x-y.unprocessed.ipa");
    assertThat(Joiner.on(' ').join(action.getArguments())).contains("--sign \"Foo Bar\"");

    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly("x/y.ipa");
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
    checkProvisioningProfileUserSpecified(
        RULE_TYPE_PAIR, false, WATCH_EXT_PROVISIONING_PROFILE_ATTR);
  }

  @Test
  public void testUserSpecifiedApplicationProvisioningProfile_deviceBuild() throws Exception {
    checkSpecifiedApplicationProvisioningProfile(false);
  }

  @Test
  public void testUserSpecifiedExtensionProvisioningProfile_deviceBuild_multiCpu()
      throws Exception {
    checkProvisioningProfileUserSpecified(
        RULE_TYPE_PAIR, true, WATCH_EXT_PROVISIONING_PROFILE_ATTR);
  }

  @Test
  public void testUserSpecifiedApplicationProvisioningProfile_deviceBuild_multiCpu()
      throws Exception {
    checkSpecifiedApplicationProvisioningProfile(true);
  }

  private void checkSpecifiedApplicationProvisioningProfile(boolean useMultiCpu) throws Exception {
    setArtifactPrefix("y");
    if (useMultiCpu) {
      useConfiguration("--ios_multi_cpus=armv7,arm64", "--cpu=ios_i386", "--watchos_cpus=armv7k");
    } else {
      useConfiguration("--cpu=ios_armv7", "--watchos_cpus=armv7k");
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
    Map<String, String> expectedProfiles =
        ImmutableMap.of(
            customProvisioningProfile.getExecPathString(),
            ReleaseBundlingSupport.PROVISIONING_PROFILE_BUNDLE_FILE);
    assertThat(profiles).isEqualTo(expectedProfiles);
  }

  @Test
  public void testExtensionMergeControlAction() throws Exception {
    addMockExtensionAndLibs("ext_infoplists = ['Info.plist']");
    useConfiguration("--watchos_sdk_version=2.2");
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
                        .setSourceFile(execPathEndingWith(mergeAction.getInputs(), "bin_lipobin"))
                        .setBundlePath("x")
                        .setExternalFileAttribute(BundleableFile.EXECUTABLE_EXTERNAL_FILE_ATTRIBUTE)
                        .build())
                .setBundleRoot("PlugIns/x.appex")
                .setBundleInfoPlistFile(
                    getMergedInfoPlist(getConfiguredTarget("//x:x")).getExecPathString())
                .setOutFile(execPathEndingWith(mergeAction.getOutputs(), "x.unprocessed.ipa"))
                .setMinimumOsVersion("2.2")
                .setSdkVersion(DEFAULT_IOS_SDK_VERSION.toString())
                .setPlatform("IOS_SIMULATOR")
                .setFallbackBundleIdentifier("example.ext.x")
                .build());
  }

  @Test
  public void testApplicationMergeControlAction() throws Exception {
    setArtifactPrefix("y");
    addMockExtensionAndLibs("app_infoplists = ['Info.plist']");
    useConfiguration("--watchos_sdk_version=2.2");
    Action mergeAction = bundleMergeAction("//x:x");
    Action action = bundleMergeControlAction("//x:x");
    assertThat(action.getInputs()).isEmpty();
    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("x/x-y.ipa-control");
    assertThat(bundleMergeControl("//x:x"))
        .isEqualTo(
            BundleMergeProtos.Control.newBuilder()
                .setBundleRoot("Watch/y.app")
                .addMergeZip(
                    MergeZip.newBuilder()
                        .setEntryNamePrefix("Watch/y.app/")
                        .setSourcePath(getBinArtifact("x.ipa", "//x:x").getExecPathString())
                        .build())
                .addMergeZip(
                    MergeZip.newBuilder()
                        .setEntryNamePrefix("Watch/y.app/")
                        .setSourcePath(
                            getBinArtifact("_watch/x/WatchKitStub.zip", "//x:x")
                                .getExecPathString())
                        .build())
                .setBundleInfoPlistFile(
                    getMergedInfoPlist(getConfiguredTarget("//x:x")).getExecPathString())
                .setOutFile(execPathEndingWith(mergeAction.getOutputs(), "x-y.unprocessed.ipa"))
                .setMinimumOsVersion("2.2")
                .setSdkVersion(DEFAULT_IOS_SDK_VERSION.toString())
                .setPlatform("IOS_SIMULATOR")
                .setFallbackBundleIdentifier("example.app.y")
                .build());
  }

  @Test
  public void testMergeExtensionBundleAction() throws Exception {
    RULE_TYPE_PAIR.scratchTargets(scratch, "ext_infoplists", "['Info.plist']");
    SpawnAction action = bundleMergeAction("//x:x");
    assertThat(Artifact.toRootRelativePaths(action.getInputs()))
        .containsExactly(
            MOCK_BUNDLEMERGE_PATH, "x/bin_lipobin", "x/x.ipa-control", "x/x-MergedInfo.plist");
    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("x/x.unprocessed.ipa");
    assertNotRequiresDarwin(action);
    assertThat(action.getEnvironment()).isEmpty();
    assertThat(action.getArguments())
        .containsExactly(
            MOCK_BUNDLEMERGE_PATH, execPathEndingWith(action.getInputs(), "x.ipa-control"))
        .inOrder();
  }

  @Test
  public void testMergeApplicationBundleAction() throws Exception {
    RULE_TYPE_PAIR.scratchTargets(scratch, "app_infoplists", "['Info.plist']");
    setArtifactPrefix("y");
    SpawnAction action = bundleMergeAction("//x:x");
    assertThat(Artifact.toRootRelativePaths(action.getInputs()))
        .containsExactly(
            MOCK_BUNDLEMERGE_PATH,
            "x/x-y.ipa-control",
            "x/x-y-MergedInfo.plist",
            "x/_watch/x/WatchKitStub.zip",
            "x/x.ipa");
    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("x/x-y.unprocessed.ipa");
    assertNotRequiresDarwin(action);
    assertThat(action.getEnvironment()).isEmpty();
    assertThat(action.getArguments())
        .containsExactly(
            MOCK_BUNDLEMERGE_PATH, execPathEndingWith(action.getInputs(), "x-y.ipa-control"))
        .inOrder();
  }

  @Test
  public void testErrorForAppIconGivenWithNoAssetCatalog() throws Exception {
    checkAssetCatalogAttributeError(
        RULE_TYPE, WATCH_APP_ICON_ATTR, WATCH_EXT_INFOPLISTS_ATTR, "['pl.plist']");
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
    ruleTypePair.scratchTargets(
        scratch, "deps", "['//lib:lib']", "app_asset_catalogs", "glob(['ac.xcassets/**'])");

    // Test that the actoolzip Action for extension has arguments and inputs obtained from
    // dependencies.
    SpawnAction extensionActoolZipAction = actoolZipActionForIpa("//x:x");
    assertThat(Artifact.toExecPaths(extensionActoolZipAction.getInputs()))
        .containsExactly("lib/ac.xcassets/foo", "lib/ac.xcassets/bar", MOCK_ACTOOLWRAPPER_PATH);
    assertContainsSublist(
        extensionActoolZipAction.getArguments(), ImmutableList.of("lib/ac.xcassets"));

    // Test that the actoolzip Action for application has arguments and inputs obtained from
    // dependencies.
    SpawnAction applicationActoolZipAction =
        (SpawnAction) getGeneratingAction(getBinArtifact("x-y.actool.zip", "//x:x"));
    assertThat(Artifact.toExecPaths(applicationActoolZipAction.getInputs()))
        .containsExactly("x/ac.xcassets/baz", "x/ac.xcassets/42", MOCK_ACTOOLWRAPPER_PATH);
    assertContainsSublist(
        applicationActoolZipAction.getArguments(), ImmutableList.of("x/ac.xcassets"));
  }

  @Test
  public void testCollectsAssetCatalogsTransitively() throws Exception {
    checkCollectsAssetCatalogsTransitively(RULE_TYPE_PAIR);
  }

  private void addTargetWithAssetCatalogs() throws IOException {
    scratch.file("x/foo.xcassets/foo");
    scratch.file("x/foo.xcassets/bar");
    scratch.file("x/a.m");
    scratch.file(
        "x/BUILD",
        "apple_binary(",
        "    name = 'bin',",
        "    srcs = ['a.m'],",
        "    platform_type = 'watchos',",
        ")",
        "",
        "apple_watch2_extension(",
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
    useConfiguration("--watchos_sdk_version=2.2");
    checkActoolActionCorrectness(
        DottedVersion.fromString("2.2"), TargetDeviceFamily.WATCH.getNameInRule().toLowerCase(),
        "watchsimulator");
  }

  @Test
  public void testMergeActionsWithAssetCatalog() throws Exception {
    addTargetWithAssetCatalogs();
    setArtifactPrefix("y");
    Artifact actoolZipOut = getBinArtifact("x-y.actool.zip", "//x:x");
    assertThat(bundleMergeAction("//x:x").getInputs()).contains(actoolZipOut);

    BundleMergeProtos.Control mergeControl = bundleMergeControl("//x:x");
    assertThat(mergeControl.getMergeZipList())
        .containsAllOf(
            MergeZip.newBuilder()
                .setEntryNamePrefix("Watch/y.app/")
                .setSourcePath(getBinArtifact("x.ipa", "//x:x").getExecPathString())
                .build(),
            MergeZip.newBuilder()
                .setEntryNamePrefix("Watch/y.app/")
                .setSourcePath(
                    getBinArtifact("_watch/x/WatchKitStub.zip", "//x:x").getExecPathString())
                .build());
  }

  private void addBinAndLibWithRawResources() throws Exception {
    addBinAndLibWithResources(
        "resources", "resource1.txt", "ja.lproj/resource2.txt", "apple_binary",
        "platform_type = 'watchos'");
    scratch.file("app_resource.txt");
    scratch.file("ext_resource.txt");
    scratch.file(
        "x/BUILD",
        "apple_watch2_extension(",
        "    name = 'x',",
        "    app_name = 'y',",
        "    binary = '//bin:bin',",
        "    app_resources = ['app_resource.txt'],",
        "    ext_resources = ['ext_resource.txt'],",
        ")");
  }

  private void addBinAndLibWithStrings() throws Exception {
    addBinAndLibWithResources("strings", "foo.strings", "ja.lproj/bar.strings", "apple_binary",
        "platform_type = 'watchos'");
    scratch.file("app.strings");
    scratch.file(
        "x/BUILD",
        "apple_watch2_extension(",
        "    name = 'x',",
        "    app_name = 'y',",
        "    binary = '//bin:bin',",
        "    app_strings = ['app.strings'],",
        ")");
  }

  @Test
  public void testCollectsRawResourceFilesTransitively() throws Exception {
    addBinAndLibWithRawResources();
    Action mergeBundleAction = bundleMergeAction("//x:x");

    assertThat(Artifact.toRootRelativePaths(mergeBundleAction.getInputs()))
        .containsAllOf("lib/resource1.txt", "bin/ja.lproj/resource2.txt");
  }

  @Test
  public void testCollectsStringsFilesTransitively() throws Exception {
    addBinAndLibWithStrings();

    Action mergeBundleAction = bundleMergeAction("//x:x");

    assertThat(Artifact.toRootRelativePaths(mergeBundleAction.getInputs()))
        .containsAllOf("x/lib/foo.strings.binary", "x/bin/ja.lproj/bar.strings.binary");
  }

  @Test
  public void testResourceFilesMergedInBundle() throws Exception {
    addBinAndLibWithRawResources();
    checkBundleablesAreMerged(
        "//x:x",
        ImmutableListMultimap.of(
            "resource1.txt", "resource1.txt",
            "ja.lproj/resource2.txt", "ja.lproj/resource2.txt"));
  }

  @Test
  public void testStringsFilesMergedInBundle() throws Exception {
    addBinAndLibWithStrings();
    checkBundleablesAreMerged(
        "//x:x",
        ImmutableListMultimap.of(
            "foo.strings.binary", "foo.strings",
            "ja.lproj/bar.strings.binary", "ja.lproj/bar.strings"));
  }

  @Test
  public void testPlistRequiresDotInName() throws Exception {
    String errorMessage =
        "'//x:Infoplist' does not produce any apple_watch2_extension "
            + "ext_infoplists files (expected .plist)";
    checkError(
        "x",
        "x",
        errorMessage,
        "apple_binary(",
        "    name = 'bin',",
        "    srcs = ['a.m'],",
        "    platform_type = 'watchos'",
        ")",
        "",
        "apple_watch2_extension(",
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
    Artifact libsbOutputZip = getBinArtifact("x-y/appsb.storyboard.zip", "//x:x");

    Control mergeControl = bundleMergeControl("//x:x");
    assertThat(mergeControl.getMergeZipList())
        .contains(
            MergeZip.newBuilder()
                .setEntryNamePrefix("Watch/y.app/")
                .setSourcePath(libsbOutputZip.getExecPathString())
                .build());
  }

  protected void addStoryboards() throws Exception {
    scratch.file("lib/libsb.storyboard");
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("storyboards", "libsb.storyboard")
        .write();

    scratch.file("bndl/bndlsb.storyboard");
    scratch.file(
        "bndl/BUILD",
        "objc_bundle_library(",
        "    name = 'bndl',",
        "    storyboards = ['ext_bndlsb.storyboard'],",
        ")");

    scratch.file(
        "x/BUILD",
        "apple_binary(",
        "    name = 'bin',",
        "    srcs = ['a.m'],",
        "    bundles = ['//bndl:bndl'],",
        "    deps = ['//lib:lib'],",
        "    storyboards = ['ext.storyboard'],",
        "    platform_type = 'watchos',",
        ")",
        "",
        "apple_watch2_extension(",
        "    name = 'x',",
        "    app_name = 'y',",
        "    app_asset_catalogs = ['foo.xcassets/foo', 'bar.xcassets/bar'],",
        "    app_storyboards = ['appsb.storyboard'],",
        "    binary = ':bin',",
        ")");

    scratch.file("x/appsb.storyboard");
    getConfiguredTarget("//x:x");
  }

  @Test
  public void testMergesPartialInfoplists() throws Exception {
    scratch.file("x/primary-Info.plist");
    setArtifactPrefix("y");
    RULE_TYPE_PAIR.scratchTargets(
        scratch,
        "app_asset_catalogs",
        "['foo.xcassets/bar']",
        "app_infoplists",
        "['primary-Info.plist']");

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

    RULE_TYPE.scratchTarget(scratch, WATCH_EXT_INFOPLISTS_ATTR, "['Ext-Info.plist']");

    BundleMergeProtos.Control control = bundleMergeControl("//x:x");

    assertThat(control.hasPrimaryBundleIdentifier()).isFalse();
    assertThat(control.getFallbackBundleIdentifier()).isEqualTo("example.ext.x");
  }

  @Test
  public void testPassesApplicationFallbackBundleIdToBundleMerging() throws Exception {
    setArtifactPrefix("y");
    scratch.file("bin/a.m");
    scratch.file("bin/App-Info.plist");

    RULE_TYPE.scratchTarget(scratch, WATCH_APP_INFOPLISTS_ATTR, "['App-Info.plist']");

    BundleMergeProtos.Control control = bundleMergeControl("//x:x");

    assertThat(control.hasPrimaryBundleIdentifier()).isFalse();
    assertThat(control.getFallbackBundleIdentifier()).isEqualTo("example.app.y");
  }

  @Test
  public void testPassesExtensionPrimaryBundleIdToBundleMerging() throws Exception {
    scratch.file("bin/a.m");
    scratch.file("bin/Ext-Info.plist");

    RULE_TYPE.scratchTarget(
        scratch,
        WATCH_EXT_INFOPLISTS_ATTR,
        "['Ext-Info.plist']",
        WATCH_EXT_BUNDLE_ID_ATTR,
        "'com.bundle.ext.id'");

    BundleMergeProtos.Control control = bundleMergeControl("//x:x");

    assertThat(control.getPrimaryBundleIdentifier()).isEqualTo("com.bundle.ext.id");
    assertThat(control.hasFallbackBundleIdentifier()).isFalse();
  }

  @Test
  public void testPassesApplicationPrimaryBundleIdToBundleMerging() throws Exception {
    setArtifactPrefix("y");
    scratch.file("bin/a.m");
    scratch.file("bin/App-Info.plist");

    RULE_TYPE.scratchTarget(
        scratch,
        WATCH_APP_INFOPLISTS_ATTR,
        "['App-Info.plist']",
        WATCH_APP_BUNDLE_ID_ATTR,
        "'com.bundle.app.id'");

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
            "x/bin_lipobin",
            "tools/objc/bundlemerge",
            "x/x.ipa-control",
            "x/x-MergedInfo.plist");
  }

  @Test
  public void testDeviceSimulatorMismatch() throws Exception {
    useConfiguration("--ios_multi_cpus=i386,x86_64", "--watchos_cpus=armv7k");
    checkError(
        "x",
        "x",
        "Building a watch extension for watch device architectures [armv7k] "
            + "requires a device ios architecture. Found [i386,x86_64] instead.",
        "apple_binary(",
        "    name = 'bin',",
        "    srcs = ['a.m'],",
        "    platform_type = 'watchos',",
        ")",
        "",
        "apple_watch2_extension(",
        "    name = 'x',",
        "    app_name = 'y',",
        "    binary = ':bin',",
        ")");
  }

  @Override
  protected void addCommonResources(BinaryRuleTypePair ruleTypePair) throws Exception {
    ruleTypePair.scratchTargets(
        scratch, "strings", "['foo.strings']", "storyboards", "['baz.storyboard']");
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
    useConfiguration("--watchos_sdk_version=2.2");
    checkCompileXibActions(RULE_TYPE_PAIR, DottedVersion.fromString("2.2"), "watch");
  }

  @Test
  public void testRegistersStoryboardCompileActions() throws Exception {
    useConfiguration("--watchos_sdk_version=2.2");
    checkRegistersStoryboardCompileActions(RULE_TYPE_PAIR, DottedVersion.fromString("2.2"),
        "watch");
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
  public void testWatchSdkDefaultMinVersion() throws Exception {
    useConfiguration("--ios_minimum_os=7.1", "--watchos_sdk_version=2.4");
    addMockExtensionAndLibs("ext_infoplists = ['Info.plist']");

    assertThat(bundleMergeControl("//x:x").getMinimumOsVersion()).isEqualTo("2.4");
  }
  
  @Test
  public void testWatchSdkMinimumOs() throws Exception {
    useConfiguration("--ios_minimum_os=7.1", "--watchos_sdk_version=2.2",
        "--watchos_minimum_os=2.0");
    addMockExtensionAndLibs("ext_infoplists = ['Info.plist']");

    assertThat(bundleMergeControl("//x:x").getMinimumOsVersion()).isEqualTo("2.0");
  }

  @Test
  public void testCheckExtensionPrimaryBundleIdInMergedPlist() throws Exception {
    RULE_TYPE_PAIR.scratchTargets(
        scratch,
        WATCH_EXT_INFOPLISTS_ATTR,
        "['Info.plist']",
        WATCH_EXT_BUNDLE_ID_ATTR,
        "'com.ext.bundle.id'");
    scratch.file("ext/Info.plist");

    checkBundleIdFlagsInPlistMergeAction(
        Optional.of("com.ext.bundle.id"),
        getVariableSubstitutionArguments(RULE_TYPE_PAIR),
        "example.ext.x");
  }

  @Test
  public void testCheckApplicationPrimaryBundleIdInMergedPlist() throws Exception {
    setArtifactPrefix("y");
    RULE_TYPE_PAIR.scratchTargets(
        scratch,
        WATCH_APP_INFOPLISTS_ATTR,
        "['Info.plist']",
        WATCH_APP_BUNDLE_ID_ATTR,
        "'com.app.bundle.id'");
    scratch.file("app/Info.plist");

    checkBundleIdFlagsInPlistMergeAction(
        Optional.of("com.app.bundle.id"),
        variableSubstitutionsForWatchApplication(),
        "example.app.y");
  }

  @Test
  public void testCheckExtensionFallbackBundleIdInMergedPlist() throws Exception {
    RULE_TYPE_PAIR.scratchTargets(scratch, WATCH_EXT_INFOPLISTS_ATTR, "['Info.plist']");
    scratch.file("ext/Info.plist");

    checkBundleIdFlagsInPlistMergeAction(
        Optional.<String>absent(),
        getVariableSubstitutionArguments(RULE_TYPE_PAIR),
        "example.ext.x");
  }

  @Test
  public void testCheckApplicationFallbackBundleIdInMergedPlist() throws Exception {
    setArtifactPrefix("y");
    RULE_TYPE_PAIR.scratchTargets(scratch, WATCH_APP_INFOPLISTS_ATTR, "['Info.plist']");
    scratch.file("app/Info.plist");

    checkBundleIdFlagsInPlistMergeAction(
        Optional.<String>absent(), variableSubstitutionsForWatchApplication(), "example.app.y");
  }

  @Test
  public void testSameStringsTwice() throws Exception {
    String targets =
        RULE_TYPE.target(
            scratch,
            "x",
            "bndl",
            "app_resources",
            "['Resources/en.lproj/foo.strings']",
            "app_strings",
            "['Resources/en.lproj/foo.strings']");
    checkError(
        "x",
        "bndl",
        "The same file was included multiple times in this rule: x/Resources/en.lproj/foo.strings",
        targets);
  }

  private ImmutableMap<String, String> variableSubstitutionsForWatchApplication() {
    return new ImmutableMap.Builder<String, String>()
        .put("EXECUTABLE_NAME", "y")
        .put("BUNDLE_NAME", "y.app")
        .put("PRODUCT_NAME", "y")
        .build();
  }

  protected void createSwiftBinaryTarget(String... lines) throws Exception {
    scratch.file("x/main.m");

    scratch.file("examples/rule/BUILD");
    scratch.file(
        "examples/rule/apple_rules.bzl",
        "def swift_rule_impl(ctx):",
        "  return struct(objc=apple_common.new_objc_provider(uses_swift=True))",
        "swift_rule = rule(implementation = swift_rule_impl, attrs = {})");

    String[] impl =
        ObjectArrays.concat(
            new String[] {
              "load('//examples/rule:apple_rules.bzl', 'swift_rule')",
              "swift_rule(name='swift_bin')",
              "apple_binary(",
              "    name = 'x',",
              "    srcs = ['main.m'],",
              "    deps = [':swift_bin'],",
              "    platform_type = 'watchos',",
              ")",
              "",
            },
            lines,
            String.class);
    scratch.file("x/BUILD", impl);
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
  // Regression test for b/30916137. Verifies that all tools are available in the watch2 extension
  // rule to handle bundling of swift objects.
  public void testSwiftSrcs() throws Exception {
    createSwiftBinaryTarget(
        "apple_watch2_extension(",
        "    name = 'ext',",
        "    app_name = 'y',",
        "    app_asset_catalogs = ['foo.xcassets/foo', 'bar.xcassets/bar'],",
        "    binary = ':x',",
        ")");

     getConfiguredTarget("//x:x");
  }

  @Override
  protected Action ipaGeneratingAction() throws Exception {
    ConfiguredTarget test = getConfiguredTarget("//x:x");
    return getGeneratingAction(getBinArtifact("x.ipa", test));
  }

  @Test
  public void testMergeBundleActionsWithNestedBundle() throws Exception {
    checkMergeBundleActionsWithNestedBundle(RULE_TYPE_PAIR, getTargetConfiguration());
  }

  @Test
  public void testIncludesStoryboardOutputZipsAsMergeZipsForExtension() throws Exception {
    addStoryboards();

    Artifact libsbOutputZip = getBinArtifact("x/libsb.storyboard.zip", "//x:x");
    Artifact extBndlsbOutputZip =
        getBinArtifact("bndl/ext_bndlsb.storyboard.zip", getConfiguredTarget("//bndl:bndl"));
    Artifact extsbOutputZip = getBinArtifact("x/ext.storyboard.zip", "//x:x");

    String bundleDir = RULE_TYPE_PAIR.getBundleDir();
    Control mergeControl = bundleMergeControl("//x:x");
    assertThat(mergeControl.getMergeZipList())
        .containsExactly(
            MergeZip.newBuilder()
                .setEntryNamePrefix(bundleDir + "/")
                .setSourcePath(libsbOutputZip.getExecPathString())
                .build(),
            MergeZip.newBuilder()
                .setEntryNamePrefix(bundleDir + "/")
                .setSourcePath(extsbOutputZip.getExecPathString())
                .build());

    Control nestedMergeControl = Iterables.getOnlyElement(mergeControl.getNestedBundleList());
    assertThat(nestedMergeControl.getMergeZipList())
        .containsExactly(
            MergeZip.newBuilder()
                .setEntryNamePrefix(bundleDir + "/bndl.bundle/")
                .setSourcePath(extBndlsbOutputZip.getExecPathString())
                .build());
  }
}
