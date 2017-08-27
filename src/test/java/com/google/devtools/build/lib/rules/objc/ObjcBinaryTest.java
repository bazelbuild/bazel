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
import static com.google.devtools.build.lib.rules.objc.BinaryLinkingTargetFactory.REQUIRES_AT_LEAST_ONE_LIBRARY_OR_SOURCE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.APP_ICON_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.LAUNCH_IMAGE_ATTR;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultiset;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Multiset;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.BundleFile;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.MergeZip;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for objc_binary. */
@RunWith(JUnit4.class)
public class ObjcBinaryTest extends ObjcRuleTestCase {
  static final RuleType RULE_TYPE = new BinaryRuleType("objc_binary");

  protected ConfiguredTarget addMockBinAndLibs(List<String> srcs) throws Exception {
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

  @Before
  public final void initializeToolsConfigMock() throws Exception {
    MockProtoSupport.setup(mockToolsConfig);
    MockObjcSupport.setup(mockToolsConfig);
  }

  @Test
  public void testCreate_runfiles() throws Exception {
    ConfiguredTarget binary = addMockBinAndLibs(ImmutableList.of("a.m"));
    RunfilesProvider runfiles = binary.getProvider(RunfilesProvider.class);
    assertThat(runfiles.getDefaultRunfiles().getArtifacts()).isEmpty();
    assertThat(Artifact.toRootRelativePaths(runfiles.getDataRunfiles().getArtifacts()))
        .containsExactly(
            "bin/bin.ipa",
            "bin/bin_bin");
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
  public void testLinkActionDuplicateInputs() throws Exception {
    checkLinkActionDuplicateInputs(RULE_TYPE, new ExtraLinkArgs());
  }

  @Test
  /**
   * Tests that bitcode is disabled for simulator builds even if enabled by flag.
   */
  public void testLinkActionsWithBitcode_simulator() throws Exception {
    useConfiguration("--xcode_version=7.1", "--apple_bitcode=embedded",
        "--ios_multi_cpus=x86_64");
    createBinaryTargetWriter("//objc:bin").setAndCreateFiles("srcs", "a.m").write();

    CommandAction linkAction = linkAction("//objc:bin");

    String commandLine = Joiner.on(" ").join(linkAction.getArguments());
    assertThat(commandLine).doesNotContain("-fembed-bitcode");
    assertThat(commandLine).doesNotContain("-fembed-bitcode-marker");
  }

  @Test
  public void testLinkActionsWithNoBitcode() throws Exception {
    useConfiguration("--xcode_version=7.1", "--apple_bitcode=none",
        "--ios_multi_cpus=arm64");
    createBinaryTargetWriter("//objc:bin").setAndCreateFiles("srcs", "a.m").write();

    CommandAction linkAction = linkAction("//objc:bin");

    String commandLine = Joiner.on(" ").join(linkAction.getArguments());
    assertThat(commandLine).doesNotContain("-fembed-bitcode");
    assertThat(commandLine).doesNotContain("-fembed-bitcode-marker");
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
  public void testProvisioningProfile_simulatorBuild() throws Exception {
    useConfiguration("--cpu=ios_i386");
    addMockBinAndLibs(ImmutableList.of("a.m"));

    Artifact provisioningProfile =
        getFileConfiguredTarget("//tools/objc:foo.mobileprovision").getArtifact();
    SpawnAction spawnAction = bundleMergeAction("//bin:bin");
    assertThat(spawnAction.getInputs()).doesNotContain(provisioningProfile);

    BundleMergeProtos.Control control = bundleMergeControl("//bin:bin");
    assertThat(mobileProvisionProfiles(control)).isEmpty();
  }

  @Test
  public void testProvisioningProfile_deviceBuild() throws Exception {
    useConfiguration("--cpu=ios_armv7");

    addMockBinAndLibs(ImmutableList.of("a.m"));

    Artifact provisioningProfile =
        getFileConfiguredTarget("//tools/objc:foo.mobileprovision").getArtifact();
    SpawnAction spawnAction = bundleMergeAction("//bin:bin");
    assertThat(spawnAction.getInputs()).contains(provisioningProfile);

    BundleMergeProtos.Control control = bundleMergeControl("//bin:bin");
    Map<String, String> profiles = mobileProvisionProfiles(control);
    ImmutableMap<String, String> expectedProfiles = ImmutableMap.of(
        provisioningProfile.getExecPathString(),
        ReleaseBundlingSupport.PROVISIONING_PROFILE_BUNDLE_FILE);
    assertThat(profiles).isEqualTo(expectedProfiles);
  }

  @Test
  public void testUserSpecifiedProvisioningProfile_deviceBuild() throws Exception {
    useConfiguration("--cpu=ios_armv7");
    scratch.file("custom/BUILD", "exports_files(['pp.mobileprovision'])");
    scratch.file("custom/pp.mobileprovision");
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createBinaryTargetWriter("//bin:bin")
        .setAndCreateFiles("srcs", "a.m")
        .setList("deps", "//lib1:lib1", "//lib2:lib2")
        .set("provisioning_profile", "'//custom:pp.mobileprovision'")
        .write();

    Artifact defaultProvisioningProfile =
        getFileConfiguredTarget("//tools/objc:foo.mobileprovision").getArtifact();
    Artifact customProvisioningProfile =
        getFileConfiguredTarget("//custom:pp.mobileprovision").getArtifact();
    SpawnAction spawnAction = bundleMergeAction("//bin:bin");
    assertThat(spawnAction.getInputs()).contains(customProvisioningProfile);
    assertThat(spawnAction.getInputs()).doesNotContain(defaultProvisioningProfile);

    BundleMergeProtos.Control control = bundleMergeControl("//bin:bin");
    Map<String, String> profiles = mobileProvisionProfiles(control);
    Map<String, String> expectedProfiles = ImmutableMap.of(
        customProvisioningProfile.getExecPathString(),
        ReleaseBundlingSupport.PROVISIONING_PROFILE_BUNDLE_FILE);
    assertThat(profiles).isEqualTo(expectedProfiles);
  }

  @Test
  public void testCreate_mergeControlAction() throws Exception {
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createBinaryTargetWriter("//bin:bin")
        .setAndCreateFiles("srcs", "a.m")
        .setList("deps", "//lib1:lib1", "//lib2:lib2")
        .set("infoplist", "'bin-Info.plist'")
        .write();

    Action mergeAction = bundleMergeAction("//bin:bin");
    Action action = bundleMergeControlAction("//bin:bin");
    assertThat(action.getInputs()).isEmpty();
    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly(
        "bin/bin.ipa-control");
    assertThat(bundleMergeControl("//bin:bin"))
        .isEqualTo(
            BundleMergeProtos.Control.newBuilder()
                .addBundleFile(
                    BundleFile.newBuilder()
                        .setSourceFile(execPathEndingWith(mergeAction.getInputs(), "bin"))
                        .setBundlePath("bin")
                        .setExternalFileAttribute(BundleableFile.EXECUTABLE_EXTERNAL_FILE_ATTRIBUTE)
                        .build())
                .setBundleRoot("Payload/bin.app")
                .setBundleInfoPlistFile(
                    execPathEndingWith(mergeAction.getInputs(), "bin-MergedInfo.plist"))
                .setOutFile(execPathEndingWith(mergeAction.getOutputs(), "bin.unprocessed.ipa"))
                .setMinimumOsVersion(DEFAULT_IOS_SDK_VERSION.toString())
                .setSdkVersion(DEFAULT_IOS_SDK_VERSION.toString())
                .setPlatform("IOS_SIMULATOR")
                .setFallbackBundleIdentifier("example.bin")
                .build());
  }

  @Test
  public void testCreate_mergeBundleAction() throws Exception {
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createBinaryTargetWriter("//bin:bin")
        .setAndCreateFiles("srcs", "a.m")
        .setList("deps", "//lib1:lib1", "//lib2:lib2")
        .set("infoplist", "'bin-Info.plist'")
        .write();

    SpawnAction action = bundleMergeAction("//bin:bin");
    assertThat(Artifact.toRootRelativePaths(action.getInputs()))
        .containsExactly(
            MOCK_BUNDLEMERGE_PATH,
            "bin/bin_lipobin",
            "bin/bin.ipa-control",
            "bin/bin-MergedInfo.plist");
    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("bin/bin.unprocessed.ipa");
    assertNotRequiresDarwin(action);
    assertThat(action.getEnvironment()).isEmpty();
    assertThat(action.getArguments())
        .containsExactly(
            MOCK_BUNDLEMERGE_PATH, execPathEndingWith(action.getInputs(), "bin.ipa-control"))
        .inOrder();
  }

  @Test
  public void testCheckPrimaryBundleIdInMergedPlist() throws Exception {
    checkPrimaryBundleIdInMergedPlist(RULE_TYPE);
  }

  @Test
  public void testCheckFallbackBundleIdInMergedPlist() throws Exception {
    checkFallbackBundleIdInMergedPlist(RULE_TYPE);
  }

  @Test
  public void testCreate_errorForNoSourceOrDep() throws Exception {
    scratch.file("x/Foo.plist");
    checkError("x", "x", REQUIRES_AT_LEAST_ONE_LIBRARY_OR_SOURCE_FILE,
        "objc_binary(name='x')");
  }

  @Test
  public void testCompileWithDotMFileInHeaders() throws Exception {
    checkCompileWithDotMFileInHeaders(RULE_TYPE);
  }

  @Test
  public void testCreate_NoDebugSymbolActionWithoutAppleFlag() throws Exception {
    checkNoDebugSymbolFileWithoutAppleFlag(RULE_TYPE);
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
    scratch.file("lib/ac.xcassets/foo");
    scratch.file("lib/ac.xcassets/bar");
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .set("asset_catalogs", "glob(['ac.xcassets/**'])")
        .write();
    scratch.file("bin/ac.xcassets/baz");
    scratch.file("bin/ac.xcassets/42");
    createBinaryTargetWriter("//bin:bin")
        .setAndCreateFiles("srcs", "src.m")
        .setList("deps", "//lib:lib")
        .set("asset_catalogs", "glob(['ac.xcassets/**'])")
        .write();

    // Test that the actoolzip Action has arguments and inputs obtained from dependencies.
    SpawnAction actoolZipAction = actoolZipActionForIpa("//bin:bin");
    assertThat(Artifact.toExecPaths(actoolZipAction.getInputs())).containsExactly(
        "lib/ac.xcassets/foo", "lib/ac.xcassets/bar", "bin/ac.xcassets/baz", "bin/ac.xcassets/42",
        MOCK_ACTOOLWRAPPER_PATH);
    assertContainsSublist(actoolZipAction.getArguments(),
        ImmutableList.of("lib/ac.xcassets", "bin/ac.xcassets"));
  }

  @Test
  public void testCcDependencyLinkoptsArePropagatedToLinkAction() throws Exception {
    useConfiguration("--experimental_disable_go", "--experimental_disable_jvm", "--cpu=ios_i386",
        "--crosstool_top=//tools/osx/crosstool:crosstool");

    scratch.file("bin/BUILD",
        "cc_library(",
        "    name = 'cclib1',",
        "    srcs = ['dep1.c'],",
        "    linkopts = ['-framework F1', '-framework F2', '-Wl,--other-opt'],",
        ")",
        "cc_library(",
        "    name = 'cclib2',",
        "    srcs = ['dep2.c'],",
        "    linkopts = ['-another-opt', '-framework F2'],",
        "    deps = ['cclib1'],",
        ")",
        "cc_library(",
        "    name = 'cclib3',",
        "    srcs = ['dep2.c'],",
        "    linkopts = ['-one-more-opt', '-framework UIKit'],",
        "    deps = ['cclib1'],",
        ")",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        "    deps = [':cclib2', ':cclib3'],",
        ")");

    // Frameworks from the CROSSTOOL "apply_implicit_frameworks" feature should be present.
    assertThat(Joiner.on(" ").join(linkAction("//bin").getArguments()))
        .contains("-framework Foundation -framework UIKit");
    // Frameworks included in linkopts by the user should get placed together with no duplicates.
    // (They may duplicate the ones inserted by the CROSSTOOL feature, but we don't test that here.)
    assertThat(Joiner.on(" ").join(linkAction("//bin").getArguments()))
        .contains("-framework F2 -framework F1");
    // Linkopts should also be grouped together.
    assertThat(Joiner.on(" ").join(linkAction("//bin").getArguments()))
        .contains("-another-opt -Wl,--other-opt -one-more-opt");
  }

  @Test
  public void testAlwaysLinkCcDependenciesAreForceLoaded() throws Exception {
    useConfiguration("--experimental_disable_go", "--experimental_disable_jvm", "--cpu=ios_i386",
        "--crosstool_top=//tools/osx/crosstool:crosstool");

    scratch.file("bin/BUILD",
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
  public void testSpecifyAppIconAndLaunchImageUsingXcassetsOfDependency() throws Exception {
    checkSpecifyAppIconAndLaunchImageUsingXcassetsOfDependency(RULE_TYPE);
  }

  @Test
  public void testCreate_actoolAction() throws Exception {
    addTargetWithAssetCatalogs(RULE_TYPE);
    checkActoolActionCorrectness(DEFAULT_IOS_SDK_VERSION);
  }

  @Test
  public void testPassesFamiliesToActool() throws Exception {
    checkPassesFamiliesToActool(RULE_TYPE);
  }

  @Test
  public void testPassesFamiliesToIbtool() throws Exception {
    checkPassesFamiliesToIbtool(RULE_TYPE);
  }

  @Test
  public void testReportsErrorsForInvalidFamiliesAttribute() throws Exception {
    checkReportsErrorsForInvalidFamiliesAttribute(RULE_TYPE);
  }

  @SuppressWarnings("deprecation") // getMergeWithoutNamePrefixZipList is deprecated
  @Test
  public void testCreate_mergeActionsWithAssetCatalog() throws Exception {
    // TODO(matvore): add this test to IosTestTest.java.
    addTargetWithAssetCatalogs(RULE_TYPE);

    Artifact actoolZipOut = getBinArtifact("x.actool.zip", "//x:x");
    assertThat(bundleMergeAction("//x:x").getInputs()).contains(actoolZipOut);

    BundleMergeProtos.Control mergeControl = bundleMergeControl("//x:x");
    assertThat(mergeControl.getMergeZipList()).containsExactly(MergeZip.newBuilder()
            .setEntryNamePrefix("Payload/x.app/")
            .setSourcePath(actoolZipOut.getExecPathString())
            .build());
  }

  private void addBinAndLibWithRawResources() throws Exception {
    addBinAndLibWithResources("resources", "resource1.txt", "ja.lproj/resource2.txt",
        "objc_binary");
  }

  private void addBinAndLibWithStrings() throws Exception {
    addBinAndLibWithResources("strings", "foo.strings", "ja.lproj/bar.strings",
        "objc_binary");
  }

  @Test
  public void testCollectsRawResourceFilesTransitively() throws Exception {
    addBinAndLibWithRawResources();
    checkCollectsResourceFilesTransitively(
        "//bin:bin",
        ImmutableList.of("lib/resource1.txt", "bin/ja.lproj/resource2.txt"),
        ImmutableList.of("lib/resource1.txt"),
        ImmutableSetMultimap.<String, Multiset<String>>of(
            "bin_bin",
            ImmutableMultiset.of("bin/ja.lproj/resource2.txt", "lib/resource1.txt"),
            "bin_static_lib_bin",
            ImmutableMultiset.of("bin/ja.lproj/resource2.txt", "lib/resource1.txt"),
            "lib_lib",
            ImmutableMultiset.of("lib/resource1.txt")));
  }

  @Test
  public void testCollectsStringsFilesTransitively() throws Exception {
    addBinAndLibWithStrings();
    checkCollectsResourceFilesTransitively(
        "//bin:bin",
        ImmutableList.of("bin/lib/foo.strings.binary", "bin/bin/ja.lproj/bar.strings.binary"),
        ImmutableList.of("lib/foo.strings.binary"),
        ImmutableSetMultimap.<String, Multiset<String>>of(
            "bin_bin",
            ImmutableMultiset.of("bin/ja.lproj/bar.strings", "lib/foo.strings"),
            "bin_static_lib_bin",
            ImmutableMultiset.of("bin/ja.lproj/bar.strings", "lib/foo.strings"),
            "lib_lib",
            ImmutableMultiset.of("lib/foo.strings")));
  }

  @Test
  public void testResourceFilesMergedInBundle() throws Exception {
    addBinAndLibWithRawResources();
    checkBundleablesAreMerged("//bin:bin",
        ImmutableListMultimap.of(
            "resource1.txt", "resource1.txt",
            "ja.lproj/resource2.txt", "ja.lproj/resource2.txt"));
  }

  @Test
  public void testStringsFilesMergedInBundle() throws Exception {
    addBinAndLibWithStrings();
    checkBundleablesAreMerged("//bin:bin",
        ImmutableListMultimap.of(
            "foo.strings.binary", "foo.strings",
            "ja.lproj/bar.strings.binary", "ja.lproj/bar.strings"));
  }

  @Test
  public void testLinksFrameworksOfSelfAndTransitiveDependencies() throws Exception {
    checkLinksFrameworksOfSelfAndTransitiveDependencies(RULE_TYPE);
  }

  @Test
  public void testLinksWeakFrameworksOfSelfAndTransitiveDependencies() throws Exception {
    checkLinksWeakFrameworksOfSelfAndTransitiveDependencies(RULE_TYPE);
  }

  @Test
  public void testMergesXcdatamodelZips() throws Exception {
    checkMergesXcdatamodelZips(RULE_TYPE);
  }

  @Test
  public void testPlistRequiresDotInName() throws Exception {
    checkError("x", "x",
        "'//x:Infoplist' does not produce any objc_binary infoplist files (expected .plist)",
        "objc_binary(",
        "    name = 'x',",
        "    srcs = ['a.m'],",
        "    infoplist = 'Infoplist'",
        ")");
  }

  @Test
  public void testLinkIncludeOrder_staticLibsFirst() throws Exception {
    checkLinkIncludeOrderStaticLibsFirst(RULE_TYPE);
  }

  @Test
  public void testLinksDylibsTransitively() throws Exception {
    checkLinksDylibsTransitively(RULE_TYPE);
  }

  @Test
  public void testPopulatesCompilationArtifacts() throws Exception {
    checkPopulatesCompilationArtifacts(RULE_TYPE);
  }

  @Test
  public void testArchivesPrecompiledObjectFiles() throws Exception {
    checkArchivesPrecompiledObjectFiles(RULE_TYPE);
  }
 
  @Test
  public void testPopulatesBundling() throws Exception {
    checkPopulatesBundling(RULE_TYPE);
  }

  @Test
  public void testRegistersStoryboardCompilationActions() throws Exception {
    checkRegistersStoryboardCompileActions(RULE_TYPE, "iphone");
  }

  @Test
  public void testSwiftStdlibActions() throws Exception {
    checkRegisterSwiftStdlibActions(RULE_TYPE, "iphonesimulator");
  }

  @Test
  public void testSwiftStdlibActionsWithToolchain() throws Exception {
    useConfiguration("--xcode_toolchain=test_toolchain");
    checkRegisterSwiftStdlibActions(RULE_TYPE, "iphonesimulator", "test_toolchain");
  }

  @Test
  public void testRegistersSwiftSupportActions() throws Exception {
    checkRegisterSwiftSupportActions(RULE_TYPE, "iphonesimulator");
  }

  @Test
  public void testRegistersSwiftSupportActionsWithToolchain() throws Exception {
    useConfiguration("--xcode_toolchain=test_toolchain");
    checkRegisterSwiftSupportActions(RULE_TYPE, "iphonesimulator", "test_toolchain");
  }

  @Test
  public void testErrorsWrongFileTypeForSrcsWhenCompiling() throws Exception {
    checkErrorsWrongFileTypeForSrcsWhenCompiling(RULE_TYPE);
  }

  @Test
  public void testObjcCopts() throws Exception {
    checkObjcCopts(RULE_TYPE);
  }

  @Test
  public void testObjcCopts_argumentOrdering() throws Exception {
    checkObjcCopts_argumentOrdering(RULE_TYPE);
  }

  @Test
  public void testMergesActoolPartialInfoplist() throws Exception {
    checkMergesPartialInfoplists(RULE_TYPE);
  }

  @Test
  public void checkDefinesFromCcLibraryDep() throws Exception {
    checkDefinesFromCcLibraryDep(RULE_TYPE);
  }

  @Test
  public void testCompileXibActions() throws Exception {
    checkCompileXibActions(RULE_TYPE);
  }

  @Test
  public void testNibZipsMergedIntoBundle() throws Exception {
    checkNibZipsMergedIntoBundle(RULE_TYPE);
  }

  @Test
  public void testAllowVariousNonBlacklistedTypesInHeaders() throws Exception {
    checkAllowVariousNonBlacklistedTypesInHeaders(RULE_TYPE);
  }

  @Test
  public void testWarningForBlacklistedTypesInHeaders() throws Exception {
    checkWarningForBlacklistedTypesInHeaders(RULE_TYPE);
  }

  @Test
  public void testCppSourceCompilesWithCppFlags() throws Exception {
    checkCppSourceCompilesWithCppFlags(RULE_TYPE);
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
  public void testNestedBundleIdIsNotAffectedByParent() throws Exception {
    scratch.file("bndl/BUILD",
        "objc_bundle_library(",
        "    name = 'bndl',",
        ")");

    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("hdrs", "lib.h")
        .setList("bundles", "//bndl:bndl")
        .write();

    createBinaryTargetWriter("//bin:bin")
        .setAndCreateFiles("srcs", "a.m")
        .setList("deps", "//lib:lib")
        .set("bundle_id", "'com.main.bundle'")
        .write();

    BundleMergeProtos.Control control = bundleMergeControl("//bin:bin");

    assertThat(control.getPrimaryBundleIdentifier()).isEqualTo("com.main.bundle");
    // The nested bndl should not get its parent's bundle_id
    assertThat(control.getNestedBundleList().get(0).getPrimaryBundleIdentifier())
      .isNotEqualTo("com.main.bundle");
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
  public void testLinkOpts() throws Exception {
    checkLinkopts(RULE_TYPE);
  }

  @Test
  public void testProtoBundlingAndLinking() throws Exception {
    checkProtoBundlingAndLinking(RULE_TYPE);
  }

  @Test
  public void testProtoBundlingWithTargetsWithNoDeps() throws Exception {
    checkProtoBundlingWithTargetsWithNoDeps(RULE_TYPE);
  }
  
  @Test
  public void testCanUseCrosstool() throws Exception {
    checkLinkingRuleCanUseCrosstool(RULE_TYPE);
  }

  @Test
  public void testBinaryStrippings() throws Exception {
    checkBinaryStripAction(RULE_TYPE);
  }

  @Test
  public void testAppleSdkVersionEnv() throws Exception {
    addMockBinAndLibs(ImmutableList.of("a.m"));
    CommandAction action = linkAction("//bin:bin");

    assertAppleSdkVersionEnv(action);
  }

  @Test
  public void testNonDefaultAppleSdkVersionEnv() throws Exception {
    useConfiguration("--ios_sdk_version=8.1");

    addMockBinAndLibs(ImmutableList.of("a.m"));
    CommandAction action = linkAction("//bin:bin");

    assertAppleSdkVersionEnv(action, "8.1");
  }

  @Test
  public void testAppleSdkDefaultPlatformEnv() throws Exception {
    addMockBinAndLibs(ImmutableList.of("a.m"));
    CommandAction action = linkAction("//bin:bin");

    assertAppleSdkPlatformEnv(action, "iPhoneSimulator");
  }

  @Test
  public void testAppleSdkDevicePlatformEnv() throws Exception {
    useConfiguration("--cpu=ios_arm64");

    addMockBinAndLibs(ImmutableList.of("a.m"));
    CommandAction action = linkAction("//bin:bin");

    assertAppleSdkPlatformEnv(action, "iPhoneOS");
  }

  @Test
  public void testMergeBundleActionsWithNestedBundle() throws Exception {
    checkMergeBundleActionsWithNestedBundle(RULE_TYPE);
  }

  @Test
  public void testIncludesStoryboardOutputZipsAsMergeZips() throws Exception {
    checkIncludesStoryboardOutputZipsAsMergeZips(RULE_TYPE);
  }

  @Test
  public void testCompilationActionsForDebug() throws Exception {
    checkClangCoptsForCompilationMode(RULE_TYPE, CompilationMode.DBG, CodeCoverageMode.NONE);
  }

  @Test
  public void testClangCoptsForDebugModeWithoutGlib() throws Exception {
    checkClangCoptsForDebugModeWithoutGlib(RULE_TYPE);
  }

  @Test
  public void testCompilationActionsForOptimized() throws Exception {
    checkClangCoptsForCompilationMode(RULE_TYPE, CompilationMode.OPT, CodeCoverageMode.NONE);
  }

  @Test
  public void testCcDependency() throws Exception {
    checkCcDependency(RULE_TYPE);
  }

  @Test
  public void testLinkActionCorrect() throws Exception {
    checkLinkActionCorrect(RULE_TYPE, new ExtraLinkArgs());
  }

  @Test
  public void testFrameworkDepLinkFlags() throws Exception {
    checkFrameworkDepLinkFlags(RULE_TYPE, new ExtraLinkArgs());
  }

  @Test
  public void testLinkActionsWithEmbeddedBitcode() throws Exception {
    useConfiguration("--xcode_version=7.1", "--apple_bitcode=embedded", "--ios_multi_cpus=arm64");
    createBinaryTargetWriter("//objc:bin").setAndCreateFiles("srcs", "a.m").write();

    CommandAction linkAction = linkAction("//objc:bin");
    String commandLine = Joiner.on(" ").join(linkAction.getArguments());

    assertThat(commandLine).contains("-fembed-bitcode");
    assertThat(commandLine).contains("-Xlinker -bitcode_verify");
    assertThat(commandLine).contains("-Xlinker -bitcode_hide_symbols");
  }

  @Test
  public void testLinkActionsWithEmbeddedBitcodeMarkers() throws Exception {
    useConfiguration(
        "--xcode_version=7.1", "--apple_bitcode=embedded_markers", "--ios_multi_cpus=arm64");
    createBinaryTargetWriter("//objc:bin").setAndCreateFiles("srcs", "a.m").write();

    CommandAction linkAction = linkAction("//objc:bin");

    assertThat(Joiner.on(" ").join(linkAction.getArguments())).contains("-fembed-bitcode-marker");
  }

  @Test
  public void testCompilationActionsForDebugInGcovCoverage() throws Exception {
    checkClangCoptsForCompilationMode(RULE_TYPE, CompilationMode.DBG,
        CodeCoverageMode.GCOV);
  }

  @Test
  public void testCompilationActionsForDebugInLlvmCovCoverage() throws Exception {
    checkClangCoptsForCompilationMode(RULE_TYPE, CompilationMode.DBG,
        CodeCoverageMode.LLVMCOV);
  }

  @Test
  public void testCompilationActionsForOptimizedInGcovCoverage() throws Exception {
    checkClangCoptsForCompilationMode(RULE_TYPE, CompilationMode.OPT,
        CodeCoverageMode.GCOV);
  }

  @Test
  public void testCompilationActionsForOptimizedInLlvmCovCoverage() throws Exception {
    checkClangCoptsForCompilationMode(RULE_TYPE, CompilationMode.OPT,
        CodeCoverageMode.LLVMCOV);
  }

  @Test
  public void testXcodeVersionEnv() throws Exception {
    useConfiguration("--xcode_version=5.8");

    addMockBinAndLibs(ImmutableList.of("a.m"));
    CommandAction action = linkAction("//bin:bin");

    assertXcodeVersionEnv(action, "5.8");
  }

  @Test
  public void testCompileWithTextualHeaders() throws Exception {
    checkCompileWithTextualHeaders(RULE_TYPE);
  }

  @Test
  public void testCompilesWithHdrs() throws Exception {
    checkCompilesWithHdrs(RULE_TYPE);
  }

  @Test
  public void testCompilesSources() throws Exception {
    checkCompilesSources(RULE_TYPE);
  }

  @Test
  public void testCreate_debugSymbolActionWithAppleFlag() throws Exception {
    useConfiguration("--apple_generate_dsym");
    RULE_TYPE.scratchTarget(scratch, "srcs", "['a.m']");
    ConfiguredTarget target = getConfiguredTarget("//x:x");

    Artifact artifact = getBinArtifact("x.app.dSYM.temp.zip", target);
    String execPath = artifact.getExecPath().getParentDirectory().toString();
    CommandAction linkAction = (CommandAction) getGeneratingAction(artifact);
    assertThat(linkAction.getArguments()).containsAllOf(
        "DSYM_HINT_LINKED_BINARY=" + execPath + "/x_bin",
        "DSYM_HINT_DSYM_PATH=" + execPath + "/x.app.dSYM.temp",
        "DSYM_HINT_DSYM_BUNDLE_ZIP=" + artifact.getExecPathString());

    Artifact plistArtifact = getBinArtifact("x.app.dSYM/Contents/Info.plist", target);
    Artifact debugSymbolArtifact =
        getBinArtifact("x.app.dSYM/Contents/Resources/DWARF/x_bin", target);
    SpawnAction plistAction = (SpawnAction) getGeneratingAction(plistArtifact);
    SpawnAction debugSymbolAction = (SpawnAction) getGeneratingAction(debugSymbolArtifact);
    assertThat(debugSymbolAction).isEqualTo(plistAction);

    String dsymUnzipActionArg =
        "unzip -p "
            + execPath
            + "/x.app.dSYM.temp.zip"
            + " Contents/Info.plist > "
            + plistArtifact.getExecPathString()
            + " && unzip -p "
            + execPath
            + "/x.app.dSYM.temp.zip"
            + " Contents/Resources/DWARF/x_bin > "
            + debugSymbolArtifact.getExecPathString();
    assertThat(plistAction.getArguments()).contains(dsymUnzipActionArg);
  }

  @Test
  public void testTargetHasDebugSymbols() throws Exception {
    checkTargetHasDebugSymbols(RULE_TYPE);
  }

  @Test
  public void testFilesToCompileOutputGroup() throws Exception {
    checkFilesToCompileOutputGroup(RULE_TYPE);
  }

  @Test
  public void testCustomModuleMap() throws Exception {
    checkCustomModuleMap(RULE_TYPE);
  }

  @Test
  public void testGenruleDependency() throws Exception {
    checkGenruleDependency(RULE_TYPE);
  }
}
