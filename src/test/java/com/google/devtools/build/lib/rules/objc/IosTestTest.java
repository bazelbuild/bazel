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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.apple.XcodeVersionProperties;
import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions.ObjcCrosstoolMode;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.xcode.plmerge.proto.PlMergeProtos;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for ios_test. */
@RunWith(JUnit4.class)
public class IosTestTest extends ObjcRuleTestCase {
  protected static final RuleType RULE_TYPE = new BinaryRuleType("ios_test");

  @Before
  public final void setUpToolsConfigMock() throws Exception  {
    MockObjcSupport.setupIosTest(mockToolsConfig);
    MockObjcSupport.setupIosSimDevice(mockToolsConfig);
    MockProtoSupport.setup(mockToolsConfig);
    MockObjcSupport.setup(mockToolsConfig);

    invalidatePackages();
  }

  @Test
  public void testRunfiles() throws Exception {
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    ConfiguredTarget target = addSimpleIosTest(
        "bin", "bin", ImmutableList.of("src.m"), ImmutableList.of("//lib1:lib1", "//lib2:lib2"));

    ImmutableList<String> expectedRunfiles =
        ImmutableList.of(
            "bin/bin.ipa",
            "tools/objc/xctest_app.ipa",
            "bin/bin_test_script",
            "tools/objc/StdRedirect.dylib",
            "tools/objc/testrunner");

    RunfilesProvider runfiles = target.getProvider(RunfilesProvider.class);
    assertThat(Artifact.toRootRelativePaths(runfiles.getDefaultRunfiles().getArtifacts()))
        .containsExactlyElementsIn(expectedRunfiles);
    assertThat(Artifact.toRootRelativePaths(runfiles.getDataRunfiles().getArtifacts()))
        .containsExactlyElementsIn(expectedRunfiles);
  }

  @Test
  public void testRunfilesInCoverage() throws Exception {
    useConfiguration("--collect_code_coverage", "--instrument_test_targets");
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    ConfiguredTarget target =
        addSimpleIosTest(
            "bin",
            "bin",
            ImmutableList.of("src.m"),
            ImmutableList.of("//lib1:lib1", "//lib2:lib2"));

    ImmutableList<String> expectedRunfiles =
        ImmutableList.of(
            "tools/objc/mcov",
            "bin/src.m",
            "lib1/a.m",
            "lib1/b.m",
            "lib2/a.m",
            "lib2/b.m",
            "tools/objc/objc_dummy.mm",
            "lib1/private.h",
            "lib1/hdr.h",
            "lib2/private.h",
            "lib2/hdr.h");

    RunfilesProvider runfiles = target.getProvider(RunfilesProvider.class);
    assertThat(Artifact.toRootRelativePaths(runfiles.getDefaultRunfiles().getArtifacts()))
        .containsAllIn(expectedRunfiles);
    assertThat(Artifact.toRootRelativePaths(runfiles.getDataRunfiles().getArtifacts()))
        .containsAllIn(expectedRunfiles);
  }

  @Test
  public void testInstrumentedFilesInCoverage() throws Exception {
    useConfiguration("--collect_code_coverage", "--instrument_test_targets");
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();

    ConfiguredTarget target =
        addSimpleIosTest(
            "bin",
            "bin",
            ImmutableList.of("src.m"),
            ImmutableList.of("//lib1:lib1", "//lib2:lib2"));

    InstrumentedFilesProvider instrumentedFilesProvider =
        target.getProvider(InstrumentedFilesProvider.class);
    assertThat(Artifact.toRootRelativePaths(instrumentedFilesProvider.getInstrumentedFiles()))
        .containsExactly(
            "bin/src.m",
            "lib1/a.m",
            "lib1/b.m",
            "lib2/a.m",
            "lib2/b.m",
            "tools/objc/objc_dummy.mm",
            "lib1/private.h",
            "lib1/hdr.h",
            "lib2/private.h",
            "lib2/hdr.h");
    assertThat(
            Artifact.toRootRelativePaths(
                instrumentedFilesProvider.getInstrumentationMetadataFiles()))
        .containsExactly(
            "bin/_objs/bin/bin/src.gcno",
            "lib1/_objs/lib1/lib1/a.gcno",
            "lib1/_objs/lib1/lib1/b.gcno",
            "lib2/_objs/lib2/lib2/a.gcno",
            "lib2/_objs/lib2/lib2/b.gcno",
            "tools/objc/_objs/xctest_appbin/tools/objc/objc_dummy.gcno");
  }

  @Test
  public void testBuildIpa() throws Exception {
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();

    ConfiguredTarget target =
        addSimpleIosTest(
            "bin",
            "bin",
            ImmutableList.of("src.m"),
            ImmutableList.of("//lib1:lib1", "//lib2:lib2"));

    Iterable<Artifact> filesToBuild = target.getProvider(FileProvider.class).getFilesToBuild();
    ImmutableList<String> expectedFilesToBuild =
        ImmutableList.of(
            "bin/bin.ipa",
            "tools/objc/xctest_app.ipa");
    assertThat(Artifact.toRootRelativePaths(filesToBuild))
        .containsExactlyElementsIn(expectedFilesToBuild);
  }

  @Test
  public void testXcTestAppIpaIsInFilesToBuild() throws Exception {
    scratch.file("x/BUILD",
        "ios_application(",
        "    name = 'xctest_app',",
        "    binary = ':xctest_app_bin',",
        "    infoplist = 'Info.plist',",
        ")",
        "",
        "objc_binary(",
        "    name = 'xctest_app_bin',",
        "    srcs = ['a.m'],",
        ")",
        "",
        "ios_test(",
        "    name = 'x',",
        "    xctest = 1,",
        "    xctest_app = ':xctest_app',",
        "    srcs = ['test.m'],",
        ")");
    ConfiguredTarget target = getConfiguredTarget("//x:x");
    Iterable<Artifact> filesToBuild = target.getProvider(FileProvider.class).getFilesToBuild();
    assertThat(filesToBuild).contains(getBinArtifact("xctest_app.ipa", target));
  }

  @Test
  public void testIpaIsImplicitOutput() throws Exception {
    addSimpleIosTest("bin", "bin", ImmutableList.of("src.m"), ImmutableList.<String>of());
    assertThat(getConfiguredTarget("//bin:bin.ipa")).isNotNull();
  }

  @Test
  public void testXcTest() throws Exception {
    setUpXCTestClient();
    ConfiguredTarget target = getConfiguredTarget("//test:XcTest");

    ImmutableList<String> expectedRunfiles =
        ImmutableList.of(
            "test/XcTest.ipa",
            "test/testApp.ipa",
            "test/XcTest_test_script",
            "tools/objc/StdRedirect.dylib",
            "tools/objc/testrunner");
    RunfilesProvider runfiles = target.getProvider(RunfilesProvider.class);
    assertThat(Artifact.toRootRelativePaths(runfiles.getDefaultRunfiles().getArtifacts()))
        .containsExactlyElementsIn(expectedRunfiles);
    assertThat(Artifact.toRootRelativePaths(runfiles.getDataRunfiles().getArtifacts()))
        .containsExactlyElementsIn(expectedRunfiles);
  }

  @Test
  public void testXcTestInCoverage() throws Exception {
    useConfiguration("--collect_code_coverage", "--instrument_test_targets");
    setUpXCTestClient();
    ConfiguredTarget target = getConfiguredTarget("//test:XcTest");

    InstrumentedFilesProvider instrumentedFilesProvider =
        target.getProvider(InstrumentedFilesProvider.class);
    assertThat(Artifact.toRootRelativePaths(instrumentedFilesProvider.getInstrumentedFiles()))
        .containsExactly("test/src.m", "test/test-src.m");
    assertThat(
            Artifact.toRootRelativePaths(
                instrumentedFilesProvider.getInstrumentationMetadataFiles()))
        .containsExactly(
            "test/_objs/XcTest/test/test-src.gcno", "test/_objs/testAppBin/test/src.gcno");
  }

  @Test
  public void testXcTestInCoverageFilter() throws Exception {
    useConfiguration("--collect_code_coverage", "--instrumentation_filter=-XcTest$");
    setUpXCTestClient();
    ConfiguredTarget target = getConfiguredTarget("//test:XcTest");


    InstrumentedFilesProvider instrumentedFilesProvider =
        target.getProvider(InstrumentedFilesProvider.class);

    // Missing "test/test-src.m" since the target including it has been excluded.
    assertThat(Artifact.toRootRelativePaths(instrumentedFilesProvider.getInstrumentedFiles()))
        .containsExactly("test/src.m");
  }

  @Test
  public void testXcTest_linkAction() throws Exception {
    setUpXCTestClient();
    CommandAction action = linkAction("//test:XcTest");

    String commandLine = Joiner.on(" ").join(action.getArguments());
    assertThat(commandLine).contains("-bundle");
    assertThat(commandLine).contains("-Xlinker -rpath -Xlinker @loader_path/Frameworks");
  }

  @Test
  public void testXcTest_linkAction_Crosstool() throws Exception {
    useConfiguration(ObjcCrosstoolMode.ALL);
    testXcTest_linkAction();
  }

  @Test
  public void testVariableSubstitution() throws Exception {
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();

    addSimpleIosTest(
        "bin", "bin", ImmutableList.of("src.m"), ImmutableList.of("//lib1:lib1", "//lib2:lib2"));

    PlMergeProtos.Control control = plMergeControl("//bin:bin");
    Map<String, String> substitutions = control.getVariableSubstitutionMapMap();
    assertThat(substitutions)
        .containsExactlyEntriesIn(
            ImmutableMap.<String, String>of(
                "EXECUTABLE_NAME", "bin",
                "BUNDLE_NAME", "bin.xctest",
                "PRODUCT_NAME", "bin"));
  }

  protected void setUpXCTestClient() throws Exception {
    scratch.file("/test/XcTest-Info.plist");
    scratch.file("/test/App-Info.plist");
    scratch.file("/test/src.m");
    scratch.file("/test/test-src.m");

    scratch.file("test/BUILD",
        "objc_binary(",
        "    name = 'testAppBin',",
        "    srcs = ['src.m'],",
        ")",
        "ios_application(",
        "    name = 'testApp',",
        "    binary = ':testAppBin',",
        ")",
        "ios_test(",
        "    name = 'XcTest',",
        "    srcs = ['test-src.m'],",
        "    xctest = True,",
        "    xctest_app = ':testApp',",
        ")");
  }

  @Test
  public void testCreate_recognizesDylibsAttribute() throws Exception {
    createBinaryTargetWriter("//bin:bin").setAndCreateFiles("srcs", "a.m").write();
    scratch.file("test/BUILD",
        "ios_application(",
        "    name = 'testApp',",
        "    binary = '//bin:bin',",
        ")",
        "ios_test(",
        "    name = 'test',",
        "    srcs = ['test-src.m'],",
        "    xctest = 1,",
        "    xctest_app = ':testApp',",
        "    sdk_dylibs = ['libdy'],",
        ")");
    CommandAction action = linkAction("//test:test");
    assertThat(Joiner.on(" ").join(action.getArguments())).contains("-ldy");
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
  public void testRegistersSwiftStdlibActions() throws Exception {
    checkRegisterSwiftStdlibActions(RULE_TYPE, "iphonesimulator");
  }

  @Test
  public void testRegistersSwiftSupportActions() throws Exception {
    checkRegisterSwiftSupportActions(RULE_TYPE, "iphonesimulator");
  }

  @Test
  public void testRegistersSwiftStdlibActionsWithToolchain() throws Exception {
    useConfiguration("--xcode_toolchain=test_toolchain");
    checkRegisterSwiftStdlibActions(RULE_TYPE, "iphonesimulator", "test_toolchain");
  }

  @Test
  public void testRegistersSwiftSupportActionsWithToolchain() throws Exception {
    useConfiguration("--xcode_toolchain=test_toolchain");
    checkRegisterSwiftSupportActions(RULE_TYPE, "iphonesimulator", "test_toolchain");
  }

  @Test
  public void testAddsStoryboardZipsToFilesToBuild() throws Exception {
    ConfiguredTarget target = createTargetWithStoryboards(RULE_TYPE);

    assertThat(getFilesToBuild(getConfiguredTarget("//x:x")))
        .containsAllOf(
            getBinArtifact("x/1.storyboard.zip", target),
            getBinArtifact("x/2.storyboard.zip", target));
  }

  @Test
  public void testAddsXcdatamodelZipsToFilesToBuild() throws Exception {
    RULE_TYPE.scratchTarget(scratch,
        "datamodels", "['modela.xcdatamodel/a', 'modelb.xcdatamodeld/modelb1/a']");
    ConfiguredTarget target = getConfiguredTarget("//x:x");
    assertThat(getFilesToBuild(target))
        .containsAllOf(
            getBinArtifact("x/modela.zip", target),
            getBinArtifact("x/modelb.zip", target));
  }

  @Test
  public void testHasDefaultInfoplistForXcTest() throws Exception {
    createBinaryTargetWriter("//bin:bin").setAndCreateFiles("srcs", "a.m").write();
    scratch.file("x/BUILD",
        "ios_application(",
        "    name = 'testApp',",
        "    binary = '//bin:bin',",
        ")",
        "ios_test(",
        "    name = 'x',",
        "    srcs = ['x-src.m'],",
        "    xctest = 1,",
        "    xctest_app = ':testApp',",
        ")");
    PlMergeProtos.Control control = plMergeControl("//x:x");
    assertThat(control.getSourceFileList())
        .contains(getSourceArtifact("tools/objc/xctest.plist").getExecPathString());
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
  public void testErrorsWrongFileTypeForSrcsWhenCompiling() throws Exception {
    checkErrorsWrongFileTypeForSrcsWhenCompiling(RULE_TYPE);
  }

  @Test
  public void testErrorForNoSources() throws Exception {
    createBinaryTargetWriter("//bin:bin").setAndCreateFiles("srcs", "a.m").write();
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    checkError("x", "x",
        IosTest.REQUIRES_SOURCE_ERROR,
        "ios_application(",
        "    name = 'testApp',",
        "    binary = '//bin:bin',",
        ")",
        "ios_test(",
        "    name = 'x',",
        "    xctest = 1,",
        "    xctest_app = ':testApp',",
        "    deps = ['//lib:lib'],",
        ")");
  }

  private void checkTestScript(Map<String, String> templateArguments, String... extraAttrs)
      throws Exception {
    createBinaryTargetWriter("//bin:bin").setAndCreateFiles("srcs", "a.m").write();
    scratch.file("x/BUILD",
        "ios_application(",
        "    name = 'testApp',",
        "    binary = '//bin:bin',",
        ")",
        "ios_test(",
        "    name = 'x',",
        "    srcs = ['a.m'],",
        "    xctest = 1,",
        "    xctest_app = ':testApp',",
        Joiner.on(",").join(extraAttrs),
        ")");
    TemplateExpansionAction action =
        (TemplateExpansionAction)
            getGeneratingAction(getBinArtifact("x_test_script", getConfiguredTarget("//x:x")));

    for (Map.Entry<String, String> templateArgument : templateArguments.entrySet()) {
      assertThat(action.getSubstitutions()).contains(
          Substitution.of("%(" + templateArgument.getKey() + ")s", templateArgument.getValue()));
    }
  }

  @Test
  public void testTargetDeviceUsesIosVersionAttributeIfGiven() throws Exception {
    scratch.file("devices/BUILD",
        "ios_device(",
        "    name = 'dev',",
        "    type = 'test_type',",
        "    ios_version = '42.9993',",
        ")");
    checkTestScript(
        ImmutableMap.of("simulator_sdk", "42.9993", "device_type", "test_type"),
        "target_device = '//devices:dev'");
  }

  @Test
  public void testObjcCopts() throws Exception {
    useConfiguration("--objccopt=-foo");

    scratch.file("x/a.m");
    addSimpleIosTest("bin", "bin", ImmutableList.of("a.m"), ImmutableList.<String>of());
    List<String> args = compileAction("//bin:bin", "a.o").getArguments();
    assertThat(args).contains("-foo");
  }

  @Test
  public void testObjcCopts_argumentOrdering() throws Exception {
    useConfiguration("--objccopt=-foo");

    scratch.file("x/a.m");
    addSimpleIosTest(
        "bin", "bin", ImmutableList.of("a.m"), ImmutableList.<String>of(), "copts=['-bar']");
    List<String> args = compileAction("//bin:bin", "a.o").getArguments();
    assertThat(args).containsAllOf("-fobjc-arc", "-foo", "-bar").inOrder();
  }

  @Test
  public void testGetsDefinesFromTestRig() throws Exception {
    scratch.file("x/BUILD",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['lib.m'],",
        "    defines = ['LIB_DEFINE=1'],",
        ")",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        "    defines = ['BIN_DEFINE=1'],",
        "    deps = [':lib'],",
        ")",
        "ios_application(",
        "    name = 'testApp',",
        "    binary = ':bin',",
        ")",
        "ios_test(",
        "    name = 'test',",
        "    srcs = ['test.m'],",
        "    defines = ['TEST_DEFINE=1'],",
        "    xctest = 1,",
        "    xctest_app = ':testApp',",
        ")");
    assertContainsSublist(compileAction("//x:test", "test.o").getArguments(),
        ImmutableList.of("-DLIB_DEFINE=1", "-DBIN_DEFINE=1", "-DTEST_DEFINE=1"));
  }

  @Test
  public void testGetsSdkDylibsFromTestRig() throws Exception {
    scratch.file("x/BUILD",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['lib.m'],",
        "    sdk_dylibs = ['lib_dylib'],",
        ")",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        "    sdk_dylibs = ['bin_dylib'],",
        "    deps = [':lib'],",
        ")",
        "ios_application(",
        "    name = 'testApp',",
        "    binary = ':bin',",
        ")",
        "ios_test(",
        "    name = 'test',",
        "    srcs = ['test.m'],",
        "    sdk_dylibs = ['test_dylib'],",
        "    xctest = 1,",
        "    xctest_app = ':testApp',",
        ")");

    String linkArgs = Joiner.on(' ').join(linkAction("//x:test").getArguments());
    assertThat(linkArgs).contains("-l_dylib");
    assertThat(linkArgs).contains("-lbin_dylib");
    assertThat(linkArgs).contains("-ltest_dylib");
  }

  @Test
  public void testGetsSdkFrameworksFromTestRig() throws Exception {
    scratch.file("x/BUILD",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['lib.m'],",
        "    sdk_frameworks = ['lib_fx'],",
        "    weak_sdk_frameworks = ['lib_wfx'],",
        ")",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        "    sdk_frameworks = ['bin_fx'],",
        "    weak_sdk_frameworks = ['bin_wfx'],",
        "    deps = [':lib'],",
        ")",
        "ios_application(",
        "    name = 'testApp',",
        "    binary = ':bin',",
        ")",
        "ios_test(",
        "    name = 'test',",
        "    srcs = ['test.m'],",
        "    sdk_frameworks = ['test_fx'],",
        "    weak_sdk_frameworks = ['test_wfx'],",
        "    xctest = 1,",
        "    xctest_app = ':testApp',",
        ")");

    String linkArgs = Joiner.on(' ').join(linkAction("//x:test").getArguments());
    assertThat(linkArgs).contains("-framework lib_fx");
    assertThat(linkArgs).contains("-weak_framework lib_wfx");
    assertThat(linkArgs).contains("-framework bin_fx");
    assertThat(linkArgs).contains("-weak_framework bin_wfx");
    assertThat(linkArgs).contains("-framework test_fx");
    assertThat(linkArgs).contains("-weak_framework test_wfx");
  }

  @Test
  public void testLinkIncludeOrder_staticLibsFirst() throws Exception {
    checkLinkIncludeOrderStaticLibsFirst(RULE_TYPE);
  }

  @Test
  public void testMergesActoolPartialInfoplist() throws Exception {
    checkMergesPartialInfoplists(RULE_TYPE);
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
  public void testActoolActionCorrectness() throws Exception {
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

  @Test
  public void testCppSourceCompilesWithCppFlags() throws Exception {
    checkCppSourceCompilesWithCppFlags(RULE_TYPE);
  }

  @Test
  public void testNoMultiCpu() throws Exception {
    useConfiguration("--ios_multi_cpus=i386", "--ios_cpu=armv7");

    checkError("x", "x",
        IosTest.NO_MULTI_CPUS_ERROR,
        "ios_test(",
        "    name = 'x',",
        "    srcs = ['a.m'],",
        ")");
  }

  @Test
  public void testXcTestAppFromSkylarkRule() throws Exception {
    scratch.file("examples/rule/BUILD",
        "exports_files(['test.ipa'])");
    scratch.file(
        "examples/rule/apple_rules.bzl",
        "def _skylark_xctest_app_impl(ctx):",
        "   artifact = list(ctx.attr.test_ipa.files)[0]",
        "   objc_provider = apple_common.new_objc_provider(define=depset(['TEST_DEFINE']))",
        "   xctest_app_provider = apple_common.new_xctest_app_provider(",
        "       bundle_loader=artifact, ipa=artifact, objc_provider=objc_provider)",
        "   return struct(",
        "      xctest_app=xctest_app_provider,",
        "   )",
        "skylark_xctest_app = rule(implementation = _skylark_xctest_app_impl,",
        "   attrs = {",
        "     'test_ipa': attr.label(",
        "       allow_single_file=True,",
        "       default=Label('//examples/rule:test.ipa')),",
        "   })");

    scratch.file(
        "examples/ios_test/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "load('/examples/rule/apple_rules', 'skylark_xctest_app')",
        "skylark_xctest_app(",
        "    name = 'my_xctest_app',",
        ")",
        "ios_test(",
        "  name = 'my_tests',",
        "  srcs = ['tests.m'],",
        "  xctest_app = ':my_xctest_app',",
        ")");

    ConfiguredTarget testTarget = getConfiguredTarget("//examples/ios_test:my_tests");

    assertThat(
            Artifact.toRootRelativePaths(
                testTarget.getProvider(FileProvider.class).getFilesToBuild()))
        .contains("examples/rule/test.ipa");
  }

  @Test
  public void testApplePlatformAndXcode() throws Exception {
    scratch.file("test/BUILD",
        "xcode_version(",
        "  name = 'test_xcode',",
        "  version = '2.3',",
        ")",
        "ios_device(",
        "  name = 'test_device',",
        "  ios_version = '3.4',",
        "  xcode = ':test_xcode',",
        "  type = 'iChimpanzee',",
        ")",
        "ios_test(",
        "  name = 'some_test',",
        "  srcs = ['SomeTest.m'],",
        "  xctest = 0,",
        "  target_device = ':test_device',",
        ")");

    scratch.file("test/SomeTest.m");

    ConfiguredTarget testTarget = getConfiguredTarget("//test:some_test");

    TestRunnerAction testAction = (TestRunnerAction) getGeneratingAction(
        Iterables.getOnlyElement(TestProvider.getTestStatusArtifacts(testTarget)));
    TemplateExpansionAction templateExpansionAction =
        getTestScriptGenerationAction(getConfiguredTarget("//test:some_test"));

    assertThat(templateExpansionAction.getSubstitutions())
        .contains(Substitution.of("%(simulator_sdk)s", "3.4"));
    assertThat(testAction.getExtraTestEnv()).containsEntry("XCODE_VERSION_OVERRIDE", "2.3");
  }

  @Test
  public void testAppleEnvironmentVariables_configurationXcodeVersion() throws Exception {
    useConfiguration("--xcode_version=5.8");

    scratch.file("test/BUILD",
        "ios_device(",
        "  name = 'test_device',",
        "  type = 'iChimpanzee',",
        ")",
        "ios_test(",
        "  name = 'some_test',",
        "  srcs = ['SomeTest.m'],",
        "  xctest = 0,",
        "  target_device = ':test_device',",
        ")");

    scratch.file("test/SomeTest.m");

    ConfiguredTarget testTarget = getConfiguredTarget("//test:some_test");

    TestRunnerAction testAction = (TestRunnerAction) getGeneratingAction(
        Iterables.getOnlyElement(TestProvider.getTestStatusArtifacts(testTarget)));
    TemplateExpansionAction templateExpansionAction =
        getTestScriptGenerationAction(getConfiguredTarget("//test:some_test"));

    assertThat(templateExpansionAction.getSubstitutions()).contains(
        Substitution.of("%(simulator_sdk)s", XcodeVersionProperties.DEFAULT_IOS_SDK_VERSION));
    assertThat(testAction.getExtraTestEnv())
        .containsEntry("XCODE_VERSION_OVERRIDE", "5.8");
  }

  @Test
  public void testRunnerSubstitution() throws Exception {
    addSimpleIosTest("test", "some_test", ImmutableList.of("a.m"), ImmutableList.<String>of());

    TemplateExpansionAction action =
        getTestScriptGenerationAction(getConfiguredTarget("//test:some_test"));
    assertThat(action.getSubstitutions()).containsExactly(
        Substitution.of("%(memleaks)s", "false"),

        Substitution.of("%(test_app_ipa)s", "test/some_test.ipa"),
        Substitution.of("%(test_app_name)s", "some_test"),
        Substitution.of("%(test_bundle_path)s", "test/some_test.ipa"),

        Substitution.of("%(xctest_app_ipa)s", "tools/objc/xctest_app.ipa"),
        Substitution.of("%(xctest_app_name)s", "xctest_app"),
        Substitution.of("%(test_host_path)s", "tools/objc/xctest_app.ipa"),

        Substitution.of("%(plugin_jars)s", ""),
        Substitution.of("%(device_type)s", "iChimpanzee"),
        Substitution.of("%(locale)s", "en"),
        Substitution.of("%(simulator_sdk)s", "9.8"),
        Substitution.of("%(testrunner_binary)s", "tools/objc/testrunner"),
        Substitution.of("%(std_redirect_dylib_path)s", "tools/objc/StdRedirect.dylib"),
        Substitution.of("%(test_env)s", ""),
        Substitution.of("%(test_type)s", "XCTEST")
    );
  }

  @Test
  public void testNonXcTestSubstitution() throws Exception {
    scratch.file("test/BUILD",
        "ios_test(",
        "  name = 'some_test',",
        "  srcs = ['SomeTest.m'],",
        "  xctest = 0,",
        ")");

    scratch.file("test/SomeTest.m");

    ConfiguredTarget target = getConfiguredTarget("//test:some_test");

    TemplateExpansionAction action =
        getTestScriptGenerationAction(target);
    assertThat(action.getSubstitutions()).containsExactly(
        Substitution.of("%(memleaks)s", "false"),

        Substitution.of("%(test_app_ipa)s", "test/some_test.ipa"),
        Substitution.of("%(test_bundle_path)s", "test/some_test.ipa"),
        Substitution.of("%(test_app_name)s", "some_test"),

        Substitution.of("%(xctest_app_ipa)s", ""),
        Substitution.of("%(xctest_app_name)s", ""),
        Substitution.of("%(test_host_path)s", ""),

        Substitution.of("%(plugin_jars)s", ""),
        Substitution.of("%(device_type)s", "iChimpanzee"),
        Substitution.of("%(locale)s", "en"),
        Substitution.of("%(simulator_sdk)s", "9.8"),
        Substitution.of("%(testrunner_binary)s", "tools/objc/testrunner"),
        Substitution.of("%(std_redirect_dylib_path)s", "tools/objc/StdRedirect.dylib"),
        Substitution.of("%(test_env)s", ""),
        Substitution.of("%(test_type)s", "KIF")
    );

    assertRunfilesContainsRootRelativePaths(target,
        "test/some_test.ipa",
        "test/some_test_test_script",
        "tools/objc/testrunner");
  }

  @Test
  public void testRunnerWithDevice() throws Exception {
    scratch.file("test/BUILD",
        "ios_test(",
        "  name = 'some_test_with_device',",
        "  srcs = ['SomeOtherTest.m'],",
        "  xctest = 1,",
        "  xctest_app = ':testApp',",
        "  target_device = ':device',",
        ")",
        "ios_device(",
        "  name = 'device',",
        "  ios_version = '1.2',",
        "  type = 'iMarmoset',",
        "  locale = 'en-gb'",
        ")",
        "ios_application(",
        "    name = 'testApp',",
        "    binary = ':bin',",
        ")",
        "objc_binary(name = 'bin',",
        "  srcs = ['app.m'],",
        ")");

    scratch.file("test/SomeOtherTest.m");
    scratch.file("test/app.m");

    TemplateExpansionAction action =
        getTestScriptGenerationAction(getConfiguredTarget("//test:some_test_with_device"));
    assertThat(action.getSubstitutions()).containsExactly(
        Substitution.of("%(memleaks)s", "false"),

        Substitution.of("%(test_app_ipa)s", "test/some_test_with_device.ipa"),
        Substitution.of("%(test_app_name)s", "some_test_with_device"),
        Substitution.of("%(test_bundle_path)s", "test/some_test_with_device.ipa"),

        Substitution.of("%(xctest_app_ipa)s", "test/testApp.ipa"),
        Substitution.of("%(xctest_app_name)s", "testApp"),
        Substitution.of("%(test_host_path)s", "test/testApp.ipa"),

        Substitution.of("%(plugin_jars)s", ""),
        Substitution.of("%(device_type)s", "iMarmoset"),
        Substitution.of("%(locale)s", "en-gb"),
        Substitution.of("%(simulator_sdk)s", "1.2"),
        Substitution.of("%(testrunner_binary)s", "tools/objc/testrunner"),
        Substitution.of("%(std_redirect_dylib_path)s", "tools/objc/StdRedirect.dylib"),
        Substitution.of("%(test_env)s", ""),
        Substitution.of("%(test_type)s", "XCTEST")
    );
  }

  @Test
  public void testPlugins() throws Exception {
    scratch.file("test/BUILD",
        "ios_test(",
        "  name = 'one_plugin',",
        "  srcs = ['SomeTest.m'],",
        "  xctest = 1,",
        "  xctest_app = ':testApp',",
        "  plugins = [':a_plugin_deploy.jar'],",
        ")",
        "ios_test(",
        "  name = 'two_plugins',",
        "  srcs = ['SomeOtherTest.m'],",
        "  xctest = 1,",
        "  xctest_app = ':testApp',",
        "  plugins = [':a_plugin_deploy.jar', ':b_plugin_deploy.jar'],",
        ")",
        "java_binary(",
        "  name = 'a_plugin',",
        "  srcs = ['A.java'],",
        "  main_class = 'A',",
        ")",
        "java_binary(",
        "  name = 'b_plugin',",
        "  srcs = ['B.java'],",
        "  main_class = 'B',",
        ")",
        "ios_application(",
        "    name = 'testApp',",
        "    binary = ':bin',",
        ")",
        "objc_binary(name = 'bin',",
        "  srcs = ['app.m'],",
        ")");

    scratch.file("test/SomeTest.m");
    scratch.file("test/SomeOtherTest.m");
    scratch.file("test/app.m");
    scratch.file("test/A.java");
    scratch.file("test/B.java");

    ConfiguredTarget onePluginTarget = getConfiguredTarget("//test:one_plugin");
    TemplateExpansionAction onePluginAction =
        getTestScriptGenerationAction(onePluginTarget);
    assertThat(onePluginAction.getSubstitutions()).contains(
        Substitution.of("%(plugin_jars)s", "test/a_plugin_deploy.jar"));
    assertRunfilesContainsRootRelativePaths(onePluginTarget, "test/a_plugin_deploy.jar");

    ConfiguredTarget twoPluginsTarget = getConfiguredTarget("//test:two_plugins");
    TemplateExpansionAction twoPluginsAction =
        getTestScriptGenerationAction(twoPluginsTarget);
    assertThat(twoPluginsAction.getSubstitutions()).contains(
        Substitution.of("%(plugin_jars)s", "test/a_plugin_deploy.jar:test/b_plugin_deploy.jar")
    );
    assertRunfilesContainsRootRelativePaths(twoPluginsTarget,
        "test/a_plugin_deploy.jar", "test/b_plugin_deploy.jar");
  }

  private TemplateExpansionAction getTestScriptGenerationAction(ConfiguredTarget target)
      throws Exception {
    Artifact testScript = getBinArtifact(target.getLabel().getName() + "_test_script", target);
    return (TemplateExpansionAction) getGeneratingAction(testScript);
  }

  private void assertRunfilesContainsRootRelativePaths(
      ConfiguredTarget target, String... expectedRunfiles) {
    RunfilesProvider runfiles = target.getProvider(RunfilesProvider.class);
    ImmutableList<String> listToAvoidDumbUnsafeVarargsWarning =
        ImmutableList.copyOf(expectedRunfiles);
    assertThat(Artifact.toRootRelativePaths(runfiles.getDefaultRunfiles().getArtifacts()))
        .containsAllIn(listToAvoidDumbUnsafeVarargsWarning);
    assertThat(Artifact.toRootRelativePaths(runfiles.getDataRunfiles().getArtifacts()))
        .containsAllIn(listToAvoidDumbUnsafeVarargsWarning);
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
  public void testLinkingRuleCanUseCrosstool() throws Exception {
    checkLinkingRuleCanUseCrosstool(RULE_TYPE);
  }

  @Test
  public void testBinaryStrippings() throws Exception {
    checkBinaryStripAction(RULE_TYPE, "-S");
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
  public void testProtobufPropagatedHeaderSearchPaths() throws Exception {
    scratch.file(
        "test/BUILD",
        "ios_test(",
        "  name = 'protos_test',",
        "  srcs = ['SomeTest.m'],",
        "  xctest = 1,",
        "  xctest_app = ':protos_app',",
        ")",
        "ios_application(",
        "    name = 'protos_app',",
        "    binary = ':protos_bin',",
        ")",
        "objc_binary(",
        "  name = 'protos_bin',",
        "  srcs = ['app.m'],",
        "  deps = [':protos_objc'],",
        ")",
        "objc_proto_library(",
        "  name = 'protos_objc',",
        "  deps = [':protos_lib'],",
        "  portable_proto_filters = ['filter.pbascii'],",
        ")",
        "proto_library(",
        "  name = 'protos_lib',",
        "  srcs = ['a.proto'],",
        ")");

    ObjcProvider appProvider =
        getConfiguredTarget("//test:protos_app")
            .get(XcTestAppProvider.SKYLARK_CONSTRUCTOR)
            .getObjcProvider();
    ConfiguredTarget binTarget = getConfiguredTarget("//test:protos_bin");
    Artifact protoHeader =
        getBinArtifact("_generated_protos/protos_bin/test/A.pbobjc.h", binTarget);

    assertThat(PathFragment.safePathStrings(appProvider.get(ObjcProvider.INCLUDE)))
        .containsAllOf(
            "objcproto/include",
            protoHeader.getExecPath().getParentDirectory().getParentDirectory().toString());
  }

  @Test
  public void testCcDependency() throws Exception {
    checkCcDependency(RULE_TYPE, "xctest", "0");
  }

  @Test
  public void testPassesTestRigAppAsBundleLoaderFlagToLinker() throws Exception {
    useConfiguration("--cpu=ios_x86_64",
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--experimental_disable_go");
    scratch.file("x/BUILD",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        ")",
        "ios_application(",
        "    name = 'testApp',",
        "    binary = ':bin',",
        ")",
        "ios_test(",
        "    name = 'test',",
        "    srcs = ['test.m'],",
        "    xctest = 1,",
        "    xctest_app = ':testApp',",
        ")");
    CommandAction testLinkAction = linkAction("//x:test");
    Action appLipoAction = lipoBinAction("//x:testApp");
    Artifact rigBinary = Iterables.getOnlyElement(appLipoAction.getOutputs());

    String linkArgs = Joiner.on(' ').join(testLinkAction.getArguments());
    assertThat(linkArgs).contains("-bundle_loader " + rigBinary.getExecPath());
    assertThat(testLinkAction.getInputs()).contains(rigBinary);
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
  public void testMultiArchUserHeaderSearchPathsUsed() throws Exception {
     // Usually, an ios_test would depend on apple_binary through a skylark_ios_application in its
     // 'binary' attribute.  Since we don't have skylark_ios_application here, we use the 'deps'
     // attribute instead.
     scratch.file("x/BUILD",
        "genrule(",
        "    name = 'gen_hdrs',",
        "    outs = ['generated.h'],",
        "    cmd = 'echo hello > \\$@',",
        ")",
        "apple_binary(",
        "    name = 'apple_bin',",
        "    srcs = ['apple_bin.m'],",
        "    platform_type = 'ios',",
        "    hdrs = ['generated.h'],",
        ")",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        ")",
        "ios_application(",
        "    name = 'testApp',",
        "    binary = ':bin',",
        ")",
        "ios_test(",
        "    name = 'test',",
        "    srcs = ['test.m'],",
        "    xctest = 1,",
        "    xctest_app = ':testApp',",
        "    deps = [':apple_bin']",
        ")");
     CommandAction compileAction = compileAction("//x:test", "test.o");
     // The genfiles root for child configurations must be present in the compile action so that
     // generated headers can be resolved.
     assertThat(Joiner.on(" ").join(compileAction.getArguments())).contains("-iquote "
         + configurationGenfiles("x86_64", ConfigurationDistinguisher.APPLEBIN_IOS,
             defaultMinimumOs(ConfigurationDistinguisher.APPLEBIN_IOS)));
  }

  @Test
  public void testXcTest_linkAction_inCoverageMode() throws Exception {
    useConfiguration("--collect_code_coverage");
    setUpXCTestClient();
    CommandAction action = linkAction("//test:XcTest");
    assertThat(Joiner.on(" ").join(action.getArguments())).contains("-bundle");
    for (String linkerCoverageFlag : CompilationSupport.LINKER_COVERAGE_FLAGS) {
      assertThat(Joiner.on(" ").join(action.getArguments())).contains(linkerCoverageFlag);
    }
  }

  @Test
  public void testXcTest_linkAction_inLLVMCoverageMode() throws Exception {
    useConfiguration("--collect_code_coverage", "--experimental_use_llvm_covmap");
    setUpXCTestClient();
    CommandAction action = linkAction("//test:XcTest");
    assertThat(Joiner.on(" ").join(action.getArguments())).contains("-bundle");
    for (String linkerCoverageFlag : CompilationSupport.LINKER_LLVM_COVERAGE_FLAGS) {
      assertThat(Joiner.on(" ").join(action.getArguments())).contains(linkerCoverageFlag);
    }
  }

  @Test
  public void testCompilesWithHdrs() throws Exception {
    checkCompilesWithHdrs(RULE_TYPE);
  }

  @Test
  public void testGetsHeadersFromTestRig() throws Exception {
    scratch.file(
        "x/BUILD",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['lib.m'],",
        "    hdrs = ['lib.h'],",
        ")",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        "    hdrs = ['bin.h'],",
        "    deps = [':lib'],",
        ")",
        "ios_application(",
        "    name = 'testApp',",
        "    binary = ':bin',",
        ")",
        "ios_test(",
        "    name = 'test',",
        "    srcs = ['test.m'],",
        "    hdrs = ['test.h'],",
        "    xctest = 1,",
        "    xctest_app = ':testApp',",
        ")");
    Iterable<Artifact> compileInputs =
        compileAction("//x:test", "test.o").getPossibleInputsForTesting();
    assertThat(Artifact.toExecPaths(compileInputs)).containsAllOf("x/lib.h", "x/bin.h", "x/test.h");
  }

  @Test
  public void testReceivesTransitivelyPropagatedDefines() throws Exception {
    checkReceivesTransitivelyPropagatedDefines(RULE_TYPE);
  }

  @Test
  public void testSdkIncludesUsedInCompileAction() throws Exception {
    checkSdkIncludesUsedInCompileAction(RULE_TYPE);
  }
}
