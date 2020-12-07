// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.baseArtifactNames;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.baseNamesOf;
import static org.junit.Assume.assumeTrue;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.truth.IterableSubject;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for {@link CcCommon}. */
@RunWith(JUnit4.class)
public class CcCommonTest extends BuildViewTestCase {

  private static final String STATIC_LIB = "statically/libstatically.a";

  @Before
  public final void createBuildFiles() throws Exception {
    // Having lots of setUp code leads to bad running time. Don't add anything here!
    scratch.file("empty/BUILD",
        "cc_library(name = 'emptylib')",
        "cc_binary(name = 'emptybinary')");

    scratch.file("foo/BUILD",
        "cc_library(name = 'foo',",
        "           srcs = ['foo.cc'])");

    scratch.file("bar/BUILD",
        "cc_library(name = 'bar',",
        "           srcs = ['bar.cc'])");
  }

  @Test
  public void testSameCcFileTwice() throws Exception {
    scratch.file(
        "a/BUILD",
        "cc_library(name='a', srcs=['a1', 'a2'])",
        "filegroup(name='a1', srcs=['a.cc'])",
        "filegroup(name='a2', srcs=['a.cc'])");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:a");
    assertContainsEvent("Artifact 'a/a.cc' is duplicated");
  }

  @Test
  public void testSameHeaderFileTwice() throws Exception {
    scratch.file(
        "a/BUILD",
        "package(features=['parse_headers'])",
        "cc_library(name='a', srcs=['a1', 'a2', 'a.cc'])",
        "filegroup(name='a1', srcs=['a.h'])",
        "filegroup(name='a2', srcs=['a.h'])");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:a");
    assertNoEvents();
  }

  @Test
  public void testEmptyLibrary() throws Exception {
    ConfiguredTarget emptylib = getConfiguredTarget("//empty:emptylib");
    // We create .a for empty libraries, for simplicity (in Blaze).
    // But we avoid creating .so files for empty libraries,
    // because those have a potentially significant run-time startup cost.
    if (emptyShouldOutputStaticLibrary()) {
      assertThat(baseNamesOf(getFilesToBuild(emptylib))).isEqualTo("libemptylib.a");
    } else {
      assertThat(getFilesToBuild(emptylib).toList()).isEmpty();
    }
    assertThat(
            emptylib
                .get(CcInfo.PROVIDER)
                .getCcLinkingContext()
                .getDynamicLibrariesForRuntime(/* linkingStatically= */ false)
                .isEmpty())
        .isTrue();
  }

  protected boolean emptyShouldOutputStaticLibrary() {
    return !getAnalysisMock().isThisBazel();
  }

  @Test
  public void testEmptyBinary() throws Exception {
    ConfiguredTarget emptybin = getConfiguredTarget("//empty:emptybinary");
    assertThat(baseNamesOf(getFilesToBuild(emptybin)))
        .isEqualTo("emptybinary");
  }

  private List<String> getCopts(String target) throws Exception {
    ConfiguredTarget cLib = getConfiguredTarget(target);
    Artifact object = getOutputGroup(cLib, OutputGroupInfo.FILES_TO_COMPILE).getSingleton();
    CppCompileAction compileAction = (CppCompileAction) getGeneratingAction(object);
    return compileAction.getCompilerOptions();
  }

  @Test
  public void testCopts() throws Exception {
    scratch.file(
        "copts/BUILD",
        "cc_library(name = 'c_lib',",
        "    srcs = ['foo.cc'],",
        "    copts = [ '-Wmy-warning', '-frun-faster' ])");
    assertThat(getCopts("//copts:c_lib")).containsAtLeast("-Wmy-warning", "-frun-faster");
  }

  @Test
  public void testCoptsTokenization() throws Exception {
    scratch.file(
        "copts/BUILD",
        "cc_library(name = 'c_lib',",
        "    srcs = ['foo.cc'],",
        "    copts = ['-Wmy-warning -frun-faster'])");
    List<String> copts = getCopts("//copts:c_lib");
    assertThat(copts).containsAtLeast("-Wmy-warning", "-frun-faster");
  }

  @Test
  public void testCoptsNoTokenization() throws Exception {
    scratch.file(
        "copts/BUILD",
        "package(features = ['no_copts_tokenization'])",
        "cc_library(name = 'c_lib',",
        "    srcs = ['foo.cc'],",
        "    copts = ['-Wmy-warning -frun-faster'])");
    List<String> copts = getCopts("//copts:c_lib");
    assertThat(copts).contains("-Wmy-warning -frun-faster");
  }

  /**
   * Test that we handle ".a" files in cc_library srcs correctly when linking dynamically. In
   * particular, if srcs contains only the ".a" file for a library, with no corresponding ".so",
   * then we need to link in the ".a" file even when we're linking dynamically. If srcs contains
   * both ".a" and ".so" then we should only link in the ".so".
   */
  @Test
  public void testArchiveInCcLibrarySrcs() throws Exception {
    useConfiguration("--cpu=k8");
    ConfiguredTarget archiveInSrcsTest =
        scratchConfiguredTarget(
            "archive_in_srcs",
            "archive_in_srcs_test",
            "cc_test(name = 'archive_in_srcs_test',",
            "           srcs = ['archive_in_srcs_test.cc'],",
            "           deps = [':archive_in_srcs_lib'],",
            "           linkstatic = 0,)",
            "cc_library(name = 'archive_in_srcs_lib',",
            "           srcs = ['libstatic.a', 'libboth.a', 'libboth.so'])");
    List<String> artifactNames = baseArtifactNames(getLinkerInputs(archiveInSrcsTest));
    assertThat(artifactNames).containsAtLeast("libboth.so", "libstatic.a");
    assertThat(artifactNames).doesNotContain("libboth.a");
  }

  private Iterable<Artifact> getLinkerInputs(ConfiguredTarget target) {
    Artifact executable = getExecutable(target);
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(executable);
    return linkAction.getLinkCommandLine().getLinkerInputArtifacts().toList();
  }

  @Test
  public void testDylibLibrarySuffixIsStripped() throws Exception {
    ConfiguredTarget archiveInSrcsTest =
        scratchConfiguredTarget(
            "archive_in_src_darwin",
            "archive_in_srcs",
            "cc_binary(name = 'archive_in_srcs',",
            "    srcs = ['libarchive.34.dylib'])");

    Artifact executable = getExecutable(archiveInSrcsTest);
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(executable);
    assertThat(linkAction.getLinkCommandLine().toString()).contains(" -larchive.34 ");
  }

  @Test
  public void testLinkStaticStatically() throws Exception {
    ConfiguredTarget statically =
        scratchConfiguredTarget(
            "statically",
            "statically",
            "cc_library(name = 'statically',",
            "           srcs = ['statically.cc'],",
            "           linkstatic=1)");
    assertThat(
            statically
                .get(CcInfo.PROVIDER)
                .getCcLinkingContext()
                .getDynamicLibrariesForRuntime(/* linkingStatically= */ false)
                .isEmpty())
        .isTrue();
    Artifact staticallyDotA = getFilesToBuild(statically).getSingleton();
    assertThat(getGeneratingAction(staticallyDotA)).isInstanceOf(CppLinkAction.class);
    PathFragment dotAPath = staticallyDotA.getExecPath();
    assertThat(dotAPath.getPathString()).endsWith(STATIC_LIB);
  }

  @Test
  public void testIsolatedDefines() throws Exception {
    ConfiguredTarget isolatedDefines =
        scratchConfiguredTarget(
            "isolated_defines",
            "defineslib",
            "cc_library(name = 'defineslib',",
            "           srcs = ['defines.cc'],",
            "           defines = ['FOO', 'BAR'])");
    assertThat(isolatedDefines.get(CcInfo.PROVIDER).getCcCompilationContext().getDefines().toList())
        .containsExactly("FOO", "BAR")
        .inOrder();
  }

  @Test
  public void testStartEndLib() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_START_END_LIB));
    useConfiguration(
        // Prevent Android from trying to setup ARM crosstool by forcing it on system cpu.
        "--fat_apk_cpu=k8", "--start_end_lib");
    scratch.file(
        "test/BUILD",
        "cc_library(name='lib',",
        "           srcs=['lib.c'])",
        "cc_binary(name='bin',",
        "          srcs=['bin.c'])");

    ConfiguredTarget target = getConfiguredTarget("//test:bin");
    CppLinkAction action = (CppLinkAction) getGeneratingAction(getExecutable(target));
    for (Artifact input : action.getInputs().toList()) {
      String name = input.getFilename();
      assertThat(!CppFileTypes.ARCHIVE.matches(name) && !CppFileTypes.PIC_ARCHIVE.matches(name))
          .isTrue();
    }
  }

  @Test
  public void testStartEndLibThroughFeature() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_START_END_LIB));
    useConfiguration("--start_end_lib");
    scratch.file(
        "test/BUILD",
        "cc_library(name='lib', srcs=['lib.c'])",
        "cc_binary(name='bin', srcs=['bin.c'])");

    ConfiguredTarget target = getConfiguredTarget("//test:bin");
    CppLinkAction action = (CppLinkAction) getGeneratingAction(getExecutable(target));
    for (Artifact input : action.getInputs().toList()) {
      String name = input.getFilename();
      assertThat(!CppFileTypes.ARCHIVE.matches(name) && !CppFileTypes.PIC_ARCHIVE.matches(name))
          .isTrue();
    }
  }

  @Test
  public void testTempsWithDifferentExtensions() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_PIC));
    invalidatePackages();
    useConfiguration("--cpu=k8", "--save_temps");
    scratch.file(
        "ananas/BUILD",
        "cc_library(name='ananas',",
        "           srcs=['1.c', '2.cc', '3.cpp', '4.S', '5.h', '6.hpp', '7.inc', '8.inl',",
        "                 '9.tlh', 'A.tli'])");

    ConfiguredTarget ananas = getConfiguredTarget("//ananas:ananas");
    Iterable<String> temps =
        ActionsTestUtil.baseArtifactNames(getOutputGroup(ananas, OutputGroupInfo.TEMP_FILES));
    assertThat(temps)
        .containsExactly(
            "1.pic.i", "1.pic.s",
            "2.pic.ii", "2.pic.s",
            "3.pic.ii", "3.pic.s");
  }

  /**
   * Returns the {@link IterableSubject} for the {@link OutputGroupInfo#TEMP_FILES} generated when
   * {@code testTarget} is built for {@code cpu}.
   */
  private IterableSubject assertTempsForTarget(String testTarget) throws Exception {
    useConfiguration("--save_temps");
    ConfiguredTarget target = getConfiguredTarget(testTarget);
    assertThat(target).isNotNull();

    List<String> temps =
        ActionsTestUtil.baseArtifactNames(getOutputGroup(target, OutputGroupInfo.TEMP_FILES));

    // Return the IterableSubject for the temp files.
    return assertWithMessage("k8").that(temps);
  }

  @Test
  public void testTempsForCcWithPic() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_PIC));
    invalidatePackages();
    assertTempsForTarget("//foo:foo").containsExactly("foo.pic.ii", "foo.pic.s");
  }

  @Test
  public void testTempsForCcWithoutPic() throws Exception {
    assertTempsForTarget("//foo:foo").containsExactly("foo.ii", "foo.s");
  }

  @Test
  public void testTempsForCWithPic() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_PIC));
    invalidatePackages();
    useConfiguration();

    scratch.file("csrc/BUILD", "cc_library(name='csrc', srcs=['foo.c'])");
    assertTempsForTarget("//csrc:csrc").containsExactly("foo.pic.i", "foo.pic.s");
  }

  @Test
  public void testTempsForCWithoutPic() throws Exception {
    scratch.file("csrc/BUILD", "cc_library(name='csrc', srcs=['foo.c'])");
    assertTempsForTarget("//csrc:csrc").containsExactly("foo.i", "foo.s");
  }

  @Test
  public void testAlwaysLinkYieldsLo() throws Exception {
    ConfiguredTarget alwaysLink =
        scratchConfiguredTarget(
            "always_link",
            "always_link",
            "cc_library(name = 'always_link',",
            "           alwayslink = 1,",
            "           srcs = ['always_link.cc'])");
    assertThat(baseNamesOf(getFilesToBuild(alwaysLink))).contains("libalways_link.lo");
  }

  @Test
  public void testPicModeAssembly() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(CppRuleClasses.SUPPORTS_PIC, CppRuleClasses.PIC));
    invalidatePackages();
    useConfiguration("--cpu=k8");
    scratch.file("a/BUILD", "cc_library(name='preprocess', srcs=['preprocess.S'])");
    List<String> argv = getCppCompileAction("//a:preprocess").getArguments();
    assertThat(argv).contains("-fPIC");
  }

  private CppCompileAction getCppCompileAction(String label) throws Exception {
    ConfiguredTarget target = getConfiguredTarget(label);
    List<CppCompileAction> compilationSteps =
        actionsTestUtil()
            .findTransitivePrerequisitesOf(
                getFilesToBuild(target).toList().get(0), CppCompileAction.class);
    return compilationSteps.get(0);
  }

  @Test
  public void testIsolatedIncludes() throws Exception {
    // Tests the (immediate) effect of declaring the includes attribute on a
    // cc_library.

    scratch.file(
        "bang/BUILD",
        "cc_library(name = 'bang',",
        "           srcs = ['bang.cc'],",
        "           includes = ['bang_includes'])");

    ConfiguredTarget foo = getConfiguredTarget("//bang:bang");

    String includesRoot = "bang/bang_includes";
    assertThat(foo.get(CcInfo.PROVIDER).getCcCompilationContext().getSystemIncludeDirs())
        .containsAtLeast(
            PathFragment.create(includesRoot),
            targetConfig.getGenfilesFragment(RepositoryName.MAIN).getRelative(includesRoot));
  }

  @Test
  public void testDisabledGenfilesDontShowUpInSystemIncludePaths() throws Exception {
    scratch.file(
        "bang/BUILD",
        "cc_library(name = 'bang',",
        "           srcs = ['bang.cc'],",
        "           includes = ['bang_includes'])");
    String includesRoot = "bang/bang_includes";

    useConfiguration("--noincompatible_merge_genfiles_directory");
    ConfiguredTarget foo = getConfiguredTarget("//bang:bang");
    PathFragment genfilesDir =
        targetConfig.getGenfilesFragment(RepositoryName.MAIN).getRelative(includesRoot);
    assertThat(foo.get(CcInfo.PROVIDER).getCcCompilationContext().getSystemIncludeDirs())
        .contains(genfilesDir);

    useConfiguration("--incompatible_merge_genfiles_directory");
    foo = getConfiguredTarget("//bang:bang");
    assertThat(foo.get(CcInfo.PROVIDER).getCcCompilationContext().getSystemIncludeDirs())
        .doesNotContain(genfilesDir);
  }

  @Test
  public void testUseIsystemForIncludes() throws Exception {
    // Tests the effect of --use_isystem_for_includes.
    useConfiguration("--incompatible_merge_genfiles_directory=false");
    scratch.file(
        "no_includes/BUILD",
        "cc_library(name = 'no_includes',",
        "           srcs = ['no_includes.cc'])");
    ConfiguredTarget noIncludes = getConfiguredTarget("//no_includes:no_includes");

    scratch.file(
        "bang/BUILD",
        "cc_library(name = 'bang',",
        "           srcs = ['bang.cc'],",
        "           includes = ['bang_includes'])");

    ConfiguredTarget foo = getConfiguredTarget("//bang:bang");

    String includesRoot = "bang/bang_includes";
    List<PathFragment> expected =
        new ImmutableList.Builder<PathFragment>()
            .addAll(
                noIncludes.get(CcInfo.PROVIDER).getCcCompilationContext().getSystemIncludeDirs())
            .add(PathFragment.create(includesRoot))
            .add(targetConfig.getGenfilesFragment(RepositoryName.MAIN).getRelative(includesRoot))
            .add(targetConfig.getBinFragment(RepositoryName.MAIN).getRelative(includesRoot))
            .build();
    assertThat(foo.get(CcInfo.PROVIDER).getCcCompilationContext().getSystemIncludeDirs())
        .containsExactlyElementsIn(expected);
  }

  @Test
  public void testCcTestDisallowsAlwaysLink() throws Exception {
    scratch.file(
        "cc/common/BUILD",
        "cc_library(name = 'lib1',",
        "           srcs = ['foo1.cc'],",
        "           deps = ['//left'])",
        "",
        "cc_test(name = 'testlib',",
        "       deps = [':lib1'],",
        "       alwayslink=1)");
    reporter.removeHandler(failFastHandler);
    getPackageManager().getPackage(reporter, PackageIdentifier.createInMainRepo("cc/common"));
    assertContainsEvent(
        "//cc/common:testlib: no such attribute 'alwayslink'" + " in 'cc_test' rule");
  }

  @Test
  public void testCcTestBuiltWithFissionHasDwp() throws Exception {
    // Tests that cc_tests built statically and with Fission will have the .dwp file
    // in their runfiles.
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.PER_OBJECT_DEBUG_INFO));
    useConfiguration("--cpu=k8", "--build_test_dwp", "--dynamic_mode=off", "--fission=yes");
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "mypackage", "mytest", "cc_test(name = 'mytest', srcs = ['mytest.cc'])");

    NestedSet<Artifact> runfiles = collectRunfiles(target);
    assertThat(baseArtifactNames(runfiles)).contains("mytest.dwp");
  }

  @Test
  public void testCcLibraryBadIncludesWarnedAndIgnored() throws Exception {
    checkWarning(
        "badincludes",
        "flaky_lib",
        // message:
        "in includes attribute of cc_library rule //badincludes:flaky_lib: "
            + "ignoring invalid absolute path '//third_party/procps/proc'",
        // build file:
        "cc_library(name = 'flaky_lib',",
        "   srcs = [ 'ok.cc' ],",
        "   includes = [ '//third_party/procps/proc' ])");
  }

  @Test
  public void testCcLibraryUplevelIncludesWarned() throws Exception {
    checkWarning(
        "third_party/uplevel",
        "lib",
        // message:
        "in includes attribute of cc_library rule //third_party/uplevel:lib: '../bar' resolves to "
            + "'third_party/bar' not below the relative path of its package 'third_party/uplevel'. "
            + "This will be an error in the future",
        // build file:
        "licenses(['unencumbered'])",
        "cc_library(name = 'lib',",
        "           srcs = ['foo.cc'],",
        "           includes = ['../bar'])");
  }

  @Test
  public void testCcLibraryNonThirdPartyIncludesWarned() throws Exception {
    if (getAnalysisMock().isThisBazel()) {
      return;
    }

    checkWarning(
        "topdir",
        "lib",
        // message:
        "in includes attribute of cc_library rule //topdir:lib: './' resolves to 'topdir' not "
            + "in 'third_party'. This will be an error in the future",
        // build file:
        "cc_library(name = 'lib',",
        "           srcs = ['foo.cc'],",
        "           includes = ['./'])");
  }

  @Test
  public void testCcLibraryThirdPartyIncludesNotWarned() throws Exception {
    eventCollector.clear();
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "third_party/pkg",
            "lib",
            "licenses(['unencumbered'])",
            "cc_library(name = 'lib',",
            "           srcs = ['foo.cc'],",
            "           includes = ['./'])");
    assertThat(view.hasErrors(target)).isFalse();
    assertNoEvents();
  }

  @Test
  public void testCcLibraryExternalIncludesNotWarned() throws Exception {
    eventCollector.clear();
    FileSystemUtils.appendIsoLatin1(
        scratch.resolve("WORKSPACE"),
        "local_repository(",
        "    name = 'pkg',",
        "    path = '/foo')");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            new ModifiedFileSet.Builder().modify(PathFragment.create("WORKSPACE")).build(),
            Root.fromPath(rootDirectory));
    FileSystemUtils.createDirectoryAndParents(scratch.resolve("/foo/bar"));
    scratch.file("/foo/WORKSPACE", "workspace(name = 'pkg')");
    scratch.file(
        "/foo/bar/BUILD",
        "cc_library(name = 'lib',",
        "           srcs = ['foo.cc'],",
        "           includes = ['./'])");
    Label label = Label.parseAbsolute("@pkg//bar:lib", ImmutableMap.of());
    ConfiguredTarget target = view.getConfiguredTargetForTesting(reporter, label, targetConfig);
    assertThat(view.hasErrors(target)).isFalse();
    assertNoEvents();
  }

  @Test
  public void testCcLibraryRootIncludesError() throws Exception {
    checkError(
        "third_party/root",
        "lib",
        // message:
        "in includes attribute of cc_library rule //third_party/root:lib: '../..' resolves to the "
            + "workspace root, which would allow this rule and all of its transitive dependents to "
            + "include any file in your workspace. Please include only what you need",
        // build file:
        "licenses(['unencumbered'])",
        "cc_library(name = 'lib',",
        "           srcs = ['foo.cc'],",
        "           includes = ['../..'])");
  }

  @Test
  public void testStaticallyLinkedBinaryNeedsSharedObject() throws Exception {
    scratch.file(
        "third_party/sophos_av_pua/BUILD",
        "licenses(['notice'])",
        "cc_library(name = 'savi',",
        "           srcs = [ 'lib/libsavi.so' ])");
    ConfiguredTarget wrapsophos =
        scratchConfiguredTarget(
            "quality/malware/support",
            "wrapsophos",
            "cc_library(name = 'sophosengine',",
            "           srcs = [ 'sophosengine.cc' ],",
            "           deps = [ '//third_party/sophos_av_pua:savi' ])",
            "cc_binary(name = 'wrapsophos',",
            "          srcs = [ 'wrapsophos.cc' ],",
            "          deps = [ ':sophosengine' ],",
            "          linkstatic=1)");

    List<String> artifactNames = baseArtifactNames(getLinkerInputs(wrapsophos));
    assertThat(artifactNames).contains("libsavi.so");
  }

  @Test
  public void testExpandLabelInLinkoptsAgainstSrc() throws Exception {
    scratch.file(
        "coolthing/BUILD",
        "genrule(name = 'build-that',",
        "  srcs = [ 'foo' ],",
        "  outs = [ 'nicelib.a' ],",
        "  cmd = 'cat  $< > $@')");
    // In reality the linkopts might contain several externally-provided
    // '.a' files with cyclic dependencies amongst them, but in this test
    // it suffices to show that one label in linkopts was resolved.
    scratch.file(
        "myapp/BUILD",
        "cc_binary(name = 'myapp',",
        "    srcs = [ '//coolthing:nicelib.a' ],",
        "    linkopts = [ '//coolthing:nicelib.a' ])");
    ConfiguredTarget theLib = getConfiguredTarget("//coolthing:build-that");
    ConfiguredTarget theApp = getConfiguredTarget("//myapp:myapp");
    // make sure we did not print warnings about the linkopt
    assertNoEvents();
    // make sure the binary is dependent on the static lib
    Action linkAction = getGeneratingAction(getFilesToBuild(theApp).getSingleton());
    ImmutableList<Artifact> filesToBuild = getFilesToBuild(theLib).toList();
    assertThat(linkAction.getInputs().toSet()).containsAtLeastElementsIn(filesToBuild);
  }

  @Test
  public void testCcLibraryWithDashStatic() throws Exception {
    assumeTrue(OS.getCurrent() != OS.DARWIN);
    checkWarning(
        "badlib",
        "lib_with_dash_static",
        // message:
        "in linkopts attribute of cc_library rule //badlib:lib_with_dash_static: "
            + "Using '-static' here won't work. Did you mean to use 'linkstatic=1' instead?",
        // build file:
        "cc_library(name = 'lib_with_dash_static',",
        "   srcs = [ 'ok.cc' ],",
        "   linkopts = [ '-static' ])");
  }

  @Test
  public void testCcLibraryWithDashStaticOnDarwin() throws Exception {
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "darwin");
    useConfiguration("--cpu=darwin");
    checkError(
        "badlib",
        "lib_with_dash_static",
        // message:
        "in linkopts attribute of cc_library rule //badlib:lib_with_dash_static: "
            + "Apple builds do not support statically linked binaries",
        // build file:
        "cc_library(name = 'lib_with_dash_static',",
        "   srcs = [ 'ok.cc' ],",
        "   linkopts = [ '-static' ])");
  }

  @Test
  public void testStampTests() throws Exception {
    scratch.file(
        "test/BUILD",
        "cc_test(name ='a', srcs = ['a.cc'])",
        "cc_test(name ='b', srcs = ['b.cc'], stamp = 0)",
        "cc_test(name ='c', srcs = ['c.cc'], stamp = 1)",
        "cc_binary(name ='d', srcs = ['d.cc'])",
        "cc_binary(name ='e', srcs = ['e.cc'], stamp = 0)",
        "cc_binary(name ='f', srcs = ['f.cc'], stamp = 1)");

    assertStamping(false, "//test:a");
    assertStamping(false, "//test:b");
    assertStamping(true, "//test:c");
    assertStamping(true, "//test:d");
    assertStamping(false, "//test:e");
    assertStamping(true, "//test:f");

    useConfiguration("--stamp");
    assertStamping(false, "//test:a");
    assertStamping(false, "//test:b");
    assertStamping(true, "//test:c");
    assertStamping(true, "//test:d");
    assertStamping(false, "//test:e");
    assertStamping(true, "//test:f");

    useConfiguration("--nostamp");
    assertStamping(false, "//test:a");
    assertStamping(false, "//test:b");
    assertStamping(true, "//test:c");
    assertStamping(false, "//test:d");
    assertStamping(false, "//test:e");
    assertStamping(true, "//test:f");
  }

  private void assertStamping(boolean enabled, String label) throws Exception {
    assertThat(AnalysisUtils.isStampingEnabled(getRuleContext(getConfiguredTarget(label))))
        .isEqualTo(enabled);
  }

  @Test
  public void testIncludeRelativeHeadersAboveExecRoot() throws Exception {
    checkError(
        "test",
        "bad_relative_include",
        "Path references a path above the execution root.",
        "cc_library(name='bad_relative_include', srcs=[], includes=['../..'])");
  }

  @Test
  public void testIncludeAbsoluteHeaders() throws Exception {
    checkWarning(
        "test",
        "bad_absolute_include",
        "ignoring invalid absolute path",
        "cc_library(name='bad_absolute_include', srcs=[], includes=['/usr/include/'])");
  }

  /** Tests that shared libraries of the form "libfoo.so.1.2" are permitted within "srcs". */
  @Test
  public void testVersionedSharedLibrarySupport() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "mypackage",
            "mybinary",
            "cc_binary(name = 'mybinary',",
            "           srcs = ['mybinary.cc'],",
            "           deps = [':mylib'])",
            "cc_library(name = 'mylib',",
            "           srcs = ['libshared.so', 'libshared.so.1.1', 'foo.cc'])");
    List<String> artifactNames = baseArtifactNames(getLinkerInputs(target));
    assertThat(artifactNames).containsAtLeast("libshared.so", "libshared.so.1.1");
  }

  @Test
  public void testNoHeaderInHdrsWarning() throws Exception {
    checkWarning(
        "hdrs_filetypes",
        "foo",
        "in hdrs attribute of cc_library rule //hdrs_filetypes:foo: file 'foo.a' "
            + "from target '//hdrs_filetypes:foo.a' is not allowed in hdrs",
        "cc_library(name = 'foo',",
        "    srcs = [],",
        "    hdrs = ['foo.a'])");
  }

  @Test
  public void testLibraryInHdrs() throws Exception {
    scratchConfiguredTarget("a", "a",
        "cc_library(name='a', srcs=['a.cc'], hdrs=[':b'])",
        "cc_library(name='b', srcs=['b.cc'])");
  }

  @Test
  public void testExpandedLinkopts() throws Exception {
    scratch.file(
        "a/BUILD",
        "genrule(name = 'linker', cmd='generate', outs=['a.lds'])",
        "cc_binary(",
        "    name='bin',",
        "    srcs=['b.cc'],",
        "    linkopts=['-Wl,@$(location a.lds)'],",
        "    deps=['a.lds'])");
    ConfiguredTarget target = getConfiguredTarget("//a:bin");
    CppLinkAction action =
        (CppLinkAction) getGeneratingAction(getFilesToBuild(target).getSingleton());
    assertThat(MockCcSupport.getLinkopts(action.getLinkCommandLine()))
        .containsExactly(
            String.format(
                "-Wl,@%s/a/a.lds",
                getTargetConfiguration()
                    .getGenfilesDirectory(RepositoryName.MAIN)
                    .getExecPath()
                    .getPathString()));
  }

  @Test
  public void testProvidesLinkerScriptToLinkAction() throws Exception {
    scratch.file(
        "a/BUILD",
        "cc_binary(",
        "    name='bin',",
        "    srcs=['b.cc'],",
        "    linkopts=['-Wl,@$(location a.lds)'],",
        "    deps=['a.lds'])");
    ConfiguredTarget target = getConfiguredTarget("//a:bin");
    CppLinkAction action =
        (CppLinkAction) getGeneratingAction(getFilesToBuild(target).getSingleton());
    NestedSet<Artifact> linkInputs = action.getInputs();
    assertThat(ActionsTestUtil.baseArtifactNames(linkInputs)).contains("a.lds");
  }

  @Test
  public void testIncludeManglingSmoke() throws Exception {
    scratch.file(
        "third_party/a/BUILD",
        "licenses(['notice'])",
        "cc_library(name='a', hdrs=['v1/b/c.h'], strip_include_prefix='v1', include_prefix='lib')");

    ConfiguredTarget lib = getConfiguredTarget("//third_party/a");
    CcCompilationContext ccCompilationContext = lib.get(CcInfo.PROVIDER).getCcCompilationContext();
    assertThat(ActionsTestUtil.prettyArtifactNames(ccCompilationContext.getDeclaredIncludeSrcs()))
        .containsExactly("third_party/a/_virtual_includes/a/lib/b/c.h", "third_party/a/v1/b/c.h");
    assertThat(ccCompilationContext.getIncludeDirs())
        .containsExactly(
            getTargetConfiguration()
                .getBinFragment(RepositoryName.MAIN)
                .getRelative("third_party/a/_virtual_includes/a"));
  }

  @Test
  public void testUpLevelReferencesInIncludeMangling() throws Exception {
    scratch.file(
        "third_party/a/BUILD",
        "licenses(['notice'])",
        "cc_library(name='sip', srcs=['a.h'], strip_include_prefix='a/../b')",
        "cc_library(name='ip', srcs=['a.h'], include_prefix='a/../b')",
        "cc_library(name='ipa', srcs=['a.h'], include_prefix='/foo')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//third_party/a:sip");
    assertContainsEvent("should not contain uplevel references");

    eventCollector.clear();
    getConfiguredTarget("//third_party/a:ip");
    assertContainsEvent("should not contain uplevel references");

    eventCollector.clear();
    getConfiguredTarget("//third_party/a:ipa");
    assertContainsEvent("should be a relative path");
  }

  @Test
  public void testAbsoluteAndRelativeStripPrefix() throws Exception {
    scratch.file("third_party/a/BUILD",
        "licenses(['notice'])",
        "cc_library(name='relative', hdrs=['v1/b.h'], strip_include_prefix='v1')",
        "cc_library(name='absolute', hdrs=['v1/b.h'], strip_include_prefix='/third_party')");

    CcCompilationContext relative =
        getConfiguredTarget("//third_party/a:relative")
            .get(CcInfo.PROVIDER)
            .getCcCompilationContext();
    CcCompilationContext absolute =
        getConfiguredTarget("//third_party/a:absolute")
            .get(CcInfo.PROVIDER)
            .getCcCompilationContext();

    assertThat(ActionsTestUtil.prettyArtifactNames(relative.getDeclaredIncludeSrcs()))
        .containsExactly("third_party/a/_virtual_includes/relative/b.h", "third_party/a/v1/b.h");
    assertThat(ActionsTestUtil.prettyArtifactNames(absolute.getDeclaredIncludeSrcs()))
        .containsExactly(
            "third_party/a/_virtual_includes/absolute/a/v1/b.h", "third_party/a/v1/b.h");
  }

  @Test
  public void testArtifactNotUnderStripPrefix() throws Exception {
    scratch.file("third_party/a/BUILD",
        "licenses(['notice'])",
        "cc_library(name='a', hdrs=['v1/b.h'], strip_include_prefix='v2')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//third_party/a:a");
    assertContainsEvent(
        "header 'third_party/a/v1/b.h' is not under the specified strip prefix 'third_party/a/v2'");
  }

  @Test
  public void testSymlinkActionIsNotRegisteredWhenIncludePrefixDoesntChangePath() throws Exception {
    scratch.file(
        "third_party/BUILD",
        "licenses(['notice'])",
        "cc_library(name='a', hdrs=['a.h'], include_prefix='third_party')");

    CcCompilationContext ccCompilationContext =
        getConfiguredTarget("//third_party:a").get(CcInfo.PROVIDER).getCcCompilationContext();
    assertThat(ActionsTestUtil.prettyArtifactNames(ccCompilationContext.getDeclaredIncludeSrcs()))
        .doesNotContain("third_party/_virtual_includes/a/third_party/a.h");
  }

  @Test
  public void
  testConfigureFeaturesDoesntCrashOnCollidingFeaturesExceptionButReportsRuleErrorCleanly()
      throws Exception {
    reporter.removeHandler(failFastHandler);
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures("same_symbol_provided_configuration"));
    useConfiguration("--features=a1", "--features=a2");

    scratch.file("x/BUILD", "cc_library(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    getConfiguredTarget("//x:foo");
    assertContainsEvent("Symbol a is provided by all of the following features: a1 a2");
  }

  @Test
  public void testSupportsPicFeatureResultsInPICObjectGenerated() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(CppRuleClasses.NO_LEGACY_FEATURES, CppRuleClasses.SUPPORTS_PIC)
                .withActionConfigs(
                    CppActionNames.CPP_LINK_STATIC_LIBRARY,
                    CppActionNames.CPP_COMPILE,
                    CppActionNames.CPP_LINK_NODEPS_DYNAMIC_LIBRARY));
    useConfiguration("--cpu=k8");

    scratch.file("x/BUILD", "cc_library(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    RuleConfiguredTarget ccLibrary = (RuleConfiguredTarget) getConfiguredTarget("//x:foo");
    ImmutableList<ActionAnalysisMetadata> actions = ccLibrary.getActions();
    ImmutableList<String> outputs =
        actions.stream()
            .map(ActionAnalysisMetadata::getPrimaryOutput)
            .map(Artifact::getFilename)
            .collect(ImmutableList.toImmutableList());
    assertThat(outputs).contains("a.pic.o");
  }

  @Test
  public void testWhenSupportsPicDisabledPICObjectAreNotGenerated() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(CppRuleClasses.NO_LEGACY_FEATURES)
                .withActionConfigs(
                    CppActionNames.CPP_LINK_STATIC_LIBRARY,
                    CppActionNames.CPP_COMPILE,
                    CppActionNames.CPP_LINK_NODEPS_DYNAMIC_LIBRARY));
    useConfiguration("--features=-supports_pic");

    scratch.file("x/BUILD", "cc_library(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    RuleConfiguredTarget ccLibrary = (RuleConfiguredTarget) getConfiguredTarget("//x:foo");
    ImmutableList<ActionAnalysisMetadata> actions = ccLibrary.getActions();
    ImmutableList<String> outputs =
        actions.stream()
            .map(ActionAnalysisMetadata::getPrimaryOutput)
            .map(Artifact::getFilename)
            .collect(ImmutableList.toImmutableList());
    assertThat(outputs).doesNotContain("a.pic.o");
  }

  @Test
  public void testWhenSupportsPicDisabledButForcePicSetPICAreGenerated() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(CppRuleClasses.NO_LEGACY_FEATURES, CppRuleClasses.SUPPORTS_PIC)
                .withActionConfigs(
                    CppActionNames.CPP_LINK_STATIC_LIBRARY,
                    CppActionNames.CPP_COMPILE,
                    CppActionNames.CPP_LINK_NODEPS_DYNAMIC_LIBRARY));
    useConfiguration("--force_pic", "--cpu=k8");

    scratch.file("x/BUILD", "cc_library(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    RuleConfiguredTarget ccLibrary = (RuleConfiguredTarget) getConfiguredTarget("//x:foo");
    ImmutableList<ActionAnalysisMetadata> actions = ccLibrary.getActions();
    ImmutableList<String> outputs =
        actions.stream()
            .map(ActionAnalysisMetadata::getPrimaryOutput)
            .map(Artifact::getFilename)
            .collect(ImmutableList.toImmutableList());
    assertThat(outputs).contains("a.pic.o");
  }

  @Test
  public void testWhenSupportsPicNotPresentAndForcePicPassedIsError() throws Exception {
    reporter.removeHandler(failFastHandler);
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(CppRuleClasses.NO_LEGACY_FEATURES)
                .withActionConfigs(
                    CppActionNames.CPP_LINK_STATIC_LIBRARY,
                    CppActionNames.CPP_LINK_NODEPS_DYNAMIC_LIBRARY,
                    CppActionNames.CPP_COMPILE));
    useConfiguration("--force_pic", "--features=-supports_pic");

    scratch.file("x/BUILD", "cc_library(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    getConfiguredTarget("//x:foo");
    assertContainsEvent(
        "PIC compilation is requested but the toolchain does not support it"
            + " (feature named 'supports_pic' is not enabled");
  }

  @Test
  public void testCompilationParameterFile() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.COMPILER_PARAM_FILE));
    scratch.file("a/BUILD", "cc_library(name='foo', srcs=['foo.cc'])");
    CppCompileAction cppCompileAction = getCppCompileAction("//a:foo");
    assertThat(
            cppCompileAction.getArguments().stream()
                .map(x -> removeOutDirectory(x))
                .collect(ImmutableList.toImmutableList()))
        .containsExactly("/usr/bin/mock-gcc", "@/k8-fastbuild/bin/a/_objs/foo/foo.o.params");
  }

  @Test
  public void testClangClParameters() throws Exception {
    if (!AnalysisMock.get().isThisBazel()) {
      return;
    }
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(
                    CppRuleClasses.TARGETS_WINDOWS,
                    CppRuleClasses.COPY_DYNAMIC_LIBRARIES_TO_BINARY));
    scratch.file(
        "a/BUILD",
        "cc_library(",
        "    name='foo',",
        "    srcs=['foo.cc'],",
        "    copts=[",
        "        '/imsvc', 'SYSTEM_INCLUDE_1',",
        "        '-imsvcSYSTEM_INCLUDE_2',",
        "        '/ISTANDARD_INCLUDE',",
        "        '/FI', 'forced_include_1',",
        "        '-FIforced_include_2',",
        "    ],",
        ")");
    CppCompileAction cppCompileAction = getCppCompileAction("//a:foo");

    PathFragment systemInclude1 = PathFragment.create("SYSTEM_INCLUDE_1");
    PathFragment systemInclude2 = PathFragment.create("SYSTEM_INCLUDE_2");
    PathFragment standardInclude = PathFragment.create("STANDARD_INCLUDE");

    assertThat(cppCompileAction.getSystemIncludeDirs()).contains(systemInclude1);
    assertThat(cppCompileAction.getSystemIncludeDirs()).contains(systemInclude2);
    assertThat(cppCompileAction.getSystemIncludeDirs()).doesNotContain(standardInclude);

    assertThat(cppCompileAction.getIncludeDirs()).doesNotContain(systemInclude1);
    assertThat(cppCompileAction.getIncludeDirs()).doesNotContain(systemInclude2);
    assertThat(cppCompileAction.getIncludeDirs()).contains(standardInclude);
  }

  @Test
  public void testCcLibraryLoadedThroughMacro() throws Exception {
    setupTestCcLibraryLoadedThroughMacro(/* loadMacro= */ true);
    assertThat(getConfiguredTarget("//a:a")).isNotNull();
    assertNoEvents();
  }

  @Test
  public void testCcLibraryNotLoadedThroughMacro() throws Exception {
    setupTestCcLibraryLoadedThroughMacro(/* loadMacro= */ false);
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:a");
    assertContainsEvent("rules are deprecated");
  }

  private void setupTestCcLibraryLoadedThroughMacro(boolean loadMacro) throws Exception {
    useConfiguration("--incompatible_load_cc_rules_from_bzl");
    scratch.file(
        "a/BUILD",
        getAnalysisMock().ccSupport().getMacroLoadStatement(loadMacro, "cc_library"),
        "cc_library(name='a', srcs=['a.cc'])");
  }

  @Test
  public void testFdoProfileLoadedThroughMacro() throws Exception {
    setuptestFdoProfileLoadedThroughMacro(/* loadMacro= */ true);
    assertThat(getConfiguredTarget("//a:a")).isNotNull();
    assertNoEvents();
  }

  @Test
  public void testFdoProfileNotLoadedThroughMacro() throws Exception {
    setuptestFdoProfileLoadedThroughMacro(/* loadMacro= */ false);
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:a");
    assertContainsEvent("rules are deprecated");
  }

  private void setuptestFdoProfileLoadedThroughMacro(boolean loadMacro) throws Exception {
    useConfiguration("--incompatible_load_cc_rules_from_bzl");
    scratch.file(
        "a/BUILD",
        getAnalysisMock().ccSupport().getMacroLoadStatement(loadMacro, "fdo_profile"),
        "fdo_profile(name='a', profile='profile.xfdo')");
  }

  @Test
  public void testFdoPrefetchHintsLoadedThroughMacro() throws Exception {
    setupTestFdoPrefetchHintsLoadedThroughMacro(/* loadMacro= */ true);
    assertThat(getConfiguredTarget("//a:a")).isNotNull();
    assertNoEvents();
  }

  @Test
  public void testFdoPrefetchHintsNotLoadedThroughMacro() throws Exception {
    setupTestFdoPrefetchHintsLoadedThroughMacro(/* loadMacro= */ false);
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:a");
    assertContainsEvent("rules are deprecated");
  }

  private void setupTestFdoPrefetchHintsLoadedThroughMacro(boolean loadMacro) throws Exception {
    useConfiguration("--incompatible_load_cc_rules_from_bzl");
    scratch.file(
        "a/BUILD",
        getAnalysisMock().ccSupport().getMacroLoadStatement(loadMacro, "fdo_prefetch_hints"),
        "fdo_prefetch_hints(",
        "    name = 'a',",
        "    profile = 'profile.afdo',",
        ")");
  }

  private String removeOutDirectory(String s) {
    return s.replace("blaze-out", "").replace("bazel-out", "");
  }

  @Test
  public void testNoCoptsDisabled() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("x/BUILD", "cc_library(name = 'foo', srcs = ['a.cc'], nocopts = 'abc')");
    useConfiguration("--incompatible_disable_nocopts");
    getConfiguredTarget("//x:foo");
    assertContainsEvent(
        "This attribute was removed. See https://github.com/bazelbuild/bazel/issues/8706 for"
            + " details.");
  }
}
