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

import static com.google.common.collect.Iterables.filter;
import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.baseArtifactNames;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.baseNamesOf;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.pkgcache.LoadingFailedException;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;
import java.util.List;

/**
 * A test for {@link CcCommon}.
 */
@RunWith(JUnit4.class)
public class CcCommonConfiguredTargetTest extends BuildViewTestCase {

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
    scratch.file("a/BUILD",
        "cc_library(name='a', srcs=['a1', 'a2'])",
        "filegroup(name='a1', srcs=['a.cc'])",
        "filegroup(name='a2', srcs=['a.cc'])");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:a");
    assertContainsEvent("Artifact 'a/a.cc' is duplicated");
  }

  @Test
  public void testSameHeaderFileTwice() throws Exception {
    scratch.file("a/BUILD",
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
      assertEquals("libemptylib.a", baseNamesOf(getFilesToBuild(emptylib)));
    } else {
      assertThat(getFilesToBuild(emptylib)).isEmpty();
    }
    assertTrue(
        emptylib
            .getProvider(CcExecutionDynamicLibrariesProvider.class)
            .getExecutionDynamicLibraryArtifacts()
            .isEmpty());
  }

  protected static boolean emptyShouldOutputStaticLibrary() {
    return !TestConstants.THIS_IS_BAZEL;
  }

  @Test
  public void testEmptyBinary() throws Exception {
    ConfiguredTarget emptybin = getConfiguredTarget("//empty:emptybinary");
    assertEquals("emptybinary", baseNamesOf(getFilesToBuild(emptybin)));
  }

  private List<String> getCopts(String target) throws Exception {
    ConfiguredTarget cLib = getConfiguredTarget(target);
    Artifact object = getOnlyElement(getOutputGroup(cLib, OutputGroupProvider.FILES_TO_COMPILE));
    CppCompileAction compileAction = (CppCompileAction) getGeneratingAction(object);
    return compileAction.getCompilerOptions();
  }

  @Test
  public void testCopts() throws Exception {
    scratch.file("copts/BUILD",
        "cc_library(name = 'c_lib',",
        "    srcs = ['foo.cc'],",
        "    copts = [ '-Wmy-warning', '-frun-faster' ])");
    MoreAsserts.assertContainsSublist(getCopts("//copts:c_lib"), "-Wmy-warning", "-frun-faster");
  }

  @Test
  public void testCoptsTokenization() throws Exception {
    scratch.file("copts/BUILD",
        "cc_library(name = 'c_lib',",
        "    srcs = ['foo.cc'],",
        "    copts = ['-Wmy-warning -frun-faster'])");
    List<String> copts = getCopts("//copts:c_lib");
    MoreAsserts.assertContainsSublist(copts, "-Wmy-warning", "-frun-faster");
    assertContainsEvent("each item in the list should contain only one option");
  }

  @Test
  public void testCoptsNoTokenization() throws Exception {
    scratch.file("copts/BUILD",
        "package(features = ['no_copts_tokenization'])",
        "cc_library(name = 'c_lib',",
        "    srcs = ['foo.cc'],",
        "    copts = ['-Wmy-warning -frun-faster'])");
    List<String> copts = getCopts("//copts:c_lib");
    MoreAsserts.assertContainsSublist(copts, "-Wmy-warning -frun-faster");
  }

  /**
   * Test that we handle ".a" files in cc_library srcs correctly when
   * linking dynamically.  In particular, if srcs contains only the ".a"
   * file for a library, with no corresponding ".so", then we need
   * to link in the ".a" file even when we're linking dynamically.
   * If srcs contains both ".a" and ".so" then we should only link
   * in the ".so".
   */
  @Test
  public void testArchiveInCcLibrarySrcs() throws Exception {
    ConfiguredTarget archiveInSrcsTest =
        scratchConfiguredTarget(
            "archive_in_srcs",
            "archive_in_srcs_test",
            "cc_test(name = 'archive_in_srcs_test',",
            "           srcs = ['archive_in_srcs_test.cc'],",
            "           deps = [':archive_in_srcs_lib'])",
            "cc_library(name = 'archive_in_srcs_lib',",
            "           srcs = ['libstatic.a', 'libboth.a', 'libboth.so'])");
    Iterable<Artifact> libraries = getLinkerInputs(archiveInSrcsTest);
    assertThat(baseArtifactNames(libraries))
        .containsAllOf("archive_in_srcs_test.pic.o", "libboth.so", "libstatic.a");
  }

  private Iterable<Artifact> getLinkerInputs(ConfiguredTarget target) {
    Artifact executable = getExecutable(target);
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(executable);
    return LinkerInputs.toLibraryArtifacts(linkAction.getLinkCommandLine().getLinkerInputs());
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
    assertTrue(
        statically
            .getProvider(CcExecutionDynamicLibrariesProvider.class)
            .getExecutionDynamicLibraryArtifacts()
            .isEmpty());
    Artifact staticallyDotA = getOnlyElement(getFilesToBuild(statically));
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
    assertThat(isolatedDefines.getProvider(CppCompilationContext.class).getDefines())
        .containsExactly("FOO", "BAR")
        .inOrder();
  }

  @Test
  public void testStartEndLib() throws Exception {
    CrosstoolConfigurationHelper.overwriteCrosstoolWithToolchain(
        directories.getWorkspace(),
        CrosstoolConfig.CToolchain.newBuilder().setSupportsStartEndLib(true).buildPartial());
    useConfiguration(
        // Prevent Android from trying to setup ARM crosstool by forcing it on system cpu.
        "--fat_apk_cpu=" + CrosstoolConfigurationHelper.defaultCpu(),
        "--start_end_lib");
    scratch.file(
        "test/BUILD",
        "cc_library(name='lib',",
        "           srcs=['lib.c'])",
        "cc_binary(name='bin',",
        "          srcs=['bin.c'])");

    ConfiguredTarget target = getConfiguredTarget("//test:bin");
    CppLinkAction action = (CppLinkAction) getGeneratingAction(getExecutable(target));
    for (Artifact input : action.getInputs()) {
      String name = input.getFilename();
      assertTrue(!CppFileTypes.ARCHIVE.matches(name) && !CppFileTypes.PIC_ARCHIVE.matches(name));
    }
  }

  @Test
  public void testTempsWithDifferentExtensions() throws Exception {
    useConfiguration("--save_temps");
    scratch.file(
        "ananas/BUILD",
        "cc_library(name='ananas',",
        "           srcs=['1.c', '2.cc', '3.cpp', '4.S', '5.h', '6.hpp'])");

    ConfiguredTarget ananas = getConfiguredTarget("//ananas:ananas");
    Iterable<String> temps =
        ActionsTestUtil.baseArtifactNames(getOutputGroup(ananas, OutputGroupProvider.TEMP_FILES));
    assertThat(temps)
        .containsExactly(
            "1.pic.i", "1.pic.s",
            "2.pic.ii", "2.pic.s",
            "3.pic.ii", "3.pic.s");
  }

  @Test
  public void testTempsForCc() throws Exception {
    useConfiguration("--save_temps");
    ConfiguredTarget fooTarget = getConfiguredTarget("//foo:foo");
    List<Artifact> temps =
        ImmutableList.copyOf(getOutputGroup(fooTarget, OutputGroupProvider.TEMP_FILES));
    assertThat(temps).hasSize(2);

    // Assert that the two temps are the .i and .s files we expect.
    getOnlyElement(filter(temps, fileTypePredicate(CppFileTypes.PIC_PREPROCESSED_CPP)));
    getOnlyElement(filter(temps, fileTypePredicate(CppFileTypes.PIC_ASSEMBLER)));
  }

  @Test
  public void testTempsForCcNoPIC() throws Exception {
    useConfiguration("--save_temps", "--cpu=piii");
    ConfiguredTarget fooTarget = getConfiguredTarget("//foo:foo");
    List<Artifact> temps =
        ImmutableList.copyOf(getOutputGroup(fooTarget, OutputGroupProvider.TEMP_FILES));
    assertThat(temps).hasSize(2);

    // Assert that the two temps are the .i and .s files we expect.
    getOnlyElement(filter(temps, fileTypePredicate(CppFileTypes.PREPROCESSED_CPP)));
    getOnlyElement(filter(temps, fileTypePredicate(CppFileTypes.ASSEMBLER)));
  }

  @Test
  public void testTempsForC() throws Exception {
    useConfiguration("--save_temps");
    // Now try with a .c source file.
    scratch.file("csrc/BUILD", "cc_library(name='csrc',", "           srcs=['foo.c'])");
    ConfiguredTarget csrcTarget = getConfiguredTarget("//csrc:csrc");
    List<Artifact> cTemps =
        ImmutableList.copyOf(getOutputGroup(csrcTarget, OutputGroupProvider.TEMP_FILES));
    assertThat(cTemps).hasSize(2);

    // Assert that the two temps are the .ii and .s files we expect.
    getOnlyElement(filter(cTemps, fileTypePredicate(CppFileTypes.PIC_PREPROCESSED_C)));
    getOnlyElement(filter(cTemps, fileTypePredicate(CppFileTypes.PIC_ASSEMBLER)));
  }

  @Test
  public void testTempsForTwoCc() throws Exception {
    useConfiguration("--save_temps");

    // For two source files we're expecting 4 temps.
    scratch.file(
        "twosrc/BUILD", "cc_library(name='twosrc',", "           srcs=['foo1.cc', 'foo2.cc'])");
    ConfiguredTarget twoSrcTarget = getConfiguredTarget("//twosrc:twosrc");
    assertThat(getOutputGroup(twoSrcTarget, OutputGroupProvider.TEMP_FILES)).hasSize(4);
  }

  private static Predicate<Artifact> fileTypePredicate(final FileType type) {
    return new Predicate<Artifact>() {
      @Override
      public boolean apply(Artifact artifact) {
        return type.matches(artifact.getFilename());
      }
    };
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

  /**
   * Tests that nocopts= "-fPIC" takes '-fPIC' out of a compile invocation even if the
   * crosstool requires fPIC compilation (i.e. nocoopts overrides crosstool settings on
   * a rule-specific basis).
   */
  @Test
  public void testNoCoptfPicOverride() throws Exception {
    CrosstoolConfigurationHelper.overwriteCrosstoolWithToolchain(
        directories.getWorkspace(),
        CrosstoolConfig.CToolchain.newBuilder().setNeedsPic(true).buildPartial());
    useConfiguration(
        // Prevent Android from trying to setup ARM crosstool by forcing it on system cpu.
        "--fat_apk_cpu=" + CrosstoolConfigurationHelper.defaultCpu());

    scratch.file(
        "a/BUILD",
        "cc_binary(name = 'pic',",
        "           srcs = [ 'binary.cc' ])",
        "cc_binary(name = 'libpic.so',",
        "           srcs = [ 'binary.cc' ])",
        "cc_library(name = 'piclib',",
        "           srcs = [ 'library.cc' ])",
        "cc_binary(name = 'nopic',",
        "           srcs = [ 'binary.cc' ],",
        "           nocopts = '-fPIC')",
        "cc_binary(name = 'libnopic.so',",
        "           srcs = [ 'binary.cc' ],",
        "           nocopts = '-fPIC')",
        "cc_library(name = 'nopiclib',",
        "           srcs = [ 'library.cc' ],",
        "           nocopts = '-fPIC')");

    assertThat(getCppCompileAction("//a:pic").getArgv()).contains("-fPIC");
    assertThat(getCppCompileAction("//a:libpic.so").getArgv()).contains("-fPIC");
    assertThat(getCppCompileAction("//a:piclib").getArgv()).contains("-fPIC");
    assertThat(getCppCompileAction("//a:nopic").getArgv()).doesNotContain("-fPIC");
    assertThat(getCppCompileAction("//a:libnopic.so").getArgv()).doesNotContain("-fPIC");
    assertThat(getCppCompileAction("//a:nopiclib").getArgv()).doesNotContain("-fPIC");
  }

  private CppCompileAction getCppCompileAction(String label) throws Exception {
    ConfiguredTarget target = getConfiguredTarget(label);
    List<CppCompileAction> compilationSteps =
        actionsTestUtil()
            .findTransitivePrerequisitesOf(
                getFilesToBuild(target).iterator().next(), CppCompileAction.class);
    return compilationSteps.get(0);
  }

  @Test
  public void testIsolatedIncludes() throws Exception {
    // Tests the (immediate) effect of declaring the includes attribute on a
    // cc_library.

    useConfiguration("--use_isystem_for_includes=false");

    scratch.file(
        "bang/BUILD",
        "cc_library(name = 'bang',",
        "           srcs = ['bang.cc'],",
        "           includes = ['bang_includes'])");

    ConfiguredTarget foo = getConfiguredTarget("//bang:bang");

    String includesRoot = "bang/bang_includes";
    List<PathFragment> expected =
        ImmutableList.of(
            new PathFragment(includesRoot),
            targetConfig.getGenfilesFragment().getRelative(includesRoot));
    assertEquals(expected, foo.getProvider(CppCompilationContext.class).getIncludeDirs());
  }

  @Test
  public void testUseIsystemForIncludes() throws Exception {
    // Tests the effect of --use_isystem_for_includes.

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
            .addAll(noIncludes.getProvider(CppCompilationContext.class).getSystemIncludeDirs())
            .add(new PathFragment(includesRoot))
            .add(targetConfig.getGenfilesFragment().getRelative(includesRoot))
            .build();
    assertThat(foo.getProvider(CppCompilationContext.class).getSystemIncludeDirs())
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
    getPackageManager().getPackage(reporter, PackageIdentifier.createInDefaultRepo("cc/common"));
    assertContainsEvent(
        "//cc/common:testlib: no such attribute 'alwayslink'" + " in 'cc_test' rule");
  }

  @Test
  public void testCcTestBuiltWithFissionHasDwp() throws Exception {
    // Tests that cc_tests built statically and with Fission will have the .dwp file
    // in their runfiles.

    useConfiguration("--build_test_dwp", "--dynamic_mode=off", "--linkopt=-static",
        "--fission=yes");
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "mypackage",
            "mytest",
            "cc_test(name = 'mytest', ",
            "         srcs = ['mytest.cc'])");

    Iterable<Artifact> runfiles = collectRunfiles(target);
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

    Iterable<Artifact> libraries = getLinkerInputs(wrapsophos);

    // The "libsavi.a" below is the empty ".a" file created by Blaze for the
    // "savi" cc_library rule (empty since it has no ".cc" files in "srcs").
    // The "libsavi.so" below is the "lib/libsavi.so" file from "srcs".
    //
    // TODO(blaze-team): (2009) the order here is a bit odd; it would make more sense
    // to put the library for the rule ("libsavi.a") before the ".so" file
    // from "srcs" ("libsavi.so").  I think this is because we currently
    // list all the .so files for a rule before all the .a files for the rule.
    assertThat(baseArtifactNames(libraries))
        .containsAllOf("wrapsophos.pic.o", "libsophosengine.a", "libsavi.so");
    if (emptyShouldOutputStaticLibrary()) {
      assertThat(baseArtifactNames(libraries)).contains("libsavi.a");
    }
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
    Action linkAction = getGeneratingAction(Iterables.getOnlyElement(getFilesToBuild(theApp)));
    ImmutableList<Artifact> filesToBuild = ImmutableList.copyOf(getFilesToBuild(theLib));
    assertTrue(ImmutableSet.copyOf(linkAction.getInputs()).containsAll(filesToBuild));
  }

  @Test
  public void testMissingLabelInLinkopts() throws Exception {
    scratch.file(
        "linklow/BUILD",
        "genrule(name = 'linklow_linker_script',",
        "  srcs = [ 'default_linker_script' ],",
        "  tools = [ 'default_linker_script' ],",
        "  outs = [ 'linklow.lds' ],",
        "  cmd = 'cat  $< > $@')");
    checkError(
        "ocean/scoring2",
        "ms-ascorer",
        // error:
        "could not resolve label '//linklow:linklow_linker_script'",
        "cc_binary(name = 'ms-ascorer',",
        "    srcs = [ ],",
        "    deps = [ ':ascorer-servlet'],",
        "    linkopts = [ '-static', '-Xlinker', '-script', '//linklow:linklow_linker_script'])",
        "cc_library(name = 'ascorer-servlet')");
  }

  @Test
  public void testCcLibraryWithDashStatic() throws Exception {
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
  public void testStampTests() throws Exception {
    scratch.file("test/BUILD",
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
    assertEquals(
        enabled, AnalysisUtils.isStampingEnabled(getRuleContext(getConfiguredTarget(label))));
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

  @Test
  public void testSelectPreferredLibrariesInvariant() {
    // All combinations of libraries:
    // a - static+pic+shared
    // b - static+pic
    // c - static+shared
    // d - static
    // e - pic+shared
    // f - pic
    // g - shared
    CcLinkingOutputs linkingOutputs =
        CcLinkingOutputs.builder()
            .addStaticLibraries(
                ImmutableList.copyOf(
                    LinkerInputs.opaqueLibrariesToLink(
                        Arrays.asList(
                            getSourceArtifact("liba.a"),
                            getSourceArtifact("libb.a"),
                            getSourceArtifact("libc.a"),
                            getSourceArtifact("libd.a")))))
            .addPicStaticLibraries(
                ImmutableList.copyOf(
                    LinkerInputs.opaqueLibrariesToLink(
                        Arrays.asList(
                            getSourceArtifact("liba.pic.a"),
                            getSourceArtifact("libb.pic.a"),
                            getSourceArtifact("libe.pic.a"),
                            getSourceArtifact("libf.pic.a")))))
            .addDynamicLibraries(
                ImmutableList.copyOf(
                    LinkerInputs.opaqueLibrariesToLink(
                        Arrays.asList(
                            getSourceArtifact("liba.so"),
                            getSourceArtifact("libc.so"),
                            getSourceArtifact("libe.so"),
                            getSourceArtifact("libg.so")))))
            .build();

    // Whether linkShared is true or false, this should return the identical results.
    List<Artifact> sharedLibraries1 =
        FileType.filterList(
            LinkerInputs.toLibraryArtifacts(linkingOutputs.getPreferredLibraries(true, false)),
            CppFileTypes.SHARED_LIBRARY);
    List<Artifact> sharedLibraries2 =
        FileType.filterList(
            LinkerInputs.toLibraryArtifacts(linkingOutputs.getPreferredLibraries(true, true)),
            CppFileTypes.SHARED_LIBRARY);
    assertEquals(sharedLibraries1, sharedLibraries2);
  }

  /**
   * Tests that shared libraries of the form "libfoo.so.1.2" are permitted within "srcs".
   */
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
    Iterable<Artifact> libraries = getLinkerInputs(target);
    assertThat(baseArtifactNames(libraries))
        .containsAllOf("mybinary.pic.o", "libmylib.a", "libshared.so", "libshared.so.1.1");
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
  public void testExplicitBadStl() throws Exception {
    scratch.file("x/BUILD",
        "cc_binary(name = 'x', srcs = ['x.cc'])");

    reporter.removeHandler(failFastHandler);
    try {
      useConfiguration("--experimental_stl=//x:blah");
      update(Arrays.asList("//x:x"), true, 10, false, new EventBus());
      fail("found non-existing target");
    } catch (LoadingFailedException expected) {
      assertThat(expected.getMessage()).contains("Failed to load required STL target: '//x:blah'");
    }

    try {
      useConfiguration("--experimental_stl=//blah");
      update(Arrays.asList("//x:x"), true, 10, false, new EventBus());
      fail("found non-existsing target");
    } catch (LoadingFailedException expected) {
      assertThat(expected.getMessage())
          .contains("Failed to load required STL target: '//blah:blah'");
    }

    // Without -k.
    try {
      useConfiguration("--experimental_stl=//blah");
      update(Arrays.asList("//x:x"), false, 10, false, new EventBus());
      fail("found non-existsing target");
    } catch (LoadingFailedException expected) {
      assertThat(expected.getMessage()).contains("Loading failed; build aborted");
    }
  }
}
