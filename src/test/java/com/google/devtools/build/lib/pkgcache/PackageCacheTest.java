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
package com.google.devtools.build.lib.pkgcache;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.SkylarkSemanticsOptions;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.DiffAwareness;
import com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutor;
import com.google.devtools.build.lib.skyframe.SkyValueDirtinessChecker;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.InvocationPolicyEnforcer;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.IOException;
import java.util.List;
import java.util.UUID;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for package loading.
 */
@RunWith(JUnit4.class)
public class PackageCacheTest extends FoundationTestCase {

  private AnalysisMock analysisMock;
  private ConfiguredRuleClassProvider ruleClassProvider;
  private SkyframeExecutor skyframeExecutor;

  @Before
  public final void initializeSkyframeExecutor() throws Exception {
    initializeSkyframeExecutor(/*doPackageLoadingChecks=*/ true);
  }

  private void initializeSkyframeExecutor(boolean doPackageLoadingChecks) throws Exception {
    analysisMock = AnalysisMock.get();
    ruleClassProvider = analysisMock.createRuleClassProvider();
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(outputBase, outputBase),
            rootDirectory,
            analysisMock.getProductName());
    PackageFactory.BuilderForTesting packageFactoryBuilder =
        analysisMock.getPackageFactoryBuilderForTesting(directories);
    if (!doPackageLoadingChecks) {
      packageFactoryBuilder.disableChecks();
    }
    skyframeExecutor =
        SequencedSkyframeExecutor.create(
            packageFactoryBuilder.build(ruleClassProvider, fileSystem),
            fileSystem,
            directories,
            null, /* workspaceStatusActionFactory */
            ruleClassProvider.getBuildInfoFactories(),
            ImmutableList.<DiffAwareness.Factory>of(),
            Predicates.<PathFragment>alwaysFalse(),
            analysisMock.getSkyFunctions(directories),
            ImmutableList.<SkyValueDirtinessChecker>of(),
            PathFragment.EMPTY_FRAGMENT,
            BazelSkyframeExecutorConstants.CROSS_REPOSITORY_LABEL_VIOLATION_STRATEGY,
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY,
            BazelSkyframeExecutorConstants.ACTION_ON_IO_EXCEPTION_READING_BUILD_FILE);
    TestConstants.processSkyframeExecutorForTesting(skyframeExecutor);
    setUpSkyframe(parsePackageCacheOptions(), parseSkylarkSemanticsOptions());
  }

  private void setUpSkyframe(
      PackageCacheOptions packageCacheOptions,
      SkylarkSemanticsOptions skylarkSemanticsOptions) {
    PathPackageLocator pkgLocator =
        PathPackageLocator.create(
            null,
            packageCacheOptions.packagePath,
            reporter,
            rootDirectory,
            rootDirectory,
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY);
    packageCacheOptions.showLoadingProgress = true;
    packageCacheOptions.globbingThreads = 7;
    skyframeExecutor.preparePackageLoading(
        pkgLocator,
        packageCacheOptions,
        skylarkSemanticsOptions,
        analysisMock.getDefaultsPackageContent(),
        UUID.randomUUID(),
        ImmutableMap.<String, String>of(),
        ImmutableMap.<String, String>of(),
        new TimestampGranularityMonitor(BlazeClock.instance()));
    skyframeExecutor.setDeletedPackages(
        ImmutableSet.copyOf(packageCacheOptions.getDeletedPackages()));
  }

  private OptionsParser parse(String... options) throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(
        PackageCacheOptions.class, SkylarkSemanticsOptions.class);
    parser.parse("--default_visibility=public");
    parser.parse(options);

    InvocationPolicyEnforcer optionsPolicyEnforcer = analysisMock.getInvocationPolicyEnforcer();
    try {
      optionsPolicyEnforcer.enforce(parser);
    } catch (OptionsParsingException e) {
      throw new IllegalStateException(e);
    }

    return parser;
  }

  private PackageCacheOptions parsePackageCacheOptions(String... options) throws Exception {
    return parse(options).getOptions(PackageCacheOptions.class);
  }

  private SkylarkSemanticsOptions parseSkylarkSemanticsOptions(String... options) throws Exception {
    return parse(options).getOptions(SkylarkSemanticsOptions.class);
  }

  protected void setOptions(String... options) throws Exception {
    setUpSkyframe(
        parsePackageCacheOptions(options),
        parseSkylarkSemanticsOptions(options));
  }

  private PackageManager getPackageManager() {
    return skyframeExecutor.getPackageManager();
  }

  private void invalidatePackages() throws InterruptedException {
    skyframeExecutor.invalidateFilesUnderPathForTesting(
        reporter, ModifiedFileSet.EVERYTHING_MODIFIED, rootDirectory);
  }

  private Package getPackage(String packageName)
      throws NoSuchPackageException, InterruptedException {
    return getPackageManager().getPackage(reporter,
        PackageIdentifier.createInMainRepo(packageName));
  }

  private Target getTarget(Label label)
      throws NoSuchPackageException, NoSuchTargetException, InterruptedException {
    return getPackageManager().getTarget(reporter, label);
  }

  private Target getTarget(String label) throws Exception {
    return getTarget(Label.parseAbsolute(label));
  }

  private void createPkg1() throws IOException {
    scratch.file("pkg1/BUILD", "cc_library(name = 'foo') # a BUILD file");
  }

  // Check that a substring is present in an error message.
  private void checkGetPackageFails(String packageName, String expectedMessage) throws Exception {
    try {
      getPackage(packageName);
      fail();
    } catch (NoSuchPackageException e) {
      assertThat(e).hasMessageThat().contains(expectedMessage);
    }
  }

  @Test
  public void testGetPackage() throws Exception {
    createPkg1();
    Package pkg1 = getPackage("pkg1");
    assertThat(pkg1.getName()).isEqualTo("pkg1");
    assertThat(pkg1.getFilename().toString()).isEqualTo("/workspace/pkg1/BUILD");
    assertThat(getPackageManager().getPackage(reporter, PackageIdentifier.createInMainRepo("pkg1")))
        .isSameAs(pkg1);
  }

  @Test
  public void testASTIsNotRetained() throws Exception {
    createPkg1();
    Package pkg1 = getPackage("pkg1");
    MoreAsserts.assertInstanceOfNotReachable(pkg1, BuildFileAST.class);
  }

  @Test
  public void testGetNonexistentPackage() throws Exception {
    checkGetPackageFails("not-there",
                         "no such package 'not-there': "
                         + "BUILD file not found on package path");
  }

  @Test
  public void testGetPackageWithInvalidName() throws Exception {
    scratch.file("invalidpackagename&42/BUILD", "cc_library(name = 'foo') # a BUILD file");
    checkGetPackageFails(
        "invalidpackagename&42",
        "no such package 'invalidpackagename&42': Invalid package name 'invalidpackagename&42'");
  }

  @Test
  public void testGetTarget() throws Exception {
    createPkg1();
    Label label = Label.parseAbsolute("//pkg1:foo");
    Target target = getTarget(label);
    assertThat(target.getLabel()).isEqualTo(label);
  }

  @Test
  public void testGetNonexistentTarget() throws Exception {
    createPkg1();
    try {
      getTarget("//pkg1:not-there");
      fail();
    } catch (NoSuchTargetException e) {
      assertThat(e).hasMessage("no such target '//pkg1:not-there': target 'not-there' "
          + "not declared in package 'pkg1' defined by /workspace/pkg1/BUILD");
    }
  }

  /**
   * A missing package is one for which no BUILD file can be found.  The
   * PackageCache caches failures of this kind until the next sync.
   */
  @Test
  public void testRepeatedAttemptsToParseMissingPackage() throws Exception {
    checkGetPackageFails("missing",
                         "no such package 'missing': "
                         + "BUILD file not found on package path");

    // Still missing:
    checkGetPackageFails("missing",
                         "no such package 'missing': "
                         + "BUILD file not found on package path");

    // Update the BUILD file on disk so "missing" is no longer missing:
    scratch.file("missing/BUILD",
                 "# an ok build file");

    // Still missing:
    checkGetPackageFails("missing",
                         "no such package 'missing': "
                         + "BUILD file not found on package path");

    invalidatePackages();

    // Found:
    Package missing = getPackage("missing");

    assertThat(missing.getName()).isEqualTo("missing");
  }

  /**
   * A broken package is one that exists but contains lexer/parser/evaluator errors. The
   * PackageCache only makes one attempt to parse each package once found.
   *
   * <p>Depending on the strictness of the PackageFactory, parsing a broken package may cause a
   * Package object to be returned (possibly missing some rules) or an exception to be thrown. For
   * this test we need that strict behavior.
   *
   * <p>Note: since the PackageCache.setStrictPackageCreation method was deleted (since it wasn't
   * used by any significant clients) creating a "broken" build file got trickier--syntax errors are
   * not enough.  For now, we create an unreadable BUILD file, which will cause an IOException to be
   * thrown. This test seems less valuable than it once did.
   */
  @Test
  public void testParseBrokenPackage() throws Exception {
    reporter.removeHandler(failFastHandler);

    Path brokenBuildFile = scratch.file("broken/BUILD");
    brokenBuildFile.setReadable(false);

    try {
      getPackage("broken");
      fail();
    } catch (BuildFileContainsErrorsException e) {
      assertThat(e).hasMessageThat().contains("/workspace/broken/BUILD (Permission denied)");
    }
    eventCollector.clear();

    // Update the BUILD file on disk so "broken" is no longer broken:
    scratch.overwriteFile("broken/BUILD",
                 "# an ok build file");

    invalidatePackages(); //  resets cache of failures

    Package broken = getPackage("broken");
    assertThat(broken.getName()).isEqualTo("broken");
    assertNoEvents();
  }

  @Test
  public void testMovedBuildFileCausesReloadAfterSync() throws Exception {
    Path buildFile1 = scratch.file("pkg/BUILD",
                                  "cc_library(name = 'foo')");
    Path buildFile2 = scratch.file("/otherroot/pkg/BUILD",
                                  "cc_library(name = 'bar')");
    setOptions("--package_path=/workspace:/otherroot");

    Package oldPkg = getPackage("pkg");
    assertThat(getPackage("pkg")).isSameAs(oldPkg); // change not yet visible
    assertThat(oldPkg.getFilename()).isEqualTo(buildFile1);
    assertThat(oldPkg.getSourceRoot()).isEqualTo(rootDirectory);

    buildFile1.delete();
    invalidatePackages();

    Package newPkg = getPackage("pkg");
    assertThat(newPkg).isNotSameAs(oldPkg);
    assertThat(newPkg.getFilename()).isEqualTo(buildFile2);
    assertThat(newPkg.getSourceRoot()).isEqualTo(scratch.dir("/otherroot"));

    // TODO(bazel-team): (2009) test BUILD file moves in the other direction too.
  }

  private Path rootDir1;
  private Path rootDir2;

  private void setUpCacheWithTwoRootLocator() throws Exception {
    // Root 1:
    //   /a/BUILD
    //   /b/BUILD
    //   /c/d
    //   /c/e
    //
    // Root 2:
    //   /b/BUILD
    //   /c/BUILD
    //   /c/d/BUILD
    //   /f/BUILD
    //   /f/g
    //   /f/g/h/BUILD

    rootDir1 = scratch.dir("/workspace");
    rootDir2 = scratch.dir("/otherroot");

    createBuildFile(rootDir1, "a", "foo.txt", "bar/foo.txt");
    createBuildFile(rootDir1, "b", "foo.txt", "bar/foo.txt");

    rootDir1.getRelative("c").createDirectory();
    rootDir1.getRelative("c/d").createDirectory();
    rootDir1.getRelative("c/e").createDirectory();

    createBuildFile(rootDir2, "c", "d", "d/foo.txt", "foo.txt", "bar/foo.txt", "e", "e/foo.txt");
    createBuildFile(rootDir2, "c/d", "foo.txt");
    createBuildFile(rootDir2, "f", "g/foo.txt", "g/h", "g/h/foo.txt", "foo.txt");
    createBuildFile(rootDir2, "f/g/h", "foo.txt");

    setOptions("--package_path=/workspace:/otherroot");
  }

  protected Path createBuildFile(Path workspace, String packageName,
      String... targets) throws IOException {
    String[] lines = new String[targets.length];

    for (int i = 0; i < targets.length; i++) {
      lines[i] = "sh_library(name='" + targets[i] + "')";
    }

    return scratch.file(workspace + "/" + packageName + "/BUILD", lines);
  }

  private void assertLabelValidity(boolean expected, String labelString)
      throws Exception {
    Label label = Label.parseAbsolute(labelString);

    boolean actual = false;
    String error = null;
    try {
      getTarget(label);
      actual = true;
    } catch (NoSuchPackageException | NoSuchTargetException e) {
      error = e.getMessage();
    }
    if (actual != expected) {
      fail("assertLabelValidity(" + label + ") "
           + actual + ", not equal to expected value " + expected
           + " (error=" + error + ")");
    }
  }

  private void assertPackageLoadingFails(String pkgName, String expectedError) throws Exception {
    Package pkg = getPackage(pkgName);
    assertThat(pkg.containsErrors()).isTrue();
    assertContainsEvent(expectedError);
  }

  @Test
  public void testLocationForLabelCrossingSubpackage() throws Exception {
    scratch.file("e/f/BUILD");
    scratch.file("e/BUILD",
        "# Whatever",
        "filegroup(name='fg', srcs=['f/g'])");
    reporter.removeHandler(failFastHandler);
    List<Event> events = getPackage("e").getEvents();
    assertThat(events).hasSize(1);
    assertThat(events.get(0).getLocation().getStartLineAndColumn().getLine()).isEqualTo(2);
  }

  /** Static tests (i.e. no changes to filesystem, nor calls to sync). */
  @Test
  public void testLabelValidity() throws Exception {
    reporter.removeHandler(failFastHandler);
    setUpCacheWithTwoRootLocator();

    scratch.file(rootDir2 + "/c/d/foo.txt");

    assertLabelValidity(true, "//a:foo.txt");
    assertLabelValidity(true, "//a:bar/foo.txt");
    assertLabelValidity(false, "//a/bar:foo.txt"); //  no such package a/bar

    assertLabelValidity(true, "//b:foo.txt");
    assertLabelValidity(true, "//b:bar/foo.txt");
    assertLabelValidity(false, "//b/bar:foo.txt"); // no such package b/bar

    assertLabelValidity(true, "//c:foo.txt");
    assertLabelValidity(true, "//c:bar/foo.txt");
    assertLabelValidity(false, "//c/bar:foo.txt"); // no such package c/bar

    assertLabelValidity(true, "//c:foo.txt");

    assertLabelValidity(false, "//c:d/foo.txt"); // crosses boundary of c/d
    assertLabelValidity(true, "//c/d:foo.txt");

    assertLabelValidity(true, "//c:foo.txt");
    assertLabelValidity(true, "//c:e");
    assertLabelValidity(true, "//c:e/foo.txt");
    assertLabelValidity(false, "//c/e:foo.txt"); // no such package c/e

    assertLabelValidity(true, "//f:foo.txt");
    assertLabelValidity(true, "//f:g/foo.txt");
    assertLabelValidity(false, "//f/g:foo.txt"); // no such package f/g
    assertLabelValidity(false, "//f:g/h/foo.txt"); // crosses boundary of f/g/h
    assertLabelValidity(false, "//f/g:h/foo.txt"); // no such package f/g
    assertLabelValidity(true, "//f/g/h:foo.txt");
  }

  /** Dynamic tests of label validity. */
  @Test
  public void testAddedBuildFileCausesLabelToBecomeInvalid() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("pkg/BUILD",
                 "           cc_library(name = 'foo', ",
                 "           srcs = ['x/y.cc'])");

    assertLabelValidity(true, "//pkg:x/y.cc");

    // The existence of this file makes 'x/y.cc' an invalid reference.
    scratch.file("pkg/x/BUILD");

    // but not yet...
    assertLabelValidity(true, "//pkg:x/y.cc");

    invalidatePackages();

    // now:
    assertPackageLoadingFails("pkg",
        "Label '//pkg:x/y.cc' crosses boundary of subpackage 'pkg/x' "
        + "(perhaps you meant to put the colon here: '//pkg/x:y.cc'?)");
  }

  @Test
  public void testDeletedPackages() throws Exception {
    // PackageLoader doesn't support --deleted_packages.
    initializeSkyframeExecutor(/*doPackageLoadingChecks=*/ false);
    reporter.removeHandler(failFastHandler);
    setUpCacheWithTwoRootLocator();
    createBuildFile(rootDir1, "c", "d/x");
    // Now package c exists in both roots, and c/d exists in only in the second
    // root.  It's as if we've merged c and c/d in the first root.

    // c/d is still a subpackage--found in the second root:
    assertThat(getPackage("c/d").getFilename()).isEqualTo(rootDir2.getRelative("c/d/BUILD"));

    // Subpackage labels are still valid...
    assertLabelValidity(true, "//c/d:foo.txt");
    // ...and this crosses package boundaries:
    assertLabelValidity(false, "//c:d/x");
    assertPackageLoadingFails("c",
        "Label '//c:d/x' crosses boundary of subpackage 'c/d' (have you deleted c/d/BUILD? "
        + "If so, use the --deleted_packages=c/d option)");

    assertThat(getPackageManager().isPackage(reporter, PackageIdentifier.createInMainRepo("c/d")))
        .isTrue();

    setOptions("--deleted_packages=c/d");
    invalidatePackages();

    assertThat(getPackageManager().isPackage(reporter, PackageIdentifier.createInMainRepo("c/d")))
        .isFalse();

    // c/d is no longer a subpackage--even though there's a BUILD file in the
    // second root:
    try {
      getPackage("c/d");
      fail();
    } catch (NoSuchPackageException e) {
      assertThat(e).hasMessage(
          "no such package 'c/d': Package is considered deleted due to --deleted_packages");
    }

    // Labels in the subpackage are no longer valid...
    assertLabelValidity(false, "//c/d:x");
    // ...and now d is just a subdirectory of c:
    assertLabelValidity(true, "//c:d/x");
  }

  @Test
  public void testPackageFeatures() throws Exception {
    scratch.file("peach/BUILD",
        "package(features = ['crosstool_default_false'])",
        "cc_library(name = 'cc', srcs = ['cc.cc'])");
    Rule cc = (Rule) getTarget("//peach:cc");
    assertThat(cc.getFeatures()).hasSize(1);
  }

  @Test
  public void testBrokenPackageOnMultiplePackagePathEntries() throws Exception {
    reporter.removeHandler(failFastHandler);
    setOptions("--package_path=.:.");
    scratch.file("x/y/BUILD");
    scratch.file("x/BUILD",
        "genrule(name = 'x',",
        "srcs = [],",
        "outs = ['y/z.h'],",
        "cmd  = '')");
    Package p = getPackage("x");
    assertThat(p.containsErrors()).isTrue();
  }
}
