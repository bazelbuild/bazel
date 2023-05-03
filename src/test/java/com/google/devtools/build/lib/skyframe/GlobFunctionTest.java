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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Functions;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.io.FileSymlinkCycleUniquenessFunction;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.packages.Globber;
import com.google.devtools.build.lib.packages.Globber.Operation;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.GlobValue.InvalidGlobPatternException;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStateKey;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.Differencer.DiffWithDelta.Delta;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link GlobFunction}.
 */
public abstract class GlobFunctionTest {
  private static final EvaluationContext EVALUATION_OPTIONS =
      EvaluationContext.newBuilder()
          .setKeepGoing(false)
          .setParallelism(SkyframeExecutor.DEFAULT_THREAD_COUNT)
          .setEventHandler(NullEventHandler.INSTANCE)
          .build();

  @RunWith(JUnit4.class)
  public static class GlobFunctionAlwaysUseDirListingTest extends GlobFunctionTest {
    @Override
    protected boolean alwaysUseDirListing() {
      return true;
    }
  }

  @RunWith(JUnit4.class)
  public static class RegularGlobFunctionTest extends GlobFunctionTest {
    @Override
    protected boolean alwaysUseDirListing() {
      return false;
    }
  }

  private CustomInMemoryFs fs;
  private MemoizingEvaluator evaluator;
  private RecordingDifferencer differencer;
  private Path root;
  private Path writableRoot;
  private Path pkgPath;
  private AtomicReference<PathPackageLocator> pkgLocator;

  private static final PackageIdentifier PKG_ID = PackageIdentifier.createInMainRepo("pkg");

  @Before
  public final void setUp() throws Exception  {
    fs = new CustomInMemoryFs(new ManualClock());
    root = fs.getPath("/root/workspace");
    writableRoot = fs.getPath("/writableRoot/workspace");
    pkgPath = root.getRelative(PKG_ID.getPackageFragment());

    pkgLocator =
        new AtomicReference<>(
            new PathPackageLocator(
                fs.getPath("/output_base"),
                ImmutableList.of(Root.fromPath(writableRoot), Root.fromPath(root)),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));

    differencer = new SequencedRecordingDifferencer();
    evaluator = new InMemoryMemoizingEvaluator(createFunctionMap(), differencer);
    PrecomputedValue.BUILD_ID.set(differencer, UUID.randomUUID());
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator.get());
    PrecomputedValue.STARLARK_SEMANTICS.set(differencer, StarlarkSemantics.DEFAULT);
    RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE.set(
        differencer, Optional.empty());

    createTestFiles();
  }

  private Map<SkyFunctionName, SkyFunction> createFunctionMap() {
    AtomicReference<ImmutableSet<PackageIdentifier>> deletedPackages =
        new AtomicReference<>(ImmutableSet.of());
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(root, root, root),
            root,
            /* defaultSystemJavabase= */ null,
            TestConstants.PRODUCT_NAME);
    ExternalFilesHelper externalFilesHelper =
        ExternalFilesHelper.createForTesting(
            pkgLocator,
            ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS,
            directories);

    Map<SkyFunctionName, SkyFunction> skyFunctions = new HashMap<>();
    skyFunctions.put(SkyFunctions.GLOB, new GlobFunction(alwaysUseDirListing()));
    skyFunctions.put(
        SkyFunctions.DIRECTORY_LISTING_STATE,
        new DirectoryListingStateFunction(externalFilesHelper, SyscallCache.NO_CACHE));
    skyFunctions.put(SkyFunctions.DIRECTORY_LISTING, new DirectoryListingFunction());
    skyFunctions.put(
        SkyFunctions.PACKAGE_LOOKUP,
        new PackageLookupFunction(
            deletedPackages,
            CrossRepositoryLabelViolationStrategy.ERROR,
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY,
            BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER));
    skyFunctions.put(
        SkyFunctions.IGNORED_PACKAGE_PREFIXES,
        BazelSkyframeExecutorConstants.IGNORED_PACKAGE_PREFIXES_FUNCTION);
    skyFunctions.put(
        FileStateKey.FILE_STATE,
        new FileStateFunction(
            Suppliers.ofInstance(new TimestampGranularityMonitor(BlazeClock.instance())),
            SyscallCache.NO_CACHE,
            externalFilesHelper));
    skyFunctions.put(FileValue.FILE, new FileFunction(pkgLocator, directories));
    skyFunctions.put(
        FileSymlinkCycleUniquenessFunction.NAME, new FileSymlinkCycleUniquenessFunction());
    AnalysisMock analysisMock = AnalysisMock.get();
    RuleClassProvider ruleClassProvider = analysisMock.createRuleClassProvider();
    skyFunctions.put(
        WorkspaceFileValue.WORKSPACE_FILE,
        new WorkspaceFileFunction(
            ruleClassProvider,
            analysisMock
                .getPackageFactoryBuilderForTesting(directories)
                .build(ruleClassProvider, fs),
            directories,
            /*bzlLoadFunctionForInlining=*/ null));
    skyFunctions.put(
        SkyFunctions.EXTERNAL_PACKAGE,
        new ExternalPackageFunction(BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER));
    skyFunctions.put(
        SkyFunctions.LOCAL_REPOSITORY_LOOKUP,
        new LocalRepositoryLookupFunction(BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER));
    return skyFunctions;
  }

  protected abstract boolean alwaysUseDirListing();

  private void createTestFiles() throws IOException {
    pkgPath.createDirectoryAndParents();
    FileSystemUtils.createEmptyFile(pkgPath.getRelative("BUILD"));
    for (String dir :
        ImmutableList.of(
            "foo/bar/wiz", "foo/barnacle/wiz", "food/barnacle/wiz", "fool/barnacle/wiz")) {
      pkgPath.getRelative(dir).createDirectoryAndParents();
    }
    FileSystemUtils.createEmptyFile(pkgPath.getRelative("foo/bar/wiz/file"));

    // Used for testing the behavior of globbing into nested subpackages.
    for (String dir : ImmutableList.of("a1/b1/c", "a2/b2/c")) {
      pkgPath.getRelative(dir).createDirectoryAndParents();
    }
    FileSystemUtils.createEmptyFile(pkgPath.getRelative("a2/b2/BUILD"));
  }

  @Test
  public void testSimple() throws Exception {
    assertGlobMatches("food", /* => */ "food");
  }

  @Test
  public void testIgnoreList() throws Exception {
    FileSystemUtils.writeContentAsLatin1(root.getRelative(".bazelignore"), "pkg/foo/bar");
    assertGlobMatches("foo/**", "foo/barnacle/wiz", "foo/barnacle", "foo");
    differencer.invalidate(
        ImmutableList.of(
            FileStateValue.key(
                RootedPath.toRootedPath(
                    Root.fromPath(root), PathFragment.create(".bazelignore")))));

    FileSystemUtils.createEmptyFile(root.getRelative(".bazelignore"));
    assertGlobMatches(
        "foo/**",
        "foo/bar/wiz",
        "foo/bar/wiz/file",
        "foo/bar",
        "foo/barnacle/wiz",
        "foo/barnacle",
        "foo");
  }

  @Test
  public void testStartsWithStar() throws Exception {
    assertGlobMatches("*oo", /* => */ "foo");
  }

  @Test
  public void testStartsWithStarWithMiddleStar() throws Exception {
    assertGlobMatches("*f*o", /* => */ "foo");
  }

  @Test
  public void testSingleMatchEqual() throws Exception {
    assertGlobsEqual("*oo", "*f*o"); // both produce "foo"
  }

  @Test
  public void testEndsWithStar() throws Exception {
    assertGlobMatches("foo*", /* => */ "foo", "food", "fool");
  }

  @Test
  public void testEndsWithStarWithMiddleStar() throws Exception {
    assertGlobMatches("f*oo*", /* => */ "foo", "food", "fool");
  }

  @Test
  public void testMultipleMatchesEqual() throws Exception {
    assertGlobsEqual("foo*", "f*oo*"); // both produce "foo", "food", "fool"
  }

  @Test
  public void testMiddleStar() throws Exception {
    assertGlobMatches("f*o", /* => */ "foo");
  }

  @Test
  public void testTwoMiddleStars() throws Exception {
    assertGlobMatches("f*o*o", /* => */ "foo");
  }

  @Test
  public void testSingleStarPatternWithNamedChild() throws Exception {
    assertGlobMatches("*/bar", /* => */ "foo/bar");
  }

  @Test
  public void testDeepSubpackages() throws Exception {
    assertGlobMatches("*/*/c", /* => */ "a1/b1/c");
  }

  @Test
  public void testSingleStarPatternWithChildGlob() throws Exception {
    assertGlobMatches(
        "*/bar*", /* => */ "foo/bar", "foo/barnacle", "food/barnacle", "fool/barnacle");
  }

  @Test
  public void testSingleStarAsChildGlob() throws Exception {
    assertGlobMatches("foo/*/wiz", /* => */ "foo/bar/wiz", "foo/barnacle/wiz");
  }

  @Test
  public void testNoAsteriskAndFilesDontExist() throws Exception {
    // Note un-UNIX like semantics:
    assertGlobMatches("ceci/n'est/pas/une/globbe" /* => nothing */);
  }

  @Test
  public void testSingleAsteriskUnderNonexistentDirectory() throws Exception {
    // Note un-UNIX like semantics:
    assertGlobMatches("not-there/*" /* => nothing */);
  }

  @Test
  public void testDifferentGlobsSameResultEqual() throws Exception {
    // Once the globs are run, it doesn't matter what pattern ran; only the output.
    assertGlobsEqual("not-there/*", "syzygy/*"); // Both produce nothing.
  }

  @Test
  public void testGlobUnderFile() throws Exception {
    assertGlobMatches("foo/bar/wiz/file/*" /* => nothing */);
  }

  @Test
  public void testGlobEqualsHashCode() throws Exception {
    // Each "equality group" forms a set of elements that are all equals() to one another,
    // and also produce the same hashCode.
    new EqualsTester()
        .addEqualityGroup(
            runGlob("no-such-file", Globber.Operation.FILES_AND_DIRS)) // Matches nothing.
        .addEqualityGroup(
            runGlob("BUILD", Globber.Operation.FILES_AND_DIRS),
            runGlob("BUILD", Globber.Operation.FILES)) // Matches BUILD.
        .addEqualityGroup(
            runGlob("**", Globber.Operation.FILES_AND_DIRS)) // Matches lots of things.
        .addEqualityGroup(
            runGlob("f*o/bar*", Globber.Operation.FILES_AND_DIRS),
            runGlob(
                "foo/bar*", Globber.Operation.FILES_AND_DIRS)) // Matches foo/bar and foo/barnacle.
        .testEquals();
  }

  @Test
  public void testGlobDoesNotCrossPackageBoundary() throws Exception {
    FileSystemUtils.createEmptyFile(pkgPath.getRelative("foo/BUILD"));
    // "foo/bar" should not be in the results because foo is a separate package.
    assertGlobMatches("f*/*", /* => */ "food/barnacle", "fool/barnacle");
  }

  @Test
  public void testGlobDirectoryMatchDoesNotCrossPackageBoundary() throws Exception {
    FileSystemUtils.createEmptyFile(pkgPath.getRelative("foo/bar/BUILD"));
    // "foo/bar" should not be in the results because foo/bar is a separate package.
    assertGlobMatches("foo/*", /* => */ "foo/barnacle");
  }

  @Test
  public void testStarStarDoesNotCrossPackageBoundary() throws Exception {
    FileSystemUtils.createEmptyFile(pkgPath.getRelative("foo/bar/BUILD"));
    // "foo/bar" should not be in the results because foo/bar is a separate package.
    assertGlobMatches("foo/**", /* => */ "foo/barnacle/wiz", "foo/barnacle", "foo");
  }

  @Test
  public void testGlobDoesNotCrossPackageBoundaryUnderOtherPackagePath() throws Exception {
    writableRoot.getRelative("pkg/foo/bar").createDirectoryAndParents();
    FileSystemUtils.createEmptyFile(writableRoot.getRelative("pkg/foo/bar/BUILD"));
    // "foo/bar" should not be in the results because foo/bar is detected as a separate package,
    // even though it is under a different package path.
    assertGlobMatches("foo/**", /* => */ "foo/barnacle/wiz", "foo/barnacle", "foo");
  }

  @Test
  public void testGlobDoesNotCrossRepositoryBoundary() throws Exception {
    FileSystemUtils.appendIsoLatin1(
        root.getRelative("WORKSPACE"), "local_repository(name='local', path='pkg/foo')");
    FileSystemUtils.createEmptyFile(pkgPath.getRelative("foo/WORKSPACE"));
    FileSystemUtils.createEmptyFile(pkgPath.getRelative("foo/BUILD"));
    // "foo/bar" should not be in the results because foo is a separate repository.
    assertGlobMatches("f*/*", /* => */ "food/barnacle", "fool/barnacle");
  }

  @Test
  public void testGlobDirectoryMatchDoesNotCrossRepositoryBoundary() throws Exception {
    FileSystemUtils.appendIsoLatin1(
        root.getRelative("WORKSPACE"), "local_repository(name='local', path='pkg/foo/bar')");
    FileSystemUtils.createEmptyFile(pkgPath.getRelative("foo/bar/WORKSPACE"));
    FileSystemUtils.createEmptyFile(pkgPath.getRelative("foo/bar/BUILD"));
    // "foo/bar" should not be in the results because foo/bar is a separate repository.
    assertGlobMatches("foo/*", /* => */ "foo/barnacle");
  }

  @Test
  public void testStarStarDoesNotCrossRepositoryBoundary() throws Exception {
    FileSystemUtils.appendIsoLatin1(
        root.getRelative("WORKSPACE"), "local_repository(name='local', path='pkg/foo/bar')");
    FileSystemUtils.createEmptyFile(pkgPath.getRelative("foo/bar/WORKSPACE"));
    FileSystemUtils.createEmptyFile(pkgPath.getRelative("foo/bar/BUILD"));
    // "foo/bar" should not be in the results because foo/bar is a separate repository.
    assertGlobMatches("foo/**", /* => */ "foo/barnacle/wiz", "foo/barnacle", "foo");
  }

  @Test
  public void testGlobDoesNotCrossRepositoryBoundaryUnderOtherPackagePath() throws Exception {
    FileSystemUtils.appendIsoLatin1(
        root.getRelative("WORKSPACE"),
        "local_repository(name='local', path='"
            + writableRoot.getRelative("pkg/foo/bar").getPathString()
            + "')");
    writableRoot.getRelative("pkg/foo/bar").createDirectoryAndParents();
    FileSystemUtils.createEmptyFile(writableRoot.getRelative("pkg/foo/bar/WORKSPACE"));
    FileSystemUtils.createEmptyFile(writableRoot.getRelative("pkg/foo/bar/BUILD"));
    // "foo/bar" should not be in the results because foo/bar is detected as a separate package,
    // even though it is under a different package path.
    assertGlobMatches("foo/**", /* => */ "foo/barnacle/wiz", "foo/barnacle", "foo");
  }

  private void assertGlobMatches(String pattern, String... expecteds) throws Exception {
    assertGlobMatches(pattern, Globber.Operation.FILES_AND_DIRS, expecteds);
  }

  private void assertGlobMatches(
      String pattern, Globber.Operation globberOperation, String... expecteds) throws Exception {
    // The order requirement is not strictly necessary -- a change to GlobFunction semantics that
    // changes the output order is fine, but we require that the order be the same here to detect
    // potential non-determinism in output order, which would be bad.
    // The current order in the case of "**" or "*" is roughly that of nestedset.Order.STABLE_ORDER,
    // putting subdirectories before directories, but putting ordinary files after their parent
    // directories.
    assertThat(
            Iterables.transform(
                runGlob(pattern, globberOperation).getMatches().toList(),
                Functions.toStringFunction()))
        .containsExactlyElementsIn(ImmutableList.copyOf(expecteds))
        .inOrder();
  }

  private void assertGlobWithoutDirsMatches(String pattern, String... expecteds) throws Exception {
    assertGlobMatches(pattern, Globber.Operation.FILES, expecteds);
  }

  private void assertGlobsEqual(String pattern1, String pattern2) throws Exception {
    GlobValue value1 = runGlob(pattern1, Globber.Operation.FILES_AND_DIRS);
    GlobValue value2 = runGlob(pattern2, Globber.Operation.FILES_AND_DIRS);
    new EqualsTester()
        .addEqualityGroup(value1, value2)
        .testEquals();
  }

  private GlobValue runGlob(String pattern, Globber.Operation globberOperation) throws Exception {
    SkyKey skyKey =
        GlobValue.key(
            PKG_ID, Root.fromPath(root), pattern, globberOperation, PathFragment.EMPTY_FRAGMENT);
    EvaluationResult<SkyValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), EVALUATION_OPTIONS);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    return (GlobValue) result.get(skyKey);
  }

  @Test
  public void testGlobWithoutWildcards() throws Exception {
    String pattern = "foo/bar/wiz/file";

    assertGlobMatches(pattern, "foo/bar/wiz/file");
    // Ensure that the glob depends on the FileValue and not on the DirectoryListingValue.
    pkgPath.getRelative("foo/bar/wiz/file").delete();
    // Nothing has been invalidated yet, so the cached result is returned.
    assertGlobMatches(pattern, "foo/bar/wiz/file");

    if (alwaysUseDirListing()) {
      differencer.invalidate(
          ImmutableList.of(
              FileStateValue.key(
                  RootedPath.toRootedPath(
                      Root.fromPath(root), pkgPath.getRelative("foo/bar/wiz/file")))));
      // The result should not rely on the FileStateValue, so it's still a cache hit.
      assertGlobMatches(pattern, "foo/bar/wiz/file");

      differencer.invalidate(
          ImmutableList.of(
              DirectoryListingStateValue.key(
                  RootedPath.toRootedPath(
                      Root.fromPath(root), pkgPath.getRelative("foo/bar/wiz")))));
    } else {
      differencer.invalidate(
          ImmutableList.of(
              DirectoryListingStateValue.key(
                  RootedPath.toRootedPath(
                      Root.fromPath(root), pkgPath.getRelative("foo/bar/wiz")))));
      // The result should not rely on the DirectoryListingValue, so it's still a cache hit.
      assertGlobMatches(pattern, "foo/bar/wiz/file");

      differencer.invalidate(
          ImmutableList.of(
              FileStateValue.key(
                  RootedPath.toRootedPath(
                      Root.fromPath(root), pkgPath.getRelative("foo/bar/wiz/file")))));
    }
    // This should have invalidated the glob result.
    assertGlobMatches(pattern /* => nothing */);
  }

  @Test
  public void testIllegalPatterns() throws Exception {
    assertIllegalPattern("foo**bar");
    assertIllegalPattern("?");
    assertIllegalPattern("");
    assertIllegalPattern(".");
    assertIllegalPattern("/foo");
    assertIllegalPattern("./foo");
    assertIllegalPattern("foo/");
    assertIllegalPattern("foo/./bar");
    assertIllegalPattern("../foo/bar");
    assertIllegalPattern("foo//bar");
  }

  @Test
  public void testIllegalRecursivePatterns() throws Exception {
    for (String prefix : Lists.newArrayList("", "*/", "**/", "ba/")) {
      String suffix = ("/" + prefix).substring(0, prefix.length());
      for (String pattern : Lists.newArrayList("**fo", "fo**", "**fo**", "fo**fo", "fo**fo**fo")) {
        assertIllegalPattern(prefix + pattern);
        assertIllegalPattern(pattern + suffix);
      }
    }
  }

  private void assertIllegalPattern(String pattern) {
    assertThrows(
        "invalid pattern not detected: " + pattern,
        InvalidGlobPatternException.class,
        () ->
            GlobValue.key(
                PKG_ID,
                Root.fromPath(root),
                pattern,
                Globber.Operation.FILES_AND_DIRS,
                PathFragment.EMPTY_FRAGMENT));
  }

  /**
   * Tests that globs can contain Java regular expression special characters
   */
  @Test
  public void testSpecialRegexCharacter() throws Exception {
    Path aDotB = pkgPath.getChild("a.b");
    FileSystemUtils.createEmptyFile(aDotB);
    FileSystemUtils.createEmptyFile(pkgPath.getChild("aab"));
    // Note: this contains two asterisks because otherwise a RE is not built,
    // as an optimization.
    assertThat(
            new UnixGlob.Builder(pkgPath, SyscallCache.NO_CACHE)
                .addPattern("*a.b*")
                .globInterruptible())
        .containsExactly(aDotB);
  }

  @Test
  public void testMatchesCallWithNoCache() {
    assertThat(UnixGlob.matches("*a*b", "CaCb", null)).isTrue();
  }

  @Test
  public void testHiddenFiles() throws Exception {
    for (String dir : ImmutableList.of(".hidden", "..also.hidden", "not.hidden")) {
      pkgPath.getRelative(dir).createDirectoryAndParents();
    }
    // Note that these are not in the result: ".", ".."
    assertGlobMatches(
        "*", "..also.hidden", ".hidden", "BUILD", "a1", "a2", "foo", "food", "fool", "not.hidden");
    assertGlobMatches("*.hidden", "not.hidden");
  }

  @Test
  public void testDoubleStar() throws Exception {
    assertGlobMatches(
        "**",
        "a1/b1/c",
        "a1/b1",
        "a1",
        "a2",
        "foo/bar/wiz",
        "foo/bar/wiz/file",
        "foo/bar",
        "foo/barnacle/wiz",
        "foo/barnacle",
        "foo",
        "food/barnacle/wiz",
        "food/barnacle",
        "food",
        "fool/barnacle/wiz",
        "fool/barnacle",
        "fool",
        "BUILD");
  }

  @Test
  public void testDoubleStarExcludeDirs() throws Exception {
    assertGlobWithoutDirsMatches("**", "foo/bar/wiz/file", "BUILD");
  }

  @Test
  public void testDoubleDoubleStar() throws Exception {
    assertGlobMatches(
        "**/**",
        "a1/b1/c",
        "a1/b1",
        "a1",
        "a2",
        "foo/bar/wiz",
        "foo/bar/wiz/file",
        "foo/bar",
        "foo/barnacle/wiz",
        "foo/barnacle",
        "foo",
        "food/barnacle/wiz",
        "food/barnacle",
        "food",
        "fool/barnacle/wiz",
        "fool/barnacle",
        "fool",
        "BUILD");
  }

  @Test
  public void testDirectoryWithDoubleStar() throws Exception {
    assertGlobMatches(
        "foo/**",
        "foo/bar/wiz",
        "foo/bar/wiz/file",
        "foo/bar",
        "foo/barnacle/wiz",
        "foo/barnacle",
        "foo");
  }

  @Test
  public void testDoubleStarPatternWithNamedChild() throws Exception {
    assertGlobMatches("**/bar", "foo/bar");
  }

  @Test
  public void testDoubleStarPatternWithErrorChild() throws Exception {
    FileSystemUtils.ensureSymbolicLink(pkgPath.getChild("self"), "self");

    IOException ioException =
        assertThrows(IOException.class, () -> runGlob("**/self", Operation.FILES));
    assertThat(ioException).hasMessageThat().matches("Symlink cycle");
  }

  @Test
  public void testDoubleStarPatternWithChildGlob() throws Exception {
    assertGlobMatches("**/ba*", "foo/bar", "foo/barnacle", "food/barnacle", "fool/barnacle");
  }

  @Test
  public void testDoubleStarAsChildGlob() throws Exception {
    FileSystemUtils.createEmptyFile(pkgPath.getRelative("foo/barnacle/wiz/wiz"));
    pkgPath.getRelative("foo/barnacle/baz/wiz").createDirectoryAndParents();

    assertGlobMatches(
        "foo/**/wiz",
        "foo/bar/wiz",
        "foo/barnacle/wiz",
        "foo/barnacle/baz/wiz",
        "foo/barnacle/wiz/wiz");
  }

  @Test
  public void testDoubleStarUnderNonexistentDirectory() throws Exception {
    assertGlobMatches("not-there/**" /* => nothing */);
  }

  @Test
  public void testDoubleStarUnderFile() throws Exception {
    assertGlobMatches("foo/bar/wiz/file/**" /* => nothing */);
  }

  /** Regression test for b/225434889: Value with exception will not crash. */
  @Test
  public void symlinkFileValueWithError() throws Exception {
    FileSystemUtils.ensureSymbolicLink(pkgPath.getChild("self"), "self");

    IOException ioException =
        assertThrows(IOException.class, () -> runGlob("self", Operation.FILES_AND_DIRS));
    assertThat(ioException).hasMessageThat().matches("Symlink cycle");
  }

  @Test
  public void symlinkSubdirValueWithError() throws Exception {
    Path cycle = pkgPath.getChild("cycle");
    FileSystemUtils.ensureSymbolicLink(cycle.getChild("self"), "self");
    FileSystemUtils.ensureSymbolicLink(pkgPath.getChild("symlink"), cycle);

    IOException ioException =
        assertThrows(IOException.class, () -> runGlob("symlink/self", Operation.FILES_AND_DIRS));
    assertThat(ioException).hasMessageThat().matches("Symlink cycle");
  }

  /** Regression test for b/13319874: Directory listing crash. */
  @Test
  public void testResilienceToFilesystemInconsistencies_directoryExistence() throws Exception {
    // Our custom filesystem says "pkgPath/BUILD" exists but "pkgPath" does not exist.
    fs.stubStat(pkgPath, null);
    RootedPath pkgRootedPath = RootedPath.toRootedPath(Root.fromPath(root), pkgPath);
    FileStateValue pkgDirFileStateValue =
        FileStateValue.create(pkgRootedPath, SyscallCache.NO_CACHE, /*tsgm=*/ null);
    FileValue pkgDirValue =
        FileValue.value(
            ImmutableList.of(pkgRootedPath),
            null,
            null,
            pkgRootedPath,
            pkgDirFileStateValue,
            pkgRootedPath,
            pkgDirFileStateValue);
    differencer.inject(ImmutableMap.of(FileValue.key(pkgRootedPath), Delta.justNew(pkgDirValue)));
    String expectedMessage = "/root/workspace/pkg is no longer an existing directory";
    SkyKey skyKey =
        GlobValue.key(
            PKG_ID,
            Root.fromPath(root),
            "*/foo",
            Globber.Operation.FILES_AND_DIRS,
            PathFragment.EMPTY_FRAGMENT);
    EvaluationResult<GlobValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), EVALUATION_OPTIONS);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skyKey);
    assertThat(errorInfo.getException()).isInstanceOf(InconsistentFilesystemException.class);
    assertThat(errorInfo.getException()).hasMessageThat().contains(expectedMessage);
  }

  @Test
  public void testResilienceToFilesystemInconsistencies_subdirectoryExistence() throws Exception {
    // Our custom filesystem says directory "pkgPath/foo/bar" contains a subdirectory "wiz" but a
    // direct stat on "pkgPath/foo/bar/wiz" says it does not exist.
    Path fooBarDir = pkgPath.getRelative("foo/bar");
    fs.stubStat(fooBarDir.getRelative("wiz"), null);
    RootedPath fooBarDirRootedPath = RootedPath.toRootedPath(Root.fromPath(root), fooBarDir);
    SkyValue fooBarDirListingValue =
        DirectoryListingStateValue.create(
            ImmutableList.of(new Dirent("wiz", Dirent.Type.DIRECTORY)));
    differencer.inject(
        ImmutableMap.of(
            DirectoryListingStateValue.key(fooBarDirRootedPath),
            Delta.justNew(fooBarDirListingValue)));
    String expectedMessage = "/root/workspace/pkg/foo/bar/wiz is no longer an existing directory.";
    SkyKey skyKey =
        GlobValue.key(
            PKG_ID,
            Root.fromPath(root),
            "**/wiz",
            Globber.Operation.FILES_AND_DIRS,
            PathFragment.EMPTY_FRAGMENT);
    EvaluationResult<GlobValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), EVALUATION_OPTIONS);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skyKey);
    assertThat(errorInfo.getException()).isInstanceOf(InconsistentFilesystemException.class);
    assertThat(errorInfo.getException()).hasMessageThat().contains(expectedMessage);
  }

  @Test
  public void testResilienceToFilesystemInconsistencies_symlinkType() throws Exception {
    RootedPath wizRootedPath =
        RootedPath.toRootedPath(Root.fromPath(root), pkgPath.getRelative("foo/bar/wiz"));
    RootedPath fileRootedPath =
        RootedPath.toRootedPath(Root.fromPath(root), pkgPath.getRelative("foo/bar/wiz/file"));
    final FileStatus realStat = fileRootedPath.asPath().stat();
    fs.stubStat(
        fileRootedPath.asPath(),
        new FileStatus() {

          @Override
          public boolean isFile() {
            // The stat says foo/bar/wiz/file is a real file, not a symlink.
            return true;
          }

          @Override
          public boolean isSpecialFile() {
            return false;
          }

          @Override
          public boolean isDirectory() {
            return false;
          }

          @Override
          public boolean isSymbolicLink() {
            return false;
          }

          @Override
          public long getSize() throws IOException {
            return realStat.getSize();
          }

          @Override
          public long getLastModifiedTime() throws IOException {
            return realStat.getLastModifiedTime();
          }

          @Override
          public long getLastChangeTime() throws IOException {
            return realStat.getLastChangeTime();
          }

          @Override
          public long getNodeId() throws IOException {
            return realStat.getNodeId();
          }
        });
    // But the dir listing say foo/bar/wiz/file is a symlink.
    SkyValue wizDirListingValue =
        DirectoryListingStateValue.create(
            ImmutableList.of(new Dirent("file", Dirent.Type.SYMLINK)));
    differencer.inject(
        ImmutableMap.of(
            DirectoryListingStateValue.key(wizRootedPath), Delta.justNew(wizDirListingValue)));
    String expectedMessage =
        "readdir and stat disagree about whether " + fileRootedPath.asPath() + " is a symlink";
    SkyKey skyKey =
        GlobValue.key(
            PKG_ID,
            Root.fromPath(root),
            "foo/bar/wiz/*",
            Globber.Operation.FILES_AND_DIRS,
            PathFragment.EMPTY_FRAGMENT);
    EvaluationResult<GlobValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), EVALUATION_OPTIONS);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skyKey);
    assertThat(errorInfo.getException()).isInstanceOf(InconsistentFilesystemException.class);
    assertThat(errorInfo.getException()).hasMessageThat().contains(expectedMessage);
  }

  @Test
  public void testSymlinks() throws Exception {
    pkgPath.getRelative("symlinks").createDirectoryAndParents();
    FileSystemUtils.ensureSymbolicLink(pkgPath.getRelative("symlinks/dangling.txt"), "nope");
    FileSystemUtils.createEmptyFile(pkgPath.getRelative("symlinks/yup"));
    FileSystemUtils.ensureSymbolicLink(pkgPath.getRelative("symlinks/existing.txt"), "yup");
    assertGlobMatches("symlinks/*.txt", "symlinks/existing.txt");
  }

  private static final class CustomInMemoryFs extends InMemoryFileSystem {
    private final Map<PathFragment, FileStatus> stubbedStats = Maps.newHashMap();

    CustomInMemoryFs(ManualClock manualClock) {
      super(manualClock, DigestHashFunction.SHA256);
    }

    public void stubStat(Path path, @Nullable FileStatus stubbedResult) {
      stubbedStats.put(path.asFragment(), stubbedResult);
    }

    @Override
    public FileStatus statIfFound(PathFragment path, boolean followSymlinks) throws IOException {
      if (stubbedStats.containsKey(path)) {
        return stubbedStats.get(path);
      }
      return super.statIfFound(path, followSymlinks);
    }
  }

  private void assertSubpackageMatches(String pattern, String... expecteds) throws Exception {
    assertThat(
            Iterables.transform(
                runGlob(pattern, Globber.Operation.SUBPACKAGES).getMatches().toList(),
                Functions.toStringFunction()))
        .containsExactlyElementsIn(ImmutableList.copyOf(expecteds));
  }

  private void makeEmptyPackage(Path newPackagePath) throws Exception {
    newPackagePath.createDirectoryAndParents();
    FileSystemUtils.createEmptyFile(newPackagePath.getRelative("BUILD"));
  }

  private void makeEmptyPackage(String path) throws Exception {
    makeEmptyPackage(pkgPath.getRelative(path));
  }

  @Test
  public void subpackages_simple() throws Exception {
    makeEmptyPackage("horse");
    makeEmptyPackage("monkey");
    makeEmptyPackage("horse/saddle");

    // "horse/saddle" should not be in the results because horse/saddle is too deep. a2/b2 added by
    // setup().
    assertSubpackageMatches("**", /* => */ "a2/b2", "horse", "monkey");
  }

  @Test
  public void subpackages_empty() throws Exception {
    assertSubpackageMatches("foo/*");
    assertSubpackageMatches("foo/**");
  }

  @Test
  public void subpackages_oneLevelDeep() throws Exception {
    makeEmptyPackage("base/sub");
    makeEmptyPackage("base/sub2");
    makeEmptyPackage("base/sub3");

    assertSubpackageMatches("base/*", /* => */ "base/sub", "base/sub2", "base/sub3");
    assertSubpackageMatches("base/**", /* => */ "base/sub", "base/sub2", "base/sub3");
  }

  @Test
  public void subpackages_oneLevel_notDeepEnough() throws Exception {
    makeEmptyPackage("base/sub/pkg");
    makeEmptyPackage("base/sub2/pkg");
    makeEmptyPackage("base/sub3/pkg");

    // * doesn't go deep enough
    assertSubpackageMatches("base/*");
    // But if we go with ** it works fine.
    assertSubpackageMatches("base/**", /* => */ "base/sub/pkg", "base/sub2/pkg", "base/sub3/pkg");
  }

  @Test
  public void subpackages_deepRecurse() throws Exception {
    makeEmptyPackage("base/sub/1");
    makeEmptyPackage("base/sub/2");
    makeEmptyPackage("base/sub2/3");
    makeEmptyPackage("base/sub2/4");
    makeEmptyPackage("base/sub3/5");
    makeEmptyPackage("base/sub3/6");

    FileSystemUtils.createEmptyFile(pkgPath.getRelative("foo/bar/BUILD"));
    // "foo/bar" should not be in the results because foo/bar is a separate package.
    assertSubpackageMatches(
        "base/*/*",
        "base/sub/1",
        "base/sub/2",
        "base/sub2/3",
        "base/sub2/4",
        "base/sub3/5",
        "base/sub3/6");

    assertSubpackageMatches(
        "base/**",
        "base/sub/1",
        "base/sub/2",
        "base/sub2/3",
        "base/sub2/4",
        "base/sub3/5",
        "base/sub3/6");
  }

  @Test
  public void subpackages_middleWidlcard() throws Exception {
    makeEmptyPackage("base/sub1/same");
    makeEmptyPackage("base/sub2/same");
    makeEmptyPackage("base/sub3/same");
    makeEmptyPackage("base/sub4/same");
    makeEmptyPackage("base/sub5/same");
    makeEmptyPackage("base/sub6/same");

    assertSubpackageMatches(
        "base/*/same",
        "base/sub1/same",
        "base/sub2/same",
        "base/sub3/same",
        "base/sub4/same",
        "base/sub5/same",
        "base/sub6/same");

    assertSubpackageMatches(
        "base/**/same",
        "base/sub1/same",
        "base/sub2/same",
        "base/sub3/same",
        "base/sub4/same",
        "base/sub5/same",
        "base/sub6/same");
  }

  @Test
  public void subpackages_noWildcard() throws Exception {
    makeEmptyPackage("sub1");
    makeEmptyPackage("sub2");
    makeEmptyPackage("sub3/deep");
    makeEmptyPackage("sub4/deeper/deeper");

    assertSubpackageMatches("sub");
    assertSubpackageMatches("sub1", "sub1");
    assertSubpackageMatches("sub2", "sub2");
    assertSubpackageMatches("sub3/deep", "sub3/deep");
    assertSubpackageMatches("sub4/deeper/deeper", "sub4/deeper/deeper");
  }

  @Test
  public void subpackages_testSymlinks() throws Exception {
    Path newPackagePath = pkgPath.getRelative("path/to/pkg");
    makeEmptyPackage(newPackagePath);

    pkgPath.getRelative("symlinks").createDirectoryAndParents();
    FileSystemUtils.ensureSymbolicLink(pkgPath.getRelative("symlinks/deeplink"), newPackagePath);
    FileSystemUtils.ensureSymbolicLink(pkgPath.getRelative("shallowlink"), newPackagePath);

    assertSubpackageMatches("**", "a2/b2", "symlinks/deeplink", "path/to/pkg", "shallowlink");
    assertSubpackageMatches("*", "shallowlink");

    assertSubpackageMatches("symlinks/**", "symlinks/deeplink");
    assertSubpackageMatches("symlinks/*", "symlinks/deeplink");
  }
}
