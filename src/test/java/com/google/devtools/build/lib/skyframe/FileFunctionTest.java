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
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.skyframe.SkyframeExecutor.DEFAULT_THREAD_COUNT;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;
import static org.junit.Assert.fail;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import com.google.devtools.build.skyframe.BuildDriver;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.ErrorInfoSubject;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FileFunction}. */
@RunWith(JUnit4.class)
public class FileFunctionTest {
  private CustomInMemoryFs fs;
  private Path pkgRoot;
  private Path outputBase;
  private PathPackageLocator pkgLocator;
  private boolean fastDigest;
  private ManualClock manualClock;
  private RecordingDifferencer differencer;

  @Before
  public final void createFsAndRoot() throws Exception {
    fastDigest = true;
    manualClock = new ManualClock();
    createFsAndRoot(new CustomInMemoryFs(manualClock));
  }

  private void createFsAndRoot(CustomInMemoryFs fs) throws IOException {
    this.fs = fs;
    pkgRoot = fs.getRootDirectory().getRelative("root");
    outputBase = fs.getRootDirectory().getRelative("output_base");
    pkgLocator =
        new PathPackageLocator(
            outputBase,
            ImmutableList.of(pkgRoot),
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY);
    FileSystemUtils.createDirectoryAndParents(pkgRoot);
  }

  private SequentialBuildDriver makeDriver() {
    return makeDriver(ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS);
  }

  private SequentialBuildDriver makeDriver(ExternalFileAction externalFileAction) {
    AtomicReference<PathPackageLocator> pkgLocatorRef = new AtomicReference<>(pkgLocator);
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(pkgRoot, outputBase), pkgRoot, TestConstants.PRODUCT_NAME);
    ExternalFilesHelper externalFilesHelper =
        new ExternalFilesHelper(pkgLocatorRef, externalFileAction, directories);
    differencer = new SequencedRecordingDifferencer();
    MemoizingEvaluator evaluator =
        new InMemoryMemoizingEvaluator(
            ImmutableMap.<SkyFunctionName, SkyFunction>builder()
                .put(
                    SkyFunctions.FILE_STATE,
                    new FileStateFunction(
                        new AtomicReference<TimestampGranularityMonitor>(), externalFilesHelper))
                .put(
                    SkyFunctions.FILE_SYMLINK_CYCLE_UNIQUENESS,
                    new FileSymlinkCycleUniquenessFunction())
                .put(
                    SkyFunctions.FILE_SYMLINK_INFINITE_EXPANSION_UNIQUENESS,
                    new FileSymlinkInfiniteExpansionUniquenessFunction())
                .put(SkyFunctions.FILE, new FileFunction(pkgLocatorRef))
                .put(
                    SkyFunctions.PACKAGE,
                    new PackageFunction(null, null, null, null, null, null, null))
                .put(
                    SkyFunctions.PACKAGE_LOOKUP,
                    new PackageLookupFunction(
                        new AtomicReference<>(ImmutableSet.<PackageIdentifier>of()),
                        CrossRepositoryLabelViolationStrategy.ERROR,
                        BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY))
                .put(
                    SkyFunctions.WORKSPACE_AST,
                    new WorkspaceASTFunction(TestRuleClassProvider.getRuleClassProvider()))
                .put(
                    SkyFunctions.WORKSPACE_FILE,
                    new WorkspaceFileFunction(
                        TestRuleClassProvider.getRuleClassProvider(),
                        TestConstants.PACKAGE_FACTORY_BUILDER_FACTORY_FOR_TESTING
                            .builder()
                            .build(TestRuleClassProvider.getRuleClassProvider(), fs),
                        directories))
                .put(SkyFunctions.EXTERNAL_PACKAGE, new ExternalPackageFunction())
                .put(SkyFunctions.LOCAL_REPOSITORY_LOOKUP, new LocalRepositoryLookupFunction())
                .build(),
            differencer);
    PrecomputedValue.BUILD_ID.set(differencer, UUID.randomUUID());
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator);
    RepositoryDelegatorFunction.REPOSITORY_OVERRIDES.set(
        differencer, ImmutableMap.<RepositoryName, PathFragment>of());
    PrecomputedValue.SKYLARK_SEMANTICS.set(differencer, SkylarkSemantics.DEFAULT_SEMANTICS);
    return new SequentialBuildDriver(evaluator);
  }

  private FileValue valueForPath(Path path) throws InterruptedException {
    return valueForPathHelper(pkgRoot, path);
  }

  private FileValue valueForPathOutsidePkgRoot(Path path) throws InterruptedException {
    return valueForPathHelper(fs.getRootDirectory(), path);
  }

  private FileValue valueForPathHelper(Path root, Path path) throws InterruptedException {
    PathFragment pathFragment = path.relativeTo(root);
    RootedPath rootedPath = RootedPath.toRootedPath(root, pathFragment);
    SequentialBuildDriver driver = makeDriver();
    SkyKey key = FileValue.key(rootedPath);
    EvaluationResult<FileValue> result =
        driver.evaluate(
            ImmutableList.of(key), false, DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);
    assertThat(result.hasError()).isFalse();
    return result.get(key);
  }

  @Test
  public void testFileValueHashCodeAndEqualsContract() throws Exception {
    Path pathA = file(pkgRoot + "a", "a");
    Path pathB = file(pkgRoot + "b", "b");
    FileValue valueA1 = valueForPathOutsidePkgRoot(pathA);
    FileValue valueA2 = valueForPathOutsidePkgRoot(pathA);
    FileValue valueB1 = valueForPathOutsidePkgRoot(pathB);
    FileValue valueB2 = valueForPathOutsidePkgRoot(pathB);
    new EqualsTester()
        .addEqualityGroup(valueA1, valueA2)
        .addEqualityGroup(valueB1, valueB2)
        .testEquals();
  }

  @Test
  public void testIsDirectory() throws Exception {
    assertThat(valueForPath(file("a")).isDirectory()).isFalse();
    assertThat(valueForPath(path("nonexistent")).isDirectory()).isFalse();
    assertThat(valueForPath(directory("dir")).isDirectory()).isTrue();

    assertThat(valueForPath(symlink("sa", "a")).isDirectory()).isFalse();
    assertThat(valueForPath(symlink("smissing", "missing")).isDirectory()).isFalse();
    assertThat(valueForPath(symlink("sdir", "dir")).isDirectory()).isTrue();
    assertThat(valueForPath(symlink("ssdir", "sdir")).isDirectory()).isTrue();
  }

  @Test
  public void testIsFile() throws Exception {
    assertThat(valueForPath(file("a")).isFile()).isTrue();
    assertThat(valueForPath(path("nonexistent")).isFile()).isFalse();
    assertThat(valueForPath(directory("dir")).isFile()).isFalse();

    assertThat(valueForPath(symlink("sa", "a")).isFile()).isTrue();
    assertThat(valueForPath(symlink("smissing", "missing")).isFile()).isFalse();
    assertThat(valueForPath(symlink("sdir", "dir")).isFile()).isFalse();
    assertThat(valueForPath(symlink("ssfile", "sa")).isFile()).isTrue();
  }

  @Test
  public void testSimpleIndependentFiles() throws Exception {
    file("a");
    file("b");

    Set<RootedPath> seenFiles = Sets.newHashSet();
    seenFiles.addAll(getFilesSeenAndAssertValueChangesIfContentsOfFileChanges("a", false, "b"));
    seenFiles.addAll(getFilesSeenAndAssertValueChangesIfContentsOfFileChanges("b", false, "a"));
    assertThat(seenFiles).containsExactly(rootedPath("a"), rootedPath("b"), rootedPath(""));
  }

  @Test
  public void testSimpleSymlink() throws Exception {
    symlink("a", "b");
    file("b");

    assertValueChangesIfContentsOfFileChanges("a", false, "b");
    assertValueChangesIfContentsOfFileChanges("b", true, "a");
  }

  @Test
  public void testTransitiveSymlink() throws Exception {
    symlink("a", "b");
    symlink("b", "c");
    file("c");

    assertValueChangesIfContentsOfFileChanges("a", false, "b");
    assertValueChangesIfContentsOfFileChanges("a", false, "c");
    assertValueChangesIfContentsOfFileChanges("b", true, "a");
    assertValueChangesIfContentsOfFileChanges("c", true, "b");
    assertValueChangesIfContentsOfFileChanges("c", true, "a");
  }

  @Test
  public void testFileUnderDirectorySymlink() throws Exception {
    symlink("a", "b/c");
    symlink("b", "d");
    assertValueChangesIfContentsOfDirectoryChanges("b", true, "a/e");
  }

  @Test
  public void testSymlinkInDirectory() throws Exception {
    symlink("a/aa", "ab");
    file("a/ab");

    assertValueChangesIfContentsOfFileChanges("a/aa", false, "a/ab");
    assertValueChangesIfContentsOfFileChanges("a/ab", true, "a/aa");
  }

  @Test
  public void testRelativeSymlink() throws Exception {
    symlink("a/aa/aaa", "../ab/aba");
    file("a/ab/aba");
    assertValueChangesIfContentsOfFileChanges("a/ab/aba", true, "a/aa/aaa");
  }

  @Test
  public void testDoubleRelativeSymlink() throws Exception {
    symlink("a/b/c/d", "../../e/f");
    file("a/e/f");
    assertValueChangesIfContentsOfFileChanges("a/e/f", true, "a/b/c/d");
  }

  @Test
  public void testExternalRelativeSymlink() throws Exception {
    symlink("a", "../outside");
    file("b");
    file("../outside");
    Set<RootedPath> seenFiles = Sets.newHashSet();
    seenFiles.addAll(getFilesSeenAndAssertValueChangesIfContentsOfFileChanges("b", false, "a"));
    seenFiles.addAll(
        getFilesSeenAndAssertValueChangesIfContentsOfFileChanges("../outside", true, "a"));
    assertThat(seenFiles)
        .containsExactly(
            rootedPath("a"),
            rootedPath(""),
            RootedPath.toRootedPath(fs.getRootDirectory(), PathFragment.EMPTY_FRAGMENT),
            RootedPath.toRootedPath(fs.getRootDirectory(), PathFragment.create("outside")));
  }

  @Test
  public void testAbsoluteSymlink() throws Exception {
    symlink("a", "/absolute");
    file("b");
    file("/absolute");
    Set<RootedPath> seenFiles = Sets.newHashSet();
    seenFiles.addAll(getFilesSeenAndAssertValueChangesIfContentsOfFileChanges("b", false, "a"));
    seenFiles.addAll(
        getFilesSeenAndAssertValueChangesIfContentsOfFileChanges("/absolute", true, "a"));
    assertThat(seenFiles)
        .containsExactly(
            rootedPath("a"),
            rootedPath(""),
            RootedPath.toRootedPath(fs.getRootDirectory(), PathFragment.EMPTY_FRAGMENT),
            RootedPath.toRootedPath(fs.getRootDirectory(), PathFragment.create("absolute")));
  }

  @Test
  public void testAbsoluteSymlinkToExternal() throws Exception {
    String externalPath =
        outputBase.getRelative(Label.EXTERNAL_PACKAGE_NAME).getRelative("a/b").getPathString();
    symlink("a", externalPath);
    file("b");
    file(externalPath);
    Set<RootedPath> seenFiles = Sets.newHashSet();
    seenFiles.addAll(getFilesSeenAndAssertValueChangesIfContentsOfFileChanges("b", false, "a"));
    seenFiles.addAll(
        getFilesSeenAndAssertValueChangesIfContentsOfFileChanges(externalPath, true, "a"));
    Path root = fs.getRootDirectory();
    assertThat(seenFiles)
        .containsExactly(
            rootedPath("WORKSPACE"),
            rootedPath("a"),
            rootedPath(""),
            RootedPath.toRootedPath(root, PathFragment.EMPTY_FRAGMENT),
            RootedPath.toRootedPath(root, PathFragment.create("output_base")),
            RootedPath.toRootedPath(root, PathFragment.create("output_base/external")),
            RootedPath.toRootedPath(root, PathFragment.create("output_base/external/a")),
            RootedPath.toRootedPath(root, PathFragment.create("output_base/external/a/b")));
  }

  @Test
  public void testSymlinkAsAncestor() throws Exception {
    file("a/b/c/d");
    symlink("f", "a/b/c");
    assertValueChangesIfContentsOfFileChanges("a/b/c/d", true, "f/d");
  }

  @Test
  public void testSymlinkAsAncestorNested() throws Exception {
    file("a/b/c/d");
    symlink("f", "a/b");
    assertValueChangesIfContentsOfFileChanges("a/b/c/d", true, "f/c/d");
  }

  @Test
  public void testTwoSymlinksInAncestors() throws Exception {
    file("a/aa/aaa/aaaa");
    symlink("b/ba/baa", "../../a/aa");
    symlink("c/ca", "../b/ba");

    assertValueChangesIfContentsOfFileChanges("c/ca", true, "c/ca/baa/aaa/aaaa");
    assertValueChangesIfContentsOfFileChanges("b/ba/baa", true, "c/ca/baa/aaa/aaaa");
    assertValueChangesIfContentsOfFileChanges("a/aa/aaa/aaaa", true, "c/ca/baa/aaa/aaaa");
  }

  @Test
  public void testSelfReferencingSymlink() throws Exception {
    symlink("a", "a");
    assertError("a");
  }

  @Test
  public void testMutuallyReferencingSymlinks() throws Exception {
    symlink("a", "b");
    symlink("b", "a");
    assertError("a");
  }

  @Test
  public void testRecursiveNestingSymlink() throws Exception {
    symlink("a/a", "../a");
    assertError("a/a/b");
  }

  @Test
  public void testBrokenSymlink() throws Exception {
    symlink("a", "b");
    Set<RootedPath> seenFiles = Sets.newHashSet();
    seenFiles.addAll(getFilesSeenAndAssertValueChangesIfContentsOfFileChanges("b", true, "a"));
    seenFiles.addAll(getFilesSeenAndAssertValueChangesIfContentsOfFileChanges("a", false, "b"));
    assertThat(seenFiles).containsExactly(rootedPath("a"), rootedPath("b"), rootedPath(""));
  }

  @Test
  public void testBrokenDirectorySymlink() throws Exception {
    symlink("a", "b");
    file("c");

    assertValueChangesIfContentsOfDirectoryChanges("a", true, "a/aa");
    // This just creates the directory "b", which doesn't change the value for "a/aa", since "a/aa"
    // still has real path "b/aa" and still doesn't exist.
    assertValueChangesIfContentsOfDirectoryChanges("b", false, "a/aa");
    assertValueChangesIfContentsOfFileChanges("c", false, "a/aa");
  }

  @Test
  public void testTraverseIntoVirtualNonDirectory() throws Exception {
    file("dir/a");
    symlink("vdir", "dir");
    // The following evaluation should not throw IOExceptions.
    assertNoError("vdir/a/aa/aaa");
  }

  @Test
  public void testFileCreation() throws Exception {
    FileValue a = valueForPath(path("file"));
    Path p = file("file");
    FileValue b = valueForPath(p);
    assertThat(a.equals(b)).isFalse();
  }

  @Test
  public void testEmptyFile() throws Exception {
    final byte[] digest = new byte[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    createFsAndRoot(
        new CustomInMemoryFs(manualClock) {
          @Override
          protected byte[] getFastDigest(Path path, HashFunction hf) throws IOException {
            return digest;
          }
        });
    Path p = file("file");
    p.setLastModifiedTime(0L);
    FileValue a = valueForPath(p);
    p.setLastModifiedTime(1L);
    assertThat(valueForPath(p)).isNotEqualTo(a);
    p.setLastModifiedTime(0L);
    assertThat(valueForPath(p)).isEqualTo(a);
    FileSystemUtils.writeContentAsLatin1(p, "content");
    // Same digest, but now non-empty.
    assertThat(valueForPath(p)).isNotEqualTo(a);
  }

  @Test
  public void testUnreadableFileWithNoFastDigest() throws Exception {
    Path p = file("unreadable");
    p.chmod(0);
    p.setLastModifiedTime(0L);

    FileValue value = valueForPath(p);
    assertThat(value.exists()).isTrue();
    assertThat(value.getDigest()).isNull();

    p.setLastModifiedTime(10L);
    assertThat(valueForPath(p)).isNotEqualTo(value);

    p.setLastModifiedTime(0L);
    assertThat(valueForPath(p)).isEqualTo(value);
  }

  @Test
  public void testUnreadableFileWithFastDigest() throws Exception {
    final byte[] expectedDigest =
        MessageDigest.getInstance("md5").digest("blah".getBytes(StandardCharsets.UTF_8));

    createFsAndRoot(
        new CustomInMemoryFs(manualClock) {
          @Override
          protected byte[] getFastDigest(Path path, HashFunction hf) {
            return path.getBaseName().equals("unreadable") ? expectedDigest : null;
          }
        });

    Path p = file("unreadable");
    p.chmod(0);

    FileValue value = valueForPath(p);
    assertThat(value.exists()).isTrue();
    assertThat(value.getDigest()).isNotNull();
  }

  @Test
  public void testFileModificationModTime() throws Exception {
    fastDigest = false;
    Path p = file("file");
    FileValue a = valueForPath(p);
    p.setLastModifiedTime(42);
    FileValue b = valueForPath(p);
    assertThat(a.equals(b)).isFalse();
  }

  @Test
  public void testFileModificationDigest() throws Exception {
    fastDigest = true;
    Path p = file("file");
    FileValue a = valueForPath(p);
    FileSystemUtils.writeContentAsLatin1(p, "goop");
    FileValue b = valueForPath(p);
    assertThat(a.equals(b)).isFalse();
  }

  @Test
  public void testModTimeVsDigest() throws Exception {
    Path p = file("somefile", "fizzley");

    fastDigest = true;
    FileValue aMd5 = valueForPath(p);
    fastDigest = false;
    FileValue aModTime = valueForPath(p);
    assertThat(aModTime).isNotEqualTo(aMd5);
    new EqualsTester().addEqualityGroup(aMd5).addEqualityGroup(aModTime).testEquals();
  }

  @Test
  public void testFileDeletion() throws Exception {
    Path p = file("file");
    FileValue a = valueForPath(p);
    p.delete();
    FileValue b = valueForPath(p);
    assertThat(a.equals(b)).isFalse();
  }

  @Test
  public void testFileTypeChange() throws Exception {
    Path p = file("file");
    FileValue a = valueForPath(p);
    p.delete();
    p = symlink("file", "foo");
    FileValue b = valueForPath(p);
    p.delete();
    FileSystemUtils.createDirectoryAndParents(pkgRoot.getRelative("file"));
    FileValue c = valueForPath(p);
    assertThat(a.equals(b)).isFalse();
    assertThat(b.equals(c)).isFalse();
    assertThat(a.equals(c)).isFalse();
  }

  @Test
  public void testSymlinkTargetChange() throws Exception {
    Path p = symlink("symlink", "foo");
    FileValue a = valueForPath(p);
    p.delete();
    p = symlink("symlink", "bar");
    FileValue b = valueForPath(p);
    assertThat(b).isNotEqualTo(a);
  }

  @Test
  public void testSymlinkTargetContentsChangeCTime() throws Exception {
    fastDigest = false;
    Path fooPath = file("foo");
    FileSystemUtils.writeContentAsLatin1(fooPath, "foo");
    Path p = symlink("symlink", "foo");
    FileValue a = valueForPath(p);
    manualClock.advanceMillis(1);
    fooPath.chmod(0555);
    manualClock.advanceMillis(1);
    FileValue b = valueForPath(p);
    assertThat(b).isNotEqualTo(a);
  }

  @Test
  public void testSymlinkTargetContentsChangeDigest() throws Exception {
    fastDigest = true;
    Path fooPath = file("foo");
    FileSystemUtils.writeContentAsLatin1(fooPath, "foo");
    Path p = symlink("symlink", "foo");
    FileValue a = valueForPath(p);
    FileSystemUtils.writeContentAsLatin1(fooPath, "bar");
    FileValue b = valueForPath(p);
    assertThat(b).isNotEqualTo(a);
  }

  @Test
  public void testRealPath() throws Exception {
    file("file");
    directory("directory");
    file("directory/file");
    symlink("directory/link", "file");
    symlink("directory/doublelink", "link");
    symlink("directory/parentlink", "../file");
    symlink("directory/doubleparentlink", "../link");
    symlink("link", "file");
    symlink("deadlink", "missing_file");
    symlink("dirlink", "directory");
    symlink("doublelink", "link");
    symlink("doubledirlink", "dirlink");

    checkRealPath("file");
    checkRealPath("link");
    checkRealPath("doublelink");

    for (String dir : new String[] {"directory", "dirlink", "doubledirlink"}) {
      checkRealPath(dir);
      checkRealPath(dir + "/file");
      checkRealPath(dir + "/link");
      checkRealPath(dir + "/doublelink");
      checkRealPath(dir + "/parentlink");
    }

    assertRealPath("missing", "missing");
    assertRealPath("deadlink", "missing_file");
  }

  @Test
  public void testRealPathRelativeSymlink() throws Exception {
    directory("dir");
    symlink("dir/link", "../dir2");
    directory("dir2");
    symlink("dir2/filelink", "../dest");
    file("dest");

    checkRealPath("dir/link/filelink");
  }

  @Test
  public void testSymlinkAcrossPackageRoots() throws Exception {
    Path otherPkgRoot = fs.getRootDirectory().getRelative("other_root");
    pkgLocator =
        new PathPackageLocator(
            outputBase,
            ImmutableList.of(pkgRoot, otherPkgRoot),
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY);
    symlink("a", "/other_root/b");
    assertValueChangesIfContentsOfFileChanges("/other_root/b", true, "a");
  }

  @Test
  public void testFilesOutsideRootIsReEvaluated() throws Exception {
    Path file = file("/outsideroot");
    SequentialBuildDriver driver = makeDriver();
    SkyKey key = skyKey("/outsideroot");
    EvaluationResult<SkyValue> result;
    result =
        driver.evaluate(
            ImmutableList.of(key), false, DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);
    if (result.hasError()) {
      fail(String.format("Evaluation error for %s: %s", key, result.getError()));
    }
    FileValue oldValue = (FileValue) result.get(key);
    assertThat(oldValue.exists()).isTrue();

    file.delete();
    differencer.invalidate(ImmutableList.of(fileStateSkyKey("/outsideroot")));
    result =
        driver.evaluate(
            ImmutableList.of(key), false, DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);
    if (result.hasError()) {
      fail(String.format("Evaluation error for %s: %s", key, result.getError()));
    }
    FileValue newValue = (FileValue) result.get(key);
    assertThat(newValue).isNotSameAs(oldValue);
    assertThat(newValue.exists()).isFalse();
  }

  @Test
  public void testFilesOutsideRootWhenExternalAssumedNonExistentAndImmutable() throws Exception {
    file("/outsideroot");

    SequentialBuildDriver driver =
        makeDriver(ExternalFileAction.ASSUME_NON_EXISTENT_AND_IMMUTABLE_FOR_EXTERNAL_PATHS);
    SkyKey key = skyKey("/outsideroot");
    EvaluationResult<SkyValue> result =
        driver.evaluate(
            ImmutableList.of(key), false, DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);

    assertThatEvaluationResult(result).hasNoError();
    FileValue value = (FileValue) result.get(key);
    assertThat(value).isNotNull();
    assertThat(value.exists()).isFalse();
  }

  @Test
  public void testAbsoluteSymlinksToFilesOutsideRootWhenExternalAssumedNonExistentAndImmutable()
      throws Exception {
    file("/outsideroot");
    symlink("a", "/outsideroot");

    SequentialBuildDriver driver =
        makeDriver(ExternalFileAction.ASSUME_NON_EXISTENT_AND_IMMUTABLE_FOR_EXTERNAL_PATHS);
    SkyKey key = skyKey("a");
    EvaluationResult<SkyValue> result =
        driver.evaluate(
            ImmutableList.of(key), false, DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);

    assertThatEvaluationResult(result).hasNoError();
    FileValue value = (FileValue) result.get(key);
    assertThat(value).isNotNull();
    assertThat(value.exists()).isFalse();
  }

  @Test
  public void testAbsoluteSymlinksReferredByInternalFilesToFilesOutsideRootWhenExternalAssumedNonExistentAndImmutable()
      throws Exception {
    file("/outsideroot/src/foo/bar");
    symlink("/root/src", "/outsideroot/src");

    SequentialBuildDriver driver =
        makeDriver(ExternalFileAction.ASSUME_NON_EXISTENT_AND_IMMUTABLE_FOR_EXTERNAL_PATHS);
    SkyKey key = skyKey("/root/src/foo/bar");
    EvaluationResult<SkyValue> result =
        driver.evaluate(
            ImmutableList.of(key), false, DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);

    assertThatEvaluationResult(result).hasNoError();
    FileValue value = (FileValue) result.get(key);
    assertThat(value).isNotNull();
    assertThat(value.exists()).isFalse();
  }

  @Test
  public void testRelativeSymlinksToFilesOutsideRootWhenExternalAssumedNonExistentAndImmutable()
      throws Exception {
    file("../outsideroot");
    symlink("a", "../outsideroot");
    SequentialBuildDriver driver =
        makeDriver(ExternalFileAction.ASSUME_NON_EXISTENT_AND_IMMUTABLE_FOR_EXTERNAL_PATHS);
    SkyKey key = skyKey("a");
    EvaluationResult<SkyValue> result =
        driver.evaluate(
            ImmutableList.of(key), false, DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);

    assertThatEvaluationResult(result).hasNoError();
    FileValue value = (FileValue) result.get(key);
    assertThat(value).isNotNull();
    assertThat(value.exists()).isFalse();
  }

  @Test
  public void testAbsoluteSymlinksBackIntoSourcesOkWhenExternalDisallowed() throws Exception {
    Path file = file("insideroot");
    symlink("a", file.getPathString());

    SequentialBuildDriver driver =
        makeDriver(ExternalFileAction.ASSUME_NON_EXISTENT_AND_IMMUTABLE_FOR_EXTERNAL_PATHS);
    SkyKey key = skyKey("a");
    EvaluationResult<SkyValue> result =
        driver.evaluate(
            ImmutableList.of(key), false, DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);

    assertThatEvaluationResult(result).hasNoError();
    FileValue value = (FileValue) result.get(key);
    assertThat(value).isNotNull();
    assertThat(value.exists()).isTrue();
    assertThat(value.realRootedPath().getRelativePath().getPathString()).isEqualTo("insideroot");
  }

  @SuppressWarnings({"rawtypes", "unchecked"})
  private static Set<RootedPath> filesSeen(MemoizingEvaluator graph) {
    return ImmutableSet.copyOf(
        (Iterable<RootedPath>)
            (Iterable)
                Iterables.transform(
                    Iterables.filter(
                        graph.getValues().keySet(),
                        SkyFunctionName.functionIs(SkyFunctions.FILE_STATE)),
                    new Function<SkyKey, Object>() {
                      @Override
                      public Object apply(SkyKey skyKey) {
                        return skyKey.argument();
                      }
                    }));
  }

  @Test
  public void testSize() throws Exception {
    Path file = file("file");
    int fileSize = 20;
    FileSystemUtils.writeContentAsLatin1(file, Strings.repeat("a", fileSize));
    assertThat(valueForPath(file).getSize()).isEqualTo(fileSize);
    Path dir = directory("directory");
    file(dir.getChild("child").getPathString());
    try {
      valueForPath(dir).getSize();
      fail();
    } catch (IllegalStateException e) {
      // Expected.
    }
    Path nonexistent = fs.getPath("/root/noexist");
    try {
      valueForPath(nonexistent).getSize();
      fail();
    } catch (IllegalStateException e) {
      // Expected.
    }
    Path symlink = symlink("link", "/root/file");
    // Symlink stores size of target, not link.
    assertThat(valueForPath(symlink).getSize()).isEqualTo(fileSize);
    assertThat(symlink.delete()).isTrue();
    symlink = symlink("link", "/root/directory");
    try {
      valueForPath(symlink).getSize();
      fail();
    } catch (IllegalStateException e) {
      // Expected.
    }
    assertThat(symlink.delete()).isTrue();
    symlink = symlink("link", "/root/noexist");
    try {
      valueForPath(symlink).getSize();
      fail();
    } catch (IllegalStateException e) {
      // Expected.
    }
  }

  @Test
  public void testDigest() throws Exception {
    final AtomicInteger digestCalls = new AtomicInteger(0);
    int expectedCalls = 0;
    fs =
        new CustomInMemoryFs(manualClock) {
          @Override
          protected byte[] getDigest(Path path, HashFunction hf) throws IOException {
            digestCalls.incrementAndGet();
            return super.getDigest(path, hf);
          }
        };
    pkgRoot = fs.getRootDirectory().getRelative("root");
    Path file = file("file");
    FileSystemUtils.writeContentAsLatin1(file, Strings.repeat("a", 20));
    byte[] digest = file.getDigest();
    expectedCalls++;
    assertThat(digestCalls.get()).isEqualTo(expectedCalls);
    FileValue value = valueForPath(file);
    expectedCalls++;
    assertThat(digestCalls.get()).isEqualTo(expectedCalls);
    assertThat(value.getDigest()).isEqualTo(digest);
    // Digest is cached -- no filesystem access.
    assertThat(digestCalls.get()).isEqualTo(expectedCalls);
    fastDigest = false;
    digestCalls.set(0);
    value = valueForPath(file);
    // No new digest calls.
    assertThat(digestCalls.get()).isEqualTo(0);
    assertThat(value.getDigest()).isNull();
    assertThat(digestCalls.get()).isEqualTo(0);
    fastDigest = true;
    Path dir = directory("directory");
    try {
      assertThat(valueForPath(dir).getDigest()).isNull();
      fail();
    } catch (IllegalStateException e) {
      // Expected.
    }
    assertThat(digestCalls.get()).isEqualTo(0); // No digest calls made for directory.
    Path nonexistent = fs.getPath("/root/noexist");
    try {
      assertThat(valueForPath(nonexistent).getDigest()).isNull();
      fail();
    } catch (IllegalStateException e) {
      // Expected.
    }
    assertThat(digestCalls.get()).isEqualTo(0); // No digest calls made for nonexistent file.
    Path symlink = symlink("link", "/root/file");
    value = valueForPath(symlink);
    assertThat(digestCalls.get()).isEqualTo(1);
    // Symlink stores digest of target, not link.
    assertThat(value.getDigest()).isEqualTo(digest);
    assertThat(digestCalls.get()).isEqualTo(1);
    digestCalls.set(0);
    assertThat(symlink.delete()).isTrue();
    symlink = symlink("link", "/root/directory");
    // Symlink stores digest of target, not link, for directories too.
    try {
      assertThat(valueForPath(symlink).getDigest()).isNull();
      fail();
    } catch (IllegalStateException e) {
      // Expected.
    }
    assertThat(digestCalls.get()).isEqualTo(0);
  }

  @Test
  public void testDoesntStatChildIfParentDoesntExist() throws Exception {
    // Our custom filesystem says "a" does not exist, so FileFunction shouldn't bother trying to
    // think about "a/b". Test for this by having a stat of "a/b" fail with an io error, and
    // observing that we don't encounter the error.
    fs.stubStat(path("a"), null);
    fs.stubStatError(path("a/b"), new IOException("ouch!"));
    assertThat(valueForPath(path("a/b")).exists()).isFalse();
  }

  @Test
  public void testFilesystemInconsistencies_ParentIsntADirectory() throws Exception {
    file("a/b");
    // Our custom filesystem says "a/b" exists but its parent "a" is a file.
    FileStatus inconsistentParentFileStatus =
        new FileStatus() {
          @Override
          public boolean isFile() {
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
            return 0;
          }

          @Override
          public long getLastModifiedTime() throws IOException {
            return 0;
          }

          @Override
          public long getLastChangeTime() throws IOException {
            return 0;
          }

          @Override
          public long getNodeId() throws IOException {
            return 0;
          }
        };
    fs.stubStat(path("a"), inconsistentParentFileStatus);
    // Disable fast-path md5 so that we don't try try to md5 the "a" (since it actually physically
    // is a directory).
    fastDigest = false;
    SequentialBuildDriver driver = makeDriver();
    SkyKey skyKey = skyKey("a/b");
    EvaluationResult<FileValue> result =
        driver.evaluate(
            ImmutableList.of(skyKey), false, DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skyKey);
    assertThat(errorInfo.getException()).isInstanceOf(InconsistentFilesystemException.class);
    assertThat(errorInfo.getException())
        .hasMessageThat()
        .contains("file /root/a/b exists but its parent path /root/a isn't an existing directory");
  }

  @Test
  public void testFilesystemInconsistencies_GetFastDigest() throws Exception {
    file("a");
    // Our custom filesystem says "a/b" exists but "a" does not exist.
    fs.stubFastDigestError(path("a"), new IOException("nope"));
    SequentialBuildDriver driver = makeDriver();
    SkyKey skyKey = skyKey("a");
    EvaluationResult<FileValue> result =
        driver.evaluate(
            ImmutableList.of(skyKey), false, DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skyKey);
    assertThat(errorInfo.getException()).isInstanceOf(InconsistentFilesystemException.class);
    assertThat(errorInfo.getException()).hasMessageThat().contains("encountered error 'nope'");
    assertThat(errorInfo.getException()).hasMessageThat().contains("/root/a is no longer a file");
  }

  @Test
  public void testFilesystemInconsistencies_GetFastDigestAndIsReadableFailure() throws Exception {
    createFsAndRoot(
        new CustomInMemoryFs(manualClock) {
          @Override
          protected boolean isReadable(Path path) throws IOException {
            if (path.getBaseName().equals("unreadable")) {
              throw new IOException("isReadable failed");
            }
            return super.isReadable(path);
          }
        });

    Path p = file("unreadable");
    p.chmod(0);

    SequentialBuildDriver driver = makeDriver();
    SkyKey skyKey = skyKey("unreadable");
    EvaluationResult<FileValue> result =
        driver.evaluate(
            ImmutableList.of(skyKey), false, DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);
    assertThat(result.hasError()).isTrue();
    ErrorInfo errorInfo = result.getError(skyKey);
    assertThat(errorInfo.getException()).isInstanceOf(InconsistentFilesystemException.class);
    assertThat(errorInfo.getException())
        .hasMessageThat()
        .contains("encountered error 'isReadable failed'");
    assertThat(errorInfo.getException())
        .hasMessageThat()
        .contains("/root/unreadable is no longer a file");
  }

  private void runTestSymlinkCycle(boolean ancestorCycle, boolean startInCycle) throws Exception {
    symlink("a", "b");
    symlink("b", "c");
    symlink("c", "d");
    symlink("d", "e");
    symlink("e", "c");
    // We build multiple keys at once to make sure the cycle is reported exactly once.
    Map<RootedPath, ImmutableList<RootedPath>> startToCycleMap =
        ImmutableMap.<RootedPath, ImmutableList<RootedPath>>builder()
            .put(
                rootedPath("a"),
                ImmutableList.of(rootedPath("c"), rootedPath("d"), rootedPath("e")))
            .put(
                rootedPath("b"),
                ImmutableList.of(rootedPath("c"), rootedPath("d"), rootedPath("e")))
            .put(
                rootedPath("d"),
                ImmutableList.<RootedPath>of(rootedPath("d"), rootedPath("e"), rootedPath("c")))
            .put(
                rootedPath("e"),
                ImmutableList.<RootedPath>of(rootedPath("e"), rootedPath("c"), rootedPath("d")))
            .put(
                rootedPath("a/some/descendant"),
                ImmutableList.of(rootedPath("c"), rootedPath("d"), rootedPath("e")))
            .put(
                rootedPath("b/some/descendant"),
                ImmutableList.of(rootedPath("c"), rootedPath("d"), rootedPath("e")))
            .put(
                rootedPath("d/some/descendant"),
                ImmutableList.<RootedPath>of(rootedPath("d"), rootedPath("e"), rootedPath("c")))
            .put(
                rootedPath("e/some/descendant"),
                ImmutableList.<RootedPath>of(rootedPath("e"), rootedPath("c"), rootedPath("d")))
            .build();
    Map<RootedPath, ImmutableList<RootedPath>> startToPathToCycleMap =
        ImmutableMap.<RootedPath, ImmutableList<RootedPath>>builder()
            .put(rootedPath("a"), ImmutableList.of(rootedPath("a"), rootedPath("b")))
            .put(rootedPath("b"), ImmutableList.of(rootedPath("b")))
            .put(rootedPath("d"), ImmutableList.<RootedPath>of())
            .put(rootedPath("e"), ImmutableList.<RootedPath>of())
            .put(
                rootedPath("a/some/descendant"), ImmutableList.of(rootedPath("a"), rootedPath("b")))
            .put(rootedPath("b/some/descendant"), ImmutableList.of(rootedPath("b")))
            .put(rootedPath("d/some/descendant"), ImmutableList.<RootedPath>of())
            .put(rootedPath("e/some/descendant"), ImmutableList.<RootedPath>of())
            .build();
    ImmutableList<SkyKey> keys;
    if (ancestorCycle && startInCycle) {
      keys = ImmutableList.of(skyKey("d/some/descendant"), skyKey("e/some/descendant"));
    } else if (ancestorCycle && !startInCycle) {
      keys = ImmutableList.of(skyKey("a/some/descendant"), skyKey("b/some/descendant"));
    } else if (!ancestorCycle && startInCycle) {
      keys = ImmutableList.of(skyKey("d"), skyKey("e"));
    } else {
      keys = ImmutableList.of(skyKey("a"), skyKey("b"));
    }
    StoredEventHandler eventHandler = new StoredEventHandler();
    SequentialBuildDriver driver = makeDriver();
    EvaluationResult<FileValue> result =
        driver.evaluate(keys, /*keepGoing=*/ true, DEFAULT_THREAD_COUNT, eventHandler);
    assertThat(result.hasError()).isTrue();
    for (SkyKey key : keys) {
      ErrorInfo errorInfo = result.getError(key);
      // FileFunction detects symlink cycles explicitly.
      assertThat(errorInfo.getCycleInfo()).isEmpty();
      FileSymlinkCycleException fsce = (FileSymlinkCycleException) errorInfo.getException();
      RootedPath start = (RootedPath) key.argument();
      assertThat(fsce.getPathToCycle())
          .containsExactlyElementsIn(startToPathToCycleMap.get(start))
          .inOrder();
      assertThat(fsce.getCycle()).containsExactlyElementsIn(startToCycleMap.get(start)).inOrder();
    }
    // Check that the unique cycle was reported exactly once.
    assertThat(eventHandler.getEvents()).hasSize(1);
    assertThat(Iterables.getOnlyElement(eventHandler.getEvents()).getMessage())
        .contains("circular symlinks detected");
  }

  @Test
  public void testSymlinkCycle_AncestorCycle_StartInCycle() throws Exception {
    runTestSymlinkCycle(/*ancestorCycle=*/ true, /*startInCycle=*/ true);
  }

  @Test
  public void testSymlinkCycle_AncestorCycle_StartOutOfCycle() throws Exception {
    runTestSymlinkCycle(/*ancestorCycle=*/ true, /*startInCycle=*/ false);
  }

  @Test
  public void testSymlinkCycle_RegularCycle_StartInCycle() throws Exception {
    runTestSymlinkCycle(/*ancestorCycle=*/ false, /*startInCycle=*/ true);
  }

  @Test
  public void testSymlinkCycle_RegularCycle_StartOutOfCycle() throws Exception {
    runTestSymlinkCycle(/*ancestorCycle=*/ false, /*startInCycle=*/ false);
  }

  @Test
  public void testSerialization() throws Exception {
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    ObjectOutputStream oos = new ObjectOutputStream(bos);

    FileSystem oldFileSystem = Path.getFileSystemForSerialization();
    try {
      // InMemoryFS is not supported for serialization.
      FileSystem fs = FileSystems.getJavaIoFileSystem();
      Path.setFileSystemForSerialization(fs);
      pkgRoot = fs.getRootDirectory();

      FileValue a = valueForPath(fs.getPath("/"));

      Path tmp = fs.getPath(TestUtils.tmpDirFile().getAbsoluteFile() + "/file.txt");

      FileSystemUtils.writeContentAsLatin1(tmp, "test contents");

      FileValue b = valueForPath(tmp);
      Preconditions.checkState(b.isFile());
      FileValue c = valueForPath(fs.getPath("/does/not/exist"));
      oos.writeObject(a);
      oos.writeObject(b);
      oos.writeObject(c);

      ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
      ObjectInputStream ois = new ObjectInputStream(bis);

      FileValue a2 = (FileValue) ois.readObject();
      FileValue b2 = (FileValue) ois.readObject();
      FileValue c2 = (FileValue) ois.readObject();

      assertThat(a2).isEqualTo(a);
      assertThat(b2).isEqualTo(b);
      assertThat(c2).isEqualTo(c);
      assertThat(a2.equals(b2)).isFalse();
    } finally {
      Path.setFileSystemForSerialization(oldFileSystem);
    }
  }

  @Test
  public void testFileStateEquality() throws Exception {
    file("a");
    symlink("b1", "a");
    symlink("b2", "a");
    symlink("b3", "zzz");
    directory("d1");
    directory("d2");
    SkyKey file = fileStateSkyKey("a");
    SkyKey symlink1 = fileStateSkyKey("b1");
    SkyKey symlink2 = fileStateSkyKey("b2");
    SkyKey symlink3 = fileStateSkyKey("b3");
    SkyKey missing1 = fileStateSkyKey("c1");
    SkyKey missing2 = fileStateSkyKey("c2");
    SkyKey directory1 = fileStateSkyKey("d1");
    SkyKey directory2 = fileStateSkyKey("d2");
    ImmutableList<SkyKey> keys =
        ImmutableList.of(
            file, symlink1, symlink2, symlink3, missing1, missing2, directory1, directory2);

    SequentialBuildDriver driver = makeDriver();
    EvaluationResult<SkyValue> result =
        driver.evaluate(keys, false, DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);

    new EqualsTester()
        .addEqualityGroup(result.get(file))
        .addEqualityGroup(result.get(symlink1), result.get(symlink2))
        .addEqualityGroup(result.get(symlink3))
        .addEqualityGroup(result.get(missing1), result.get(missing2))
        .addEqualityGroup(result.get(directory1), result.get(directory2))
        .testEquals();
  }

  @Test
  public void testSymlinkToPackagePathBoundary() throws Exception {
    Path path = path("this/is/a/path");
    FileSystemUtils.ensureSymbolicLink(path, pkgRoot);
    assertError("this/is/a/path");
  }

  private void runTestInfiniteSymlinkExpansion(boolean symlinkToAncestor, boolean absoluteSymlink)
      throws Exception {
    Path otherPath = path("other");
    RootedPath otherRootedPath = RootedPath.toRootedPath(pkgRoot, otherPath.relativeTo(pkgRoot));
    Path ancestorPath = path("a");
    RootedPath ancestorRootedPath =
        RootedPath.toRootedPath(pkgRoot, ancestorPath.relativeTo(pkgRoot));
    FileSystemUtils.ensureSymbolicLink(otherPath, ancestorPath);
    Path intermediatePath = path("inter");
    RootedPath intermediateRootedPath =
        RootedPath.toRootedPath(pkgRoot, intermediatePath.relativeTo(pkgRoot));
    Path descendantPath = path("a/b/c/d/e");
    RootedPath descendantRootedPath =
        RootedPath.toRootedPath(pkgRoot, descendantPath.relativeTo(pkgRoot));
    if (symlinkToAncestor) {
      FileSystemUtils.ensureSymbolicLink(descendantPath, intermediatePath);
      if (absoluteSymlink) {
        FileSystemUtils.ensureSymbolicLink(intermediatePath, ancestorPath);
      } else {
        FileSystemUtils.ensureSymbolicLink(intermediatePath, ancestorRootedPath.getRelativePath());
      }
    } else {
      FileSystemUtils.ensureSymbolicLink(ancestorPath, intermediatePath);
      if (absoluteSymlink) {
        FileSystemUtils.ensureSymbolicLink(intermediatePath, descendantPath);
      } else {
        FileSystemUtils.ensureSymbolicLink(
            intermediatePath, descendantRootedPath.getRelativePath());
      }
    }
    StoredEventHandler eventHandler = new StoredEventHandler();
    SequentialBuildDriver driver = makeDriver();
    SkyKey ancestorPathKey = FileValue.key(ancestorRootedPath);
    SkyKey descendantPathKey = FileValue.key(descendantRootedPath);
    SkyKey otherPathKey = FileValue.key(otherRootedPath);
    ImmutableList<SkyKey> keys;
    ImmutableList<SkyKey> errorKeys;
    ImmutableList<RootedPath> expectedChain;
    if (symlinkToAncestor) {
      keys = ImmutableList.of(descendantPathKey, otherPathKey);
      errorKeys = ImmutableList.of(descendantPathKey);
      expectedChain =
          ImmutableList.of(descendantRootedPath, intermediateRootedPath, ancestorRootedPath);
    } else {
      keys = ImmutableList.of(ancestorPathKey, otherPathKey);
      errorKeys = keys;
      expectedChain =
          ImmutableList.of(ancestorRootedPath, intermediateRootedPath, descendantRootedPath);
    }
    EvaluationResult<FileValue> result =
        driver.evaluate(keys, /*keepGoing=*/ true, DEFAULT_THREAD_COUNT, eventHandler);
    assertThat(result.hasError()).isTrue();
    for (SkyKey key : errorKeys) {
      ErrorInfo errorInfo = result.getError(key);
      // FileFunction detects infinite symlink expansion explicitly.
      assertThat(errorInfo.getCycleInfo()).isEmpty();
      FileSymlinkInfiniteExpansionException fsiee =
          (FileSymlinkInfiniteExpansionException) errorInfo.getException();
      assertThat(fsiee).hasMessageThat().contains("Infinite symlink expansion");
      assertThat(fsiee.getChain()).containsExactlyElementsIn(expectedChain).inOrder();
    }
    // Check that the unique symlink expansion error was reported exactly once.
    assertThat(eventHandler.getEvents()).hasSize(1);
    assertThat(Iterables.getOnlyElement(eventHandler.getEvents()).getMessage())
        .contains("infinite symlink expansion detected");
  }

  @Test
  public void testInfiniteSymlinkExpansion_AbsoluteSymlinkToDescendant() throws Exception {
    runTestInfiniteSymlinkExpansion(/* symlinkToAncestor= */ false, /*absoluteSymlink=*/ true);
  }

  @Test
  public void testInfiniteSymlinkExpansion_RelativeSymlinkToDescendant() throws Exception {
    runTestInfiniteSymlinkExpansion(/* symlinkToAncestor= */ false, /*absoluteSymlink=*/ false);
  }

  @Test
  public void testInfiniteSymlinkExpansion_AbsoluteSymlinkToAncestor() throws Exception {
    runTestInfiniteSymlinkExpansion(/* symlinkToAncestor= */ true, /*absoluteSymlink=*/ true);
  }

  @Test
  public void testInfiniteSymlinkExpansion_RelativeSymlinkToAncestor() throws Exception {
    runTestInfiniteSymlinkExpansion(/* symlinkToAncestor= */ true, /*absoluteSymlink=*/ false);
  }

  @Test
  public void testChildOfNonexistentParent() throws Exception {
    Path ancestor = directory("this/is/an/ancestor");
    Path parent = ancestor.getChild("parent");
    Path child = parent.getChild("child");
    assertThat(valueForPath(parent).exists()).isFalse();
    assertThat(valueForPath(child).exists()).isFalse();
  }

  @Test
  public void testInjectionOverIOException() throws Exception {
    Path foo = file("foo");
    SkyKey fooKey = skyKey("foo");
    fs.stubStatError(foo, new IOException("bork"));
    BuildDriver driver = makeDriver();
    EvaluationResult<FileValue> result =
        driver.evaluate(
            ImmutableList.of(fooKey),
            /*keepGoing=*/ true,
            /*numThreads=*/ 1,
            NullEventHandler.INSTANCE);
    ErrorInfoSubject errorInfoSubject = assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(fooKey);
    errorInfoSubject.isTransient();
    errorInfoSubject
        .hasExceptionThat()
        .hasMessageThat()
        .isEqualTo("bork");
    fs.stubbedStatErrors.remove(foo);
    differencer.inject(
        fileStateSkyKey("foo"),
        FileStateValue.create(
            RootedPath.toRootedPath(pkgRoot, foo),
            new TimestampGranularityMonitor(BlazeClock.instance())));
    result =
        driver.evaluate(
            ImmutableList.of(fooKey),
            /*keepGoing=*/ true,
            /*numThreads=*/ 1,
            NullEventHandler.INSTANCE);
    assertThatEvaluationResult(result).hasNoError();
    assertThat(result.get(fooKey).exists()).isTrue();
  }

  private void checkRealPath(String pathString) throws Exception {
    Path realPath = pkgRoot.getRelative(pathString).resolveSymbolicLinks();
    assertRealPath(pathString, realPath.relativeTo(pkgRoot).toString());
  }

  private void assertRealPath(String pathString, String expectedRealPathString) throws Exception {
    SequentialBuildDriver driver = makeDriver();
    SkyKey key = skyKey(pathString);
    EvaluationResult<SkyValue> result =
        driver.evaluate(
            ImmutableList.of(key), false, DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);
    if (result.hasError()) {
      fail(String.format("Evaluation error for %s: %s", key, result.getError()));
    }
    FileValue fileValue = (FileValue) result.get(key);
    assertThat(fileValue.realRootedPath().asPath().toString())
        .isEqualTo(pkgRoot.getRelative(expectedRealPathString).toString());
  }

  /**
   * Returns a callback that, when executed, deletes the given path. Not meant to be called directly
   * by tests.
   */
  private Runnable makeDeletePathCallback(final Path toDelete) {
    return new Runnable() {
      @Override
      public void run() {
        try {
          toDelete.delete();
        } catch (IOException e) {
          e.printStackTrace();
          fail(e.getMessage());
        }
      }
    };
  }

  /**
   * Returns a callback that, when executed, writes the given bytes to the given file path. Not
   * meant to be called directly by tests.
   */
  private Runnable makeWriteFileContentCallback(final Path toChange, final byte[] contents) {
    return new Runnable() {
      @Override
      public void run() {
        OutputStream outputStream;
        try {
          outputStream = toChange.getOutputStream();
          outputStream.write(contents);
          outputStream.close();
        } catch (IOException e) {
          e.printStackTrace();
          fail(e.getMessage());
        }
      }
    };
  }

  /**
   * Returns a callback that, when executed, creates the given directory path. Not meant to be
   * called directly by tests.
   */
  private Runnable makeCreateDirectoryCallback(final Path toCreate) {
    return new Runnable() {
      @Override
      public void run() {
        try {
          toCreate.createDirectory();
        } catch (IOException e) {
          e.printStackTrace();
          fail(e.getMessage());
        }
      }
    };
  }

  /**
   * Returns a callback that, when executed, makes {@code toLink} a symlink to {@code toTarget}. Not
   * meant to be called directly by tests.
   */
  private Runnable makeSymlinkCallback(final Path toLink, final PathFragment toTarget) {
    return new Runnable() {
      @Override
      public void run() {
        try {
          FileSystemUtils.ensureSymbolicLink(toLink, toTarget);
        } catch (IOException e) {
          e.printStackTrace();
          fail(e.getMessage());
        }
      }
    };
  }

  /** Returns the files that would be changed/created if {@code path} were to be changed/created. */
  private ImmutableList<String> filesTouchedIfTouched(Path path) {
    List<String> filesToBeTouched = Lists.newArrayList();
    do {
      filesToBeTouched.add(path.getPathString());
      path = path.getParentDirectory();
    } while (!path.exists());
    return ImmutableList.copyOf(filesToBeTouched);
  }

  /**
   * Changes the contents of the FileValue for the given file in some way e.g.
   *
   * <ul>
   * <li> If it's a regular file, the contents will be changed.
   * <li> If it's a non-existent file, it will be created.
   *     <ul>
   *     and then returns the file(s) changed paired with a callback to undo the change. Not meant
   *     to be called directly by tests.
   */
  private Pair<ImmutableList<String>, Runnable> changeFile(String fileStringToChange)
      throws Exception {
    Path fileToChange = path(fileStringToChange);
    if (fileToChange.exists()) {
      final byte[] oldContents = FileSystemUtils.readContent(fileToChange);
      OutputStream outputStream = fileToChange.getOutputStream(/*append=*/ true);
      outputStream.write(new byte[] {(byte) 42}, 0, 1);
      outputStream.close();
      return Pair.of(
          ImmutableList.of(fileStringToChange),
          makeWriteFileContentCallback(fileToChange, oldContents));
    } else {
      ImmutableList<String> filesTouched = filesTouchedIfTouched(fileToChange);
      file(fileStringToChange, "new stuff");
      return Pair.of(ImmutableList.copyOf(filesTouched), makeDeletePathCallback(fileToChange));
    }
  }

  /**
   * Changes the contents of the FileValue for the given directory in some way e.g.
   *
   * <ul>
   * <li> If it exists, the directory will be deleted.
   * <li> If it doesn't exist, the directory will be created.
   *     <ul>
   *     and then returns the file(s) changed paired with a callback to undo the change. Not meant
   *     to be called directly by tests.
   */
  private Pair<ImmutableList<String>, Runnable> changeDirectory(String directoryStringToChange)
      throws Exception {
    final Path directoryToChange = path(directoryStringToChange);
    if (directoryToChange.exists()) {
      directoryToChange.delete();
      return Pair.of(
          ImmutableList.of(directoryStringToChange),
          makeCreateDirectoryCallback(directoryToChange));
    } else {
      directoryToChange.createDirectory();
      return Pair.of(
          ImmutableList.of(directoryStringToChange), makeDeletePathCallback(directoryToChange));
    }
  }

  /**
   * Performs filesystem operations to change the file or directory denoted by {@code
   * changedPathString} and returns the file(s) changed paired with a callback to undo the change.
   * Not meant to be called directly by tests.
   *
   * @param isSupposedToBeFile whether the path denoted by the given string is supposed to be a file
   *     or a directory. This is needed is the path doesn't exist yet, and so the filesystem doesn't
   *     know.
   */
  private Pair<ImmutableList<String>, Runnable> change(
      String changedPathString, boolean isSupposedToBeFile) throws Exception {
    final Path changedPath = path(changedPathString);
    if (changedPath.isSymbolicLink()) {
      ImmutableList<String> filesTouched = filesTouchedIfTouched(changedPath);
      PathFragment oldTarget = changedPath.readSymbolicLink();
      FileSystemUtils.ensureSymbolicLink(changedPath, oldTarget.getChild("__different_target__"));
      return Pair.of(filesTouched, makeSymlinkCallback(changedPath, oldTarget));
    } else if (isSupposedToBeFile) {
      return changeFile(changedPathString);
    } else {
      return changeDirectory(changedPathString);
    }
  }

  /**
   * Asserts that if the contents of {@code changedPathString} changes, then the FileValue
   * corresponding to {@code pathString} will change. Not meant to be called directly by tests.
   */
  private void assertValueChangesIfContentsOfFileChanges(
      String changedPathString, boolean changes, String pathString) throws Exception {
    getFilesSeenAndAssertValueChangesIfContentsOfFileChanges(
        changedPathString, changes, pathString);
  }

  /**
   * Asserts that if the contents of {@code changedPathString} changes, then the FileValue
   * corresponding to {@code pathString} will change. Returns the paths of all files seen.
   */
  private Set<RootedPath> getFilesSeenAndAssertValueChangesIfContentsOfFileChanges(
      String changedPathString, boolean changes, String pathString) throws Exception {
    return assertChangesIfChanges(changedPathString, true, changes, pathString);
  }

  /**
   * Asserts that if the directory {@code changedPathString} changes, then the FileValue
   * corresponding to {@code pathString} will change. Returns the paths of all files seen.
   */
  private Set<RootedPath> assertValueChangesIfContentsOfDirectoryChanges(
      String changedPathString, boolean changes, String pathString) throws Exception {
    return assertChangesIfChanges(changedPathString, false, changes, pathString);
  }

  /**
   * Asserts that if the contents of {@code changedPathString} changes, then the FileValue
   * corresponding to {@code pathString} will change. Returns the paths of all files seen. Not meant
   * to be called directly by tests.
   */
  private Set<RootedPath> assertChangesIfChanges(
      String changedPathString, boolean isFile, boolean changes, String pathString)
      throws Exception {
    SequentialBuildDriver driver = makeDriver();
    SkyKey key = skyKey(pathString);
    EvaluationResult<SkyValue> result;
    result =
        driver.evaluate(
            ImmutableList.of(key), false, DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);
    if (result.hasError()) {
      fail(String.format("Evaluation error for %s: %s", key, result.getError()));
    }
    SkyValue oldValue = result.get(key);

    Pair<ImmutableList<String>, Runnable> changeResult = change(changedPathString, isFile);
    ImmutableList<String> changedPathStrings = changeResult.first;
    Runnable undoCallback = changeResult.second;
    differencer.invalidate(
        Iterables.transform(
            changedPathStrings,
            new Function<String, SkyKey>() {
              @Override
              public SkyKey apply(String input) {
                return fileStateSkyKey(input);
              }
            }));

    result =
        driver.evaluate(
            ImmutableList.of(key), false, DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);
    if (result.hasError()) {
      fail(String.format("Evaluation error for %s: %s", key, result.getError()));
    }

    SkyValue newValue = result.get(key);
    assertWithMessage(
            String.format(
                "Changing the contents of %s %s should%s change the value for file %s.",
                isFile ? "file" : "directory",
                changedPathString,
                changes ? "" : " not",
                pathString))
        .that(changes != newValue.equals(oldValue))
        .isTrue();

    // Restore the original file.
    undoCallback.run();
    return filesSeen(driver.getGraphForTesting());
  }

  /**
   * Asserts that trying to construct a FileValue for {@code path} succeeds. Returns the paths of
   * all files seen.
   */
  private Set<RootedPath> assertNoError(String pathString) throws Exception {
    SequentialBuildDriver driver = makeDriver();
    SkyKey key = skyKey(pathString);
    EvaluationResult<FileValue> result;
    result =
        driver.evaluate(
            ImmutableList.of(key), false, DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);
    assertWithMessage(
            "Did not expect error while evaluating " + pathString + ", got " + result.get(key))
        .that(result.hasError())
        .isFalse();
    return filesSeen(driver.getGraphForTesting());
  }

  /**
   * Asserts that trying to construct a FileValue for {@code path} fails. Returns the paths of all
   * files seen.
   */
  private Set<RootedPath> assertError(String pathString) throws Exception {
    SequentialBuildDriver driver = makeDriver();
    SkyKey key = skyKey(pathString);
    EvaluationResult<FileValue> result;
    result =
        driver.evaluate(
            ImmutableList.of(key), false, DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);
    assertWithMessage("Expected error while evaluating " + pathString + ", got " + result.get(key))
        .that(result.hasError())
        .isTrue();
    assertThat(
            !Iterables.isEmpty(result.getError().getCycleInfo())
                || result.getError().getException() != null)
        .isTrue();
    return filesSeen(driver.getGraphForTesting());
  }

  private Path file(String fileName) throws Exception {
    Path path = path(fileName);
    FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
    FileSystemUtils.createEmptyFile(path);
    return path;
  }

  private Path file(String fileName, String contents) throws Exception {
    Path path = path(fileName);
    FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
    FileSystemUtils.writeContentAsLatin1(path, contents);
    return path;
  }

  private Path directory(String directoryName) throws Exception {
    Path path = path(directoryName);
    FileSystemUtils.createDirectoryAndParents(path);
    return path;
  }

  private Path symlink(String link, String target) throws Exception {
    Path path = path(link);
    FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
    path.createSymbolicLink(PathFragment.create(target));
    return path;
  }

  private Path path(String rootRelativePath) {
    return pkgRoot.getRelative(PathFragment.create(rootRelativePath));
  }

  private RootedPath rootedPath(String pathString) {
    Path path = path(pathString);
    for (Path root : pkgLocator.getPathEntries()) {
      if (path.startsWith(root)) {
        return RootedPath.toRootedPath(root, path);
      }
    }
    return RootedPath.toRootedPath(fs.getRootDirectory(), path);
  }

  private SkyKey skyKey(String pathString) {
    return FileValue.key(rootedPath(pathString));
  }

  private SkyKey fileStateSkyKey(String pathString) {
    return FileStateValue.key(rootedPath(pathString));
  }

  private class CustomInMemoryFs extends InMemoryFileSystem {

    private final Map<Path, FileStatus> stubbedStats = Maps.newHashMap();
    private final Map<Path, IOException> stubbedStatErrors = Maps.newHashMap();
    private final Map<Path, IOException> stubbedFastDigestErrors = Maps.newHashMap();

    public CustomInMemoryFs(ManualClock manualClock) {
      super(manualClock);
    }

    public void stubFastDigestError(Path path, IOException error) {
      stubbedFastDigestErrors.put(path, error);
    }

    @Override
    protected byte[] getFastDigest(Path path, HashFunction hashFunction) throws IOException {
      if (stubbedFastDigestErrors.containsKey(path)) {
        throw stubbedFastDigestErrors.get(path);
      }
      return fastDigest ? getDigest(path) : null;
    }

    public void stubStat(Path path, @Nullable FileStatus stubbedResult) {
      stubbedStats.put(path, stubbedResult);
    }

    public void stubStatError(Path path, IOException error) {
      stubbedStatErrors.put(path, error);
    }

    @Override
    public FileStatus stat(Path path, boolean followSymlinks) throws IOException {
      if (stubbedStatErrors.containsKey(path)) {
        throw stubbedStatErrors.get(path);
      }
      if (stubbedStats.containsKey(path)) {
        return stubbedStats.get(path);
      }
      return super.stat(path, followSymlinks);
    }
  }
}
