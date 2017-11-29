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
import static com.google.devtools.build.lib.actions.FilesetTraversalParams.PackageBoundaryMode.CROSS;
import static com.google.devtools.build.lib.actions.FilesetTraversalParams.PackageBoundaryMode.DONT_CROSS;
import static com.google.devtools.build.lib.actions.FilesetTraversalParams.PackageBoundaryMode.REPORT_ERROR;

import com.google.common.base.Preconditions;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.FilesetTraversalParams;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.PackageBoundaryMode;
import com.google.devtools.build.lib.actions.FilesetTraversalParamsFactory;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.FilesetEntry.SymlinkBehavior;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
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
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FilesetEntryFunction}. */
@RunWith(JUnit4.class)
public final class FilesetEntryFunctionTest extends FoundationTestCase {
  private MemoizingEvaluator evaluator;
  private SequentialBuildDriver driver;
  private RecordingDifferencer differencer;
  private AtomicReference<PathPackageLocator> pkgLocator;

  @Before
  public final void setUp() throws Exception  {
    pkgLocator =
        new AtomicReference<>(
            new PathPackageLocator(
                outputBase,
                ImmutableList.of(rootDirectory),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    AtomicReference<ImmutableSet<PackageIdentifier>> deletedPackages =
        new AtomicReference<>(ImmutableSet.<PackageIdentifier>of());
    ExternalFilesHelper externalFilesHelper =
        new ExternalFilesHelper(
            pkgLocator,
            ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS,
            new BlazeDirectories(
                new ServerDirectories(outputBase, outputBase),
                rootDirectory,
                TestConstants.PRODUCT_NAME));

    Map<SkyFunctionName, SkyFunction> skyFunctions = new HashMap<>();

    skyFunctions.put(SkyFunctions.FILE_STATE, new FileStateFunction(
        new AtomicReference<TimestampGranularityMonitor>(), externalFilesHelper));
    skyFunctions.put(SkyFunctions.FILE, new FileFunction(pkgLocator));
    skyFunctions.put(SkyFunctions.DIRECTORY_LISTING, new DirectoryListingFunction());
    skyFunctions.put(
        SkyFunctions.DIRECTORY_LISTING_STATE,
        new DirectoryListingStateFunction(externalFilesHelper));
    skyFunctions.put(
        SkyFunctions.RECURSIVE_FILESYSTEM_TRAVERSAL, new RecursiveFilesystemTraversalFunction());
    skyFunctions.put(
        SkyFunctions.PACKAGE_LOOKUP,
        new PackageLookupFunction(
            deletedPackages,
            CrossRepositoryLabelViolationStrategy.ERROR,
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    skyFunctions.put(SkyFunctions.BLACKLISTED_PACKAGE_PREFIXES,
        new BlacklistedPackagePrefixesFunction());
    skyFunctions.put(SkyFunctions.FILESET_ENTRY, new FilesetEntryFunction());
    skyFunctions.put(SkyFunctions.LOCAL_REPOSITORY_LOOKUP, new LocalRepositoryLookupFunction());

    differencer = new SequencedRecordingDifferencer();
    evaluator = new InMemoryMemoizingEvaluator(skyFunctions, differencer);
    driver = new SequentialBuildDriver(evaluator);
    PrecomputedValue.BUILD_ID.set(differencer, UUID.randomUUID());
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator.get());
    PrecomputedValue.BLACKLISTED_PACKAGE_PREFIXES_FILE.set(differencer,
        PathFragment.EMPTY_FRAGMENT);
  }

  private Artifact getSourceArtifact(String path) throws Exception {
    return new Artifact(PathFragment.create(path), Root.asSourceRoot(rootDirectory));
  }

  private Artifact createSourceArtifact(String path) throws Exception {
    Artifact result = getSourceArtifact(path);
    createFile(result, "foo");
    return result;
  }

  private static RootedPath rootedPath(Artifact artifact) {
    return RootedPath.toRootedPath(artifact.getRoot().getPath(), artifact.getRootRelativePath());
  }

  private static RootedPath childOf(Artifact artifact, String relative) {
    return RootedPath.toRootedPath(
        artifact.getRoot().getPath(), artifact.getRootRelativePath().getRelative(relative));
  }

  private static RootedPath siblingOf(Artifact artifact, String relative) {
    PathFragment parent =
        Preconditions.checkNotNull(artifact.getRootRelativePath().getParentDirectory());
    return RootedPath.toRootedPath(artifact.getRoot().getPath(), parent.getRelative(relative));
  }

  private void createFile(Path path, String... contents) throws Exception {
    if (!path.getParentDirectory().exists()) {
      scratch.dir(path.getParentDirectory().getPathString());
    }
    scratch.file(path.getPathString(), contents);
  }

  private void createFile(Artifact artifact, String... contents) throws Exception {
    createFile(artifact.getPath(), contents);
  }

  private RootedPath createFile(RootedPath path, String... contents) throws Exception {
    createFile(path.asPath(), contents);
    return path;
  }

  private <T extends SkyValue> EvaluationResult<T> eval(SkyKey key) throws Exception {
    return driver.evaluate(
        ImmutableList.of(key),
        false,
        SkyframeExecutor.DEFAULT_THREAD_COUNT,
        NullEventHandler.INSTANCE);
  }

  private FilesetEntryValue evalFilesetTraversal(FilesetTraversalParams params) throws Exception {
    SkyKey key = FilesetEntryKey.key(params);
    EvaluationResult<FilesetEntryValue> result = eval(key);
    assertThat(result.hasError()).isFalse();
    return result.get(key);
  }

  private static FilesetOutputSymlink symlink(String from, Artifact to) {
    return new FilesetOutputSymlink(PathFragment.create(from), to.getPath().asFragment());
  }

  private static FilesetOutputSymlink symlink(String from, String to) {
    return new FilesetOutputSymlink(PathFragment.create(from), PathFragment.create(to));
  }

  private static FilesetOutputSymlink symlink(String from, RootedPath to) {
    return new FilesetOutputSymlink(PathFragment.create(from), to.asPath().asFragment());
  }

  private void assertSymlinksCreatedInOrder(
      FilesetTraversalParams request, FilesetOutputSymlink... expectedSymlinks) throws Exception {
    List<FilesetOutputSymlink> expected = Arrays.asList(expectedSymlinks);
    Collection<FilesetOutputSymlink> actual =
        Collections2.transform(
            evalFilesetTraversal(request).getSymlinks(),
            // Strip the metadata from the actual results.
            (input) -> new FilesetOutputSymlink(input.name, input.target));
    assertThat(actual).containsExactlyElementsIn(expected).inOrder();
  }

  private static Label label(String label) throws Exception {
    return Label.parseAbsolute(label);
  }

  @Test
  public void testFileTraversalForFile() throws Exception {
    Artifact file = createSourceArtifact("foo/file.real");
    FilesetTraversalParams params =
        FilesetTraversalParamsFactory.fileTraversal(
            /*ownerLabel=*/ label("//foo"),
            /*fileToTraverse=*/ file,
            PathFragment.create("output-name"),
            /*symlinkBehaviorMode=*/ SymlinkBehavior.COPY,
            /*pkgBoundaryMode=*/ DONT_CROSS);
    assertSymlinksCreatedInOrder(params, symlink("output-name", file));
  }

  private void assertFileTraversalForFileSymlink(SymlinkBehavior symlinks) throws Exception {
    Artifact file = createSourceArtifact("foo/file.real");
    Artifact symlink = getSourceArtifact("foo/file.sym");
    symlink.getPath().createSymbolicLink(PathFragment.create("file.real"));

    FilesetTraversalParams params =
        FilesetTraversalParamsFactory.fileTraversal(
            /*ownerLabel=*/ label("//foo"),
            /*fileToTraverse=*/ symlink,
            PathFragment.create("output-name"),
            /*symlinkBehaviorMode=*/ symlinks,
            /*pkgBoundaryMode=*/ DONT_CROSS);
    switch (symlinks) {
      case COPY:
        assertSymlinksCreatedInOrder(params, symlink("output-name", "file.real"));
        break;
      case DEREFERENCE:
        assertSymlinksCreatedInOrder(params, symlink("output-name", file));
        break;
      default:
        throw new IllegalStateException(symlinks.toString());
    }
  }

  @Test
  public void testFileTraversalForFileSymlinkNoFollow() throws Exception {
    assertFileTraversalForFileSymlink(SymlinkBehavior.COPY);
  }

  @Test
  public void testFileTraversalForFileSymlinkFollow() throws Exception {
    assertFileTraversalForFileSymlink(SymlinkBehavior.DEREFERENCE);
  }

  @Test
  public void testFileTraversalForDirectory() throws Exception {
    Artifact dir = getSourceArtifact("foo/dir_real");
    RootedPath fileA = createFile(childOf(dir, "file.a"), "hello");
    RootedPath fileB = createFile(childOf(dir, "sub/file.b"), "world");

    FilesetTraversalParams params =
        FilesetTraversalParamsFactory.fileTraversal(
            /*ownerLabel=*/ label("//foo"),
            /*fileToTraverse=*/ dir,
            PathFragment.create("output-name"),
            /*symlinkBehaviorMode=*/ SymlinkBehavior.COPY,
            /*pkgBoundaryMode=*/ DONT_CROSS);
    assertSymlinksCreatedInOrder(
        params, symlink("output-name/file.a", fileA), symlink("output-name/sub/file.b", fileB));
  }

  private void assertFileTraversalForDirectorySymlink(SymlinkBehavior symlinks) throws Exception {
    Artifact dir = getSourceArtifact("foo/dir_real");
    Artifact symlink = getSourceArtifact("foo/dir_sym");
    createFile(childOf(dir, "file.a"), "hello");
    createFile(childOf(dir, "sub/file.b"), "world");
    symlink.getPath().createSymbolicLink(PathFragment.create("dir_real"));

    FilesetTraversalParams params =
        FilesetTraversalParamsFactory.fileTraversal(
            /*ownerLabel=*/ label("//foo"),
            /*fileToTraverse=*/ symlink,
            PathFragment.create("output-name"),
            /*symlinkBehaviorMode=*/ symlinks,
            /*pkgBoundaryMode=*/ DONT_CROSS);
    switch (symlinks) {
      case COPY:
        assertSymlinksCreatedInOrder(params, symlink("output-name", "dir_real"));
        break;
      case DEREFERENCE:
        assertSymlinksCreatedInOrder(params, symlink("output-name", dir));
        break;
      default:
        throw new IllegalStateException(symlinks.toString());
    }
  }

  @Test
  public void testFileTraversalForDirectorySymlinkFollow() throws Exception {
    assertFileTraversalForDirectorySymlink(SymlinkBehavior.COPY);
  }

  @Test
  public void testFileTraversalForDirectorySymlinkNoFollow() throws Exception {
    assertFileTraversalForDirectorySymlink(SymlinkBehavior.DEREFERENCE);
  }

  private void assertRecursiveTraversalForDirectory(
      SymlinkBehavior symlinks, PackageBoundaryMode pkgBoundaryMode) throws Exception {
    Artifact dir = getSourceArtifact("foo/dir");
    RootedPath fileA = createFile(childOf(dir, "file.a"), "blah");
    RootedPath fileAsym = childOf(dir, "subdir/file.a.sym");
    RootedPath buildFile = createFile(childOf(dir, "subpkg/BUILD"), "blah");
    RootedPath fileB = createFile(childOf(dir, "subpkg/file.b"), "blah");
    fileAsym.asPath().getParentDirectory().createDirectory();
    fileAsym.asPath().createSymbolicLink(PathFragment.create("../file.a"));

    FilesetOutputSymlink outA = symlink("output-name/file.a", childOf(dir, "file.a"));
    FilesetOutputSymlink outAsym = null;
    FilesetOutputSymlink outBuild = symlink("output-name/subpkg/BUILD", buildFile);
    FilesetOutputSymlink outB = symlink("output-name/subpkg/file.b", fileB);
    switch (symlinks) {
      case COPY:
        outAsym = symlink("output-name/subdir/file.a.sym", "../file.a");
        break;
      case DEREFERENCE:
        outAsym = symlink("output-name/subdir/file.a.sym", fileA);
        break;
      default:
        throw new IllegalStateException(symlinks.toString());
    }

    FilesetTraversalParams params =
        FilesetTraversalParamsFactory.recursiveTraversalOfDirectory(
            /*ownerLabel=*/ label("//foo"),
            /*directoryToTraverse=*/ dir,
            PathFragment.create("output-name"),
            /*excludes=*/ null,
            /*symlinkBehaviorMode=*/ symlinks,
            /*pkgBoundaryMode=*/ pkgBoundaryMode);
    switch (pkgBoundaryMode) {
      case CROSS:
        assertSymlinksCreatedInOrder(params, outA, outAsym, outBuild, outB);
        break;
      case DONT_CROSS:
        assertSymlinksCreatedInOrder(params, outA, outAsym);
        break;
      case REPORT_ERROR:
        SkyKey key = FilesetEntryKey.key(params);
        EvaluationResult<SkyValue> result = eval(key);
        assertThat(result.hasError()).isTrue();
        assertThat(result.getError(key).getException())
            .hasMessageThat()
            .contains("'foo/dir' crosses package boundary into package rooted at foo/dir/subpkg");
        break;
      default:
        throw new IllegalStateException(pkgBoundaryMode.toString());
    }
  }

  @Test
  public void testRecursiveTraversalForDirectoryCrossNoFollow() throws Exception {
    assertRecursiveTraversalForDirectory(SymlinkBehavior.COPY, CROSS);
  }

  @Test
  public void testRecursiveTraversalForDirectoryDontCrossNoFollow() throws Exception {
    assertRecursiveTraversalForDirectory(SymlinkBehavior.COPY, DONT_CROSS);
  }

  @Test
  public void testRecursiveTraversalForDirectoryReportErrorNoFollow() throws Exception {
    assertRecursiveTraversalForDirectory(SymlinkBehavior.COPY, REPORT_ERROR);
  }

  @Test
  public void testRecursiveTraversalForDirectoryCrossFollow() throws Exception {
    assertRecursiveTraversalForDirectory(SymlinkBehavior.DEREFERENCE, CROSS);
  }

  @Test
  public void testRecursiveTraversalForDirectoryDontCrossFollow() throws Exception {
    assertRecursiveTraversalForDirectory(SymlinkBehavior.DEREFERENCE, DONT_CROSS);
  }

  @Test
  public void testRecursiveTraversalForDirectoryReportErrorFollow() throws Exception {
    assertRecursiveTraversalForDirectory(SymlinkBehavior.DEREFERENCE, REPORT_ERROR);
  }

  private void assertRecursiveTraversalForDirectorySymlink(
      SymlinkBehavior symlinks, PackageBoundaryMode pkgBoundaryMode) throws Exception {
    Artifact dir = getSourceArtifact("foo/dir_real");
    Artifact symlink = getSourceArtifact("foo/dir_sym");
    createFile(childOf(dir, "file.a"), "blah");
    RootedPath fileAsym = childOf(dir, "subdir/file.a.sym");
    createFile(childOf(dir, "subpkg/BUILD"), "blah");
    createFile(childOf(dir, "subpkg/file.b"), "blah");
    fileAsym.asPath().getParentDirectory().createDirectory();
    fileAsym.asPath().createSymbolicLink(PathFragment.create("../file.a"));
    symlink.getPath().createSymbolicLink(PathFragment.create("dir_real"));

    FilesetOutputSymlink outA = symlink("output-name/file.a", childOf(symlink, "file.a"));
    FilesetOutputSymlink outASym = null;
    FilesetOutputSymlink outBuild =
        symlink("output-name/subpkg/BUILD", childOf(symlink, "subpkg/BUILD"));
    FilesetOutputSymlink outB =
        symlink("output-name/subpkg/file.b", childOf(symlink, "subpkg/file.b"));
    switch (symlinks) {
      case COPY:
        outASym = symlink("output-name/subdir/file.a.sym", "../file.a");
        break;
      case DEREFERENCE:
        outASym = symlink("output-name/subdir/file.a.sym", childOf(dir, "file.a"));
        break;
      default:
        throw new IllegalStateException(symlinks.toString());
    }

    FilesetTraversalParams params =
        FilesetTraversalParamsFactory.recursiveTraversalOfDirectory(
            /*ownerLabel=*/ label("//foo"),
            /*directoryToTraverse=*/ symlink,
            PathFragment.create("output-name"),
            /*excludes=*/ null,
            /*symlinkBehaviorMode=*/ symlinks,
            /*pkgBoundaryMode=*/ pkgBoundaryMode);
    switch (pkgBoundaryMode) {
      case CROSS:
        assertSymlinksCreatedInOrder(params, outA, outASym, outBuild, outB);
        break;
      case DONT_CROSS:
        assertSymlinksCreatedInOrder(params, outA, outASym);
        break;
      case REPORT_ERROR:
        SkyKey key = FilesetEntryKey.key(params);
        EvaluationResult<SkyValue> result = eval(key);
        assertThat(result.hasError()).isTrue();
        assertThat(result.getError(key).getException())
            .hasMessageThat()
            .contains(
                "'foo/dir_sym' crosses package boundary into package rooted at foo/dir_sym/subpkg");
        break;
      default:
        throw new IllegalStateException(pkgBoundaryMode.toString());
    }
  }

  @Test
  public void testRecursiveTraversalForDirectorySymlinkNoFollowCross() throws Exception {
    assertRecursiveTraversalForDirectorySymlink(SymlinkBehavior.COPY, CROSS);
  }

  @Test
  public void testRecursiveTraversalForDirectorySymlinkNoFollowDontCross() throws Exception {
    assertRecursiveTraversalForDirectorySymlink(SymlinkBehavior.COPY, DONT_CROSS);
  }

  @Test
  public void testRecursiveTraversalForDirectorySymlinkNoFollowReportError() throws Exception {
    assertRecursiveTraversalForDirectorySymlink(SymlinkBehavior.COPY, REPORT_ERROR);
  }

  @Test
  public void testRecursiveTraversalForDirectorySymlinkFollowCross() throws Exception {
    assertRecursiveTraversalForDirectorySymlink(SymlinkBehavior.DEREFERENCE, CROSS);
  }

  @Test
  public void testRecursiveTraversalForDirectorySymlinkFollowDontCross() throws Exception {
    assertRecursiveTraversalForDirectorySymlink(SymlinkBehavior.DEREFERENCE, DONT_CROSS);
  }

  @Test
  public void testRecursiveTraversalForDirectorySymlinkFollowReportError() throws Exception {
    assertRecursiveTraversalForDirectorySymlink(SymlinkBehavior.DEREFERENCE, REPORT_ERROR);
  }

  private void assertRecursiveTraversalForPackage(
      SymlinkBehavior symlinks, PackageBoundaryMode pkgBoundaryMode) throws Exception {
    Artifact buildFile = createSourceArtifact("foo/BUILD");
    Artifact subpkgBuildFile = createSourceArtifact("foo/subpkg/BUILD");
    Artifact subpkgSymlink = getSourceArtifact("foo/subpkg_sym");

    RootedPath fileA = createFile(siblingOf(buildFile, "file.a"), "blah");
    RootedPath fileAsym = siblingOf(buildFile, "subdir/file.a.sym");
    RootedPath fileB = createFile(siblingOf(subpkgBuildFile, "file.b"), "blah");

    scratch.dir(fileAsym.asPath().getParentDirectory().getPathString());
    fileAsym.asPath().createSymbolicLink(PathFragment.create("../file.a"));
    subpkgSymlink.getPath().createSymbolicLink(PathFragment.create("subpkg"));

    FilesetOutputSymlink outBuild = symlink("output-name/BUILD", buildFile);
    FilesetOutputSymlink outA = symlink("output-name/file.a", fileA);
    FilesetOutputSymlink outAsym = null;
    FilesetOutputSymlink outSubpkgBuild = symlink("output-name/subpkg/BUILD", subpkgBuildFile);
    FilesetOutputSymlink outSubpkgB = symlink("output-name/subpkg/file.b", fileB);
    FilesetOutputSymlink outSubpkgSymBuild;
    switch (symlinks) {
      case COPY:
        outAsym = symlink("output-name/subdir/file.a.sym", "../file.a");
        outSubpkgSymBuild = symlink("output-name/subpkg_sym", "subpkg");
        break;
      case DEREFERENCE:
        outAsym = symlink("output-name/subdir/file.a.sym", fileA);
        outSubpkgSymBuild = symlink("output-name/subpkg_sym", getSourceArtifact("foo/subpkg"));
        break;
      default:
        throw new IllegalStateException(symlinks.toString());
    }

    FilesetTraversalParams params =
        FilesetTraversalParamsFactory.recursiveTraversalOfPackage(
            /*ownerLabel=*/ label("//foo"),
            /* buildFile= */ buildFile,
            PathFragment.create("output-name"),
            /*excludes=*/ null,
            /*symlinkBehaviorMode=*/ symlinks,
            /*pkgBoundaryMode=*/ pkgBoundaryMode);
    switch (pkgBoundaryMode) {
      case CROSS:
        assertSymlinksCreatedInOrder(
            params, outBuild, outA, outSubpkgSymBuild, outAsym, outSubpkgBuild, outSubpkgB);
        break;
      case DONT_CROSS:
        assertSymlinksCreatedInOrder(params, outBuild, outA, outAsym);
        break;
      case REPORT_ERROR:
        SkyKey key = FilesetEntryKey.key(params);
        EvaluationResult<SkyValue> result = eval(key);
        assertThat(result.hasError()).isTrue();
        assertThat(result.getError(key).getException())
            .hasMessageThat()
            .contains("'foo' crosses package boundary into package rooted at foo/subpkg");
        break;
      default:
        throw new IllegalStateException(pkgBoundaryMode.toString());
    }
  }

  @Test
  public void testRecursiveTraversalForPackageNoFollowCross() throws Exception {
    assertRecursiveTraversalForPackage(SymlinkBehavior.COPY, CROSS);
  }

  @Test
  public void testRecursiveTraversalForPackageNoFollowDontCross() throws Exception {
    assertRecursiveTraversalForPackage(SymlinkBehavior.COPY, DONT_CROSS);
  }

  @Test
  public void testRecursiveTraversalForPackageNoFollowReportError() throws Exception {
    assertRecursiveTraversalForPackage(SymlinkBehavior.COPY, REPORT_ERROR);
  }

  @Test
  public void testRecursiveTraversalForPackageFollowCross() throws Exception {
    assertRecursiveTraversalForPackage(SymlinkBehavior.DEREFERENCE, CROSS);
  }

  @Test
  public void testRecursiveTraversalForPackageFollowDontCross() throws Exception {
    assertRecursiveTraversalForPackage(SymlinkBehavior.DEREFERENCE, DONT_CROSS);
  }

  @Test
  public void testRecursiveTraversalForPackageFollowReportError() throws Exception {
    assertRecursiveTraversalForPackage(SymlinkBehavior.DEREFERENCE, REPORT_ERROR);
  }

  @Test
  public void testNestedFileFilesetTraversal() throws Exception {
    Artifact path1 = getSourceArtifact("foo/bar.file");
    createFile(path1, "blah");
    Artifact path2 = getSourceArtifact("foo/baz.file");
    createFile(path2, "what");
    FilesetTraversalParams inner1 =
        FilesetTraversalParamsFactory.fileTraversal(
            /*ownerLabel=*/ label("//foo"),
            /*fileToTraverse=*/ path1,
            PathFragment.create("inner-out1"),
            /*symlinkBehaviorMode=*/ SymlinkBehavior.COPY,
            /*pkgBoundaryMode=*/ DONT_CROSS);
    FilesetTraversalParams inner2 =
        FilesetTraversalParamsFactory.fileTraversal(
            /*ownerLabel=*/ label("//foo"),
            /*fileToTraverse=*/ path2,
            PathFragment.create("inner-out2"),
            /*symlinkBehaviorMode=*/ SymlinkBehavior.COPY,
            /*pkgBoundaryMode=*/ DONT_CROSS);
    FilesetTraversalParams outer =
        FilesetTraversalParamsFactory.nestedTraversal(
            /*ownerLabel=*/ label("//foo:bar"),
            /*nested=*/ ImmutableList.of(inner1, inner2),
            PathFragment.create("outer-out"),
            /*excludes=*/ null);
    assertSymlinksCreatedInOrder(
        outer,
        symlink("outer-out/inner-out1", rootedPath(path1)),
        symlink("outer-out/inner-out2", rootedPath(path2)));
  }

  @Test
  public void testMultiLevelNesting() throws Exception {
    Artifact path1 = getSourceArtifact("foo/bar.file");
    createFile(path1, "blah");
    Artifact path2 = getSourceArtifact("foo/baz.file");
    createFile(path2, "what");
    Artifact path3 = getSourceArtifact("foo/hw.file");
    createFile(path3, "hello");
    FilesetTraversalParams inner1 =
        FilesetTraversalParamsFactory.fileTraversal(
            /*ownerLabel=*/ label("//foo"),
            /*fileToTraverse=*/ path1,
            PathFragment.create("inner-out1"),
            /*symlinkBehaviorMode=*/ SymlinkBehavior.COPY,
            /*pkgBoundaryMode=*/ DONT_CROSS);
    FilesetTraversalParams inner2 =
        FilesetTraversalParamsFactory.fileTraversal(
            /*ownerLabel=*/ label("//foo"),
            /*fileToTraverse=*/ path2,
            PathFragment.create("inner-out2"),
            /*symlinkBehaviorMode=*/ SymlinkBehavior.COPY,
            /*pkgBoundaryMode=*/ DONT_CROSS);
    FilesetTraversalParams middle1 =
        FilesetTraversalParamsFactory.nestedTraversal(
            /*ownerLabel=*/ label("//foo:middle1"),
            /*nested=*/ ImmutableList.of(inner1, inner2),
            PathFragment.create("middle-out1"),
            /*excludes=*/ null);

    FilesetTraversalParams inner3 =
        FilesetTraversalParamsFactory.fileTraversal(
            /*ownerLabel=*/ label("//foo:inner3"),
            /*fileToTraverse=*/ path3,
            PathFragment.create("inner-out3"),
            /*symlinkBehaviorMode=*/ SymlinkBehavior.COPY,
            /*pkgBoundaryMode=*/ DONT_CROSS);
    FilesetTraversalParams middle2 =
        FilesetTraversalParamsFactory.nestedTraversal(
            /*ownerLabel=*/ label("//foo:middle2"),
            /*nested=*/ ImmutableList.of(inner3),
            PathFragment.create("middle-out2"),
            /*excludes=*/ null);

    FilesetTraversalParams outer =
        FilesetTraversalParamsFactory.nestedTraversal(
            /*ownerLabel=*/ label("//foo:bar"),
            /*nested=*/ ImmutableList.of(middle1, middle2),
            PathFragment.create("outer-out"),
            /*excludes=*/ null);
    assertSymlinksCreatedInOrder(
        outer,
        symlink("outer-out/middle-out1/inner-out1", rootedPath(path1)),
        symlink("outer-out/middle-out1/inner-out2", rootedPath(path2)),
        symlink("outer-out/middle-out2/inner-out3", rootedPath(path3)));
  }

  private void assertNestedRecursiveFilesetTraversal(boolean useInnerDir) throws Exception {
    Artifact dir = getSourceArtifact("foo/dir");
    RootedPath fileA = createFile(childOf(dir, "file.a"), "hello");
    RootedPath fileB = createFile(childOf(dir, "file.b"), "hello");
    RootedPath fileC = createFile(childOf(dir, "sub/file.c"), "world");

    FilesetTraversalParams inner =
        FilesetTraversalParamsFactory.recursiveTraversalOfDirectory(
            /*ownerLabel=*/ label("//foo"),
            /*directoryToTraverse=*/ dir,
            PathFragment.create(useInnerDir ? "inner-dir" : ""),
            /*excludes=*/ null,
            /*symlinkBehaviorMode=*/ SymlinkBehavior.COPY,
            /*pkgBoundaryMode=*/ DONT_CROSS);
    FilesetTraversalParams outer =
        FilesetTraversalParamsFactory.nestedTraversal(
            /*ownerLabel=*/ label("//foo"),
            /*nested=*/ ImmutableList.of(inner),
            PathFragment.create("outer-dir"),
            /*excludes=*/ ImmutableSet.of("file.a", "sub/file.c"));

    if (useInnerDir) {
      assertSymlinksCreatedInOrder(
          outer,
          // no file is excluded, since no files from "inner" are top-level in the outer Fileset
          symlink("outer-dir/inner-dir/file.a", fileA),
          symlink("outer-dir/inner-dir/file.b", fileB),
          symlink("outer-dir/inner-dir/sub/file.c", fileC)); // only top-level files are excluded
    } else {
      assertSymlinksCreatedInOrder(
          outer,
          // file.a can be excluded because it's top-level (there's no output directory for "inner")
          symlink("outer-dir/file.b", fileB),
          symlink("outer-dir/sub/file.c", fileC)); // only top-level files could be excluded
    }
  }

  @Test
  public void testNestedRecursiveFilesetTraversalWithInnerDestDir() throws Exception {
    assertNestedRecursiveFilesetTraversal(true);
  }

  @Test
  public void testNestedRecursiveFilesetTraversalWithoutInnerDestDir() throws Exception {
    assertNestedRecursiveFilesetTraversal(false);
  }

  @Test
  public void testFileTraversalForDanglingSymlink() throws Exception {
    Artifact linkName = getSourceArtifact("foo/dangling.sym");
    RootedPath linkTarget = createFile(siblingOf(linkName, "target.file"), "blah");
    linkName.getPath().createSymbolicLink(PathFragment.create("target.file"));
    linkTarget.asPath().delete();

    FilesetTraversalParams params =
        FilesetTraversalParamsFactory.fileTraversal(
            /*ownerLabel=*/ label("//foo"),
            /*fileToTraverse=*/ linkName,
            PathFragment.create("output-name"),
            /*symlinkBehaviorMode=*/ SymlinkBehavior.COPY,
            /*pkgBoundaryMode=*/ DONT_CROSS);
    assertSymlinksCreatedInOrder(params); // expect empty results
  }

  private void assertExclusionOfDanglingSymlink(SymlinkBehavior symlinkBehavior) throws Exception {
    Artifact buildFile = getSourceArtifact("foo/BUILD");
    createFile(buildFile);

    Artifact linkName = getSourceArtifact("foo/file.sym");
    Artifact linkTarget = getSourceArtifact("foo/file.actual");
    createFile(linkTarget);
    linkName.getPath().createSymbolicLink(PathFragment.create("file.actual"));

    // Ensure the symlink and its target would be included if they weren't explicitly excluded.
    FilesetTraversalParams params =
        FilesetTraversalParamsFactory.recursiveTraversalOfPackage(
            /* ownerLabel */ label("//foo"),
            /* buildFile */ buildFile,
            PathFragment.create("output-name"),
            /* excludes */ ImmutableSet.<String>of(),
            /* symlinkBehaviorMode */ symlinkBehavior,
            /* pkgBoundaryMode */ PackageBoundaryMode.DONT_CROSS);
    assertSymlinksCreatedInOrder(
        params,
        symlink("output-name/BUILD", buildFile),
        symlink("output-name/file.actual", linkTarget),
        symlinkBehavior == SymlinkBehavior.COPY
            ? symlink("output-name/file.sym", "file.actual")
            : symlink("output-name/file.sym", linkTarget));

    // Delete the symlink's target to make it dangling.
    // Exclude the symlink and make sure it's not included.
    linkTarget.getPath().delete();
    differencer.invalidate(ImmutableList.of(FileStateValue.key(rootedPath(linkTarget))));
    params =
        FilesetTraversalParamsFactory.recursiveTraversalOfPackage(
            /* ownerLabel */ label("//foo"),
            /* buildFile */ buildFile,
            PathFragment.create("output-name"),
            /* excludes */ ImmutableSet.of("file.sym"),
            /* symlinkBehaviorMode */ symlinkBehavior,
            /* pkgBoundaryMode */ PackageBoundaryMode.DONT_CROSS);
    assertSymlinksCreatedInOrder(params, symlink("output-name/BUILD", buildFile));
  }

  @Test
  public void testExclusionOfDanglingSymlinkWithSymlinkModeCopy() throws Exception {
    assertExclusionOfDanglingSymlink(SymlinkBehavior.COPY);
  }

  @Test
  public void testExclusionOfDanglingSymlinkWithSymlinkModeDereference() throws Exception {
    assertExclusionOfDanglingSymlink(SymlinkBehavior.DEREFERENCE);
  }

  @Test
  public void testExcludes() throws Exception {
    Artifact buildFile = getSourceArtifact("foo/BUILD");
    createFile(buildFile);
    Artifact outerFile = getSourceArtifact("foo/outerfile.txt");
    createFile(outerFile);
    Artifact innerFile = getSourceArtifact("foo/dir/innerfile.txt");
    createFile(innerFile);

    FilesetTraversalParams params =
        FilesetTraversalParamsFactory.recursiveTraversalOfPackage(
            /* ownerLabel */ label("//foo"),
            /* buildFile */ buildFile,
            PathFragment.create("output-name"),
            /* excludes */ ImmutableSet.of(),
            /* symlinkBehaviorMode */ SymlinkBehavior.COPY,
            /* pkgBoundaryMode */ PackageBoundaryMode.DONT_CROSS);
    assertSymlinksCreatedInOrder(
        params,
        symlink("output-name/BUILD", buildFile),
        symlink("output-name/outerfile.txt", outerFile),
        symlink("output-name/dir/innerfile.txt", innerFile));

    // Make sure the file within the excluded directory is no longer present.
    params = FilesetTraversalParamsFactory.recursiveTraversalOfPackage(
            /* ownerLabel */ label("//foo"),
            /* buildFile */ buildFile,
            PathFragment.create("output-name"),
            /* excludes */ ImmutableSet.of("dir"),
            /* symlinkBehaviorMode */ SymlinkBehavior.COPY,
            /* pkgBoundaryMode */ PackageBoundaryMode.DONT_CROSS);
    assertSymlinksCreatedInOrder(
        params,
        symlink("output-name/BUILD", buildFile),
        symlink("output-name/outerfile.txt", outerFile));
  }

  @Test
  public void testFileTraversalForNonExistentFile() throws Exception {
    Artifact path = getSourceArtifact("foo/non-existent");
    FilesetTraversalParams params =
        FilesetTraversalParamsFactory.fileTraversal(
            /*ownerLabel=*/ label("//foo"),
            /*fileToTraverse=*/ path,
            PathFragment.create("output-name"),
            /*symlinkBehaviorMode=*/ SymlinkBehavior.COPY,
            /*pkgBoundaryMode=*/ DONT_CROSS);
    assertSymlinksCreatedInOrder(params); // expect empty results
  }

  @Test
  public void testRecursiveTraversalForDanglingSymlink() throws Exception {
    Artifact linkName = getSourceArtifact("foo/dangling.sym");
    RootedPath linkTarget = createFile(siblingOf(linkName, "target.file"), "blah");
    linkName.getPath().createSymbolicLink(PathFragment.create("target.file"));
    linkTarget.asPath().delete();

    FilesetTraversalParams params =
        FilesetTraversalParamsFactory.recursiveTraversalOfDirectory(
            /*ownerLabel=*/ label("//foo"),
            /*directoryToTraverse=*/ linkName,
            PathFragment.create("output-name"),
            /*excludes=*/ null,
            /*symlinkBehaviorMode=*/ SymlinkBehavior.COPY,
            /*pkgBoundaryMode=*/ DONT_CROSS);
    assertSymlinksCreatedInOrder(params); // expect empty results
  }

  @Test
  public void testRecursiveTraversalForNonExistentFile() throws Exception {
    Artifact path = getSourceArtifact("foo/non-existent");

    FilesetTraversalParams params =
        FilesetTraversalParamsFactory.recursiveTraversalOfDirectory(
            /*ownerLabel=*/ label("//foo"),
            /*directoryToTraverse=*/ path,
            PathFragment.create("output-name"),
            /*excludes=*/ null,
            /*symlinkBehaviorMode=*/ SymlinkBehavior.COPY,
            /*pkgBoundaryMode=*/ DONT_CROSS);
    assertSymlinksCreatedInOrder(params); // expect empty results
  }

  /**
   * Tests that the fingerprint is a function of all arguments of the factory method.
   *
   * <p>Implementations must provide:
   * <ul>
   * <li>two different values (a domain) for each argument of the factory method and whether or not
   * it is expected to influence the fingerprint
   * <li>a way to instantiate {@link FilesetTraversalParams} with a given set of arguments from the
   * specified domains
   * </ul>
   *
   * <p>The tests will instantiate pairs of {@link FilesetTraversalParams} objects with only a given
   * attribute differing, and observe whether the fingerprints differ (if they are expected to) or
   * are the same (otherwise).
   */
  private abstract static class FingerprintTester {
    private final Map<String, Domain> domains;

    FingerprintTester(Map<String, Domain> domains) {
      this.domains = domains;
    }

    abstract FilesetTraversalParams create(Map<String, ?> kwArgs) throws Exception;

    private Map<String, ?> getDefaultArgs() {
      return getKwArgs(null);
    }

    private Map<String, ?> getKwArgs(@Nullable String useAlternateFor) {
      Map<String, Object> values = new HashMap<>();
      for (Map.Entry<String, Domain> d : domains.entrySet()) {
        values.put(
            d.getKey(),
            d.getKey().equals(useAlternateFor) ? d.getValue().valueA : d.getValue().valueB);
      }
      return values;
    }

    public void doTest() throws Exception {
      Fingerprint fp = new Fingerprint();

      create(getDefaultArgs()).fingerprint(fp);
      String primary = fp.hexDigestAndReset();

      for (String argName : domains.keySet()) {
        create(getKwArgs(argName)).fingerprint(fp);
        String secondary = fp.hexDigestAndReset();

        if (domains.get(argName).includedInFingerprint) {
          assertWithMessage(
                  "Argument '"
                      + argName
                      + "' was expected to be included in the"
                      + " fingerprint, but wasn't")
              .that(primary)
              .isNotEqualTo(secondary);
        } else {
          assertWithMessage(
                  "Argument '"
                      + argName
                      + "' was expected not to be included in the"
                      + " fingerprint, but was")
              .that(primary)
              .isEqualTo(secondary);
        }
      }
    }
  }

  private static final class Domain {
    boolean includedInFingerprint;
    Object valueA;
    Object valueB;

    Domain(boolean includedInFingerprint, Object valueA, Object valueB) {
      this.includedInFingerprint = includedInFingerprint;
      this.valueA = valueA;
      this.valueB = valueB;
    }
  }

  private static Domain partOfFingerprint(Object valueA, Object valueB) {
    return new Domain(true, valueA, valueB);
  }

  private static Domain notPartOfFingerprint(Object valueA, Object valueB) {
    return new Domain(false, valueA, valueB);
  }

  @Test
  public void testFingerprintOfFileTraversal() throws Exception {
    new FingerprintTester(
        ImmutableMap.<String, Domain>of(
            "ownerLabel", notPartOfFingerprint("//foo", "//bar"),
            "fileToTraverse", partOfFingerprint("foo/file.a", "bar/file.b"),
            "destPath", partOfFingerprint("out1", "out2"),
            "symlinkBehaviorMode",
                partOfFingerprint(SymlinkBehavior.COPY, SymlinkBehavior.DEREFERENCE),
            "pkgBoundaryMode", partOfFingerprint(CROSS, DONT_CROSS))) {
      @Override
      FilesetTraversalParams create(Map<String, ?> kwArgs) throws Exception {
        return FilesetTraversalParamsFactory.fileTraversal(
            label((String) kwArgs.get("ownerLabel")),
            getSourceArtifact((String) kwArgs.get("fileToTraverse")),
            PathFragment.create((String) kwArgs.get("destPath")),
            ((SymlinkBehavior) kwArgs.get("symlinkBehaviorMode")),
            (PackageBoundaryMode) kwArgs.get("pkgBoundaryMode"));
      }
    }.doTest();
  }

  @Test
  public void testFingerprintOfDirectoryTraversal() throws Exception {
    new FingerprintTester(
        ImmutableMap.<String, Domain>builder()
            .put("ownerLabel", notPartOfFingerprint("//foo", "//bar"))
            .put("directoryToTraverse", partOfFingerprint("foo/dir_a", "bar/dir_b"))
            .put("destPath", partOfFingerprint("out1", "out2"))
            .put(
                "excludes",
                partOfFingerprint(ImmutableSet.<String>of(), ImmutableSet.<String>of("blah")))
            .put(
                "symlinkBehaviorMode",
                partOfFingerprint(SymlinkBehavior.COPY, SymlinkBehavior.DEREFERENCE))
            .put("pkgBoundaryMode", partOfFingerprint(CROSS, DONT_CROSS))
            .build()) {
      @SuppressWarnings("unchecked")
      @Override
      FilesetTraversalParams create(Map<String, ?> kwArgs) throws Exception {
        return FilesetTraversalParamsFactory.recursiveTraversalOfDirectory(
            label((String) kwArgs.get("ownerLabel")),
            getSourceArtifact((String) kwArgs.get("directoryToTraverse")),
            PathFragment.create((String) kwArgs.get("destPath")),
            (Set<String>) kwArgs.get("excludes"),
            ((SymlinkBehavior) kwArgs.get("symlinkBehaviorMode")),
            (PackageBoundaryMode) kwArgs.get("pkgBoundaryMode"));
      }
    }.doTest();
  }

  @Test
  public void testFingerprintOfPackageTraversal() throws Exception {
    new FingerprintTester(
        ImmutableMap.<String, Domain>builder()
            .put("ownerLabel", notPartOfFingerprint("//foo", "//bar"))
            .put("buildFile", partOfFingerprint("foo/BUILD", "bar/BUILD"))
            .put("destPath", partOfFingerprint("out1", "out2"))
            .put(
                "excludes",
                partOfFingerprint(ImmutableSet.<String>of(), ImmutableSet.<String>of("blah")))
            .put(
                "symlinkBehaviorMode",
                partOfFingerprint(SymlinkBehavior.COPY, SymlinkBehavior.DEREFERENCE))
            .put("pkgBoundaryMode", partOfFingerprint(CROSS, DONT_CROSS))
            .build()) {
      @SuppressWarnings("unchecked")
      @Override
      FilesetTraversalParams create(Map<String, ?> kwArgs) throws Exception {
        return FilesetTraversalParamsFactory.recursiveTraversalOfPackage(
            label((String) kwArgs.get("ownerLabel")),
            getSourceArtifact((String) kwArgs.get("buildFile")),
            PathFragment.create((String) kwArgs.get("destPath")),
            (Set<String>) kwArgs.get("excludes"),
            ((SymlinkBehavior) kwArgs.get("symlinkBehaviorMode")),
            (PackageBoundaryMode) kwArgs.get("pkgBoundaryMode"));
      }
    }.doTest();
  }

  @Test
  public void testFingerprintOfNestedTraversal() throws Exception {
    FilesetTraversalParams n1 =
        FilesetTraversalParamsFactory.fileTraversal(
            /*ownerLabel=*/ label("//blah"),
            /*fileToTraverse=*/ getSourceArtifact("blah/file.a"),
            PathFragment.create("output-name"),
            /*symlinkBehaviorMode=*/ SymlinkBehavior.COPY,
            /*pkgBoundaryMode=*/ DONT_CROSS);

    FilesetTraversalParams n2 =
        FilesetTraversalParamsFactory.fileTraversal(
            /*ownerLabel=*/ label("//blah"),
            /*fileToTraverse=*/ getSourceArtifact("meow/file.b"),
            PathFragment.create("output-name"),
            /*symlinkBehaviorMode=*/ SymlinkBehavior.COPY,
            /*pkgBoundaryMode=*/ DONT_CROSS);
    ImmutableList<FilesetTraversalParams> nested1 = ImmutableList.of(n1, n2);

    FilesetTraversalParams n3 =
        FilesetTraversalParamsFactory.fileTraversal(
            /*ownerLabel=*/ label("//blah"),
            /*fileToTraverse=*/ getSourceArtifact("brrr/file.c"),
            PathFragment.create("output-name"),
            /*symlinkBehaviorMode=*/ SymlinkBehavior.COPY,
            /*pkgBoundaryMode=*/ DONT_CROSS);

    FilesetTraversalParams n4 =
        FilesetTraversalParamsFactory.fileTraversal(
            /*ownerLabel=*/ label("//blah"),
            /*fileToTraverse=*/ getSourceArtifact("hurr/file.d"),
            PathFragment.create("output-name"),
            /*symlinkBehaviorMode=*/ SymlinkBehavior.COPY,
            /*pkgBoundaryMode=*/ DONT_CROSS);
    ImmutableList<FilesetTraversalParams> nested2 = ImmutableList.of(n3, n4);

    new FingerprintTester(
        ImmutableMap.<String, Domain>of(
            "ownerLabel", notPartOfFingerprint("//foo", "//bar"),
            "nested", partOfFingerprint(nested1, nested2),
            "destDir", partOfFingerprint("out1", "out2"),
            "excludes",
                partOfFingerprint(ImmutableSet.<String>of(), ImmutableSet.<String>of("x")))) {
      @SuppressWarnings("unchecked")
      @Override
      FilesetTraversalParams create(Map<String, ?> kwArgs) throws Exception {
        return FilesetTraversalParamsFactory.nestedTraversal(
            label((String) kwArgs.get("ownerLabel")),
            (ImmutableList<FilesetTraversalParams>) kwArgs.get("nested"),
            PathFragment.create((String) kwArgs.get("destDir")),
            (Set<String>) kwArgs.get("excludes"));
      }
    }.doTest();
  }
}
