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
import static com.google.devtools.build.lib.actions.FilesetTraversalParams.PackageBoundaryMode.CROSS;
import static com.google.devtools.build.lib.actions.FilesetTraversalParams.PackageBoundaryMode.DONT_CROSS;
import static com.google.devtools.build.lib.actions.FilesetTraversalParams.PackageBoundaryMode.REPORT_ERROR;
import static com.google.devtools.build.lib.skyframe.RecursiveFilesystemTraversalValue.ResolvedFileFactoryForTesting.danglingSymlinkForTesting;
import static com.google.devtools.build.lib.skyframe.RecursiveFilesystemTraversalValue.ResolvedFileFactoryForTesting.regularFileForTesting;
import static com.google.devtools.build.lib.skyframe.RecursiveFilesystemTraversalValue.ResolvedFileFactoryForTesting.symlinkToDirectoryForTesting;
import static com.google.devtools.build.lib.skyframe.RecursiveFilesystemTraversalValue.ResolvedFileFactoryForTesting.symlinkToFileForTesting;

import com.google.common.base.Preconditions;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.PackageBoundaryMode;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.skyframe.RecursiveFilesystemTraversalFunction.FileOperationException;
import com.google.devtools.build.lib.skyframe.RecursiveFilesystemTraversalValue.ResolvedFile;
import com.google.devtools.build.lib.skyframe.RecursiveFilesystemTraversalValue.TraversalRequest;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver;
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
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Pattern;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RecursiveFilesystemTraversalFunction}. */
@RunWith(JUnit4.class)
public final class RecursiveFilesystemTraversalFunctionTest extends FoundationTestCase {
  private RecordingEvaluationProgressReceiver progressReceiver;
  private MemoizingEvaluator evaluator;
  private SequentialBuildDriver driver;
  private RecordingDifferencer differencer;
  private AtomicReference<PathPackageLocator> pkgLocator;

  @Before
  public final void setUp() throws Exception  {
    AnalysisMock analysisMock = AnalysisMock.get();
    pkgLocator =
        new AtomicReference<>(
            new PathPackageLocator(
                outputBase,
                ImmutableList.of(rootDirectory),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    AtomicReference<ImmutableSet<PackageIdentifier>> deletedPackages =
        new AtomicReference<>(ImmutableSet.<PackageIdentifier>of());
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(rootDirectory, outputBase),
            rootDirectory,
            analysisMock.getProductName());
    ExternalFilesHelper externalFilesHelper = new ExternalFilesHelper(
        pkgLocator, ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS, directories);

    ConfiguredRuleClassProvider ruleClassProvider = analysisMock.createRuleClassProvider();
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
    skyFunctions.put(SkyFunctions.PACKAGE,
        new PackageFunction(null, null, null, null, null, null, null));
    skyFunctions.put(SkyFunctions.WORKSPACE_AST, new WorkspaceASTFunction(ruleClassProvider));
    skyFunctions.put(
        SkyFunctions.WORKSPACE_FILE,
        new WorkspaceFileFunction(
            ruleClassProvider,
            analysisMock
                .getPackageFactoryBuilderForTesting(directories)
                .build(ruleClassProvider, scratch.getFileSystem()),
            directories));
    skyFunctions.put(SkyFunctions.EXTERNAL_PACKAGE, new ExternalPackageFunction());
    skyFunctions.put(SkyFunctions.LOCAL_REPOSITORY_LOOKUP, new LocalRepositoryLookupFunction());
    skyFunctions.put(
        SkyFunctions.FILE_SYMLINK_INFINITE_EXPANSION_UNIQUENESS,
        new FileSymlinkInfiniteExpansionUniquenessFunction());
    skyFunctions.put(
        SkyFunctions.FILE_SYMLINK_CYCLE_UNIQUENESS, new FileSymlinkCycleUniquenessFunction());

    progressReceiver = new RecordingEvaluationProgressReceiver();
    differencer = new SequencedRecordingDifferencer();
    evaluator = new InMemoryMemoizingEvaluator(skyFunctions, differencer, progressReceiver);
    driver = new SequentialBuildDriver(evaluator);
    PrecomputedValue.BUILD_ID.set(differencer, UUID.randomUUID());
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator.get());
    PrecomputedValue.BLACKLISTED_PACKAGE_PREFIXES_FILE.set(
        differencer, PathFragment.EMPTY_FRAGMENT);
  }

  private Artifact sourceArtifact(String path) {
    return new Artifact(PathFragment.create(path), Root.asSourceRoot(rootDirectory));
  }

  private Artifact sourceArtifactUnderPackagePath(String path, String packagePath) {
    return new Artifact(
        PathFragment.create(path), Root.asSourceRoot(rootDirectory.getRelative(packagePath)));
  }

  private Artifact derivedArtifact(String path) {
    PathFragment execPath = PathFragment.create("out").getRelative(path);
    Path fullPath = rootDirectory.getRelative(execPath);
    Artifact output =
        new Artifact(
            fullPath,
            Root.asDerivedRoot(rootDirectory, rootDirectory.getRelative("out")),
            execPath);
    return output;
  }

  private static RootedPath rootedPath(Artifact artifact) {
    return RootedPath.toRootedPath(artifact.getRoot().getPath(), artifact.getRootRelativePath());
  }

  private RootedPath rootedPath(String path, String packagePath) {
    return RootedPath.toRootedPath(
        rootDirectory.getRelative(packagePath), PathFragment.create(path));
  }

  private static RootedPath childOf(Artifact artifact, String relative) {
    return RootedPath.toRootedPath(
        artifact.getRoot().getPath(), artifact.getRootRelativePath().getRelative(relative));
  }

  private static RootedPath childOf(RootedPath path, String relative) {
    return RootedPath.toRootedPath(path.getRoot(), path.getRelativePath().getRelative(relative));
  }

  private static RootedPath parentOf(RootedPath path) {
    PathFragment parent = Preconditions.checkNotNull(path.getRelativePath().getParentDirectory());
    return RootedPath.toRootedPath(path.getRoot(), parent);
  }

  private static RootedPath siblingOf(RootedPath path, String relative) {
    PathFragment parent = Preconditions.checkNotNull(path.getRelativePath().getParentDirectory());
    return RootedPath.toRootedPath(path.getRoot(), parent.getRelative(relative));
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
    scratch.dir(parentOf(path).asPath().getPathString());
    createFile(path.asPath(), contents);
    return path;
  }

  private static TraversalRequest fileLikeRoot(Artifact file, PackageBoundaryMode pkgBoundaryMode) {
    return new TraversalRequest(
        rootedPath(file), !file.isSourceArtifact(), pkgBoundaryMode, false, null, null);
  }

  private static TraversalRequest pkgRoot(
      RootedPath pkgDirectory, PackageBoundaryMode pkgBoundaryMode) {
    return new TraversalRequest(pkgDirectory, false, pkgBoundaryMode, true, null, null);
  }

  private <T extends SkyValue> EvaluationResult<T> eval(SkyKey key) throws Exception {
    return driver.evaluate(
        ImmutableList.of(key),
        false,
        SkyframeExecutor.DEFAULT_THREAD_COUNT,
        NullEventHandler.INSTANCE);
  }

  private RecursiveFilesystemTraversalValue evalTraversalRequest(TraversalRequest params)
      throws Exception {
    SkyKey key = rftvSkyKey(params);
    EvaluationResult<RecursiveFilesystemTraversalValue> result = eval(key);
    assertThat(result.hasError()).isFalse();
    return result.get(key);
  }

  private static SkyKey rftvSkyKey(TraversalRequest params) {
    return RecursiveFilesystemTraversalValue.key(params);
  }

  /**
   * Asserts that the requested SkyValue can be built and results in the expected set of files.
   *
   * <p>The metadata of files is ignored in comparing the actual results with the expected ones.
   * The returned object however contains the actual metadata.
   */
  @SafeVarargs
  private final RecursiveFilesystemTraversalValue traverseAndAssertFiles(
      TraversalRequest params, ResolvedFile... expectedFilesIgnoringMetadata) throws Exception {
    RecursiveFilesystemTraversalValue result = evalTraversalRequest(params);
    Set<ResolvedFile> actual = new HashSet<>();
    for (ResolvedFile act : result.getTransitiveFiles()) {
      // Strip metadata so only the type and path of the objects are compared.
      actual.add(act.stripMetadataForTesting());
    }
    // First just assert equality of the keys, so in case of a mismatch the error message is easier
    // to read.
    assertThat(actual).containsExactly((Object[]) expectedFilesIgnoringMetadata);

    // The returned object still has the unstripped metadata.
    return result;
  }

  private void appendToFile(RootedPath rootedPath, String content) throws Exception {
    Path path = rootedPath.asPath();
    if (path.exists()) {
      try (OutputStream os = path.getOutputStream(/*append=*/ true)) {
        os.write(content.getBytes(StandardCharsets.UTF_8));
      }
      differencer.invalidate(ImmutableList.of(FileStateValue.key(rootedPath)));
    } else {
      createFile(path, content);
    }
  }

  private void appendToFile(Artifact file, String content) throws Exception {
    appendToFile(rootedPath(file), content);
  }

  private void invalidateDirectory(RootedPath path) {
    differencer.invalidate(ImmutableList.of(DirectoryListingStateValue.key(path)));
  }

  private void invalidateDirectory(Artifact directoryArtifact) {
    invalidateDirectory(rootedPath(directoryArtifact));
  }

  private static final class RecordingEvaluationProgressReceiver
      extends EvaluationProgressReceiver.NullEvaluationProgressReceiver {
    Set<SkyKey> invalidations;
    Set<SkyValue> evaluations;

    RecordingEvaluationProgressReceiver() {
      clear();
    }

    void clear() {
      invalidations = Sets.newConcurrentHashSet();
      evaluations = Sets.newConcurrentHashSet();
    }

    @Override
    public void invalidated(SkyKey skyKey, InvalidationState state) {
      invalidations.add(skyKey);
    }

    @Override
    public void evaluated(
        SkyKey skyKey, Supplier<SkyValue> skyValueSupplier, EvaluationState state) {
      SkyValue value = skyValueSupplier.get();
      if (value != null) {
        evaluations.add(value);
      }
    }
  }

  private void assertTraversalRootHashesAre(
      boolean equal, RecursiveFilesystemTraversalValue a, RecursiveFilesystemTraversalValue b)
          throws Exception {
    if (equal) {
      assertThat(a.getResolvedRoot().get().hashCode())
          .isEqualTo(b.getResolvedRoot().get().hashCode());
    } else {
      assertThat(a.getResolvedRoot().get().hashCode())
          .isNotEqualTo(b.getResolvedRoot().get().hashCode());
    }
  }

  private void assertTraversalRootHashesAreEqual(
      RecursiveFilesystemTraversalValue a, RecursiveFilesystemTraversalValue b) throws Exception {
    assertTraversalRootHashesAre(true, a, b);
  }

  private void assertTraversalRootHashesAreNotEqual(
      RecursiveFilesystemTraversalValue a, RecursiveFilesystemTraversalValue b) throws Exception {
    assertTraversalRootHashesAre(false, a, b);
  }

  private void assertTraversalOfFile(Artifact rootArtifact) throws Exception {
    TraversalRequest traversalRoot = fileLikeRoot(rootArtifact, DONT_CROSS);
    RootedPath rootedPath = createFile(rootedPath(rootArtifact), "foo");

    // Assert that the SkyValue is built and looks right.
    ResolvedFile expected = regularFileForTesting(rootedPath);
    RecursiveFilesystemTraversalValue v1 = traverseAndAssertFiles(traversalRoot, expected);
    assertThat(progressReceiver.invalidations).isEmpty();
    assertThat(progressReceiver.evaluations).contains(v1);
    progressReceiver.clear();

    // Edit the file and verify that the value is rebuilt.
    appendToFile(rootArtifact, "bar");
    RecursiveFilesystemTraversalValue v2 = traverseAndAssertFiles(traversalRoot, expected);
    assertThat(progressReceiver.invalidations).contains(rftvSkyKey(traversalRoot));
    assertThat(progressReceiver.evaluations).contains(v2);
    assertThat(v2).isNotEqualTo(v1);
    assertTraversalRootHashesAreNotEqual(v1, v2);

    progressReceiver.clear();
  }

  @Test
  public void testTraversalOfSourceFile() throws Exception {
    assertTraversalOfFile(sourceArtifact("foo/bar.txt"));
  }

  @Test
  public void testTraversalOfGeneratedFile() throws Exception {
    assertTraversalOfFile(derivedArtifact("foo/bar.txt"));
  }

  @Test
  public void testTraversalOfSymlinkToFile() throws Exception {
    Artifact linkNameArtifact = sourceArtifact("foo/baz/qux.sym");
    Artifact linkTargetArtifact = sourceArtifact("foo/bar/baz.txt");
    PathFragment linkValue = PathFragment.create("../bar/baz.txt");
    TraversalRequest traversalRoot = fileLikeRoot(linkNameArtifact, DONT_CROSS);
    createFile(linkTargetArtifact);
    scratch.dir(linkNameArtifact.getExecPath().getParentDirectory().getPathString());
    rootDirectory.getRelative(linkNameArtifact.getExecPath()).createSymbolicLink(linkValue);

    // Assert that the SkyValue is built and looks right.
    RootedPath symlinkNamePath = rootedPath(linkNameArtifact);
    RootedPath symlinkTargetPath = rootedPath(linkTargetArtifact);
    ResolvedFile expected = symlinkToFileForTesting(symlinkTargetPath, symlinkNamePath, linkValue);
    RecursiveFilesystemTraversalValue v1 = traverseAndAssertFiles(traversalRoot, expected);
    assertThat(progressReceiver.invalidations).isEmpty();
    assertThat(progressReceiver.evaluations).contains(v1);
    progressReceiver.clear();

    // Edit the target of the symlink and verify that the value is rebuilt.
    appendToFile(linkTargetArtifact, "bar");
    RecursiveFilesystemTraversalValue v2 = traverseAndAssertFiles(traversalRoot, expected);
    assertThat(progressReceiver.invalidations).contains(rftvSkyKey(traversalRoot));
    assertThat(progressReceiver.evaluations).contains(v2);
    assertThat(v2).isNotEqualTo(v1);
    assertTraversalRootHashesAreNotEqual(v1, v2);
  }

  @Test
  public void testTraversalOfTransitiveSymlinkToFile() throws Exception {
    Artifact directLinkArtifact = sourceArtifact("direct/file.sym");
    Artifact transitiveLinkArtifact = sourceArtifact("transitive/sym.sym");
    RootedPath fileA = createFile(rootedPath(sourceArtifact("a/file.a")));
    RootedPath directLink = rootedPath(directLinkArtifact);
    RootedPath transitiveLink = rootedPath(transitiveLinkArtifact);
    PathFragment directLinkPath = PathFragment.create("../a/file.a");
    PathFragment transitiveLinkPath = PathFragment.create("../direct/file.sym");

    parentOf(directLink).asPath().createDirectory();
    parentOf(transitiveLink).asPath().createDirectory();
    directLink.asPath().createSymbolicLink(directLinkPath);
    transitiveLink.asPath().createSymbolicLink(transitiveLinkPath);

    traverseAndAssertFiles(
        fileLikeRoot(directLinkArtifact, DONT_CROSS),
        symlinkToFileForTesting(fileA, directLink, directLinkPath));

    traverseAndAssertFiles(
        fileLikeRoot(transitiveLinkArtifact, DONT_CROSS),
        symlinkToFileForTesting(fileA, transitiveLink, transitiveLinkPath));
  }

  private void assertTraversalOfDirectory(Artifact directoryArtifact) throws Exception {
    // Create files under the directory.
    // Use the root + root-relative path of the rootArtifact to create these files, rather than
    // using the rootDirectory + execpath of the rootArtifact. The resulting paths are the same
    // but the RootedPaths are different:
    // in the 1st case, it is: RootedPath(/root/execroot, relative), in the second it is
    // in the 2nd case, it is: RootedPath(/root, execroot/relative).
    // Creating the files will also create the parent directories.
    RootedPath file1 = createFile(childOf(directoryArtifact, "bar.txt"));
    RootedPath file2 = createFile(childOf(directoryArtifact, "baz/qux.txt"));

    TraversalRequest traversalRoot = fileLikeRoot(directoryArtifact, DONT_CROSS);

    // Assert that the SkyValue is built and looks right.
    ResolvedFile expected1 = regularFileForTesting(file1);
    ResolvedFile expected2 = regularFileForTesting(file2);
    RecursiveFilesystemTraversalValue v1 =
        traverseAndAssertFiles(traversalRoot, expected1, expected2);
    assertThat(progressReceiver.invalidations).isEmpty();
    assertThat(progressReceiver.evaluations).contains(v1);
    progressReceiver.clear();

    // Add a new file to the directory and see that the value is rebuilt.
    RootedPath file3 = createFile(childOf(directoryArtifact, "foo.txt"));
    invalidateDirectory(directoryArtifact);
    ResolvedFile expected3 = regularFileForTesting(file3);
    RecursiveFilesystemTraversalValue v2 =
        traverseAndAssertFiles(traversalRoot, expected1, expected2, expected3);
    assertThat(progressReceiver.invalidations).contains(rftvSkyKey(traversalRoot));
    assertThat(progressReceiver.evaluations).contains(v2);
    // Directories always have the same hash code, but that is fine because their contents are also
    // part of the RecursiveFilesystemTraversalValue, so v1 and v2 are unequal.
    assertThat(v2).isNotEqualTo(v1);
    assertTraversalRootHashesAreEqual(v1, v2);
    progressReceiver.clear();

    // Edit a file in the directory and see that the value is rebuilt.
    appendToFile(file1, "bar");
    RecursiveFilesystemTraversalValue v3 =
        traverseAndAssertFiles(traversalRoot, expected1, expected2, expected3);
    assertThat(progressReceiver.invalidations).contains(rftvSkyKey(traversalRoot));
    assertThat(progressReceiver.evaluations).contains(v3);
    assertThat(v3).isNotEqualTo(v2);
    // Directories always have the same hash code, but that is fine because their contents are also
    // part of the RecursiveFilesystemTraversalValue, so v2 and v3 are unequal.
    assertTraversalRootHashesAreEqual(v2, v3);
    progressReceiver.clear();

    // Add a new file *outside* of the directory and see that the value is *not* rebuilt.
    Artifact someFile = sourceArtifact("somewhere/else/a.file");
    createFile(someFile, "new file");
    appendToFile(someFile, "not all changes are treated equal");
    RecursiveFilesystemTraversalValue v4 =
        traverseAndAssertFiles(traversalRoot, expected1, expected2, expected3);
    assertThat(v4).isEqualTo(v3);
    assertTraversalRootHashesAreEqual(v3, v4);
    assertThat(progressReceiver.invalidations).doesNotContain(rftvSkyKey(traversalRoot));
  }

  @Test
  public void testTraversalOfSourceDirectory() throws Exception {
    assertTraversalOfDirectory(sourceArtifact("dir"));
  }

  @Test
  public void testTraversalOfGeneratedDirectory() throws Exception {
    assertTraversalOfDirectory(derivedArtifact("dir"));
  }

  @Test
  public void testTraversalOfTransitiveSymlinkToDirectory() throws Exception {
    Artifact directLinkArtifact = sourceArtifact("direct/dir.sym");
    Artifact transitiveLinkArtifact = sourceArtifact("transitive/sym.sym");
    RootedPath fileA = createFile(rootedPath(sourceArtifact("a/file.a")));
    RootedPath directLink = rootedPath(directLinkArtifact);
    RootedPath transitiveLink = rootedPath(transitiveLinkArtifact);
    PathFragment directLinkPath = PathFragment.create("../a");
    PathFragment transitiveLinkPath = PathFragment.create("../direct/dir.sym");

    parentOf(directLink).asPath().createDirectory();
    parentOf(transitiveLink).asPath().createDirectory();
    directLink.asPath().createSymbolicLink(directLinkPath);
    transitiveLink.asPath().createSymbolicLink(transitiveLinkPath);

    // Expect the file as if was a child of the direct symlink, not of the actual directory.
    traverseAndAssertFiles(
        fileLikeRoot(directLinkArtifact, DONT_CROSS),
        symlinkToDirectoryForTesting(parentOf(fileA), directLink, directLinkPath),
        regularFileForTesting(childOf(directLinkArtifact, "file.a")));

    // Expect the file as if was a child of the transitive symlink, not of the actual directory.
    traverseAndAssertFiles(
        fileLikeRoot(transitiveLinkArtifact, DONT_CROSS),
        symlinkToDirectoryForTesting(parentOf(fileA), transitiveLink, transitiveLinkPath),
        regularFileForTesting(childOf(transitiveLinkArtifact, "file.a")));
  }

  @Test
  public void testTraversePackage() throws Exception {
    Artifact buildFile = sourceArtifact("pkg/BUILD");
    RootedPath buildFilePath = createFile(rootedPath(buildFile));
    RootedPath file1 = createFile(siblingOf(buildFile, "subdir/file.a"));

    traverseAndAssertFiles(
        pkgRoot(parentOf(buildFilePath), DONT_CROSS),
        regularFileForTesting(buildFilePath),
        regularFileForTesting(file1));
  }

  @Test
  public void testTraversalOfSymlinkToDirectory() throws Exception {
    Artifact linkNameArtifact = sourceArtifact("link/foo.sym");
    Artifact linkTargetArtifact = sourceArtifact("dir");
    RootedPath linkName = rootedPath(linkNameArtifact);
    PathFragment linkValue = PathFragment.create("../dir");
    RootedPath file1 = createFile(childOf(linkTargetArtifact, "file.1"));
    createFile(childOf(linkTargetArtifact, "sub/file.2"));
    scratch.dir(parentOf(linkName).asPath().getPathString());
    linkName.asPath().createSymbolicLink(linkValue);

    // Assert that the SkyValue is built and looks right.
    TraversalRequest traversalRoot = fileLikeRoot(linkNameArtifact, DONT_CROSS);
    ResolvedFile expected1 =
        symlinkToDirectoryForTesting(rootedPath(linkTargetArtifact), linkName, linkValue);
    ResolvedFile expected2 = regularFileForTesting(childOf(linkNameArtifact, "file.1"));
    ResolvedFile expected3 = regularFileForTesting(childOf(linkNameArtifact, "sub/file.2"));
    // We expect to see all the files from the symlink'd directory, under the symlink's path, not
    // under the symlink target's path.
    RecursiveFilesystemTraversalValue v1 =
        traverseAndAssertFiles(traversalRoot, expected1, expected2, expected3);
    assertThat(progressReceiver.invalidations).isEmpty();
    assertThat(progressReceiver.evaluations).contains(v1);
    progressReceiver.clear();

    // Add a new file to the directory and see that the value is rebuilt.
    createFile(childOf(linkTargetArtifact, "file.3"));
    invalidateDirectory(linkTargetArtifact);
    ResolvedFile expected4 = regularFileForTesting(childOf(linkNameArtifact, "file.3"));
    RecursiveFilesystemTraversalValue v2 =
        traverseAndAssertFiles(traversalRoot, expected1, expected2, expected3, expected4);
    assertThat(progressReceiver.invalidations).contains(rftvSkyKey(traversalRoot));
    assertThat(progressReceiver.evaluations).contains(v2);
    assertThat(v2).isNotEqualTo(v1);
    assertTraversalRootHashesAreNotEqual(v1, v2);
    progressReceiver.clear();

    // Edit a file in the directory and see that the value is rebuilt.
    appendToFile(file1, "bar");
    RecursiveFilesystemTraversalValue v3 =
        traverseAndAssertFiles(traversalRoot, expected1, expected2, expected3, expected4);
    assertThat(progressReceiver.invalidations).contains(rftvSkyKey(traversalRoot));
    assertThat(progressReceiver.evaluations).contains(v3);
    assertThat(v3).isNotEqualTo(v2);
    assertTraversalRootHashesAreNotEqual(v2, v3);
    progressReceiver.clear();

    // Add a new file *outside* of the directory and see that the value is *not* rebuilt.
    Artifact someFile = sourceArtifact("somewhere/else/a.file");
    createFile(someFile, "new file");
    appendToFile(someFile, "not all changes are treated equal");
    RecursiveFilesystemTraversalValue v4 =
        traverseAndAssertFiles(traversalRoot, expected1, expected2, expected3, expected4);
    assertThat(v4).isEqualTo(v3);
    assertTraversalRootHashesAreEqual(v3, v4);
    assertThat(progressReceiver.invalidations).doesNotContain(rftvSkyKey(traversalRoot));
  }

  @Test
  public void testTraversalOfDanglingSymlink() throws Exception {
    Artifact linkArtifact = sourceArtifact("a/dangling.sym");
    RootedPath link = rootedPath(linkArtifact);
    PathFragment linkTarget = PathFragment.create("non_existent");
    parentOf(link).asPath().createDirectory();
    link.asPath().createSymbolicLink(linkTarget);
    traverseAndAssertFiles(
        fileLikeRoot(linkArtifact, DONT_CROSS), danglingSymlinkForTesting(link, linkTarget));
  }

  @Test
  public void testTraversalOfDanglingSymlinkInADirectory() throws Exception {
    Artifact dirArtifact = sourceArtifact("a");
    RootedPath file = createFile(childOf(dirArtifact, "file.txt"));
    RootedPath link = rootedPath(sourceArtifact("a/dangling.sym"));
    PathFragment linkTarget = PathFragment.create("non_existent");
    parentOf(link).asPath().createDirectory();
    link.asPath().createSymbolicLink(linkTarget);
    traverseAndAssertFiles(
        fileLikeRoot(dirArtifact, DONT_CROSS),
        regularFileForTesting(file),
        danglingSymlinkForTesting(link, linkTarget));
  }

  private void assertTraverseSubpackages(PackageBoundaryMode traverseSubpackages) throws Exception {
    Artifact pkgDirArtifact = sourceArtifact("pkg1/foo");
    Artifact subpkgDirArtifact = sourceArtifact("pkg1/foo/subdir/subpkg");
    RootedPath pkgBuildFile = childOf(pkgDirArtifact, "BUILD");
    RootedPath subpkgBuildFile = childOf(subpkgDirArtifact, "BUILD");
    scratch.dir(rootedPath(pkgDirArtifact).asPath().getPathString());
    scratch.dir(rootedPath(subpkgDirArtifact).asPath().getPathString());
    createFile(pkgBuildFile);
    createFile(subpkgBuildFile);

    TraversalRequest traversalRoot = pkgRoot(parentOf(pkgBuildFile), traverseSubpackages);

    ResolvedFile expected1 = regularFileForTesting(pkgBuildFile);
    ResolvedFile expected2 = regularFileForTesting(subpkgBuildFile);
    switch (traverseSubpackages) {
      case CROSS:
        traverseAndAssertFiles(traversalRoot, expected1, expected2);
        break;
      case DONT_CROSS:
        traverseAndAssertFiles(traversalRoot, expected1);
        break;
      case REPORT_ERROR:
        SkyKey key = rftvSkyKey(traversalRoot);
        EvaluationResult<SkyValue> result = eval(key);
        assertThat(result.hasError()).isTrue();
        assertThat(result.getError().getException())
            .hasMessageThat()
            .contains("crosses package boundary into package rooted at");
        break;
      default:
        throw new IllegalStateException(traverseSubpackages.toString());
    }
  }

  @Test
  public void testTraverseSubpackages() throws Exception {
    assertTraverseSubpackages(CROSS);
  }

  @Test
  public void testDoNotTraverseSubpackages() throws Exception {
    assertTraverseSubpackages(DONT_CROSS);
  }

  @Test
  public void testReportErrorWhenTraversingSubpackages() throws Exception {
    assertTraverseSubpackages(REPORT_ERROR);
  }

  @Test
  public void testSwitchPackageRootsWhenUsingMultiplePackagePaths() throws Exception {
    // Layout:
    //   pp1://a/BUILD
    //   pp1://a/file.a
    //   pp1://a/b.sym -> b/   (only created later)
    //   pp1://a/b/
    //   pp1://a/b/file.fake
    //   pp1://a/subdir/file.b
    //
    //   pp2://a/BUILD
    //   pp2://a/b/
    //   pp2://a/b/BUILD
    //   pp2://a/b/file.a
    //   pp2://a/subdir.fake/
    //   pp2://a/subdir.fake/file.fake
    //
    // Notice that pp1://a/b will be overlaid by pp2://a/b as the latter has a BUILD file and that
    // takes precedence. On the other hand the package definition pp2://a/BUILD will be ignored
    // since package //a is already defined under pp1.
    //
    // Notice also that pp1://a/b.sym is a relative symlink pointing to b/. This should be resolved
    // to the definition of //a/b/ under pp1, not under pp2.

    // Set the package paths.
    pkgLocator.set(
        new PathPackageLocator(
            outputBase,
            ImmutableList.of(rootDirectory.getRelative("pp1"), rootDirectory.getRelative("pp2")),
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator.get());

    Artifact aBuildArtifact = sourceArtifactUnderPackagePath("a/BUILD", "pp1");
    Artifact bBuildArtifact = sourceArtifactUnderPackagePath("a/b/BUILD", "pp2");

    RootedPath pp1aBuild = createFile(rootedPath(aBuildArtifact));
    RootedPath pp1aFileA = createFile(siblingOf(pp1aBuild, "file.a"));
    RootedPath pp1bFileFake = createFile(siblingOf(pp1aBuild, "b/file.fake"));
    RootedPath pp1aSubdirFileB = createFile(siblingOf(pp1aBuild, "subdir/file.b"));

    RootedPath pp2aBuild = createFile(rootedPath("a/BUILD", "pp2"));
    RootedPath pp2bBuild = createFile(rootedPath(bBuildArtifact));
    RootedPath pp2bFileA = createFile(siblingOf(pp2bBuild, "file.a"));
    createFile(siblingOf(pp2aBuild, "subdir.fake/file.fake"));

    // Traverse //a including subpackages. The result should contain the pp1-definition of //a and
    // the pp2-definition of //a/b.
    traverseAndAssertFiles(
        pkgRoot(parentOf(rootedPath(aBuildArtifact)), CROSS),
        regularFileForTesting(pp1aBuild),
        regularFileForTesting(pp1aFileA),
        regularFileForTesting(pp1aSubdirFileB),
        regularFileForTesting(pp2bBuild),
        regularFileForTesting(pp2bFileA));

    // Traverse //a excluding subpackages. The result should only contain files from //a and not
    // from //a/b.
    traverseAndAssertFiles(
        pkgRoot(parentOf(rootedPath(aBuildArtifact)), DONT_CROSS),
        regularFileForTesting(pp1aBuild),
        regularFileForTesting(pp1aFileA),
        regularFileForTesting(pp1aSubdirFileB));

    // Create a relative symlink pp1://a/b.sym -> b/. It will be resolved to the subdirectory
    // pp1://a/b, even though a package definition pp2://a/b exists.
    RootedPath pp1aBsym = siblingOf(pp1aFileA, "b.sym");
    pp1aBsym.asPath().createSymbolicLink(PathFragment.create("b"));
    invalidateDirectory(parentOf(pp1aBsym));

    // Traverse //a excluding subpackages. The relative symlink //a/b.sym points to the subdirectory
    // a/b, i.e. the pp1-definition, even though there is a pp2-defined package //a/b and we expect
    // to see b.sym/b.fake (not b/b.fake).
    traverseAndAssertFiles(
        pkgRoot(parentOf(rootedPath(aBuildArtifact)), DONT_CROSS),
        regularFileForTesting(pp1aBuild),
        regularFileForTesting(pp1aFileA),
        regularFileForTesting(childOf(pp1aBsym, "file.fake")),
        symlinkToDirectoryForTesting(parentOf(pp1bFileFake), pp1aBsym, PathFragment.create("b")),
        regularFileForTesting(pp1aSubdirFileB));
  }

  @Test
  public void testFileDigestChangeCausesRebuild() throws Exception {
    Artifact artifact = sourceArtifact("foo/bar.txt");
    RootedPath path = rootedPath(artifact);
    createFile(path, "hello");

    // Assert that the SkyValue is built and looks right.
    TraversalRequest params = fileLikeRoot(artifact, DONT_CROSS);
    ResolvedFile expected = regularFileForTesting(path);
    RecursiveFilesystemTraversalValue v1 = traverseAndAssertFiles(params, expected);
    assertThat(progressReceiver.evaluations).contains(v1);
    progressReceiver.clear();

    // Change the digest of the file. See that the value is rebuilt.
    appendToFile(path, "world");
    RecursiveFilesystemTraversalValue v2 = traverseAndAssertFiles(params, expected);
    assertThat(progressReceiver.invalidations).contains(rftvSkyKey(params));
    assertThat(v2).isNotEqualTo(v1);
    assertTraversalRootHashesAreNotEqual(v1, v2);
  }

  @Test
  public void testFileMtimeChangeDoesNotCauseRebuildIfDigestIsUnchanged() throws Exception {
    Artifact artifact = sourceArtifact("foo/bar.txt");
    RootedPath path = rootedPath(artifact);
    createFile(path, "hello");

    // Assert that the SkyValue is built and looks right.
    TraversalRequest params = fileLikeRoot(artifact, DONT_CROSS);
    ResolvedFile expected = regularFileForTesting(path);
    RecursiveFilesystemTraversalValue v1 = traverseAndAssertFiles(params, expected);
    assertThat(progressReceiver.evaluations).contains(v1);
    progressReceiver.clear();

    // Change the mtime of the file but not the digest. See that the value is *not* rebuilt.
    long mtime = path.asPath().getLastModifiedTime();
    mtime += 1000000L; // more than the timestamp granularity of any filesystem
    path.asPath().setLastModifiedTime(mtime);
    RecursiveFilesystemTraversalValue v2 = traverseAndAssertFiles(params, expected);
    assertThat(v2).isEqualTo(v1);
    assertTraversalRootHashesAreEqual(v1, v2);
  }

  @Test
  public void testRegexp() throws Exception {
    Artifact wantedArtifact = sourceArtifact("foo/bar/baz.txt");
    Artifact unwantedArtifact = sourceArtifact("foo/boo/baztxt.bak");
    RootedPath wantedPath = rootedPath(wantedArtifact);
    createFile(wantedPath, "hello");
    createFile(unwantedArtifact, "nope");
    Artifact pkgDirArtifact = sourceArtifact("foo");
    RootedPath dir = rootedPath(pkgDirArtifact);
    scratch.dir(dir.asPath().getPathString());

    TraversalRequest traversalRoot =
        new TraversalRequest(
            dir, false, PackageBoundaryMode.REPORT_ERROR, true, null, Pattern.compile(".*\\.txt"));

    ResolvedFile expected = regularFileForTesting(wantedPath);
    traverseAndAssertFiles(traversalRoot, expected);
  }

  @Test
  public void testGeneratedDirectoryConflictsWithPackage() throws Exception {
    Artifact genDir = derivedArtifact("a/b");
    createFile(rootedPath(sourceArtifact("a/b/c/file.real")));
    createFile(rootedPath(derivedArtifact("a/b/c/file.fake")));
    createFile(sourceArtifact("a/b/c/BUILD"));

    SkyKey key = rftvSkyKey(fileLikeRoot(genDir, CROSS));
    EvaluationResult<SkyValue> result = eval(key);
    assertThat(result.hasError()).isTrue();
    ErrorInfo error = result.getError(key);
    assertThat(error.isTransitivelyTransient()).isFalse();
    assertThat(error.getException())
        .hasMessageThat()
        .contains("Generated directory a/b/c conflicts with package under the same path.");
  }

  @Test
  public void unboundedSymlinkExpansionError() throws Exception {
    Artifact bazLink = sourceArtifact("foo/baz.sym");
    Path parentDir = scratch.dir("foo");
    bazLink.getPath().createSymbolicLink(parentDir);
    SkyKey key = rftvSkyKey(pkgRoot(parentOf(rootedPath(bazLink)), DONT_CROSS));
    EvaluationResult<SkyValue> result = eval(key);
    assertThat(result.hasError()).isTrue();
    ErrorInfo error = result.getError(key);
    assertThat(error.getException()).isInstanceOf(FileOperationException.class);
    assertThat(error.getException()).hasMessageThat().contains("Infinite symlink expansion");
  }

  @Test
  public void symlinkChainError() throws Exception {
    scratch.dir("a");
    Artifact fooLink = sourceArtifact("a/foo.sym");
    Artifact barLink = sourceArtifact("a/bar.sym");
    Artifact bazLink = sourceArtifact("a/baz.sym");
    fooLink.getPath().createSymbolicLink(barLink.getPath());
    barLink.getPath().createSymbolicLink(bazLink.getPath());
    bazLink.getPath().createSymbolicLink(fooLink.getPath());

    SkyKey key = rftvSkyKey(pkgRoot(parentOf(rootedPath(bazLink)), DONT_CROSS));
    EvaluationResult<SkyValue> result = eval(key);
    assertThat(result.hasError()).isTrue();
    ErrorInfo error = result.getError(key);
    assertThat(error.getException()).isInstanceOf(FileOperationException.class);
    assertThat(error.getException()).hasMessageThat().contains("Symlink cycle");
  }
}
