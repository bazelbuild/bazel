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
import static com.google.devtools.build.lib.actions.ActionInputHelper.treeFileArtifact;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Runnables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionLookupValue.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.actions.util.TestAction;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.BasicFilesystemDirtinessChecker;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileStatusWithDigest;
import com.google.devtools.build.lib.vfs.FileStatusWithDigestAdapter;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.Differencer.Diff;
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
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link FilesystemValueChecker}.
 */
@RunWith(JUnit4.class)
public class FilesystemValueCheckerTest {

  private RecordingDifferencer differencer;
  private MemoizingEvaluator evaluator;
  private SequentialBuildDriver driver;
  private MockFileSystem fs;
  private Path pkgRoot;

  @Before
  public final void setUp() throws Exception  {
    ImmutableMap.Builder<SkyFunctionName, SkyFunction> skyFunctions = ImmutableMap.builder();

    fs = new MockFileSystem();
    pkgRoot = fs.getPath("/testroot");
    FileSystemUtils.createDirectoryAndParents(pkgRoot);
    FileSystemUtils.createEmptyFile(pkgRoot.getRelative("WORKSPACE"));

    AtomicReference<PathPackageLocator> pkgLocator =
        new AtomicReference<>(
            new PathPackageLocator(
                fs.getPath("/output_base"),
                ImmutableList.of(pkgRoot),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(pkgRoot, pkgRoot), pkgRoot, TestConstants.PRODUCT_NAME);
    ExternalFilesHelper externalFilesHelper = new ExternalFilesHelper(
        pkgLocator, ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS, directories);
    skyFunctions.put(SkyFunctions.FILE_STATE, new FileStateFunction(
        new AtomicReference<TimestampGranularityMonitor>(), externalFilesHelper));
    skyFunctions.put(SkyFunctions.FILE, new FileFunction(pkgLocator));
    skyFunctions.put(
        SkyFunctions.FILE_SYMLINK_CYCLE_UNIQUENESS, new FileSymlinkCycleUniquenessFunction());
    skyFunctions.put(
        SkyFunctions.FILE_SYMLINK_INFINITE_EXPANSION_UNIQUENESS,
        new FileSymlinkInfiniteExpansionUniquenessFunction());
    skyFunctions.put(SkyFunctions.PACKAGE,
        new PackageFunction(null, null, null, null, null, null, null));
    skyFunctions.put(
        SkyFunctions.PACKAGE_LOOKUP,
        new PackageLookupFunction(
            new AtomicReference<>(ImmutableSet.<PackageIdentifier>of()),
            CrossRepositoryLabelViolationStrategy.ERROR,
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    skyFunctions.put(SkyFunctions.WORKSPACE_AST,
        new WorkspaceASTFunction(TestRuleClassProvider.getRuleClassProvider()));
    skyFunctions.put(SkyFunctions.WORKSPACE_FILE,
        new WorkspaceFileFunction(TestRuleClassProvider.getRuleClassProvider(),
            TestConstants.PACKAGE_FACTORY_BUILDER_FACTORY_FOR_TESTING.builder().build(
                TestRuleClassProvider.getRuleClassProvider(), fs),
            directories));
    skyFunctions.put(SkyFunctions.EXTERNAL_PACKAGE, new ExternalPackageFunction());

    differencer = new SequencedRecordingDifferencer();
    evaluator = new InMemoryMemoizingEvaluator(skyFunctions.build(), differencer);
    driver = new SequentialBuildDriver(evaluator);
    PrecomputedValue.BUILD_ID.set(differencer, UUID.randomUUID());
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator.get());
  }

  @Test
  public void testEmpty() throws Exception {
    FilesystemValueChecker checker = new FilesystemValueChecker(null, null);
    assertEmptyDiff(getDirtyFilesystemKeys(evaluator, checker));
  }

  @Test
  public void testSimple() throws Exception {
    FilesystemValueChecker checker = new FilesystemValueChecker(null, null);

    Path path = fs.getPath("/foo");
    FileSystemUtils.createEmptyFile(path);
    assertEmptyDiff(getDirtyFilesystemKeys(evaluator, checker));

    SkyKey skyKey = FileStateValue.key(
        RootedPath.toRootedPath(fs.getRootDirectory(), PathFragment.create("foo")));
    EvaluationResult<SkyValue> result =
        driver.evaluate(
            ImmutableList.of(skyKey),
            false,
            SkyframeExecutor.DEFAULT_THREAD_COUNT,
            NullEventHandler.INSTANCE);
    assertThat(result.hasError()).isFalse();

    assertEmptyDiff(getDirtyFilesystemKeys(evaluator, checker));

    FileSystemUtils.writeContentAsLatin1(path, "hello");
    assertDiffWithNewValues(getDirtyFilesystemKeys(evaluator, checker), skyKey);

    // The dirty bits are not reset until the FileValues are actually revalidated.
    assertDiffWithNewValues(getDirtyFilesystemKeys(evaluator, checker), skyKey);

    differencer.invalidate(ImmutableList.of(skyKey));
    result =
        driver.evaluate(
            ImmutableList.of(skyKey),
            false,
            SkyframeExecutor.DEFAULT_THREAD_COUNT,
            NullEventHandler.INSTANCE);
    assertThat(result.hasError()).isFalse();
    assertEmptyDiff(getDirtyFilesystemKeys(evaluator, checker));
  }

  /**
   * Tests that an already-invalidated value can still be marked changed: symlink points at sym1.
   * Invalidate symlink by changing sym1 from pointing at path to point to sym2. This only dirties
   * (rather than changes) symlink because sym2 still points at path, so all symlink stats remain
   * the same. Then do a null build, change sym1 back to point at path, and change symlink to not be
   * a symlink anymore. The fact that it is not a symlink should be detected.
   */
  @Test
  public void testDirtySymlink() throws Exception {
    FilesystemValueChecker checker = new FilesystemValueChecker(null, null);

    Path path = fs.getPath("/foo");
    FileSystemUtils.writeContentAsLatin1(path, "foo contents");
    // We need the intermediate sym1 and sym2 so that we can dirty a child of symlink without
    // actually changing the FileValue calculated for symlink (if we changed the contents of foo,
    // the the FileValue created for symlink would notice, since it stats foo).
    Path sym1 = fs.getPath("/sym1");
    Path sym2 = fs.getPath("/sym2");
    Path symlink = fs.getPath("/bar");
    FileSystemUtils.ensureSymbolicLink(symlink, sym1);
    FileSystemUtils.ensureSymbolicLink(sym1, path);
    FileSystemUtils.ensureSymbolicLink(sym2, path);
    SkyKey fooKey =
        FileValue.key(RootedPath.toRootedPath(fs.getRootDirectory(), PathFragment.create("foo")));
    RootedPath symlinkRootedPath =
        RootedPath.toRootedPath(fs.getRootDirectory(), PathFragment.create("bar"));
    SkyKey symlinkKey = FileValue.key(symlinkRootedPath);
    SkyKey symlinkFileStateKey = FileStateValue.key(symlinkRootedPath);
    RootedPath sym1RootedPath =
        RootedPath.toRootedPath(fs.getRootDirectory(), PathFragment.create("sym1"));
    SkyKey sym1FileStateKey = FileStateValue.key(sym1RootedPath);
    Iterable<SkyKey> allKeys = ImmutableList.of(symlinkKey, fooKey);

    // First build -- prime the graph.
    EvaluationResult<FileValue> result =
        driver.evaluate(
            allKeys, false, SkyframeExecutor.DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);
    assertThat(result.hasError()).isFalse();
    FileValue symlinkValue = result.get(symlinkKey);
    FileValue fooValue = result.get(fooKey);
    assertWithMessage(symlinkValue.toString()).that(symlinkValue.isSymlink()).isTrue();
    // Digest is not always available, so use size as a proxy for contents.
    assertThat(symlinkValue.getSize()).isEqualTo(fooValue.getSize());
    assertEmptyDiff(getDirtyFilesystemKeys(evaluator, checker));

    // Before second build, move sym1 to point to sym2.
    assertThat(sym1.delete()).isTrue();
    FileSystemUtils.ensureSymbolicLink(sym1, sym2);
    assertDiffWithNewValues(getDirtyFilesystemKeys(evaluator, checker), sym1FileStateKey);

    differencer.invalidate(ImmutableList.of(sym1FileStateKey));
    result =
        driver.evaluate(
            ImmutableList.<SkyKey>of(),
            false,
            SkyframeExecutor.DEFAULT_THREAD_COUNT,
            NullEventHandler.INSTANCE);
    assertThat(result.hasError()).isFalse();
    assertDiffWithNewValues(getDirtyFilesystemKeys(evaluator, checker), sym1FileStateKey);

    // Before third build, move sym1 back to original (so change pruning will prevent signaling of
    // its parents, but change symlink for real.
    assertThat(sym1.delete()).isTrue();
    FileSystemUtils.ensureSymbolicLink(sym1, path);
    assertThat(symlink.delete()).isTrue();
    FileSystemUtils.writeContentAsLatin1(symlink, "new symlink contents");
    assertDiffWithNewValues(getDirtyFilesystemKeys(evaluator, checker), symlinkFileStateKey);
    differencer.invalidate(ImmutableList.of(symlinkFileStateKey));
    result =
        driver.evaluate(
            allKeys, false, SkyframeExecutor.DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);
    assertThat(result.hasError()).isFalse();
    symlinkValue = result.get(symlinkKey);
    assertWithMessage(symlinkValue.toString()).that(symlinkValue.isSymlink()).isFalse();
    assertThat(result.get(fooKey)).isEqualTo(fooValue);
    assertThat(symlinkValue.getSize()).isNotEqualTo(fooValue.getSize());
    assertEmptyDiff(getDirtyFilesystemKeys(evaluator, checker));
  }

  @Test
  public void testExplicitFiles() throws Exception {
    FilesystemValueChecker checker = new FilesystemValueChecker(null, null);

    Path path1 = fs.getPath("/foo1");
    Path path2 = fs.getPath("/foo2");
    FileSystemUtils.createEmptyFile(path1);
    FileSystemUtils.createEmptyFile(path2);
    assertEmptyDiff(getDirtyFilesystemKeys(evaluator, checker));

    SkyKey key1 =
        FileStateValue.key(
            RootedPath.toRootedPath(fs.getRootDirectory(), PathFragment.create("foo1")));
    SkyKey key2 =
        FileStateValue.key(
            RootedPath.toRootedPath(fs.getRootDirectory(), PathFragment.create("foo2")));
    Iterable<SkyKey> skyKeys = ImmutableList.of(key1, key2);
    EvaluationResult<SkyValue> result =
        driver.evaluate(
            skyKeys, false, SkyframeExecutor.DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);
    assertThat(result.hasError()).isFalse();

    assertEmptyDiff(getDirtyFilesystemKeys(evaluator, checker));

    FileSystemUtils.writeContentAsLatin1(path1, "hello1");
    FileSystemUtils.writeContentAsLatin1(path1, "hello2");
    path1.setLastModifiedTime(27);
    path2.setLastModifiedTime(42);
    assertDiffWithNewValues(getDirtyFilesystemKeys(evaluator, checker), key1, key2);

    differencer.invalidate(skyKeys);
    result =
        driver.evaluate(
            skyKeys, false, SkyframeExecutor.DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);
    assertThat(result.hasError()).isFalse();
    assertEmptyDiff(getDirtyFilesystemKeys(evaluator, checker));
  }

  @Test
  public void testFileWithIOExceptionNotConsideredDirty() throws Exception {
    Path path = fs.getPath("/testroot/foo");
    path.getParentDirectory().createDirectory();
    path.createSymbolicLink(PathFragment.create("bar"));

    fs.readlinkThrowsIoException = true;
    SkyKey fileKey = FileStateValue.key(
        RootedPath.toRootedPath(pkgRoot, PathFragment.create("foo")));
    EvaluationResult<SkyValue> result =
        driver.evaluate(
            ImmutableList.of(fileKey),
            false,
            SkyframeExecutor.DEFAULT_THREAD_COUNT,
            NullEventHandler.INSTANCE);
    assertThat(result.hasError()).isTrue();

    fs.readlinkThrowsIoException = false;
    FilesystemValueChecker checker = new FilesystemValueChecker(null, null);
    Diff diff = getDirtyFilesystemKeys(evaluator, checker);
    assertThat(diff.changedKeysWithoutNewValues()).isEmpty();
    assertThat(diff.changedKeysWithNewValues()).isEmpty();
  }

  @Test
  public void testFilesInCycleNotConsideredDirty() throws Exception {
    Path path1 = pkgRoot.getRelative("foo1");
    Path path2 = pkgRoot.getRelative("foo2");
    Path path3 = pkgRoot.getRelative("foo3");
    FileSystemUtils.ensureSymbolicLink(path1, path2);
    FileSystemUtils.ensureSymbolicLink(path2, path3);
    FileSystemUtils.ensureSymbolicLink(path3, path1);
    SkyKey fileKey1 = FileValue.key(RootedPath.toRootedPath(pkgRoot, path1));

    EvaluationResult<SkyValue> result =
        driver.evaluate(
            ImmutableList.of(fileKey1),
            false,
            SkyframeExecutor.DEFAULT_THREAD_COUNT,
            NullEventHandler.INSTANCE);
    assertThat(result.hasError()).isTrue();

    FilesystemValueChecker checker = new FilesystemValueChecker(null, null);
    Diff diff = getDirtyFilesystemKeys(evaluator, checker);
    assertThat(diff.changedKeysWithoutNewValues()).isEmpty();
    assertThat(diff.changedKeysWithNewValues()).isEmpty();
  }

  public void checkDirtyActions(BatchStat batchStatter, boolean forceDigests) throws Exception {
    Artifact out1 = createDerivedArtifact("fiz");
    Artifact out2 = createDerivedArtifact("pop");

    FileSystemUtils.writeContentAsLatin1(out1.getPath(), "hello");
    FileSystemUtils.writeContentAsLatin1(out2.getPath(), "fizzlepop");

    SkyKey actionLookupKey =
        ActionLookupValue.key(
            new ActionLookupKey() {
              @Override
              protected SkyFunctionName getType() {
                return SkyFunctionName.FOR_TESTING;
              }
            });
    SkyKey actionKey1 = ActionExecutionValue.key(actionLookupKey, 0);
    SkyKey actionKey2 = ActionExecutionValue.key(actionLookupKey, 1);
    differencer.inject(
        ImmutableMap.<SkyKey, SkyValue>of(
            actionKey1,
                actionValue(
                    new TestAction(
                        Runnables.doNothing(), ImmutableSet.<Artifact>of(), ImmutableSet.of(out1)),
                    forceDigests),
            actionKey2,
                actionValue(
                    new TestAction(
                        Runnables.doNothing(), ImmutableSet.<Artifact>of(), ImmutableSet.of(out2)),
                    forceDigests)));
    assertThat(
            driver
                .evaluate(ImmutableList.<SkyKey>of(), false, 1, NullEventHandler.INSTANCE)
                .hasError())
        .isFalse();
    assertThat(new FilesystemValueChecker(null, null).getDirtyActionValues(evaluator.getValues(),
        batchStatter, ModifiedFileSet.EVERYTHING_MODIFIED)).isEmpty();

    FileSystemUtils.writeContentAsLatin1(out1.getPath(), "goodbye");
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(), batchStatter, ModifiedFileSet.EVERYTHING_MODIFIED))
        .containsExactly(actionKey1);
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    new ModifiedFileSet.Builder().modify(out1.getExecPath()).build()))
        .containsExactly(actionKey1);
    assertThat(
            new FilesystemValueChecker(null, null).getDirtyActionValues(evaluator.getValues(),
                batchStatter,
                new ModifiedFileSet.Builder().modify(
                    out1.getExecPath().getParentDirectory()).build())).isEmpty();
    assertThat(
        new FilesystemValueChecker(null, null).getDirtyActionValues(evaluator.getValues(),
            batchStatter, ModifiedFileSet.NOTHING_MODIFIED)).isEmpty();
  }

  public void checkDirtyTreeArtifactActions(BatchStat batchStatter)
      throws Exception {
    // Normally, an Action specifies the contents of a TreeArtifact when it executes.
    // To decouple FileSystemValueTester checking from Action execution, we inject TreeArtifact
    // contents into ActionExecutionValues.

    Artifact out1 = createTreeArtifact("one");
    TreeFileArtifact file11 = treeFileArtifact(out1, "fizz");
    FileSystemUtils.createDirectoryAndParents(out1.getPath());
    FileSystemUtils.writeContentAsLatin1(file11.getPath(), "buzz");

    Artifact out2 = createTreeArtifact("two");
    FileSystemUtils.createDirectoryAndParents(out2.getPath().getChild("subdir"));
    TreeFileArtifact file21 = treeFileArtifact(out2, "moony");
    TreeFileArtifact file22 = treeFileArtifact(out2, "subdir/wormtail");
    FileSystemUtils.writeContentAsLatin1(file21.getPath(), "padfoot");
    FileSystemUtils.writeContentAsLatin1(file22.getPath(), "prongs");

    Artifact outEmpty = createTreeArtifact("empty");
    FileSystemUtils.createDirectoryAndParents(outEmpty.getPath());

    Artifact outUnchanging = createTreeArtifact("untouched");
    FileSystemUtils.createDirectoryAndParents(outUnchanging.getPath());

    Artifact last = createTreeArtifact("zzzzzzzzzz");
    FileSystemUtils.createDirectoryAndParents(last.getPath());

    SkyKey actionLookupKey =
        ActionLookupValue.key(
            new ActionLookupKey() {
              @Override
              protected SkyFunctionName getType() {
                return SkyFunctionName.FOR_TESTING;
              }
            });
    SkyKey actionKey1 = ActionExecutionValue.key(actionLookupKey, 0);
    SkyKey actionKey2 = ActionExecutionValue.key(actionLookupKey, 1);
    SkyKey actionKeyEmpty = ActionExecutionValue.key(actionLookupKey, 2);
    SkyKey actionKeyUnchanging = ActionExecutionValue.key(actionLookupKey, 3);
    SkyKey actionKeyLast = ActionExecutionValue.key(actionLookupKey, 4);
    differencer.inject(
        ImmutableMap.<SkyKey, SkyValue>of(
            actionKey1,
            actionValueWithTreeArtifacts(ImmutableList.of(file11)),
            actionKey2,
            actionValueWithTreeArtifacts(ImmutableList.of(file21, file22)),
            actionKeyEmpty,
            actionValueWithEmptyDirectory(outEmpty),
            actionKeyUnchanging,
            actionValueWithEmptyDirectory(outUnchanging),
            actionKeyLast,
            actionValueWithEmptyDirectory(last)));

    assertThat(
            driver
                .evaluate(ImmutableList.<SkyKey>of(), false, 1, NullEventHandler.INSTANCE)
                .hasError())
        .isFalse();
    assertThat(new FilesystemValueChecker(null, null).getDirtyActionValues(evaluator.getValues(),
        batchStatter, ModifiedFileSet.EVERYTHING_MODIFIED)).isEmpty();

    // Touching the TreeArtifact directory should have no effect
    FileSystemUtils.touchFile(out1.getPath());
    assertThat(
        new FilesystemValueChecker(null, null).getDirtyActionValues(evaluator.getValues(),
            batchStatter, ModifiedFileSet.EVERYTHING_MODIFIED)).isEmpty();
    // Neither should touching a subdirectory.
    FileSystemUtils.touchFile(out2.getPath().getChild("subdir"));
    assertThat(
        new FilesystemValueChecker(null, null).getDirtyActionValues(evaluator.getValues(),
            batchStatter, ModifiedFileSet.EVERYTHING_MODIFIED)).isEmpty();

    /* **** Tests for directories **** */

    // Removing a directory (even if empty) should have an effect
    outEmpty.getPath().delete();
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    new ModifiedFileSet.Builder().modify(outEmpty.getExecPath()).build()))
        .containsExactly(actionKeyEmpty);
    // Symbolic links should count as dirty
    Path dummyEmptyDir = fs.getPath("/bin").getRelative("symlink");
    FileSystemUtils.createDirectoryAndParents(dummyEmptyDir);
    FileSystemUtils.ensureSymbolicLink(outEmpty.getPath(), dummyEmptyDir);
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    new ModifiedFileSet.Builder().modify(outEmpty.getExecPath()).build()))
        .containsExactly(actionKeyEmpty);

    // We're done fiddling with this... restore the original state
    outEmpty.getPath().delete();
    FileSystemUtils.deleteTree(dummyEmptyDir);
    FileSystemUtils.createDirectoryAndParents(outEmpty.getPath());

    /* **** Tests for files and directory contents ****/

    // Test that file contents matter. This is covered by existing tests already,
    // so it's just a sanity check.
    FileSystemUtils.writeContentAsLatin1(file11.getPath(), "goodbye");
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    new ModifiedFileSet.Builder().modify(file11.getExecPath()).build()))
        .containsExactly(actionKey1);

    // Test that directory contents (and nested contents) matter
    Artifact out1new = treeFileArtifact(out1, "julius/caesar");
    FileSystemUtils.createDirectoryAndParents(out1.getPath().getChild("julius"));
    FileSystemUtils.writeContentAsLatin1(out1new.getPath(), "octavian");
    // even for empty directories
    Artifact outEmptyNew = treeFileArtifact(outEmpty, "marcus");
    FileSystemUtils.writeContentAsLatin1(outEmptyNew.getPath(), "aurelius");
    // so does removing
    file21.getPath().delete();
    // now, let's test our changes are actually visible
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(), batchStatter, ModifiedFileSet.EVERYTHING_MODIFIED))
        .containsExactly(actionKey1, actionKey2, actionKeyEmpty);
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    new ModifiedFileSet.Builder()
                        .modify(file21.getExecPath())
                        .modify(out1new.getExecPath())
                        .modify(outEmptyNew.getExecPath())
                        .build()))
        .containsExactly(actionKey1, actionKey2, actionKeyEmpty);
    // We also check that if the modified file set does not contain our modified files on disk,
    // we are not going to check and return them.
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    new ModifiedFileSet.Builder()
                        .modify(file21.getExecPath())
                        .modify(outEmptyNew.getExecPath())
                        .build()))
        .containsExactly(actionKey2, actionKeyEmpty);
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    new ModifiedFileSet.Builder()
                        .modify(file21.getExecPath())
                        .modify(out1new.getExecPath())
                        .build()))
        .containsExactly(actionKey1, actionKey2);
    // Check modifying the last (lexicographically) tree artifact.
    last.getPath().delete();
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    new ModifiedFileSet.Builder()
                        .modify(file21.getExecPath())
                        .modify(out1new.getExecPath())
                        .modify(last.getExecPath())
                        .build()))
        .containsExactly(actionKey1, actionKey2, actionKeyLast);
    // Check ModifiedFileSet without the last (lexicographically) tree artifact.
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    new ModifiedFileSet.Builder()
                        .modify(file21.getExecPath())
                        .modify(out1new.getExecPath())
                        .build()))
        .containsExactly(actionKey1, actionKey2);
    // Restore
    last.getPath().delete();
    FileSystemUtils.createDirectoryAndParents(last.getPath());
    // We add a test for NOTHING_MODIFIED, because FileSystemValueChecker doesn't
    // pay attention to file sets for TreeArtifact directory listings.
    assertThat(
        new FilesystemValueChecker(null, null).getDirtyActionValues(evaluator.getValues(),
            batchStatter, ModifiedFileSet.NOTHING_MODIFIED)).isEmpty();
  }

  private Artifact createDerivedArtifact(String relPath) throws IOException {
    Path outputPath = fs.getPath("/bin");
    outputPath.createDirectory();
    return new Artifact(
        outputPath.getRelative(relPath), Root.asDerivedRoot(fs.getPath("/"), outputPath));
  }

  private Artifact createTreeArtifact(String relPath) throws IOException {
    Path outputDir = fs.getPath("/bin");
    Path outputPath = outputDir.getRelative(relPath);
    outputDir.createDirectory();
    Root derivedRoot = Root.asDerivedRoot(fs.getPath("/"), outputDir);
    return new SpecialArtifact(outputPath, derivedRoot,
        derivedRoot.getExecPath().getRelative(outputPath.relativeTo(derivedRoot.getPath())),
        ArtifactOwner.NULL_OWNER, SpecialArtifactType.TREE);
  }

  @Test
  public void testDirtyActions() throws Exception {
    checkDirtyActions(null, false);
  }

  @Test
  public void testDirtyActionsBatchStat() throws Exception {
    checkDirtyActions(
        new BatchStat() {
          @Override
          public List<FileStatusWithDigest> batchStat(
              boolean useDigest, boolean includeLinks, Iterable<PathFragment> paths)
              throws IOException {
            List<FileStatusWithDigest> stats = new ArrayList<>();
            for (PathFragment pathFrag : paths) {
              stats.add(
                  FileStatusWithDigestAdapter.adapt(
                      fs.getRootDirectory().getRelative(pathFrag).statIfFound(Symlinks.NOFOLLOW)));
            }
            return stats;
          }
        },
        false);
  }

  @Test
  public void testDirtyActionsBatchStatWithDigest() throws Exception {
    checkDirtyActions(
        new BatchStat() {
          @Override
          public List<FileStatusWithDigest> batchStat(
              boolean useDigest, boolean includeLinks, Iterable<PathFragment> paths)
              throws IOException {
            List<FileStatusWithDigest> stats = new ArrayList<>();
            for (PathFragment pathFrag : paths) {
              final Path path = fs.getRootDirectory().getRelative(pathFrag);
              stats.add(statWithDigest(path, path.statIfFound(Symlinks.NOFOLLOW)));
            }
            return stats;
          }
        },
        true);
  }

  @Test
  public void testDirtyActionsBatchStatFallback() throws Exception {
    checkDirtyActions(
        new BatchStat() {
          @Override
          public List<FileStatusWithDigest> batchStat(
              boolean useDigest, boolean includeLinks, Iterable<PathFragment> paths)
              throws IOException {
            throw new IOException("try again");
          }
        },
        false);
  }

  @Test
  public void testDirtyTreeArtifactActions() throws Exception {
    checkDirtyTreeArtifactActions(null);
  }

  @Test
  public void testDirtyTreeArtifactActionsBatchStat() throws Exception {
    checkDirtyTreeArtifactActions(
        new BatchStat() {
          @Override
          public List<FileStatusWithDigest> batchStat(
              boolean useDigest, boolean includeLinks, Iterable<PathFragment> paths)
              throws IOException {
            List<FileStatusWithDigest> stats = new ArrayList<>();
            for (PathFragment pathFrag : paths) {
              stats.add(
                  FileStatusWithDigestAdapter.adapt(
                      fs.getRootDirectory().getRelative(pathFrag).statIfFound(Symlinks.NOFOLLOW)));
            }
            return stats;
          }
        });
  }

  // TODO(bazel-team): Add some tests for FileSystemValueChecker#changedKeys*() methods.
  // Presently these appear to be untested.

  private ActionExecutionValue actionValue(Action action, boolean forceDigest) {
    Map<Artifact, FileValue> artifactData = new HashMap<>();
    for (Artifact output : action.getOutputs()) {
      try {
        Path path = output.getPath();
        FileStatusWithDigest stat =
            forceDigest ? statWithDigest(path, path.statIfFound(Symlinks.NOFOLLOW)) : null;
        artifactData.put(output,
            ActionMetadataHandler.fileValueFromArtifact(output, stat, null));
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
    }
    return new ActionExecutionValue(
        artifactData,
        ImmutableMap.<Artifact, TreeArtifactValue>of(),
        ImmutableMap.<Artifact, FileArtifactValue>of());
  }

  private ActionExecutionValue actionValueWithEmptyDirectory(Artifact emptyDir) {
    TreeArtifactValue emptyValue = TreeArtifactValue.create
        (ImmutableMap.<TreeFileArtifact, FileArtifactValue>of());

    return new ActionExecutionValue(
        ImmutableMap.<Artifact, FileValue>of(),
        ImmutableMap.of(emptyDir, emptyValue),
        ImmutableMap.<Artifact, FileArtifactValue>of());
  }

  private ActionExecutionValue actionValueWithTreeArtifacts(List<TreeFileArtifact> contents) {
    Map<Artifact, FileValue> fileData = new HashMap<>();
    Map<Artifact, Map<TreeFileArtifact, FileArtifactValue>> directoryData = new HashMap<>();

    for (TreeFileArtifact output : contents) {
      try {
        Map<TreeFileArtifact, FileArtifactValue> dirDatum =
            directoryData.get(output.getParent());
        if (dirDatum == null) {
          dirDatum = new HashMap<>();
          directoryData.put(output.getParent(), dirDatum);
        }
        FileValue fileValue = ActionMetadataHandler.fileValueFromArtifact(output, null, null);
        dirDatum.put(output, FileArtifactValue.create(output, fileValue));
        fileData.put(output, fileValue);
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
    }

    Map<Artifact, TreeArtifactValue> treeArtifactData = new HashMap<>();
    for (Map.Entry<Artifact, Map<TreeFileArtifact, FileArtifactValue>> dirDatum :
        directoryData.entrySet()) {
      treeArtifactData.put(dirDatum.getKey(), TreeArtifactValue.create(dirDatum.getValue()));
    }

    return new ActionExecutionValue(fileData, treeArtifactData,
        ImmutableMap.<Artifact, FileArtifactValue>of());
  }

  @Test
  public void testPropagatesRuntimeExceptions() throws Exception {
    Collection<SkyKey> values = ImmutableList.of(
        FileValue.key(RootedPath.toRootedPath(pkgRoot, PathFragment.create("foo"))));
    driver.evaluate(
        values, false, SkyframeExecutor.DEFAULT_THREAD_COUNT, NullEventHandler.INSTANCE);
    FilesystemValueChecker checker = new FilesystemValueChecker(null, null);

    assertEmptyDiff(getDirtyFilesystemKeys(evaluator, checker));

    fs.statThrowsRuntimeException = true;
    try {
      getDirtyFilesystemKeys(evaluator, checker);
      fail();
    } catch (RuntimeException e) {
      assertThat(e).hasMessage("bork");
    }
  }

  private static void assertEmptyDiff(Diff diff) {
    assertDiffWithNewValues(diff);
  }

  private static void assertDiffWithNewValues(Diff diff, SkyKey... keysWithNewValues) {
    assertThat(diff.changedKeysWithoutNewValues()).isEmpty();
    assertThat(diff.changedKeysWithNewValues().keySet())
        .containsExactlyElementsIn(Arrays.asList(keysWithNewValues));
  }

  private class MockFileSystem extends InMemoryFileSystem {

    boolean statThrowsRuntimeException;
    boolean readlinkThrowsIoException;

    MockFileSystem() {
      super();
    }

    @Override
    public FileStatus stat(Path path, boolean followSymlinks) throws IOException {
      if (statThrowsRuntimeException) {
        throw new RuntimeException("bork");
      }
      return super.stat(path, followSymlinks);
    }

    @Override
    protected PathFragment readSymbolicLink(Path path) throws IOException {
      if (readlinkThrowsIoException) {
        throw new IOException("readlink failed");
      }
      return super.readSymbolicLink(path);
    }
  }

  private static FileStatusWithDigest statWithDigest(final Path path, final FileStatus stat) {
    return new FileStatusWithDigest() {
      @Nullable
      @Override
      public byte[] getDigest() throws IOException {
        return path.getDigest();
      }

      @Override
      public boolean isFile() {
        return stat.isFile();
      }

      @Override
      public boolean isSpecialFile() {
        return stat.isSpecialFile();
      }

      @Override
      public boolean isDirectory() {
        return stat.isDirectory();
      }

      @Override
      public boolean isSymbolicLink() {
        return stat.isSymbolicLink();
      }

      @Override
      public long getSize() throws IOException {
        return stat.getSize();
      }

      @Override
      public long getLastModifiedTime() throws IOException {
        return stat.getLastModifiedTime();
      }

      @Override
      public long getLastChangeTime() throws IOException {
        return stat.getLastChangeTime();
      }

      @Override
      public long getNodeId() throws IOException {
        return stat.getNodeId();
      }
    };
  }

  private static Diff getDirtyFilesystemKeys(MemoizingEvaluator evaluator,
      FilesystemValueChecker checker) throws InterruptedException {
    return checker.getDirtyKeys(evaluator.getValues(), new BasicFilesystemDirtinessChecker());
  }
}
