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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.hash.HashCode;
import com.google.common.util.concurrent.Runnables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.TestAction;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.BasicFilesystemDirtinessChecker;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestPackageFactoryBuilderFactory;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.testutil.TimestampGranularityUtils;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileStatusWithDigest;
import com.google.devtools.build.lib.vfs.FileStatusWithDigestAdapter;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.Differencer.Diff;
import com.google.devtools.build.skyframe.EvaluationContext;
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
import java.util.Collections;
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
  private static final EvaluationContext EVALUATION_OPTIONS =
      EvaluationContext.newBuilder()
          .setKeepGoing(false)
          .setNumThreads(SkyframeExecutor.DEFAULT_THREAD_COUNT)
          .setEventHander(NullEventHandler.INSTANCE)
          .build();

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
                ImmutableList.of(Root.fromPath(pkgRoot)),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(pkgRoot, pkgRoot, pkgRoot),
            pkgRoot,
            /* defaultSystemJavabase= */ null,
            TestConstants.PRODUCT_NAME);
    ExternalFilesHelper externalFilesHelper = ExternalFilesHelper.createForTesting(
        pkgLocator, ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS, directories);
    skyFunctions.put(
        FileStateValue.FILE_STATE,
        new FileStateFunction(
            new AtomicReference<TimestampGranularityMonitor>(),
            new AtomicReference<>(UnixGlob.DEFAULT_SYSCALLS),
            externalFilesHelper));
    skyFunctions.put(FileValue.FILE, new FileFunction(pkgLocator));
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
    skyFunctions.put(
        WorkspaceFileValue.WORKSPACE_FILE,
        new WorkspaceFileFunction(
            TestRuleClassProvider.getRuleClassProvider(),
            TestPackageFactoryBuilderFactory.getInstance()
                .builder(directories)
                .build(TestRuleClassProvider.getRuleClassProvider(), fs),
            directories,
            /*skylarkImportLookupFunctionForInlining=*/ null));
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

    SkyKey skyKey =
        FileStateValue.key(
            RootedPath.toRootedPath(Root.absoluteRoot(fs), PathFragment.create("/foo")));
    EvaluationResult<SkyValue> result =
        driver.evaluate(ImmutableList.of(skyKey), EVALUATION_OPTIONS);
    assertThat(result.hasError()).isFalse();

    assertEmptyDiff(getDirtyFilesystemKeys(evaluator, checker));

    FileSystemUtils.writeContentAsLatin1(path, "hello");
    assertDiffWithNewValues(getDirtyFilesystemKeys(evaluator, checker), skyKey);

    // The dirty bits are not reset until the FileValues are actually revalidated.
    assertDiffWithNewValues(getDirtyFilesystemKeys(evaluator, checker), skyKey);

    differencer.invalidate(ImmutableList.of(skyKey));
    result = driver.evaluate(ImmutableList.of(skyKey), EVALUATION_OPTIONS);
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
    // the FileValue created for symlink would notice, since it stats foo).
    Path sym1 = fs.getPath("/sym1");
    Path sym2 = fs.getPath("/sym2");
    Path symlink = fs.getPath("/bar");
    FileSystemUtils.ensureSymbolicLink(symlink, sym1);
    FileSystemUtils.ensureSymbolicLink(sym1, path);
    FileSystemUtils.ensureSymbolicLink(sym2, path);
    SkyKey fooKey =
        FileValue.key(RootedPath.toRootedPath(Root.absoluteRoot(fs), PathFragment.create("/foo")));
    RootedPath symlinkRootedPath =
        RootedPath.toRootedPath(Root.absoluteRoot(fs), PathFragment.create("/bar"));
    SkyKey symlinkKey = FileValue.key(symlinkRootedPath);
    SkyKey symlinkFileStateKey = FileStateValue.key(symlinkRootedPath);
    RootedPath sym1RootedPath =
        RootedPath.toRootedPath(Root.absoluteRoot(fs), PathFragment.create("/sym1"));
    SkyKey sym1FileStateKey = FileStateValue.key(sym1RootedPath);
    Iterable<SkyKey> allKeys = ImmutableList.of(symlinkKey, fooKey);

    // First build -- prime the graph.
    EvaluationResult<FileValue> result = driver.evaluate(allKeys, EVALUATION_OPTIONS);
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
    result = driver.evaluate(ImmutableList.<SkyKey>of(), EVALUATION_OPTIONS);
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
    result = driver.evaluate(allKeys, EVALUATION_OPTIONS);
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
            RootedPath.toRootedPath(Root.absoluteRoot(fs), PathFragment.create("/foo1")));
    SkyKey key2 =
        FileStateValue.key(
            RootedPath.toRootedPath(Root.absoluteRoot(fs), PathFragment.create("/foo2")));
    Iterable<SkyKey> skyKeys = ImmutableList.of(key1, key2);
    EvaluationResult<SkyValue> result = driver.evaluate(skyKeys, EVALUATION_OPTIONS);
    assertThat(result.hasError()).isFalse();

    assertEmptyDiff(getDirtyFilesystemKeys(evaluator, checker));

    // Wait for the timestamp granularity to elapse, so updating the files will observably advance
    // their ctime.
    TimestampGranularityUtils.waitForTimestampGranularity(
        System.currentTimeMillis(), OutErr.SYSTEM_OUT_ERR);
    // Update path1's contents and mtime. This will update the file's ctime.
    FileSystemUtils.writeContentAsLatin1(path1, "hello1");
    path1.setLastModifiedTime(27);
    // Update path2's mtime but not its contents. We expect that an mtime change suffices to update
    // the ctime.
    path2.setLastModifiedTime(42);
    // Assert that both files changed. The change detection relies, among other things, on ctime
    // change.
    assertDiffWithNewValues(getDirtyFilesystemKeys(evaluator, checker), key1, key2);

    differencer.invalidate(skyKeys);
    result = driver.evaluate(skyKeys, EVALUATION_OPTIONS);
    assertThat(result.hasError()).isFalse();
    assertEmptyDiff(getDirtyFilesystemKeys(evaluator, checker));
  }

  @Test
  public void testFileWithIOExceptionNotConsideredDirty() throws Exception {
    Path path = fs.getPath("/testroot/foo");
    path.getParentDirectory().createDirectory();
    path.createSymbolicLink(PathFragment.create("bar"));

    fs.readlinkThrowsIoException = true;
    SkyKey fileKey =
        FileStateValue.key(
            RootedPath.toRootedPath(Root.fromPath(pkgRoot), PathFragment.create("foo")));
    EvaluationResult<SkyValue> result =
        driver.evaluate(ImmutableList.of(fileKey), EVALUATION_OPTIONS);
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
    SkyKey fileKey1 = FileValue.key(RootedPath.toRootedPath(Root.fromPath(pkgRoot), path1));

    EvaluationResult<SkyValue> result =
        driver.evaluate(ImmutableList.of(fileKey1), EVALUATION_OPTIONS);
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

    TimestampGranularityMonitor tsgm = new TimestampGranularityMonitor(BlazeClock.instance());
    ActionLookupKey actionLookupKey =
        new ActionLookupKey() {
          @Override
          public SkyFunctionName functionName() {
            return SkyFunctionName.FOR_TESTING;
          }
        };
    SkyKey actionKey1 = ActionLookupData.create(actionLookupKey, 0);
    SkyKey actionKey2 = ActionLookupData.create(actionLookupKey, 1);

    pretendBuildTwoArtifacts(out1, actionKey1, out2, actionKey2, batchStatter, tsgm);

    // Change the file but not its size
    FileSystemUtils.writeContentAsLatin1(out1.getPath(), "hallo");
    checkActionDirtiedByFile(out1, actionKey1, batchStatter, tsgm);
    pretendBuildTwoArtifacts(out1, actionKey1, out2, actionKey2, batchStatter, tsgm);

    // Now try with a different size
    FileSystemUtils.writeContentAsLatin1(out1.getPath(), "hallo2");
    checkActionDirtiedByFile(out1, actionKey1, batchStatter, tsgm);
  }

  private void pretendBuildTwoArtifacts(
      Artifact out1,
      SkyKey actionKey1,
      Artifact out2,
      SkyKey actionKey2,
      BatchStat batchStatter,
      TimestampGranularityMonitor tsgm)
      throws InterruptedException {
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(1)
            .setEventHander(NullEventHandler.INSTANCE)
            .build();

    tsgm.setCommandStartTime();
    differencer.inject(
        ImmutableMap.<SkyKey, SkyValue>of(
            actionKey1,
                actionValue(
                    new TestAction(
                        Runnables.doNothing(),
                        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
                        ImmutableSet.of(out1))),
            actionKey2,
                actionValue(
                    new TestAction(
                        Runnables.doNothing(),
                        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
                        ImmutableSet.of(out2)))));
    assertThat(driver.evaluate(ImmutableList.<SkyKey>of(), evaluationContext).hasError()).isFalse();
    assertThat(
            new FilesystemValueChecker(tsgm, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    ModifiedFileSet.EVERYTHING_MODIFIED,
                    /* trustRemoteArtifacts= */ false))
        .isEmpty();

    tsgm.waitForTimestampGranularity(OutErr.SYSTEM_OUT_ERR);
  }

  private void checkActionDirtiedByFile(
      Artifact file, SkyKey actionKey, BatchStat batchStatter, TimestampGranularityMonitor tsgm)
      throws InterruptedException {
    assertThat(
            new FilesystemValueChecker(tsgm, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    ModifiedFileSet.EVERYTHING_MODIFIED,
                    /* trustRemoteArtifacts= */ false))
        .containsExactly(actionKey);
    assertThat(
            new FilesystemValueChecker(tsgm, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    new ModifiedFileSet.Builder().modify(file.getExecPath()).build(),
                    /* trustRemoteArtifacts= */ false))
        .containsExactly(actionKey);
    assertThat(
            new FilesystemValueChecker(tsgm, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    new ModifiedFileSet.Builder()
                        .modify(file.getExecPath().getParentDirectory())
                        .build(),
                    /* trustRemoteArtifacts= */ false))
        .isEmpty();
    assertThat(
            new FilesystemValueChecker(tsgm, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    ModifiedFileSet.NOTHING_MODIFIED,
                    /* trustRemoteArtifacts= */ false))
        .isEmpty();
  }

  private void checkDirtyTreeArtifactActions(BatchStat batchStatter) throws Exception {
    // Normally, an Action specifies the contents of a TreeArtifact when it executes.
    // To decouple FileSystemValueTester checking from Action execution, we inject TreeArtifact
    // contents into ActionExecutionValues.

    SpecialArtifact out1 = createTreeArtifact("one");
    TreeFileArtifact file11 =
        ActionsTestUtil.createTreeFileArtifactWithNoGeneratingAction(out1, "fizz");
    FileSystemUtils.createDirectoryAndParents(out1.getPath());
    FileSystemUtils.writeContentAsLatin1(file11.getPath(), "buzz");

    SpecialArtifact out2 = createTreeArtifact("two");
    FileSystemUtils.createDirectoryAndParents(out2.getPath().getChild("subdir"));
    TreeFileArtifact file21 =
        ActionsTestUtil.createTreeFileArtifactWithNoGeneratingAction(out2, "moony");
    TreeFileArtifact file22 =
        ActionsTestUtil.createTreeFileArtifactWithNoGeneratingAction(out2, "subdir/wormtail");
    FileSystemUtils.writeContentAsLatin1(file21.getPath(), "padfoot");
    FileSystemUtils.writeContentAsLatin1(file22.getPath(), "prongs");

    SpecialArtifact outEmpty = createTreeArtifact("empty");
    FileSystemUtils.createDirectoryAndParents(outEmpty.getPath());

    SpecialArtifact outUnchanging = createTreeArtifact("untouched");
    FileSystemUtils.createDirectoryAndParents(outUnchanging.getPath());

    SpecialArtifact last = createTreeArtifact("zzzzzzzzzz");
    FileSystemUtils.createDirectoryAndParents(last.getPath());

    ActionLookupKey actionLookupKey =
        new ActionLookupKey() {
          @Override
          public SkyFunctionName functionName() {
            return SkyFunctionName.FOR_TESTING;
          }
        };
    SkyKey actionKey1 = ActionLookupData.create(actionLookupKey, 0);
    SkyKey actionKey2 = ActionLookupData.create(actionLookupKey, 1);
    SkyKey actionKeyEmpty = ActionLookupData.create(actionLookupKey, 2);
    SkyKey actionKeyUnchanging = ActionLookupData.create(actionLookupKey, 3);
    SkyKey actionKeyLast = ActionLookupData.create(actionLookupKey, 4);
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

    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(1)
            .setEventHander(NullEventHandler.INSTANCE)
            .build();
    assertThat(driver.evaluate(ImmutableList.<SkyKey>of(), evaluationContext).hasError()).isFalse();
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    ModifiedFileSet.EVERYTHING_MODIFIED,
                    /* trustRemoteArtifacts= */ false))
        .isEmpty();

    // Touching the TreeArtifact directory should have no effect
    FileSystemUtils.touchFile(out1.getPath());
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    ModifiedFileSet.EVERYTHING_MODIFIED,
                    /* trustRemoteArtifacts= */ false))
        .isEmpty();
    // Neither should touching a subdirectory.
    FileSystemUtils.touchFile(out2.getPath().getChild("subdir"));
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    ModifiedFileSet.EVERYTHING_MODIFIED,
                    /* trustRemoteArtifacts= */ false))
        .isEmpty();

    /* **** Tests for directories **** */

    // Removing a directory (even if empty) should have an effect
    outEmpty.getPath().delete();
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    new ModifiedFileSet.Builder().modify(outEmpty.getExecPath()).build(),
                    /* trustRemoteArtifacts= */ false))
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
                    new ModifiedFileSet.Builder().modify(outEmpty.getExecPath()).build(),
                    /* trustRemoteArtifacts= */ false))
        .containsExactly(actionKeyEmpty);

    // We're done fiddling with this... restore the original state
    outEmpty.getPath().delete();
    dummyEmptyDir.deleteTree();
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
                    new ModifiedFileSet.Builder().modify(file11.getExecPath()).build(),
                    /* trustRemoteArtifacts= */ false))
        .containsExactly(actionKey1);

    // Test that directory contents (and nested contents) matter
    Artifact out1new =
        ActionsTestUtil.createTreeFileArtifactWithNoGeneratingAction(out1, "julius/caesar");
    FileSystemUtils.createDirectoryAndParents(out1.getPath().getChild("julius"));
    FileSystemUtils.writeContentAsLatin1(out1new.getPath(), "octavian");
    // even for empty directories
    Artifact outEmptyNew =
        ActionsTestUtil.createTreeFileArtifactWithNoGeneratingAction(outEmpty, "marcus");
    FileSystemUtils.writeContentAsLatin1(outEmptyNew.getPath(), "aurelius");
    // so does removing
    file21.getPath().delete();
    // now, let's test our changes are actually visible
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    ModifiedFileSet.EVERYTHING_MODIFIED,
                    /* trustRemoteArtifacts= */ false))
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
                        .build(),
                    /* trustRemoteArtifacts= */ false))
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
                        .build(),
                    /* trustRemoteArtifacts= */ false))
        .containsExactly(actionKey2, actionKeyEmpty);
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    new ModifiedFileSet.Builder()
                        .modify(file21.getExecPath())
                        .modify(out1new.getExecPath())
                        .build(),
                    /* trustRemoteArtifacts= */ false))
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
                        .build(),
                    /* trustRemoteArtifacts= */ false))
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
                        .build(),
                    /* trustRemoteArtifacts= */ false))
        .containsExactly(actionKey1, actionKey2);
    // Restore
    last.getPath().delete();
    FileSystemUtils.createDirectoryAndParents(last.getPath());
    // We add a test for NOTHING_MODIFIED, because FileSystemValueChecker doesn't
    // pay attention to file sets for TreeArtifact directory listings.
    assertThat(
            new FilesystemValueChecker(null, null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    batchStatter,
                    ModifiedFileSet.NOTHING_MODIFIED,
                    /* trustRemoteArtifacts= */ false))
        .isEmpty();
  }

  private Artifact createDerivedArtifact(String relPath) throws IOException {
    String outSegment = "bin";
    Path outputPath = fs.getPath("/" + outSegment);
    outputPath.createDirectory();
    return ActionsTestUtil.createArtifact(
        ArtifactRoot.asDerivedRoot(fs.getPath("/"), outSegment), outputPath.getRelative(relPath));
  }

  private SpecialArtifact createTreeArtifact(String relPath) throws IOException {
    String outSegment = "bin";
    Path outputDir = fs.getPath("/" + outSegment);
    Path outputPath = outputDir.getRelative(relPath);
    outputDir.createDirectory();
    ArtifactRoot derivedRoot = ArtifactRoot.asDerivedRoot(fs.getPath("/"), outSegment);
    return new SpecialArtifact(
        derivedRoot,
        derivedRoot.getExecPath().getRelative(derivedRoot.getRoot().relativize(outputPath)),
        ActionsTestUtil.NULL_ARTIFACT_OWNER,
        SpecialArtifactType.TREE);
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
                      fs.getPath("/").getRelative(pathFrag).statIfFound(Symlinks.NOFOLLOW)));
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
              final Path path = fs.getPath("/").getRelative(pathFrag);
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
                      fs.getPath("/").getRelative(pathFrag).statIfFound(Symlinks.NOFOLLOW)));
            }
            return stats;
          }
        });
  }

  // TODO(bazel-team): Add some tests for FileSystemValueChecker#changedKeys*() methods.
  // Presently these appear to be untested.

  private ActionExecutionValue actionValue(Action action) {
    Map<Artifact, FileArtifactValue> artifactData = new HashMap<>();
    for (Artifact output : action.getOutputs()) {
      try {
        Path path = output.getPath();
        FileArtifactValue noDigest =
            ActionMetadataHandler.fileArtifactValueFromArtifact(
                output,
                FileStatusWithDigestAdapter.adapt(path.statIfFound(Symlinks.NOFOLLOW)),
                null);
        FileArtifactValue withDigest =
            FileArtifactValue.createFromInjectedDigest(
                noDigest, path.getDigest(), !output.isConstantMetadata());
        artifactData.put(output, withDigest);
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
    }
    return ActionExecutionValue.create(
        artifactData,
        ImmutableMap.<Artifact, TreeArtifactValue>of(),
        /*outputSymlinks=*/ null,
        /*discoveredModules=*/ null,
        /*actionDependsOnBuildId=*/ false);
  }

  private ActionExecutionValue actionValueWithEmptyDirectory(Artifact emptyDir) {
    TreeArtifactValue emptyValue = TreeArtifactValue.create
        (ImmutableMap.<TreeFileArtifact, FileArtifactValue>of());

    return ActionExecutionValue.create(
        ImmutableMap.of(),
        ImmutableMap.of(emptyDir, emptyValue),
        /*outputSymlinks=*/ null,
        /*discoveredModules=*/ null,
        /*actionDependsOnBuildId=*/ false);
  }

  private ActionExecutionValue actionValueWithTreeArtifacts(List<TreeFileArtifact> contents) {
    Map<Artifact, FileArtifactValue> fileData = new HashMap<>();
    Map<Artifact, Map<TreeFileArtifact, FileArtifactValue>> directoryData = new HashMap<>();

    for (TreeFileArtifact output : contents) {
      try {
        Map<TreeFileArtifact, FileArtifactValue> dirDatum =
            directoryData.get(output.getParent());
        if (dirDatum == null) {
          dirDatum = new HashMap<>();
          directoryData.put(output.getParent(), dirDatum);
        }
        Path path = output.getPath();
        FileArtifactValue noDigest =
            ActionMetadataHandler.fileArtifactValueFromArtifact(
                output,
                FileStatusWithDigestAdapter.adapt(path.statIfFound(Symlinks.NOFOLLOW)),
                null);
        FileArtifactValue withDigest =
            FileArtifactValue.createFromInjectedDigest(
                noDigest, path.getDigest(), !output.isConstantMetadata());
        dirDatum.put(output, withDigest);
        fileData.put(output, withDigest);
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
    }

    Map<Artifact, TreeArtifactValue> treeArtifactData = new HashMap<>();
    for (Map.Entry<Artifact, Map<TreeFileArtifact, FileArtifactValue>> dirDatum :
        directoryData.entrySet()) {
      treeArtifactData.put(dirDatum.getKey(), TreeArtifactValue.create(dirDatum.getValue()));
    }

    return ActionExecutionValue.create(
        fileData,
        treeArtifactData,
        /*outputSymlinks=*/ null,
        /*discoveredModules=*/ null,
        /*actionDependsOnBuildId=*/ false);
  }

  private ActionExecutionValue actionValueWithRemoteTreeArtifact(
      SpecialArtifact output, Map<PathFragment, RemoteFileArtifactValue> children) {
    ImmutableMap.Builder<TreeFileArtifact, FileArtifactValue> childFileValues =
        ImmutableMap.builder();
    for (Map.Entry<PathFragment, RemoteFileArtifactValue> child : children.entrySet()) {
      childFileValues.put(
          ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
              output, child.getKey(), output.getArtifactOwner()),
          child.getValue());
    }
    TreeArtifactValue treeArtifactValue = TreeArtifactValue.create(childFileValues.build());
    return ActionExecutionValue.create(
        ImmutableMap.of(),
        Collections.singletonMap(output, treeArtifactValue),
        /* outputSymlinks= */ null,
        /* discoveredModules= */ null,
        /* actionDependsOnBuildId= */ false);
  }

  private ActionExecutionValue actionValueWithRemoteArtifact(
      Artifact output, RemoteFileArtifactValue value) {
    return ActionExecutionValue.create(
        Collections.singletonMap(output, value),
        ImmutableMap.of(),
        /* outputSymlinks= */ null,
        /* discoveredModules= */ null,
        /* actionDependsOnBuildId= */ false);
  }

  private RemoteFileArtifactValue createRemoteFileArtifactValue(String contents) {
    byte[] data = contents.getBytes();
    DigestHashFunction hashFn = fs.getDigestFunction();
    HashCode hash = hashFn.getHashFunction().hashBytes(data);
    return new RemoteFileArtifactValue(hash.asBytes(), data.length, -1);
  }

  @Test
  public void testRemoteAndLocalArtifacts() throws Exception {
    // Test that injected remote artifacts are trusted by the FileSystemValueChecker
    // if it is configured to trust remote artifacts, and that local files always take precedence
    // over remote files.
    ActionLookupKey actionLookupKey =
        new ActionLookupKey() {
          @Override
          public SkyFunctionName functionName() {
            return SkyFunctionName.FOR_TESTING;
          }
        };
    SkyKey actionKey1 = ActionLookupData.create(actionLookupKey, 0);
    SkyKey actionKey2 = ActionLookupData.create(actionLookupKey, 1);

    Artifact out1 = createDerivedArtifact("foo");
    Artifact out2 = createDerivedArtifact("bar");
    Map<SkyKey, SkyValue> metadataToInject = new HashMap<>();
    metadataToInject.put(
        actionKey1,
        actionValueWithRemoteArtifact(out1, createRemoteFileArtifactValue("foo-content")));
    metadataToInject.put(
        actionKey2,
        actionValueWithRemoteArtifact(out2, createRemoteFileArtifactValue("bar-content")));
    differencer.inject(metadataToInject);

    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(1)
            .setEventHander(NullEventHandler.INSTANCE)
            .build();
    assertThat(
            driver.evaluate(ImmutableList.of(actionKey1, actionKey2), evaluationContext).hasError())
        .isFalse();
    assertThat(
            new FilesystemValueChecker(/* tsgm= */ null, /* lastExecutionTimeRange= */ null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    /* batchStatter= */ null,
                    ModifiedFileSet.EVERYTHING_MODIFIED,
                    /* trustRemoteArtifacts= */ true))
        .isEmpty();

    // Create the "out1" artifact on the filesystem and test that it invalidates the generating
    // action's SkyKey.
    FileSystemUtils.writeContentAsLatin1(out1.getPath(), "new-foo-content");
    assertThat(
            new FilesystemValueChecker(/* tsgm= */ null, /* lastExecutionTimeRange= */ null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    /* batchStatter= */ null,
                    ModifiedFileSet.EVERYTHING_MODIFIED,
                    /* trustRemoteArtifacts= */ true))
        .containsExactly(actionKey1);
  }

  @Test
  public void testRemoteAndLocalTreeArtifacts() throws Exception {
    // Test that injected remote tree artifacts are trusted by the FileSystemValueChecker
    // and that local files always takes preference over remote files.
    ActionLookupKey actionLookupKey =
        new ActionLookupKey() {
          @Override
          public SkyFunctionName functionName() {
            return SkyFunctionName.FOR_TESTING;
          }
        };
    SkyKey actionKey = ActionLookupData.create(actionLookupKey, 0);

    SpecialArtifact treeArtifact = createTreeArtifact("dir");
    treeArtifact.getPath().createDirectoryAndParents();
    Map<PathFragment, RemoteFileArtifactValue> treeArtifactMetadata = new HashMap<>();
    treeArtifactMetadata.put(
        PathFragment.create("foo"), createRemoteFileArtifactValue("foo-content"));
    treeArtifactMetadata.put(
        PathFragment.create("bar"), createRemoteFileArtifactValue("bar-content"));

    Map<SkyKey, SkyValue> metadataToInject = new HashMap<>();
    metadataToInject.put(
        actionKey, actionValueWithRemoteTreeArtifact(treeArtifact, treeArtifactMetadata));
    differencer.inject(metadataToInject);

    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(1)
            .setEventHander(NullEventHandler.INSTANCE)
            .build();
    assertThat(driver.evaluate(ImmutableList.of(actionKey), evaluationContext).hasError())
        .isFalse();
    assertThat(
            new FilesystemValueChecker(/* tsgm= */ null, /* lastExecutionTimeRange= */ null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    /* batchStatter= */ null,
                    ModifiedFileSet.EVERYTHING_MODIFIED,
                    /* trustRemoteArtifacts= */ false))
        .isEmpty();

    // Create dir/foo on the local disk and test that it invalidates the associated sky key.
    TreeFileArtifact fooArtifact =
        ActionsTestUtil.createTreeFileArtifactWithNoGeneratingAction(treeArtifact, "foo");
    FileSystemUtils.writeContentAsLatin1(fooArtifact.getPath(), "new-foo-content");
    assertThat(
            new FilesystemValueChecker(/* tsgm= */ null, /* lastExecutionTimeRange= */ null)
                .getDirtyActionValues(
                    evaluator.getValues(),
                    /* batchStatter= */ null,
                    ModifiedFileSet.EVERYTHING_MODIFIED,
                    /* trustRemoteArtifacts= */ false))
        .containsExactly(actionKey);
  }

  @Test
  public void testPropagatesRuntimeExceptions() throws Exception {
    Collection<SkyKey> values =
        ImmutableList.of(
            FileValue.key(
                RootedPath.toRootedPath(Root.fromPath(pkgRoot), PathFragment.create("foo"))));
    driver.evaluate(values, EVALUATION_OPTIONS);
    FilesystemValueChecker checker = new FilesystemValueChecker(null, null);

    assertEmptyDiff(getDirtyFilesystemKeys(evaluator, checker));

    fs.statThrowsRuntimeException = true;
    RuntimeException e =
        assertThrows(RuntimeException.class, () -> getDirtyFilesystemKeys(evaluator, checker));
    assertThat(e).hasMessageThat().isEqualTo("bork");
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
    public FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
      if (statThrowsRuntimeException) {
        throw new RuntimeException("bork");
      }
      return super.statIfFound(path, followSymlinks);
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
