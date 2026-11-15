// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.withSettings;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionTemplate;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.RunfilesArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.BasicActionLookupValue;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.RunfilesArtifactValue;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue.ActionTemplateExpansionKey;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.InOrder;

/** Tests for {@link RemoteRewoundActionSynchronizer}. */
@RunWith(TestParameterInjector.class)
public final class RemoteRewoundActionSynchronizerTest {
  private static final long DEADLOCK_TIMEOUT_MILLIS = 10_000;

  private RemoteActionInputFetcher actionInputFetcher;
  private WalkableGraph graph;
  private RemoteRewoundActionSynchronizer synchronizer;

  @Before
  public void setUp() {
    actionInputFetcher = mock(RemoteActionInputFetcher.class);
    graph = mock(WalkableGraph.class);
    synchronizer = new RemoteRewoundActionSynchronizer(actionInputFetcher, graph);
  }

  @Test
  public void rewind_cancelsAllRegisteredTasksOnce() throws Exception {
    Action action = newAction();
    var first = mock(RemoteRewoundActionSynchronizer.Cancellable.class);
    var second = mock(RemoteRewoundActionSynchronizer.Cancellable.class);
    var unusedFirst = synchronizer.registerOutputUploadTask(action, first);
    var unusedSecond = synchronizer.registerOutputUploadTask(action, second);

    rewind(action);
    rewind(action);

    verify(first, times(1)).requestCancellation();
    verify(first, times(1)).awaitCompletion();
    verify(second, times(1)).requestCancellation();
    verify(second, times(1)).awaitCompletion();
  }

  @Test
  public void rewind_interruptedWhileAwaiting_cancelsAndAwaitsEveryTask() throws Exception {
    Action action = newAction();
    var first = mock(RemoteRewoundActionSynchronizer.Cancellable.class);
    var second = mock(RemoteRewoundActionSynchronizer.Cancellable.class);
    doThrow(new InterruptedException()).doNothing().when(first).awaitCompletion();
    var unusedFirst = synchronizer.registerOutputUploadTask(action, first);
    var unusedSecond = synchronizer.registerOutputUploadTask(action, second);

    assertThrows(InterruptedException.class, () -> rewind(action));

    InOrder order = inOrder(first, second);
    order.verify(first).requestCancellation();
    order.verify(second).requestCancellation();
    order.verify(first, times(2)).awaitCompletion();
    order.verify(second).awaitCompletion();
    verify(actionInputFetcher, never()).handleRewoundActionOutputs(any());

    var retry = new TestThread(() -> rewind(action));
    retry.start();
    retry.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
    verify(actionInputFetcher).handleRewoundActionOutputs(any());
  }

  @Test
  public void unregisterHandle_preventsCancellation() throws Exception {
    Action action = newAction();
    var task = mock(RemoteRewoundActionSynchronizer.Cancellable.class);
    Runnable unregister = synchronizer.registerOutputUploadTask(action, task);

    unregister.run();
    rewind(action);

    verify(task, never()).requestCancellation();
    verify(task, never()).awaitCompletion();
    assertThat(synchronizer.hasRegisteredOutputUploadTasks(action)).isFalse();
  }

  @Test
  public void predecessorUnregisterHandle_doesNotRemoveReplacementTask() throws Exception {
    Action action = newAction();
    var predecessor = mock(RemoteRewoundActionSynchronizer.Cancellable.class);
    Runnable unregisterPredecessor = synchronizer.registerOutputUploadTask(action, predecessor);
    rewind(action);
    var replacement = mock(RemoteRewoundActionSynchronizer.Cancellable.class);
    var unused = synchronizer.registerOutputUploadTask(action, replacement);

    unregisterPredecessor.run();
    rewind(action);

    verify(predecessor, times(1)).requestCancellation();
    verify(predecessor, times(1)).awaitCompletion();
    verify(replacement, times(1)).requestCancellation();
    verify(replacement, times(1)).awaitCompletion();
  }

  @Test
  public void unregisterHandle_usesIdentityComparison() throws Exception {
    Action action = newAction();
    var first = new EqualCancellable();
    var second = new EqualCancellable();
    Runnable unregisterFirst = synchronizer.registerOutputUploadTask(action, first);
    var unused = synchronizer.registerOutputUploadTask(action, second);

    unregisterFirst.run();
    rewind(action);

    assertThat(first.cancellations.get()).isEqualTo(0);
    assertThat(second.cancellations.get()).isEqualTo(1);
  }

  /**
   * Shared actions are equivalent actions of different owners generating the same outputs, which
   * SkyframeActionExecutor executes as one. A consumer of one owner's output excludes the rewinding
   * of the other owner's action, which deletes the same file, and that rewinding cancels the
   * uploads registered by either owner.
   */
  @Test
  public void sharedAction_rewoundUnderOtherOwner_waitsForConsumerAndCancelsUpload(
      @TestParameter boolean inflateFineLocksFirst) throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;
    var otherOwner = ActionsTestUtil.YET_ANOTHER_NULL_ARTIFACT_OWNER;
    PathFragment sharedExecPath = root.getExecPath().getRelative("shared");
    DerivedArtifact output = DerivedArtifact.create(root, sharedExecPath, owner);
    output.setGeneratingActionKey(ActionLookupData.create(owner, 0));
    Action action = newAction(ImmutableList.of(output), ImmutableList.of());
    DerivedArtifact otherOutput = DerivedArtifact.create(root, sharedExecPath, otherOwner);
    otherOutput.setGeneratingActionKey(ActionLookupData.create(otherOwner, 0));
    Action otherAction = newAction(ImmutableList.of(otherOutput), ImmutableList.of());
    DerivedArtifact consumerOutput =
        (DerivedArtifact) ActionsTestUtil.createArtifact(root, "consumer.out");
    consumerOutput.setGeneratingActionKey(ActionLookupData.create(owner, 1));
    Action consumer = newAction(ImmutableList.of(consumerOutput), ImmutableList.of(output));
    mockActions(owner, ImmutableList.of(action, consumer));
    mockActions(otherOwner, ImmutableList.of(otherAction));
    var upload = mock(RemoteRewoundActionSynchronizer.Cancellable.class);
    var unused = synchronizer.registerOutputUploadTask(action, upload);

    maybeInflateFineLocks(inflateFineLocksFirst);

    var otherPreparation = new TestThread(() -> rewind(otherAction));
    try (SilentCloseable execution =
        synchronizer.enterActionExecution(
            consumer, /* wasRewound= */ false, mock(InputMetadataProvider.class))) {
      otherPreparation.start();
      waitUntilState(otherPreparation, Thread.State.WAITING);
    }
    otherPreparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
    verify(upload).requestCancellation();
    verify(upload).awaitCompletion();
    assertThat(synchronizer.hasRegisteredOutputUploadTasks(action)).isFalse();
  }

  /**
   * Input discovery may read scheduling dependencies (e.g. headers reachable via {@code #include})
   * that are not declared inputs, so their producers must not be rewound while it is running.
   */
  @Test
  public void inputDiscovery_locksProducersOfSchedulingDependencies(
      @TestParameter boolean inflateFineLocksFirst) throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;
    DerivedArtifact header = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "header.h");
    header.setGeneratingActionKey(ActionLookupData.create(owner, 0));
    Action headerProducer = newAction(ImmutableList.of(header), ImmutableList.of());
    DerivedArtifact source = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "source.cc");
    source.setGeneratingActionKey(ActionLookupData.create(owner, 1));
    Action sourceProducer = newAction(ImmutableList.of(source), ImmutableList.of());
    DerivedArtifact object = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "object.o");
    object.setGeneratingActionKey(ActionLookupData.create(owner, 2));
    Action compile =
        newAction(
            ImmutableList.of(object),
            /* inputs= */ ImmutableList.of(source),
            /* schedulingDependencies= */ ImmutableList.of(header));
    mockActions(owner, ImmutableList.of(headerProducer, sourceProducer, compile));
    InputMetadataProvider metadataProvider = mock(InputMetadataProvider.class);

    maybeInflateFineLocks(inflateFineLocksFirst);

    var headerProducerPreparation = new TestThread(() -> rewind(headerProducer));
    try (SilentCloseable discovery =
        synchronizer.enterInputDiscovery(
            compile, NestedSetBuilder.create(Order.STABLE_ORDER, source), metadataProvider)) {
      headerProducerPreparation.start();
      waitUntilState(headerProducerPreparation, Thread.State.WAITING);
    }
    headerProducerPreparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
  }

  /**
   * The output of a symlink action may be materialized as a symlink to its target, so the producer
   * of the target must not be rewound while a consumer of the output runs, even across a chain of
   * symlink actions.
   */
  @Test
  public void symlinkTargetProducerRewound_waitsForSymlinkConsumer(
      @TestParameter boolean inflateFineLocksFirst) throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;
    DerivedArtifact target = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "target");
    target.setGeneratingActionKey(ActionLookupData.create(owner, 0));
    Action targetProducer = newAction(ImmutableList.of(target), ImmutableList.of());
    DerivedArtifact symlink = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "symlink");
    symlink.setGeneratingActionKey(ActionLookupData.create(owner, 1));
    Action symlinkAction = newAction(ImmutableList.of(symlink), ImmutableList.of(target));
    when(symlinkAction.mayInsensitivelyPropagateInputs()).thenReturn(true);
    DerivedArtifact symlinkToSymlink =
        (DerivedArtifact) ActionsTestUtil.createArtifact(root, "symlink_to_symlink");
    symlinkToSymlink.setGeneratingActionKey(ActionLookupData.create(owner, 2));
    Action symlinkToSymlinkAction =
        newAction(ImmutableList.of(symlinkToSymlink), ImmutableList.of(symlink));
    when(symlinkToSymlinkAction.mayInsensitivelyPropagateInputs()).thenReturn(true);
    DerivedArtifact consumerOutput =
        (DerivedArtifact) ActionsTestUtil.createArtifact(root, "consumer.out");
    consumerOutput.setGeneratingActionKey(ActionLookupData.create(owner, 3));
    Action consumer =
        newAction(ImmutableList.of(consumerOutput), ImmutableList.of(symlinkToSymlink));
    mockActions(
        owner, ImmutableList.of(targetProducer, symlinkAction, symlinkToSymlinkAction, consumer));

    maybeInflateFineLocks(inflateFineLocksFirst);

    var targetProducerPreparation = new TestThread(() -> rewind(targetProducer));
    try (SilentCloseable execution =
        synchronizer.enterActionExecution(
            consumer, /* wasRewound= */ false, mock(InputMetadataProvider.class))) {
      targetProducerPreparation.start();
      waitUntilState(targetProducerPreparation, Thread.State.WAITING);
    }
    targetProducerPreparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
  }

  /**
   * An action consuming a runfiles tree is guarded by the keys of the actions generating the
   * artifacts the tree contains, which are taken from its metadata rather than from all runfiles
   * trees known to the metadata provider. The action generating the runfiles tree itself isn't
   * excluded, as it doesn't write to disk.
   */
  @Test
  public void runfilesTreeConsumer_excludesOnlyProducersOfItsRunfiles(
      @TestParameter boolean inflateFineLocksFirst) throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;
    DerivedArtifact file = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "file");
    file.setGeneratingActionKey(ActionLookupData.create(owner, 0));
    Action producer = newAction(ImmutableList.of(file), ImmutableList.of());
    DerivedArtifact unrelatedFile =
        (DerivedArtifact) ActionsTestUtil.createArtifact(root, "unrelated");
    unrelatedFile.setGeneratingActionKey(ActionLookupData.create(owner, 1));
    Action unrelatedProducer = newAction(ImmutableList.of(unrelatedFile), ImmutableList.of());
    RunfilesArtifact runfiles = ActionsTestUtil.createRunfilesArtifact(root, "out/runfiles");
    runfiles.setGeneratingActionKey(ActionLookupData.create(owner, 2));
    Action runfilesAction = newAction(ImmutableList.of(runfiles), ImmutableList.of(file));
    DerivedArtifact consumerOutput =
        (DerivedArtifact) ActionsTestUtil.createArtifact(root, "consumer.out");
    consumerOutput.setGeneratingActionKey(ActionLookupData.create(owner, 3));
    Action consumer = newAction(ImmutableList.of(consumerOutput), ImmutableList.of(runfiles));
    mockActions(owner, ImmutableList.of(producer, unrelatedProducer, runfilesAction, consumer));

    RunfilesTree runfilesTree = mock(RunfilesTree.class);
    when(runfilesTree.getArtifacts()).thenReturn(NestedSetBuilder.create(Order.STABLE_ORDER, file));
    RunfilesTree unrelatedRunfilesTree = mock(RunfilesTree.class);
    when(unrelatedRunfilesTree.getArtifacts())
        .thenReturn(NestedSetBuilder.create(Order.STABLE_ORDER, unrelatedFile));
    InputMetadataProvider metadataProvider = mock(InputMetadataProvider.class);
    when(metadataProvider.getRunfilesTrees())
        .thenReturn(ImmutableList.of(runfilesTree, unrelatedRunfilesTree));
    var runfilesValue =
        new RunfilesArtifactValue(
            runfilesTree,
            ImmutableList.of(file),
            ImmutableList.of(FileArtifactValue.createForNormalFile(new byte[32], null, 0)),
            ImmutableList.of(),
            ImmutableList.of(),
            ImmutableList.of(),
            ImmutableList.of());
    when(metadataProvider.getRunfilesMetadata(runfiles)).thenReturn(runfilesValue);

    maybeInflateFineLocks(inflateFineLocksFirst);

    var preparation = new TestThread(() -> rewind(producer));
    var unrelatedPreparation = new TestThread(() -> rewind(unrelatedProducer));
    var runfilesPreparation = new TestThread(() -> rewind(runfilesAction));
    try (SilentCloseable execution =
        synchronizer.enterActionExecution(consumer, /* wasRewound= */ false, metadataProvider)) {
      preparation.start();
      waitUntilState(preparation, Thread.State.WAITING);
      unrelatedPreparation.start();
      unrelatedPreparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
      runfilesPreparation.start();
      runfilesPreparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
    }
    preparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
  }

  /**
   * Runfiles trees are hidden top-level outputs and thus only reach {@link
   * RemoteImportantOutputHandler} through the metadata provider, but the rewinding of the actions
   * generating their artifacts must still be excluded while the handler processes them.
   */
  @Test
  public void outputProcessing_waitsForRewoundRunfilesProducer() throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;
    DerivedArtifact runfile = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "runfile");
    runfile.setGeneratingActionKey(ActionLookupData.create(owner, 0));
    Action producer = newAction(ImmutableList.of(runfile), ImmutableList.of());
    mockActions(owner, ImmutableList.of(producer));
    InputMetadataProvider metadataProvider =
        mockTopLevelMetadataProvider(
            /* runfiles= */ ImmutableList.of(runfile),
            /* remoteArtifacts= */ ImmutableList.of(runfile));
    when(actionInputFetcher.prefetchFiles(any(), any(), any(), any(), any(), any()))
        .thenReturn(immediateVoidFuture());
    RemoteImportantOutputHandler handler = newImportantOutputHandler(runfile);

    var processing =
        new TestThread(
            () ->
                assertThat(
                        handler
                            .processOutputsAndGetLostArtifacts(
                                /* importantOutputs= */ ImmutableList.of(), metadataProvider)
                            .isEmpty())
                    .isTrue());
    try (SilentCloseable preparation =
        synchronizer.enterActionPreparation(producer, /* wasRewound= */ true)) {
      processing.start();
      waitUntilState(processing, Thread.State.WAITING);
    }
    processing.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
  }

  @Test
  public void outputProcessing_excludesOnlyRewindingOfProducersOfOutputsAndRunfiles(
      @TestParameter boolean inflateFineLocksFirst) throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;
    DerivedArtifact importantOutput =
        (DerivedArtifact) ActionsTestUtil.createArtifact(root, "important");
    importantOutput.setGeneratingActionKey(ActionLookupData.create(owner, 0));
    Action importantOutputProducer =
        newAction(ImmutableList.of(importantOutput), ImmutableList.of());
    DerivedArtifact runfile = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "runfile");
    runfile.setGeneratingActionKey(ActionLookupData.create(owner, 1));
    Action runfileProducer = newAction(ImmutableList.of(runfile), ImmutableList.of());
    DerivedArtifact unrelated = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "unrelated");
    unrelated.setGeneratingActionKey(ActionLookupData.create(owner, 2));
    Action unrelatedProducer = newAction(ImmutableList.of(unrelated), ImmutableList.of());
    mockActions(
        owner, ImmutableList.of(importantOutputProducer, runfileProducer, unrelatedProducer));
    InputMetadataProvider metadataProvider =
        mockTopLevelMetadataProvider(
            /* runfiles= */ ImmutableList.of(runfile),
            /* remoteArtifacts= */ ImmutableList.of(importantOutput, runfile));
    // The processing holds its locks while the download is pending.
    SettableFuture<Void> download = SettableFuture.create();
    when(actionInputFetcher.prefetchFiles(any(), any(), any(), any(), any(), any()))
        .thenReturn(download);
    RemoteImportantOutputHandler handler = newImportantOutputHandler(importantOutput, runfile);

    maybeInflateFineLocks(inflateFineLocksFirst);

    var processing =
        new TestThread(
            () ->
                assertThat(
                        handler
                            .processOutputsAndGetLostArtifacts(
                                ImmutableList.of(importantOutput), metadataProvider)
                            .isEmpty())
                    .isTrue());
    processing.start();
    waitUntilState(processing, Thread.State.WAITING);

    var importantOutputPreparation = new TestThread(() -> rewind(importantOutputProducer));
    importantOutputPreparation.start();
    waitUntilState(importantOutputPreparation, Thread.State.WAITING);
    var runfilePreparation = new TestThread(() -> rewind(runfileProducer));
    runfilePreparation.start();
    waitUntilState(runfilePreparation, Thread.State.WAITING);
    var unrelatedPreparation = new TestThread(() -> rewind(unrelatedProducer));
    unrelatedPreparation.start();
    unrelatedPreparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);

    download.set(null);
    processing.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
    importantOutputPreparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
    runfilePreparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
  }

  /**
   * A consumer of a tree artifact declared by an {@link ActionTemplate} excludes the rewinding of
   * every expanded action populating it, including one whose only output in the tree is an empty
   * subdirectory and which thus doesn't generate any of the tree's files.
   */
  @Test
  public void expandedActionRewound_emptySubdirectoryProducer_waitsForTreeConsumer(
      @TestParameter boolean inflateFineLocksFirst) throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;
    TreeArtifact tree = newTreeArtifact(root, "tree", ActionLookupData.create(owner, 0));
    ActionTemplateExpansionKey expansion = ActionTemplateExpansionValue.key(owner, 0);
    TreeArtifact subdirectory =
        TreeArtifact.createSubTreeArtifact(tree, PathFragment.create("empty"), expansion);
    subdirectory.setGeneratingActionKey(ActionLookupData.create(expansion, 0));
    Action producer = newAction(ImmutableList.of(subdirectory), ImmutableList.of());
    DerivedArtifact consumerOutput =
        (DerivedArtifact) ActionsTestUtil.createArtifact(root, "consumer.out");
    consumerOutput.setGeneratingActionKey(ActionLookupData.create(owner, 1));
    Action consumer = newAction(ImmutableList.of(consumerOutput), ImmutableList.of(tree));

    InputMetadataProvider metadataProvider = mock(InputMetadataProvider.class);
    mockTemplateExpansion(owner, ImmutableList.of(producer));

    maybeInflateFineLocks(inflateFineLocksFirst);

    var preparation = new TestThread(() -> rewind(producer));
    try (SilentCloseable execution =
        synchronizer.enterActionExecution(consumer, /* wasRewound= */ false, metadataProvider)) {
      preparation.start();
      waitUntilState(preparation, Thread.State.WAITING);
    }
    preparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
  }

  /**
   * A consumer of one output tree of an {@link ActionTemplate} doesn't depend on expanded actions
   * that only populate another output tree of the same template and thus must not wait for their
   * rewinding, whereas it does exclude the rewinding of the actions populating its own tree.
   */
  @Test
  public void expandedActionRewound_treeConsumer_excludesOnlyProducersOfItsTree(
      @TestParameter boolean inflateFineLocksFirst) throws Exception {
    TwoTreeTemplate template = mockTwoTreeTemplate();
    InputMetadataProvider metadataProvider = mock(InputMetadataProvider.class);

    maybeInflateFineLocks(inflateFineLocksFirst);

    var firstPreparation = new TestThread(() -> rewind(template.firstProducer()));
    var secondPreparation = new TestThread(() -> rewind(template.secondProducer()));
    try (SilentCloseable execution =
        synchronizer.enterActionExecution(
            template.consumer(), /* wasRewound= */ false, metadataProvider)) {
      secondPreparation.start();
      secondPreparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
      firstPreparation.start();
      waitUntilState(firstPreparation, Thread.State.WAITING);
    }
    firstPreparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
  }

  /**
   * A consumer that arrives while the producer of another output tree of the same {@link
   * ActionTemplate} holds its write lock is admitted and still excludes the rewinding of the
   * producers of its own tree.
   */
  @Test
  public void expandedActionRewound_treeConsumer_admittedWhileProducerOfOtherTreeRewinds()
      throws Exception {
    TwoTreeTemplate template = mockTwoTreeTemplate();
    InputMetadataProvider metadataProvider = mock(InputMetadataProvider.class);

    try (SilentCloseable secondPreparation =
        synchronizer.enterActionPreparation(template.secondProducer(), /* wasRewound= */ true)) {
      var consumerExecution =
          new TestThread(
              () -> {
                try (SilentCloseable execution =
                    synchronizer.enterActionExecution(
                        template.consumer(), /* wasRewound= */ false, metadataProvider)) {}
              });
      consumerExecution.start();
      consumerExecution.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);

      var firstPreparation = new TestThread(() -> rewind(template.firstProducer()));
      try (SilentCloseable execution =
          synchronizer.enterActionExecution(
              template.consumer(), /* wasRewound= */ false, metadataProvider)) {
        firstPreparation.start();
        waitUntilState(firstPreparation, Thread.State.WAITING);
      }
      firstPreparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
    }
  }

  /**
   * Regression test for a lock cycle between the rewound expanded action P of an action template, a
   * rewound consumer C of the tree artifact P populates, and an action D expanded from a downstream
   * template that consumes both P's file and C's output:
   *
   * <ol>
   *   <li>D holds the read lock of P and waits for the read lock of C, which C holds for writing.
   *   <li>P waits for the write lock of its own key, which D holds for reading.
   *   <li>C, which depends on P but not on D, requests the read lock of P.
   * </ol>
   *
   * <p>C must be admitted even though P is waiting for the same lock as a writer. Any lock that
   * queues readers behind waiting writers would make C wait for P and close the cycle. {@link
   * java.util.concurrent.locks.ReentrantReadWriteLock} does so in both fairness modes and {@link
   * java.util.concurrent.locks.StampedLock} does so in {@code readLock}, but not in {@code
   * readLockInterruptibly}, which only waits for a writer that holds the lock.
   */
  @Test
  public void expandedActionRewound_treeConsumerNotQueuedBehindWaitingProducer() throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;

    // The upstream template (action 0 of the owner) declares a tree artifact that its single
    // expanded action P populates with a file.
    TreeArtifact upstreamTree =
        newTreeArtifact(root, "upstream", ActionLookupData.create(owner, 0));
    ActionTemplateExpansionKey upstreamExpansion = ActionTemplateExpansionValue.key(owner, 0);
    TreeFileArtifact upstreamFile =
        TreeFileArtifact.createTemplateExpansionOutput(upstreamTree, "file", upstreamExpansion);
    upstreamFile.setGeneratingActionKey(ActionLookupData.create(upstreamExpansion, 0));
    Action upstreamAction = newAction(ImmutableList.of(upstreamFile), ImmutableList.of());

    // The regular action C (action 1 of the owner) consumes the whole upstream tree artifact.
    DerivedArtifact treeConsumerOutput =
        (DerivedArtifact) ActionsTestUtil.createArtifact(root, "tree_consumer.out");
    treeConsumerOutput.setGeneratingActionKey(ActionLookupData.create(owner, 1));
    Action treeConsumer =
        newAction(ImmutableList.of(treeConsumerOutput), ImmutableList.of(upstreamTree));

    // The downstream template (action 2 of the owner) is expanded over the upstream tree artifact.
    // Its expanded action D consumes the file generated by P as well as the output of C and
    // acquires their read locks in this order.
    TreeArtifact downstreamTree =
        newTreeArtifact(root, "downstream", ActionLookupData.create(owner, 2));
    ActionTemplateExpansionKey downstreamExpansion = ActionTemplateExpansionValue.key(owner, 2);
    TreeFileArtifact downstreamFile =
        TreeFileArtifact.createTemplateExpansionOutput(downstreamTree, "file", downstreamExpansion);
    downstreamFile.setGeneratingActionKey(ActionLookupData.create(downstreamExpansion, 0));
    Action downstreamAction =
        newAction(
            ImmutableList.of(downstreamFile), ImmutableList.of(upstreamFile, treeConsumerOutput));

    InputMetadataProvider metadataProvider = mock(InputMetadataProvider.class);
    mockActions(
        owner,
        ImmutableList.of(mock(ActionTemplate.class), treeConsumer, mock(ActionTemplate.class)));
    mockTemplateExpansion(owner, /* actionIndex= */ 0, ImmutableList.of(upstreamAction));

    // C is rewound and prepares for its re-execution, which makes it hold the write lock guarding
    // its output until the end of its execution.
    SilentCloseable treeConsumerPreparation =
        synchronizer.enterActionPreparation(treeConsumer, /* wasRewound= */ true);
    // D enters execution, acquires the read lock guarding the upstream file and then blocks on the
    // read lock guarding the output of C.
    var downstreamExecution =
        new TestThread(
            () -> {
              try (SilentCloseable execution =
                  synchronizer.enterActionExecution(
                      downstreamAction, /* wasRewound= */ false, metadataProvider)) {}
            });
    downstreamExecution.start();
    waitUntilState(downstreamExecution, Thread.State.WAITING);
    // P is rewound and prepares for its re-execution, which blocks on the read lock held by D.
    var upstreamPreparation = new TestThread(() -> rewind(upstreamAction));
    upstreamPreparation.start();
    waitUntilState(upstreamPreparation, Thread.State.WAITING);
    // C enters execution, which requires the read lock guarding the upstream tree artifact. It must
    // not wait for P, which waits for D, which waits for C.
    var treeConsumerExecution =
        new TestThread(
            () -> {
              try (treeConsumerPreparation;
                  SilentCloseable execution =
                      synchronizer.enterActionExecution(
                          treeConsumer, /* wasRewound= */ true, metadataProvider)) {}
            });
    treeConsumerExecution.start();

    treeConsumerExecution.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
    downstreamExecution.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
    upstreamPreparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
  }

  /**
   * While a rewound action waits for a consumer of its outputs that started before the fine locks
   * were inflated, other consumers are admitted.
   */
  @Test
  public void rewoundActionAwaitingRegisteredConsumer_admitsOtherConsumers() throws Exception {
    var fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    var root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;
    var file = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "file");
    file.setGeneratingActionKey(ActionLookupData.create(owner, 0));
    var producer = newAction(ImmutableList.of(file), ImmutableList.of());
    var otherFile = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "other");
    otherFile.setGeneratingActionKey(ActionLookupData.create(owner, 1));
    var otherProducer = newAction(ImmutableList.of(otherFile), ImmutableList.of());
    var consumerOutput = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "consumer.out");
    consumerOutput.setGeneratingActionKey(ActionLookupData.create(owner, 2));
    var consumer = newAction(ImmutableList.of(consumerOutput), ImmutableList.of(file));
    var otherConsumerOutput =
        (DerivedArtifact) ActionsTestUtil.createArtifact(root, "other_consumer.out");
    otherConsumerOutput.setGeneratingActionKey(ActionLookupData.create(owner, 3));
    var otherConsumer =
        newAction(ImmutableList.of(otherConsumerOutput), ImmutableList.of(otherFile));
    mockActions(owner, ImmutableList.of(producer, otherProducer, consumer, otherConsumer));
    var metadataProvider = mock(InputMetadataProvider.class);

    var preparation = new TestThread(() -> rewind(producer));
    var otherExecution =
        new TestThread(
            () -> {
              try (var _ =
                  synchronizer.enterActionExecution(
                      otherConsumer, /* wasRewound= */ false, metadataProvider)) {}
            });
    try (var _ =
        synchronizer.enterActionExecution(consumer, /* wasRewound= */ false, metadataProvider)) {
      preparation.start();
      waitUntilState(preparation, Thread.State.WAITING);
      otherExecution.start();
      otherExecution.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
    }
    preparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
  }

  /**
   * A rewound action reads the metadata provider of a consumer that registered before the fine
   * locks were inflated to find out whether it consumes the rewound action's outputs. Since the
   * consumer may mutate the provider once unregistered, unregistration waits for the rewound
   * action.
   */
  @Test
  public void unregistration_waitsForRewoundActionReadingConsumerMetadata() throws Exception {
    var fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    var root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;
    var file = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "file");
    file.setGeneratingActionKey(ActionLookupData.create(owner, 0));
    var producer = newAction(ImmutableList.of(file), ImmutableList.of());
    var unrelatedFile = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "unrelated");
    unrelatedFile.setGeneratingActionKey(ActionLookupData.create(owner, 1));
    var unrelatedProducer = newAction(ImmutableList.of(unrelatedFile), ImmutableList.of());
    var runfiles = ActionsTestUtil.createRunfilesArtifact(root, "out/runfiles");
    runfiles.setGeneratingActionKey(ActionLookupData.create(owner, 2));
    var runfilesAction = newAction(ImmutableList.of(runfiles), ImmutableList.of(file));
    var consumerOutput = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "consumer.out");
    consumerOutput.setGeneratingActionKey(ActionLookupData.create(owner, 3));
    var consumer = newAction(ImmutableList.of(consumerOutput), ImmutableList.of(runfiles));
    mockActions(owner, ImmutableList.of(producer, unrelatedProducer, runfilesAction, consumer));
    var runfilesValue =
        new RunfilesArtifactValue(
            mock(RunfilesTree.class),
            ImmutableList.of(file),
            ImmutableList.of(FileArtifactValue.createForNormalFile(new byte[32], null, 0)),
            ImmutableList.of(),
            ImmutableList.of(),
            ImmutableList.of(),
            ImmutableList.of());
    var metadataRead = new CountDownLatch(1);
    var metadataReleased = new CountDownLatch(1);
    var metadataProvider = mock(InputMetadataProvider.class);
    when(metadataProvider.getRunfilesMetadata(runfiles))
        .thenAnswer(
            _ -> {
              metadataRead.countDown();
              metadataReleased.await();
              return runfilesValue;
            });

    var execution =
        synchronizer.enterActionExecution(consumer, /* wasRewound= */ false, metadataProvider);
    var unrelatedPreparation = new TestThread(() -> rewind(unrelatedProducer));
    unrelatedPreparation.start();
    var unregistration = new TestThread(execution::close);
    try {
      assertThat(metadataRead.await(DEADLOCK_TIMEOUT_MILLIS, MILLISECONDS)).isTrue();
      unregistration.start();
      waitUntilState(unregistration, Thread.State.BLOCKED);
    } finally {
      metadataReleased.countDown();
    }

    unregistration.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
    unrelatedPreparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
  }

  @Test
  public void enterActionExecution_interrupted_throws() throws Exception {
    Action action = newAction();
    InputMetadataProvider metadataProvider = mock(InputMetadataProvider.class);

    try {
      Thread.currentThread().interrupt();
      assertThrows(
          InterruptedException.class,
          () ->
              synchronizer.enterActionExecution(action, /* wasRewound= */ false, metadataProvider));
      assertThat(Thread.currentThread().isInterrupted()).isFalse();
    } finally {
      // Avoid leaking the interrupt into other tests if an assertion fails.
      Thread.interrupted();
    }
  }

  /**
   * A consumer interrupted while waiting for the read lock of one input releases the read locks it
   * has already acquired, so that the producers of its other inputs can still be rewound.
   */
  @Test
  public void consumerInterruptedWhileAwaitingReadLock_releasesAcquiredReadLocks()
      throws Exception {
    var fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    var root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;
    var first = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "first");
    first.setGeneratingActionKey(ActionLookupData.create(owner, 0));
    var firstProducer = newAction(ImmutableList.of(first), ImmutableList.of());
    var second = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "second");
    second.setGeneratingActionKey(ActionLookupData.create(owner, 1));
    var secondProducer = newAction(ImmutableList.of(second), ImmutableList.of());
    var consumerOutput = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "consumer.out");
    consumerOutput.setGeneratingActionKey(ActionLookupData.create(owner, 2));
    var consumer = newAction(ImmutableList.of(consumerOutput), ImmutableList.of(first, second));
    mockActions(owner, ImmutableList.of(firstProducer, secondProducer, consumer));

    // The consumer acquires the read lock of the first input and then blocks on the second one.
    var execution =
        new TestThread(
            () ->
                assertThrows(
                    InterruptedException.class,
                    () ->
                        synchronizer.enterActionExecution(
                            consumer, /* wasRewound= */ false, mock(InputMetadataProvider.class))));
    try (var _ = synchronizer.enterActionPreparation(secondProducer, /* wasRewound= */ true)) {
      execution.start();
      waitUntilState(execution, Thread.State.WAITING);
      execution.interrupt();
      execution.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
    }

    var firstPreparation = new TestThread(() -> rewind(firstProducer));
    firstPreparation.start();
    firstPreparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
  }

  /**
   * Scanning a registered consumer flattens its inputs, which isn't interruptible and thus must not
   * start once the rewound action has been interrupted.
   */
  @Test
  public void rewoundActionInterruptedBeforeScanningRegisteredConsumer_doesNotScanIt()
      throws Exception {
    var fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    var root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;
    var file = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "file");
    file.setGeneratingActionKey(ActionLookupData.create(owner, 0));
    var producer = newAction(ImmutableList.of(file), ImmutableList.of());
    var consumerOutput = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "consumer.out");
    consumerOutput.setGeneratingActionKey(ActionLookupData.create(owner, 1));
    var consumer = newAction(ImmutableList.of(consumerOutput), ImmutableList.of(file));
    mockActions(owner, ImmutableList.of(producer, consumer));

    try (var _ =
        synchronizer.enterActionExecution(
            consumer, /* wasRewound= */ false, mock(InputMetadataProvider.class))) {
      try {
        Thread.currentThread().interrupt();
        assertThrows(
            InterruptedException.class,
            () -> synchronizer.enterActionPreparation(producer, /* wasRewound= */ true));
        assertThat(Thread.currentThread().isInterrupted()).isFalse();
      } finally {
        // Avoid leaking the interrupt into other tests if an assertion fails.
        Thread.interrupted();
      }
      verify(graph, never()).getValue(any());
      verify(actionInputFetcher, never()).handleRewoundActionOutputs(any());
    }

    rewind(producer);
    verify(actionInputFetcher).handleRewoundActionOutputs(any());
  }

  @Test
  public void rewoundActionInterruptedWhileAwaitingRegisteredConsumer_canBeRewoundAgain()
      throws Exception {
    var fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    var root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;
    var file = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "file");
    file.setGeneratingActionKey(ActionLookupData.create(owner, 0));
    var producer = newAction(ImmutableList.of(file), ImmutableList.of());
    var consumerOutput = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "consumer.out");
    consumerOutput.setGeneratingActionKey(ActionLookupData.create(owner, 1));
    var consumer = newAction(ImmutableList.of(consumerOutput), ImmutableList.of(file));
    mockActions(owner, ImmutableList.of(producer, consumer));

    var preparation =
        new TestThread(() -> assertThrows(InterruptedException.class, () -> rewind(producer)));
    try (var _ =
        synchronizer.enterActionExecution(
            consumer, /* wasRewound= */ false, mock(InputMetadataProvider.class))) {
      preparation.start();
      waitUntilState(preparation, Thread.State.WAITING);
      preparation.interrupt();
      preparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
    }

    rewind(producer);
  }

  /**
   * A consumer that races with the first rewound action either finds the fine locks inflated or is
   * found registered by the rewound action, so it never executes while the rewound action holds its
   * write lock. The race is only hit with some probability per round.
   */
  @Test
  public void consumersRacingWithFirstRewinding_neverOverlapRewoundAction() throws Exception {
    var fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    var root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;
    var file = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "file");
    file.setGeneratingActionKey(ActionLookupData.create(owner, 0));
    var consumerOutput = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "consumer.out");
    consumerOutput.setGeneratingActionKey(ActionLookupData.create(owner, 1));
    var producer = new ActionsTestUtil.MockAction(ImmutableList.of(), ImmutableSet.of(file));
    var consumer =
        new ActionsTestUtil.MockAction(ImmutableList.of(file), ImmutableSet.of(consumerOutput));
    // These are used in every round, so their invocations shouldn't be recorded.
    var graph = mock(WalkableGraph.class, withSettings().stubOnly());
    when(graph.getValue(owner))
        .thenReturn(new BasicActionLookupValue(ImmutableList.of(producer, consumer)));
    var actionInputFetcher = mock(RemoteActionInputFetcher.class, withSettings().stubOnly());
    var metadataProvider = mock(InputMetadataProvider.class);

    int rounds = 3000;
    int consumerCount = 4;
    var rewindingThread = Thread.currentThread();
    var barrier = new CyclicBarrier(consumerCount + 1);
    var currentRound = new AtomicReference<RacingRound>();
    var overlapped = new AtomicBoolean();
    // Set by the first thread that fails, which makes the other threads stop.
    var aborted = new AtomicBoolean();
    var consumerExecutions = new ArrayList<TestThread>();
    for (int i = 0; i < consumerCount; i++) {
      int consumerIndex = i;
      var consumerExecution =
          new TestThread(
              () -> {
                try {
                  for (int j = 0; j < rounds; j++) {
                    barrier.await(DEADLOCK_TIMEOUT_MILLIS, MILLISECONDS);
                    var round = currentRound.get();
                    round.awaitParticipants();
                    round.delayArrival();
                    try (var _ =
                        round.synchronizer.enterActionExecution(
                            consumer, /* wasRewound= */ false, metadataProvider)) {
                      // Stay until the rewound action either waits, possibly for this consumer,
                      // holds its write lock or has released it.
                      while (!round.rewinding.get()
                          && !round.rewound.get()
                          && rewindingThread.getState() != Thread.State.WAITING) {
                        round.yieldUnlessAborted();
                      }
                      if (round.rewinding.get()) {
                        overlapped.set(true);
                      }
                    } finally {
                      round.finishedConsumers.set(consumerIndex, 1);
                    }
                  }
                } catch (Throwable t) {
                  // Only the first failure is reported, the others are caused by the abort.
                  if (!aborted.getAndSet(true)) {
                    barrier.reset();
                    throw t;
                  }
                }
              });
      consumerExecution.start();
      consumerExecutions.add(consumerExecution);
    }

    try {
      for (int i = 0; i < rounds; i++) {
        var round =
            new RacingRound(
                new RemoteRewoundActionSynchronizer(actionInputFetcher, graph),
                consumerCount,
                aborted);
        currentRound.set(round);
        barrier.await(DEADLOCK_TIMEOUT_MILLIS, MILLISECONDS);
        round.awaitParticipants();
        try (var _ = round.synchronizer.enterActionPreparation(producer, /* wasRewound= */ true)) {
          round.rewinding.set(true);
          // Keep the write lock until every consumer has either finished or blocked on it, so that
          // a consumer that wrongly executes concurrently observes the flag. Only waiting for a
          // lock makes a consumer WAITING here, whereas contending for a monitor makes it BLOCKED.
          for (int j = 0; j < consumerCount; j++) {
            while (round.finishedConsumers.get(j) == 0
                && consumerExecutions.get(j).getState() != Thread.State.WAITING) {
              round.yieldUnlessAborted();
            }
          }
          round.rewinding.set(false);
        }
        round.rewound.set(true);
      }
    } catch (Throwable t) {
      boolean firstFailure = !aborted.getAndSet(true);
      consumerExecutions.forEach(Thread::interrupt);
      if (firstFailure) {
        for (TestThread consumerExecution : consumerExecutions) {
          try {
            consumerExecution.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
          } catch (Throwable e) {
            t.addSuppressed(e);
          }
        }
        throw t;
      }
    }
    // If a consumer failed, joining it reports its failure.
    for (TestThread consumerExecution : consumerExecutions) {
      consumerExecution.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
    }
    assertThat(overlapped.get()).isFalse();
  }

  /**
   * The state shared by the threads of one round of {@link
   * #consumersRacingWithFirstRewinding_neverOverlapRewoundAction}. It isn't reused across rounds,
   * so that a thread leaving a round can't observe the flags of the next one.
   */
  private static final class RacingRound {
    // Staggering the consumers hits the race more often than having them all arrive at once, since
    // they have to read the fine locks right as the rewinding thread inflates them.
    private static final long MAX_ARRIVAL_DELAY_NANOS = 2000;

    final RemoteRewoundActionSynchronizer synchronizer;
    final AtomicBoolean rewinding = new AtomicBoolean();
    final AtomicBoolean rewound = new AtomicBoolean();
    final AtomicIntegerArray finishedConsumers;
    private final AtomicInteger arrivals = new AtomicInteger();
    // Shared by all rounds.
    private final AtomicBoolean aborted;

    RacingRound(
        RemoteRewoundActionSynchronizer synchronizer, int consumerCount, AtomicBoolean aborted) {
      this.synchronizer = synchronizer;
      this.finishedConsumers = new AtomicIntegerArray(consumerCount);
      this.aborted = aborted;
    }

    /**
     * Waits for the consumers and the rewinding thread to arrive. Threads leave a {@link
     * CyclicBarrier} one after another as they are woken up, which takes long enough for the last
     * thread to arrive to run ahead of the others and thus rarely hits the race. This also ensures
     * that every thread has left the barrier before any thread inspects the state of another.
     */
    void awaitParticipants() {
      arrivals.incrementAndGet();
      while (arrivals.get() <= finishedConsumers.length()) {
        yieldUnlessAborted();
      }
    }

    /** Called in a spin loop, which a failure in another thread would otherwise never end. */
    @SuppressWarnings("ThreadPriorityCheck")
    void yieldUnlessAborted() {
      if (aborted.get()) {
        throw new IllegalStateException("Aborted after another thread failed");
      }
      Thread.yield();
    }

    void delayArrival() {
      long deadline =
          System.nanoTime() + ThreadLocalRandom.current().nextLong(MAX_ARRIVAL_DELAY_NANOS);
      while (System.nanoTime() < deadline) {
        Thread.onSpinWait();
      }
    }
  }

  /** An {@link ActionTemplate} with two output trees and a consumer of the first one. */
  private record TwoTreeTemplate(Action firstProducer, Action secondProducer, Action consumer) {}

  private TwoTreeTemplate mockTwoTreeTemplate() throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;
    var templateKey = ActionLookupData.create(owner, 0);
    ActionTemplateExpansionKey expansion = ActionTemplateExpansionValue.key(owner, 0);
    TreeArtifact firstTree = newTreeArtifact(root, "first", templateKey);
    TreeArtifact secondTree = newTreeArtifact(root, "second", templateKey);
    TreeFileArtifact firstFile =
        TreeFileArtifact.createTemplateExpansionOutput(firstTree, "file", expansion);
    firstFile.setGeneratingActionKey(ActionLookupData.create(expansion, 0));
    TreeFileArtifact secondFile =
        TreeFileArtifact.createTemplateExpansionOutput(secondTree, "file", expansion);
    secondFile.setGeneratingActionKey(ActionLookupData.create(expansion, 1));
    Action firstProducer = newAction(ImmutableList.of(firstFile), ImmutableList.of());
    Action secondProducer = newAction(ImmutableList.of(secondFile), ImmutableList.of());
    DerivedArtifact consumerOutput =
        (DerivedArtifact) ActionsTestUtil.createArtifact(root, "consumer.out");
    consumerOutput.setGeneratingActionKey(ActionLookupData.create(owner, 1));
    Action consumer = newAction(ImmutableList.of(consumerOutput), ImmutableList.of(firstTree));
    mockTemplateExpansion(owner, ImmutableList.of(firstProducer, secondProducer));
    return new TwoTreeTemplate(firstProducer, secondProducer, consumer);
  }

  /**
   * Makes the given owner's only action an {@link ActionTemplate} whose expansion consists of the
   * given actions.
   */
  private void mockTemplateExpansion(ActionLookupKey owner, ImmutableList<Action> expandedActions)
      throws InterruptedException {
    mockActions(owner, ImmutableList.of(mock(ActionTemplate.class)));
    mockTemplateExpansion(owner, /* actionIndex= */ 0, expandedActions);
  }

  /** Makes the given actions the expansion of the given owner's template at the given index. */
  private void mockTemplateExpansion(
      ActionLookupKey owner, int actionIndex, ImmutableList<Action> expandedActions)
      throws InterruptedException {
    when(graph.getValue(ActionTemplateExpansionValue.key(owner, actionIndex)))
        .thenReturn(new ActionTemplateExpansionValue(ImmutableList.copyOf(expandedActions)));
  }

  /** Makes the given actions, in order, the actions of the given owner. */
  private void mockActions(
      ActionLookupKey owner, ImmutableList<? extends ActionAnalysisMetadata> actions)
      throws InterruptedException {
    ActionLookupValue ownerValue = mock(ActionLookupValue.class);
    when(ownerValue.getActions()).thenReturn(ImmutableList.<ActionAnalysisMetadata>copyOf(actions));
    when(graph.getValue(owner)).thenReturn(ownerValue);
  }

  /**
   * Returns a metadata provider that, just like the one {@code CompletionFunction} passes to the
   * {@link RemoteImportantOutputHandler}, provides a runfiles tree containing the given runfiles
   * and remote metadata for the given artifacts.
   */
  private static InputMetadataProvider mockTopLevelMetadataProvider(
      ImmutableList<Artifact> runfiles, ImmutableList<Artifact> remoteArtifacts) throws Exception {
    RunfilesTree runfilesTree = mock(RunfilesTree.class);
    when(runfilesTree.getArtifacts())
        .thenReturn(NestedSetBuilder.wrap(Order.STABLE_ORDER, runfiles));
    InputMetadataProvider metadataProvider = mock(InputMetadataProvider.class);
    when(metadataProvider.getRunfilesTrees()).thenReturn(ImmutableList.of(runfilesTree));
    FileArtifactValue remoteMetadata = FileArtifactValue.createForRemoteFile(new byte[32], 1, 1);
    for (Artifact artifact : remoteArtifacts) {
      when(metadataProvider.getInputMetadata(artifact)).thenReturn(remoteMetadata);
    }
    return metadataProvider;
  }

  /**
   * Returns a handler for a build that downloads the given top-level outputs, which are registered
   * with the output checker just like the outputs and runfiles of top-level targets are after their
   * analysis.
   */
  private RemoteImportantOutputHandler newImportantOutputHandler(Artifact... topLevelOutputs) {
    var remoteOutputChecker =
        new RemoteOutputChecker("build", RemoteOutputsMode.TOPLEVEL, ImmutableList.of());
    for (Artifact output : topLevelOutputs) {
      remoteOutputChecker.addOutputToDownload(output);
    }
    return new RemoteImportantOutputHandler(
        graph, remoteOutputChecker, actionInputFetcher, synchronizer);
  }

  @SuppressWarnings("ThreadPriorityCheck")
  private static void waitUntilState(Thread thread, Thread.State expectedState) {
    long deadline = System.nanoTime() + MILLISECONDS.toNanos(DEADLOCK_TIMEOUT_MILLIS);
    Thread.State state;
    while ((state = thread.getState()) != expectedState) {
      assertThat(state).isNotEqualTo(Thread.State.TERMINATED);
      if (System.nanoTime() - deadline > 0) {
        throw new AssertionError(
            "%s is still %s instead of %s".formatted(thread.getName(), state, expectedState));
      }
      Thread.yield();
    }
  }

  /**
   * Inflates the fine locks by rewinding an unrelated action if requested. Otherwise, consumers
   * entered before the first rewinding only register themselves.
   */
  private void maybeInflateFineLocks(boolean inflate) throws Exception {
    if (inflate) {
      rewind(newAction());
    }
  }

  private void rewind(Action action) throws Exception {
    try (SilentCloseable ignored = synchronizer.enterActionPreparation(action, true)) {
      // Cancellation happens while entering preparation.
    }
  }

  private static Action newAction() {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    DerivedArtifact output = (DerivedArtifact) ActionsTestUtil.createArtifact(outputRoot, "output");
    output.setGeneratingActionKey(ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);
    return newAction(ImmutableList.of(output), ImmutableList.of());
  }

  private static Action newAction(
      ImmutableList<? extends Artifact> outputs, ImmutableList<? extends Artifact> inputs) {
    return newAction(outputs, inputs, /* schedulingDependencies= */ ImmutableList.of());
  }

  private static Action newAction(
      ImmutableList<? extends Artifact> outputs,
      ImmutableList<? extends Artifact> inputs,
      ImmutableList<? extends Artifact> schedulingDependencies) {
    Action action = mock(Action.class);
    when(action.getPrimaryOutput()).thenReturn(outputs.get(0));
    when(action.getOutputs()).thenReturn(ImmutableList.copyOf(outputs));
    when(action.getInputs()).thenReturn(NestedSetBuilder.wrap(Order.STABLE_ORDER, inputs));
    when(action.getSchedulingDependencies())
        .thenReturn(NestedSetBuilder.wrap(Order.STABLE_ORDER, schedulingDependencies));
    return action;
  }

  private static TreeArtifact newTreeArtifact(
      ArtifactRoot root, String name, ActionLookupData generatingActionKey) {
    TreeArtifact tree =
        TreeArtifact.create(
            root, root.getExecPath().getRelative(name), generatingActionKey.getActionLookupKey());
    tree.setGeneratingActionKey(generatingActionKey);
    return tree;
  }

  private static final class EqualCancellable
      implements RemoteRewoundActionSynchronizer.Cancellable {
    private final AtomicInteger cancellations = new AtomicInteger();

    @Override
    public void requestCancellation() {
      cancellations.incrementAndGet();
    }

    @Override
    public void awaitCompletion() {}

    @Override
    public boolean equals(Object obj) {
      return obj instanceof EqualCancellable;
    }

    @Override
    public int hashCode() {
      return 1;
    }
  }
}
