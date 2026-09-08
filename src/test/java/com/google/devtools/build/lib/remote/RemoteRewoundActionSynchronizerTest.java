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
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionTemplate;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.RunfilesArtifactValue;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue.ActionTemplateExpansionKey;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.InOrder;

/** Tests for {@link RemoteRewoundActionSynchronizer}. */
@RunWith(JUnit4.class)
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

  @Test
  public void outputProcessing_runfilesTree_locksOnlyItsProducers() throws Exception {
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
    SpecialArtifact runfiles = ActionsTestUtil.createRunfilesArtifact(root, "out/runfiles");
    runfiles.setGeneratingActionKey(ActionLookupData.create(owner, 2));
    Action runfilesAction = newAction(ImmutableList.of(runfiles), ImmutableList.of(file));

    RunfilesTree runfilesTree = mock(RunfilesTree.class);
    when(runfilesTree.getMapping()).thenReturn(ImmutableSortedMap.of(file.getExecPath(), file));
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

    // Switch to fine locks before processing just one of the provider's runfiles trees.
    rewind(newAction());
    var preparation = new TestThread(() -> rewind(producer));
    var unrelatedPreparation = new TestThread(() -> rewind(unrelatedProducer));
    var runfilesPreparation = new TestThread(() -> rewind(runfilesAction));
    try (SilentCloseable processing =
        synchronizer.enterProcessOutputsAndGetLostArtifacts(
            ImmutableList.of(runfiles), metadataProvider)) {
      preparation.start();
      waitUntilBlocked(preparation);
      unrelatedPreparation.start();
      unrelatedPreparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
      runfilesPreparation.start();
      runfilesPreparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
    }
    preparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
  }

  /**
   * Regression test for a deadlock between the rewound expanded action of an action template, a
   * rewound consumer of the tree artifact it populates and an action expanded from a downstream
   * template that consumes both an individual file of that tree artifact and the output of the tree
   * consumer.
   */
  @Test
  public void expandedActionRewound_consumerFromOtherExpansion_doesNotDeadlock() throws Exception {
    runExpandedActionRewound(ConsumerKind.EXPANDED_ACTION);
  }

  @Test
  public void expandedActionRewound_ordinaryConsumerOfTreeFile_doesNotDeadlock() throws Exception {
    runExpandedActionRewound(ConsumerKind.ORDINARY_ACTION);
  }

  @Test
  public void expandedActionRewound_outputProcessing_doesNotDeadlock() throws Exception {
    runExpandedActionRewound(ConsumerKind.OUTPUT_PROCESSING);
  }

  /**
   * Verifies that a consumer of a tree artifact excludes the rewinding of an expanded action whose
   * only output in that tree is an empty subdirectory.
   */
  @Test
  public void expandedActionRewound_emptySubdirectoryProducer_waitsForTreeConsumer()
      throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;
    SpecialArtifact tree = newTreeArtifact(root, "tree", ActionLookupData.create(owner, 0));
    ActionTemplateExpansionKey expansion = ActionTemplateExpansionValue.key(owner, 0);
    SpecialArtifact subdirectory =
        SpecialArtifact.createSubTreeArtifact(tree, PathFragment.create("empty"), expansion);
    subdirectory.setGeneratingActionKey(ActionLookupData.create(expansion, 0));
    Action producer = newAction(ImmutableList.of(subdirectory), ImmutableList.of());
    DerivedArtifact consumerOutput =
        (DerivedArtifact) ActionsTestUtil.createArtifact(root, "consumer.out");
    consumerOutput.setGeneratingActionKey(ActionLookupData.create(owner, 1));
    Action consumer = newAction(ImmutableList.of(consumerOutput), ImmutableList.of(tree));

    InputMetadataProvider metadataProvider = mock(InputMetadataProvider.class);
    ActionLookupValue ownerValue = mock(ActionLookupValue.class);
    when(ownerValue.getActions()).thenReturn(ImmutableList.of(mock(ActionTemplate.class)));
    when(graph.getValue(owner)).thenReturn(ownerValue);
    var expansionValue = new ActionTemplateExpansionValue(ImmutableList.of(producer));
    when(graph.getValue(expansion)).thenReturn(expansionValue);

    // Switch to the fine locks with an unrelated rewound action first: the coarse lock would
    // exclude the producer regardless of which keys guard the tree.
    rewind(newAction());

    var preparation = new TestThread(() -> rewind(producer));
    try (SilentCloseable execution =
        synchronizer.enterActionExecution(consumer, /* wasRewound= */ false, metadataProvider)) {
      preparation.start();
      waitUntilBlocked(preparation);
    }
    preparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
  }

  /** Regression test for the lock cycle caused by sharing a key between output trees. */
  @Test
  public void expandedActionsInDifferentTrees_doNotDeadlockWithTreeConsumer() throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;
    var templateKey = ActionLookupData.create(owner, 0);
    var expansion = ActionTemplateExpansionValue.key(owner, 0);
    SpecialArtifact firstTree = newTreeArtifact(root, "first", templateKey);
    SpecialArtifact secondTree = newTreeArtifact(root, "second", templateKey);
    TreeFileArtifact firstFile =
        TreeFileArtifact.createTemplateExpansionOutput(firstTree, "file", expansion);
    firstFile.setGeneratingActionKey(ActionLookupData.create(expansion, 0));
    TreeFileArtifact secondFile =
        TreeFileArtifact.createTemplateExpansionOutput(secondTree, "file", expansion);
    secondFile.setGeneratingActionKey(ActionLookupData.create(expansion, 1));
    DerivedArtifact middle = (DerivedArtifact) ActionsTestUtil.createArtifact(root, "middle");
    middle.setGeneratingActionKey(ActionLookupData.create(owner, 1));
    Action firstProducer = newAction(ImmutableList.of(firstFile), ImmutableList.of());
    Action consumer = newAction(ImmutableList.of(middle), ImmutableList.of(firstTree));
    Action secondProducer = newAction(ImmutableList.of(secondFile), ImmutableList.of(middle));

    ActionLookupValue ownerValue = mock(ActionLookupValue.class);
    when(ownerValue.getActions()).thenReturn(ImmutableList.of(mock(ActionTemplate.class)));
    when(graph.getValue(owner)).thenReturn(ownerValue);
    when(graph.getValue(expansion))
        .thenReturn(
            new ActionTemplateExpansionValue(ImmutableList.of(firstProducer, secondProducer)));
    InputMetadataProvider metadataProvider = mock(InputMetadataProvider.class);

    SilentCloseable consumerPreparation =
        synchronizer.enterActionPreparation(consumer, /* wasRewound= */ true);
    SilentCloseable secondPreparation =
        synchronizer.enterActionPreparation(secondProducer, /* wasRewound= */ true);
    var consumerExecution =
        new TestThread(
            () -> {
              try (consumerPreparation;
                  SilentCloseable unused =
                      synchronizer.enterActionExecution(consumer, false, metadataProvider)) {}
            });
    var secondExecution =
        new TestThread(
            () -> {
              try (secondPreparation;
                  SilentCloseable unused =
                      synchronizer.enterActionExecution(secondProducer, false, metadataProvider)) {}
    });
    consumerExecution.start();
    // The old implementation blocks here on the shared template key; the fixed implementation
    // can complete immediately because the consumer only reads the first producer's key.
    Thread.sleep(100);
    secondExecution.start();
    consumerExecution.join(DEADLOCK_TIMEOUT_MILLIS);
    secondExecution.join(DEADLOCK_TIMEOUT_MILLIS);
    assertWithMessage("acyclic actions must not form a lock cycle")
        .that(consumerExecution.isAlive() && secondExecution.isAlive())
        .isFalse();
    consumerExecution.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
    secondExecution.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
  }

  private enum ConsumerKind {
    EXPANDED_ACTION,
    ORDINARY_ACTION,
    OUTPUT_PROCESSING
  }

  private void runExpandedActionRewound(ConsumerKind consumerKind) throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out");
    var owner = ActionsTestUtil.NULL_ARTIFACT_OWNER;

    // The upstream template (action 0 of the owner) declares a tree artifact that its single
    // expanded action populates with a file.
    SpecialArtifact upstreamTree =
        newTreeArtifact(root, "upstream", ActionLookupData.create(owner, 0));
    ActionTemplateExpansionKey upstreamExpansion = ActionTemplateExpansionValue.key(owner, 0);
    TreeFileArtifact upstreamFile =
        TreeFileArtifact.createTemplateExpansionOutput(upstreamTree, "file", upstreamExpansion);
    upstreamFile.setGeneratingActionKey(ActionLookupData.create(upstreamExpansion, 0));
    Action upstreamAction = newAction(ImmutableList.of(upstreamFile), ImmutableList.of());

    // A regular action (action 1 of the owner) consumes the whole upstream tree artifact.
    DerivedArtifact treeConsumerOutput =
        (DerivedArtifact) ActionsTestUtil.createArtifact(root, "tree_consumer.out");
    treeConsumerOutput.setGeneratingActionKey(ActionLookupData.create(owner, 1));
    Action treeConsumer =
        newAction(ImmutableList.of(treeConsumerOutput), ImmutableList.of(upstreamTree));

    // The downstream template (action 2 of the owner) is expanded over the upstream tree artifact.
    // Its expanded action consumes the file generated by the upstream action as well as the output
    // of the tree consumer and acquires their read locks in this order.
    SpecialArtifact downstreamTree =
        newTreeArtifact(root, "downstream", ActionLookupData.create(owner, 2));
    ActionTemplateExpansionKey downstreamExpansion = ActionTemplateExpansionValue.key(owner, 2);
    TreeFileArtifact downstreamFile =
        TreeFileArtifact.createTemplateExpansionOutput(downstreamTree, "file", downstreamExpansion);
    downstreamFile.setGeneratingActionKey(ActionLookupData.create(downstreamExpansion, 0));
    DerivedArtifact ordinaryOutput =
        (DerivedArtifact) ActionsTestUtil.createArtifact(root, "ordinary_consumer.out");
    ordinaryOutput.setGeneratingActionKey(ActionLookupData.create(owner, 3));
    ImmutableList<Artifact> downstreamInputs = ImmutableList.of(upstreamFile, treeConsumerOutput);
    Action downstreamAction =
        newAction(
            ImmutableList.of(
                consumerKind == ConsumerKind.EXPANDED_ACTION ? downstreamFile : ordinaryOutput),
            downstreamInputs);

    InputMetadataProvider metadataProvider = mock(InputMetadataProvider.class);
    ActionLookupValue ownerValue = mock(ActionLookupValue.class);
    when(ownerValue.getActions()).thenReturn(ImmutableList.of(mock(ActionTemplate.class)));
    when(graph.getValue(owner)).thenReturn(ownerValue);
    var expansionValue = new ActionTemplateExpansionValue(ImmutableList.of(upstreamAction));
    when(graph.getValue(upstreamExpansion)).thenReturn(expansionValue);

    // The tree consumer is rewound and prepares for its re-execution, which makes it hold the
    // write lock guarding its output until the end of its execution.
    SilentCloseable treeConsumerPreparation =
        synchronizer.enterActionPreparation(treeConsumer, /* wasRewound= */ true);
    // The downstream action enters execution, acquires the read lock guarding the upstream file
    // and then blocks on the read lock guarding the output of the tree consumer.
    var downstreamExecution =
        new TestThread(
            () -> {
              try (SilentCloseable unused =
                  consumerKind == ConsumerKind.OUTPUT_PROCESSING
                      ? synchronizer.enterProcessOutputsAndGetLostArtifacts(
                          downstreamInputs, metadataProvider)
                      : synchronizer.enterActionExecution(
                          downstreamAction, /* wasRewound= */ false, metadataProvider)) {}
            });
    downstreamExecution.start();
    waitUntilBlocked(downstreamExecution);
    // The upstream action is rewound and prepares for its re-execution, which blocks on a read
    // lock held by the downstream action.
    var upstreamPreparation =
        new TestThread(
            () -> {
              try (SilentCloseable unused =
                  synchronizer.enterActionPreparation(upstreamAction, /* wasRewound= */ true)) {}
            });
    upstreamPreparation.start();
    waitUntilBlocked(upstreamPreparation);
    // The tree consumer enters execution, which requires the read lock guarding the upstream tree
    // artifact. It must not wait for the upstream action, which waits for the downstream action,
    // which waits for the tree consumer.
    var treeConsumerExecution =
        new TestThread(
            () -> {
              try (SilentCloseable unused =
                  synchronizer.enterActionExecution(
                      treeConsumer, /* wasRewound= */ true, metadataProvider)) {
              } finally {
                treeConsumerPreparation.close();
              }
            });
    treeConsumerExecution.start();

    treeConsumerExecution.join(DEADLOCK_TIMEOUT_MILLIS);
    assertWithMessage(
            "deadlock: the tree consumer waits for the upstream action, which waits for the"
                + " downstream action, which waits for the tree consumer")
        .that(treeConsumerExecution.isAlive())
        .isFalse();
    treeConsumerExecution.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
    downstreamExecution.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
    upstreamPreparation.joinAndAssertState(DEADLOCK_TIMEOUT_MILLIS);
  }

  private static void waitUntilBlocked(Thread thread) {
    Thread.State state;
    while ((state = thread.getState()) != Thread.State.WAITING) {
      assertThat(state).isNotEqualTo(Thread.State.TERMINATED);
      Thread.yield();
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
    Action action = mock(Action.class);
    when(action.getPrimaryOutput()).thenReturn(outputs.get(0));
    when(action.getOutputs()).thenReturn(ImmutableList.copyOf(outputs));
    when(action.getInputs()).thenReturn(NestedSetBuilder.wrap(Order.STABLE_ORDER, inputs));
    return action;
  }

  private static SpecialArtifact newTreeArtifact(
      ArtifactRoot root, String name, ActionLookupData generatingActionKey) {
    SpecialArtifact tree =
        SpecialArtifact.create(
            root,
            root.getExecPath().getRelative(name),
            generatingActionKey.getActionLookupKey(),
            SpecialArtifactType.TREE);
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
