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
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue.ActionTemplateExpansionKey;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
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
  private RemoteRewoundActionSynchronizer synchronizer;

  @Before
  public void setUp() {
    actionInputFetcher = mock(RemoteActionInputFetcher.class);
    synchronizer = new RemoteRewoundActionSynchronizer(actionInputFetcher);
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

  /**
   * Regression test for a deadlock between the rewound expanded action of an action template, a
   * rewound consumer of the tree artifact it populates and an action expanded from a downstream
   * template that consumes both an individual file of that tree artifact and the output of the tree
   * consumer.
   */
  @Test
  public void expandedActionRewound_consumerFromOtherExpansion_doesNotDeadlock() throws Exception {
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
    Action downstreamAction =
        newAction(
            ImmutableList.of(downstreamFile), ImmutableList.of(upstreamFile, treeConsumerOutput));

    InputMetadataProvider metadataProvider = mock(InputMetadataProvider.class);
    when(metadataProvider.getRunfilesTrees()).thenReturn(ImmutableList.of());

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
                  synchronizer.enterActionExecution(
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
