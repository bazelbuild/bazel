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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.vfs.OutputService.RewoundActionSynchronizer;
import com.google.devtools.build.lib.vfs.RewindingSynchronizer;
import com.google.devtools.build.lib.vfs.RewindingSynchronizer.TransferableWriteLock;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A {@link RewoundActionSynchronizer} implementation for Bazel's remote filesystem, which is backed
 * by actual files on disk and requires synchronization to ensure that action outputs aren't deleted
 * while they are being read.
 */
final class RemoteRewoundActionSynchronizer implements RewoundActionSynchronizer {
  /** A task with a cancellation callback. */
  public interface Cancellable {
    void cancel() throws InterruptedException;
  }

  private final RemoteActionInputFetcher actionInputFetcher;
  // Rewound actions are producers that replace their outputs in place, so they are synchronized
  // with the actions reading those outputs by the same locks that synchronize repository fetches
  // with the actions reading repository contents. A rewound action takes the write lock of the key
  // guarding its outputs before it prepares for execution, while any action takes read locks for
  // the keys of the generating actions of its inputs before it starts executing.
  private final RewindingSynchronizer rewindingSynchronizer;
  private final ConcurrentHashMap<ActionExecutionMetadata, Cancellable> outputUploadTasks =
      new ConcurrentHashMap<>();

  public RemoteRewoundActionSynchronizer(
      RemoteActionInputFetcher actionInputFetcher, RewindingSynchronizer rewindingSynchronizer) {
    this.actionInputFetcher = actionInputFetcher;
    this.rewindingSynchronizer = rewindingSynchronizer;
  }

  /*
  Proof of deadlock freedom:

  As long as the coarse lock is used, there can't be any deadlock because there is only a single
  read-write lock, whose write lock is only acquired by a producer that holds no other lock.

  Now assume that there is a deadlock while the per-key locks are used. First, note that the logic
  in ImportantOutputHandler that is guarded by enterProcessOutputsAndGetLostArtifacts does not block
  on any (rewound or non-rewound) action executions while it holds read locks and can thus be
  ignored in the following. Consider the directed labeled "wait-for" graph defined as follows:

  * Nodes are the currently active producers: action executions, each identified with the
    ActionLookupData of the action it is (or will be) executing, and repository fetches, each
    identified with the RepositoryName of the repo it fetches.
  * For each pair of producers P_1 and P_2, there is an edge from P_1 to P_2 labeled with XY(P_3)
    if P_1 is waiting for the X lock of P_3 and P_2 currently holds the Y lock of P_3, where X and
    Y are either R (for read) or W (for write). The resulting graph may have parallel edges with
    distinct labels.

  Let C be any directed cycle in the graph representing a deadlock, let P_1 -[XY(P_3)]-> P_2 be an
  edge in C and consider the following cases for the pair XY:

  * RR: Since a read-write lock whose read lock is held by at least one thread doesn't
        block any other thread from acquiring its read lock, this case doesn't occur.
  * WW: The write lock of P_3 is only ever (attempted to be) acquired by P_3 itself when it
        replaces its outputs, which means that the edge would necessarily be of the shape
        P_3 -[WW(P_3)]-> P_3. But this isn't possible since a producer acquires its write lock in
        exactly one place (enterActionPreparationForRewinding for an action, fetch for a repo) and
        not recursively.
  * WR: In this case, P_1 attempts to acquire a write lock, which only happens when P_1 is a
        rewound action about to prepare for its (re-)execution or a repo fetch about to replace its
        repo root. This means that the edge is necessarily of the shape P_1 -[WR(P_1)]-> P_2. A
        producer doesn't hold any lock while waiting for its own write lock: a rewound action
        hasn't reached enterActionExecution yet in SkyframeActionExecutor, a repo fetch never
        takes read locks, and all past executions of the action have released all their locks due
        to use of try-with-resources. This means that P_1 can't have any incoming edges in the
        wait-for graph, which is a contradiction to the assumption that it is contained in the
        directed cycle C.

   We conclude that XY = RW. Since the write lock of P_3 is only ever acquired by P_3 itself, all
   edges in C are of the form P_1 -[RW(P_2)]-> P_2. But by construction of inputKeysFor and of the
   repo read locks in SkyframeActionExecutor, the action P_1 is attempting to acquire the read
   locks of all its inputs' producers, and thus P_1 depends on one of the outputs of P_2 (*).

   Applied to all edges of C, we conclude that there is a corresponding directed cycle in the
   combined action and repository dependency graph, which is a contradiction since Skyframe
   disallows dependency cycles.

   Notes:
   * The proof would not go through at (*) if the per-key locks were replaced by a Striped lock
     structure with a fixed number of locks. In fact, this gives rise to a deadlock if the number of
     stripes is at least 2, but low enough that distinct generating actions hash to the same stripe.
   * An action acquires its repo read locks after its action output read locks, which is required
     since a rewound action holds the write lock on its outputs while acquiring its repo read
     locks.
   */

  @Override
  public SilentCloseable enterActionPreparation(Action action, boolean wasRewound)
      throws InterruptedException {
    // Skyframe schedules non-rewound actions such that they never run concurrently with actions
    // that consume their outputs.
    if (!wasRewound) {
      return () -> {};
    }
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.ACTION_LOCK, "action.enterActionPreparation")) {
      return enterActionPreparationForRewinding(action);
    }
  }

  private SilentCloseable enterActionPreparationForRewinding(Action action)
      throws InterruptedException {
    // This action is about to replace outputs that other actions may already be reading.
    rewindingSynchronizer.markReplacementsPossible();
    TransferableWriteLock writeLock = rewindingSynchronizer.acquireWriteLock(outputKeyFor(action));
    try {
      prepareOutputsForRewinding(action);
    } catch (InterruptedException e) {
      writeLock.close();
      throw e;
    }
    return writeLock;
  }

  /**
   * Cancels all async tasks that operate on the action's outputs and resets any cached data about
   * their prefetching state.
   */
  private void prepareOutputsForRewinding(Action action) throws InterruptedException {
    Cancellable task = outputUploadTasks.remove(action);
    if (task != null) {
      task.cancel();
    }
    actionInputFetcher.handleRewoundActionOutputs(action.getOutputs());
  }

  @Override
  public SilentCloseable enterActionExecution(Action action, InputMetadataProvider metadataProvider)
      throws InterruptedException {
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.ACTION_LOCK, "action.enterActionExecution")) {
      return lockArtifactsForConsumption(
          () -> action.getInputs().toList().iterator(), metadataProvider);
    }
  }

  /**
   * Guards a call to {@link
   * com.google.devtools.build.lib.remote.RemoteImportantOutputHandler#processOutputsAndGetLostArtifacts}.
   */
  public SilentCloseable enterProcessOutputsAndGetLostArtifacts(
      Iterable<Artifact> importantOutputs, InputMetadataProvider fullMetadataProvider)
      throws InterruptedException {
    try (SilentCloseable c =
        Profiler.instance()
            .profile(ProfilerTask.ACTION_LOCK, "action.enterProcessOutputsAndGetLostArtifacts")) {
      return lockArtifactsForConsumption(importantOutputs, fullMetadataProvider);
    }
  }

  /**
   * Registers a cancellation callback for an upload of action outputs that may still be running
   * after the action has completed.
   */
  public void registerOutputUploadTask(ActionExecutionMetadata action, Cancellable task) {
    // We don't expect to have multiple output upload tasks for the same action registered at the
    // same time.
    outputUploadTasks.merge(
        action,
        task,
        (oldTask, newTask) -> {
          throw new IllegalStateException(
              "Attempted to register multiple output upload tasks for %s: %s and %s"
                  .formatted(action, oldTask, newTask));
        });
  }

  private SilentCloseable lockArtifactsForConsumption(
      Iterable<Artifact> artifacts, InputMetadataProvider metadataProvider)
      throws InterruptedException {
    return rewindingSynchronizer.acquireReadLocks(() -> inputKeysFor(artifacts, metadataProvider));
  }

  private static ImmutableList<ActionLookupData> inputKeysFor(
      Iterable<Artifact> artifacts, InputMetadataProvider metadataProvider) {
    var allArtifacts =
        Iterables.concat(
            artifacts,
            Iterables.concat(
                Iterables.transform(
                    metadataProvider.getRunfilesTrees(),
                    runfilesTree -> runfilesTree.getArtifacts().toList())));
    return ImmutableList.copyOf(
        Iterables.transform(
            Iterables.filter(allArtifacts, artifact -> artifact instanceof DerivedArtifact),
            artifact -> ((DerivedArtifact) artifact).getGeneratingActionKey()));
  }

  private static ActionLookupData outputKeyFor(Action action) {
    return ((DerivedArtifact) action.getPrimaryOutput()).getGeneratingActionKey();
  }
}
