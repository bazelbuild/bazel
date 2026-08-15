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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionTemplate;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.concurrent.ReaderPreferringReadWriteLock;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue;
import com.google.devtools.build.lib.vfs.OutputService.RewoundActionSynchronizer;
import com.google.devtools.build.lib.vfs.RewindingSynchronizer;
import com.google.devtools.build.lib.vfs.RewindingSynchronizer.TransferableWriteLock;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.errorprone.annotations.CheckReturnValue;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A {@link RewoundActionSynchronizer} implementation for Bazel's remote filesystem, which is backed
 * by actual files on disk and requires synchronization to ensure that action outputs aren't deleted
 * while they are being read.
 */
public final class RemoteRewoundActionSynchronizer implements RewoundActionSynchronizer {
  /** A task whose cancellation can be requested separately from awaiting its completion. */
  public interface Cancellable {
    /** Requests cancellation without awaiting the task's completion. */
    void requestCancellation();

    /** Waits until the task no longer accesses the outputs of the action it belongs to. */
    void awaitCompletion() throws InterruptedException;
  }

  private final AbstractActionInputPrefetcher actionInputFetcher;
  private final WalkableGraph graph;

  // Rewound actions are producers that replace their outputs in place, so they are synchronized
  // with the actions reading those outputs by the same locks that synchronize repository fetches
  // with the actions reading repository contents. A rewound action takes the write lock of its own
  // key before it prepares for execution, while any action takes read locks for the keys guarding
  // its inputs before it starts executing (see lockKeysFor).
  private final RewindingSynchronizer rewindingSynchronizer;

  // An action generally has at most one such task in flight, but nothing prevents an action from
  // executing multiple spawns whose outputs are uploaded concurrently.
  private final ConcurrentHashMap<ActionLookupData, ImmutableList<Cancellable>> outputUploadTasks =
      new ConcurrentHashMap<>();

  public RemoteRewoundActionSynchronizer(
      AbstractActionInputPrefetcher actionInputFetcher,
      WalkableGraph graph,
      RewindingSynchronizer rewindingSynchronizer) {
    this.actionInputFetcher = actionInputFetcher;
    this.graph = graph;
    this.rewindingSynchronizer = rewindingSynchronizer;
  }

  /*
  Proof of deadlock freedom:

  Rewound actions and repository fetches are both producers that replace outputs which consumers
  may already be reading. Each kind is synchronized by its own RewindingSynchronizer, which starts
  out with a single coarse lock and switches to per-key locks when the first producer acquires a
  write lock. A coarse lock can't be part of a deadlock on its own: it is a single read-write lock
  whose write lock is only acquired by a producer that holds no other lock.

  For the per-key locks, we show that a cycle of lock waits would imply a cycle of dependencies,
  which Skyframe disallows. Throughout, "X depends on Y" means that the Skyframe node evaluating
  producer X transitively depends on the node evaluating producer Y.

  1. Relating lock keys to dependencies between producers.

  Every write-lock key identifies a producer: an action (see actionKeyFor) or a repository. By
  enterActionPreparationForRewinding, only a rewound action acquires the write lock of its own key.
  It does so before it prepares for execution, holds the lock until the end of its execution and
  acquires no other write lock. A repository fetch acquires only the write lock of its own
  repository before replacing the repository's contents and holds it until it is done.

  By inputKeysFor, an action acquires the read lock of the key of each action that generates one of
  its inputs, including the artifacts of its runfiles trees, before it starts executing. For a tree
  artifact input, this is the action that generates the tree artifact, or the actions expanded
  from an ActionTemplate that populate it, including producers of empty subdirectories. In either
  case, the reader depends on that action: ActionExecutionFunction requests all inputs, including
  discovered ones, before executing, and ArtifactFunction resolves an artifact by requesting its
  generating action, or, for a tree artifact declared by a template, exactly the expanded actions
  that populate it. After that, SkyframeActionExecutor acquires the read locks of the repositories
  containing the action's source inputs, on whose fetches the action depends in the same way.

  Thus a producer that holds or waits for the read lock of K depends on any producer that can
  acquire the write lock of K.

  2. Ruling out cycles of lock waits.

  Consider a directed "wait-for" graph with one node per active producer or call to
  enterProcessOutputsAndGetLostArtifacts. We refer to producer nodes by the producer they are
  evaluating or preparing to evaluate. An edge A -[XY(K)]-> B means that A is waiting for the X lock
  of K while B holds its Y lock, where R means read and W means write. The graph may have several
  edges between the same pair of nodes. A reader only ever waits for the holder of a write lock,
  never for a queued writer: RewindingSynchronizer admits readers while a writer waits for other
  readers.

  Suppose there is a deadlock, and choose a directed cycle C in this graph. Consider any edge
  A -[XY(K)]-> B in C:

  * RR or WW: Readers never wait for other readers. Only the producer identified by K acquires the
    write lock of K (step 1) and Skyframe evaluates a producer at most once at a time, so no two
    writers of K exist. Readers therefore wait only for writers, and writers only for readers.

  * WR: A waits for a write lock, the only one it ever acquires. A rewound action waits for it in
    enterActionPreparation and holds no locks from this execution because read-lock acquisition in
    enterActionExecution has not begun, and previous executions have released their locks through
    try-with-resources. A repository fetch never holds read locks. A therefore has no incoming edge
    and cannot belong to C.

  * RW: A waits to read a key that B holds for writing. By step 1, B is the producer identified by
    K.

  Every edge of C is therefore an RW edge, whose target holds a write lock. Calls to
  enterProcessOutputsAndGetLostArtifacts hold no write locks, so they cannot belong to C either.

  C is therefore a cycle of RW edges between producers. By step 1, each edge follows a dependency,
  so C implies a cycle of dependencies, which Skyframe disallows.

  Notes:

  * Step 1 relies on lock keys preserving dependencies. A Striped structure with a fixed number of
    locks would let unrelated producers share a lock, so a reader would no longer necessarily
    depend on the writer of its key. Such collisions can cause deadlock with two or more stripes.

  * With two synchronizers, the order in which an action takes its locks matters: it acquires its
    repository read locks only after the read locks of its inputs' generating actions (see
    SkyframeActionExecutor). A rewound action holds the write lock of its own key while it
    acquires repository read locks, which a coarse lock may queue behind a waiting repository
    fetch. A consumer of its outputs must therefore not hold a repository read lock while waiting
    for that write lock, or the fetch could wait for the consumer, which waits for the rewound
    action, which waits for the fetch.
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
    TransferableWriteLock writeLock;
    try (SilentCloseable c =
        Profiler.instance()
            .profile(ProfilerTask.ACTION_LOCK, "action.awaitRewoundActionConsumers")) {
      writeLock = rewindingSynchronizer.acquireWriteLock(actionKeyFor(action));
    }
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.INFO, "action.prepareOutputsForRewinding")) {
      prepareOutputsForRewinding(action);
    } catch (Throwable t) {
      writeLock.close();
      throw t;
    }
    return writeLock;
  }

  /**
   * Cancels all async tasks that operate on the action's outputs and resets any cached data about
   * their prefetching state.
   */
  private void prepareOutputsForRewinding(Action action) throws InterruptedException {
    ImmutableList<Cancellable> tasks = outputUploadTasks.remove(actionKeyFor(action));
    if (tasks != null) {
      // Request cancellation from every task before awaiting any one of them so that an
      // interruption while awaiting cannot leave later tasks running without cancellation.
      for (Cancellable task : tasks) {
        task.requestCancellation();
      }

      InterruptedException interruption = null;
      for (Cancellable task : tasks) {
        while (true) {
          try {
            task.awaitCompletion();
            break;
          } catch (InterruptedException e) {
            // The tasks have already been removed from the registry, so abandoning one here would
            // let a retry delete outputs it still accesses. Finish awaiting every task and only
            // then propagate the interruption.
            if (interruption == null) {
              interruption = e;
            }
          }
        }
      }
      if (Thread.interrupted()) {
        if (interruption == null) {
          interruption = new InterruptedException();
        }
      }
      if (interruption != null) {
        throw interruption;
      }
    }
    actionInputFetcher.handleRewoundActionOutputs(action.getOutputs());
  }

  @Override
  public SilentCloseable enterActionExecution(
      Action action, boolean wasRewound, InputMetadataProvider metadataProvider)
      throws InterruptedException {
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.ACTION_LOCK, "action.enterActionExecution")) {
      return lockArtifactsForConsumption(action.getInputs().toList(), metadataProvider);
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
   *
   * <p>The returned callback must be run once the upload has completed so that the task doesn't
   * remain registered (and thus retained) for the rest of the build.
   *
   * @return a callback that unregisters this exact task
   */
  @CheckReturnValue
  public Runnable registerOutputUploadTask(ActionExecutionMetadata action, Cancellable task) {
    ActionLookupData key = actionKeyFor(action);
    // merge is atomic with respect to the removal of the entry in prepareOutputsForRewinding.
    outputUploadTasks.merge(
        key,
        ImmutableList.of(task),
        (oldTasks, newTasks) ->
            ImmutableList.<Cancellable>builder().addAll(oldTasks).addAll(newTasks).build());
    return () -> unregisterOutputUploadTask(key, task);
  }

  @VisibleForTesting
  boolean hasRegisteredOutputUploadTasks(ActionExecutionMetadata action) {
    return outputUploadTasks.containsKey(actionKeyFor(action));
  }

  private void unregisterOutputUploadTask(ActionLookupData key, Cancellable task) {
    outputUploadTasks.computeIfPresent(
        key,
        (unusedKey, tasks) -> {
          // Identity comparison: a task is only ever registered once, and a task registered by a
          // re-execution of the action must not be unregistered by its predecessor.
          var remainingTasks = tasks.stream().filter(t -> t != task).collect(toImmutableList());
          return remainingTasks.isEmpty() ? null : remainingTasks;
        });
  }

  private SilentCloseable lockArtifactsForConsumption(
      Iterable<Artifact> artifacts, InputMetadataProvider metadataProvider)
      throws InterruptedException {
    return rewindingSynchronizer.acquireReadLocks(
        () -> ImmutableList.copyOf(inputKeysFor(artifacts, metadataProvider)));
  }

  /**
   * Lazily returns the keys of the locks that guard the given artifacts (see {@link #lockKeysFor}).
   */
  private Iterable<ActionLookupData> inputKeysFor(
      Iterable<Artifact> artifacts, InputMetadataProvider metadataProvider) {
    return Iterables.concat(
        Iterables.transform(
            Iterables.filter(artifacts, DerivedArtifact.class),
            artifact -> lockKeysFor(artifact, metadataProvider)));
  }

  /**
   * Returns the keys of the locks that guard the given artifact. Runfiles trees are expanded into
   * the artifacts they contain. Tree artifacts declared by an {@link ActionTemplate} use the keys
   * of the expanded actions that populate them. Other artifacts use the key of their generating
   * action.
   *
   * <p>Consumers hold these read locks while executing and a rewound generating action holds the
   * write lock of its own key while re-executing (see {@link #actionKeyFor}).
   */
  private Iterable<ActionLookupData> lockKeysFor(
      DerivedArtifact artifact, InputMetadataProvider metadataProvider) {
    if (artifact.isRunfilesTree()) {
      return inputKeysFor(
          checkNotNull(metadataProvider.getRunfilesMetadata(artifact), artifact).getAllArtifacts(),
          metadataProvider);
    }
    ActionLookupData key = artifact.getGeneratingActionKey();
    if (!artifact.isTreeArtifact()) {
      return ImmutableList.of(key);
    }
    try {
      var owner =
          (ActionLookupValue) checkNotNull(graph.getValue(key.getActionLookupKey()), artifact);
      if (!(owner.getActions().get(key.getActionIndex()) instanceof ActionTemplate)) {
        // This tree artifact is the output of a regular action and thus always consumed as a whole.
        return ImmutableList.of(key);
      }
      // Crucially, action template expansion is never rewound and can thus be queried without
      // locking.
      var expansionKey =
          ActionTemplateExpansionValue.key(key.getActionLookupKey(), key.getActionIndex());
      var expansion =
          (ActionTemplateExpansionValue) checkNotNull(graph.getValue(expansionKey), artifact);
      return expansion.getGeneratingActionKeys(artifact);
    } catch (InterruptedException e) {
      // Bazel's in-memory graph lookups do not throw InterruptedException.
      throw new IllegalStateException(e);
    }
  }

  /**
   * Returns the key that uniquely identifies the given action: the generating action key of its
   * outputs. A rewound action holds the write lock of this key while re-executing and consumers of
   * its outputs hold its read lock while executing (see {@link #lockKeysFor}).
   */
  private static ActionLookupData actionKeyFor(ActionExecutionMetadata action) {
    return ((DerivedArtifact) action.getPrimaryOutput()).getGeneratingActionKey();
  }
}
