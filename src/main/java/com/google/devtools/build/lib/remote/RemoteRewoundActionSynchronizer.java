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

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionTemplate;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue;
import com.google.devtools.build.lib.vfs.OutputService.RewoundActionSynchronizer;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.errorprone.annotations.CheckReturnValue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.BooleanSupplier;
import javax.annotation.Nullable;

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
  // Evaluated lazily because Skycache's mode is configured during analysis, which may start after
  // this synchronizer is constructed. It remains fixed throughout execution. Even analysis-only
  // Skycache can retain execution values retrieved by a previous build.
  private final BooleanSupplier useFineLocks;

  // An action generally has at most one such task in flight, but nothing prevents an action from
  // executing multiple spawns whose outputs are uploaded concurrently.
  private final ConcurrentHashMap<ActionLookupData, ImmutableList<Cancellable>> outputUploadTasks =
      new ConcurrentHashMap<>();

  // A single coarse lock is used to synchronize rewound actions (writers) and both rewound and
  // non-rewound actions (readers) as long as no rewound action has attempted to prepare for its
  // execution.
  // This ensures high throughput and low memory footprint for the common case of no rewound
  // actions. In this case, there won't be any writers and the performance characteristics of a
  // ReentrantReadWriteLock are comparable to that of an atomic counter. A StampedLock would not be
  // a good fit as its performance regresses with 127 or more concurrent readers.
  // Note that it wouldn't be correct to only start using this lock once an action is rewound,
  // because a non-rewound action consuming its non-lost outputs could have already started
  // executing.
  @Nullable private volatile ReadWriteLock coarseLock = new ReentrantReadWriteLock();

  // A fine-grained lock structure that is switched to when the first rewound action attempts to
  // prepare for its execution. This structure is used to ensure that rewound actions do not
  // delete their outputs while they are being read by other actions, while still allowing
  // rewound actions and non-rewound actions to run concurrently (i.e., not force the equivalent
  // of --jobs=1 for as long as a rewound action is running, as the coarse lock would).
  // A rewound action will acquire the write lock on its own key before it prepares for execution,
  // while any action will acquire a read lock on the key of each action generating one of its
  // inputs (see inputKeysFor) before it starts executing.
  //
  // The values of this cache are weakly referenced to ensure that locks are cleaned up when they
  // are no longer needed. Holders of a ReadersOrWritersLock reference the lock itself, which keeps
  // it in the cache for as long as it is held, whereas the read and write lock views of the JDK's
  // ReentrantReadWriteLock don't reference their parent lock (JDK-8189598). ReadersOrWritersLock
  // also has a smaller footprint. Crucially, it admits readers even when a writer is waiting;
  // queuing readers behind waiting writers can cause deadlock even with a nonfair JDK lock.
  // Its admission of multiple writers is not relied upon: only the action identified by a key
  // ever acquires the write lock of that key.
  @Nullable private volatile LoadingCache<ActionLookupData, ReadersOrWritersLock> fineLocks;

  public RemoteRewoundActionSynchronizer(
      AbstractActionInputPrefetcher actionInputFetcher,
      WalkableGraph graph,
      BooleanSupplier useFineLocks) {
    this.actionInputFetcher = actionInputFetcher;
    this.graph = graph;
    this.useFineLocks = useFineLocks;
  }

  /*
  Proof of deadlock freedom:

  The coarse lock cannot participate in a cycle of lock acquisitions in this synchronizer.
  When switching to fine locks, readers and the writer release it before acquiring fine locks.
  Otherwise, a rewound action holds its write lock throughout preparation and execution, and can
  reentrantly acquire its read lock. No reader upgrades it to a write lock.

  For the fine locks, we show that a cycle of lock waits would imply a cycle of dependencies,
  which Skyframe disallows. Throughout, "X depends on Y" means that the Skyframe node executing
  action X transitively depends on the node executing action Y.

  1. Relate lock keys to dependencies between actions.

  Every write-lock key identifies an action (see actionKeyFor). By enterActionPreparationForRewinding,
  only a rewound action acquires the write lock of its own key. It does so before it prepares for
  execution, holds the lock until the end of its execution and acquires no other write lock.

  By inputKeysFor, an action acquires the read lock of the key of each action that generates one of
  its inputs, including the artifacts of its runfiles trees, before it starts executing. For a tree
  artifact input, this is the action that generates the tree artifact, or the actions expanded
  from an ActionTemplate that populate it, including producers of empty subdirectories. In either
  case, the reader depends on that action: ActionExecutionFunction requests all inputs, including
  discovered ones, before executing, and ArtifactFunction resolves an artifact by requesting its
  generating action, or, for a tree artifact declared by a template, exactly the expanded actions
  that populate it.

  Thus an action that holds or waits for the read lock of K depends on any action that can acquire
  the write lock of K. An empty expansion has no producers to lock.
  Calls to enterProcessOutputsAndGetLostArtifacts acquire read locks by the same rule.

  2. Classify the edges of a possible lock cycle.

  Consider a directed "wait-for" graph with one node per active action execution or call to
  enterProcessOutputsAndGetLostArtifacts. We refer to action nodes by the action they are
  executing or preparing to execute. An edge A -[XY(K)]-> B means that A is waiting for the X lock
  of K while B holds its Y lock, where R means read and W means write. The graph may have several
  edges between the same pair of nodes.

  Suppose there is a deadlock, and choose a directed cycle C in this graph. Consider any edge
  A -[XY(K)]-> B in C:

  * RR or WW: ReadersOrWritersLock allows multiple readers or multiple writers, but never both.
    Readers therefore wait only for writers, and writers only for readers, ruling out both cases.

  * WR: A waits for a write lock in enterActionPreparation, the only write lock it ever acquires.
    It holds no locks from this execution because read-lock acquisition in enterActionExecution
    has not begun, and previous executions have released their locks through try-with-resources.
    A therefore has no incoming edge and cannot belong to C.

  * RW: A waits to read a key that B holds for writing. By step 1, B is the action identified by
    K and A depends on B.

  Every edge of C is therefore an RW edge, whose target holds a write lock. Calls to
  enterProcessOutputsAndGetLostArtifacts hold no write locks, so they cannot belong to C either.

  3. Derive a dependency cycle.

  By step 2, C is a cycle of RW edges between actions, each of which follows a dependency. C is
  therefore a cycle of dependencies, which Skyframe disallows.

  Notes:

  * Step 1 relies on lock keys preserving action dependencies. A Striped structure with a fixed
    number of locks would let unrelated actions share a lock, so a reader would no longer
    necessarily depend on the writer of its key. Such collisions can cause deadlock with two or
    more stripes.
  * Step 1 also relies on a tree artifact being guarded by the keys of the actions that populate
    it rather than by the key of the ActionTemplate declaring it: a consumer of one tree artifact
    does not depend on expanded actions that only populate other tree artifacts of the same
    template.
  * Step 2 relies on an action acquiring at most one write lock. A rewound action holding one
    write lock while waiting for another could have incoming edges and thus take part in a cycle,
    for example with a reader that acquires the same two locks in the opposite order.
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
    SilentCloseable lock = acquireWriteLock(action);
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.INFO, "action.prepareOutputsForRewinding")) {
      prepareOutputsForRewinding(action);
    } catch (Throwable t) {
      lock.close();
      throw t;
    }
    return lock;
  }

  private SilentCloseable acquireWriteLock(Action action) throws InterruptedException {
    var localCoarseLock = coarseLock;
    if (localCoarseLock != null) {
      boolean switchToFineLocks = useFineLocks.getAsBoolean();
      // Switch to using the fine locks under the protection of the coarse write lock, unless
      // execution values may have been retrieved without their action template expansions.
      try (SilentCloseable c =
          Profiler.instance().profile(ProfilerTask.ACTION_LOCK, "action.prepareFirstRewinding")) {
        localCoarseLock.writeLock().lockInterruptibly();
      }
      if (!switchToFineLocks) {
        // Skycache can supply execution values without their action template expansions. Retain
        // the coarse lock so that consumers of these values are still protected during rewinding.
        return localCoarseLock.writeLock()::unlock;
      }
      try {
        // Check again under the lock to avoid a race between multiple rewound actions attempting
        // to prepare for execution at the same time.
        if (fineLocks == null) {
          fineLocks =
              Caffeine.newBuilder()
                  .weakValues()
                  .build((ActionLookupData _) -> new ReadersOrWritersLock());
          // Must be assigned after fineLocks as lockArtifactsForConsumption relies on a null
          // coarseLock implying a non-null fineLocks.
          coarseLock = null;
        }
      } finally {
        localCoarseLock.writeLock().unlock();
      }
    }

    var writeLock = fineLocks.get(actionKeyFor(action));
    try (SilentCloseable c =
        Profiler.instance()
            .profile(ProfilerTask.ACTION_LOCK, "action.awaitRewoundActionConsumers")) {
      writeLock.lockWriteInterruptibly();
    }
    return writeLock::unlockWrite;
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
    var localCoarseLock = coarseLock;
    if (localCoarseLock != null) {
      // Common case for builds without any rewound actions: acquire the single lock that is never
      // acquired by a writer.
      localCoarseLock.readLock().lockInterruptibly();
    }
    // Read the fine locks after acquiring the coarse lock to allow the fine locks to be inflated
    // lazily.
    var localFineLocks = fineLocks;
    if (localFineLocks == null) {
      // Continuation of the common case for builds without any rewound actions: the fine locks
      // have not been inflated.
      return localCoarseLock.readLock()::unlock;
    }

    // At this point, there has been at least one rewound action that has inflated the fine locks.
    // We need to switch to them.
    if (localCoarseLock != null) {
      localCoarseLock.readLock().unlock();
    }
    var locks = localFineLocks.getAll(inputKeysFor(artifacts, metadataProvider)).values();
    var locksToUnlockBuilder =
        ImmutableList.<ReadersOrWritersLock>builderWithExpectedSize(locks.size());
    try {
      for (var lock : locks) {
        lock.lockReadInterruptibly();
        locksToUnlockBuilder.add(lock);
      }
    } catch (Throwable e) {
      for (var lock : locksToUnlockBuilder.build().reverse()) {
        lock.unlockRead();
      }
      throw e;
    }
    var locksToUnlock = locksToUnlockBuilder.build().reverse();
    return () -> locksToUnlock.forEach(ReadersOrWritersLock::unlockRead);
  }

  /**
   * Returns the keys of the locks that guard the given artifacts as well as all artifacts in the
   * metadata provider's runfiles trees (see {@link #lockKeysFor}).
   */
  private ImmutableSet<ActionLookupData> inputKeysFor(
      Iterable<Artifact> artifacts, InputMetadataProvider metadataProvider)
      throws InterruptedException {
    var allArtifacts =
        Iterables.concat(
            artifacts,
            Iterables.concat(
                Iterables.transform(
                    metadataProvider.getRunfilesTrees(),
                    runfilesTree -> runfilesTree.getArtifacts().toList())));
    var keys = ImmutableSet.<ActionLookupData>builder();
    for (Artifact artifact : allArtifacts) {
      if (artifact instanceof DerivedArtifact derivedArtifact) {
        keys.addAll(lockKeysFor(derivedArtifact));
      }
    }
    return keys.build();
  }

  /**
   * Returns the key that uniquely identifies the given action: the generating action key of its
   * outputs. A rewound action holds the write lock of this key while re-executing and consumers of
   * its outputs hold its read lock while executing (see {@link #lockKeysFor}).
   */
  private static ActionLookupData actionKeyFor(ActionExecutionMetadata action) {
    return ((DerivedArtifact) action.getPrimaryOutput()).getGeneratingActionKey();
  }

  /**
   * Returns the keys of the locks that guard the given artifact: the key of its generating action
   * or, for a tree artifact declared by an {@link
   * com.google.devtools.build.lib.actions.ActionTemplate}, the keys of the expanded actions that
   * populate it, including producers of empty subdirectories. Consumers hold these read locks while
   * executing and a rewound generating action holds the write lock of its own key while
   * re-executing (see {@link #actionKeyFor}).
   */
  private ImmutableList<ActionLookupData> lockKeysFor(DerivedArtifact artifact)
      throws InterruptedException {
    ActionLookupData ownKey = artifact.getGeneratingActionKey();
    if (!artifact.isTreeArtifact()) {
      return ImmutableList.of(ownKey);
    }
    ActionLookupValue owner =
        (ActionLookupValue) checkNotNull(graph.getValue(ownKey.getActionLookupKey()), artifact);
    if (!(owner.getActions().get(ownKey.getActionIndex()) instanceof ActionTemplate)) {
      return ImmutableList.of(ownKey);
    }
    // Expansion values are not rewound and remain available during execution even in
    // non-incremental builds. Use values, not graph edges, which such builds discard.
    var expansionKey =
        ActionTemplateExpansionValue.key(ownKey.getActionLookupKey(), ownKey.getActionIndex());
    var expansion =
        (ActionTemplateExpansionValue) checkNotNull(graph.getValue(expansionKey), artifact);
    return expansion.getGeneratingActionKeys(artifact);
  }
}
