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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.common.annotations.VisibleForTesting;
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
import com.google.errorprone.annotations.CheckReturnValue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.concurrent.locks.StampedLock;
import javax.annotation.Nullable;

/**
 * A {@link RewoundActionSynchronizer} implementation for Bazel's remote filesystem, which is backed
 * by actual files on disk and requires synchronization to ensure that action outputs aren't deleted
 * while they are being read.
 */
final class RemoteRewoundActionSynchronizer implements RewoundActionSynchronizer {
  /** A task whose cancellation can be requested separately from awaiting its completion. */
  public interface Cancellable {
    /** Requests cancellation without awaiting the task's completion. */
    void requestCancellation();

    /** Waits until the task no longer accesses the outputs of the action it belongs to. */
    void awaitCompletion() throws InterruptedException;
  }

  private final RemoteActionInputFetcher actionInputFetcher;

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
  // A rewound action will acquire a write lock on its lookup data before it prepares for
  // execution, while any action will acquire a read lock on the lookup data of any generating
  // action of its inputs before it starts executing.
  // The values of this cache are weakly referenced to ensure that locks are cleaned up when they
  // are no longer needed.
  @Nullable private volatile LoadingCache<ActionLookupData, ReadWriteLock> fineLocks;

  public RemoteRewoundActionSynchronizer(RemoteActionInputFetcher actionInputFetcher) {
    this.actionInputFetcher = actionInputFetcher;
  }

  /*
  Proof of deadlock freedom:

  As long as the coarse lock is used, there can't be any deadlock because there is only a single
  read-write lock.

  Now assume that there is a deadlock while the fine locks are used. First, note that the logic in
  ImportantOutputHandler that is guarded by enterProcessOutputsAndGetLostArtifacts does not block
  on any (rewound or non-rewound) action executions while it holds read locks and can thus be
  ignored in the following. Consider the directed labeled "wait-for" graph defined as follows:

  * Nodes are given by the currently active Skyframe action execution threads, each of which is
    identified with the action it is (or will be) executing. Actions are in one-to-one
    correspondence with the ActionLookupData that is used as the key in the fine locks map.
  * For each pair of actions A_1 and A_2, there is an edge from A_1 to A_2 labeled with XY(K)
    if A_1 is waiting for the X lock of the key K and A_2 currently holds the Y lock of K, where X
    and Y are either R (for read) or W (for write). The resulting graph may have parallel edges
    with distinct labels.

  Say that an action A "covers" a key K if A is the action identified by K, or if K identifies an
  ActionTemplate and A is one of its expanded actions. By construction of outputKeyFor, the write
  lock of K is only ever acquired by actions covering K, and every action covers exactly one key.

  Let C be any directed cycle in the graph representing a deadlock, let A_1 -[XY(K)]-> A_2 be an
  edge in C and consider the following cases for the pair XY:

  * RR: Since a read-write lock whose read lock is held by at least one thread doesn't
        block any other thread from acquiring its read lock, this case doesn't occur.
  * WW and WR: In both cases, A_1 attempts to acquire a write lock, which only happens when A_1 is
        a rewound action about to prepare for its (re-)execution. While a rewound action is waiting
        for a write lock in enterActionPreparation, it doesn't hold any locks: enterActionExecution
        hasn't been called yet in SkyframeActionExecutor, it only ever acquires the single write
        lock it is waiting for, and all past executions of the action have released all their locks
        due to use of try-with-resources. This means that A_1 can't have any incoming edges in the
        wait-for graph, which is a contradiction to the assumption that it is contained in the
        directed cycle C.

   We conclude that XY = RW, so all edges in C are of the form A_1 -[RW(K)]-> A_2 with A_2 covering
   K. Since every node of C also has an incoming edge, every node of C holds a write lock and thus
   covers the key of that lock.

   By construction of inputKeysFor, A_1 is waiting for R(K) because it has an input guarded by K,
   which is either an output of the action identified by K, or a file in a tree artifact declared
   by the ActionTemplate identified by K. In the latter case, if the input is an individual file
   rather than the tree artifact itself, then A_1 is an expanded action of that template and thus
   covers K - but A_1 covers exactly one key, namely the one of the write lock it holds, which A_2
   holds instead. So A_1 depends on the tree artifact in its entirety and thus on all actions
   covering K, in particular on A_2 (*).

   Applied to all edges of C, we conclude that there is a corresponding directed cycle in the
   action graph, which is a contradiction since Bazel disallows dependency cycles.

   Notes:
   * The proof would not go through at (*) if fineLocks were replaced by a Striped lock structure
     with a fixed number of locks. In fact, this gives rise to a deadlock if the number of stripes
     is at least 2, but low enough that distinct generating actions hash to the same stripe.
   * It is crucial that an action only ever acquires a single write lock: a rewound action holding
     one write lock while waiting for another could deadlock with a reader acquiring the same two
     locks in the opposite order, and readers acquire their locks in an arbitrary order.
   * A rewound action must skip the read lock of the key guarding its own outputs, which it already
     holds the write lock of: the locks aren't reentrant, so an expanded action consuming the
     outputs of another action from the same expansion would otherwise deadlock with itself.
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
    var localCoarseLock = coarseLock;
    if (localCoarseLock != null) {
      // This is the first time a rewound action has attempted to prepare for its execution.
      // Switch to using the fine locks under the protection of the coarse write lock.
      try (SilentCloseable c =
          Profiler.instance().profile(ProfilerTask.ACTION_LOCK, "action.prepareFirstRewinding")) {
        localCoarseLock.writeLock().lockInterruptibly();
      }
      try {
        // Check again under the lock to avoid a race between multiple rewound actions attempting
        // to prepare for execution at the same time.
        if (fineLocks == null) {
          fineLocks =
              Caffeine.newBuilder()
                  .weakValues()
                  // ReentrantReadWriteLock would not work here as its individual read and write
                  // locks do not strongly reference the parent lock, which would lead to locks
                  // being cleaned up while they are still held
                  // (https://bugs.openjdk.org/browse/JDK-8189598). This can be worked around by
                  // using a construction similar to Guava's Striped helpers. StampedLock is both
                  // more memory-efficient and its views do strongly reference the parent lock
                  // (https://github.com/openjdk/jdk/blob/b349f661ea5f14b258191134714a7e712c90ef3e/src/java.base/share/classes/java/util/concurrent/locks/StampedLock.java#L1039),
                  // TODO: Investigate the effect of fair locks on build wall time.
                  .build((ActionLookupData unused) -> new StampedLock().asReadWriteLock());
          // Must be assigned after fineLocks as lockArtifactsForConsumption relies on a null
          // coarseLock implying a non-null fineLocks.
          coarseLock = null;
        }
      } finally {
        localCoarseLock.writeLock().unlock();
      }
    }

    var writeLock = fineLocks.get(outputKeyFor(action)).writeLock();
    try (SilentCloseable c =
        Profiler.instance()
            .profile(ProfilerTask.ACTION_LOCK, "action.awaitRewoundActionConsumers")) {
      writeLock.lockInterruptibly();
    }
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.INFO, "action.prepareOutputsForRewinding")) {
      prepareOutputsForRewinding(action);
    } catch (Throwable t) {
      writeLock.unlock();
      throw t;
    }
    return writeLock::unlock;
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
      return lockArtifactsForConsumption(
          action.getInputs().toList(),
          metadataProvider,
          // A rewound action already holds the write lock on the key guarding its outputs and the
          // locks aren't reentrant. Actions generated by an ActionTemplate can consume the outputs
          // of other actions from the same expansion, which are guarded by the same key.
          wasRewound ? outputKeyFor(action) : null);
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
      return lockArtifactsForConsumption(
          importantOutputs, fullMetadataProvider, /* writeLockedKey= */ null);
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
      Iterable<Artifact> artifacts,
      InputMetadataProvider metadataProvider,
      @Nullable ActionLookupData writeLockedKey)
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
    // We need to switch to it.
    if (localCoarseLock != null) {
      localCoarseLock.readLock().unlock();
    }
    var allReadWriteLocks =
        localFineLocks.getAll(inputKeysFor(artifacts, metadataProvider, writeLockedKey)).values();
    var locksToUnlockBuilder =
        ImmutableList.<Lock>builderWithExpectedSize(allReadWriteLocks.size());
    try {
      for (var readWriteLock : allReadWriteLocks) {
        var readLock = readWriteLock.readLock();
        readLock.lockInterruptibly();
        locksToUnlockBuilder.add(readLock);
      }
    } catch (Throwable e) {
      for (var readLock : locksToUnlockBuilder.build()) {
        readLock.unlock();
      }
      throw e;
    }
    var locksToUnlock = locksToUnlockBuilder.build();
    return () -> locksToUnlock.forEach(Lock::unlock);
  }

  private static Iterable<ActionLookupData> inputKeysFor(
      Iterable<Artifact> artifacts,
      InputMetadataProvider metadataProvider,
      @Nullable ActionLookupData writeLockedKey) {
    var allArtifacts =
        Iterables.concat(
            artifacts,
            Iterables.concat(
                Iterables.transform(
                    metadataProvider.getRunfilesTrees(),
                    runfilesTree -> runfilesTree.getArtifacts().toList())));
    var result =
        Iterables.transform(
            Iterables.filter(allArtifacts, artifact -> artifact instanceof DerivedArtifact),
            artifact -> lockKeyFor((DerivedArtifact) artifact));
    if (writeLockedKey == null) {
      return result;
    }
    return Iterables.filter(result, key -> !key.equals(writeLockedKey));
  }

  /** Returns the key that uniquely identifies the given action. */
  private static ActionLookupData actionKeyFor(ActionExecutionMetadata action) {
    return ((DerivedArtifact) action.getPrimaryOutput()).getGeneratingActionKey();
  }

  /**
   * Returns the key of the lock that guards the given artifact, which is the generating action key
   * of the outermost tree artifact containing it, or its own if it isn't contained in one.
   *
   * <p>This is the artifact's own generating action key except for the outputs of an {@link
   * com.google.devtools.build.lib.actions.ActionTemplate} expansion, which are guarded by the key
   * of the template: they are only ever consumed as part of a tree artifact the template declares,
   * either by actions outside the expansion, which depend on that tree artifact, or by other
   * actions of the same expansion, which depend on individual files in it.
   */
  private static ActionLookupData lockKeyFor(DerivedArtifact artifact) {
    var outermost = artifact;
    for (var parent = artifact.getParent(); parent != null; parent = parent.getParent()) {
      outermost = parent;
    }
    return outermost.getGeneratingActionKey();
  }

  /**
   * Returns the key of the lock that guards the outputs of the given action, which is the key its
   * consumers acquire the read lock of.
   */
  private static ActionLookupData outputKeyFor(Action action) {
    return lockKeyFor((DerivedArtifact) action.getPrimaryOutput());
  }
}
