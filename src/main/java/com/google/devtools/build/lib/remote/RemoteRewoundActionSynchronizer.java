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
import com.google.common.collect.ImmutableSet;
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
import java.util.Collection;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
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
  // A rewound action will acquire the write locks on the keys guarding its outputs (see
  // writeLockKeys) before it prepares for execution, while any action will acquire a read lock on
  // the key guarding each of its inputs (see readLockKeys) before it starts executing.
  //
  // Writers of a key deliberately don't exclude each other. The actions covering a key are either
  // the single action identified by it, or the expanded actions of the ActionTemplate identified
  // by it (see lockKeyForOutermostParent). Expanded actions generate disjoint outputs under the
  // tree artifact declared by the template, so writers of a key only ever conflict with its
  // readers, never with each other's outputs. A writer that consumes individual files generated by
  // a sibling of the same expansion holds the read lock on that sibling's own key while executing
  // (see lockKey), so a later rewind of the sibling waits for it like for any other consumer
  // instead of invalidating the files it is reading. Excluding writers from each other would
  // serialize the re-execution of an entire template expansion.
  //
  // Readers wait only for the writers of a key to finish and writers wait only for its readers to
  // finish, never for members of their own group (see ReadersOrWritersLock, which unlike
  // ReentrantReadWriteLock and StampedLock specifically provides this guarantee). A thread waiting
  // for a read lock is therefore never waiting for another reader, which is what the RR case of the
  // deadlock proof below relies on.
  //
  // The values of this cache are weakly referenced to ensure that locks are cleaned up when they
  // are no longer needed.
  @Nullable private volatile LoadingCache<ActionLookupData, ReadersOrWritersLock> fineLocks;

  public RemoteRewoundActionSynchronizer(AbstractActionInputPrefetcher actionInputFetcher) {
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
  ActionTemplate and A is one of its expanded actions. By construction of writeLockKeys, the
  write locks of K are only ever acquired by actions covering K. An expanded action covers two
  keys, its template's and its own; every other action covers exactly one.

  Let C be any directed cycle in the graph representing a deadlock, let A_1 -[XY(K)]-> A_2 be an
  edge in C and consider the following cases for the pair XY:

  * RR: A thread waiting for a fine read lock is only ever waiting for the writers of that key to
        finish (see the comment on fineLocks). If another thread holds the read lock, the key has
        no writers, so this case doesn't occur.
  * WW: The writers of a key don't exclude each other (see the comment on fineLocks), so this case
        doesn't occur.
  * WR: A_1 attempts to acquire a write lock, which only happens when A_1 is a rewound action about
        to prepare for its (re-)execution. If A_1 is waiting for the first write lock it acquires
        in enterActionPreparation, it doesn't hold any locks: enterActionExecution hasn't been
        called yet in SkyframeActionExecutor, the write locks are the first locks it acquires, and
        all past executions of the action have released all their locks due to use of
        try-with-resources. This means that A_1 can't have any incoming edges in the wait-for
        graph, which is a contradiction to the assumption that it is contained in the directed
        cycle C. Otherwise, A_1 is a rewound expanded action waiting for the write lock on its own
        key while already holding the write lock on its template's key (see writeLockKeys), and
        A_2 holds the read lock on A_1's own key. By construction of lockKey, only consumers
        of individual files generated by A_1 acquire that read lock, and such consumers are
        expanded actions of the same template, so A_2 covers the same key as A_1 and depends on
        A_1 (**).

   We conclude that every edge of C is either an RW(K) edge, or a WR(K) edge of the second kind
   above with K the own key of the waiting expanded action, whose endpoints both cover the same
   template key, see (**).

   By construction of lockKey, the waiting action of an RW(K) edge waits because it has an
   input guarded by K: either an output of the single action identified by K, the whole tree
   artifact declared by the ActionTemplate identified by K, or an individual file generated by the
   expanded action identified by K if K is an own key. In each case the waiting action depends on
   all actions covering K (*): a whole tree depends on every expanded action of its template, and
   an own key is covered only by its expanded action. Moreover, if K is an own key, the waiting
   action is an expanded action of the same template, since individual files are only ever
   consumed within their expansion - so both endpoints of the edge cover the same template key.

   Now collapse all nodes of C covering the same template key (or the same key of a non-expanded
   action) into a single node. Every surviving edge is an RW(K) edge with K a template key or the
   key of a non-expanded action, whose waiting action depends on all actions covering K - in
   particular on the action at which C exits the collapsed node the edge points to. If any edge
   survives, chaining these dependencies around the collapsed cycle yields a directed cycle in
   the action graph, which is a contradiction since Bazel disallows dependency cycles.

   Otherwise, C lies entirely within the actions covering a single template key. A WR(K) edge of
   C starts at a writer that holds only the write lock of the template key (own keys are acquired
   last, see writeLockKeys). Its incoming edges could only come from readers waiting on that
   template key - but a sibling never acquires the read lock of its own template key, since no
   expanded action can depend on the tree it contributes to, so the writer has no incoming edges
   within the class and C contains no WR edges at all. C thus consists only of RW edges on own
   keys, each of which points from an action to a sibling it depends on, again yielding a cycle
   in the action graph and a contradiction.

   Notes:
   * The proof would not go through at (*) if fineLocks were replaced by a Striped lock structure
     with a fixed number of locks. In fact, this gives rise to a deadlock if the number of stripes
     is at least 2, but low enough that distinct generating actions hash to the same stripe.
   * A rewound expanded action acquires two write locks, but in a fixed order (the template key
     strictly before its own key), and it never holds any lock of another action's keys while
     waiting for a write lock. A rewound action holding one write lock while waiting for the
     write lock of an unrelated key could deadlock with a reader acquiring read locks on the same
     two keys in the opposite order.
   * A rewound action must not acquire the read locks of the keys guarding its own outputs, which
     it already holds the write locks of, as a reader waits for all writers of the key including
     itself and would thus self-deadlock. This never happens: lockKey maps same-expansion inputs
     to the keys of their generating sibling actions, so an input key of an action could only
     equal one of its own output keys if the action consumed an output it generates itself or the
     tree artifact it contributes to, either of which would be a dependency cycle.
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
                  .build((ActionLookupData _) -> new ReadersOrWritersLock());
          // Must be assigned after fineLocks as lockForConsumption relies on a null coarseLock
          // implying a non-null fineLocks.
          coarseLock = null;
        }
      } finally {
        localCoarseLock.writeLock().unlock();
      }
    }

    SilentCloseable unlock;
    try (SilentCloseable c =
        Profiler.instance()
            .profile(ProfilerTask.ACTION_LOCK, "action.awaitRewoundActionConsumers")) {
      unlock = lockWriteInterruptibly(fineLocks.getAll(writeLockKeys(action)).values());
    }
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.INFO, "action.prepareOutputsForRewinding")) {
      prepareOutputsForRewinding(action);
    } catch (Throwable t) {
      unlock.close();
      throw t;
    }
    return unlock;
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
      return lockForConsumption(readLockKeys(action.getInputs().toList(), metadataProvider));
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
      return lockForConsumption(readLockKeys(importantOutputs, fullMetadataProvider));
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

  private SilentCloseable lockForConsumption(Iterable<ActionLookupData> readLockKeys)
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
    return lockReadInterruptibly(localFineLocks.getAll(readLockKeys).values());
  }

  /**
   * Returns the keys of the locks that guard the given artifacts as well as all artifacts in the
   * metadata provider's runfiles trees.
   */
  private static Iterable<ActionLookupData> readLockKeys(
      Iterable<Artifact> artifacts, InputMetadataProvider metadataProvider) {
    var allArtifacts =
        Iterables.concat(
            artifacts,
            Iterables.concat(
                Iterables.transform(
                    metadataProvider.getRunfilesTrees(),
                    runfilesTree -> runfilesTree.getArtifacts().toList())));
    return Iterables.transform(
        Iterables.filter(allArtifacts, artifact -> artifact instanceof DerivedArtifact),
        artifact -> lockKey((DerivedArtifact) artifact));
  }

  /** Returns the key that uniquely identifies the given action. */
  private static ActionLookupData actionKeyFor(ActionExecutionMetadata action) {
    return lockKey((DerivedArtifact) action.getPrimaryOutput());
  }

  /**
   * Returns all keys of locks that guard outputs of the given action, in the order in which a
   * rewound action must acquire their write locks.
   */
  private static ImmutableSet<ActionLookupData> writeLockKeys(Action action) {
    var primaryOutput = (DerivedArtifact) action.getPrimaryOutput();
    // For actions expanded from an action template, rewinding first acquires a lock for the
    // containing tree artifact and then an additional key for the individual output within it. For
    // all other actions, the two keys below coincide and the set has a single element.
    return ImmutableSet.of(lockKeyForOutermostParent(primaryOutput), lockKey(primaryOutput));
  }

  /**
   * Returns the key of the lock that guards the given artifact individually: its own generating
   * action key. Consumers of the artifact hold the read lock of this key while executing (see
   * readLockKeys) and a rewound generating action holds the write lock while re-executing (see
   * writeLockKeys).
   */
  private static ActionLookupData lockKey(DerivedArtifact artifact) {
    return artifact.getGeneratingActionKey();
  }

  /**
   * Returns the key of the lock that guards the outermost tree artifact containing the given
   * artifact, or {@link #lockKey} if it isn't contained in one. For an output of an {@link
   * com.google.devtools.build.lib.actions.ActionTemplate} expansion this is the key of the
   * template.
   */
  private static ActionLookupData lockKeyForOutermostParent(DerivedArtifact artifact) {
    var outermost = artifact;
    while (outermost.hasParent()) {
      outermost = outermost.getParent();
    }
    return lockKey(outermost);
  }

  private static SilentCloseable lockReadInterruptibly(Collection<ReadersOrWritersLock> locks)
      throws InterruptedException {
    return lockInterruptibly(locks, /* read= */ true);
  }

  private static SilentCloseable lockWriteInterruptibly(Collection<ReadersOrWritersLock> locks)
      throws InterruptedException {
    return lockInterruptibly(locks, /* read= */ false);
  }

  private static SilentCloseable lockInterruptibly(
      Collection<ReadersOrWritersLock> locks, boolean read) throws InterruptedException {
    var locksToUnlockBuilder =
        ImmutableList.<ReadersOrWritersLock>builderWithExpectedSize(locks.size());
    try {
      for (var lock : locks) {
        if (read) {
          lock.lockReadInterruptibly();
        } else {
          lock.lockWriteInterruptibly();
        }
        locksToUnlockBuilder.add(lock);
      }
    } catch (Throwable e) {
      for (var lock : locksToUnlockBuilder.build().reverse()) {
        if (read) {
          lock.unlockRead();
        } else {
          lock.unlockWrite();
        }
      }
      throw e;
    }
    var locksToUnlock = locksToUnlockBuilder.build().reverse();
    if (read) {
      return () -> locksToUnlock.forEach(ReadersOrWritersLock::unlockRead);
    } else {
      return () -> locksToUnlock.forEach(ReadersOrWritersLock::unlockWrite);
    }
  }
}
