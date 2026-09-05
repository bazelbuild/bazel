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
  // Writers of a key deliberately don't exclude each other (see ReadersOrWritersLock, which
  // provides this unlike ReentrantReadWriteLock and StampedLock): the write locks of a key are
  // only ever acquired by the single action identified by it or by the expanded actions of the
  // ActionTemplate identified by it, whose outputs are disjoint and whose consumption of each
  // other's outputs is guarded by separate keys (see writeLockKeys). Excluding them from each
  // other would serialize the re-execution of an entire template expansion.
  //
  // The values of this cache are weakly referenced to ensure that locks are cleaned up when they
  // are no longer needed.
  @Nullable private volatile LoadingCache<ActionLookupData, ReadersOrWritersLock> fineLocks;

  public RemoteRewoundActionSynchronizer(AbstractActionInputPrefetcher actionInputFetcher) {
    this.actionInputFetcher = actionInputFetcher;
  }

  /*
  Proof of deadlock freedom:

  The coarse lock cannot cause a deadlock because no action holding it waits for another lock
  from this synchronizer. Readers release it before acquiring fine locks, and the writer releases
  it after switching to fine locks.

  For the fine locks, we show that a cycle of lock waits would imply a cycle of action dependencies.
  Bazel disallows dependency cycles, so this rules out deadlock.

  1. Relate lock keys to action dependencies.

  By writeLockKeys, a rewound ordinary action acquires a write lock only for its own key. A rewound
  action generated by an ActionTemplate acquires two: first its template's, then its own.
  Several actions from the same expansion may hold the template's write lock at once.

  The readLockKeys and lockKey methods choose the read-lock key K for each input according to its
  generating action and its consumer:

  * If K identifies an ordinary action, the input is an output of that action, so the reader
    depends on it.
  * If K identifies an ActionTemplate, the input is either a whole tree artifact declared by the
    template or an individual file in it. A reader of an individual file uses this key only when
    it belongs to a different template expansion, which depends on the whole tree artifact.
    In either case, the reader depends on every action in the producing expansion.
  * If K is an expanded action's own key, the input is a file produced by that action, so the
    reader depends on it. The reader uses this key only when it belongs to the same expansion.

  Thus an action that holds or waits for the read lock of K depends on every action that can
  acquire a write lock for K.

  2. Classify the edges of a possible deadlock cycle.

  Consider a directed "wait-for" graph with one node per active Skyframe action execution thread.
  We refer to each node by the action it is executing or preparing to execute. An edge
  A -[XY(K)]-> B means that A is waiting for the X lock of K while B holds its Y lock, where R
  means read and W means write. The graph may have edges from an action to itself and several
  edges between the same pair of actions.

  The output-processing work guarded by enterProcessOutputsAndGetLostArtifacts does not wait
  for action executions while holding read locks. It therefore cannot complete a cycle of waits
  and can be omitted from the graph.

  Suppose there is a deadlock, and choose a directed cycle C in this graph. Consider any edge
  A -[XY(K)]-> B in C:

  * RR or WW: ReadersOrWritersLock allows multiple readers or multiple writers, but never both.
    Readers therefore wait only for writers, and writers only for readers, ruling out both cases.

  * RW: A waits to read a key that B holds for writing, so step 1 tells us that A depends on B.

  * WR: A waits for a write lock in enterActionPreparation. If this is its first write lock, it
    holds no locks from this execution because read-lock acquisition in enterActionExecution has
    not begun. Previous executions have released their locks through try-with-resources, so A
    holds no locks at all. A therefore has no incoming edge and cannot belong to C.

    If A is waiting for its second write lock, it must be an expanded action. The order imposed
    by writeLockKeys means that A holds its template's write lock and is waiting for its own.
    Since B holds a read lock for A's own key, step 1 tells us that B depends on A and belongs
    to the same expansion.

  3. Rule out cycles between expansions and within an expansion.

  Group the actions by template expansion, with each ordinary action forming a group of its own.
  Step 2 shows that every WR edge in C stays within a group, as does every RW edge for an expanded
  action's own key. Thus only RW edges for template keys or ordinary action keys can cross
  between groups.

  If C crosses between groups, consider two consecutive crossing edges, A -> B and D -> E.
  The part of C from B to D stays within one group, since there is no crossing edge between them.
  By step 1, A depends on every action in B's group and therefore on D. Applying this reasoning
  to each consecutive pair of crossing edges gives a cycle of action dependencies, which Bazel
  disallows. This argument does not require the edges between B and D to be action dependencies.

  If C stays within one group, suppose it contains a WR edge starting at A. By step 2, A holds
  only its template's write lock, so any incoming edge must come from an action waiting for that
  lock. It cannot come from a writer, because writers never wait for other writers. It cannot
  come from a reader in the same group either: by step 1, that reader would depend on every action
  in its own expansion, including itself. Thus A has no incoming edge within the group and cannot
  belong to C.

  A cycle within one group must therefore consist entirely of RW edges. Since each of these is
  an action dependency, this would also be a dependency cycle, ruling out the remaining case.

  Note: Step 1 relies on lock keys preserving action dependencies. A Striped structure with a
  fixed number of locks would let unrelated actions share a lock. A reader would then no longer
  necessarily depend on every writer of the same key. Such collisions can cause deadlock with
  two or more stripes.
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
          // Must be assigned after fineLocks as lockForReading relies on a null coarseLock
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
      // The write locks must be acquired in the order of writeLockKeys, which getAll does not
      // guarantee to preserve.
      unlock =
          acquireWriteLocks(
              writeLockKeys(action).stream().map(fineLocks::get).collect(toImmutableList()));
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
    ImmutableList<Cancellable> tasks = outputUploadTasks.remove(actionKey(action));
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
      return lockForReading(
          readLockKeys(
              action.getInputs().toList(),
              metadataProvider,
              lockKeyForOutermostParent((DerivedArtifact) action.getPrimaryOutput())));
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
      return lockForReading(
          readLockKeys(importantOutputs, fullMetadataProvider, /* consumerTemplateKey= */ null));
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
    ActionLookupData key = actionKey(action);
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
    return outputUploadTasks.containsKey(actionKey(action));
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

  private SilentCloseable lockForReading(Iterable<ActionLookupData> keys)
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
    return acquireReadLocks(localFineLocks.getAll(keys).values());
  }

  /**
   * Returns the keys of the locks that guard the given artifacts as well as all artifacts in the
   * metadata provider's runfiles trees.
   *
   * @param consumerTemplateKey the {@link #lockKeyForOutermostParent} of the consuming action's
   *     primary output, or null if the consumer isn't an action
   */
  private static Iterable<ActionLookupData> readLockKeys(
      Iterable<Artifact> artifacts,
      InputMetadataProvider metadataProvider,
      @Nullable ActionLookupData consumerTemplateKey) {
    var allArtifacts =
        Iterables.concat(
            artifacts,
            Iterables.concat(
                Iterables.transform(
                    metadataProvider.getRunfilesTrees(),
                    runfilesTree -> runfilesTree.getArtifacts().toList())));
    return Iterables.transform(
        Iterables.filter(allArtifacts, artifact -> artifact instanceof DerivedArtifact),
        artifact -> {
          var derivedArtifact = (DerivedArtifact) artifact;
          var templateKey = lockKeyForOutermostParent(derivedArtifact);
          // Individual files generated by an expanded action are guarded by that action's own key
          // only for consumers of the same expansion; any other consumer is guarded by the key of
          // the template like a consumer of the whole tree artifact.
          return templateKey.equals(consumerTemplateKey) ? lockKey(derivedArtifact) : templateKey;
        });
  }

  /** Returns the key that uniquely identifies the given action. */
  private static ActionLookupData actionKey(ActionExecutionMetadata action) {
    return lockKey((DerivedArtifact) action.getPrimaryOutput());
  }

  /**
   * Returns the keys of the locks that guard the outputs of the given action, in the order in which
   * a rewound action must acquire their write locks.
   *
   * <p>The outputs of an action expanded from an {@link
   * com.google.devtools.build.lib.actions.ActionTemplate} are guarded by two keys: that of the
   * template, whose read lock is held by consumers of the tree artifact it declares, and its own,
   * whose read lock is held by actions of the same expansion that consume individual files it
   * generates (see lockKey). For all other actions, both keys coincide.
   */
  private static ImmutableSet<ActionLookupData> writeLockKeys(Action action) {
    var primaryOutput = (DerivedArtifact) action.getPrimaryOutput();
    return ImmutableSet.of(lockKeyForOutermostParent(primaryOutput), lockKey(primaryOutput));
  }

  /**
   * Returns the key of the lock that guards the given artifact individually: its own generating
   * action key. Consumers of the artifact hold the read lock of this key while executing, unless it
   * is an individual file of a tree artifact populated by a template expansion they aren't part of
   * (see readLockKeys), and a rewound generating action holds the write lock while re-executing
   * (see writeLockKeys).
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

  private static SilentCloseable acquireReadLocks(Collection<ReadersOrWritersLock> locks)
      throws InterruptedException {
    return acquireLocks(locks, /* read= */ true);
  }

  private static SilentCloseable acquireWriteLocks(Collection<ReadersOrWritersLock> locks)
      throws InterruptedException {
    return acquireLocks(locks, /* read= */ false);
  }

  private static SilentCloseable acquireLocks(
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
