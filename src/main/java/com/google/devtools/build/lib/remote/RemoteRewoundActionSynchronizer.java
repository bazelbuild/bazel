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
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionTemplate;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue;
import com.google.devtools.build.lib.vfs.OutputService.RewoundActionSynchronizer;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.errorprone.annotations.CheckReturnValue;
import java.lang.invoke.VarHandle;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

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

  // An action generally has at most one such task in flight, but nothing prevents an action from
  // executing multiple spawns whose outputs are uploaded concurrently.
  private final ConcurrentHashMap<PathFragment, ImmutableList<Cancellable>> outputUploadTasks =
      new ConcurrentHashMap<>();

  // As long as no rewound action has attempted to prepare for its execution, consumers acquire no
  // locks and only register themselves here. A rewound action waits for the registered consumers
  // that read its outputs, while all other consumers keep running.
  // This keeps the common case of no rewound actions cheap. Registration can't be deferred until an
  // action is rewound, because a non-rewound action consuming its non-lost outputs could have
  // already started executing.
  // The set only holds consumers that are currently executing, discovering inputs or processing
  // top-level outputs. Its size thus scales with the number of concurrently running actions, which
  // is capped at BuildRequestOptions.MAX_JOBS (5000), even with async execution. An entry costs
  // about 100 bytes and only references state that the consumer keeps alive anyway, until a
  // rewound action caches its lock keys.
  private final Set<RegisteredConsumer> registeredConsumers = ConcurrentHashMap.newKeySet();

  // A fine-grained lock structure that is inflated when the first rewound action attempts to
  // prepare for its execution and used by all consumers from then on. This structure is used to
  // ensure that rewound actions do not delete their outputs while they are being read by other
  // actions, while still allowing rewound actions and non-rewound actions to run concurrently.
  // A rewound action will acquire the write lock on its own key before it prepares for execution,
  // while any action will acquire a read lock on the key of each action generating one of its
  // inputs (see inputKeysFor) before it starts executing.
  //
  // ReaderPreferringReadWriteLock is used as java.util.concurrent locks may queue readers behind
  // waiting writers even while other readers hold the lock, which would cause deadlocks (see the
  // proof below)
  //
  // The values of this cache are weakly referenced to ensure that locks are cleaned up when they
  // are no longer needed.
  @Nullable private volatile LoadingCache<PathFragment, ReaderPreferringReadWriteLock> fineLocks;

  public RemoteRewoundActionSynchronizer(
      AbstractActionInputPrefetcher actionInputFetcher, WalkableGraph graph) {
    this.actionInputFetcher = actionInputFetcher;
    this.graph = graph;
  }

  /*
  Proof of deadlock freedom:

  A consumer that registered itself before the fine locks were inflated acquires no locks and thus
  never waits for one. A rewound action only waits for such consumers before it acquires its write
  lock and thus while it holds no locks (see WR below), so these waits can't be part of a cycle.

  For the fine locks, we show that a cycle of lock waits would imply a cycle of dependencies,
  which Skyframe disallows. Throughout, "X depends on Y" means that the Skyframe node executing
  action X transitively depends on the node executing action Y.

  1. Relating lock keys to dependencies between actions.

  Every write-lock key identifies an action up to sharing (see actionKeyFor) as shared actions are
  executed and rewound as one by SkyframeActionExecutor. Below, an "action" always refers to its set
  of shared actions, By enterActionPreparationForRewinding, only a rewound action acquires the write
  lock of its own key. It does so before it prepares for execution, holds the lock until the end of
  its execution and acquires no other write lock.

  By inputKeysFor, an action acquires the read lock of the key of each action that generates one of
  its inputs, including the artifacts of its runfiles trees, before it starts executing. While it
  discovers inputs, it additionally holds the read locks of the keys of the actions generating its
  scheduling dependencies, which are the artifacts it may read during discovery. For a tree
  artifact input, this is the action that generates the tree artifact, or the actions expanded
  from an ActionTemplate that populate it, including producers of empty subdirectories. In either
  case, the reader depends on that action: ActionExecutionFunction requests the evaluated inputs
  (including inputs restored from the action cache) and scheduling dependencies before discovering
  inputs, and all discovered inputs before executing. ArtifactFunction resolves an artifact by
  requesting its generating action, or, for a tree artifact declared by a template, exactly the
  expanded actions that populate it.

  Thus an action that holds or waits for the read lock of K depends on any action that can acquire
  the write lock of K.

  2. Ruling out cycles of lock waits.

  Consider a directed "wait-for" graph with one node per active action execution or call to
  enterProcessOutputsAndGetLostArtifacts. We refer to action nodes by the action they are
  executing or preparing to execute. An edge A -[XY(K)]-> B means that A is waiting for the X lock
  of K while B holds its Y lock, where R means read and W means write. The graph may have several
  edges between the same pair of nodes.

  Suppose there is a deadlock, and choose a directed cycle C in this graph. Consider any edge
  A -[XY(K)]-> B in C:

  * RR or WW: Readers never wait for other readers. Only the action identified by K acquires the
    write lock of K (step 1) and SkyframeActionExecutor executes an action, shared members
    included, at most once at a time, so no two writers of K exist. Readers therefore wait only
    for writers, and writers only for readers.

  * WR: A waits for a write lock in enterActionPreparation, the only write lock it ever acquires.
    It holds no locks from this execution because read-lock acquisition in enterActionExecution
    has not begun, and previous executions have released their locks through try-with-resources.
    A therefore has no incoming edge and cannot belong to C.

  * RW: A waits to read a key that B holds for writing. By step 1, B is the action identified by K.

  Every edge of C is therefore an RW edge, whose target holds a write lock. Calls to
  enterProcessOutputsAndGetLostArtifacts hold no write locks, so they cannot belong to C either.

  C is therefore a cycle of RW edges between actions. By step 1, each edge follows a dependency,
  so C implies a cycle of dependencies, which Skyframe disallows.

  Notes:

  * Step 1 relies on lock keys preserving action dependencies. A Striped structure with a fixed
    number of locks would let unrelated actions share a lock, so a reader would no longer
    necessarily depend on the writer of its key. Such collisions can cause deadlock with two or
    more stripes.
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
    if (fineLocks == null) {
      synchronized (this) {
        if (fineLocks == null) {
          fineLocks =
              Caffeine.newBuilder()
                  .weakValues()
                  .build((PathFragment _) -> new ReaderPreferringReadWriteLock());
        }
      }
    }
    // Pairs with the fence in lockArtifactsForConsumption so that every consumer that doesn't see
    // the fine locks is seen here.
    VarHandle.fullFence();
    PathFragment key = actionKeyFor(action);
    try (var _ =
        Profiler.instance().profile(ProfilerTask.ACTION_LOCK, "action.awaitRegisteredConsumers")) {
      for (RegisteredConsumer consumer : registeredConsumers) {
        if (Thread.interrupted()) {
          throw new InterruptedException();
        }
        consumer.awaitIfConsuming(key);
      }
    }

    var writeLock = fineLocks.get(key);
    try (SilentCloseable c =
        Profiler.instance()
            .profile(ProfilerTask.ACTION_LOCK, "action.awaitRewoundActionConsumers")) {
      writeLock.lockWriteInterruptibly();
    }
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.INFO, "action.prepareOutputsForRewinding")) {
      prepareOutputsForRewinding(action);
    } catch (Throwable t) {
      writeLock.unlockWrite();
      throw t;
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
      NestedSet<Artifact> inputs = action.getInputs();
      return lockArtifactsForConsumption(inputs::toList, metadataProvider);
    }
  }

  @Override
  public SilentCloseable enterInputDiscovery(
      Action action, NestedSet<Artifact> evaluatedInputs, InputMetadataProvider metadataProvider)
      throws InterruptedException {
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.ACTION_LOCK, "action.enterInputDiscovery")) {
      return lockArtifactsForConsumption(
          () ->
              Iterables.concat(
                  evaluatedInputs.toList(), action.getSchedulingDependencies().toList()),
          metadataProvider);
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
      return lockArtifactsForConsumption(() -> importantOutputs, fullMetadataProvider);
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
    PathFragment key = actionKeyFor(action);
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

  private void unregisterOutputUploadTask(PathFragment key, Cancellable task) {
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
      Supplier<Iterable<Artifact>> artifacts, InputMetadataProvider metadataProvider)
      throws InterruptedException {
    // Registration never blocks, so check for interruption explicitly.
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }
    var localFineLocks = fineLocks;
    if (localFineLocks == null) {
      // Until the first rewound action inflates the fine locks, which most builds never do,
      // consumers register instead of acquiring locks. Their artifacts are then only computed if a
      // rewound action scans them.
      var consumer = new RegisteredConsumer(artifacts, metadataProvider);
      registeredConsumers.add(consumer);
      // Pairs with the fence in enterActionPreparationForRewinding so that every rewound action
      // finds this consumer registered if the fine locks are still null below.
      VarHandle.fullFence();
      localFineLocks = fineLocks;
      if (localFineLocks == null) {
        return consumer::unregister;
      }
      consumer.unregister();
    }

    // At this point, there has been at least one rewound action that has inflated the fine locks.
    var locks = localFineLocks.getAll(inputKeysFor(artifacts.get(), metadataProvider)).values();
    var locksToUnlockBuilder =
        ImmutableList.<ReaderPreferringReadWriteLock>builderWithExpectedSize(locks.size());
    try {
      for (var lock : locks) {
        lock.lockReadInterruptibly();
        locksToUnlockBuilder.add(lock);
      }
    } catch (Throwable e) {
      for (var lock : locksToUnlockBuilder.build()) {
        lock.unlockRead();
      }
      throw e;
    }
    var locksToUnlock = locksToUnlockBuilder.build();
    return () -> locksToUnlock.forEach(ReaderPreferringReadWriteLock::unlockRead);
  }

  /** A consumer that started before the fine locks were inflated and thus holds no locks. */
  private final class RegisteredConsumer {
    private final Supplier<Iterable<Artifact>> artifacts;
    private final InputMetadataProvider metadataProvider;

    @GuardedBy("this")
    private boolean registered = true;

    @GuardedBy("this")
    @Nullable
    private ImmutableSet<PathFragment> lockKeys;

    RegisteredConsumer(
        Supplier<Iterable<Artifact>> artifacts, InputMetadataProvider metadataProvider) {
      this.artifacts = artifacts;
      this.metadataProvider = metadataProvider;
    }

    void unregister() {
      registeredConsumers.remove(this);
      // Waits for a rewound action that is computing the lock keys, since the consumer may mutate
      // its metadata provider once unregistered.
      synchronized (this) {
        registered = false;
        notifyAll();
      }
    }

    /** Waits until this consumer is unregistered if it consumes artifacts guarded by the key. */
    synchronized void awaitIfConsuming(PathFragment key) throws InterruptedException {
      if (!registered) {
        return;
      }
      if (lockKeys == null) {
        // Rewound actions tend to arrive in bursts. Building the set costs about as much as a
        // single scan of the inputs and makes later scans of this consumer nearly free.
        lockKeys = ImmutableSet.copyOf(inputKeysFor(artifacts.get(), metadataProvider));
      }
      if (!lockKeys.contains(key)) {
        return;
      }
      while (registered) {
        wait();
      }
    }
  }

  /**
   * Lazily returns the keys of the locks that guard the given artifacts (see {@link #lockKeysFor}).
   */
  private Iterable<PathFragment> inputKeysFor(
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
  private Iterable<PathFragment> lockKeysFor(
      DerivedArtifact artifact, InputMetadataProvider metadataProvider) {
    if (artifact.isRunfilesTree()) {
      return inputKeysFor(
          checkNotNull(metadataProvider.getRunfilesMetadata(artifact), artifact).getAllArtifacts(),
          metadataProvider);
    }
    ActionLookupData key = artifact.getGeneratingActionKey();
    try {
      var owner =
          (ActionLookupValue) checkNotNull(graph.getValue(key.getActionLookupKey()), artifact);
      ActionAnalysisMetadata generatingAction = owner.getActions().get(key.getActionIndex());
      if (artifact.isTreeArtifact() && generatingAction instanceof ActionTemplate) {
        // Crucially, action template expansion is never rewound and can thus be queried without
        // locking.
        var expansionKey =
            ActionTemplateExpansionValue.key(key.getActionLookupKey(), key.getActionIndex());
        var expansion =
            (ActionTemplateExpansionValue) checkNotNull(graph.getValue(expansionKey), artifact);
        return Lists.transform(
            expansion.getGeneratingActionKeys(artifact),
            expandedKey -> actionKeyFor(expansion.getActions().get(expandedKey.getActionIndex())));
      }
      // A tree artifact that is the output of a regular action is always consumed as a whole.
      return ImmutableList.of(actionKeyFor(generatingAction));
    } catch (InterruptedException e) {
      // Bazel's in-memory graph lookups do not throw InterruptedException.
      throw new IllegalStateException(e);
    }
  }

  /**
   * Returns the exec path of the action's primary output, which identifies the given action up to
   * sharing. A rewound action holds the write lock of this key while re-executing and consumers of
   * its outputs hold its read lock while executing (see {@link #lockKeysFor}).
   */
  private static PathFragment actionKeyFor(ActionAnalysisMetadata action) {
    return action.getPrimaryOutput().getExecPath();
  }
}
