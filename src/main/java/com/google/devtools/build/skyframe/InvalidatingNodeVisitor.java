// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.ErrorClassifier;
import com.google.devtools.build.lib.concurrent.ForkJoinQuiescingExecutor;
import com.google.devtools.build.lib.concurrent.QuiescingExecutor;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.ThinNodeEntry.DirtyType;
import com.google.devtools.build.skyframe.ThinNodeEntry.MarkedDirtyResult;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;

/**
 * A visitor that is useful for invalidating transitive dependencies of Skyframe nodes.
 *
 * <p>Interruptibility: It is safe to interrupt the invalidation process at any time. Consider a
 * graph and a set of modified nodes. Then the reverse transitive closure of the modified nodes is
 * the set of dirty nodes. We provide interruptibility by making sure that the following invariant
 * holds at any time:
 *
 * <p>If a node is dirty, but not removed (or marked as dirty) yet, then either it or any of its
 * transitive dependencies must be in the {@link #pendingVisitations} set. Furthermore, reverse dep
 * pointers must always point to existing nodes.
 *
 * <p>Thread-safety: This class should only be instantiated and called on a single thread, but
 * internally it spawns many worker threads to process the graph. The thread-safety of the workers
 * on the graph can be delicate, and is documented below. Moreover, no other modifications to the
 * graph can take place while invalidation occurs.
 *
 * <p>This is intended only for use in alternative {@code MemoizingEvaluator} implementations.
 */
public abstract class InvalidatingNodeVisitor<GraphT extends QueryableGraph> {

  // Default thread count is equal to the number of cores to exploit
  // that level of hardware parallelism, since invalidation should be CPU-bound.
  // We may consider increasing this in the future.
  private static final int DEFAULT_THREAD_COUNT = Runtime.getRuntime().availableProcessors();
  private static final int EXPECTED_PENDING_SET_SIZE = DEFAULT_THREAD_COUNT * 8;
  private static final int EXPECTED_VISITED_SET_SIZE = 1024;

  private static final ErrorClassifier errorClassifier =
      new ErrorClassifier() {
        @Override
        protected ErrorClassification classifyException(Exception e) {
          return e instanceof RuntimeException
              ? ErrorClassification.CRITICAL_AND_LOG
              : ErrorClassification.NOT_CRITICAL;
        }
      };

  protected final GraphT graph;
  protected final DirtyTrackingProgressReceiver progressReceiver;
  // Aliased to InvalidationState.pendingVisitations.
  protected final Set<Pair<SkyKey, InvalidationType>> pendingVisitations;
  protected final QuiescingExecutor executor;

  protected InvalidatingNodeVisitor(
      GraphT graph, DirtyTrackingProgressReceiver progressReceiver, InvalidationState state) {
    this.executor =
        new AbstractQueueVisitor(
            /*parallelism=*/ DEFAULT_THREAD_COUNT,
            /*keepAliveTime=*/ 15,
            /*units=*/ TimeUnit.SECONDS,
            /*failFastOnException=*/ true,
            "skyframe-invalidator",
            errorClassifier);
    this.graph = Preconditions.checkNotNull(graph);
    this.progressReceiver = Preconditions.checkNotNull(progressReceiver);
    this.pendingVisitations = state.pendingValues;
  }

  protected InvalidatingNodeVisitor(
      GraphT graph,
      DirtyTrackingProgressReceiver progressReceiver,
      InvalidationState state,
      ForkJoinPool forkJoinPool) {
    this.executor = ForkJoinQuiescingExecutor.newBuilder()
        .withOwnershipOf(forkJoinPool)
        .setErrorClassifier(errorClassifier)
        .build();
    this.graph = Preconditions.checkNotNull(graph);
    this.progressReceiver = Preconditions.checkNotNull(progressReceiver);
    this.pendingVisitations = state.pendingValues;
  }

  /** Initiates visitation and waits for completion. */
  void run() throws InterruptedException {
    // Make a copy to avoid concurrent modification confusing us as to which nodes were passed by
    // the caller, and which are added by other threads during the run. Since no tasks have been
    // started yet (the queueDirtying calls start them), this is thread-safe.
    for (final Pair<SkyKey, InvalidationType> visitData :
        ImmutableList.copyOf(pendingVisitations)) {
      executor.execute(
          new Runnable() {
            @Override
            public void run() {
              visit(ImmutableList.of(visitData.first), visitData.second);
            }
          });
    }
    try {
      executor.awaitQuiescence(/*interruptWorkers=*/ true);
    } catch (IllegalStateException e) {
      // TODO(mschaller): Remove this wrapping after debugging the invalidation-after-OOMing-eval
      // problem. The wrapping provides a stack trace showing what caused the invalidation.
      throw new IllegalStateException(e);
    }

    // Note: implementations that do not support interruption also do not update pendingVisitations.
    Preconditions.checkState(!getSupportInterruptions() || pendingVisitations.isEmpty(),
        "All dirty nodes should have been processed: %s", pendingVisitations);
  }

  protected abstract boolean getSupportInterruptions();

  @VisibleForTesting
  CountDownLatch getInterruptionLatchForTestingOnly() {
    return executor.getInterruptionLatchForTestingOnly();
  }

  /** Enqueues nodes for invalidation. Elements of {@code keys} may not exist in the graph. */
  @ThreadSafe
  abstract void visit(Iterable<SkyKey> keys, InvalidationType invalidationType);

  @VisibleForTesting
  enum InvalidationType {
    /** The node is dirty and must be recomputed. */
    CHANGED,
    /** The node is dirty, but may be marked clean later during change pruning. */
    DIRTIED,
    /** The node is deleted. */
    DELETED;
  }

  /**
   * Invalidation state object that keeps track of which nodes need to be invalidated, but have not
   * been dirtied/deleted yet. This supports interrupts - by only deleting a node from this set
   * when all its parents have been invalidated, we ensure that no information is lost when an
   * interrupt comes in.
   */
  static class InvalidationState {

    private final Set<Pair<SkyKey, InvalidationType>> pendingValues =
        Collections.newSetFromMap(
            new ConcurrentHashMap<Pair<SkyKey, InvalidationType>, Boolean>(
                EXPECTED_PENDING_SET_SIZE, .75f, DEFAULT_THREAD_COUNT));
    private final InvalidationType defaultUpdateType;

    private InvalidationState(InvalidationType defaultUpdateType) {
      this.defaultUpdateType = Preconditions.checkNotNull(defaultUpdateType);
    }

    void update(Iterable<SkyKey> diff) {
      Iterables.addAll(pendingValues, Iterables.transform(diff,
          new Function<SkyKey, Pair<SkyKey, InvalidationType>>() {
            @Override
            public Pair<SkyKey, InvalidationType> apply(SkyKey skyKey) {
              return Pair.of(skyKey, defaultUpdateType);
            }
          }));
    }

    @VisibleForTesting
    boolean isEmpty() {
      return pendingValues.isEmpty();
    }

    @VisibleForTesting
    Set<Pair<SkyKey, InvalidationType>> getInvalidationsForTesting() {
      return ImmutableSet.copyOf(pendingValues);
    }
  }

  static class DirtyingInvalidationState extends InvalidationState {
    public DirtyingInvalidationState() {
      super(InvalidationType.CHANGED);
    }
  }

  static class DeletingInvalidationState extends InvalidationState {
    DeletingInvalidationState() {
      super(InvalidationType.DELETED);
    }
  }

  /** A node-deleting implementation. */
  static class DeletingNodeVisitor extends InvalidatingNodeVisitor<InMemoryGraph> {

    private final Set<SkyKey> visited = Sets.newConcurrentHashSet();
    private final boolean traverseGraph;

    DeletingNodeVisitor(
        InMemoryGraph graph,
        DirtyTrackingProgressReceiver progressReceiver,
        InvalidationState state,
        boolean traverseGraph) {
      super(graph, progressReceiver, state);
      this.traverseGraph = traverseGraph;
    }

    @Override
    protected boolean getSupportInterruptions() {
      return true;
    }

    @Override
    public void visit(Iterable<SkyKey> keys, InvalidationType invalidationType) {
      Preconditions.checkState(invalidationType == InvalidationType.DELETED, keys);
      ImmutableList.Builder<SkyKey> unvisitedKeysBuilder = ImmutableList.builder();
      for (SkyKey key : keys) {
        if (visited.add(key)) {
          unvisitedKeysBuilder.add(key);
        }
      }
      ImmutableList<SkyKey> unvisitedKeys = unvisitedKeysBuilder.build();
      for (SkyKey key : unvisitedKeys) {
        pendingVisitations.add(Pair.of(key, InvalidationType.DELETED));
      }
      final Map<SkyKey, ? extends NodeEntry> entries =
          graph.getBatch(null, Reason.INVALIDATION, unvisitedKeys);
      for (final SkyKey key : unvisitedKeys) {
        executor.execute(
            new Runnable() {
              @Override
              public void run() {
                NodeEntry entry = entries.get(key);
                Pair<SkyKey, InvalidationType> invalidationPair =
                    Pair.of(key, InvalidationType.DELETED);
                if (entry == null) {
                  pendingVisitations.remove(invalidationPair);
                  return;
                }

                if (traverseGraph) {
                  // Propagate deletion upwards.
                  visit(entry.getAllReverseDepsForNodeBeingDeleted(), InvalidationType.DELETED);

                  // Unregister this node as an rdep from its direct deps, since reverse dep
                  // edges cannot point to non-existent nodes. To know whether the child has this
                  // node as an "in-progress" rdep to be signaled, or just as a known rdep, we
                  // look at the deps that this node declared during its last (presumably
                  // interrupted) evaluation. If a dep is in this set, then it was notified to
                  // signal this node, and so the rdep will be an in-progress rdep, if the dep
                  // itself isn't done. Otherwise it will be a normal rdep. That information is
                  // used to remove this node as an rdep from the correct list of rdeps in the
                  // child -- because of our compact storage of rdeps, checking which list
                  // contains this parent could be expensive.
                  Set<SkyKey> signalingDeps =
                      entry.isDone()
                          ? ImmutableSet.<SkyKey>of()
                          : entry.getTemporaryDirectDeps().toSet();
                  Iterable<SkyKey> directDeps;
                  try {
                    directDeps =
                        entry.isDone()
                            ? entry.getDirectDeps()
                            : entry.getAllDirectDepsForIncompleteNode();
                  } catch (InterruptedException e) {
                    throw new IllegalStateException(
                        "Deletion cannot happen on a graph that may have blocking operations: "
                            + key
                            + ", "
                            + entry,
                        e);
                  }
                  Map<SkyKey, ? extends NodeEntry> depMap =
                      graph.getBatch(key, Reason.INVALIDATION, directDeps);
                  for (Map.Entry<SkyKey, ? extends NodeEntry> directDepEntry : depMap.entrySet()) {
                    NodeEntry dep = directDepEntry.getValue();
                    if (dep != null) {
                      if (dep.isDone() || !signalingDeps.contains(directDepEntry.getKey())) {
                        try {
                          dep.removeReverseDep(key);
                        } catch (InterruptedException e) {
                          throw new IllegalStateException(
                              "Deletion cannot happen on a graph that may have blocking "
                                  + "operations: "
                                  + key
                                  + ", "
                                  + entry,
                              e);
                        }
                      } else {
                        // This step is not strictly necessary, since all in-progress nodes are
                        // deleted during graph cleaning, which happens in a single
                        // DeletingNodeVisitor visitation, aka the one right now. We leave this
                        // here in case the logic changes.
                        dep.removeInProgressReverseDep(key);
                      }
                    }
                  }
                }

                // Allow custom key-specific logic to update dirtiness status.
                progressReceiver.invalidated(
                    key, EvaluationProgressReceiver.InvalidationState.DELETED);
                // Actually remove the node.
                graph.remove(key);

                // Remove the node from the set as the last operation.
                pendingVisitations.remove(invalidationPair);
              }
            });
      }
    }
  }

  /** A node-dirtying implementation. */
  static class DirtyingNodeVisitor extends InvalidatingNodeVisitor<QueryableGraph> {

    private final Set<SkyKey> changed =
        Collections.newSetFromMap(
            new ConcurrentHashMap<SkyKey, Boolean>(
                EXPECTED_VISITED_SET_SIZE, .75f, DEFAULT_THREAD_COUNT));
    private final Set<SkyKey> dirtied =
        Collections.newSetFromMap(
            new ConcurrentHashMap<SkyKey, Boolean>(
                EXPECTED_VISITED_SET_SIZE, .75f, DEFAULT_THREAD_COUNT));
    private final boolean supportInterruptions;

    protected DirtyingNodeVisitor(
        QueryableGraph graph,
        DirtyTrackingProgressReceiver progressReceiver,
        InvalidationState state) {
      super(graph, progressReceiver, state);
      this.supportInterruptions = true;
    }

    /**
     * Use cases that do not require support for interruptibility can avoid unnecessary work by
     * passing {@code false} for {@param supportInterruptions}.
     */
    protected DirtyingNodeVisitor(
        QueryableGraph graph,
        DirtyTrackingProgressReceiver progressReceiver,
        InvalidationState state,
        ForkJoinPool forkJoinPool,
        boolean supportInterruptions) {
      super(graph, progressReceiver, state, forkJoinPool);
      this.supportInterruptions = supportInterruptions;
    }

    @Override
    protected boolean getSupportInterruptions() {
      return supportInterruptions;
    }

    @Override
    void visit(Iterable<SkyKey> keys, InvalidationType invalidationType) {
      Preconditions.checkState(invalidationType != InvalidationType.DELETED, keys);
      visit(keys, invalidationType, null);
    }

    /**
     * Queues a task to dirty the nodes named by {@param keys}. May be called from multiple threads.
     * It is possible that the same node is enqueued many times. However, we require that a node
     * is only actually marked dirty/changed once, with two exceptions:
     *
     * (1) If a node is marked dirty, it can subsequently be marked changed. This can occur if, for
     * instance, FileValue workspace/foo/foo.cc is marked dirty because FileValue workspace/foo is
     * marked changed (and every FileValue depends on its parent). Then FileValue
     * workspace/foo/foo.cc is itself changed (this can even happen on the same build).
     *
     * (2) If a node is going to be marked both dirty and changed, as, for example, in the previous
     * case if both workspace/foo/foo.cc and workspace/foo have been changed in the same build, the
     * thread marking workspace/foo/foo.cc dirty may race with the one marking it changed, and so
     * try to mark it dirty after it has already been marked changed. In that case, the
     * {@link NodeEntry} ignores the second marking.
     *
     * The invariant that we do not process a (SkyKey, InvalidationType) pair twice is enforced by
     * the {@link #changed} and {@link #dirtied} sets.
     *
     * The "invariant" is also enforced across builds by checking to see if the entry is already
     * marked changed, or if it is already marked dirty and we are just going to mark it dirty
     * again.
     *
     * If either of the above tests shows that we have already started a task to mark this entry
     * dirty/changed, or that it is already marked dirty/changed, we do not continue this task.
     */
    @ThreadSafe
    private void visit(
        Iterable<SkyKey> keys,
        final InvalidationType invalidationType,
        @Nullable SkyKey enqueueingKeyForExistenceCheck) {
      final boolean isChanged = (invalidationType == InvalidationType.CHANGED);
      Set<SkyKey> setToCheck = isChanged ? changed : dirtied;
      int size = Iterables.size(keys);
      ArrayList<SkyKey> keysToGet = new ArrayList<>(size);
      for (SkyKey key : keys) {
        if (setToCheck.add(key)) {
          Preconditions.checkState(
              !isChanged || key.functionName().getHermeticity() != FunctionHermeticity.HERMETIC,
              key);
          keysToGet.add(key);
        }
      }
      if (supportInterruptions) {
        for (SkyKey key : keysToGet) {
          pendingVisitations.add(Pair.of(key, invalidationType));
        }
      }
      final Map<SkyKey, ? extends ThinNodeEntry> entries;
      try {
        entries = graph.getBatch(null, Reason.INVALIDATION, keysToGet);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        // This can only happen if the main thread has been interrupted, and so the
        // AbstractQueueVisitor is shutting down. We haven't yet removed the pending visitations, so
        // we can resume next time.
        return;
      }
      if (enqueueingKeyForExistenceCheck != null && entries.size() != keysToGet.size()) {
        Set<SkyKey> missingKeys = Sets.difference(ImmutableSet.copyOf(keysToGet), entries.keySet());
        throw new IllegalStateException(
            String.format(
                "key(s) %s not in the graph, but enqueued for dirtying by %s",
                Iterables.limit(missingKeys, 10), enqueueingKeyForExistenceCheck));
      }
      for (final SkyKey key : keysToGet) {
        executor.execute(
            () -> {
              ThinNodeEntry entry = entries.get(key);

              if (entry == null) {
                if (supportInterruptions) {
                  pendingVisitations.remove(Pair.of(key, invalidationType));
                }
                return;
              }

              if (entry.isChanged() || (!isChanged && entry.isDirty())) {
                // If this node is already marked changed, or we are only marking this node
                // dirty, and it already is, move along.
                if (supportInterruptions) {
                  pendingVisitations.remove(Pair.of(key, invalidationType));
                }
                return;
              }

              // It is not safe to interrupt the logic from this point until the end of the method.
              // Any exception thrown should be unrecoverable.
              // This entry remains in the graph in this dirty state until it is re-evaluated.
              MarkedDirtyResult markedDirtyResult;
              try {
                markedDirtyResult = entry.markDirty(isChanged ? DirtyType.CHANGE : DirtyType.DIRTY);
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                // This can only happen if the main thread has been interrupted, and so the
                // AbstractQueueVisitor is shutting down. We haven't yet removed the pending
                // visitation, so we can resume next time.
                return;
              } catch (IllegalStateException e) {
                // Debugging for #10912.
                throw new IllegalStateException("Crash caused by " + key, e);
              }
              if (markedDirtyResult == null) {
                // Another thread has already dirtied this node. Don't do anything in this thread.
                if (supportInterruptions) {
                  pendingVisitations.remove(Pair.of(key, invalidationType));
                }
                return;
              }
              // Propagate dirtiness upwards and mark this node dirty/changed. Reverse deps should
              // only be marked dirty (because only a dependency of theirs has changed).
              visit(markedDirtyResult.getReverseDepsUnsafe(), InvalidationType.DIRTIED, key);

              progressReceiver.invalidated(key, EvaluationProgressReceiver.InvalidationState.DIRTY);
              // Remove the node from the set as the last operation.
              if (supportInterruptions) {
                pendingVisitations.remove(Pair.of(key, invalidationType));
              }
            });
      }
    }
  }
}
