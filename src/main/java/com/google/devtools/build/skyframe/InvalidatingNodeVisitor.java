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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.ErrorClassifier;
import com.google.devtools.build.lib.concurrent.ExecutorParams;
import com.google.devtools.build.lib.concurrent.ForkJoinQuiescingExecutor;
import com.google.devtools.build.lib.concurrent.QuiescingExecutor;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.ThinNodeEntry.MarkedDirtyResult;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
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
public abstract class InvalidatingNodeVisitor<TGraph extends ThinNodeQueryableGraph> {

  // Default thread count is equal to the number of cores to exploit
  // that level of hardware parallelism, since invalidation should be CPU-bound.
  // We may consider increasing this in the future.
  private static final int DEFAULT_THREAD_COUNT = Runtime.getRuntime().availableProcessors();
  private static final int EXPECTED_PENDING_SET_SIZE = DEFAULT_THREAD_COUNT * 8;
  private static final int EXPECTED_VISITED_SET_SIZE = 1024;

  private static final boolean MUST_EXIST = true;

  private static final ErrorClassifier errorClassifier =
      new ErrorClassifier() {
        @Override
        protected ErrorClassification classifyException(Exception e) {
          return e instanceof RuntimeException
              ? ErrorClassification.CRITICAL_AND_LOG
              : ErrorClassification.NOT_CRITICAL;
        }
      };

  protected final TGraph graph;
  @Nullable protected final EvaluationProgressReceiver invalidationReceiver;
  protected final DirtyKeyTracker dirtyKeyTracker;
  // Aliased to InvalidationState.pendingVisitations.
  protected final Set<Pair<SkyKey, InvalidationType>> pendingVisitations;
  protected final QuiescingExecutor executor;

  protected InvalidatingNodeVisitor(
      TGraph graph,
      @Nullable EvaluationProgressReceiver invalidationReceiver,
      InvalidationState state,
      DirtyKeyTracker dirtyKeyTracker) {
    this(
        graph, invalidationReceiver, state, dirtyKeyTracker, AbstractQueueVisitor.EXECUTOR_FACTORY);
  }

  protected InvalidatingNodeVisitor(
      TGraph graph,
      @Nullable EvaluationProgressReceiver invalidationReceiver,
      InvalidationState state,
      DirtyKeyTracker dirtyKeyTracker,
      Function<ExecutorParams, ? extends ExecutorService> executorFactory) {
    this.executor =
        new AbstractQueueVisitor(
            /*concurrent=*/ true,
            /*parallelism=*/ DEFAULT_THREAD_COUNT,
            /*keepAliveTime=*/ 1,
            /*units=*/ TimeUnit.SECONDS,
            /*failFastOnException=*/ true,
            /*failFastOnInterrupt=*/ true,
            "skyframe-invalidator",
            executorFactory,
            errorClassifier);
    this.graph = Preconditions.checkNotNull(graph);
    this.invalidationReceiver = invalidationReceiver;
    this.dirtyKeyTracker = Preconditions.checkNotNull(dirtyKeyTracker);
    this.pendingVisitations = state.pendingValues;
  }

  protected InvalidatingNodeVisitor(
      TGraph graph,
      @Nullable EvaluationProgressReceiver invalidationReceiver,
      InvalidationState state,
      DirtyKeyTracker dirtyKeyTracker,
      ForkJoinPool forkJoinPool) {
    this.executor = new ForkJoinQuiescingExecutor(forkJoinPool, errorClassifier);
    this.graph = Preconditions.checkNotNull(graph);
    this.invalidationReceiver = invalidationReceiver;
    this.dirtyKeyTracker = Preconditions.checkNotNull(dirtyKeyTracker);
    this.pendingVisitations = state.pendingValues;
  }

  /** Initiates visitation and waits for completion. */
  void run() throws InterruptedException {
    // Make a copy to avoid concurrent modification confusing us as to which nodes were passed by
    // the caller, and which are added by other threads during the run. Since no tasks have been
    // started yet (the queueDirtying calls start them), this is thread-safe.
    for (final Pair<SkyKey, InvalidationType> visitData :
        ImmutableList.copyOf(pendingVisitations)) {
      // The caller may have specified non-existent SkyKeys, or there may be stale SkyKeys in
      // pendingVisitations that have already been deleted. In both these cases, the nodes will not
      // exist in the graph, so we must be tolerant of that case.
      executor.execute(new Runnable() {
        @Override
        public void run() {
          visit(ImmutableList.of(visitData.first), visitData.second, !MUST_EXIST);
        }
      });
    }
    executor.awaitQuiescence(/*interruptWorkers=*/ true);

    // Note: implementations that do not support interruption also do not update pendingVisitations.
    Preconditions.checkState(!getSupportInterruptions() || pendingVisitations.isEmpty(),
        "All dirty nodes should have been processed: %s", pendingVisitations);
  }

  protected abstract boolean getSupportInterruptions();

  @VisibleForTesting
  public CountDownLatch getInterruptionLatchForTestingOnly() {
    return executor.getInterruptionLatchForTestingOnly();
  }

  protected void informInvalidationReceiver(SkyKey key,
      EvaluationProgressReceiver.InvalidationState state) {
    if (invalidationReceiver != null) {
      invalidationReceiver.invalidated(key, state);
    }
  }

  /** Enqueues nodes for invalidation. */
  @ThreadSafe
  abstract void visit(Iterable<SkyKey> keys, InvalidationType second, boolean mustExist);

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

  public static class DirtyingInvalidationState extends InvalidationState {
    public DirtyingInvalidationState() {
      super(InvalidationType.CHANGED);
    }
  }

  static class DeletingInvalidationState extends InvalidationState {
    public DeletingInvalidationState() {
      super(InvalidationType.DELETED);
    }
  }

  /** A node-deleting implementation. */
  static class DeletingNodeVisitor extends InvalidatingNodeVisitor<DirtiableGraph> {

    private final Set<SkyKey> visited = Sets.newConcurrentHashSet();
    private final boolean traverseGraph;

    protected DeletingNodeVisitor(DirtiableGraph graph,
        EvaluationProgressReceiver invalidationReceiver, InvalidationState state,
        boolean traverseGraph, DirtyKeyTracker dirtyKeyTracker) {
      super(graph, invalidationReceiver, state, dirtyKeyTracker);
      this.traverseGraph = traverseGraph;
    }

    @Override
    protected boolean getSupportInterruptions() {
      return true;
    }

    @Override
    public void visit(Iterable<SkyKey> keys, InvalidationType invalidationType, boolean mustExist) {
      Preconditions.checkState(invalidationType == InvalidationType.DELETED, keys);
      Builder<SkyKey> unvisitedKeysBuilder = ImmutableList.builder();
      for (SkyKey key : keys) {
        if (visited.add(key)) {
          unvisitedKeysBuilder.add(key);
        }
      }
      ImmutableList<SkyKey> unvisitedKeys = unvisitedKeysBuilder.build();
      for (SkyKey key : unvisitedKeys) {
        pendingVisitations.add(Pair.of(key, InvalidationType.DELETED));
      }
      final Map<SkyKey, NodeEntry> entries = graph.getBatch(unvisitedKeys);
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
                  visit(entry.getReverseDeps(), InvalidationType.DELETED, !MUST_EXIST);

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
                      entry.isDone() ? ImmutableSet.<SkyKey>of() : entry.getTemporaryDirectDeps();
                  Iterable<SkyKey> directDeps =
                      entry.isDone()
                          ? entry.getDirectDeps()
                          : entry.getAllDirectDepsForIncompleteNode();
                  Map<SkyKey, NodeEntry> depMap = graph.getBatch(directDeps);
                  for (Map.Entry<SkyKey, NodeEntry> directDepEntry : depMap.entrySet()) {
                    NodeEntry dep = directDepEntry.getValue();
                    if (dep != null) {
                      if (dep.isDone() || !signalingDeps.contains(directDepEntry.getKey())) {
                        dep.removeReverseDep(key);
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
                informInvalidationReceiver(
                    key, EvaluationProgressReceiver.InvalidationState.DELETED);
                // Actually remove the node.
                graph.remove(key);
                dirtyKeyTracker.notDirty(key);

                // Remove the node from the set as the last operation.
                pendingVisitations.remove(invalidationPair);
              }
            });
      }
    }
  }

  /** A node-dirtying implementation. */
  static class DirtyingNodeVisitor extends InvalidatingNodeVisitor<ThinNodeQueryableGraph> {

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
        ThinNodeQueryableGraph graph,
        EvaluationProgressReceiver invalidationReceiver,
        InvalidationState state,
        DirtyKeyTracker dirtyKeyTracker,
        Function<ExecutorParams, ? extends ExecutorService> executorFactory) {
      super(graph, invalidationReceiver, state, dirtyKeyTracker, executorFactory);
      this.supportInterruptions = true;
    }

    /**
     * Use cases that do not require support for interruptibility can avoid unnecessary work by
     * passing {@code false} for {@param supportInterruptions}.
     */
    protected DirtyingNodeVisitor(
        ThinNodeQueryableGraph graph,
        EvaluationProgressReceiver invalidationReceiver,
        InvalidationState state,
        DirtyKeyTracker dirtyKeyTracker,
        ForkJoinPool forkJoinPool,
        boolean supportInterruptions) {
      super(graph, invalidationReceiver, state, dirtyKeyTracker, forkJoinPool);
      this.supportInterruptions = supportInterruptions;
    }

    @Override
    protected boolean getSupportInterruptions() {
      return supportInterruptions;
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
    @Override
    @ThreadSafe
    public void visit(
        Iterable<SkyKey> keys, final InvalidationType invalidationType, final boolean mustExist) {
      Preconditions.checkState(invalidationType != InvalidationType.DELETED, keys);
      final boolean isChanged = (invalidationType == InvalidationType.CHANGED);
      Set<SkyKey> setToCheck = isChanged ? changed : dirtied;
      int size = Iterables.size(keys);
      ArrayList<SkyKey> keysToGet = new ArrayList<>(size);
      for (SkyKey key : keys) {
        if (setToCheck.add(key)) {
          keysToGet.add(key);
        }
      }
      if (supportInterruptions) {
        for (SkyKey key : keysToGet) {
          pendingVisitations.add(Pair.of(key, invalidationType));
        }
      }
      final Map<SkyKey, ? extends ThinNodeEntry> entries = graph.getBatch(keysToGet);
      for (final SkyKey key : keysToGet) {
        executor.execute(
            new Runnable() {
              @Override
              public void run() {
                ThinNodeEntry entry = entries.get(key);

                if (entry == null) {
                  Preconditions.checkState(
                      !mustExist,
                      "%s does not exist in the graph but was enqueued for dirtying by another "
                          + "node",
                      key);
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

                // It is not safe to interrupt the logic from this point until the end of the
                // method.
                // Any exception thrown should be unrecoverable.
                // This entry remains in the graph in this dirty state until it is re-evaluated.
                MarkedDirtyResult markedDirtyResult = entry.markDirty(isChanged);
                if (markedDirtyResult == null) {
                  // Another thread has already dirtied this node. Don't do anything in this thread.
                  if (supportInterruptions) {
                    pendingVisitations.remove(Pair.of(key, invalidationType));
                  }
                  return;
                }
                // Propagate dirtiness upwards and mark this node dirty/changed. Reverse deps should
                // only be marked dirty (because only a dependency of theirs has changed).
                visit(
                    markedDirtyResult.getReverseDepsUnsafe(), InvalidationType.DIRTIED, MUST_EXIST);

                informInvalidationReceiver(key, EvaluationProgressReceiver.InvalidationState.DIRTY);
                dirtyKeyTracker.dirty(key);
                // Remove the node from the set as the last operation.
                if (supportInterruptions) {
                  pendingVisitations.remove(Pair.of(key, invalidationType));
                }
              }
            });
      }
    }
  }
}
