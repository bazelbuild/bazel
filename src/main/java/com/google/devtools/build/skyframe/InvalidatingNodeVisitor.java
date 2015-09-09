// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Pair;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ThreadPoolExecutor;
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
public abstract class InvalidatingNodeVisitor<TGraph extends ThinNodeQueryableGraph>
    extends AbstractQueueVisitor {

  // Default thread count is equal to the number of cores to exploit
  // that level of hardware parallelism, since invalidation should be CPU-bound.
  // We may consider increasing this in the future.
  private static final int DEFAULT_THREAD_COUNT = Runtime.getRuntime().availableProcessors();

  private static final boolean MUST_EXIST = true;

  protected final TGraph graph;
  @Nullable protected final EvaluationProgressReceiver invalidationReceiver;
  protected final DirtyKeyTracker dirtyKeyTracker;
  // Aliased to InvalidationState.pendingVisitations.
  protected final Set<Pair<SkyKey, InvalidationType>> pendingVisitations;

  protected InvalidatingNodeVisitor(
      TGraph graph,
      @Nullable EvaluationProgressReceiver invalidationReceiver,
      InvalidationState state,
      DirtyKeyTracker dirtyKeyTracker) {
    this(graph, invalidationReceiver, state, dirtyKeyTracker, EXECUTOR_FACTORY);
  }

  protected InvalidatingNodeVisitor(
      TGraph graph,
      @Nullable EvaluationProgressReceiver invalidationReceiver,
      InvalidationState state,
      DirtyKeyTracker dirtyKeyTracker,
      Function<ThreadPoolExecutorParams, ThreadPoolExecutor> executorFactory) {
    super(/*concurrent=*/true,
        /*corePoolSize=*/DEFAULT_THREAD_COUNT,
        /*maxPoolSize=*/DEFAULT_THREAD_COUNT,
        /*keepAliveTime=*/1,
        /*units=*/TimeUnit.SECONDS,
        /*failFastOnException=*/true,
        /*failFastOnInterrupt=*/true,
        "skyframe-invalidator",
        executorFactory);
    this.graph = Preconditions.checkNotNull(graph);
    this.invalidationReceiver = invalidationReceiver;
    this.dirtyKeyTracker = Preconditions.checkNotNull(dirtyKeyTracker);
    this.pendingVisitations = state.pendingValues;
  }

  /**
   * Initiates visitation and waits for completion.
   */
  void run() throws InterruptedException {
    // Make a copy to avoid concurrent modification confusing us as to which nodes were passed by
    // the caller, and which are added by other threads during the run. Since no tasks have been
    // started yet (the queueDirtying calls start them), this is thread-safe.
    for (Pair<SkyKey, InvalidationType> visitData : ImmutableList.copyOf(pendingVisitations)) {
      // The caller may have specified non-existent SkyKeys, or there may be stale SkyKeys in
      // pendingVisitations that have already been deleted. In both these cases, the nodes will not
      // exist in the graph, so we must be tolerant of that case.
      visit(visitData.first, visitData.second, !MUST_EXIST);
    }
    work(/*failFastOnInterrupt=*/true);
    Preconditions.checkState(pendingVisitations.isEmpty(),
        "All dirty nodes should have been processed: %s", pendingVisitations);
  }

  protected abstract long count();

  protected void informInvalidationReceiver(SkyKey key,
      EvaluationProgressReceiver.InvalidationState state) {
    if (invalidationReceiver != null) {
      invalidationReceiver.invalidated(key, state);
    }
  }

  /**
   * Enqueues a node for invalidation.
   */
  @ThreadSafe
  abstract void visit(SkyKey key, InvalidationType second, boolean mustExist);

  @VisibleForTesting
  enum InvalidationType {
    /**
     * The node is dirty and must be recomputed.
     */
    CHANGED,
    /**
     * The node is dirty, but may be marked clean later during change pruning.
     */
    DIRTIED,
    /**
     * The node is deleted.
     */
    DELETED;
  }

  /**
   * Invalidation state object that keeps track of which nodes need to be invalidated, but have not
   * been dirtied/deleted yet. This supports interrupts - by only deleting a node from this set
   * when all its parents have been invalidated, we ensure that no information is lost when an
   * interrupt comes in.
   */
  static class InvalidationState {
    private final Set<Pair<SkyKey, InvalidationType>> pendingValues = Sets.newConcurrentHashSet();
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

  /**
   * A node-deleting implementation.
   */
  static class DeletingNodeVisitor extends InvalidatingNodeVisitor<DirtiableGraph> {

    private final Set<SkyKey> visitedValues = Sets.newConcurrentHashSet();
    private final boolean traverseGraph;

    protected DeletingNodeVisitor(DirtiableGraph graph,
        EvaluationProgressReceiver invalidationReceiver, InvalidationState state,
        boolean traverseGraph, DirtyKeyTracker dirtyKeyTracker) {
      super(graph, invalidationReceiver, state, dirtyKeyTracker);
      this.traverseGraph = traverseGraph;
    }

    @Override
    protected long count() {
      return visitedValues.size();
    }

    @Override
    public void visit(final SkyKey key, InvalidationType invalidationType, boolean mustExist) {
      Preconditions.checkState(invalidationType == InvalidationType.DELETED, key);
      if (!visitedValues.add(key)) {
        return;
      }
      final Pair<SkyKey, InvalidationType> invalidationPair = Pair.of(key, invalidationType);
      pendingVisitations.add(invalidationPair);
      enqueue(
          new Runnable() {
            @Override
            public void run() {
              NodeEntry entry = graph.get(key);
              if (entry == null) {
                pendingVisitations.remove(invalidationPair);
                return;
              }

              if (traverseGraph) {
                // Propagate deletion upwards.
                for (SkyKey reverseDep : entry.getReverseDeps()) {
                  visit(reverseDep, InvalidationType.DELETED, !MUST_EXIST);
                }
                Iterable<SkyKey> directDeps =
                    entry.isDone() ? entry.getDirectDeps() : entry.getTemporaryDirectDeps();
                // Unregister this node from direct deps, since reverse dep edges cannot point to
                // non-existent nodes.
                for (SkyKey directDep : directDeps) {
                  NodeEntry dep = graph.get(directDep);
                  if (dep != null) {
                    dep.removeReverseDep(key);
                  }
                }
              }
              // Allow custom key-specific logic to update dirtiness status.
              informInvalidationReceiver(key, EvaluationProgressReceiver.InvalidationState.DELETED);
              // Actually remove the node.
              graph.remove(key);
              dirtyKeyTracker.notDirty(key);

              // Remove the node from the set as the last operation.
              pendingVisitations.remove(invalidationPair);
            }
          });
    }
  }

  /**
   * A node-dirtying implementation.
   */
  static class DirtyingNodeVisitor extends InvalidatingNodeVisitor<ThinNodeQueryableGraph> {

    private final Set<Pair<SkyKey, InvalidationType>> visited = Sets.newConcurrentHashSet();

    protected DirtyingNodeVisitor(
        ThinNodeQueryableGraph graph,
        EvaluationProgressReceiver invalidationReceiver,
        InvalidationState state,
        DirtyKeyTracker dirtyKeyTracker,
        Function<ThreadPoolExecutorParams, ThreadPoolExecutor> executorFactory) {
      super(graph, invalidationReceiver, state, dirtyKeyTracker, executorFactory);
    }

    @Override
    protected long count() {
      return visited.size();
    }

    /**
     * Queues a task to dirty the node named by {@code key}. May be called from multiple threads.
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
     * the {@link #visited} set.
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
    public void visit(final SkyKey key, final InvalidationType invalidationType,
        final boolean mustExist) {
      Preconditions.checkState(invalidationType != InvalidationType.DELETED, key);
      final boolean isChanged = (invalidationType == InvalidationType.CHANGED);
      final Pair<SkyKey, InvalidationType> invalidationPair = Pair.of(key, invalidationType);
      if (!visited.add(invalidationPair)) {
        return;
      }
      pendingVisitations.add(invalidationPair);
      enqueue(
          new Runnable() {
            @Override
            public void run() {
              ThinNodeEntry entry = graph.get(key);

              if (entry == null) {
                Preconditions.checkState(
                    !mustExist,
                    "%s does not exist in the graph but was enqueued for dirtying by another node",
                    key);
                pendingVisitations.remove(invalidationPair);
                return;
              }

              if (entry.isChanged() || (!isChanged && entry.isDirty())) {
                // If this node is already marked changed, or we are only marking this node
                // dirty, and it already is, move along.
                pendingVisitations.remove(invalidationPair);
                return;
              }

              // This entry remains in the graph in this dirty state until it is re-evaluated.
              Iterable<SkyKey> deps = entry.markDirty(isChanged);
              // It is not safe to interrupt the logic from this point until the end of the method.
              // Any exception thrown should be unrecoverable.
              if (deps == null) {
                // Another thread has already dirtied this node. Don't do anything in this thread.
                pendingVisitations.remove(invalidationPair);
                return;
              }
              // Propagate dirtiness upwards and mark this node dirty/changed. Reverse deps
              // should only be marked dirty (because only a dependency of theirs has changed).
              for (SkyKey reverseDep : entry.getReverseDeps()) {
                visit(reverseDep, InvalidationType.DIRTIED, MUST_EXIST);
              }

              // Remove this node as a reverse dep from its children, since we have reset it and
              // it no longer lists its children as direct deps.
              Map<SkyKey, ? extends ThinNodeEntry> children = graph.getBatch(deps);
              if (children.size() != Iterables.size(deps)) {
                Set<SkyKey> depsSet = ImmutableSet.copyOf(deps);
                throw new IllegalStateException(
                    "Mismatch in getBatch: "
                        + key
                        + ", "
                        + entry
                        + "\n"
                        + Sets.difference(depsSet, children.keySet())
                        + "\n"
                        + Sets.difference(children.keySet(), depsSet));
              }
              for (ThinNodeEntry child : children.values()) {
                child.removeReverseDep(key);
              }

              informInvalidationReceiver(key, EvaluationProgressReceiver.InvalidationState.DIRTY);
              dirtyKeyTracker.dirty(key);
              // Remove the node from the set as the last operation.
              pendingVisitations.remove(invalidationPair);
            }
          });
    }
  }
}
