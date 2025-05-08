// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;

import com.github.benmanes.caffeine.cache.Cache;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.concurrent.MultiThreadPoolsQuiescingExecutor;
import com.google.devtools.build.lib.concurrent.MultiThreadPoolsQuiescingExecutor.ThreadPoolType;
import com.google.devtools.build.lib.concurrent.QuiescingExecutor;
import com.google.devtools.build.skyframe.ParallelEvaluatorContext.RunnableMaker;
import com.google.devtools.build.skyframe.SkyFunction.Environment.ClassToInstanceMapSkyKeyComputeState;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/**
 * Threadpool manager for {@link ParallelEvaluator}. Wraps a {@link QuiescingExecutor} and keeps
 * track of pending nodes.
 */
class NodeEntryVisitor {
  private final QuiescingExecutor quiescingExecutor;
  private final AtomicBoolean preventNewEvaluations = new AtomicBoolean(false);
  private final Set<RuntimeException> crashes = Sets.newConcurrentHashSet();
  private final InflightTrackingProgressReceiver progressReceiver;

  /**
   * Function that allows this visitor to execute the appropriate {@link Runnable} when given a
   * {@link SkyKey} to evaluate.
   */
  private final RunnableMaker runnableMaker;

  private final RunnableMaker partialReevaluationRunnableMaker;
  private final Cache<SkyKey, SkyKeyComputeState> stateCache;

  /**
   * This state enum is used with {@link #partialReevaluationStates} to describe, for each {@link
   * SkyKey} opting into partial reevaluation, a state describing its partial reevaluation status.
   *
   * <p>Along with the values specified in the enum, the absence of an entry for a key in the map
   * means something: that no evaluation of the key's {@link SkyFunction} is currently happening.
   */
  enum PartialReevaluationState {
    /**
     * This state means that an evaluation of the key's {@link SkyFunction} has been called for via
     * either {@link #enqueueEvaluation} or {@link #enqueuePartialReevaluation}. The evaluation
     * might be currently underway, or may be pending in {@link #quiescingExecutor}, or is about to
     * be scheduled with {@link #quiescingExecutor}.
     */
    EVALUATING,

    /**
     * This state means that either {@link #enqueueEvaluation} or {@link
     * #enqueuePartialReevaluation} was called for the key while it was already in an {@link
     * #EVALUATING} state. Because it is unknown whether the "current" {@link SkyFunction}
     * evaluation (i.e. the one associated with its original {@code null} to {@code EVALUATING}
     * state transition) has been able to observe the newly completed signaling dep's value, the
     * signaled dep must be given another chance.
     *
     * <p>After that current evaluation completes, it will be scheduled again.
     */
    EVALUATING_SIGNALED,
  }

  private final ConcurrentHashMap<SkyKey, PartialReevaluationState> partialReevaluationStates =
      new ConcurrentHashMap<>();

  private class PartialReevaluationRunnableMaker implements RunnableMaker {
    @Override
    public Runnable make(SkyKey key) {
      Runnable inner = runnableMaker.make(key);
      return () -> {
        PartialReevaluationState state = PartialReevaluationState.EVALUATING;
        while (state == PartialReevaluationState.EVALUATING) {
          inner.run();
          state =
              partialReevaluationStates.compute(
                  key,
                  (k, s) -> {
                    checkNotNull(s, "Null state during evaluation: %s", k);
                    switch (s) {
                      case EVALUATING:
                        // Note that returning null from this compute function causes the entry to
                        // be removed from the map.
                        return null;
                      case EVALUATING_SIGNALED:
                        return PartialReevaluationState.EVALUATING;
                    }
                    throw new AssertionError(s);
                  });
        }
      };
    }
  }

  NodeEntryVisitor(
      QuiescingExecutor quiescingExecutor,
      InflightTrackingProgressReceiver progressReceiver,
      RunnableMaker runnableMaker,
      Cache<SkyKey, SkyKeyComputeState> stateCache) {
    this.quiescingExecutor = quiescingExecutor;
    this.progressReceiver = progressReceiver;
    this.runnableMaker = runnableMaker;
    this.partialReevaluationRunnableMaker = new PartialReevaluationRunnableMaker();
    this.stateCache = stateCache;
  }

  void waitForCompletion() throws InterruptedException {
    quiescingExecutor.awaitQuiescence(/* interruptWorkers= */ true);
  }

  /**
   * Enqueue {@code key} for evaluation.
   *
   * <p>This won't immediately enqueue {@code key} if {@code key.supportsPartialReevaluation()} and
   * a partial reevaluation is currently running, but that reevaluation will be immediately followed
   * by another reevaluation.
   */
  void enqueueEvaluation(SkyKey key, @Nullable SkyKey signalingDep) {
    if (key.supportsPartialReevaluation()) {
      enqueuePartialReevaluation(key, signalingDep);
    } else {
      innerEnqueueEvaluation(key, runnableMaker);
    }
  }

  /**
   * Registers a listener with all passed futures that causes the node to be re-enqueued when all
   * futures are completed.
   */
  void registerExternalDeps(SkyKey skyKey, NodeEntry entry, List<ListenableFuture<?>> externalDeps)
      throws InterruptedException {
    // Generally speaking, there is no ordering guarantee for listeners registered with a single
    // listenable future. If we used a listener here, there would be a potential race condition
    // between re-enqueuing the key and notifying the quiescing executor, in which case the executor
    // could shut down even though the work is not done yet. That would be bad.
    //
    // However, the whenAllComplete + run API guarantees that the Runnable is run before the
    // returned future completes, i.e., before the quiescing executor is notified.
    ListenableFuture<?> future =
        Futures.whenAllComplete(externalDeps)
            .run(
                () -> {
                  if (entry.signalDep(entry.getVersion(), null)) {
                    enqueueEvaluation(skyKey, null);
                  }
                },
                MoreExecutors.directExecutor());
    quiescingExecutor.dependOnFuture(future);
  }

  /**
   * Returns whether any new evaluations should be prevented.
   *
   * <p>If called from within node evaluation, the caller may use the return value to determine
   * whether it is responsible for throwing an exception to halt evaluation at the executor level.
   */
  boolean shouldPreventNewEvaluations() {
    return preventNewEvaluations.get();
  }

  /**
   * Stop any new evaluations from being enqueued. Returns whether this was the first thread to
   * request a halt.
   *
   * <p>If called from within node evaluation, the caller may use the return value to determine
   * whether it is responsible for throwing an exception to halt evaluation at the executor level.
   */
  boolean preventNewEvaluations() {
    return preventNewEvaluations.compareAndSet(false, true);
  }

  void noteCrash(RuntimeException e) {
    crashes.add(e);
  }

  Collection<RuntimeException> getCrashes() {
    return crashes;
  }

  @VisibleForTesting
  CountDownLatch getExceptionLatchForTestingOnly() {
    return quiescingExecutor.getExceptionLatchForTestingOnly();
  }

  private void enqueuePartialReevaluation(SkyKey key, @Nullable SkyKey signalingDep) {
    PartialReevaluationMailbox mailbox = getMailbox(key);
    if (signalingDep != null) {
      mailbox.signal(signalingDep);
    } else {
      mailbox.enqueuedNotByDeps();
    }

    PartialReevaluationState reevaluationState =
        partialReevaluationStates.compute(
            key,
            (k, s) ->
                s == null
                    ? PartialReevaluationState.EVALUATING
                    : PartialReevaluationState.EVALUATING_SIGNALED);
    if (reevaluationState.equals(PartialReevaluationState.EVALUATING)) {
      innerEnqueueEvaluation(key, partialReevaluationRunnableMaker);
    }
  }

  private PartialReevaluationMailbox getMailbox(SkyKey key) {
    return PartialReevaluationMailbox.from(
        (ClassToInstanceMapSkyKeyComputeState)
            stateCache.get(key, k -> new ClassToInstanceMapSkyKeyComputeState()));
  }

  private void innerEnqueueEvaluation(SkyKey key, RunnableMaker runnableMakerToUse) {
    if (shouldPreventNewEvaluations()) {
      // If an error happens in nokeep_going mode, we still want to mark these nodes as inflight,
      // otherwise cleanup will not happen properly.
      progressReceiver.enqueueAfterError(key);
      return;
    }
    progressReceiver.enqueueing(key);

    var runnable = runnableMakerToUse.make(key);
    if (quiescingExecutor
        instanceof MultiThreadPoolsQuiescingExecutor multiThreadPoolsQuiescingExecutor) {
      ThreadPoolType threadPoolType;
      if (key instanceof CPUHeavySkyKey) {
        threadPoolType = ThreadPoolType.CPU_HEAVY;
      } else if (multiThreadPoolsQuiescingExecutor.hasSeparatePoolForExecutionTasks()
          && key instanceof ExecutionPhaseSkyKey) {
        // Only possible with --experimental_merged_skyframe_analysis_execution.
        threadPoolType = ThreadPoolType.EXECUTION_PHASE;
      } else {
        threadPoolType = ThreadPoolType.REGULAR;
      }
      multiThreadPoolsQuiescingExecutor.execute(
          runnable,
          threadPoolType,
          /* shouldStallAwaitingSignal= */ key instanceof StallableSkykey);
    } else {
      quiescingExecutor.execute(runnable);
    }
  }
}
