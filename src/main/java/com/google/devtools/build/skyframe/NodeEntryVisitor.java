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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.concurrent.ErrorClassifier;
import com.google.devtools.build.lib.concurrent.MultiThreadPoolsQuiescingExecutor;
import com.google.devtools.build.lib.concurrent.MultiThreadPoolsQuiescingExecutor.ThreadPoolType;
import com.google.devtools.build.lib.concurrent.QuiescingExecutor;
import com.google.devtools.build.skyframe.ParallelEvaluatorContext.RunnableMaker;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Threadpool manager for {@link ParallelEvaluator}. Wraps a {@link QuiescingExecutor} and keeps
 * track of pending nodes.
 */
class NodeEntryVisitor {
  static final ErrorClassifier NODE_ENTRY_VISITOR_ERROR_CLASSIFIER =
      new ErrorClassifier() {
        @Override
        protected ErrorClassification classifyException(Exception e) {
          if (e instanceof SchedulerException) {
            return ErrorClassification.CRITICAL;
          }
          if (e instanceof RuntimeException) {
            // We treat non-SchedulerException RuntimeExceptions as more severe than
            // SchedulerExceptions so that AbstractQueueVisitor will propagate instances of the
            // former. They indicate actual Blaze bugs, rather than normal Skyframe evaluation
            // control flow.
            return ErrorClassification.CRITICAL_AND_LOG;
          }
          return ErrorClassification.NOT_CRITICAL;
        }
      };

  private final QuiescingExecutor quiescingExecutor;
  private final AtomicBoolean preventNewEvaluations = new AtomicBoolean(false);
  private final Set<RuntimeException> crashes = Sets.newConcurrentHashSet();
  private final DirtyTrackingProgressReceiver progressReceiver;
  /**
   * Function that allows this visitor to execute the appropriate {@link Runnable} when given a
   * {@link SkyKey} to evaluate.
   */
  private final RunnableMaker runnableMaker;

  NodeEntryVisitor(
      QuiescingExecutor quiescingExecutor,
      DirtyTrackingProgressReceiver progressReceiver,
      RunnableMaker runnableMaker) {
    this.quiescingExecutor = quiescingExecutor;
    this.progressReceiver = progressReceiver;
    this.runnableMaker = runnableMaker;
  }

  void waitForCompletion() throws InterruptedException {
    quiescingExecutor.awaitQuiescence(/*interruptWorkers=*/ true);
  }

  /**
   * Enqueue {@code key} for evaluation, at {@code evaluationPriority} if this visitor is using a
   * priority queue.
   *
   * <p>{@code evaluationPriority} is used to minimize evaluation "sprawl": inefficiencies coming
   * from incompletely evaluating many nodes, versus focusing on finishing the evaluation of nodes
   * that have already started evaluating. Sprawl can be expensive because an incompletely evaluated
   * node keeps state in Skyframe, and often in external caches, that uses memory.
   *
   * <p>In general, {@code evaluationPriority} should be higher when restarting a node that has
   * already started evaluation, and lower when enqueueing a node that no other tasks depend on.
   * Setting {@code evaluationPriority} to the same value for all children of a parent has good
   * results experimentally, since it prioritizes batches of work that can be used together.
   * Similarly, prioritizing deeper nodes (depth-first search of the evaluation graph) also has good
   * results experimentally, since it minimizes sprawl.
   */
  void enqueueEvaluation(SkyKey key, int evaluationPriority) {
    if (shouldPreventNewEvaluations()) {
      // If an error happens in nokeep_going mode, we still want to mark these nodes as inflight,
      // otherwise cleanup will not happen properly.
      progressReceiver.enqueueAfterError(key);
      return;
    }
    progressReceiver.enqueueing(key);
    if (quiescingExecutor instanceof MultiThreadPoolsQuiescingExecutor) {
      ThreadPoolType threadPoolType =
          key instanceof CPUHeavySkyKey ? ThreadPoolType.CPU_HEAVY : ThreadPoolType.REGULAR;
      ((MultiThreadPoolsQuiescingExecutor) quiescingExecutor)
          .execute(runnableMaker.make(key, evaluationPriority), threadPoolType);
    } else {
      quiescingExecutor.execute(runnableMaker.make(key, evaluationPriority));
    }
  }

  /**
   * Registers a listener with all passed futures that causes the node to be re-enqueued (at the
   * given {@code evaluationPriority}) when all futures are completed.
   */
  void registerExternalDeps(
      SkyKey skyKey,
      NodeEntry entry,
      List<ListenableFuture<?>> externalDeps,
      int evaluationPriority)
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
                    enqueueEvaluation(skyKey, evaluationPriority);
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
}
