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
import com.google.common.base.Function;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.ErrorClassifier;
import com.google.devtools.build.lib.concurrent.ForkJoinQuiescingExecutor;
import com.google.devtools.build.lib.concurrent.QuiescingExecutor;
import java.util.Collection;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
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
  private final Set<SkyKey> inflightNodes = Sets.newConcurrentHashSet();
  private final Set<RuntimeException> crashes = Sets.newConcurrentHashSet();
  private final DirtyKeyTracker dirtyKeyTracker;
  private final EvaluationProgressReceiver progressReceiver;
  /**
   * Function that allows this visitor to execute the appropriate {@link Runnable} when given a
   * {@link SkyKey} to evaluate.
   */
  private final Function<SkyKey, Runnable> runnableMaker;

  NodeEntryVisitor(
      ForkJoinPool forkJoinPool,
      DirtyKeyTracker dirtyKeyTracker,
      EvaluationProgressReceiver progressReceiver,
      Function<SkyKey, Runnable> runnableMaker) {
    quiescingExecutor =
        new ForkJoinQuiescingExecutor(forkJoinPool, NODE_ENTRY_VISITOR_ERROR_CLASSIFIER);
    this.dirtyKeyTracker = dirtyKeyTracker;
    this.progressReceiver = progressReceiver;
    this.runnableMaker = runnableMaker;
  }

  NodeEntryVisitor(
      int threadCount,
      DirtyKeyTracker dirtyKeyTracker,
      EvaluationProgressReceiver progressReceiver,
      Function<SkyKey, Runnable> runnableMaker) {
    quiescingExecutor =
        new AbstractQueueVisitor(
            /*concurrent*/ true,
            threadCount,
            /*keepAliveTime=*/ 1,
            TimeUnit.SECONDS,
            /*failFastOnException*/ true,
            "skyframe-evaluator",
            NODE_ENTRY_VISITOR_ERROR_CLASSIFIER);
    this.dirtyKeyTracker = dirtyKeyTracker;
    this.progressReceiver = progressReceiver;
    this.runnableMaker = runnableMaker;
  }

  void waitForCompletion() throws InterruptedException {
    quiescingExecutor.awaitQuiescence(/*interruptWorkers=*/ true);
  }

  void enqueueEvaluation(SkyKey key) {
    // We unconditionally add the key to the set of in-flight nodes because even if evaluation is
    // never scheduled we still want to remove the previously created NodeEntry from the graph.
    // Otherwise we would leave the graph in a weird state (wasteful garbage in the best case and
    // inconsistent in the worst case).
    boolean newlyEnqueued = inflightNodes.add(key);
    // All nodes enqueued for evaluation will be either verified clean, re-evaluated, or cleaned
    // up after being in-flight when an error happens in nokeep_going mode or in the event of an
    // interrupt. In any of these cases, they won't be dirty anymore.
    if (newlyEnqueued) {
      dirtyKeyTracker.notDirty(key);
    }
    if (preventNewEvaluations.get()) {
      return;
    }
    if (newlyEnqueued && progressReceiver != null) {
      progressReceiver.enqueueing(key);
    }
    quiescingExecutor.execute(runnableMaker.apply(key));
  }

  /**
   * Stop any new evaluations from being enqueued. Returns whether this was the first thread to
   * request a halt. If true, this thread should proceed to throw an exception. If false, another
   * thread already requested a halt and will throw an exception, and so this thread can simply end.
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

  void notifyDone(SkyKey key) {
    inflightNodes.remove(key);
  }

  boolean isInflight(SkyKey key) {
    return inflightNodes.contains(key);
  }

  Set<SkyKey> getInflightNodes() {
    return inflightNodes;
  }

  @VisibleForTesting
  CountDownLatch getExceptionLatchForTestingOnly() {
    return quiescingExecutor.getExceptionLatchForTestingOnly();
  }
}
