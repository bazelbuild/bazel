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
package com.google.devtools.build.lib.concurrent;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.atomic.AtomicIntegerFieldUpdater;

/**
 * A {@link ForkJoinTask} representing an action executed by a {@link QuiescingExecutor}.
 *
 * <p>Avoids the wrapper allocation overhead of {@link AbstractQueueVisitor.WrappedRunnable} and
 * {@link ForkJoinTask#adapt} by integrating lifecycle management (remaining task counting, error
 * handling, fail-fast gating, and worker thread tracking) directly into the task itself.
 */
public abstract class QuiescingTask extends ForkJoinTask<Void>
    implements Runnable, Comparable<QuiescingTask> {
  private static final AtomicIntegerFieldUpdater<QuiescingTask> DECREMENTED_UPDATER =
      AtomicIntegerFieldUpdater.newUpdater(QuiescingTask.class, "decremented");

  private final AbstractQueueVisitor visitor;
  private volatile boolean ran;

  @SuppressWarnings("unused") // Accessed via DECREMENTED_UPDATER
  private volatile int decremented;

  protected QuiescingTask(AbstractQueueVisitor visitor) {
    this.visitor = checkNotNull(visitor);
  }

  public final AbstractQueueVisitor getQueueVisitor() {
    return visitor;
  }

  @Override
  public final Void getRawResult() {
    return null;
  }

  @Override
  protected final void setRawResult(Void mustBeNull) {}

  void decrementRemainingTasksOnce() {
    if (DECREMENTED_UPDATER.compareAndSet(this, 0, 1)) {
      visitor.decrementRemainingTasks();
    }
  }

  boolean hasRun() {
    return ran;
  }

  @CanIgnoreReturnValue
  @Override
  protected final boolean exec() {
    ran = true;
    Thread thread = null;
    boolean addedJob = false;
    try {
      thread = Thread.currentThread();
      visitor.addJob(thread);
      addedJob = true;
      if (visitor.blockNewActions()) {
        return true;
      }
      runCore();
    } catch (Throwable t) {
      visitor.maybeSaveUnhandledThrowable(t, /* markToStopJobs= */ true);
    } finally {
      try {
        if (thread != null && addedJob) {
          visitor.removeJob(thread);
        }
      } finally {
        decrementRemainingTasksOnce();
      }
    }
    return true;
  }

  @Override
  public final void run() {
    exec();
  }

  /**
   * Scheduling priority tier when queued in a {@link java.util.concurrent.PriorityBlockingQueue}
   * (earlier enum constants run first).
   */
  public enum PriorityTier {
    /**
     * Tasks resuming or assisting work already in progress (including raw {@link Runnable}s without
     * a graph depth) so they can finish and release held state.
     */
    IN_PROGRESS,
    /** Default tier for unstarted tasks, ordered deepest-first via {@link #priorityDepth()}. */
    DEPTH_ORDERED,
  }

  /**
   * Priority tier when scheduled in a priority queue.
   *
   * <p>Defaults to {@link PriorityTier#DEPTH_ORDERED} so tasks that do not specify a tier are
   * treated as unstarted work and do not preempt {@link PriorityTier#IN_PROGRESS} tasks.
   */
  public PriorityTier priorityTier() {
    return PriorityTier.DEPTH_ORDERED;
  }

  /**
   * Graph depth (deeper/larger runs first within the same priority tier).
   *
   * <p>Defaults to {@code 0} (shallowest root depth) so tasks without a recorded graph depth do not
   * preempt deeper descendants in {@link PriorityTier#DEPTH_ORDERED}, and tasks in {@link
   * PriorityTier#IN_PROGRESS} share equal depth.
   */
  public int priorityDepth() {
    return 0;
  }

  @Override
  public final int compareTo(QuiescingTask other) {
    int tierCmp = this.priorityTier().compareTo(other.priorityTier());
    if (tierCmp != 0) {
      return tierCmp;
    }
    return Integer.compare(other.priorityDepth(), this.priorityDepth());
  }

  /** The core work of this task. */
  public abstract void runCore() throws Exception;
}
