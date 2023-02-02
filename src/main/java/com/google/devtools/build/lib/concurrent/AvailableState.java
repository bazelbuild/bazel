// Copyright 2022 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.MoreObjects.toStringHelper;
import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.annotations.VisibleForTesting;

/** Tracks available resources and tasks for {@link TieredPriorityExecutor}. */
final class AvailableState {
  private static final int INT16_MASK = 0xFFFF;

  private static final int THREADS_MASK = INT16_MASK;
  private static final int ONE_THREAD = 1;

  private static final int CPU_PERMITS_BIT_OFFSET = 16;
  private static final int ONE_CPU_PERMIT = 1 << CPU_PERMITS_BIT_OFFSET;

  /**
   * The threads and CPU permits packed together to reduce cost.
   *
   * <ul>
   *   <li><i>threads</i>, stored in the lower 16 bits, is the number of threads available.
   *   <li><i>cpuPermits</i>, stored in the upper 16 bits, is the number of CPU permits available.
   * </ul>
   *
   * <p>It's often possible to combine multiple operations into one, for example, adding both a
   * thread and a permit can be achieved by adding {@link #ONE_THREAD} and {@link #ONE_CPU_PERMIT}.
   * In contrast, if we used {@code short}s here, they would need to first be expanded into {@code
   * int}s for arithmetic operations, then cast back into {@code short}s anyway, resulting in more
   * {@code int} operations to accomplish the same goal.
   *
   * <p>Both <i>threads</i> and <i>cpuPermits</i> values are set negative during cancellation. This
   * allows resource tokens to be returned but doesn't allow any to be acquired, while allowing the
   * counts to be tracked for quiescence.
   */
  private int threadsAndCpuPermits;

  /** The number of non-CPU-heavy tasks in the queue. */
  private int tasks;
  /** The number of CPU-heavy tasks in the queue. */
  private int cpuHeavyTasks;

  AvailableState(int threads, int cpuPermits) {
    checkArgument(threads <= INT16_MASK, threads);
    checkArgument(cpuPermits <= INT16_MASK, cpuPermits);
    this.threadsAndCpuPermits = (cpuPermits << CPU_PERMITS_BIT_OFFSET) | threads;
  }

  /**
   * Constructs a default state to serve as the {@code out} argument of methods below.
   *
   * <p>The transition methods are called from a loop and shouldn't produce per-iteration garbage.
   */
  AvailableState() {}

  @VisibleForTesting
  int threads() {
    // Note: this extends the sign from 16 to 32 bits. Negative numbers are used for cancellation.
    return (threadsAndCpuPermits << 16) >> 16;
  }

  @VisibleForTesting
  int cpuPermits() {
    // Note: this preserves the sign. CPU permits can be negative under cancellation.
    return threadsAndCpuPermits >> CPU_PERMITS_BIT_OFFSET;
  }

  @VisibleForTesting
  int tasksForTesting() {
    return tasks;
  }

  @VisibleForTesting
  int cpuHeavyTasksForTesting() {
    return cpuHeavyTasks;
  }

  boolean isCancelled() {
    return tasks < 0;
  }

  /**
   * Outputs a state that blocks future task processing by making all resources unavailable.
   *
   * <p>This is implemented by subtracting {@code (threads+1)} and {@code (cpuPermits+1)} from
   * respective resources so they can still be tracked for quiescence. The resulting state is such
   * that {@code threads=-1, cpuPermits=-1} is quiescent. Under cancellation, resources may be
   * returned but none can be acquired.
   */
  void cancel(int threads, int cpuPermits, AvailableState out) {
    int targetThreads = threads() - (threads + 1);
    int targetCpuPermits = cpuPermits() - (cpuPermits + 1);
    out.threadsAndCpuPermits =
        (targetCpuPermits << CPU_PERMITS_BIT_OFFSET) | (targetThreads & THREADS_MASK);
    out.tasks = Integer.MIN_VALUE;
    out.cpuHeavyTasks = Integer.MIN_VALUE;
  }

  boolean isQuiescent(int poolSize) {
    return threads() == (isCancelled() ? -1 : poolSize);
  }

  boolean tryAcquireThread(AvailableState out) {
    if (threads() <= 0) {
      return false;
    }
    out.threadsAndCpuPermits = threadsAndCpuPermits - ONE_THREAD;
    out.tasks = tasks;
    out.cpuHeavyTasks = cpuHeavyTasks;
    return true;
  }

  boolean tryAcquireTask(AvailableState out) {
    if (tasks <= 0) {
      return false;
    }
    out.threadsAndCpuPermits = threadsAndCpuPermits;
    out.tasks = tasks - 1;
    out.cpuHeavyTasks = cpuHeavyTasks;
    return true;
  }

  boolean tryAcquireCpuHeavyTask(AvailableState out) {
    if (cpuHeavyTasks <= 0) {
      return false;
    }
    out.threadsAndCpuPermits = threadsAndCpuPermits;
    out.tasks = tasks;
    out.cpuHeavyTasks = cpuHeavyTasks - 1;
    return true;
  }

  void releaseThread(AvailableState out) {
    out.threadsAndCpuPermits = threadsAndCpuPermits + ONE_THREAD;
    out.tasks = tasks;
    out.cpuHeavyTasks = cpuHeavyTasks;
  }

  void releaseCpuPermit(AvailableState out) {
    out.threadsAndCpuPermits = threadsAndCpuPermits + ONE_CPU_PERMIT;
    out.tasks = tasks;
    out.cpuHeavyTasks = cpuHeavyTasks;
  }

  void releaseThreadAndCpuPermit(AvailableState out) {
    out.threadsAndCpuPermits = threadsAndCpuPermits + ONE_THREAD + ONE_CPU_PERMIT;
    out.tasks = tasks;
    out.cpuHeavyTasks = cpuHeavyTasks;
  }

  void releaseTask(AvailableState out) {
    out.threadsAndCpuPermits = threadsAndCpuPermits;
    out.tasks = tasks + 1;
    out.cpuHeavyTasks = cpuHeavyTasks;
  }

  boolean tryAcquireCpuHeavyTaskAndPermit(AvailableState out) {
    if (cpuHeavyTasks <= 0 || cpuPermits() <= 0) {
      return false;
    }
    out.threadsAndCpuPermits = threadsAndCpuPermits - ONE_CPU_PERMIT;
    out.tasks = tasks;
    out.cpuHeavyTasks = cpuHeavyTasks - 1;
    return true;
  }

  boolean tryAcquireTaskAndReleaseCpuPermit(AvailableState out) {
    if (tasks <= 0) {
      return false;
    }
    out.threadsAndCpuPermits = threadsAndCpuPermits + ONE_CPU_PERMIT;
    out.tasks = tasks - 1;
    out.cpuHeavyTasks = cpuHeavyTasks;
    return true;
  }

  boolean tryAcquireThreadAndCpuPermitElseReleaseCpuHeavyTask(AvailableState out) {
    out.tasks = tasks;
    if (threads() > 0 && cpuPermits() > 0) {
      out.threadsAndCpuPermits = threadsAndCpuPermits - (ONE_THREAD + ONE_CPU_PERMIT);
      out.cpuHeavyTasks = cpuHeavyTasks;
      return true;
    }
    out.threadsAndCpuPermits = threadsAndCpuPermits;
    out.cpuHeavyTasks = cpuHeavyTasks + 1;
    return false;
  }

  @Override
  public String toString() {
    return toStringHelper(this)
        .add("threads", threads())
        .add("cpuPermits", cpuPermits())
        .add("tasks", tasks)
        .add("cpuHeavyTasks", cpuHeavyTasks)
        .toString();
  }
}
