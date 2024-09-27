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

package com.google.devtools.build.lib.server;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.errorprone.annotations.concurrent.GuardedBy;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.util.ArrayList;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * Runs cleanup-related tasks during an idle period in the server.
 *
 * <p>By default, there is a single final task responsible for running garbage collection.
 * Additional tasks may be added at construction time and execute serially before the final task.
 * When {@link #idle} is called, signaling the beginning of an idle period, execution of idle tasks
 * begins. When {@link #busy} is called, signaling the end of an idle period, all pending idle tasks
 * are interrupted.
 *
 * <p>A fresh instance must be constructed to manage each individual idle period.
 */
public final class IdleTaskManager {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private enum State {
    INITIALIZED,
    IDLE,
    BUSY
  }

  @GuardedBy("this")
  private State state = State.INITIALIZED;

  // Use a single-threaded ScheduledThreadPoolExecutor to ensure that tasks execute serially.
  private final ScheduledThreadPoolExecutor executor =
      new ScheduledThreadPoolExecutor(
          1, new ThreadFactoryBuilder().setNameFormat("idle-server-tasks-%d").build());

  private final ImmutableList<IdleTask> registeredTasks;
  private final boolean stateKeptAfterBuild;

  private final ArrayList<ScheduledFuture<?>> taskFutures = new ArrayList<>();

  /**
   * Creates a new {@link IdleTaskManager}.
   *
   * @param idleTasks idle tasks registered during the previous build
   * @param stateKeptAfterBuild whether state from the previous build was kept
   */
  public IdleTaskManager(ImmutableList<IdleTask> idleTasks, boolean stateKeptAfterBuild) {
    this.registeredTasks = idleTasks;
    this.stateKeptAfterBuild = stateKeptAfterBuild;
  }

  /**
   * Called by the main thread when the server becomes idle.
   *
   * <p>Does not block, but may schedule tasks in the background.
   */
  public synchronized void idle() {
    checkState(state == State.INITIALIZED);
    state = State.IDLE;

    // Schedule tasks in the order they were registered.
    for (IdleTask task : registeredTasks) {
      taskFutures.add(executor.schedule(task::run, task.delay().toSeconds(), TimeUnit.SECONDS));
    }

    // Schedule the final task to run after everything else.
    // Note that this is effectively enforced by the fact that the executor is single-threaded and
    // executes tasks in the order they are scheduled.
    taskFutures.add(
        executor.schedule(
            () -> runGc(stateKeptAfterBuild),
            // If state was kept after the build, wait for a few seconds before triggering GC, to
            // avoid unnecessarily slowing down an immediately following incremental build.
            stateKeptAfterBuild ? 10 : 0,
            TimeUnit.SECONDS));
  }

  /**
   * Called by the main thread when the server gets to work.
   *
   * <p>Interrupts any pending idle tasks and blocks for their completion before returning.
   */
  public synchronized void busy() {
    checkState(state == State.IDLE);
    state = State.BUSY;

    // Interrupt pending tasks.
    var unused = executor.shutdownNow();

    // Wait for all tasks to complete, so they cannot interfere with a subsequent command.
    boolean interrupted = false;
    while (true) {
      try {
        executor.awaitTermination(Long.MAX_VALUE, TimeUnit.HOURS);
        break;
      } catch (InterruptedException e) {
        // It's unsafe to leak tasks - keep trying and reset the interrupt bit later.
        interrupted = true;
      }
    }

    for (ScheduledFuture<?> taskFuture : taskFutures) {
      try {
        taskFuture.get(0, TimeUnit.SECONDS);
      } catch (ExecutionException e) {
        logger.atWarning().withCause(e.getCause()).log("Unexpected exception from idle task");
      } catch (TimeoutException | CancellationException e) {
        // Expected if the task hadn't yet started running or was interrupted mid-run.
      } catch (InterruptedException e) {
        // We ourselves were interrupted, not the task.
        interrupted = true;
      }
    }

    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  @VisibleForTesting long runGcCalled;

  private void runGc(boolean stateKeptAfterBuild) {
    runGcCalled = System.nanoTime();

    MemoryMXBean memBean = ManagementFactory.getMemoryMXBean();
    MemoryUsage before = memBean.getHeapMemoryUsage();
    try (var p = GoogleAutoProfilerUtils.logged("Idle GC")) {
      System.gc();
    }
    MemoryUsage after = memBean.getHeapMemoryUsage();

    logger.atInfo().log(
        "[Idle GC] used: %s -> %s, committed: %s -> %s",
        StringUtilities.prettyPrintBytes(before.getUsed()),
        StringUtilities.prettyPrintBytes(after.getUsed()),
        StringUtilities.prettyPrintBytes(before.getCommitted()),
        StringUtilities.prettyPrintBytes(after.getCommitted()));
  }
}
