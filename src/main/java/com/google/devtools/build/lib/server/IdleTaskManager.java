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
import static java.util.concurrent.TimeUnit.SECONDS;

import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.errorprone.annotations.concurrent.GuardedBy;
import java.util.ArrayList;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeoutException;

/**
 * Runs cleanup-related tasks during an idle period in the server.
 *
 * <p>A fresh instance must be constructed to manage each individual idle period. The idle period
 * begins when {@link #idle} is called and ends when {@link #busy} is called.
 */
public final class IdleTaskManager {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private enum State {
    INITIALIZED,
    IDLE,
    BUSY
  }

  private static final class IdleTaskWrapper implements Runnable {
    private final IdleTask task;

    IdleTaskWrapper(IdleTask task) {
      this.task = task;
    }

    @Override
    public void run() {
      String name = task.displayName();
      try {
        logger.atInfo().log("%s idle task started", name);
        task.run();
        logger.atInfo().log("%s idle task finished", name);
      } catch (IdleTaskException e) {
        logger.atWarning().withCause(e.getCause()).log("%s idle task failed", name);
      } catch (InterruptedException e) {
        // There's no point in restoring the interrupt bit since this thread belongs to an executor
        // service that is shutting down.
        logger.atWarning().withCause(e).log("%s idle task interrupted", name);
      }
    }
  }

  @GuardedBy("this")
  private State state = State.INITIALIZED;

  // Use a single-threaded ScheduledThreadPoolExecutor to ensure that tasks execute serially.
  private final ScheduledThreadPoolExecutor executor =
      new ScheduledThreadPoolExecutor(
          1, new ThreadFactoryBuilder().setNameFormat("idle-server-tasks-%d").build());

  private final ImmutableList<IdleTask> registeredTasks;

  private final ArrayList<ScheduledFuture<?>> taskFutures = new ArrayList<>();

  /**
   * Creates a new {@link IdleTaskManager}.
   *
   * @param idleTasks tasks to run while idle
   */
  public IdleTaskManager(ImmutableList<IdleTask> idleTasks) {
    this.registeredTasks = idleTasks;
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
      taskFutures.add(
          executor.schedule(new IdleTaskWrapper(task), task.delay().toSeconds(), SECONDS));
    }
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
        executor.awaitTermination(Long.MAX_VALUE, SECONDS);
        break;
      } catch (InterruptedException e) {
        // It's unsafe to leak tasks - keep trying and restore the interrupt bit later.
        interrupted = true;
      }
    }

    for (ScheduledFuture<?> taskFuture : taskFutures) {
      try {
        taskFuture.get(0, SECONDS);
      } catch (ExecutionException e) {
        // Must be an unchecked exception since all checked exceptions thrown by an IdleTask are
        // handled by its IdleTaskWrapper.
        throw new IllegalStateException("Unexpected exception thrown by idle task", e.getCause());
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
}
