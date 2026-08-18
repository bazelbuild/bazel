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

import java.util.Objects;
import java.util.concurrent.CountDownLatch;
import javax.annotation.Nullable;

/**
 * A one-shot task whose cancellation can wait until its execution has quiesced.
 *
 * @param <E> the checked exception thrown by the task body, if any
 */
public final class CancellableTask<E extends Exception> {

  /** The work performed by a {@link CancellableTask}. */
  @FunctionalInterface
  public interface Task<E extends Exception> {
    void run() throws E;
  }

  private final Task<E> task;
  private final CountDownLatch done = new CountDownLatch(1);

  private boolean claimed;
  private boolean cancelledBeforeStart;
  @Nullable private Thread runner;

  public CancellableTask(Task<E> task) {
    this.task = Objects.requireNonNull(task);
  }

  /**
   * Runs the task synchronously unless cancellation prevented it from starting.
   *
   * @return whether the task ran
   */
  public boolean runIfNotCancelled() throws E {
    synchronized (this) {
      if (cancelledBeforeStart) {
        return false;
      }
      if (claimed) {
        throw new IllegalStateException("task executed more than once");
      }
      claimed = true;
      runner = Thread.currentThread();
    }
    try {
      task.run();
      return true;
    } finally {
      synchronized (this) {
        runner = null;
      }
      done.countDown();
    }
  }

  /**
   * Cancels the task and waits until it no longer executes.
   *
   * <p>If the task has not started, this prevents it from starting. Otherwise, it optionally
   * interrupts the thread executing the task and waits for that thread to leave the task body.
   * Unlike {@link java.util.concurrent.Future#cancel}, a normal return therefore guarantees that
   * the task has quiesced.
   *
   * <p>The return value allows the caller that prevented execution to perform any completion work
   * that would otherwise have been the task's responsibility.
   *
   * <p>Note that {@code mayInterruptIfRunning} may interrupt the executing thread just after it has
   * left the task body, so the task must be run on a thread that either does no further work or
   * tolerates a spurious interrupt.
   *
   * @return whether this call prevented the task from starting
   */
  public boolean cancelAndAwait(boolean mayInterruptIfRunning) throws InterruptedException {
    Thread localRunner;
    synchronized (this) {
      if (!claimed) {
        claimed = true;
        cancelledBeforeStart = true;
        done.countDown();
        return true;
      }
      localRunner = runner;
    }
    if (mayInterruptIfRunning && localRunner != null) {
      localRunner.interrupt();
    }
    // Waiting for oneself is impossible. This exception is only relevant to reentrant
    // cancellation from the task body; external cancellation still always awaits quiescence.
    if (localRunner != Thread.currentThread()) {
      done.await();
    }
    return false;
  }

  /**
   * An uninterruptible variant of {@link #cancelAndAwait} that restores the interrupt bit before
   * returning.
   */
  public boolean cancelAndAwaitUninterruptibly(boolean mayInterruptIfRunning) {
    boolean interrupted = false;
    try {
      while (true) {
        try {
          return cancelAndAwait(mayInterruptIfRunning);
        } catch (InterruptedException e) {
          interrupted = true;
        }
      }
    } finally {
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
    }
  }
}
