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

import com.google.common.util.concurrent.Uninterruptibles;
import com.google.errorprone.annotations.concurrent.GuardedBy;
import java.util.Objects;
import java.util.concurrent.CountDownLatch;
import javax.annotation.Nullable;

/**
 * A one-shot task whose cancellation can wait until its execution has quiesced.
 *
 * <p>An optional completion action runs exactly once when the task's lifecycle ends, no matter how
 * it ends: on the task thread after the body has run (even if it threw), or on the thread of the
 * canceler that prevented the task from starting. Once the completion action begins, cancellation
 * does not interrupt it. Quiescence includes the completion action: a normal return from {@link
 * #runIfNotCancelled}, {@link #awaitCompletion}, {@link #awaitCompletionUninterruptibly}, or {@link
 * #cancelAndAwait} guarantees that both the body and the completion action have finished. {@link
 * #requestCancellation} only initiates cancellation and does not provide this guarantee.
 *
 * <p>This class serves as a replacement for {@link java.util.concurrent.Future} when canceled
 * non-pure tasks may still interfere with other tasks until they have completed. Since {@link
 * java.util.concurrent.Future#cancel} is specified to immediately mark the future as completed, it
 * is unfortunately not possible for any conforming implementation to provide awaitable
 * cancellation.
 *
 * @param <E> the checked exception thrown by the task body, if any
 */
public final class CancellableTask<E extends Exception> {

  private enum Phase {
    NOT_STARTED,
    RUNNING,
    COMPLETING_AFTER_RUNNING,
    COMPLETING_WITHOUT_RUNNING,
    COMPLETED_AFTER_RUNNING,
    COMPLETED_WITHOUT_RUNNING
  }

  /** The work performed by a {@link CancellableTask}. */
  @FunctionalInterface
  public interface Task<E extends Exception> {
    void run() throws E;
  }

  private final Task<E> task;
  private final Runnable onCompletion;
  private final CountDownLatch done = new CountDownLatch(1);

  @GuardedBy("this")
  private Phase phase = Phase.NOT_STARTED;

  @GuardedBy("this")
  @Nullable
  private Thread executingThread;

  public CancellableTask(Task<E> task) {
    this(task, () -> {});
  }

  /**
   * Creates a task with a completion action.
   *
   * <p>The completion action must not throw or reenter this task. It may begin with its thread's
   * interrupt bit set if the task body was interrupted, but cancellation does not interrupt it once
   * it has begun.
   */
  public CancellableTask(Task<E> task, Runnable onCompletion) {
    this.task = Objects.requireNonNull(task);
    this.onCompletion = Objects.requireNonNull(onCompletion);
  }

  /**
   * Runs the task synchronously unless cancellation prevented it from starting.
   *
   * <p>A normal return guarantees that the completion action has finished, including when
   * cancellation prevented the task body from running.
   *
   * @return whether the task ran
   */
  public boolean runIfNotCancelled() throws E {
    boolean wasCancelled;
    synchronized (this) {
      wasCancelled =
          switch (phase) {
            case NOT_STARTED -> {
              phase = Phase.RUNNING;
              executingThread = Thread.currentThread();
              yield false;
            }
            case COMPLETING_WITHOUT_RUNNING, COMPLETED_WITHOUT_RUNNING -> {
              if (executingThread == Thread.currentThread()) {
                throw new IllegalStateException(
                    "completion action must not attempt to execute its task, this would deadlock");
              }
              yield true;
            }
            default -> throw new IllegalStateException("task executed more than once");
          };
    }
    if (wasCancelled) {
      Uninterruptibles.awaitUninterruptibly(done);
      return false;
    }
    try {
      task.run();
      return true;
    } finally {
      synchronized (this) {
        phase = Phase.COMPLETING_AFTER_RUNNING;
      }
      runCompletion();
    }
  }

  /**
   * Requests cancellation without waiting for the task to quiesce.
   *
   * <p>If the task has not started, this prevents it from starting and runs the completion action
   * on its behalf before returning. Otherwise, it interrupts the thread executing the task body,
   * which may still be executing when this method returns. Call {@link #awaitCompletion} afterward
   * if quiescence is required without requesting cancellation a second time.
   *
   * @throws IllegalStateException if called from the task body or completion action
   */
  public void requestCancellation() {
    boolean preventedStart;
    synchronized (this) {
      preventedStart =
          switch (phase) {
            case NOT_STARTED -> {
              phase = Phase.COMPLETING_WITHOUT_RUNNING;
              executingThread = Thread.currentThread();
              yield true;
            }
            case RUNNING -> {
              if (executingThread == Thread.currentThread()) {
                throw new IllegalStateException("task cannot cancel itself");
              }
              // Interrupt while holding the state lock so that the task cannot enter its completion
              // action between publishing the executing thread and receiving the interrupt.
              Objects.requireNonNull(executingThread).interrupt();
              yield false;
            }
            case COMPLETING_AFTER_RUNNING, COMPLETING_WITHOUT_RUNNING -> {
              if (executingThread == Thread.currentThread()) {
                throw new IllegalStateException("task cannot cancel itself");
              }
              yield false;
            }
            case COMPLETED_AFTER_RUNNING, COMPLETED_WITHOUT_RUNNING -> false;
          };
    }
    if (preventedStart) {
      // The task will never run, so run the completion action on its behalf. Signal completion only
      // afterward so that concurrent awaiters only return once it has finished.
      runCompletion();
    }
  }

  /**
   * Waits until the task body and completion action have finished without requesting cancellation.
   *
   * <p>If neither execution nor cancellation has started, this waits until one of them does.
   *
   * @throws IllegalStateException if called from the task body or completion action, neither of
   *     which can await the task
   */
  public void awaitCompletion() throws InterruptedException {
    checkCanAwait();
    done.await();
  }

  /**
   * An uninterruptible variant of {@link #awaitCompletion} that restores the interrupt bit before
   * returning.
   *
   * @throws IllegalStateException if called from the task body or completion action, neither of
   *     which can await the task
   */
  public void awaitCompletionUninterruptibly() {
    checkCanAwait();
    Uninterruptibles.awaitUninterruptibly(done);
  }

  /**
   * Cancels the task and waits until it no longer executes.
   *
   * <p>If the task has not started, this prevents it from starting and runs the completion action
   * on its behalf. Otherwise, it interrupts the thread executing the task body and waits for the
   * body and completion action to finish. Once the completion action begins, cancellation waits for
   * it without interrupting its thread. Unlike {@link java.util.concurrent.Future#cancel}, a normal
   * return therefore guarantees that the task has quiesced.
   *
   * @throws IllegalStateException if called from the task body or completion action, neither of
   *     which can await the task
   */
  public void cancelAndAwait() throws InterruptedException {
    requestCancellation();
    awaitCompletion();
  }

  private void checkCanAwait() {
    synchronized (this) {
      if (executingThread == Thread.currentThread()) {
        throw new IllegalStateException("task cannot await itself");
      }
    }
  }

  /** Runs the completion action and publishes terminal state even if the action throws. */
  private void runCompletion() {
    try {
      onCompletion.run();
    } finally {
      synchronized (this) {
        executingThread = null;
        phase =
            phase == Phase.COMPLETING_WITHOUT_RUNNING
                ? Phase.COMPLETED_WITHOUT_RUNNING
                : Phase.COMPLETED_AFTER_RUNNING;
      }
      done.countDown();
    }
  }
}
