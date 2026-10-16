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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.TestUtils.WAIT_TIMEOUT_MILLISECONDS;
import static com.google.devtools.build.lib.testutil.TestUtils.WAIT_TIMEOUT_SECONDS;
import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.testutil.TestThread;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link CancellableTask}. */
@RunWith(TestParameterInjector.class)
public final class CancellableTaskTest {

  /** The cancellation entry point under test, which must not affect the cancellation semantics. */
  private enum CancelVariant {
    INTERRUPTIBLE {
      @Override
      void cancel(CancellableTask<?> task) throws InterruptedException {
        task.cancelAndAwait();
      }
    },
    REQUEST_THEN_AWAIT {
      @Override
      void cancel(CancellableTask<?> task) throws InterruptedException {
        task.requestCancellation();
        task.awaitCompletion();
      }
    };

    abstract void cancel(CancellableTask<?> task) throws InterruptedException;
  }

  @Test
  public void runTwice_throws() throws Exception {
    var task = new CancellableTask<>(() -> {});

    assertThat(task.runIfNotCancelled()).isTrue();

    assertThrows(IllegalStateException.class, task::runIfNotCancelled);
  }

  @Test
  public void cancelBeforeRun_preventsTaskFromRunning(@TestParameter CancelVariant variant)
      throws Exception {
    var ran = new AtomicBoolean();
    var task = new CancellableTask<>(() -> ran.set(true));

    variant.cancel(task);

    assertThat(task.runIfNotCancelled()).isFalse();
    assertThat(ran.get()).isFalse();
  }

  @Test
  public void cancelTwiceBeforeRun_returnsWithoutBlocking(@TestParameter CancelVariant variant)
      throws Exception {
    var task = new CancellableTask<>(() -> {});

    variant.cancel(task);
    variant.cancel(task);
  }

  @Test
  public void cancelDuringRun_interruptsAndAwaitsTask(@TestParameter CancelVariant variant)
      throws Exception {
    var taskStarted = new Semaphore(0);
    var taskInterrupted = new Semaphore(0);
    var taskMayFinish = new Semaphore(0);
    var cancellationReturned = new Semaphore(0);
    var task =
        new CancellableTask<>(
            () -> {
              taskStarted.release();
              try {
                taskMayFinish.acquire();
              } catch (InterruptedException e) {
                taskInterrupted.release();
                taskMayFinish.acquireUninterruptibly();
              }
            });
    var runner = new TestThread(() -> assertThat(task.runIfNotCancelled()).isTrue());
    runner.start();
    assertThat(taskStarted.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();

    var canceller =
        new TestThread(
            () -> {
              variant.cancel(task);
              cancellationReturned.release();
            });
    canceller.start();
    try {
      assertThat(cancellationReturned.tryAcquire(100, MILLISECONDS)).isFalse();
      assertThat(taskInterrupted.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();
    } finally {
      taskMayFinish.release();
    }

    assertThat(cancellationReturned.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();
    runner.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
    canceller.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
  }

  @Test
  public void cancelAfterRun_returnsWithoutBlocking(@TestParameter CancelVariant variant)
      throws Exception {
    var task = new CancellableTask<>(() -> {});

    assertThat(task.runIfNotCancelled()).isTrue();

    variant.cancel(task);
    // Repeated cancellation is allowed and must not block either.
    variant.cancel(task);
  }

  @Test
  public void completion_runsOnceAfterBody() throws Exception {
    var events = new ArrayList<String>();
    var task = new CancellableTask<>(() -> events.add("body"), () -> events.add("completion"));

    assertThat(task.runIfNotCancelled()).isTrue();

    assertThat(events).containsExactly("body", "completion").inOrder();
    // Cancellation after completion must not run the completion action again.
    task.cancelAndAwait();
    assertThat(events).containsExactly("body", "completion").inOrder();
  }

  @Test
  public void completion_runsWhenBodyThrows() {
    var completionRuns = new AtomicInteger();
    var task =
        new CancellableTask<>(
            () -> {
              throw new IOException("task failed");
            },
            completionRuns::incrementAndGet);

    assertThrows(IOException.class, task::runIfNotCancelled);

    assertThat(completionRuns.get()).isEqualTo(1);
  }

  @Test
  public void cancelBeforeRun_runsCompletionExactlyOnce(@TestParameter CancelVariant variant)
      throws Exception {
    var completionRuns = new AtomicInteger();
    var task = new CancellableTask<>(() -> {}, completionRuns::incrementAndGet);

    variant.cancel(task);
    assertThat(completionRuns.get()).isEqualTo(1);

    // Neither a repeated cancellation nor the prevented run runs the completion action again.
    variant.cancel(task);
    assertThat(task.runIfNotCancelled()).isFalse();
    assertThat(completionRuns.get()).isEqualTo(1);
  }

  @Test
  public void cancelBeforeRun_concurrentCancellerAwaitsCompletion(
      @TestParameter CancelVariant variant) throws Exception {
    var completionStarted = new Semaphore(0);
    var completionMayFinish = new Semaphore(0);
    var task =
        new CancellableTask<>(
            () -> {},
            () -> {
              completionStarted.release();
              completionMayFinish.acquireUninterruptibly();
            });
    var winner = new TestThread(() -> variant.cancel(task));
    winner.start();
    assertThat(completionStarted.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();

    var loserReturned = new Semaphore(0);
    var loser =
        new TestThread(
            () -> {
              variant.cancel(task);
              loserReturned.release();
            });
    loser.start();
    try {
      // The losing canceler must not return while the winner's completion action is running.
      assertThat(loserReturned.tryAcquire(100, MILLISECONDS)).isFalse();
    } finally {
      completionMayFinish.release();
    }

    assertThat(loserReturned.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();
    winner.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
    loser.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
  }

  @Test
  public void cancelDuringCompletion_awaitsWithoutInterruptingCompletion(
      @TestParameter CancelVariant variant) throws Exception {
    var completionStarted = new Semaphore(0);
    var completionMayFinish = new Semaphore(0);
    var completionWasInterrupted = new AtomicBoolean();
    var task =
        new CancellableTask<>(
            () -> {},
            () -> {
              completionStarted.release();
              try {
                completionMayFinish.acquire();
              } catch (InterruptedException e) {
                completionWasInterrupted.set(true);
              }
            });
    var runner = new TestThread(() -> assertThat(task.runIfNotCancelled()).isTrue());
    runner.start();
    assertThat(completionStarted.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();

    var cancellationReturned = new Semaphore(0);
    var canceller =
        new TestThread(
            () -> {
              variant.cancel(task);
              cancellationReturned.release();
            });
    canceller.start();
    try {
      // Completion is part of the terminal lifecycle, but cancellation only interrupts the body.
      assertThat(cancellationReturned.tryAcquire(100, MILLISECONDS)).isFalse();
      assertThat(completionWasInterrupted.get()).isFalse();
    } finally {
      completionMayFinish.release();
    }

    assertThat(cancellationReturned.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();
    runner.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
    canceller.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
    assertThat(completionWasInterrupted.get()).isFalse();
  }

  @Test
  public void runLosingToCancelBeforeStart_awaitsCompletion(@TestParameter CancelVariant variant)
      throws Exception {
    var completionStarted = new Semaphore(0);
    var completionMayFinish = new Semaphore(0);
    var task =
        new CancellableTask<>(
            () -> {},
            () -> {
              completionStarted.release();
              completionMayFinish.acquireUninterruptibly();
            });
    var canceller = new TestThread(() -> variant.cancel(task));
    canceller.start();
    assertThat(completionStarted.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();

    var runReturned = new Semaphore(0);
    var ran = new AtomicBoolean(true);
    var runner =
        new TestThread(
            () -> {
              ran.set(task.runIfNotCancelled());
              runReturned.release();
            });
    runner.start();
    try {
      assertThat(runReturned.tryAcquire(100, MILLISECONDS)).isFalse();
    } finally {
      completionMayFinish.release();
    }

    assertThat(runReturned.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();
    runner.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
    canceller.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
    assertThat(ran.get()).isFalse();
  }

  @Test
  public void taskThrows_propagatesExceptionAndUnblocksCanceller(
      @TestParameter CancelVariant variant) throws Exception {
    var taskStarted = new Semaphore(0);
    var taskMayThrow = new Semaphore(0);
    var task =
        new CancellableTask<IOException>(
            () -> {
              taskStarted.release();
              taskMayThrow.acquireUninterruptibly();
              throw new IOException("task failed");
            });
    var thrown = new AtomicReference<IOException>();
    var runner =
        new TestThread(() -> thrown.set(assertThrows(IOException.class, task::runIfNotCancelled)));
    runner.start();
    assertThat(taskStarted.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();

    var cancellationReturned = new Semaphore(0);
    var canceller =
        new TestThread(
            () -> {
              variant.cancel(task);
              cancellationReturned.release();
            });
    canceller.start();
    try {
      assertThat(cancellationReturned.tryAcquire(100, MILLISECONDS)).isFalse();
    } finally {
      taskMayThrow.release();
    }

    // A task that leaves its body by throwing must still unblock cancelers.
    assertThat(cancellationReturned.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();
    runner.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
    canceller.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
    assertThat(thrown.get()).hasMessageThat().isEqualTo("task failed");
  }

  @Test
  public void cancelFromTaskBody_throws(@TestParameter CancelVariant variant) throws Exception {
    var taskRef = new AtomicReference<CancellableTask<Exception>>();
    var task =
        new CancellableTask<Exception>(
            () -> assertThrows(IllegalStateException.class, () -> variant.cancel(taskRef.get())));
    taskRef.set(task);

    assertThat(task.runIfNotCancelled()).isTrue();
  }

  @Test
  public void cancelFromCompletionAfterBody_throws(@TestParameter CancelVariant variant)
      throws Exception {
    var taskRef = new AtomicReference<CancellableTask<Exception>>();
    var task =
        new CancellableTask<Exception>(
            () -> {},
            () -> assertThrows(IllegalStateException.class, () -> variant.cancel(taskRef.get())));
    taskRef.set(task);

    var runner = new TestThread(() -> assertThat(task.runIfNotCancelled()).isTrue());
    runner.start();
    runner.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
  }

  @Test
  public void cancelFromCancelBeforeStartCompletion_throws(@TestParameter CancelVariant variant)
      throws Exception {
    var taskRef = new AtomicReference<CancellableTask<Exception>>();
    var task =
        new CancellableTask<Exception>(
            () -> {},
            () -> assertThrows(IllegalStateException.class, () -> variant.cancel(taskRef.get())));
    taskRef.set(task);

    var canceller = new TestThread(() -> variant.cancel(task));
    canceller.start();
    canceller.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
    assertThat(task.runIfNotCancelled()).isFalse();
  }

  @Test
  public void runFromCancelBeforeStartCompletion_throws() throws Exception {
    var taskRef = new AtomicReference<CancellableTask<Exception>>();
    var task =
        new CancellableTask<Exception>(
            () -> {},
            () ->
                assertThrows(IllegalStateException.class, () -> taskRef.get().runIfNotCancelled()));
    taskRef.set(task);

    var canceller = new TestThread(task::cancelAndAwait);
    canceller.start();
    canceller.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
  }

  @Test
  public void completionThrowsAfterBody_stillSignalsTerminalState(
      @TestParameter CancelVariant variant) throws Exception {
    var task =
        new CancellableTask<>(
            () -> {},
            () -> {
              throw new IllegalStateException("completion failed");
            });

    assertThat(assertThrows(IllegalStateException.class, task::runIfNotCancelled))
        .hasMessageThat()
        .isEqualTo("completion failed");

    assertLaterCancellationReturns(task, variant);
  }

  @Test
  public void completionThrowsAfterCancelBeforeStart_stillSignalsTerminalState(
      @TestParameter CancelVariant variant) throws Exception {
    var task =
        new CancellableTask<>(
            () -> {},
            () -> {
              throw new IllegalStateException("completion failed");
            });

    assertThat(assertThrows(IllegalStateException.class, () -> variant.cancel(task)))
        .hasMessageThat()
        .isEqualTo("completion failed");

    assertLaterCancellationReturns(task, variant);
    assertThat(task.runIfNotCancelled()).isFalse();
  }

  private static void assertLaterCancellationReturns(CancellableTask<?> task, CancelVariant variant)
      throws Exception {
    var cancellationReturned = new Semaphore(0);
    var canceller =
        new TestThread(
            () -> {
              variant.cancel(task);
              cancellationReturned.release();
            });
    canceller.start();

    assertThat(cancellationReturned.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();
    canceller.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
  }

  @Test
  public void requestCancellation_beforeRun_preventsStartAndRunsCompletion() throws Exception {
    var ran = new AtomicBoolean();
    var completionRuns = new AtomicInteger();
    var task = new CancellableTask<>(() -> ran.set(true), completionRuns::incrementAndGet);

    task.requestCancellation();

    assertThat(completionRuns.get()).isEqualTo(1);
    assertThat(task.runIfNotCancelled()).isFalse();
    assertThat(ran.get()).isFalse();
    assertThat(completionRuns.get()).isEqualTo(1);
  }

  @Test
  public void requestCancellation_duringRun_interruptsWithoutAwaiting() throws Exception {
    var taskStarted = new Semaphore(0);
    var taskInterrupted = new Semaphore(0);
    var taskMayFinish = new Semaphore(0);
    var task =
        new CancellableTask<>(
            () -> {
              taskStarted.release();
              try {
                taskMayFinish.acquire();
              } catch (InterruptedException e) {
                taskInterrupted.release();
                taskMayFinish.acquireUninterruptibly();
              }
            });
    var runner = new TestThread(() -> assertThat(task.runIfNotCancelled()).isTrue());
    runner.start();
    assertThat(taskStarted.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();

    // Returns while the body is still running: it only finishes after the release below.
    task.requestCancellation();

    assertThat(taskInterrupted.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();
    taskMayFinish.release();
    runner.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
  }

  @Test
  public void requestCancellation_thenAwaitCompletion_doesNotInterruptTaskAgain() throws Exception {
    var taskStarted = new Semaphore(0);
    var cleanupStarted = new Semaphore(0);
    var cleanupMayFinish = new Semaphore(0);
    var interruptionCount = new AtomicInteger();
    var task =
        new CancellableTask<>(
            () -> {
              taskStarted.release();
              try {
                new Semaphore(0).acquire();
              } catch (InterruptedException e) {
                interruptionCount.incrementAndGet();
              }
              cleanupStarted.release();
              try {
                cleanupMayFinish.acquire();
              } catch (InterruptedException e) {
                interruptionCount.incrementAndGet();
              }
            });
    var runner = new TestThread(() -> assertThat(task.runIfNotCancelled()).isTrue());
    runner.start();
    assertThat(taskStarted.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();

    task.requestCancellation();
    assertThat(cleanupStarted.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();
    var waiter = new TestThread(task::awaitCompletion);
    waiter.start();
    try {
      // Awaiting a previously requested cancellation must not send another interrupt that could
      // abort the task's cleanup.
      assertThat(waiter.isAlive()).isTrue();
      assertThat(interruptionCount.get()).isEqualTo(1);
    } finally {
      cleanupMayFinish.release();
    }

    runner.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
    waiter.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
    assertThat(interruptionCount.get()).isEqualTo(1);
  }

  @Test
  public void cancelAndAwait_interruptedWhileAwaiting_retryInterruptsAgainAndAwaits()
      throws Exception {
    var taskStarted = new Semaphore(0);
    var taskMayFinish = new Semaphore(0);
    var taskInterruptions = new Semaphore(0);
    var interruptionCount = new AtomicInteger();
    var task =
        new CancellableTask<>(
            () -> {
              taskStarted.release();
              while (true) {
                try {
                  taskMayFinish.acquire();
                  return;
                } catch (InterruptedException e) {
                  interruptionCount.incrementAndGet();
                  taskInterruptions.release();
                }
              }
            });
    var runner = new TestThread(() -> assertThat(task.runIfNotCancelled()).isTrue());
    runner.start();
    assertThat(taskStarted.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();

    var cancellerThrew = new Semaphore(0);
    var cancellationReturned = new Semaphore(0);
    var canceller =
        new TestThread(
            () -> {
              assertThrows(InterruptedException.class, task::cancelAndAwait);
              cancellerThrew.release();
              // A retry interrupts the still-running task again and awaits quiescence.
              task.cancelAndAwait();
              cancellationReturned.release();
            });
    canceller.start();
    assertThat(taskInterruptions.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();

    canceller.interrupt();
    assertThat(cancellerThrew.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();
    assertThat(taskInterruptions.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();
    try {
      assertThat(cancellationReturned.tryAcquire(100, MILLISECONDS)).isFalse();
      assertThat(interruptionCount.get()).isEqualTo(2);
    } finally {
      taskMayFinish.release();
    }

    assertThat(cancellationReturned.tryAcquire(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();
    runner.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
    canceller.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
    assertThat(interruptionCount.get()).isEqualTo(2);
  }

  @Test
  public void runAfterCancel_whileInterrupted_restoresInterruptBit() throws Exception {
    var task = new CancellableTask<>(() -> {});
    task.cancelAndAwait();
    Thread.currentThread().interrupt();

    assertThat(task.runIfNotCancelled()).isFalse();

    assertThat(Thread.interrupted()).isTrue();
  }

  @Test
  public void concurrentRunAndCancel_exactlyOneClaimWins() throws Exception {
    for (int i = 0; i < 1000; i++) {
      var bodyRuns = new AtomicInteger();
      var bodyRunning = new AtomicBoolean();
      var completionRuns = new AtomicInteger();
      var task =
          new CancellableTask<>(
              () -> {
                bodyRunning.set(true);
                bodyRuns.incrementAndGet();
                bodyRunning.set(false);
              },
              completionRuns::incrementAndGet);
      var barrier = new CyclicBarrier(3);
      var ran = new AtomicBoolean();
      var runner =
          new TestThread(
              () -> {
                barrier.await();
                ran.set(task.runIfNotCancelled());
              });
      var cancellers = new TestThread[2];
      for (int c = 0; c < cancellers.length; c++) {
        cancellers[c] =
            new TestThread(
                () -> {
                  barrier.await();
                  task.cancelAndAwait();
                  // A normal return from cancellation guarantees that the body is not running
                  // and that the completion action has finished.
                  assertThat(bodyRunning.get()).isFalse();
                  assertThat(completionRuns.get()).isEqualTo(1);
                });
      }
      // Rotate which thread arrives at the barrier last: that thread trips the barrier and
      // proceeds without parking while the others are still being woken, so it tends to win the
      // claim. Rotation makes both outcomes of the race common.
      var startOrder =
          switch (i % 3) {
            case 0 -> new TestThread[] {runner, cancellers[0], cancellers[1]};
            case 1 -> new TestThread[] {cancellers[0], cancellers[1], runner};
            default -> new TestThread[] {cancellers[0], runner, cancellers[1]};
          };
      for (var thread : startOrder) {
        thread.start();
      }
      runner.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
      for (var canceller : cancellers) {
        canceller.joinAndAssertState(WAIT_TIMEOUT_MILLISECONDS);
      }

      // The claim is atomic: either the task ran to completion exactly once and no canceler
      // prevented it, or cancellation prevented it from running.
      assertThat(ran.get()).isEqualTo(bodyRuns.get() == 1);
      assertThat(bodyRuns.get()).isAtMost(1);
      assertThat(completionRuns.get()).isEqualTo(1);
    }
  }
}
