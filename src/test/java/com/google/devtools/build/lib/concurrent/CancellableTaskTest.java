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
import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.testutil.TestThread;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CancellableTask}. */
@RunWith(JUnit4.class)
public final class CancellableTaskTest {

  @Test
  public void cancelBeforeRun_preventsTaskFromRunning() throws Exception {
    var ran = new AtomicBoolean();
    var task = new CancellableTask<>(() -> ran.set(true));

    assertThat(task.cancelAndAwait(/* mayInterruptIfRunning= */ false)).isTrue();

    assertThat(task.runIfNotCancelled()).isFalse();
    assertThat(ran.get()).isFalse();
  }

  @Test
  public void cancelDuringRun_interruptsAndAwaitsTask() throws Exception {
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
    assertThat(taskStarted.tryAcquire(10, SECONDS)).isTrue();

    var canceller =
        new TestThread(
            () -> {
              assertThat(task.cancelAndAwait(/* mayInterruptIfRunning= */ true)).isFalse();
              cancellationReturned.release();
            });
    canceller.start();
    try {
      assertThat(cancellationReturned.tryAcquire(100, MILLISECONDS)).isFalse();
      assertThat(taskInterrupted.tryAcquire(10, SECONDS)).isTrue();
    } finally {
      taskMayFinish.release();
    }

    assertThat(cancellationReturned.tryAcquire(10, SECONDS)).isTrue();
    runner.joinAndAssertState(10_000);
    canceller.joinAndAssertState(10_000);
  }

  @Test
  public void cancelAfterRun_returnsWithoutBlocking() throws Exception {
    var task = new CancellableTask<>(() -> {});

    assertThat(task.runIfNotCancelled()).isTrue();

    assertThat(task.cancelAndAwait(/* mayInterruptIfRunning= */ true)).isFalse();
    // Repeated cancellation is allowed and must not block either.
    assertThat(task.cancelAndAwait(/* mayInterruptIfRunning= */ true)).isFalse();
  }

  @Test
  public void cancelTwiceBeforeRun_onlyFirstCallPreventsStart() throws Exception {
    var task = new CancellableTask<>(() -> {});

    assertThat(task.cancelAndAwait(/* mayInterruptIfRunning= */ false)).isTrue();
    assertThat(task.cancelAndAwait(/* mayInterruptIfRunning= */ false)).isFalse();
  }

  @Test
  public void runTwice_throws() throws Exception {
    var task = new CancellableTask<>(() -> {});

    assertThat(task.runIfNotCancelled()).isTrue();

    assertThrows(IllegalStateException.class, task::runIfNotCancelled);
  }

  @Test
  public void cancelUninterruptibly_whileInterrupted_restoresInterruptBit() {
    var task = new CancellableTask<>(() -> {});
    Thread.currentThread().interrupt();

    assertThat(task.cancelAndAwaitUninterruptibly(/* mayInterruptIfRunning= */ false)).isTrue();

    assertThat(Thread.interrupted()).isTrue();
  }
}
