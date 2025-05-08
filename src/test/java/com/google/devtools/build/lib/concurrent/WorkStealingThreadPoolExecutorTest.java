// Copyright 2025 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link WorkStealingThreadPoolExecutor}. */
@RunWith(JUnit4.class)
public class WorkStealingThreadPoolExecutorTest {
  private static final int PARALLELISM = 100;

  @Test
  public void execute_allTasksAreExecuted_numTasksLessThanNumWorkers() throws Exception {
    AtomicBoolean interrupted = new AtomicBoolean(false);
    AtomicInteger sum = new AtomicInteger(0);
    int numTasks = PARALLELISM / 2;
    try (var executor =
        new WorkStealingThreadPoolExecutor(PARALLELISM, Thread.ofPlatform().factory())) {
      CountDownLatch countDown = new CountDownLatch(numTasks);
      for (int i = 0; i < numTasks; i++) {
        executor.execute(
            () -> {
              sum.incrementAndGet();
              try {
                Thread.sleep(1);
              } catch (InterruptedException e) {
                interrupted.set(true);
              }
              countDown.countDown();
            });
      }
      countDown.await();
    }
    assertThat(interrupted.get()).isFalse();
    assertThat(sum.get()).isEqualTo(numTasks);
  }

  @Test
  public void execute_allTasksAreExecuted_numTasksMoreThanNumWorkers() throws Exception {
    AtomicBoolean interrupted = new AtomicBoolean(false);
    AtomicInteger sum = new AtomicInteger(0);
    int numTasks = PARALLELISM * 5;
    try (var executor =
        new WorkStealingThreadPoolExecutor(PARALLELISM, Thread.ofPlatform().factory())) {
      CountDownLatch countDown = new CountDownLatch(numTasks);
      for (int i = 0; i < numTasks; i++) {
        executor.execute(
            () -> {
              sum.incrementAndGet();
              try {
                Thread.sleep(1);
              } catch (InterruptedException e) {
                interrupted.set(true);
              }
              countDown.countDown();
            });
      }
      countDown.await();
    }
    assertThat(interrupted.get()).isFalse();
    assertThat(sum.get()).isEqualTo(numTasks);
  }

  @Test
  public void execute_reachParallelism() throws Exception {
    AtomicBoolean interrupted = new AtomicBoolean(false);
    AtomicInteger sum = new AtomicInteger(0);
    int numBatches = 5;
    try (var executor =
        new WorkStealingThreadPoolExecutor(PARALLELISM, Thread.ofPlatform().factory())) {
      for (int i = 0; i < numBatches; ++i) {
        CountDownLatch startedCountDown = new CountDownLatch(PARALLELISM);
        CountDownLatch continueCountDown = new CountDownLatch(1);
        for (int j = 0; j < PARALLELISM; j++) {
          executor.execute(
              () -> {
                startedCountDown.countDown();
                sum.incrementAndGet();

                // Clear the interruption bit
                var unused = Thread.interrupted();
                try {
                  Thread.sleep(1);
                  continueCountDown.await();
                } catch (InterruptedException e) {
                  interrupted.set(true);
                }

                // Set the interruption bit to test that pool can continue scheduling tasks even if
                // the task left the interruption bit set.
                Thread.currentThread().interrupt();
              });
        }
        startedCountDown.await();
        continueCountDown.countDown();
      }
    }
    assertThat(interrupted.get()).isFalse();
    assertThat(sum.get()).isEqualTo(numBatches * PARALLELISM);
  }

  @Test
  public void execute_taskThrowsRuntimeException_reachParallelism() throws Exception {
    AtomicBoolean interrupted = new AtomicBoolean(false);
    int numBatches = 5;
    CountDownLatch uncaughtExceptionHandlerCountDown = new CountDownLatch(numBatches * PARALLELISM);
    AtomicReference<Throwable> errorFromUncaughtExceptionHandler = new AtomicReference<>();
    try (var executor =
        new WorkStealingThreadPoolExecutor(PARALLELISM, Thread.ofPlatform().factory())) {
      for (int i = 0; i < numBatches; i++) {
        CountDownLatch startedCountDown = new CountDownLatch(PARALLELISM);
        CountDownLatch continueCountDown = new CountDownLatch(1);
        for (int j = 0; j < PARALLELISM; j++) {
          executor.execute(
              () -> {
                var thread = Thread.currentThread();
                thread.setUncaughtExceptionHandler(
                    (t, e) -> {
                      try {
                        assertThat(t).isEqualTo(thread);
                        assertThat(e).isInstanceOf(IllegalStateException.class);
                        assertThat(e).hasMessageThat().isEqualTo("test");
                      } catch (Throwable error) {
                        errorFromUncaughtExceptionHandler.set(error);
                      } finally {
                        uncaughtExceptionHandlerCountDown.countDown();
                      }
                    });
                startedCountDown.countDown();

                try {
                  continueCountDown.await();
                } catch (InterruptedException e) {
                  interrupted.set(true);
                }

                throw new IllegalStateException("test");
              });
        }
        startedCountDown.await();
        continueCountDown.countDown();
      }
    }
    uncaughtExceptionHandlerCountDown.await();
    assertThat(interrupted.get()).isFalse();
    assertThat(errorFromUncaughtExceptionHandler.get()).isNull();
  }

  @Test
  public void shutdown_remainingTasksExecuted() throws Exception {
    int numTasks = PARALLELISM * 5;
    AtomicBoolean interrupted = new AtomicBoolean(false);
    AtomicInteger numExecuted = new AtomicInteger(0);
    var executor = new WorkStealingThreadPoolExecutor(PARALLELISM, Thread.ofPlatform().factory());
    CountDownLatch shutdownCalled = new CountDownLatch(1);
    for (int i = 0; i < numTasks; i++) {
      executor.execute(
          () -> {
            try {
              shutdownCalled.await();
              numExecuted.incrementAndGet();
            } catch (InterruptedException e) {
              interrupted.set(true);
            }
          });
    }

    executor.shutdown();
    shutdownCalled.countDown();

    assertThat(executor.isShutdown()).isTrue();
    assertThrows(RejectedExecutionException.class, () -> executor.execute(() -> {}));

    boolean terminated = executor.awaitTermination(1L, TimeUnit.DAYS);
    assertThat(terminated).isTrue();
    assertThat(executor.isTerminated()).isTrue();
    assertThat(interrupted.get()).isFalse();
    assertThat(numExecuted.get()).isEqualTo(numTasks);
  }

  @Test
  public void shutdownNow_interruptTasks() throws Exception {
    int numTasks = PARALLELISM * 5;
    AtomicInteger numExecuted = new AtomicInteger(0);
    var executor = new WorkStealingThreadPoolExecutor(PARALLELISM, Thread.ofPlatform().factory());
    CountDownLatch neverAwake = new CountDownLatch(1);
    CountDownLatch startedCountDown = new CountDownLatch(PARALLELISM);
    for (int i = 0; i < numTasks; i++) {
      executor.execute(
          () -> {
            try {
              startedCountDown.countDown();
              neverAwake.await();
              numExecuted.incrementAndGet();
            } catch (InterruptedException e) {
              // Intentionally ignored
            }
          });
    }

    startedCountDown.await();
    var remainingTasks = executor.shutdownNow();
    assertThat(remainingTasks).hasSize(numTasks - PARALLELISM);

    assertThat(executor.isShutdown()).isTrue();
    assertThrows(RejectedExecutionException.class, () -> executor.execute(() -> {}));

    boolean terminated = executor.awaitTermination(1L, TimeUnit.DAYS);
    assertThat(terminated).isTrue();
    assertThat(executor.isTerminated()).isTrue();
    assertThat(numExecuted.get()).isEqualTo(0);
  }
}
