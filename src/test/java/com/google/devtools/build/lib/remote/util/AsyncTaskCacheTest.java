// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.util;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static org.junit.Assert.assertThrows;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AsyncTaskCache}. */
@RunWith(JUnit4.class)
public class AsyncTaskCacheTest {

  @Test
  public void execute_taskFinished_completed() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    SettableFuture<String> task = SettableFuture.create();
    ListenableFuture<String> future = cache.executeIfNot("key1", () -> task);
    assertThat(future.isDone()).isFalse();
    assertThat(cache.getInProgressTasks()).containsExactly("key1");

    task.set("value1");

    assertThat(future.get()).isEqualTo("value1");
    assertThat(cache.getInProgressTasks()).isEmpty();
    assertThat(cache.getFinishedTasks()).containsExactly("key1");
  }

  @Test
  public void execute_taskHasError_propagateError() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    SettableFuture<String> task = SettableFuture.create();
    ListenableFuture<String> future = cache.executeIfNot("key1", () -> task);
    var error = new IOException("error");

    task.setException(error);

    var e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isSameInstanceAs(error);
    assertThat(cache.getInProgressTasks()).isEmpty();
    assertThat(cache.getFinishedTasks()).isEmpty();
  }

  @Test
  public void execute_taskInProgress_noReExecution() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    SettableFuture<String> task = SettableFuture.create();
    AtomicInteger executionTimes = new AtomicInteger(0);
    ListenableFuture<String> future1 =
        cache.executeIfNot(
            "key1",
            () -> {
              executionTimes.incrementAndGet();
              return task;
            });
    assertThat(cache.getInProgressTasks()).containsExactly("key1");
    assertThat(cache.getFinishedTasks()).isEmpty();

    ListenableFuture<String> future2 =
        cache.executeIfNot(
            "key1",
            () -> {
              executionTimes.incrementAndGet();
              return task;
            });
    assertThat(future2.isDone()).isFalse();

    task.set("value1");

    assertThat(future1.get()).isEqualTo("value1");
    assertThat(future2.get()).isEqualTo("value1");
    assertThat(executionTimes.get()).isEqualTo(1);
    assertThat(cache.getInProgressTasks()).isEmpty();
    assertThat(cache.getFinishedTasks()).containsExactly("key1");
  }

  @Test
  public void executeForcibly_taskInProgress_reExecution() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    SettableFuture<String> task1 = SettableFuture.create();
    SettableFuture<String> task2 = SettableFuture.create();
    ListenableFuture<String> future1 = cache.execute("key1", () -> task1, /* force= */ true);
    assertThat(cache.getInProgressTasks()).containsExactly("key1");

    ListenableFuture<String> future2 = cache.execute("key1", () -> task2, /* force= */ true);

    // Unlike a non-forced execution, this does not join the in-progress one.
    task2.set("value2");
    assertThat(future2.get()).isEqualTo("value2");
    assertThat(future1.isDone()).isFalse();

    task1.set("value1");
    assertThat(future1.get()).isEqualTo("value1");
  }

  @Test
  public void executeForcibly_supersededTaskCompletesLate_staleResultNotCached() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    SettableFuture<String> staleTask = SettableFuture.create();
    ListenableFuture<String> staleFuture = cache.executeIfNot("key1", () -> staleTask);
    ListenableFuture<String> freshFuture =
        cache.execute("key1", () -> immediateFuture("fresh"), /* force= */ true);
    assertThat(freshFuture.get()).isEqualTo("fresh");

    // The superseded execution completes only after the forced one.
    staleTask.set("stale");
    assertThat(staleFuture.get()).isEqualTo("stale");

    // Its potentially stale result must not be cached over the fresh one.
    ListenableFuture<String> future =
        cache.executeIfNot(
            "key1",
            () -> {
              throw new AssertionError("should not be executed");
            });
    assertThat(future.get()).isEqualTo("fresh");
  }

  @Test
  public void execute_supplierThrows_failedFuture() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    var error = new IllegalStateException("error");

    ListenableFuture<String> future =
        cache.executeIfNot(
            "key1",
            () -> {
              throw error;
            });

    var e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isSameInstanceAs(error);
    assertThat(cache.getInProgressTasks()).isEmpty();
    assertThat(cache.getFinishedTasks()).isEmpty();
  }

  @Test
  public void execute_taskFinished_noReExecution() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicInteger executionTimes = new AtomicInteger(0);
    ListenableFuture<String> future1 =
        cache.executeIfNot(
            "key1",
            () -> {
              executionTimes.incrementAndGet();
              return immediateFuture("value1");
            });
    assertThat(future1.get()).isEqualTo("value1");
    assertThat(cache.getFinishedTasks()).containsExactly("key1");

    ListenableFuture<String> future2 =
        cache.executeIfNot(
            "key1",
            () -> {
              executionTimes.incrementAndGet();
              return immediateFuture("value2");
            });

    assertThat(future2.get()).isEqualTo("value1");
    assertThat(executionTimes.get()).isEqualTo(1);
  }

  @Test
  public void executeForcibly_taskFinished_reExecution() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicInteger executionTimes = new AtomicInteger(0);
    ListenableFuture<String> future1 =
        cache.executeIfNot(
            "key1",
            () -> {
              executionTimes.incrementAndGet();
              return immediateFuture("value1");
            });
    assertThat(future1.get()).isEqualTo("value1");
    assertThat(cache.getFinishedTasks()).containsExactly("key1");

    SettableFuture<String> task = SettableFuture.create();
    ListenableFuture<String> future2 =
        cache.execute(
            "key1",
            () -> {
              executionTimes.incrementAndGet();
              return task;
            },
            /* force= */ true);

    assertThat(future2.isDone()).isFalse();
    assertThat(executionTimes.get()).isEqualTo(2);
    assertThat(cache.getInProgressTasks()).containsExactly("key1");
    assertThat(cache.getFinishedTasks()).isEmpty();
  }

  @Test
  public void invalidate_reExecutesOnNextCall() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicInteger executionTimes = new AtomicInteger(0);
    var unused =
        cache
            .executeIfNot("key1", () -> immediateFuture("value" + executionTimes.incrementAndGet()))
            .get();

    cache.invalidate("key1");

    assertThat(cache.getFinishedTasks()).isEmpty();
    assertThat(
            cache
                .executeIfNot(
                    "key1", () -> immediateFuture("value" + executionTimes.incrementAndGet()))
                .get())
        .isEqualTo("value2");
  }

  @Test
  public void put_shortCircuitsNextCall() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();

    cache.put("key1", "value1");

    assertThat(cache.getFinishedTasks()).containsExactly("key1");
    assertThat(
            cache
                .executeIfNot(
                    "key1",
                    () -> {
                      throw new AssertionError("should not be executed");
                    })
                .get())
        .isEqualTo("value1");
  }

  @Test
  public void put_nullValue_shortCircuitsNextCall() throws Exception {
    AsyncTaskCache<String, Void> cache = AsyncTaskCache.create();

    cache.put("key1", null);

    assertThat(cache.getFinishedTasks()).containsExactly("key1");
    assertThat(
            cache
                .executeIfNot(
                    "key1",
                    () -> {
                      throw new AssertionError("should not be executed");
                    })
                .get())
        .isNull();
  }

  @Test
  public void execute_cancel_taskCancelled() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    SettableFuture<String> task = SettableFuture.create();
    ListenableFuture<String> future = cache.executeIfNot("key1", () -> task);

    assertThat(future.cancel(/* mayInterruptIfRunning= */ true)).isTrue();

    assertThat(task.isCancelled()).isTrue();
    assertThat(cache.getInProgressTasks()).isEmpty();
    assertThat(cache.getFinishedTasks()).isEmpty();
  }

  @Test
  public void execute_cancelOneOfTwoCallers_taskNotCancelled() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    SettableFuture<String> task = SettableFuture.create();
    ListenableFuture<String> future1 = cache.executeIfNot("key1", () -> task);
    ListenableFuture<String> future2 = cache.executeIfNot("key1", () -> task);
    assertThat(cache.getSubscriberCount("key1")).isEqualTo(2);

    assertThat(future1.cancel(/* mayInterruptIfRunning= */ true)).isTrue();

    assertThat(future2.isDone()).isFalse();
    assertThat(task.isCancelled()).isFalse();
    assertThat(cache.getInProgressTasks()).containsExactly("key1");
    assertThat(cache.getSubscriberCount("key1")).isEqualTo(1);
  }

  @Test
  public void execute_cancelAllCallers_taskCancelled() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    SettableFuture<String> task = SettableFuture.create();
    ListenableFuture<String> future1 = cache.executeIfNot("key1", () -> task);
    ListenableFuture<String> future2 = cache.executeIfNot("key1", () -> task);

    assertThat(future1.cancel(/* mayInterruptIfRunning= */ true)).isTrue();
    assertThat(future2.cancel(/* mayInterruptIfRunning= */ true)).isTrue();

    assertThat(task.isCancelled()).isTrue();
    assertThat(cache.getInProgressTasks()).isEmpty();
    assertThat(cache.getFinishedTasks()).isEmpty();
  }

  @Test
  public void execute_multipleTasks_completeOne() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    SettableFuture<String> task1 = SettableFuture.create();
    SettableFuture<String> task2 = SettableFuture.create();
    ListenableFuture<String> future1 = cache.executeIfNot("key1", () -> task1);
    ListenableFuture<String> future2 = cache.executeIfNot("key2", () -> task2);

    task1.set("value1");

    assertThat(future1.get()).isEqualTo("value1");
    assertThat(future2.isDone()).isFalse();
    assertThat(cache.getInProgressTasks()).containsExactly("key2");
    assertThat(cache.getFinishedTasks()).containsExactly("key1");
  }

  @Test
  public void awaitInProgressTasks_completesAfterAllTasksFinish() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    SettableFuture<String> task1 = SettableFuture.create();
    SettableFuture<String> task2 = SettableFuture.create();
    var unused1 = cache.executeIfNot("key1", () -> task1);
    var unused2 = cache.executeIfNot("key2", () -> task2);

    Thread waiter = new Thread(() -> awaitInProgressTasksUninterruptibly(cache));
    waiter.start();
    task1.set("value1");
    assertThat(waiter.isAlive()).isTrue();
    task2.setException(new IOException("error"));
    waiter.join();

    // A failed task does not fail the wait.
    assertThat(cache.getInProgressTasks()).isEmpty();
  }

  private static void awaitInProgressTasksUninterruptibly(AsyncTaskCache<?, ?> cache) {
    try {
      cache.awaitInProgressTasks();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }

  @Test
  public void execute_pendingShutdown_getCancellationError() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    ListenableFuture<String> inProgress = cache.executeIfNot("key1", () -> SettableFuture.create());
    cache.shutdown();
    assertThat(cache.isShutdown()).isTrue();
    assertThat(cache.isTerminated()).isFalse();
    assertThat(inProgress.isDone()).isFalse();

    ListenableFuture<String> future = cache.executeIfNot("key2", () -> immediateFuture("value2"));

    assertThat(future.isCancelled()).isTrue();
    assertThrows(CancellationException.class, future::get);
  }

  @Test
  public void execute_afterShutdown_getCancellationError() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    cache.shutdown();
    cache.awaitTermination();

    ListenableFuture<String> future = cache.executeIfNot("key", () -> immediateFuture("value"));

    assertThat(future.isCancelled()).isTrue();
  }

  @Test
  public void shutdownNow_cancelInProgressTasks() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    SettableFuture<String> task = SettableFuture.create();
    ListenableFuture<String> future = cache.executeIfNot("key", () -> task);
    cache.shutdown();
    assertThat(cache.isShutdown()).isTrue();
    assertThat(cache.isTerminated()).isFalse();
    assertThat(future.isDone()).isFalse();

    cache.shutdownNow();
    cache.awaitTermination();

    assertThat(cache.isShutdown()).isTrue();
    assertThat(cache.isTerminated()).isTrue();
    assertThat(task.isCancelled()).isTrue();
    assertThat(future.isCancelled()).isTrue();
  }

  @Test
  public void awaitTermination_pendingShutdown_completeAfterTaskFinished() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    SettableFuture<String> task = SettableFuture.create();
    ListenableFuture<String> future = cache.executeIfNot("key", () -> task);
    cache.shutdown();
    assertThat(cache.isShutdown()).isTrue();
    assertThat(cache.isTerminated()).isFalse();

    task.set("value");
    cache.awaitTermination();

    assertThat(cache.isShutdown()).isTrue();
    assertThat(cache.isTerminated()).isTrue();
    assertThat(future.get()).isEqualTo("value");
    assertThat(cache.getInProgressTasks()).isEmpty();
    assertThat(cache.getFinishedTasks()).isEmpty();
  }

  @Test
  public void awaitTermination_afterShutdown_complete() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    cache.shutdownNow();
    cache.awaitTermination();

    cache.awaitTermination();

    assertThat(cache.isShutdown()).isTrue();
    assertThat(cache.isTerminated()).isTrue();
  }

  @Test
  public void execute_executeAndCancelLoop_noErrors() throws Throwable {
    runExecuteAndCancelLoop(/* force= */ false);
  }

  @Test
  public void executeForcibly_executeAndCancelLoop_noErrors() throws Throwable {
    runExecuteAndCancelLoop(/* force= */ true);
  }

  private static void runExecuteAndCancelLoop(boolean force) throws Throwable {
    int taskCount = 1000;
    int maxKey = 20;
    Random random = new Random();
    // Separate pools: callers block on their futures, so they must not compete for the threads
    // that complete the tasks.
    ExecutorService callerExecutor = Executors.newFixedThreadPool(64);
    ExecutorService taskExecutor = Executors.newVirtualThreadPerTaskExecutor();
    AsyncTaskCache<String, Void> cache = AsyncTaskCache.create();
    AtomicReference<Throwable> error = new AtomicReference<>(null);
    Semaphore semaphore = new Semaphore(0);

    for (int i = 0; i < taskCount; ++i) {
      callerExecutor.execute(
          () -> {
            try {
              ListenableFuture<Void> future =
                  cache.execute("key" + random.nextInt(maxKey), () -> newTask(taskExecutor), force);
              if (!future.isDone() && random.nextBoolean()) {
                future.cancel(true);
              } else {
                try {
                  var unused = future.get();
                } catch (CancellationException e) {
                  // Another caller cancelled the shared task.
                }
              }
            } catch (Throwable e) {
              if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
              }
              error.set(e);
            } finally {
              semaphore.release();
            }
          });
    }
    semaphore.acquire(taskCount);
    callerExecutor.shutdown();
    taskExecutor.shutdown();

    if (error.get() != null) {
      throw error.get();
    }
  }

  private static ListenableFuture<Void> newTask(ExecutorService taskExecutor) {
    SettableFuture<Void> future = SettableFuture.create();
    taskExecutor.execute(
        () -> {
          try {
            Thread.sleep((long) (Math.random() * 10));
            future.set(null);
          } catch (InterruptedException e) {
            future.setException(new IOException(e));
          }
        });
    return future;
  }
}
