// Copyright 2015 The Bazel Authors. All rights reserved.
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
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static org.junit.Assert.assertThrows;

import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import java.util.Random;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TaskDeduplicator}. */
@RunWith(JUnit4.class)
public class TaskDeduplicatorTest {

  @Test
  public void executeIfNew_taskFinished_completed() throws Exception {
    var deduplicator = new TaskDeduplicator<String, String>();
    var taskFuture = SettableFuture.<String>create();

    ListenableFuture<String> result = deduplicator.executeIfNew("key1", () -> taskFuture);

    taskFuture.set("value1");

    assertThat(result.get()).isEqualTo("value1");
  }

  @Test
  public void executeIfNew_taskHasError_propagateError() {
    var deduplicator = new TaskDeduplicator<String, String>();
    var taskFuture = SettableFuture.<String>create();
    var error = new IllegalStateException("error");

    ListenableFuture<String> result = deduplicator.executeIfNew("key1", () -> taskFuture);

    taskFuture.setException(error);

    var exception = assertThrows(ExecutionException.class, result::get);
    assertThat(exception).hasCauseThat().isSameInstanceAs(error);
  }

  @Test
  public void executeIfNew_taskInProgress_noReExecution() throws Exception {
    var deduplicator = new TaskDeduplicator<String, String>();
    var taskFuture = SettableFuture.<String>create();
    var executionTimes = new AtomicInteger(0);

    ListenableFuture<String> result1 =
        deduplicator.executeIfNew(
            "key1",
            () -> {
              executionTimes.incrementAndGet();
              return taskFuture;
            });

    // Second call with the same key should return the same future, not re-execute
    ListenableFuture<String> result2 =
        deduplicator.executeIfNew(
            "key1",
            () -> {
              throw new IllegalStateException("should not be called");
            });

    assertThat(result1.isDone()).isFalse();
    assertThat(result2.isDone()).isFalse();

    taskFuture.set("value1");

    assertThat(result1.get()).isEqualTo("value1");
    assertThat(result2.get()).isEqualTo("value1");
    assertThat(executionTimes.get()).isEqualTo(1);
  }

  @Test
  public void executeIfNew_taskFinished_reExecution() throws Exception {
    var deduplicator = new TaskDeduplicator<String, String>();
    var executionTimes = new AtomicInteger(0);

    // First execution
    ListenableFuture<String> result1 =
        deduplicator.executeIfNew(
            "key1",
            () -> {
              executionTimes.incrementAndGet();
              var future = SettableFuture.<String>create();
              future.set("value1");
              return future;
            });

    assertThat(result1.get()).isEqualTo("value1");
    assertThat(executionTimes.get()).isEqualTo(1);

    // Second execution after first is finished should re-execute
    ListenableFuture<String> result2 =
        deduplicator.executeIfNew(
            "key1",
            () -> {
              executionTimes.incrementAndGet();
              var future = SettableFuture.<String>create();
              future.set("value2");
              return future;
            });

    assertThat(result2.get()).isEqualTo("value2");
    assertThat(executionTimes.get()).isEqualTo(2);
  }

  @Test
  public void executeIfNew_taskCanceled_reExecution() {
    var deduplicator = new TaskDeduplicator<String, String>();
    var executionTimes = new AtomicInteger(0);

    // First execution
    ListenableFuture<String> result1 =
        deduplicator.executeIfNew(
            "key1",
            () -> {
              executionTimes.incrementAndGet();
              return Futures.immediateCancelledFuture();
            });

    assertThat(result1.isCancelled()).isTrue();
    assertThat(executionTimes.get()).isEqualTo(1);

    // Second execution after first is finished should re-execute
    ListenableFuture<String> result2 =
        deduplicator.executeIfNew(
            "key1",
            () -> {
              executionTimes.incrementAndGet();
              return Futures.immediateCancelledFuture();
            });

    assertThat(result2.isCancelled()).isTrue();
    assertThat(executionTimes.get()).isEqualTo(2);
  }

  @Test
  public void executeIfNew_cancel_cancelled() throws Exception {
    var deduplicator = new TaskDeduplicator<String, String>();
    var taskFuture = SettableFuture.<String>create();

    ListenableFuture<String> result = deduplicator.executeIfNew("key1", () -> taskFuture);

    result.cancel(true);

    assertThat(result.isCancelled()).isTrue();
    assertThrows(CancellationException.class, result::get);
  }

  @Test
  public void executeIfNew_cancelWhenMultipleFutures_notCancelled() throws Exception {
    var deduplicator = new TaskDeduplicator<String, String>();
    var taskFuture = SettableFuture.<String>create();

    ListenableFuture<String> result1 = deduplicator.executeIfNew("key1", () -> taskFuture);
    ListenableFuture<String> result2 = deduplicator.executeIfNew("key1", () -> taskFuture);

    // Cancel one future multiple times
    result1.cancel(true);
    result1.cancel(true);

    // The task should still be running for the second future
    assertThat(result1.isCancelled()).isTrue();
    assertThat(result2.isDone()).isFalse();

    taskFuture.set("value1");

    assertThat(result2.get()).isEqualTo("value1");
  }

  @Test
  public void executeIfNew_cancelWhenMultipleFutures_allCancelled() {
    var deduplicator = new TaskDeduplicator<String, String>();
    var taskFuture = SettableFuture.<String>create();

    ListenableFuture<String> result1 = deduplicator.executeIfNew("key1", () -> taskFuture);
    ListenableFuture<String> result2 = deduplicator.executeIfNew("key1", () -> taskFuture);

    // Cancel both futures
    result1.cancel(true);
    result2.cancel(true);

    assertThat(result1.isCancelled()).isTrue();
    assertThat(result2.isCancelled()).isTrue();
  }

  @Test
  public void executeIfNew_multipleTasks_completeOne() throws Exception {
    var deduplicator = new TaskDeduplicator<String, String>();
    var taskFuture1 = SettableFuture.<String>create();
    var taskFuture2 = SettableFuture.<String>create();

    ListenableFuture<String> result1 = deduplicator.executeIfNew("key1", () -> taskFuture1);
    ListenableFuture<String> result2 = deduplicator.executeIfNew("key2", () -> taskFuture2);

    taskFuture1.set("value1");

    assertThat(result1.get()).isEqualTo("value1");
    assertThat(result2.isDone()).isFalse();

    taskFuture2.set("value2");

    assertThat(result2.get()).isEqualTo("value2");
  }

  @Test
  public void executeIfNeeded_executeAndCancelLoop_noErrors() {
    int taskCount = 1000;
    int maxKey = 20;
    var random = new Random();
    var deduplicator = new TaskDeduplicator<String, Void>();
    var throwables = new ConcurrentLinkedQueue<Throwable>();

    try (var taskExecutorService = Executors.newFixedThreadPool(50);
        var testExecutorService = Executors.newVirtualThreadPerTaskExecutor()) {
      for (int i = 0; i < taskCount; ++i) {
        testExecutorService.execute(
            () -> {
              try {
                ListenableFuture<Void> future =
                    deduplicator.executeIfNew(
                        "key" + random.nextInt(maxKey),
                        () ->
                            Futures.submitAsync(
                                () -> {
                                  Thread.sleep((long) (Math.random() * 100));
                                  return immediateVoidFuture();
                                },
                                taskExecutorService));

                if (!future.isDone() && random.nextBoolean()) {
                  future.cancel(true);
                } else {
                  future.get();
                }
              } catch (Throwable e) {
                if (e instanceof InterruptedException) {
                  Thread.currentThread().interrupt();
                }
                throwables.add(e);
              }
            });
      }
    }

    if (!throwables.isEmpty()) {
      var combinedError = new AssertionError();
      for (var throwable : throwables) {
        combinedError.addSuppressed(throwable);
      }
      throw combinedError;
    }
  }

  @Test
  public void executeIfNew_taskCompletedBeforeSecondCall_bothGetSameResult() throws Exception {
    var deduplicator = new TaskDeduplicator<String, String>();
    var taskStarted = new AtomicBoolean(false);

    // First call - task completes immediately in-flight
    ListenableFuture<String> result1 =
        deduplicator.executeIfNew(
            "key1",
            () -> {
              taskStarted.set(true);
              var future = SettableFuture.<String>create();
              future.set("value1");
              return future;
            });

    assertThat(result1.get()).isEqualTo("value1");
    assertThat(taskStarted.get()).isTrue();
    taskStarted.set(false);

    // Second call - task should execute again since first is done
    ListenableFuture<String> result2 =
        deduplicator.executeIfNew(
            "key1",
            () -> {
              taskStarted.set(true);
              var future = SettableFuture.<String>create();
              future.set("value2");
              return future;
            });

    assertThat(result2.get()).isEqualTo("value2");
    assertThat(taskStarted.get()).isTrue();
  }

  @Test
  public void executeIfNew_errorInAsyncCallable_propagated() {
    var deduplicator = new TaskDeduplicator<String, String>();
    var expectedException = new RuntimeException("task creation failed");

    var actualException =
        assertThrows(
            RuntimeException.class,
            () ->
                deduplicator.executeIfNew(
                    "key1",
                    () -> {
                      throw expectedException;
                    }));
    assertThat(actualException).isSameInstanceAs(expectedException);
  }
}
