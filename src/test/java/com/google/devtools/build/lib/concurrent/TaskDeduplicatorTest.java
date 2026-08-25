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
import static org.junit.Assert.assertThrows;

import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import java.util.Random;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TaskDeduplicator}. */
@RunWith(JUnit4.class)
@SuppressWarnings("AllowVirtualThreads")
public class TaskDeduplicatorTest {

  // Executions in the tests below that exercise attributes are described by an integer strength,
  // mirroring the use case in MerkleTreeComputer: an execution started with a higher strength
  // produces everything a weaker one would. A caller may thus join an ongoing execution that is at
  // least as strong as what it needs, but never a weaker one.
  private static final int WEAK = 0;
  private static final int STRONG = 1;

  // Tests that don't exercise attributes accept any ongoing execution.
  private static final Predicate<Object> JOIN_ANY = unusedAttributes -> true;

  private static Predicate<Integer> atLeast(int strength) {
    return ongoingStrength -> ongoingStrength >= strength;
  }

  @Test
  public void execute_taskFinished_completed() throws Exception {
    var deduplicator = new TaskDeduplicator<String, Void, String>();
    var taskFuture = SettableFuture.<String>create();

    ListenableFuture<String> result =
        deduplicator.execute("key1", /* attributes= */ null, JOIN_ANY, () -> taskFuture);

    taskFuture.set("value1");

    assertThat(result.get()).isEqualTo("value1");
  }

  @Test
  public void execute_taskHasError_propagateError() {
    var deduplicator = new TaskDeduplicator<String, Void, String>();
    var taskFuture = SettableFuture.<String>create();
    var error = new IllegalStateException("error");

    ListenableFuture<String> result =
        deduplicator.execute("key1", /* attributes= */ null, JOIN_ANY, () -> taskFuture);

    taskFuture.setException(error);

    var exception = assertThrows(ExecutionException.class, result::get);
    assertThat(exception).hasCauseThat().isSameInstanceAs(error);
  }

  @Test
  public void execute_taskInProgress_noReExecution() throws Exception {
    var deduplicator = new TaskDeduplicator<String, Void, String>();
    var taskFuture = SettableFuture.<String>create();
    var executionTimes = new AtomicInteger(0);

    ListenableFuture<String> result1 =
        deduplicator.execute(
            "key1",
            /* attributes= */ null,
            JOIN_ANY,
            () -> {
              executionTimes.incrementAndGet();
              return taskFuture;
            });

    // Second call with the same key should return the same future, not re-execute
    ListenableFuture<String> result2 =
        deduplicator.execute(
            "key1",
            /* attributes= */ null,
            JOIN_ANY,
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
  public void execute_taskFinished_reExecution() throws Exception {
    var deduplicator = new TaskDeduplicator<String, Void, String>();
    var executionTimes = new AtomicInteger(0);

    // First execution
    ListenableFuture<String> result1 =
        deduplicator.execute(
            "key1",
            /* attributes= */ null,
            JOIN_ANY,
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
        deduplicator.execute(
            "key1",
            /* attributes= */ null,
            JOIN_ANY,
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
  public void execute_taskCanceled_reExecution() {
    var deduplicator = new TaskDeduplicator<String, Void, String>();
    var executionTimes = new AtomicInteger(0);

    // First execution
    ListenableFuture<String> result1 =
        deduplicator.execute(
            "key1",
            /* attributes= */ null,
            JOIN_ANY,
            () -> {
              executionTimes.incrementAndGet();
              return Futures.immediateCancelledFuture();
            });

    assertThat(result1.isCancelled()).isTrue();
    assertThat(executionTimes.get()).isEqualTo(1);

    // Second execution after first is finished should re-execute
    ListenableFuture<String> result2 =
        deduplicator.execute(
            "key1",
            /* attributes= */ null,
            JOIN_ANY,
            () -> {
              executionTimes.incrementAndGet();
              return Futures.immediateCancelledFuture();
            });

    assertThat(result2.isCancelled()).isTrue();
    assertThat(executionTimes.get()).isEqualTo(2);
  }

  @Test
  public void execute_cancel_cancelled() throws Exception {
    var deduplicator = new TaskDeduplicator<String, Void, String>();
    var taskFuture = SettableFuture.<String>create();

    ListenableFuture<String> result =
        deduplicator.execute("key1", /* attributes= */ null, JOIN_ANY, () -> taskFuture);

    result.cancel(true);

    assertThat(result.isCancelled()).isTrue();
    assertThrows(CancellationException.class, result::get);
  }

  @Test
  public void execute_cancelWhenMultipleFutures_notCancelled() throws Exception {
    var deduplicator = new TaskDeduplicator<String, Void, String>();
    var taskFuture = SettableFuture.<String>create();

    ListenableFuture<String> result1 =
        deduplicator.execute("key1", /* attributes= */ null, JOIN_ANY, () -> taskFuture);
    ListenableFuture<String> result2 =
        deduplicator.execute("key1", /* attributes= */ null, JOIN_ANY, () -> taskFuture);

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
  public void execute_cancelWhenMultipleFutures_allCancelled() {
    var deduplicator = new TaskDeduplicator<String, Void, String>();
    var taskFuture = SettableFuture.<String>create();

    ListenableFuture<String> result1 =
        deduplicator.execute("key1", /* attributes= */ null, JOIN_ANY, () -> taskFuture);
    ListenableFuture<String> result2 =
        deduplicator.execute("key1", /* attributes= */ null, JOIN_ANY, () -> taskFuture);

    // Cancel both futures
    result1.cancel(true);
    result2.cancel(true);

    assertThat(result1.isCancelled()).isTrue();
    assertThat(result2.isCancelled()).isTrue();
    assertThat(taskFuture.isCancelled()).isTrue();
  }

  @Test
  public void execute_multipleTasks_completeOne() throws Exception {
    var deduplicator = new TaskDeduplicator<String, Void, String>();
    var taskFuture1 = SettableFuture.<String>create();
    var taskFuture2 = SettableFuture.<String>create();

    ListenableFuture<String> result1 =
        deduplicator.execute("key1", /* attributes= */ null, JOIN_ANY, () -> taskFuture1);
    ListenableFuture<String> result2 =
        deduplicator.execute("key2", /* attributes= */ null, JOIN_ANY, () -> taskFuture2);

    taskFuture1.set("value1");

    assertThat(result1.get()).isEqualTo("value1");
    assertThat(result2.isDone()).isFalse();

    taskFuture2.set("value2");

    assertThat(result2.get()).isEqualTo("value2");
  }

  @Test
  public void execute_errorInAsyncCallable_propagated() {
    var deduplicator = new TaskDeduplicator<String, Void, String>();
    var expectedException = new RuntimeException("task creation failed");

    var actualException =
        assertThrows(
            RuntimeException.class,
            () ->
                deduplicator.execute(
                    "key1",
                    /* attributes= */ null,
                    JOIN_ANY,
                    () -> {
                      throw expectedException;
                    }));
    assertThat(actualException).isSameInstanceAs(expectedException);
  }

  @Test
  public void execute_taskCanceledIndependently_allCallersCanceled() {
    var deduplicator = new TaskDeduplicator<String, Void, String>();
    var taskFuture = SettableFuture.<String>create();

    ListenableFuture<String> result1 =
        deduplicator.execute("key1", /* attributes= */ null, JOIN_ANY, () -> taskFuture);
    ListenableFuture<String> result2 =
        deduplicator.execute("key1", /* attributes= */ null, JOIN_ANY, () -> taskFuture);

    // The task is canceled by whoever runs it rather than by any of its callers, e.g. because its
    // executor was shut down.
    taskFuture.cancel(/* mayInterruptIfRunning= */ true);

    assertThat(result1.isCancelled()).isTrue();
    assertThat(result2.isCancelled()).isTrue();
    assertThrows(CancellationException.class, result1::get);
  }

  @Test
  public void execute_cancelOneFuture_otherStillSeesError() {
    var deduplicator = new TaskDeduplicator<String, Void, String>();
    var taskFuture = SettableFuture.<String>create();
    var error = new IllegalStateException("error");

    ListenableFuture<String> result1 =
        deduplicator.execute("key1", /* attributes= */ null, JOIN_ANY, () -> taskFuture);
    ListenableFuture<String> result2 =
        deduplicator.execute("key1", /* attributes= */ null, JOIN_ANY, () -> taskFuture);

    result1.cancel(true);
    taskFuture.setException(error);

    assertThat(result1.isCancelled()).isTrue();
    var exception = assertThrows(ExecutionException.class, result2::get);
    assertThat(exception).hasCauseThat().isSameInstanceAs(error);
  }

  /**
   * Canceling a future must release exactly one reference to the shared task, also after the task
   * has completed.
   *
   * <p>A stray second release is otherwise only observable through a rare race: a concurrent {@link
   * TaskDeduplicator#execute} that looks up the shared task just before it completes and retains it
   * just after would be refused a perfectly good result and would redo the work.
   */
  @Test
  public void execute_cancelOneFuture_releasesExactlyOneReference() throws Exception {
    var deduplicator = new TaskDeduplicator<String, Void, String>();
    var taskFuture = SettableFuture.<String>create();

    ListenableFuture<String> result1 =
        deduplicator.execute("key1", /* attributes= */ null, JOIN_ANY, () -> taskFuture);
    ListenableFuture<String> result2 =
        deduplicator.execute("key1", /* attributes= */ null, JOIN_ANY, () -> taskFuture);
    ListenableFuture<String> result3 =
        deduplicator.execute("key1", /* attributes= */ null, JOIN_ANY, () -> taskFuture);
    var sharedTask = deduplicator.ongoingExecutionForTesting("key1");
    assertThat(sharedTask.activeUses()).isEqualTo(3);

    result1.cancel(true);
    assertThat(sharedTask.activeUses()).isEqualTo(2);

    taskFuture.set("value1");

    assertThat(result2.get()).isEqualTo("value1");
    assertThat(result3.get()).isEqualTo("value1");
    assertThat(sharedTask.activeUses()).isEqualTo(2);
  }

  @Test
  public void execute_canJoinAcceptsOngoingExecution_joins() throws Exception {
    var deduplicator = new TaskDeduplicator<String, Integer, String>();
    var taskFuture = SettableFuture.<String>create();

    ListenableFuture<String> result1 =
        deduplicator.execute("key1", STRONG, atLeast(STRONG), () -> taskFuture);
    // A weaker caller is satisfied by the stronger ongoing execution.
    ListenableFuture<String> result2 =
        deduplicator.execute(
            "key1",
            WEAK,
            atLeast(WEAK),
            () -> {
              throw new IllegalStateException("should not be called");
            });

    taskFuture.set("value1");

    assertThat(result1.get()).isEqualTo("value1");
    assertThat(result2.get()).isEqualTo("value1");
  }

  @Test
  public void execute_canJoinRejectsOngoingExecution_startsNewExecution() throws Exception {
    var deduplicator = new TaskDeduplicator<String, Integer, String>();
    var weakTaskFuture = SettableFuture.<String>create();
    var strongTaskFuture = SettableFuture.<String>create();
    var executionTimes = new AtomicInteger(0);

    ListenableFuture<String> weakResult =
        deduplicator.execute(
            "key1",
            WEAK,
            atLeast(WEAK),
            () -> {
              executionTimes.incrementAndGet();
              return weakTaskFuture;
            });
    // The ongoing execution is too weak, so this caller has to start its own.
    ListenableFuture<String> strongResult =
        deduplicator.execute(
            "key1",
            STRONG,
            atLeast(STRONG),
            () -> {
              executionTimes.incrementAndGet();
              return strongTaskFuture;
            });

    assertThat(executionTimes.get()).isEqualTo(2);

    // The replaced execution keeps running for the caller that joined it.
    strongTaskFuture.set("strong");
    assertThat(strongResult.get()).isEqualTo("strong");
    assertThat(weakResult.isDone()).isFalse();

    weakTaskFuture.set("weak");
    assertThat(weakResult.get()).isEqualTo("weak");
  }

  @Test
  public void execute_newExecutionTakesOverSlot_laterCallersJoinIt() throws Exception {
    var deduplicator = new TaskDeduplicator<String, Integer, String>();
    var weakTaskFuture = SettableFuture.<String>create();
    var strongTaskFuture = SettableFuture.<String>create();

    var unused = deduplicator.execute("key1", WEAK, atLeast(WEAK), () -> weakTaskFuture);
    ListenableFuture<String> strongResult =
        deduplicator.execute("key1", STRONG, atLeast(STRONG), () -> strongTaskFuture);
    // The weak execution is no longer joinable, but the strong one that replaced it is.
    ListenableFuture<String> lateWeakResult =
        deduplicator.execute(
            "key1",
            WEAK,
            atLeast(WEAK),
            () -> {
              throw new IllegalStateException("should not be called");
            });

    strongTaskFuture.set("strong");

    assertThat(strongResult.get()).isEqualTo("strong");
    assertThat(lateWeakResult.get()).isEqualTo("strong");
  }

  @Test
  public void execute_canJoinRejects_cancelWhenMultipleFutures_notCancelled() throws Exception {
    var deduplicator = new TaskDeduplicator<String, Integer, String>();
    var taskFuture = SettableFuture.<String>create();

    ListenableFuture<String> result1 =
        deduplicator.execute("key1", STRONG, atLeast(STRONG), () -> taskFuture);
    ListenableFuture<String> result2 =
        deduplicator.execute("key1", WEAK, atLeast(WEAK), () -> taskFuture);

    result1.cancel(true);

    assertThat(result1.isCancelled()).isTrue();
    assertThat(result2.isDone()).isFalse();

    taskFuture.set("value1");

    assertThat(result2.get()).isEqualTo("value1");
  }

  /**
   * A caller must never end up on an execution that its {@code canJoin} predicate rejects, not even
   * one that a concurrent call starts while it is looking up the key.
   *
   * <p>This is what makes {@code MerkleTreeComputer}'s {@code KEEP_AND_REUPLOAD} policy reliable:
   * joining a {@code KEEP} computation would skip the reupload of the blobs that the remote cache
   * lost.
   */
  @Test
  public void execute_concurrentWeakerExecution_neverJoinsIt() throws Exception {
    int rounds = 2000;
    var failures = new ConcurrentLinkedQueue<String>();

    try (var testExecutorService = Executors.newVirtualThreadPerTaskExecutor()) {
      for (int round = 0; round < rounds; ++round) {
        var deduplicator = new TaskDeduplicator<String, Integer, Integer>();
        var start = new CountDownLatch(1);
        var done = new CountDownLatch(2);

        testExecutorService.execute(
            () -> {
              try {
                start.await();
                // Every execution completes with the strength it was started with.
                ListenableFuture<Integer> result =
                    deduplicator.execute(
                        "key1", STRONG, atLeast(STRONG), () -> Futures.immediateFuture(STRONG));
                if (result.get() != STRONG) {
                  failures.add("joined an execution of strength " + result.get());
                }
              } catch (Throwable e) {
                if (e instanceof InterruptedException) {
                  Thread.currentThread().interrupt();
                }
                failures.add(e.toString());
              } finally {
                done.countDown();
              }
            });
        testExecutorService.execute(
            () -> {
              try {
                start.await();
                var taskFuture = SettableFuture.<Integer>create();
                ListenableFuture<Integer> result =
                    deduplicator.execute("key1", WEAK, atLeast(WEAK), () -> taskFuture);
                // Leave the execution in flight for a moment so that it overlaps with the
                // concurrent stronger call.
                taskFuture.set(WEAK);
                var unused = result.get();
              } catch (Throwable e) {
                if (e instanceof InterruptedException) {
                  Thread.currentThread().interrupt();
                }
                failures.add(e.toString());
              } finally {
                done.countDown();
              }
            });

        start.countDown();
        done.await();
      }
    }

    assertThat(failures).isEmpty();
  }

  @Test
  public void execute_executeAndCancelLoop_noErrors() {
    int taskCount = 1000;
    int maxKey = 20;
    var random = new Random();
    var deduplicator = new TaskDeduplicator<String, Void, Void>();
    var throwables = new ConcurrentLinkedQueue<Throwable>();

    try (var taskExecutorService = Executors.newFixedThreadPool(50);
        var testExecutorService = Executors.newVirtualThreadPerTaskExecutor()) {
      for (int i = 0; i < taskCount; ++i) {
        testExecutorService.execute(
            () -> {
              try {
                ListenableFuture<Void> future =
                    deduplicator.execute(
                        "key" + random.nextInt(maxKey),
                        /* attributes= */ null,
                        JOIN_ANY,
                        () ->
                            Futures.submit(
                                () -> {
                                  Thread.sleep((long) (Math.random() * 100));
                                  return (Void) null;
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
  public void mixedOperations_executeAndCancelLoop_noErrors() {
    int taskCount = 20000;
    int maxKey = 20;
    var random = new Random();
    var deduplicator = new TaskDeduplicator<String, Integer, Integer>();
    var throwables = new ConcurrentLinkedQueue<Throwable>();

    try (var taskExecutorService = Executors.newFixedThreadPool(8);
        var testExecutorService = Executors.newVirtualThreadPerTaskExecutor()) {
      for (int i = 0; i < taskCount; ++i) {
        testExecutorService.execute(
            () -> {
              try {
                String key = "key" + random.nextInt(maxKey);
                int strength = random.nextInt(3);
                ListenableFuture<Integer> future =
                    deduplicator.execute(
                        key,
                        strength,
                        atLeast(strength),
                        () ->
                            Futures.submit(
                                () -> {
                                  Thread.sleep(0, 200_000);
                                  return strength;
                                },
                                taskExecutorService));
                if (!future.isDone() && random.nextBoolean()) {
                  future.cancel(true);
                } else {
                  // A joined execution must always be at least as strong as what was requested.
                  assertThat(future.get()).isAtLeast(strength);
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
}
