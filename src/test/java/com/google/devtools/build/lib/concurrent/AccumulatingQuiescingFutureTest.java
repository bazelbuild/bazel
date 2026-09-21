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
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutor.safeDirectExecutor;
import static java.util.concurrent.ForkJoinPool.commonPool;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.concurrent.safeexecutor.RejectionHandlingRunnable;
import com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutor;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class AccumulatingQuiescingFutureTest {

  @Test
  public void accumulateFutureResult_success_foldsResultAndDecrementsCount() throws Exception {
    var future = new AccumulatingStringFuture();
    var subtask1 = SettableFuture.<String>create();
    var subtask2 = SettableFuture.<String>create();

    future.addFuture(subtask1, safeDirectExecutor());
    future.addFuture(subtask2, safeDirectExecutor());
    future.finishRegistration();

    assertThat(future.isDone()).isFalse();

    subtask1.set("hello");
    assertThat(future.isDone()).isFalse();

    subtask2.set("world");
    assertThat(future.isDone()).isTrue();
    assertThat(future.get()).containsExactly("hello", "world").inOrder();
  }

  @Test
  public void accumulateFutureResult_throwsException_recordsExceptionAndDecrementsCount()
      throws Exception {
    AtomicBoolean doneWithErrorCalled = new AtomicBoolean(false);
    var expectedError = new RuntimeException("failed during accumulateFutureResult");

    var future =
        new AccumulatingQuiescingFuture<String, String>(safeDirectExecutor()) {
          @Override
          protected void accumulateFutureResult(String result) {
            throw expectedError;
          }

          @Override
          protected String getValue() {
            return "unused";
          }

          @Override
          protected void doneWithError(
              @Nullable Throwable primaryCause, ImmutableList<Throwable> secondaryCauses) {
            doneWithErrorCalled.set(true);
          }
        };

    var subtask = SettableFuture.<String>create();
    future.addFuture(subtask, safeDirectExecutor());
    future.finishRegistration();

    subtask.set("trigger");

    assertThat(future.isDone()).isTrue();
    assertThat(doneWithErrorCalled.get()).isTrue();
    var thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(expectedError);
  }

  @Test
  public void onFailure_recordsExceptionAndDecrementsCount() throws Exception {
    AtomicBoolean doneWithErrorCalled = new AtomicBoolean(false);
    var expectedError = new RuntimeException("subtask failed");

    var future =
        new AccumulatingQuiescingFuture<String, String>(safeDirectExecutor()) {
          @Override
          protected void accumulateFutureResult(String result) {}

          @Override
          protected String getValue() {
            return "unused";
          }

          @Override
          protected void doneWithError(
              @Nullable Throwable primaryCause, ImmutableList<Throwable> secondaryCauses) {
            doneWithErrorCalled.set(true);
          }
        };

    var subtask = SettableFuture.<String>create();
    future.addFuture(subtask, safeDirectExecutor());
    future.finishRegistration();

    subtask.setException(expectedError);

    assertThat(future.isDone()).isTrue();
    assertThat(doneWithErrorCalled.get()).isTrue();
    var thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(expectedError);
  }

  @Test
  public void addFuture_rejection_rollsBackCountAndCompletesExceptionally() throws Exception {
    AtomicBoolean doneWithErrorCalled = new AtomicBoolean(false);
    var rejectionException = new RejectedExecutionException("callback rejected");

    SafeExecutor rejectingExecutor =
        new SafeExecutor() {
          @Override
          public void execute(RejectionHandlingRunnable task) {
            task.handleRejection(rejectionException);
          }

          @Override
          public <T> void addCallback(
              ListenableFuture<T> future, FutureCallback<? super T> callback) {
            throw rejectionException;
          }

          @Override
          public Executor getInternalUnsafeExecutor() {
            return null;
          }
        };

    var future =
        new AccumulatingQuiescingFuture<String, String>(safeDirectExecutor()) {
          @Override
          protected void accumulateFutureResult(String result) {}

          @Override
          protected String getValue() {
            return "result";
          }

          @Override
          protected void doneWithError(
              @Nullable Throwable primaryCause, ImmutableList<Throwable> secondaryCauses) {
            doneWithErrorCalled.set(true);
          }
        };

    var subtask = SettableFuture.<String>create();
    future.addFuture(subtask, rejectingExecutor);
    future.finishRegistration();

    assertThat(future.isDone()).isTrue();
    assertThat(doneWithErrorCalled.get()).isTrue();
    var thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(rejectionException);
  }

  @SuppressWarnings("AssertThrowsMinimizer") // safeDirectExecutor() does not throw exceptions
  @Test
  public void addFuture_nullFuture_throwsNullPointerException() {
    var future = new AccumulatingStringFuture();
    assertThrows(NullPointerException.class, () -> future.addFuture(null, safeDirectExecutor()));
  }

  @Test
  public void addFuture_nullExecutor_throwsNullPointerException() {
    var future = new AccumulatingStringFuture();
    var subtask = SettableFuture.<String>create();
    assertThrows(NullPointerException.class, () -> future.addFuture(subtask, null));
  }

  @Test
  public void multipleSubtasks_concurrentCompletion() throws Exception {
    var future = new AccumulatingStringFuture();
    int count = 50;
    List<SettableFuture<String>> subtasks = new ArrayList<>(count);

    for (int i = 0; i < count; i++) {
      var subtask = SettableFuture.<String>create();
      subtasks.add(subtask);
      future.addFuture(subtask, safeDirectExecutor());
    }
    future.finishRegistration();

    var latch = new CountDownLatch(count);
    for (int i = 0; i < count; i++) {
      final int index = i;
      commonPool()
          .execute(
              () -> {
                subtasks.get(index).set("item-" + index);
                latch.countDown();
              });
    }

    latch.await();
    assertThat(future.isDone()).isTrue();
    assertThat(future.get()).hasSize(count);
  }

  @Test
  public void multipleSubtasks_multipleFailures_preservesSecondaryExceptions() throws Exception {
    AtomicReference<Throwable> capturedPrimary = new AtomicReference<>();
    AtomicReference<ImmutableList<Throwable>> capturedSecondaries = new AtomicReference<>();

    var future =
        new AccumulatingQuiescingFuture<String, String>(safeDirectExecutor()) {
          @Override
          protected void accumulateFutureResult(String result) {}

          @Override
          protected String getValue() {
            return "unused";
          }

          @Override
          protected void doneWithError(
              @Nullable Throwable primaryCause, ImmutableList<Throwable> secondaryCauses) {
            capturedPrimary.set(primaryCause);
            capturedSecondaries.set(secondaryCauses);
          }
        };

    var error1 = new RuntimeException("error1");
    var error2 = new RuntimeException("error2");
    var subtask1 = SettableFuture.<String>create();
    var subtask2 = SettableFuture.<String>create();

    future.addFuture(subtask1, safeDirectExecutor());
    future.addFuture(subtask2, safeDirectExecutor());
    future.finishRegistration();

    subtask1.setException(error1);
    subtask2.setException(error2);

    assertThat(future.isDone()).isTrue();
    assertThat(capturedPrimary.get()).isSameInstanceAs(error1);
    assertThat(capturedSecondaries.get()).containsExactly(error2);

    var thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(error1);
  }

  @Test
  public void immediateFutures_resolveImmediately() throws Exception {
    var future = new AccumulatingStringFuture();
    future.addFuture(immediateFuture("a"), safeDirectExecutor());
    future.addFuture(immediateFuture("b"), safeDirectExecutor());
    future.finishRegistration();

    assertThat(future.isDone()).isTrue();
    assertThat(future.get()).containsExactly("a", "b").inOrder();
  }

  @Test
  public void accumulateFuture_cancelledSubtask_cancelsAccumulatingFuture() throws Exception {
    var future = new AccumulatingStringFuture();
    var subtask1 = SettableFuture.<String>create();
    var subtask2 = SettableFuture.<String>create();

    future.addFuture(subtask1, safeDirectExecutor());
    future.addFuture(subtask2, safeDirectExecutor());
    future.finishRegistration();

    assertThat(future.isDone()).isFalse();

    subtask1.cancel(/* mayInterruptIfRunning= */ false);

    assertThat(future.isDone()).isTrue();
    assertThat(future.isCancelled()).isTrue();
    assertThrows(CancellationException.class, future::get);
  }

  private static final class AccumulatingStringFuture
      extends AccumulatingQuiescingFuture<ImmutableList<String>, String> {
    private final ConcurrentLinkedQueue<String> collected = new ConcurrentLinkedQueue<>();

    private AccumulatingStringFuture() {
      super(safeDirectExecutor());
    }

    @Override
    protected void accumulateFutureResult(String result) {
      collected.add(result);
    }

    @Override
    protected ImmutableList<String> getValue() {
      return ImmutableList.copyOf(collected);
    }
  }
}
