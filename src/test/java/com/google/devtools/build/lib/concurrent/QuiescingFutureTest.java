// Copyright 2024 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutor.safeDirectExecutor;
import static java.util.concurrent.ForkJoinPool.commonPool;
import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.concurrent.safeexecutor.RejectionHandlingRunnable;
import com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutor;
import com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutorOwner;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class QuiescingFutureTest {

  @Test
  public void immediateCompletion() throws Exception {
    var future = new ConstantQuiescingFuture();
    assertThat(future.isDone()).isFalse();

    future.decrement();

    assertThat(future.isDone()).isTrue();
    assertThat(future.get()).isEqualTo("result");
  }

  @Test
  public void finishRegistration_completesFuture() throws Exception {
    var future = new ConstantQuiescingFuture();
    assertThat(future.isDone()).isFalse();

    future.finishRegistration();

    assertThat(future.isDone()).isTrue();
    assertThat(future.get()).isEqualTo("result");
  }

  @Test
  public void exceptionPropagates() throws Exception {
    var future = new ConstantQuiescingFuture();
    assertThat(future.isDone()).isFalse();

    var error = new Throwable("failure");
    future.notifyException(error);

    assertThat(future.isDone()).isTrue();
    var thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(error);
  }

  @Test
  public void transientZeroing_doesNotPrematurelyComplete() throws Exception {
    var future = new ConstantQuiescingFuture();
    assertThat(future.isDone()).isFalse();

    future.increment();
    future.decrement(); // count reaches "0"

    assertThat(future.isDone()).isFalse();

    future.decrement();
    assertThat(future.isDone()).isTrue();
    assertThat(future.get()).isEqualTo("result");
  }

  private static final class ConstantQuiescingFuture extends QuiescingFuture<String> {
    private ConstantQuiescingFuture() {
      super(safeDirectExecutor());
    }

    @Override
    protected String getValue() {
      return "result";
    }
  }

  @Test
  public void concurrentRecursiveTasks() throws Exception {
    AtomicInteger completionCount = new AtomicInteger();
    var future = new CountingQuiescingFuture(completionCount);

    commonPool().execute(new RecurrentTask(future, completionCount, 0));
    future.decrement();

    // If this passes, it means the counter value at the time of completion included all tasks,
    // showing that there was no early completion.
    assertThat(future.get()).isEqualTo(1023);
  }

  private static final int MAX_DEPTH = 9;

  private static final class RecurrentTask implements Runnable {
    private final QuiescingFuture<?> future;
    private final AtomicInteger counter;
    private final int depth;

    private RecurrentTask(QuiescingFuture<?> future, AtomicInteger counter, int depth) {
      this.future = future;
      this.counter = counter;
      this.depth = depth;

      future.increment();
    }

    @Override
    public void run() {
      if (depth < MAX_DEPTH) {
        for (int i = 0; i < 2; i++) {
          commonPool().execute(new RecurrentTask(future, counter, depth + 1));
        }
      }
      counter.getAndIncrement();
      future.decrement();
    }
  }

  private static final class CountingQuiescingFuture extends QuiescingFuture<Integer> {
    private final AtomicInteger counter;

    private CountingQuiescingFuture(AtomicInteger counter) {
      super(safeDirectExecutor());
      this.counter = counter;
    }

    @Override
    protected Integer getValue() {
      return counter.get();
    }
  }

  @Test
  public void notifyException_callsDoneWithError_notGetValue() throws Exception {
    AtomicBoolean doneWithErrorCalled = new AtomicBoolean(false);
    AtomicBoolean getValueCalled = new AtomicBoolean(false);
    var future = new TestQuiescingFuture(doneWithErrorCalled, getValueCalled);

    var error = new RuntimeException("oops");
    future.notifyException(error);

    assertThat(future.isDone()).isTrue();
    assertThat(doneWithErrorCalled.get()).isTrue();
    assertThat(getValueCalled.get()).isFalse();

    // Future should be in an error state
    var thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(error);
  }

  @Test
  public void notifyException_multipleErrors_callsDoneWithErrorOnce() throws Exception {
    AtomicInteger doneWithErrorCallCount = new AtomicInteger(0);
    AtomicBoolean getValueCalled = new AtomicBoolean(false);
    var future =
        new TestQuiescingFuture(
            () -> doneWithErrorCallCount.getAndIncrement(), () -> getValueCalled.set(true));

    future.increment(); // Add an extra task
    future.notifyException(new RuntimeException("error1"));
    assertThat(future.isDone()).isTrue(); // Done after first exception

    future.notifyException(new RuntimeException("error2")); // Second error

    // Wait for all decrements to complete
    assertThrows(ExecutionException.class, future::get);
    assertThat(doneWithErrorCallCount.get()).isEqualTo(1);
    assertThat(getValueCalled.get()).isFalse();
  }

  @Test
  public void mixNotifyExceptionAndDecrement_callsDoneWithError() throws Exception {
    AtomicBoolean doneWithErrorCalled = new AtomicBoolean(false);
    AtomicBoolean getValueCalled = new AtomicBoolean(false);
    var future = new TestQuiescingFuture(doneWithErrorCalled, getValueCalled);

    future.increment();
    future.increment();

    future.notifyException(new RuntimeException("error"));
    assertThat(future.isDone()).isTrue(); // Done after first exception

    future.decrement();
    assertThat(doneWithErrorCalled.get()).isFalse(); // Not called yet

    future.decrement();
    assertThat(doneWithErrorCalled.get()).isTrue(); // Called after all decrements
    assertThat(getValueCalled.get()).isFalse();
  }

  @Test
  public void executorTest() throws Exception {
    AtomicBoolean executorCalled = new AtomicBoolean(false);
    var threadPool =
        new ThreadPoolExecutor(1, 1, 0L, MILLISECONDS, new LinkedBlockingQueue<>()) {
          @Override
          public void execute(Runnable command) {
            executorCalled.set(true);
            super.execute(command);
          }
        };
    try {
      var future =
          new QuiescingFuture<String>(new SafeExecutorOwner(threadPool)) {
            @Override
            protected String getValue() {
              return "executed";
            }
          };

      future.decrement();
      assertThat(future.get()).isEqualTo("executed");
      assertThat(executorCalled.get()).isTrue();
    } finally {
      threadPool.shutdown();
    }
  }

  @Test
  public void concurrentNotifyExceptionAndDecrement() throws Exception {
    CountDownLatch doneWithErrorCalled = new CountDownLatch(1);
    AtomicBoolean getValueCalled = new AtomicBoolean(false);
    var future =
        new TestQuiescingFuture(
            () -> doneWithErrorCalled.countDown(), () -> getValueCalled.set(true));

    var error = new RuntimeException("concurrent error");
    for (int i = 0; i < 10; i++) {
      future.increment();
      final int capturedIndex = i;
      commonPool()
          .execute(
              () -> {
                if (capturedIndex % 2 == 0) {
                  future.notifyException(error);
                } else {
                  future.decrement();
                }
              });
    }
    future.decrement(); // Clears the pre-increment.

    // Waits for completion
    var thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(error);

    assertThat(future.isDone()).isTrue();

    assertThat(doneWithErrorCalled.await(60, SECONDS)).isTrue();
  }

  @Test
  public void executeSubtask_runnable_success() throws Exception {
    AtomicBoolean subtaskExecuted = new AtomicBoolean(false);
    var future = new ConstantQuiescingFuture();

    future.executeSubtask(() -> subtaskExecuted.set(true), safeDirectExecutor());
    future.finishRegistration();

    assertThat(subtaskExecuted.get()).isTrue();
    assertThat(future.isDone()).isTrue();
    assertThat(future.get()).isEqualTo("result");
  }

  @Test
  public void executeSubtask_throwsException_propagatesErrorAndInvokesDoneWithError()
      throws Exception {
    AtomicBoolean doneWithErrorCalled = new AtomicBoolean(false);
    AtomicBoolean getValueCalled = new AtomicBoolean(false);
    var future = new TestQuiescingFuture(doneWithErrorCalled, getValueCalled);

    var error = new RuntimeException("subtask failure");
    future.executeSubtask(
        () -> {
          throw error;
        },
        safeDirectExecutor());
    future.finishRegistration();

    assertThat(future.isDone()).isTrue();
    assertThat(doneWithErrorCalled.get()).isTrue();
    assertThat(getValueCalled.get()).isFalse();
    var thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(error);
  }

  @Test
  public void executeSubtask_rejection_recordsExceptionAndCompletesWithError() throws Exception {
    AtomicBoolean doneWithErrorCalled = new AtomicBoolean(false);
    AtomicBoolean getValueCalled = new AtomicBoolean(false);
    var future = new TestQuiescingFuture(doneWithErrorCalled, getValueCalled);

    var rejectionException = new RejectedExecutionException("rejected");
    SafeExecutor rejectingExecutor =
        new SafeExecutor() {
          @Override
          public void execute(RejectionHandlingRunnable task) {
            task.handleRejection(rejectionException);
          }

          @Override
          public <T> void addCallback(
              ListenableFuture<T> future, FutureCallback<? super T> callback) {}

          @Override
          public Executor getInternalUnsafeExecutor() {
            return null;
          }
        };

    future.executeSubtask(() -> {}, rejectingExecutor);
    future.finishRegistration();

    assertThat(future.isDone()).isTrue();
    assertThat(doneWithErrorCalled.get()).isTrue();
    var thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(rejectionException);
  }

  @Test
  public void doneWithError_receivesPrimaryAndSecondaryExceptions() throws Exception {
    AtomicReference<Throwable> capturedPrimary = new AtomicReference<>();
    AtomicReference<ImmutableList<Throwable>> capturedSecondaries = new AtomicReference<>();

    var future =
        new QuiescingFuture<String>(safeDirectExecutor()) {
          @Override
          protected String getValue() {
            return "result";
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
    var error3 = new RuntimeException("error3");

    future.increment();
    future.increment();

    future.notifyException(error1);
    future.notifyException(error2);
    future.notifyException(error3);

    assertThat(future.isDone()).isTrue();
    assertThat(capturedPrimary.get()).isSameInstanceAs(error1);
    assertThat(capturedSecondaries.get()).containsExactly(error2, error3).inOrder();
  }

  @Test
  public void cancelBeforeQuiescence_invokesDoneWithError() throws Exception {
    AtomicBoolean doneWithErrorCalled = new AtomicBoolean(false);
    var future =
        new QuiescingFuture<String>(safeDirectExecutor()) {
          @Override
          protected String getValue() {
            return "result";
          }

          @Override
          protected void doneWithError(
              @Nullable Throwable primaryCause, ImmutableList<Throwable> secondaryCauses) {
            doneWithErrorCalled.set(true);
            assertThat(primaryCause).isNull();
          }
        };

    future.cancel(true);
    assertThat(future.isCancelled()).isTrue();
    assertThat(doneWithErrorCalled.get()).isFalse();

    future.finishRegistration();
    assertThat(doneWithErrorCalled.get()).isTrue();
  }

  @Test
  public void cancelDuringQuiescence_invokesDoneWithError() throws Exception {
    AtomicBoolean doneWithErrorCalled = new AtomicBoolean(false);
    var future =
        new QuiescingFuture<String>(safeDirectExecutor()) {
          @Override
          protected String getValue() {
            cancel(true);
            return "result";
          }

          @Override
          protected void doneWithError(
              @Nullable Throwable primaryCause, ImmutableList<Throwable> secondaryCauses) {
            doneWithErrorCalled.set(true);
          }
        };

    future.finishRegistration();
    assertThat(future.isCancelled()).isTrue();
    assertThat(doneWithErrorCalled.get()).isTrue();
  }

  private static class TestQuiescingFuture extends QuiescingFuture<String> {
    private final Runnable doneWithErrorCallback;
    private final Runnable getValueCallback;

    private TestQuiescingFuture(AtomicBoolean doneWithErrorCalled, AtomicBoolean getValueCalled) {
      this(() -> doneWithErrorCalled.set(true), () -> getValueCalled.set(true));
    }

    private TestQuiescingFuture(Runnable doneWithErrorCallback, Runnable getValueCallback) {
      super(safeDirectExecutor());
      this.doneWithErrorCallback = doneWithErrorCallback;
      this.getValueCallback = getValueCallback;
    }

    @Override
    protected String getValue() {
      getValueCallback.run();
      return "result";
    }

    @Override
    protected void doneWithError(
        @Nullable Throwable primaryCause, ImmutableList<Throwable> secondaryCauses) {
      doneWithErrorCallback.run();
    }
  }

  @Test
  public void getValueThrowsUncheckedException_completesFutureExceptionally() throws Exception {
    var expectedException = new RuntimeException("getValue exception");
    var future =
        new QuiescingFuture<String>(safeDirectExecutor()) {
          @Override
          protected String getValue() {
            throw expectedException;
          }
        };

    future.decrement();

    assertThat(future.isDone()).isTrue();
    var thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(expectedException);
  }

  @Test
  public void doneWithErrorThrowsUncheckedException_completesFutureExceptionally()
      throws Exception {
    AtomicBoolean doneWithErrorCalled = new AtomicBoolean(false);
    var doneWithErrorException = new RuntimeException("doneWithError exception");
    var future =
        new QuiescingFuture<String>(safeDirectExecutor()) {
          @Override
          protected String getValue() {
            return "result";
          }

          @Override
          protected void doneWithError(
              @Nullable Throwable primaryCause, ImmutableList<Throwable> secondaryCauses) {
            doneWithErrorCalled.set(true);
            throw doneWithErrorException;
          }
        };

    var notifiedException = new RuntimeException("notified exception");
    var thrownFromNotify =
        assertThrows(RuntimeException.class, () -> future.notifyException(notifiedException));
    assertThat(thrownFromNotify).isSameInstanceAs(doneWithErrorException);

    assertThat(future.isDone()).isTrue();
    assertThat(doneWithErrorCalled.get()).isTrue();
    var thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(notifiedException);
  }

  @SuppressWarnings("AssertThrowsMinimizer") // safeDirectExecutor() does not throw exceptions
  @Test
  public void executeSubtask_nullParameters_throwNullPointerException() {
    var future = new ConstantQuiescingFuture();
    assertThrows(
        NullPointerException.class, () -> future.executeSubtask(null, safeDirectExecutor()));
    assertThrows(NullPointerException.class, () -> future.executeSubtask(() -> {}, null));
  }

  @Test
  public void notifyException_cancellationException_transitionsToCancelled() throws Exception {
    AtomicBoolean doneWithErrorCalled = new AtomicBoolean(false);
    var future =
        new QuiescingFuture<String>(safeDirectExecutor()) {
          @Override
          protected String getValue() {
            return "result";
          }

          @Override
          protected void doneWithError(
              @Nullable Throwable primaryCause, ImmutableList<Throwable> secondaryCauses) {
            doneWithErrorCalled.set(true);
            assertThat(primaryCause).isNull();
          }
        };

    future.notifyException(new CancellationException("task cancelled"));

    assertThat(future.isDone()).isTrue();
    assertThat(future.isCancelled()).isTrue();
    assertThat(doneWithErrorCalled.get()).isTrue();
    assertThrows(CancellationException.class, future::get);
  }

  @Test
  public void notifyException_secondaryCancellationException_ignored() throws Exception {
    AtomicReference<Throwable> capturedPrimary = new AtomicReference<>();
    AtomicReference<ImmutableList<Throwable>> capturedSecondaries = new AtomicReference<>();

    var future =
        new QuiescingFuture<String>(safeDirectExecutor()) {
          @Override
          protected String getValue() {
            return "result";
          }

          @Override
          protected void doneWithError(
              @Nullable Throwable primaryCause, ImmutableList<Throwable> secondaryCauses) {
            capturedPrimary.set(primaryCause);
            capturedSecondaries.set(secondaryCauses);
          }
        };

    var primaryError = new RuntimeException("primary error");
    var cancellationError = new CancellationException("secondary cancellation");

    future.increment();
    future.notifyException(primaryError);
    future.notifyException(cancellationError);

    assertThat(future.isDone()).isTrue();
    assertThat(future.isCancelled()).isFalse();
    assertThat(capturedPrimary.get()).isSameInstanceAs(primaryError);
    assertThat(capturedSecondaries.get()).isEmpty();
  }
}
