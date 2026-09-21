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
import static com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutor.safeDirectExecutor;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.concurrent.safeexecutor.RejectionHandlingRunnable;
import com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutor;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class QuiescingFutureTaskTest {

  @Test
  public void runOnce() throws Exception {
    AtomicInteger callCount = new AtomicInteger(0);
    var task =
        new QuiescingFutureTask<String>(safeDirectExecutor()) {
          @Override
          protected void arrangeSubtasks() {
            callCount.incrementAndGet();
          }

          @Override
          protected String getValue() {
            return "result";
          }
        };

    assertThat(task.isDone()).isFalse();
    task.run();
    assertThat(task.isDone()).isTrue();
    assertThat(task.get()).isEqualTo("result");
    assertThat(callCount.get()).isEqualTo(1);

    // Running again should not call arrangeSubtasks but still result in the same value (already
    // done)
    task.run();
    assertThat(callCount.get()).isEqualTo(1);
  }

  @Test
  public void subtasksCompletion() throws Exception {
    AtomicInteger subtaskCallCount = new AtomicInteger(0);
    var task =
        new QuiescingFutureTask<String>(safeDirectExecutor()) {
          @Override
          protected void arrangeSubtasks() {
            increment();
            subtaskCallCount.incrementAndGet();
          }

          @Override
          protected String getValue() {
            return "result";
          }
        };

    task.run();
    assertThat(task.isDone()).isFalse();
    assertThat(subtaskCallCount.get()).isEqualTo(1);

    task.decrement();
    assertThat(task.isDone()).isTrue();
    assertThat(task.get()).isEqualTo("result");
  }

  @Test
  public void exceptionInArrangeSubtasks() throws Exception {
    var error = new RuntimeException("oops");
    var task =
        new QuiescingFutureTask<String>(safeDirectExecutor()) {
          @Override
          protected void arrangeSubtasks() {
            throw error;
          }

          @Override
          protected String getValue() {
            return "result";
          }
        };

    task.run();
    assertThat(task.isDone()).isTrue();
    var thrown = assertThrows(ExecutionException.class, task::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(error);
  }

  @Test
  public void doneWithErrorCalled() throws Exception {
    AtomicBoolean doneWithErrorCalled = new AtomicBoolean(false);
    var task =
        new QuiescingFutureTask<String>(safeDirectExecutor()) {
          @Override
          protected void arrangeSubtasks() {
            recordException(new RuntimeException("error"));
          }

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

    task.run();
    assertThat(task.isDone()).isTrue();
    assertThat(doneWithErrorCalled.get()).isTrue();
  }

  @Test
  public void handleRejection_beforeArrangeSubtasks() throws Exception {
    AtomicBoolean arrangeSubtasksCalled = new AtomicBoolean(false);
    AtomicBoolean doneWithErrorCalled = new AtomicBoolean(false);
    AtomicReference<Throwable> capturedPrimary = new AtomicReference<>();
    AtomicReference<ImmutableList<Throwable>> capturedSecondaries = new AtomicReference<>();

    var task =
        new QuiescingFutureTask<String>(safeDirectExecutor()) {
          @Override
          protected void arrangeSubtasks() {
            arrangeSubtasksCalled.set(true);
          }

          @Override
          protected String getValue() {
            return "result";
          }

          @Override
          protected void doneWithError(
              @Nullable Throwable primaryCause, ImmutableList<Throwable> secondaryCauses) {
            doneWithErrorCalled.set(true);
            capturedPrimary.set(primaryCause);
            capturedSecondaries.set(secondaryCauses);
          }
        };

    var rejectionException = new RejectedExecutionException("rejected before start");
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

    rejectingExecutor.execute(task);

    assertThat(task.isDone()).isTrue();
    assertThat(doneWithErrorCalled.get()).isTrue();
    assertThat(capturedPrimary.get()).isSameInstanceAs(rejectionException);
    assertThat(capturedSecondaries.get()).isEmpty();
    assertThat(arrangeSubtasksCalled.get()).isFalse();

    var thrown = assertThrows(ExecutionException.class, task::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(rejectionException);

    // Subsequent run() should NOT execute arrangeSubtasks
    task.run();
    assertThat(arrangeSubtasksCalled.get()).isFalse();
  }

  @Test
  public void handleRejection_afterArrangeSubtasks() throws Exception {
    AtomicBoolean doneWithErrorCalled = new AtomicBoolean(false);
    AtomicReference<Throwable> capturedPrimary = new AtomicReference<>();
    AtomicReference<ImmutableList<Throwable>> capturedSecondaries = new AtomicReference<>();

    var rejectionException = new RejectedExecutionException("rejection during pass 2");
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

    var task =
        new QuiescingFutureTask<String>(rejectingExecutor) {
          @Override
          protected void arrangeSubtasks() {}

          @Override
          protected String getValue() {
            return "result";
          }

          @Override
          protected void doneWithError(
              @Nullable Throwable primaryCause, ImmutableList<Throwable> secondaryCauses) {
            doneWithErrorCalled.set(true);
            capturedPrimary.set(primaryCause);
            capturedSecondaries.set(secondaryCauses);
          }
        };

    // Executes arrangeSubtasks and decrements to 0, triggering rejection on rejectingExecutor
    task.run();
    assertThat(task.isDone()).isTrue();
    assertThat(doneWithErrorCalled.get()).isTrue();
    assertThat(capturedPrimary.get()).isSameInstanceAs(rejectionException);
  }

  @Test
  public void executeSubtask_withinArrangeSubtasks_success() throws Exception {
    AtomicBoolean subtaskExecuted = new AtomicBoolean(false);
    var task =
        new QuiescingFutureTask<String>(safeDirectExecutor()) {
          @Override
          protected void arrangeSubtasks() {
            executeSubtask(() -> subtaskExecuted.set(true), safeDirectExecutor());
          }

          @Override
          protected String getValue() {
            return "result";
          }
        };

    task.run();
    assertThat(subtaskExecuted.get()).isTrue();
    assertThat(task.isDone()).isTrue();
    assertThat(task.get()).isEqualTo("result");
  }

  @Test
  public void executeSubtask_withinArrangeSubtasks_rejection() throws Exception {
    AtomicBoolean doneWithErrorCalled = new AtomicBoolean(false);

    var rejectionException = new RejectedExecutionException("subtask rejected");
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

    var task =
        new QuiescingFutureTask<String>(safeDirectExecutor()) {
          @Override
          protected void arrangeSubtasks() {
            executeSubtask(() -> {}, rejectingExecutor);
          }

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

    task.run();
    assertThat(task.isDone()).isTrue();
    assertThat(doneWithErrorCalled.get()).isTrue();
    var thrown = assertThrows(ExecutionException.class, task::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(rejectionException);
  }

  @Test
  public void multiFailure_secondaryExceptionsInQuiescingFutureTask() throws Exception {
    AtomicReference<Throwable> capturedPrimary = new AtomicReference<>();
    AtomicReference<ImmutableList<Throwable>> capturedSecondaries = new AtomicReference<>();

    var error1 = new RuntimeException("error1");
    var error2 = new RuntimeException("error2");

    var task =
        new QuiescingFutureTask<String>(safeDirectExecutor()) {
          @Override
          protected void arrangeSubtasks() {
            recordException(error1);
            recordException(error2);
          }

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

    task.run();
    assertThat(task.isDone()).isTrue();
    assertThat(capturedPrimary.get()).isSameInstanceAs(error1);
    assertThat(capturedSecondaries.get()).containsExactly(error2);
  }
}
