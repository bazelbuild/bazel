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
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static java.util.concurrent.ForkJoinPool.commonPool;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertThrows;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
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
      super(directExecutor());
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
      super(directExecutor());
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
    var future =
        new QuiescingFuture<String>(
            command -> {
              executorCalled.set(true);
              command.run();
            }) {
          @Override
          protected String getValue() {
            return "executed";
          }
        };

    future.decrement();
    assertThat(future.get()).isEqualTo("executed");
    assertThat(executorCalled.get()).isTrue();
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

  private static class TestQuiescingFuture extends QuiescingFuture<String> {
    private final Runnable doneWithErrorCallback;
    private final Runnable getValueCallback;

    private TestQuiescingFuture(AtomicBoolean doneWithErrorCalled, AtomicBoolean getValueCalled) {
      this(() -> doneWithErrorCalled.set(true), () -> getValueCalled.set(true));
    }

    private TestQuiescingFuture(Runnable doneWithErrorCallback, Runnable getValueCallback) {
      super(directExecutor());
      this.doneWithErrorCallback = doneWithErrorCallback;
      this.getValueCallback = getValueCallback;
    }

    @Override
    protected String getValue() {
      getValueCallback.run();
      return "result";
    }

    @Override
    protected void doneWithError() {
      doneWithErrorCallback.run();
    }
  }
}
