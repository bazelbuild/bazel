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
import static java.util.concurrent.ForkJoinPool.commonPool;
import static org.junit.Assert.assertThrows;

import java.util.concurrent.ExecutionException;
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
      this.counter = counter;
    }

    @Override
    protected Integer getValue() {
      return counter.get();
    }
  }
}
