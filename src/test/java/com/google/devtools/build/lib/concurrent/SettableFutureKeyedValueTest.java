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
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static java.util.concurrent.ForkJoinPool.commonPool;
import static org.junit.Assert.assertThrows;

import com.google.common.util.concurrent.SettableFuture;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiConsumer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class SettableFutureKeyedValueTest {
  private record Value(String text) {}

  private static final class FutureValue
      extends SettableFutureKeyedValue<FutureValue, String, Value> {
    private FutureValue(String key, BiConsumer<String, Value> consumer) {
      super(key, consumer);
    }
  }

  @Test
  public void takingOwnership_occursExactlyOnce() throws Exception {
    var future = new FutureValue("key", (unusedA, unusedB) -> {});
    var tryOwnSuccessCount = new AtomicInteger(0);
    final int taskCount = 100;
    var allDone = new CountDownLatch(taskCount);
    for (int i = 0; i < taskCount; i++) {
      commonPool()
          .execute(
              () -> {
                if (future.tryTakeOwnership()) {
                  tryOwnSuccessCount.getAndIncrement();
                }
                allDone.countDown();
              });
    }
    allDone.await();
    assertThat(tryOwnSuccessCount.get()).isEqualTo(1);
  }

  @Test
  public void futureFails_ifUnset() {
    var future = new FutureValue("key", (unusedA, unusedB) -> {});
    future.verifyComplete();

    var thrown = assertThrows(ExecutionException.class, future::get);

    assertThat(thrown)
        .hasMessageThat()
        .contains("future was unexpectedly unset for key, look for unchecked exceptions");
  }

  @Test
  public void completeWithValue_propagates() throws Exception {
    var setValue = new AtomicReference<Value>();
    var future =
        new FutureValue(
            "key",
            (key, value) -> {
              assertThat(key).isEqualTo("key");
              assertThat(setValue.compareAndSet(null, value)).isTrue();
            });
    var value = new Value("value");

    assertThat(future.completeWith(value)).isEqualTo(value);
    assertThat(setValue.get()).isEqualTo(value);

    future.verifyComplete();
    assertThat(future.get()).isEqualTo(value);
  }

  @Test
  public void completeWithFuture_propagates() throws Exception {
    var setValue = new AtomicReference<Value>();
    var future =
        new FutureValue(
            "key",
            (key, value) -> {
              assertThat(key).isEqualTo("key");
              assertThat(setValue.compareAndSet(null, value)).isTrue();
            });
    var value = new Value("value");

    assertThat(future.completeWith(immediateFuture(value)).get()).isEqualTo(value);
    assertThat(setValue.get()).isEqualTo(value);

    future.verifyComplete();
    assertThat(future.get()).isEqualTo(value);
  }

  @Test
  public void failWith_propagates() {
    var setValue = new AtomicReference<Value>();
    var future =
        new FutureValue(
            "key",
            (key, value) -> {
              assertThat(key).isEqualTo("key");
              assertThat(setValue.compareAndSet(null, value)).isTrue();
            });

    FutureValue result = future.failWith(new IllegalStateException("injected failure"));
    var thrown = assertThrows(ExecutionException.class, result::get);

    assertThat(thrown).hasCauseThat().isInstanceOf(IllegalStateException.class);
    assertThat(thrown).hasCauseThat().hasMessageThat().contains("injected failure");

    assertThat(setValue.get()).isNull();

    future.verifyComplete();
    var thrown2 = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown2).hasCauseThat().isSameInstanceAs(thrown.getCause());
  }

  @Test
  public void cancel_propagatesAndAllowsVerifyComplete() {
    var setValue = new AtomicReference<Value>();
    var future =
        new FutureValue(
            "key",
            (key, value) -> {
              assertThat(key).isEqualTo("key");
              assertThat(setValue.compareAndSet(null, value)).isTrue();
            });

    assertThat(future.cancel(false)).isTrue();
    assertThat(future.isCancelled()).isTrue();
    assertThrows(CancellationException.class, future::get);
    assertThat(setValue.get()).isNull();

    future.verifyComplete();
    assertThat(future.isCancelled()).isTrue();
    assertThrows(CancellationException.class, future::get);
  }

  @Test
  public void cancel_completeWithFuture_propagatesToDelegateAndAllowsVerifyComplete() {
    var setValue = new AtomicReference<Value>();
    var future =
        new FutureValue(
            "key",
            (key, value) -> {
              assertThat(key).isEqualTo("key");
              assertThat(setValue.compareAndSet(null, value)).isTrue();
            });
    SettableFuture<Value> delegate = SettableFuture.create();
    var unused = future.completeWith(delegate);

    assertThat(future.cancel(false)).isTrue();
    assertThat(future.isCancelled()).isTrue();
    assertThat(delegate.isCancelled()).isTrue();
    assertThrows(CancellationException.class, future::get);

    future.verifyComplete();
  }

  @Test
  public void cancel_alreadyCompletedWithValue_returnsFalse() {
    var future = new FutureValue("key", (k, v) -> {});
    assertThat(future.completeWith(new Value("val"))).isEqualTo(new Value("val"));

    assertThat(future.cancel(false)).isFalse();
    assertThat(future.isCancelled()).isFalse();
    future.verifyComplete();
  }

  @Test
  public void cancel_alreadyCompletedWithException_returnsFalse() {
    var future = new FutureValue("key", (k, v) -> {});
    assertThat(future.failWith(new IllegalStateException("failed"))).isSameInstanceAs(future);

    assertThat(future.cancel(false)).isFalse();
    assertThat(future.isCancelled()).isFalse();
    future.verifyComplete();
  }

  @Test
  public void cancel_twice_secondCallReturnsFalse() {
    var future = new FutureValue("key", (k, v) -> {});
    assertThat(future.cancel(false)).isTrue();
    assertThat(future.cancel(false)).isFalse();
    assertThat(future.isCancelled()).isTrue();
    future.verifyComplete();
  }

  @Test
  public void cancel_mayInterruptIfRunning_propagatesAndAllowsVerifyComplete() {
    var future = new FutureValue("key", (k, v) -> {});
    assertThat(future.cancel(true)).isTrue();
    assertThat(future.isCancelled()).isTrue();
    assertThrows(CancellationException.class, future::get);
    future.verifyComplete();
  }

  @Test
  public void verifyComplete_concurrentCancellationBetweenIsDoneAndSetException_succeeds() {
    class InterceptableFutureValue
        extends SettableFutureKeyedValue<InterceptableFutureValue, String, Value> {
      private InterceptableFutureValue() {
        super("key", (k, v) -> {});
      }

      @Override
      public boolean isDone() {
        boolean done = super.isDone();
        if (!done) {
          // Simulate concurrent cancellation landing right after the isDone() fast-path check.
          cancel(/* mayInterruptIfRunning= */ false);
        }
        return false;
      }
    }

    var future = new InterceptableFutureValue();
    future.verifyComplete();
    assertThat(future.isCancelled()).isTrue();
  }
}
