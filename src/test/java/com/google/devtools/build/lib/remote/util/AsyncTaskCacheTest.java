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
import static java.util.concurrent.TimeUnit.SECONDS;

import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.core.SingleEmitter;
import io.reactivex.rxjava3.observers.TestObserver;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AsyncTaskCache}. */
@RunWith(JUnit4.class)
public class AsyncTaskCacheTest {

  @Rule public final RxNoGlobalErrorsRule rxNoGlobalErrorsRule = new RxNoGlobalErrorsRule();

  @Test
  public void execute_noSubscription_noExecution() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicBoolean executed = new AtomicBoolean(false);

    cache.executeIfNot(
        "key1",
        Single.create(
            emitter -> {
              executed.set(true);
              emitter.onSuccess("value1");
            }));

    assertThat(executed.get()).isFalse();
    assertThat(cache.getInProgressTasks()).isEmpty();
    assertThat(cache.getFinishedTasks()).isEmpty();
  }

  @Test
  public void execute_taskFinished_completed() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicReference<SingleEmitter<String>> emitterRef = new AtomicReference<>(null);
    TestObserver<String> observer =
        cache.executeIfNot("key1", Single.create(emitterRef::set)).test();
    SingleEmitter<String> emitter = emitterRef.get();
    assertThat(emitter).isNotNull();

    emitter.onSuccess("value1");

    observer.assertValue("value1");
    assertThat(cache.getInProgressTasks()).isEmpty();
    assertThat(cache.getFinishedTasks()).containsExactly("key1");
  }

  @Test
  public void execute_taskHasError_propagateError() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicReference<SingleEmitter<String>> emitterRef = new AtomicReference<>(null);
    TestObserver<String> observer =
        cache.executeIfNot("key1", Single.create(emitterRef::set)).test();
    SingleEmitter<String> emitter = emitterRef.get();
    assertThat(emitter).isNotNull();
    Throwable error = new IllegalStateException("error");

    emitter.onError(error);

    observer.assertError(error);
    assertThat(cache.getInProgressTasks()).isEmpty();
    assertThat(cache.getFinishedTasks()).isEmpty();
  }

  @Test
  public void execute_taskInProgress_noReExecution() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicReference<SingleEmitter<String>> emitterRef = new AtomicReference<>(null);
    AtomicInteger executionTimes = new AtomicInteger(0);
    Single<String> single =
        cache.executeIfNot(
            "key1",
            Single.create(
                emitter -> {
                  executionTimes.incrementAndGet();
                  emitterRef.set(emitter);
                }));
    TestObserver<String> ob1 = single.test();
    ob1.assertEmpty();
    SingleEmitter<String> emitter = emitterRef.get();
    assertThat(emitter).isNotNull();
    assertThat(cache.getInProgressTasks()).containsExactly("key1");
    assertThat(cache.getFinishedTasks()).isEmpty();

    TestObserver<String> ob2 = single.test();
    ob2.assertEmpty();
    emitter.onSuccess("value1");

    ob1.assertValue("value1");
    ob2.assertValue("value1");
    assertThat(executionTimes.get()).isEqualTo(1);
    assertThat(cache.getInProgressTasks()).isEmpty();
    assertThat(cache.getFinishedTasks()).containsExactly("key1");
  }

  @Test
  public void executeForcibly_taskInProgress_noReExecution() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicReference<SingleEmitter<String>> emitterRef = new AtomicReference<>(null);
    AtomicInteger executionTimes = new AtomicInteger(0);
    Single<String> single =
        cache.execute(
            "key1",
            Single.create(
                emitter -> {
                  executionTimes.incrementAndGet();
                  emitterRef.set(emitter);
                }),
            /* force= */ true);
    TestObserver<String> ob1 = single.test();
    ob1.assertEmpty();
    SingleEmitter<String> emitter = emitterRef.get();
    assertThat(emitter).isNotNull();
    assertThat(cache.getInProgressTasks()).containsExactly("key1");
    assertThat(cache.getFinishedTasks()).isEmpty();

    TestObserver<String> ob2 = single.test();
    ob2.assertEmpty();
    emitter.onSuccess("value1");

    ob1.assertValue("value1");
    ob2.assertValue("value1");
    assertThat(executionTimes.get()).isEqualTo(1);
    assertThat(cache.getInProgressTasks()).isEmpty();
    assertThat(cache.getFinishedTasks()).containsExactly("key1");
  }

  @Test
  public void execute_taskFinished_noReExecution() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicReference<SingleEmitter<String>> emitterRef = new AtomicReference<>(null);
    AtomicInteger executionTimes = new AtomicInteger(0);
    Single<String> single =
        cache.executeIfNot(
            "key1",
            Single.create(
                emitter -> {
                  executionTimes.incrementAndGet();
                  emitterRef.set(emitter);
                }));
    TestObserver<String> ob1 = single.test();
    SingleEmitter<String> emitter = emitterRef.get();
    assertThat(emitter).isNotNull();
    emitter.onSuccess("value1");
    ob1.assertValue("value1");
    assertThat(cache.getFinishedTasks()).containsExactly("key1");

    TestObserver<String> ob2 = single.test();

    ob2.assertValue("value1");
    assertThat(executionTimes.get()).isEqualTo(1);
  }

  @Test
  public void executeForcibly_taskFinished_reExecution() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicReference<SingleEmitter<String>> emitterRef = new AtomicReference<>(null);
    AtomicInteger executionTimes = new AtomicInteger(0);
    Single<String> single =
        cache.execute(
            "key1",
            Single.create(
                emitter -> {
                  executionTimes.incrementAndGet();
                  emitterRef.set(emitter);
                }),
            /* force= */ true);
    TestObserver<String> ob1 = single.test();
    SingleEmitter<String> emitter = emitterRef.get();
    assertThat(emitter).isNotNull();
    emitter.onSuccess("value1");
    ob1.assertValue("value1");
    assertThat(cache.getFinishedTasks()).containsExactly("key1");

    TestObserver<String> ob2 = single.test();

    ob2.assertEmpty();
    assertThat(executionTimes.get()).isEqualTo(2);
    assertThat(cache.getInProgressTasks()).containsExactly("key1");
    assertThat(cache.getFinishedTasks()).isEmpty();
  }

  @Test
  public void execute_dispose_cancelled() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicReference<SingleEmitter<String>> emitterRef = new AtomicReference<>(null);
    TestObserver<String> observer =
        cache.executeIfNot("key1", Single.create(emitterRef::set)).test();
    SingleEmitter<String> emitter = emitterRef.get();
    assertThat(emitter).isNotNull();
    AtomicBoolean disposed = new AtomicBoolean(false);
    emitter.setCancellable(() -> disposed.set(true));

    observer.dispose();

    assertThat(disposed.get()).isTrue();
    assertThat(cache.getInProgressTasks()).isEmpty();
    assertThat(cache.getFinishedTasks()).isEmpty();
  }

  @Test
  public void execute_disposeWhenMultipleSubscriptions_notCancelled() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicReference<SingleEmitter<String>> emitterRef = new AtomicReference<>(null);
    Single<String> single = cache.executeIfNot("key1", Single.create(emitterRef::set));
    TestObserver<String> ob1 = single.test();
    TestObserver<String> ob2 = single.test();
    SingleEmitter<String> emitter = emitterRef.get();
    assertThat(emitter).isNotNull();
    AtomicBoolean disposed = new AtomicBoolean(false);
    emitter.setCancellable(() -> disposed.set(true));

    ob1.dispose();

    ob2.assertEmpty();
    assertThat(disposed.get()).isFalse();
    assertThat(cache.getInProgressTasks()).containsExactly("key1");
    assertThat(cache.getFinishedTasks()).isEmpty();
  }

  @Test
  public void execute_disposeWhenMultipleSubscriptions_cancelled() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicReference<SingleEmitter<String>> emitterRef = new AtomicReference<>(null);
    Single<String> single = cache.executeIfNot("key1", Single.create(emitterRef::set));
    TestObserver<String> ob1 = single.test();
    TestObserver<String> ob2 = single.test();
    SingleEmitter<String> emitter = emitterRef.get();
    assertThat(emitter).isNotNull();
    AtomicBoolean disposed = new AtomicBoolean(false);
    emitter.setCancellable(() -> disposed.set(true));

    ob1.dispose();
    ob2.dispose();

    assertThat(disposed.get()).isTrue();
    assertThat(cache.getInProgressTasks()).isEmpty();
    assertThat(cache.getFinishedTasks()).isEmpty();
  }

  @Test
  public void execute_multipleTasks_completeOne() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicReference<SingleEmitter<String>> emitterRef1 = new AtomicReference<>(null);
    TestObserver<String> observer1 =
        cache.executeIfNot("key1", Single.create(emitterRef1::set)).test();
    SingleEmitter<String> emitter1 = emitterRef1.get();
    assertThat(emitter1).isNotNull();
    AtomicReference<SingleEmitter<String>> emitterRef2 = new AtomicReference<>(null);
    TestObserver<String> observer2 =
        cache.executeIfNot("key2", Single.create(emitterRef2::set)).test();
    SingleEmitter<String> emitter2 = emitterRef1.get();
    assertThat(emitter2).isNotNull();

    emitter1.onSuccess("value1");

    observer1.assertValue("value1");
    observer2.assertEmpty();
    assertThat(cache.getInProgressTasks()).containsExactly("key2");
    assertThat(cache.getFinishedTasks()).containsExactly("key1");
  }

  @Test
  public void execute_executeAndDisposeLoop_noErrors() throws InterruptedException {
    AsyncTaskCache<String, Long> cache = AsyncTaskCache.create();
    Single<Long> task = Single.timer(1, SECONDS);
    AtomicReference<Throwable> error = new AtomicReference<>(null);
    AtomicInteger errorCount = new AtomicInteger(0);
    int executionCount = 100;
    Runnable runnable =
        () -> {
          try {
            for (int i = 0; i < executionCount; ++i) {
              TestObserver<Long> observer = cache.execute("key1", task, true).test();
              observer.assertNoErrors();
              observer.dispose();
            }
          } catch (Throwable t) {
            errorCount.incrementAndGet();
            error.set(t);
          }
        };
    int threadCount = 10;
    Thread[] threads = new Thread[threadCount];
    for (int i = 0; i < threadCount; ++i) {
      Thread thread = new Thread(runnable);
      threads[i] = thread;
    }

    for (Thread thread : threads) {
      thread.start();
    }
    for (Thread thread : threads) {
      thread.join();
    }

    if (error.get() != null) {
      throw new IllegalStateException(
          String.format("%s/%s errors", errorCount.get(), threadCount), error.get());
    }
  }
}
