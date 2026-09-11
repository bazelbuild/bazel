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

import com.google.common.util.concurrent.SettableFuture;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.core.SingleEmitter;
import io.reactivex.rxjava3.observers.TestObserver;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
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
  public void execute_blockedObserver_doesNotBlockOtherKey() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicReference<SingleEmitter<String>> emitterRef1 = new AtomicReference<>();
    AtomicReference<SingleEmitter<String>> emitterRef2 = new AtomicReference<>();
    Semaphore observer1Started = new Semaphore(0);
    Semaphore releaseObserver1 = new Semaphore(0);
    ExecutorService executorService = Executors.newFixedThreadPool(2);

    TestObserver<String> observer1 =
        cache
            .executeIfNot("key1", Single.create(emitterRef1::set))
            .doOnSuccess(
                unused -> {
                  observer1Started.release();
                  releaseObserver1.acquireUninterruptibly();
                })
            .test();
    TestObserver<String> observer2 =
        cache.executeIfNot("key2", Single.create(emitterRef2::set)).test();

    Future<?> completion1 = executorService.submit(() -> emitterRef1.get().onSuccess("value1"));
    try {
      assertThat(observer1Started.tryAcquire(5, SECONDS)).isTrue();

      Future<?> completion2 = executorService.submit(() -> emitterRef2.get().onSuccess("value2"));
      completion2.get(5, SECONDS);
      observer2.assertValue("value2");
    } finally {
      releaseObserver1.release();
      completion1.get(5, SECONDS);
      observer1.assertValue("value1");
      executorService.shutdownNow();
    }
  }

  @Test
  public void execute_blockedCancellation_doesNotBlockOtherKey() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicReference<SingleEmitter<String>> emitterRef1 = new AtomicReference<>();
    AtomicReference<SingleEmitter<String>> emitterRef2 = new AtomicReference<>();
    Semaphore cancellation1Started = new Semaphore(0);
    Semaphore releaseCancellation1 = new Semaphore(0);
    ExecutorService executorService = Executors.newFixedThreadPool(2);

    TestObserver<String> observer1 =
        cache.executeIfNot("key1", Single.create(emitterRef1::set)).test();
    emitterRef1
        .get()
        .setCancellable(
            () -> {
              cancellation1Started.release();
              releaseCancellation1.acquireUninterruptibly();
            });
    TestObserver<String> observer2 =
        cache.executeIfNot("key2", Single.create(emitterRef2::set)).test();

    Future<?> cancellation1 = executorService.submit(observer1::dispose);
    try {
      assertThat(cancellation1Started.tryAcquire(5, SECONDS)).isTrue();

      Future<?> completion2 = executorService.submit(() -> emitterRef2.get().onSuccess("value2"));
      completion2.get(5, SECONDS);
      observer2.assertValue("value2");
    } finally {
      releaseCancellation1.release();
      cancellation1.get(5, SECONDS);
      executorService.shutdownNow();
    }
  }

  @Test
  public void execute_blockedAlreadyFinishedCallback_doesNotBlockOtherKey() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicReference<SingleEmitter<String>> emitterRef2 = new AtomicReference<>();
    Semaphore callback1Started = new Semaphore(0);
    Semaphore releaseCallback1 = new Semaphore(0);
    ExecutorService executorService = Executors.newFixedThreadPool(2);

    cache.executeIfNot("key1", Single.just("value1")).test().assertValue("value1");
    TestObserver<String> observer2 =
        cache.executeIfNot("key2", Single.create(emitterRef2::set)).test();

    Future<?> cacheHit1 =
        executorService.submit(
            () ->
                cache
                    .execute(
                        "key1",
                        Single.just("unused"),
                        () -> {},
                        () -> {
                          callback1Started.release();
                          releaseCallback1.acquireUninterruptibly();
                        },
                        false)
                    .blockingGet());
    try {
      assertThat(callback1Started.tryAcquire(5, SECONDS)).isTrue();

      Future<?> completion2 = executorService.submit(() -> emitterRef2.get().onSuccess("value2"));
      completion2.get(5, SECONDS);
      observer2.assertValue("value2");
    } finally {
      releaseCallback1.release();
      cacheHit1.get(5, SECONDS);
      executorService.shutdownNow();
    }
  }

  private Completable newTask(ExecutorService executorService) {
    return RxFutures.toCompletable(
        () -> {
          SettableFuture<Void> future = SettableFuture.create();
          executorService.execute(
              () -> {
                try {
                  Thread.sleep((long) (Math.random() * 1000));
                  future.set(null);
                } catch (InterruptedException e) {
                  future.setException(new IOException(e));
                }
              });
          return future;
        },
        executorService);
  }

  @Test
  public void execute_executeAndDisposeLoop_noErrors() throws Throwable {
    int taskCount = 1000;
    int maxKey = 20;
    Random random = new Random();
    ExecutorService executorService = Executors.newFixedThreadPool(taskCount);
    AsyncTaskCache.NoResult<String> cache = AsyncTaskCache.NoResult.create();
    AtomicReference<Throwable> error = new AtomicReference<>(null);
    Semaphore semaphore = new Semaphore(0);

    for (int i = 0; i < taskCount; ++i) {
      executorService.execute(
          () -> {
            try {
              Completable task =
                  cache.execute("key" + random.nextInt(maxKey), newTask(executorService), true);
              TestObserver<Void> observer = task.test();
              observer.assertNoErrors();
              if (random.nextBoolean()) {
                observer.dispose();
              } else {
                observer.await();
                observer.assertNoErrors();
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

    if (error.get() != null) {
      throw error.get();
    }
  }

  @Test
  public void execute_executeWithFutureAndCancelLoop_noErrors() throws Throwable {
    int taskCount = 1000;
    int maxKey = 20;
    Random random = new Random();
    ExecutorService executorService = Executors.newFixedThreadPool(taskCount);
    AsyncTaskCache.NoResult<String> cache = AsyncTaskCache.NoResult.create();
    AtomicReference<Throwable> error = new AtomicReference<>(null);
    Semaphore semaphore = new Semaphore(0);

    for (int i = 0; i < taskCount; ++i) {
      executorService.execute(
          () -> {
            try {
              Completable download =
                  cache.execute("key" + random.nextInt(maxKey), newTask(executorService), true);
              Future<Void> future = RxFutures.toListenableFuture(download);
              if (!future.isDone() && random.nextBoolean()) {
                future.cancel(true);
              } else {
                future.get();
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

    if (error.get() != null) {
      throw error.get();
    }
  }

  @Test
  public void execute_pendingShutdown_getCancellationError() {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    cache
        .executeIfNot(
            "key1",
            Single.create(
                emitter -> {
                  // never complete
                }))
        .test()
        .assertNotComplete();
    cache.shutdown();
    assertThat(cache.isShutdown()).isTrue();
    assertThat(cache.isTerminated()).isFalse();

    TestObserver<String> ob = cache.executeIfNot("key2", Single.just("value2")).test();

    ob.assertError(e -> e instanceof CancellationException);
  }

  @Test
  public void execute_afterShutdown_getCancellationError() throws InterruptedException {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    cache.shutdown();
    cache.awaitTermination();

    TestObserver<String> ob = cache.executeIfNot("key", Single.just("value")).test();

    ob.assertError(e -> e instanceof CancellationException);
  }

  @Test
  public void shutdownNow_cancelInProgressTasks() throws InterruptedException {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    TestObserver<String> ob =
        cache
            .executeIfNot(
                "key",
                Single.create(
                    emitter -> {
                      // never complete
                    }))
            .test();
    cache.shutdown();
    assertThat(cache.isShutdown()).isTrue();
    assertThat(cache.isTerminated()).isFalse();
    ob.assertNotComplete();

    cache.shutdownNow();
    cache.awaitTermination();

    assertThat(cache.isShutdown()).isTrue();
    assertThat(cache.isTerminated()).isTrue();
    ob.assertError(e -> e instanceof CancellationException);
  }

  @Test
  public void awaitTermination_pendingShutdown_completeAfterTaskFinished()
      throws InterruptedException {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicReference<SingleEmitter<String>> emitterRef = new AtomicReference<>(null);
    TestObserver<String> ob =
        cache.executeIfNot("key", Single.create(emitterRef::set)).test().assertNotComplete();
    assertThat(emitterRef.get()).isNotNull();
    cache.shutdown();
    assertThat(cache.isShutdown()).isTrue();
    assertThat(cache.isTerminated()).isFalse();

    emitterRef.get().onSuccess("value");
    cache.awaitTermination();

    assertThat(cache.isShutdown()).isTrue();
    assertThat(cache.isTerminated()).isTrue();
    ob.assertValue("value");

    assertThat(cache.getInProgressTasks()).isEmpty();
    assertThat(cache.getFinishedTasks()).isEmpty();
  }

  @Test
  public void awaitTermination_blockedObserver_waitsForNotification() throws Exception {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    AtomicReference<SingleEmitter<String>> emitterRef = new AtomicReference<>();
    Semaphore observerStarted = new Semaphore(0);
    Semaphore releaseObserver = new Semaphore(0);
    ExecutorService executorService = Executors.newSingleThreadExecutor();

    TestObserver<String> observer =
        cache
            .executeIfNot("key", Single.create(emitterRef::set))
            .doOnSuccess(
                unused -> {
                  observerStarted.release();
                  releaseObserver.acquireUninterruptibly();
                })
            .test();
    Future<?> completion = executorService.submit(() -> emitterRef.get().onSuccess("value"));
    try {
      assertThat(observerStarted.tryAcquire(5, SECONDS)).isTrue();

      cache.shutdown();

      assertThat(cache.isShutdown()).isTrue();
      assertThat(cache.isTerminated()).isFalse();
    } finally {
      releaseObserver.release();
      completion.get(5, SECONDS);
      executorService.shutdownNow();
    }

    cache.awaitTermination();
    assertThat(cache.isTerminated()).isTrue();
    observer.assertValue("value");
  }

  @Test
  public void awaitTermination_afterShutdown_complete() throws InterruptedException {
    AsyncTaskCache<String, String> cache = AsyncTaskCache.create();
    cache.shutdownNow();
    cache.awaitTermination();

    cache.awaitTermination();

    assertThat(cache.isShutdown()).isTrue();
    assertThat(cache.isTerminated()).isTrue();
  }
}
