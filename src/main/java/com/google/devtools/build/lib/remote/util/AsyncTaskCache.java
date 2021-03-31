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

import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableSet;
import io.reactivex.rxjava3.annotations.NonNull;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.core.SingleObserver;
import io.reactivex.rxjava3.disposables.Disposable;
import io.reactivex.rxjava3.subjects.AsyncSubject;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.concurrent.GuardedBy;
import javax.annotation.concurrent.ThreadSafe;

/**
 * A cache which de-duplicates the executions and stores the results of asynchronous tasks. Each
 * task is identified by a key of type {@link KeyT} and has the result of type {@link ValueT}.
 *
 * <p>Use {@link #executeIfNot} or {@link #execute} and subscribe the returned {@link Single} to
 * start executing a task. The {@link Single} turns to completed once the task is {@code finished}.
 * Errors are propagated if any.
 *
 * <p>Calling {@code execute[IfNot]} multiple times with the same task key can get an {@link Single}
 * which connects to the same underlying execution if the task is still executing, or get a
 * completed {@link Single} if the task is already finished. Set {@code force} to {@code true } to
 * re-execute a finished task.
 *
 * <p>Dispose the {@link Single} to cancel to task execution.
 */
@ThreadSafe
public final class AsyncTaskCache<KeyT, ValueT> {
  private final Object lock = new Object();

  @GuardedBy("lock")
  private final Map<KeyT, ValueT> finished;

  @GuardedBy("lock")
  private final Map<KeyT, Execution<ValueT>> inProgress;

  public static <KeyT, ValueT> AsyncTaskCache<KeyT, ValueT> create() {
    return new AsyncTaskCache<>();
  }

  private AsyncTaskCache() {
    this.finished = new HashMap<>();
    this.inProgress = new HashMap<>();
  }

  /** Returns a set of keys for tasks which is finished. */
  public ImmutableSet<KeyT> getFinishedTasks() {
    synchronized (lock) {
      return ImmutableSet.copyOf(finished.keySet());
    }
  }

  /** Returns a set of keys for tasks which is still executing. */
  public ImmutableSet<KeyT> getInProgressTasks() {
    synchronized (lock) {
      return ImmutableSet.copyOf(inProgress.keySet());
    }
  }

  /**
   * Executes a task if it hasn't been executed.
   *
   * @param key identifies the task.
   * @return a {@link Single} which turns to completed once the task is finished or propagates the
   *     error if any.
   */
  public Single<ValueT> executeIfNot(KeyT key, Single<ValueT> task) {
    return execute(key, task, false);
  }

  private static class Execution<ValueT> {
    private final AtomicBoolean isTaskDisposed = new AtomicBoolean(false);
    private final Single<ValueT> task;
    private final AsyncSubject<ValueT> asyncSubject = AsyncSubject.create();
    private final AtomicInteger referenceCount = new AtomicInteger(0);
    private final AtomicReference<Disposable> taskDisposable = new AtomicReference<>(null);

    Execution(Single<ValueT> task) {
      this.task = task;
    }

    Single<ValueT> executeIfNot() {
      checkState(!isTaskDisposed(), "disposed");

      int subscribed = referenceCount.getAndIncrement();
      if (taskDisposable.get() == null && subscribed == 0) {
        task.subscribe(
            new SingleObserver<ValueT>() {
              @Override
              public void onSubscribe(@NonNull Disposable d) {
                taskDisposable.compareAndSet(null, d);
              }

              @Override
              public void onSuccess(@NonNull ValueT value) {
                asyncSubject.onNext(value);
                asyncSubject.onComplete();
              }

              @Override
              public void onError(@NonNull Throwable e) {
                asyncSubject.onError(e);
              }
            });
      }

      return Single.fromObservable(asyncSubject);
    }

    boolean isTaskTerminated() {
      return asyncSubject.hasComplete() || asyncSubject.hasThrowable();
    }

    boolean isTaskDisposed() {
      return isTaskDisposed.get();
    }

    void tryDisposeTask() {
      checkState(!isTaskDisposed(), "disposed");
      checkState(!isTaskTerminated(), "terminated");

      if (referenceCount.decrementAndGet() == 0) {
        isTaskDisposed.set(true);
        asyncSubject.onError(new CancellationException("disposed"));

        Disposable d = taskDisposable.get();
        if (d != null) {
          d.dispose();
        }
      }
    }
  }

  /** Returns count of subscribers for a task. */
  public int getSubscriberCount(KeyT key) {
    synchronized (lock) {
      Execution<ValueT> execution = inProgress.get(key);
      if (execution != null) {
        return execution.referenceCount.get();
      }
    }

    return 0;
  }

  /**
   * Executes a task.
   *
   * @param key identifies the task.
   * @param force re-execute a finished task if set to {@code true}.
   * @return a {@link Single} which turns to completed once the task is finished or propagates the
   *     error if any.
   */
  public Single<ValueT> execute(KeyT key, Single<ValueT> task, boolean force) {
    return Single.create(
        emitter -> {
          synchronized (lock) {
            if (!force && finished.containsKey(key)) {
              emitter.onSuccess(finished.get(key));
              return;
            }

            finished.remove(key);

            Execution<ValueT> execution =
                inProgress.computeIfAbsent(
                    key,
                    ignoredKey -> {
                      AtomicInteger subscribeTimes = new AtomicInteger(0);
                      return new Execution<>(
                          Single.defer(
                              () -> {
                                int times = subscribeTimes.incrementAndGet();
                                checkState(times == 1, "Subscribed more than once to the task");
                                return task;
                              }));
                    });

            execution
                .executeIfNot()
                .subscribe(
                    new SingleObserver<ValueT>() {
                      @Override
                      public void onSubscribe(@NonNull Disposable d) {
                        emitter.setCancellable(
                            () -> {
                              d.dispose();

                              if (!execution.isTaskTerminated()) {
                                synchronized (lock) {
                                  execution.tryDisposeTask();
                                  if (execution.isTaskDisposed()) {
                                    inProgress.remove(key);
                                  }
                                }
                              }
                            });
                      }

                      @Override
                      public void onSuccess(@NonNull ValueT value) {
                        synchronized (lock) {
                          finished.put(key, value);
                          inProgress.remove(key);
                        }

                        emitter.onSuccess(value);
                      }

                      @Override
                      public void onError(@NonNull Throwable e) {
                        synchronized (lock) {
                          inProgress.remove(key);
                        }

                        if (!emitter.isDisposed()) {
                          emitter.onError(e);
                        }
                      }
                    });
          }
        });
  }

  /** An {@link AsyncTaskCache} without result. */
  public static final class NoResult<KeyT> {
    private final AsyncTaskCache<KeyT, Optional<Void>> cache;

    public static <KeyT> AsyncTaskCache.NoResult<KeyT> create() {
      return new AsyncTaskCache.NoResult<>(AsyncTaskCache.create());
    }

    public NoResult(AsyncTaskCache<KeyT, Optional<Void>> cache) {
      this.cache = cache;
    }

    /** Same as {@link AsyncTaskCache#executeIfNot} but operates on {@link Completable}. */
    public Completable executeIfNot(KeyT key, Completable task) {
      return Completable.fromSingle(
          cache.executeIfNot(key, task.toSingleDefault(Optional.empty())));
    }

    /** Same as {@link AsyncTaskCache#executeIfNot} but operates on {@link Completable}. */
    public Completable execute(KeyT key, Completable task, boolean force) {
      return Completable.fromSingle(
          cache.execute(key, task.toSingleDefault(Optional.empty()), force));
    }

    /** Returns a set of keys for tasks which is finished. */
    public ImmutableSet<KeyT> getFinishedTasks() {
      return cache.getFinishedTasks();
    }

    /** Returns a set of keys for tasks which is still executing. */
    public ImmutableSet<KeyT> getInProgressTasks() {
      return cache.getInProgressTasks();
    }

    /** Returns count of subscribers for a task. */
    public int getSubscriberCount(KeyT key) {
      return cache.getSubscriberCount(key);
    }
  }
}
