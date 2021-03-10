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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.Observable;
import io.reactivex.rxjava3.core.Single;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
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
  @GuardedBy("this")
  private final Map<KeyT, ValueT> finished;

  @GuardedBy("this")
  private final Map<KeyT, Observable<ValueT>> inProgress;

  public static <KeyT, ValueT> AsyncTaskCache<KeyT, ValueT> create() {
    return new AsyncTaskCache<>();
  }

  private AsyncTaskCache() {
    this.finished = new HashMap<>();
    this.inProgress = new HashMap<>();
  }

  /** Returns a set of keys for tasks which is finished. */
  public ImmutableSet<KeyT> getFinishedTasks() {
    synchronized (this) {
      return ImmutableSet.copyOf(finished.keySet());
    }
  }

  /** Returns a set of keys for tasks which is still executing. */
  public ImmutableSet<KeyT> getInProgressTasks() {
    synchronized (this) {
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

  /**
   * Executes a task.
   *
   * @param key identifies the task.
   * @param force re-execute a finished task if set to {@code true}.
   * @return a {@link Single} which turns to completed once the task is finished or propagates the
   *     error if any.
   */
  public Single<ValueT> execute(KeyT key, Single<ValueT> task, boolean force) {
    return Single.defer(
        () -> {
          synchronized (this) {
            if (!force && finished.containsKey(key)) {
              return Single.just(finished.get(key));
            }

            finished.remove(key);

            Observable<ValueT> execution =
                inProgress.computeIfAbsent(
                    key,
                    missingKey -> {
                      AtomicInteger subscribeTimes = new AtomicInteger(0);
                      return Single.defer(
                              () -> {
                                int times = subscribeTimes.incrementAndGet();
                                Preconditions.checkState(
                                    times == 1, "Subscribed more than once to the task");
                                return task;
                              })
                          .doOnSuccess(
                              value -> {
                                synchronized (this) {
                                  finished.put(key, value);
                                  inProgress.remove(key);
                                }
                              })
                          .doOnError(
                              error -> {
                                synchronized (this) {
                                  inProgress.remove(key);
                                }
                              })
                          .doOnDispose(
                              () -> {
                                synchronized (this) {
                                  inProgress.remove(key);
                                }
                              })
                          .toObservable()
                          .publish()
                          .refCount();
                    });

            return Single.fromObservable(execution);
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
  }
}
