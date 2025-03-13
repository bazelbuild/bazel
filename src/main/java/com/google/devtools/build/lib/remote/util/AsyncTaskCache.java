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
import static com.google.common.base.Throwables.throwIfInstanceOf;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import io.reactivex.rxjava3.annotations.NonNull;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.CompletableEmitter;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.core.SingleObserver;
import io.reactivex.rxjava3.disposables.Disposable;
import io.reactivex.rxjava3.functions.Action;
import io.reactivex.rxjava3.subjects.AsyncSubject;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
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
 *
 * <p>Use {@link #shutdown} to shuts the cache down. Any in progress tasks will continue running
 * while new tasks will be injected with {@link CancellationException}. Use {@link
 * #awaitTermination()} after {@link #shutdown} to wait for the in progress tasks finished.
 *
 * <p>Use {@link #shutdownNow} to cancel all in progress and new tasks with exception {@link
 * CancellationException}.
 */
@ThreadSafe
public final class AsyncTaskCache<KeyT, ValueT> {
  private final Object lock = new Object();

  private static final int STATE_ACTIVE = 0;
  private static final int STATE_SHUTDOWN = 1;
  private static final int STATE_TERMINATED = 2;

  @GuardedBy("lock")
  private int state = STATE_ACTIVE;

  @GuardedBy("lock")
  private final List<CompletableEmitter> terminationSubscriber = new ArrayList<>();

  @GuardedBy("lock")
  private final Map<KeyT, ValueT> finished = new HashMap<>();

  @GuardedBy("lock")
  private final Map<KeyT, Execution> inProgress = new HashMap<>();

  public static <KeyT, ValueT> AsyncTaskCache<KeyT, ValueT> create() {
    return new AsyncTaskCache<>();
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

  /** Returns count of subscribers for a task. */
  public int getSubscriberCount(KeyT key) {
    synchronized (lock) {
      Execution task = inProgress.get(key);
      if (task != null) {
        return task.getSubscriberCount();
      }
    }

    return 0;
  }

  class Execution extends Single<ValueT> implements SingleObserver<ValueT> {
    private final KeyT key;
    private final Single<ValueT> upstream;

    @GuardedBy("lock")
    private boolean terminated = false;

    @GuardedBy("lock")
    private Disposable upstreamDisposable;

    @GuardedBy("lock")
    private final List<SingleObserver<? super ValueT>> observers = new ArrayList<>();

    private final AsyncSubject<ValueT> completion = AsyncSubject.create();

    Execution(KeyT key, Single<ValueT> upstream) {
      this.key = key;
      this.upstream = upstream;
    }

    int getSubscriberCount() {
      synchronized (lock) {
        return observers.size();
      }
    }

    @Override
    protected void subscribeActual(@NonNull SingleObserver<? super ValueT> observer) {
      synchronized (lock) {
        checkState(!terminated, "terminated");

        boolean shouldSubscribe = observers.isEmpty();

        observers.add(observer);

        observer.onSubscribe(new ExecutionDisposable(this, observer));

        if (shouldSubscribe) {
          upstream.subscribe(this);
        }
      }
    }

    @Override
    public void onSubscribe(@NonNull Disposable d) {
      synchronized (lock) {
        upstreamDisposable = d;

        if (terminated) {
          d.dispose();
        }
      }
    }

    @Override
    public void onSuccess(@NonNull ValueT value) {
      synchronized (lock) {
        if (!terminated) {
          inProgress.remove(key);
          finished.put(key, value);
          terminated = true;

          for (SingleObserver<? super ValueT> observer : ImmutableList.copyOf(observers)) {
            observer.onSuccess(value);
          }

          completion.onNext(value);
          completion.onComplete();

          maybeNotifyTermination();
        }
      }
    }

    @Override
    public void onError(@NonNull Throwable error) {
      synchronized (lock) {
        if (!terminated) {
          inProgress.remove(key);
          terminated = true;

          for (SingleObserver<? super ValueT> observer : ImmutableList.copyOf(observers)) {
            observer.onError(error);
          }

          completion.onError(error);

          maybeNotifyTermination();
        }
      }
    }

    void remove(SingleObserver<? super ValueT> observer) {
      synchronized (lock) {
        observers.remove(observer);

        if (observers.isEmpty() && !terminated) {
          inProgress.remove(key);
          terminated = true;

          if (upstreamDisposable != null) {
            upstreamDisposable.dispose();
          }
        }
      }
    }

    void cancel() {
      synchronized (lock) {
        if (!terminated) {
          if (upstreamDisposable != null) {
            upstreamDisposable.dispose();
          }

          onError(new CancellationException("cancelled"));
        }
      }
    }
  }

  class ExecutionDisposable implements Disposable {
    final Execution execution;
    final SingleObserver<? super ValueT> observer;
    AtomicBoolean isDisposed = new AtomicBoolean(false);

    ExecutionDisposable(Execution execution, SingleObserver<? super ValueT> observer) {
      this.execution = execution;
      this.observer = observer;
    }

    @Override
    public void dispose() {
      if (isDisposed.compareAndSet(false, true)) {
        execution.remove(observer);
      }
    }

    @Override
    public boolean isDisposed() {
      return isDisposed.get();
    }
  }

  /**
   * Executes a task.
   *
   * @see #execute(Object, Single, Action, Action, boolean).
   */
  public Single<ValueT> execute(KeyT key, Single<ValueT> task, boolean force) {
    return execute(key, task, () -> {}, () -> {}, force);
  }

  /**
   * Executes a task. If the task has already finished, this execution of the task is ignored unless
   * `force` is true. If the task is in progress this execution of the task is always ignored.
   *
   * <p>If the cache is already shutdown, a {@link CancellationException} will be emitted.
   *
   * @param key identifies the task.
   * @param onAlreadyRunning callback called when provided task is already running.
   * @param onAlreadyFinished callback called when provided task is already finished.
   * @param force re-execute a finished task if set to {@code true}.
   * @return a {@link Single} which turns to completed once the task is finished or propagates the
   *     error if any.
   */
  public Single<ValueT> execute(
      KeyT key,
      Single<ValueT> task,
      Action onAlreadyRunning,
      Action onAlreadyFinished,
      boolean force) {
    return Single.create(
        emitter -> {
          synchronized (lock) {
            if (state != STATE_ACTIVE) {
              emitter.onError(new CancellationException("already shutdown"));
              return;
            }

            if (!force && finished.containsKey(key)) {
              onAlreadyFinished.run();
              emitter.onSuccess(finished.get(key));
              return;
            }

            finished.remove(key);

            Execution execution = inProgress.get(key);
            if (execution != null) {
              onAlreadyRunning.run();
            } else {
              execution = new Execution(key, task);
              inProgress.put(key, execution);
            }

            // We must subscribe the execution within the scope of lock to avoid race condition
            // that:
            //    1. Two callers get the same execution instance
            //    2. One decides to dispose the execution, since no more observers, the execution
            // will change to the terminate state
            //    3. Another one try to subscribe, will get "terminated" error.
            execution.subscribe(
                new SingleObserver<ValueT>() {
                  @Override
                  public void onSubscribe(@NonNull Disposable d) {
                    emitter.setDisposable(d);
                  }

                  @Override
                  public void onSuccess(@NonNull ValueT valueT) {
                    emitter.onSuccess(valueT);
                  }

                  @Override
                  public void onError(@NonNull Throwable e) {
                    if (!emitter.isDisposed()) {
                      emitter.onError(e);
                    }
                  }
                });
          }
        });
  }

  /**
   * Initiates an orderly shutdown in which preexisting tasks continue but new tasks are immediately
   * cancelled with {@link CancellationException}.
   */
  public void shutdown() {
    synchronized (lock) {
      if (state == STATE_ACTIVE) {
        state = STATE_SHUTDOWN;
        maybeNotifyTermination();
      }
    }
  }

  /**
   * Waits for the in-progress tasks to finish. Any tasks that are submitted after the call are not
   * waited.
   */
  public void awaitInProgressTasks() throws InterruptedException {
    Completable completable =
        Completable.defer(
            () -> {
              ImmutableList<Execution> executions;
              synchronized (lock) {
                executions = ImmutableList.copyOf(inProgress.values());
              }

              if (executions.isEmpty()) {
                return Completable.complete();
              }

              return Completable.fromPublisher(
                  Flowable.fromIterable(executions)
                      .flatMapSingle(e -> Single.fromObservable(e.completion)));
            });

    try {
      completable.blockingAwait();
    } catch (RuntimeException e) {
      Throwable cause = e.getCause();
      if (cause != null) {
        throwIfInstanceOf(cause, InterruptedException.class);
      }
      throw e;
    }
  }

  /** Waits for the channel to become terminated. */
  public void awaitTermination() throws InterruptedException {
    Completable completable =
        Completable.create(
            emitter -> {
              synchronized (lock) {
                if (state == STATE_TERMINATED) {
                  emitter.onComplete();
                } else {
                  terminationSubscriber.add(emitter);

                  emitter.setCancellable(
                      () -> {
                        synchronized (lock) {
                          if (state != STATE_TERMINATED) {
                            terminationSubscriber.remove(emitter);
                          }
                        }
                      });
                }
              }
            });

    try {
      completable.blockingAwait();
    } catch (RuntimeException e) {
      Throwable cause = e.getCause();
      if (cause != null) {
        throwIfInstanceOf(cause, InterruptedException.class);
      }
      throw e;
    }
  }

  /**
   * Initiates a forceful shutdown in which preexisting and new tasks are cancelled with {@link
   * CancellationException}. Although forceful, the shutdown process is still not instantaneous;
   * {@link #isTerminated()} will likely return {@code false} immediately after this method returns.
   */
  public void shutdownNow() {
    shutdown();

    synchronized (lock) {
      if (state == STATE_SHUTDOWN) {
        for (Execution execution : ImmutableList.copyOf(inProgress.values())) {
          execution.cancel();
        }
      }
    }
  }

  /**
   * Returns whether the cache is shutdown. Shutdown cache immediately cancels any new tasks, but
   * may still have some tasks in the progress.
   */
  public boolean isShutdown() {
    synchronized (lock) {
      return state == STATE_SHUTDOWN || state == STATE_TERMINATED;
    }
  }

  /**
   * Returns whether the cache is terminated. Terminated cache have no running tasks and relevant
   * resources released.
   */
  public boolean isTerminated() {
    synchronized (lock) {
      return state == STATE_TERMINATED;
    }
  }

  @GuardedBy("lock")
  private void maybeNotifyTermination() {
    if (state == STATE_SHUTDOWN && inProgress.isEmpty()) {
      state = STATE_TERMINATED;

      for (CompletableEmitter emitter : terminationSubscriber) {
        emitter.onComplete();
      }
      terminationSubscriber.clear();
    }
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

    /** Same as {@link AsyncTaskCache#execute} but operates on {@link Completable}. */
    public Completable execute(KeyT key, Completable task, boolean force) {
      return execute(key, task, () -> {}, () -> {}, force);
    }

    /** Same as {@link AsyncTaskCache#execute} but operates on {@link Completable}. */
    public Completable execute(
        KeyT key,
        Completable task,
        Action onAlreadyRunning,
        Action onAlreadyFinished,
        boolean force) {
      return Completable.fromSingle(
          cache.execute(
              key,
              task.toSingleDefault(Optional.empty()),
              onAlreadyRunning,
              onAlreadyFinished,
              force));
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

    /**
     * Initiates an orderly shutdown in which preexisting tasks continue but new tasks are
     * immediately cancelled with {@link CancellationException}.
     */
    public void shutdown() {
      cache.shutdown();
    }

    /**
     * Waits for the in-progress tasks to finish. Any tasks that are submitted after the call are
     * not waited.
     */
    public void awaitInProgressTasks() throws InterruptedException {
      cache.awaitInProgressTasks();
    }

    /** Waits for the cache to become terminated. */
    public void awaitTermination() throws InterruptedException {
      cache.awaitTermination();
    }

    /**
     * Initiates a forceful shutdown in which preexisting and new tasks are cancelled with {@link
     * CancellationException}. Although forceful, the shutdown process is still not instantaneous;
     * {@link #isTerminated()} will likely return {@code false} immediately after this method
     * returns.
     */
    public void shutdownNow() {
      cache.shutdownNow();
    }

    /**
     * Returns whether the cache is shutdown. Shutdown cache immediately cancels any new tasks, but
     * may still have some tasks in the progress.
     */
    public boolean isShutdown() {
      return cache.isShutdown();
    }

    /**
     * Returns whether the cache is terminated. Terminated cache have no running tasks and relevant
     * resources released.
     */
    public boolean isTerminated() {
      return cache.isTerminated();
    }
  }
}
