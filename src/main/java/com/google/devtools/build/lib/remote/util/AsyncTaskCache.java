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

import static com.google.common.base.Throwables.throwIfInstanceOf;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import io.reactivex.rxjava3.annotations.NonNull;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.CompletableEmitter;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.core.SingleEmitter;
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
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;
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
  private final ArrayList<CompletableEmitter> terminationSubscriber = new ArrayList<>();

  // Concurrent so that {@link #invalidate} can run without acquiring {@code lock}, which prevents
  // lock-ordering deadlocks when invalidation is triggered from within another cache's observer
  // notification (e.g., a doFinally on an upload completion).
  private final ConcurrentHashMap<KeyT, ValueT> finished = new ConcurrentHashMap<>();

  @GuardedBy("lock")
  private Map<KeyT, Execution> inProgress = new HashMap<>();

  // Terminal notifications run without holding lock. Keep shutdown from reporting termination
  // until notifications that have already been detached from an execution finish delivery.
  @GuardedBy("lock")
  private int notificationsInProgress = 0;

  public static <KeyT, ValueT> AsyncTaskCache<KeyT, ValueT> create() {
    return new AsyncTaskCache<>();
  }

  /** Returns a set of keys for tasks which is finished. */
  public ImmutableSet<KeyT> getFinishedTasks() {
    return ImmutableSet.copyOf(finished.keySet());
  }

  /**
   * Removes any cached result for the given {@code key}, so that the next call to {@link #execute}
   * for that key re-runs the task. Does not affect in-progress tasks. Safe to call concurrently
   * with {@link #execute}.
   */
  public void invalidate(KeyT key) {
    finished.remove(key);
  }

  /**
   * Atomically replaces the cached result for {@code key} with {@code value}. The new value is
   * visible to subsequent {@link #execute} callers. Safe to call concurrently with {@link
   * #execute}.
   */
  public void put(KeyT key, ValueT value) {
    finished.put(key, value);
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

  class Execution implements SingleObserver<ValueT> {
    private final KeyT key;
    private final Single<ValueT> upstream;

    @GuardedBy("lock")
    private boolean terminated = false;

    @GuardedBy("lock")
    private Disposable upstreamDisposable;

    @GuardedBy("lock")
    private final List<Subscriber> subscribers = new ArrayList<>();

    private final AsyncSubject<ValueT> completion = AsyncSubject.create();

    Execution(KeyT key, Single<ValueT> upstream) {
      this.key = key;
      this.upstream = upstream;
    }

    int getSubscriberCount() {
      synchronized (lock) {
        return subscribers.size();
      }
    }

    Subscriber addSubscriber(SingleEmitter<ValueT> emitter) {
      synchronized (lock) {
        Subscriber subscriber = new Subscriber(this, emitter);
        subscribers.add(subscriber);
        return subscriber;
      }
    }

    void start() {
      upstream.subscribe(this);
    }

    @Override
    public void onSubscribe(@NonNull Disposable d) {
      boolean dispose;
      synchronized (lock) {
        dispose = terminated;
        if (!dispose) {
          upstreamDisposable = d;
        }
      }
      if (dispose) {
        d.dispose();
      }
    }

    @Override
    public void onSuccess(@NonNull ValueT value) {
      TerminalNotification notification;
      synchronized (lock) {
        notification = terminateLocked(value, null, false);
      }
      if (notification != null) {
        notification.deliver();
      }
    }

    @Override
    public void onError(@NonNull Throwable error) {
      TerminalNotification notification;
      synchronized (lock) {
        notification = terminateLocked(null, error, false);
      }
      if (notification != null) {
        notification.deliver();
      }
    }

    void remove(Subscriber subscriber) {
      TerminalNotification notification = null;
      synchronized (lock) {
        subscribers.remove(subscriber);
        if (subscribers.isEmpty() && !terminated) {
          notification = terminateLocked(null, new CancellationException("cancelled"), true);
        }
      }
      if (notification != null) {
        notification.deliver();
      }
    }

    void cancel() {
      TerminalNotification notification;
      synchronized (lock) {
        notification = terminateLocked(null, new CancellationException("cancelled"), true);
      }
      if (notification != null) {
        notification.deliver();
      }
    }

    @GuardedBy("lock")
    @Nullable
    private TerminalNotification terminateLocked(
        @Nullable ValueT value, @Nullable Throwable error, boolean disposeUpstream) {
      if (terminated) {
        return null;
      }

      inProgress.remove(key, this);
      if (error == null) {
        finished.put(key, value);
      }
      terminated = true;

      ImmutableList<Subscriber> subscribersToNotify = ImmutableList.copyOf(subscribers);
      subscribers.clear();
      notificationsInProgress++;

      return new TerminalNotification(
          value, error, subscribersToNotify, disposeUpstream ? upstreamDisposable : null);
    }

    /**
     * A terminal state transition captured under {@link #lock} and delivered after releasing it.
     */
    class TerminalNotification {
      @Nullable private final ValueT value;
      @Nullable private final Throwable error;
      private final ImmutableList<Subscriber> subscribersToNotify;
      @Nullable private final Disposable disposableToDispose;

      TerminalNotification(
          @Nullable ValueT value,
          @Nullable Throwable error,
          ImmutableList<Subscriber> subscribersToNotify,
          @Nullable Disposable disposableToDispose) {
        this.value = value;
        this.error = error;
        this.subscribersToNotify = subscribersToNotify;
        this.disposableToDispose = disposableToDispose;
      }

      void deliver() {
        try {
          if (disposableToDispose != null) {
            disposableToDispose.dispose();
          }

          if (error == null) {
            for (Subscriber subscriber : subscribersToNotify) {
              if (!subscriber.isDisposed()) {
                subscriber.emitter.onSuccess(value);
              }
            }
            completion.onNext(value);
            completion.onComplete();
          } else {
            for (Subscriber subscriber : subscribersToNotify) {
              if (!subscriber.isDisposed()) {
                subscriber.emitter.tryOnError(error);
              }
            }
            completion.onError(error);
          }
        } finally {
          notificationFinished();
        }
      }
    }
  }

  class Subscriber implements Disposable {
    final Execution execution;
    final SingleEmitter<ValueT> emitter;
    private final AtomicBoolean isDisposed = new AtomicBoolean(false);

    Subscriber(Execution execution, SingleEmitter<ValueT> emitter) {
      this.execution = execution;
      this.emitter = emitter;
    }

    @Override
    public void dispose() {
      if (isDisposed.compareAndSet(false, true)) {
        execution.remove(this);
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
          boolean rejected = false;
          boolean alreadyFinished = false;
          boolean alreadyRunning = false;
          boolean startExecution = false;
          @Nullable ValueT cached = null;
          @Nullable Execution execution = null;
          @Nullable Subscriber subscriber = null;

          synchronized (lock) {
            if (state != STATE_ACTIVE) {
              rejected = true;
            } else {
              if (!force) {
                cached = finished.get(key);
                alreadyFinished = cached != null;
              } else {
                finished.remove(key);
              }

              if (!alreadyFinished) {
                execution = inProgress.get(key);
                alreadyRunning = execution != null;
                if (!alreadyRunning) {
                  execution = new Execution(key, task);
                  inProgress.put(key, execution);
                  startExecution = true;
                }

                subscriber = execution.addSubscriber(emitter);
              }
            }
          }

          if (rejected) {
            emitter.tryOnError(new CancellationException("already shutdown"));
            return;
          }
          if (alreadyFinished) {
            onAlreadyFinished.run();
            emitter.onSuccess(cached);
            return;
          }

          emitter.setDisposable(subscriber);
          if (alreadyRunning) {
            onAlreadyRunning.run();
          }
          if (startExecution) {
            execution.start();
          }
        });
  }

  /**
   * Initiates an orderly shutdown in which preexisting tasks continue but new tasks are immediately
   * cancelled with {@link CancellationException}.
   */
  public void shutdown() {
    ImmutableList<CompletableEmitter> terminationSubscribers;
    synchronized (lock) {
      if (state == STATE_ACTIVE) {
        state = STATE_SHUTDOWN;
      }
      terminationSubscribers = maybeTransitionToTerminated();
    }
    notifyTerminationSubscribers(terminationSubscribers);
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
              boolean complete;
              synchronized (lock) {
                complete = state == STATE_TERMINATED;
                if (!complete) {
                  terminationSubscriber.add(emitter);
                }
              }
              if (complete) {
                emitter.onComplete();
              } else {
                emitter.setCancellable(
                    () -> {
                      synchronized (lock) {
                        if (state != STATE_TERMINATED) {
                          terminationSubscriber.remove(emitter);
                        }
                      }
                    });
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

    ImmutableList<Execution> executions;
    synchronized (lock) {
      executions =
          state == STATE_SHUTDOWN ? ImmutableList.copyOf(inProgress.values()) : ImmutableList.of();
    }
    for (Execution execution : executions) {
      execution.cancel();
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

  private void notificationFinished() {
    ImmutableList<CompletableEmitter> terminationSubscribers;
    synchronized (lock) {
      notificationsInProgress--;
      terminationSubscribers = maybeTransitionToTerminated();
    }
    notifyTerminationSubscribers(terminationSubscribers);
  }

  @GuardedBy("lock")
  private ImmutableList<CompletableEmitter> maybeTransitionToTerminated() {
    if (state == STATE_SHUTDOWN && inProgress.isEmpty() && notificationsInProgress == 0) {
      state = STATE_TERMINATED;
      ImmutableList<CompletableEmitter> subscribers = ImmutableList.copyOf(terminationSubscriber);
      terminationSubscriber.clear();
      terminationSubscriber.trimToSize();
      inProgress = new HashMap<>();
      finished.clear();
      return subscribers;
    }
    return ImmutableList.of();
  }

  private static void notifyTerminationSubscribers(ImmutableList<CompletableEmitter> subscribers) {
    for (CompletableEmitter emitter : subscribers) {
      emitter.onComplete();
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

    /**
     * @see AsyncTaskCache#invalidate
     */
    public void invalidate(KeyT key) {
      cache.invalidate(key);
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
