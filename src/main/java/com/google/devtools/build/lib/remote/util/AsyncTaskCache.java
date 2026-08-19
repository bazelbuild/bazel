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

import static com.google.common.util.concurrent.Futures.immediateCancelledFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.concurrent.TaskDeduplicator;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import javax.annotation.concurrent.ThreadSafe;

/**
 * A cache which de-duplicates the executions and stores the results of asynchronous tasks. Each
 * task is identified by a key of type {@link KeyT} and has the result of type {@link ValueT}.
 *
 * <p>Calling {@link #execute} multiple times with the same task key joins the same underlying
 * execution if the task is still executing, or returns a completed future if the task has already
 * finished successfully. Results of failed tasks are not cached.
 *
 * <p>Cancelling a returned future only cancels the underlying task once every caller that joined it
 * has cancelled its own future; the task is interrupted only if all of them requested interruption.
 *
 * <p>Use {@link #shutdown} to shut the cache down. Any in progress tasks will continue running
 * while new tasks are immediately cancelled. Use {@link #awaitTermination()} after {@link
 * #shutdown} to wait for the in progress tasks to finish. Use {@link #shutdownNow} to cancel all in
 * progress and new tasks.
 *
 * <p>This class holds no lock of its own: all mutual exclusion is provided by {@link
 * TaskDeduplicator} and {@link ConcurrentHashMap}, and no caller-provided code ever runs in a
 * critical section. Two ordering invariants make that safe:
 *
 * <ol>
 *   <li>A task's result is published to {@link #finished} and the task is unregistered from {@link
 *       #inProgress} strictly before the future handed to callers completes, so a caller that
 *       observes a task as no longer running also observes its result.
 *   <li>A task is only removed from the deduplicator after the future returned by {@link
 *       #registerTask} completes, so a caller that finds no in-flight task for a key and misses the
 *       cache lookup in {@link #execute} is guaranteed to see the cached result when it re-checks
 *       from within the deduplicator's per-key critical section.
 * </ol>
 */
@ThreadSafe
public final class AsyncTaskCache<KeyT, ValueT> {

  private static final int STATE_ACTIVE = 0;
  private static final int STATE_SHUTDOWN = 1;
  private static final int STATE_TERMINATED = 2;

  /** Stands in for a {@code null} result, which {@link ConcurrentHashMap} cannot store. */
  private static final Object NULL_VALUE = new Object();

  /** A task that has been started, but whose result has not been published yet. */
  private record InProgressTask<KeyT, ValueT>(KeyT key, ListenableFuture<ValueT> future) {}

  private final TaskDeduplicator<KeyT, ValueT> deduplicator = new TaskDeduplicator<>();
  private final ConcurrentHashMap<KeyT, Object> finished = new ConcurrentHashMap<>();
  private final Set<InProgressTask<KeyT, ValueT>> inProgress = ConcurrentHashMap.newKeySet();
  private final AtomicInteger state = new AtomicInteger(STATE_ACTIVE);
  private final SettableFuture<Void> termination = SettableFuture.create();

  public static <KeyT, ValueT> AsyncTaskCache<KeyT, ValueT> create() {
    return new AsyncTaskCache<>();
  }

  /** Returns the set of keys of tasks that have finished successfully. */
  public ImmutableSet<KeyT> getFinishedTasks() {
    return ImmutableSet.copyOf(finished.keySet());
  }

  /** Returns the set of keys of tasks that are still executing. */
  public ImmutableSet<KeyT> getInProgressTasks() {
    return inProgress.stream().map(InProgressTask::key).collect(ImmutableSet.toImmutableSet());
  }

  /** Returns the number of callers awaiting an in progress task for {@code key}. */
  @VisibleForTesting
  public int getSubscriberCount(KeyT key) {
    return deduplicator.getActiveUseCount(key);
  }

  /**
   * Removes any cached result for the given {@code key}, so that the next call to {@link #execute}
   * for that key re-runs the task. Does not affect in-progress tasks.
   */
  public void invalidate(KeyT key) {
    finished.remove(key);
  }

  /**
   * Atomically replaces the cached result for {@code key} with {@code value}. The new value is
   * visible to subsequent {@link #execute} callers.
   */
  public void put(KeyT key, ValueT value) {
    finished.put(key, wrap(value));
  }

  /**
   * Executes a task if it hasn't been executed.
   *
   * @see #execute(Object, Supplier, boolean)
   */
  public ListenableFuture<ValueT> executeIfNot(
      KeyT key, Supplier<ListenableFuture<ValueT>> task) {
    return execute(key, task, /* force= */ false);
  }

  /**
   * Executes a task, unless an equivalent one has already finished successfully or is currently
   * running.
   *
   * <p>If the cache has been shut down, a cancelled future is returned.
   *
   * @param key identifies the task.
   * @param task supplies the future for a new execution of the task. It is invoked at most once,
   *     and only if a new execution is actually started, which makes it usable to detect that case.
   *     It runs while holding exclusive access to {@code key}, so it must be short, must not block
   *     on other tasks, and must report failures through the returned future rather than by
   *     throwing.
   * @param force start a new execution even if the task has already finished or is currently
   *     running.
   * @return a future which completes once the task is finished, or propagates its error.
   */
  public ListenableFuture<ValueT> execute(
      KeyT key, Supplier<ListenableFuture<ValueT>> task, boolean force) {
    if (state.get() != STATE_ACTIVE) {
      return immediateCancelledFuture();
    }

    if (force) {
      finished.remove(key);
    } else {
      Object cached = finished.get(key);
      if (cached != null) {
        return immediateFuture(unwrap(cached));
      }
    }

    var isNew = new boolean[1];
    Supplier<ListenableFuture<ValueT>> newExecution =
        () -> {
          if (!force) {
            // Re-check from within the deduplicator's per-key critical section: if the task
            // finished between the lookup above and here, it has already been removed from the
            // deduplicator and thus can no longer be joined, but its result is guaranteed to be
            // visible by now.
            Object cached = finished.get(key);
            if (cached != null) {
              return immediateFuture(unwrap(cached));
            }
          }
          isNew[0] = true;
          return registerTask(key, task.get());
        };

    ListenableFuture<ValueT> future =
        force
            ? deduplicator.executeUnconditionally(key, newExecution)
            : deduplicator.executeIfNew(key, newExecution);

    if (isNew[0] && state.get() != STATE_ACTIVE) {
      // Raced with a shutdown. The task exists only because of this call, so cancelling the future
      // returned to this (so far only) caller also cancels the task itself.
      future.cancel(/* mayInterruptIfRunning= */ true);
    }
    return future;
  }

  /**
   * Tracks {@code task} as in progress and publishes its result to {@link #finished} on success.
   *
   * <p>The returned future completes only after both have happened, which is what establishes the
   * ordering invariants documented on this class.
   */
  private ListenableFuture<ValueT> registerTask(KeyT key, ListenableFuture<ValueT> task) {
    var inProgressTask = new InProgressTask<>(key, task);
    inProgress.add(inProgressTask);
    ListenableFuture<ValueT> result =
        Futures.transform(
            task,
            value -> {
              finished.put(key, wrap(value));
              inProgress.remove(inProgressTask);
              return value;
            },
            directExecutor());
    result.addListener(
        () -> {
          // No-op unless the task failed or was cancelled, in which case its result is not cached.
          inProgress.remove(inProgressTask);
          maybeTerminate();
        },
        directExecutor());
    return result;
  }

  /**
   * Initiates an orderly shutdown in which preexisting tasks continue but new tasks are immediately
   * cancelled.
   */
  public void shutdown() {
    if (state.compareAndSet(STATE_ACTIVE, STATE_SHUTDOWN)) {
      maybeTerminate();
    }
  }

  /**
   * Initiates a forceful shutdown in which preexisting and new tasks are cancelled. Although
   * forceful, the shutdown process is still not instantaneous; {@link #isTerminated()} will likely
   * return {@code false} immediately after this method returns.
   */
  public void shutdownNow() {
    shutdown();

    for (InProgressTask<KeyT, ValueT> task : ImmutableList.copyOf(inProgress)) {
      task.future().cancel(/* mayInterruptIfRunning= */ true);
    }
  }

  /**
   * Waits for the tasks that are in progress at the time of the call to finish. Tasks that are
   * submitted afterwards are not waited for.
   *
   * <p>Task failures are not reported: they are propagated to whoever requested the task.
   */
  public void awaitInProgressTasks() throws InterruptedException {
    ImmutableList<ListenableFuture<ValueT>> futures =
        inProgress.stream().map(InProgressTask::future).collect(ImmutableList.toImmutableList());

    try {
      var unused = Futures.successfulAsList(futures).get();
    } catch (ExecutionException e) {
      throw new IllegalStateException("successfulAsList is not expected to fail", e);
    }
  }

  /** Waits for the cache to become terminated. */
  public void awaitTermination() throws InterruptedException {
    try {
      var unused = termination.get();
    } catch (ExecutionException e) {
      throw new IllegalStateException("termination is not expected to fail", e);
    } catch (CancellationException e) {
      throw new IllegalStateException("termination is not expected to be cancelled", e);
    }
  }

  /**
   * Returns whether the cache is shut down. A shut down cache immediately cancels any new tasks,
   * but may still have some tasks in progress.
   */
  public boolean isShutdown() {
    return state.get() != STATE_ACTIVE;
  }

  /**
   * Returns whether the cache is terminated. A terminated cache has no running tasks and has
   * released the relevant resources.
   */
  public boolean isTerminated() {
    return state.get() == STATE_TERMINATED;
  }

  private void maybeTerminate() {
    // Both the writer of state and the remover of the last in progress task run this check after
    // their own write, so at least one of them observes both.
    if (state.get() == STATE_SHUTDOWN
        && inProgress.isEmpty()
        && state.compareAndSet(STATE_SHUTDOWN, STATE_TERMINATED)) {
      // Reduce retained size in case references to the cache are held after shutdown.
      finished.clear();
      termination.set(null);
    }
  }

  private static Object wrap(Object value) {
    return value != null ? value : NULL_VALUE;
  }

  @SuppressWarnings("unchecked")
  private ValueT unwrap(Object value) {
    return value != NULL_VALUE ? (ValueT) value : null;
  }
}
