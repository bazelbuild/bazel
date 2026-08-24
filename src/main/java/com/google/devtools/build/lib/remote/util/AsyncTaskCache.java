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
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

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
 * TaskDeduplicator} and {@link ConcurrentHashMap}. The only caller-provided code that runs in a
 * critical section is the task supplier, which executes in the deduplicator's per-key section and
 * must therefore be short. Two ordering invariants make this safe:
 *
 * <ol>
 *   <li>A task that completes successfully publishes its result to {@link #finished} and is
 *       unregistered from {@link #inProgress} strictly before the future handed to callers
 *       completes, so a caller that observes a task as no longer running also observes its result.
 *       (A task superseded by a forced re-execution deliberately publishes nothing: re-running it
 *       is always safe, whereas caching its stale result would not be.)
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
  private static final class InProgressTask<KeyT, ValueT> {
    private final KeyT key;
    private final ListenableFuture<ValueT> future;

    /**
     * Set when a forced re-execution detaches this task from the deduplicator; a superseded task's
     * result may be stale and must not remain published.
     */
    private volatile boolean superseded;

    InProgressTask(KeyT key, ListenableFuture<ValueT> future) {
      this.key = key;
      this.future = future;
    }

    KeyT key() {
      return key;
    }

    ListenableFuture<ValueT> future() {
      return future;
    }
  }

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
  public ListenableFuture<ValueT> executeIfNot(KeyT key, Supplier<ListenableFuture<ValueT>> task) {
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
   *     It runs while holding exclusive access to {@code key}, so it must be short and must not
   *     block on other tasks. An unchecked exception it throws is delivered through the returned
   *     future.
   * @param force disregard both a cached result and an in-flight execution and execute the task
   *     anew. Concurrent forced calls may share a single new execution, which is guaranteed to have
   *     started only after each of them detached the old one.
   * @return a future which completes once the task is finished, or propagates its error.
   */
  public ListenableFuture<ValueT> execute(
      KeyT key, Supplier<ListenableFuture<ValueT>> task, boolean force) {
    if (state.get() != STATE_ACTIVE) {
      return immediateCancelledFuture();
    }

    if (force) {
      // Detach any in-flight execution first: detaching synchronizes with its registration, so
      // the sweep below finds it in the in-progress set unless it has already completed, in which
      // case its published result is visible to the remove below. A superseded execution keeps
      // running for its existing callers, but its potentially stale result must not remain
      // published: since it is marked before the cached result is removed, it either observes the
      // mark when publishing or has already published a result that the remove drops.
      deduplicator.detach(key);
      for (InProgressTask<KeyT, ValueT> inProgressTask : inProgress) {
        if (inProgressTask.key().equals(key)) {
          inProgressTask.superseded = true;
        }
      }
      finished.remove(key);
    } else {
      Object cached = finished.get(key);
      if (cached != null) {
        return immediateFuture(unwrap(cached));
      }
    }

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
          ListenableFuture<ValueT> taskFuture;
          try {
            taskFuture = task.get();
          } catch (RuntimeException e) {
            // Deliver synchronous failures through the returned future so that callers only have
            // to handle a single failure channel.
            taskFuture = immediateFailedFuture(e);
          }
          ListenableFuture<ValueT> result = registerTask(key, taskFuture);
          if (state.get() != STATE_ACTIVE) {
            // Raced with a shutdown, whose cancellation sweep may have missed the task registered
            // just now. Cancel the underlying task directly, which takes effect even if other
            // callers have already joined it, and drop any result it may have published after
            // termination cleared the cache.
            taskFuture.cancel(/* mayInterruptIfRunning= */ true);
            finished.remove(key);
            return immediateCancelledFuture();
          }
          return result;
        };

    // With force, the explicit detach above has already made room for a new execution; a caller
    // that loses the race to start it joins an execution that began after the detach, which is
    // just as fresh.
    return deduplicator.executeIfNew(key, newExecution);
  }

  /**
   * Tracks {@code task} as in progress and, unless a forced re-execution supersedes it in the
   * meantime, publishes its result to {@link #finished} on success.
   *
   * <p>On success, the returned future completes only after the result has been published and the
   * task unregistered, which establishes the first ordering invariant documented on this class.
   */
  private ListenableFuture<ValueT> registerTask(KeyT key, ListenableFuture<ValueT> task) {
    var inProgressTask = new InProgressTask<>(key, task);
    inProgress.add(inProgressTask);
    ListenableFuture<ValueT> result =
        Futures.transform(
            task,
            value -> {
              if (!inProgressTask.superseded) {
                Object wrapped = wrap(value);
                finished.put(key, wrapped);
                if (inProgressTask.superseded) {
                  // A forced re-execution superseded this one between the check above and the
                  // put, so this result may be stale. The value-conditioned remove keeps a
                  // distinct result already published by the forced execution; removing an equal
                  // one merely costs a re-execution on the next call.
                  finished.remove(key, wrapped);
                }
              }
              inProgress.remove(inProgressTask);
              return value;
            },
            directExecutor());
    result.addListener(
        () -> {
          // Only removes anything if the task failed or was cancelled; the transform above has
          // already unregistered a successful task.
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
