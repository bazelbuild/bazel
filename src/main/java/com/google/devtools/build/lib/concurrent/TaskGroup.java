// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.base.Throwables.throwIfInstanceOf;
import static com.google.common.base.Throwables.throwIfUnchecked;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** An API for structured concurrency, inspired by JDK's {@code StructuredTaskScope}. */
public class TaskGroup<T, R> implements AutoCloseable {
  private final ThreadFactory threadFactory;
  private final Policy<? super T> policy;
  private final Joiner<? super T, ? extends R> joiner;

  private final Thread owner;
  private final Set<Thread> threads;

  private enum TaskGroupState {
    NEW,
    FORKED, // subtasks forked, need to join.
    JOIN_STARTED, // join started, can no longer fork
    JOIN_COMPLETED, // join completed
    CLOSED;
  }

  // state, only accessed by owner thread
  private TaskGroupState state;

  // set or read by any thread
  private final AtomicBoolean cancelled;

  // set to 1 + number of subtasks forked and not yet joined
  private final IncrementableCountDownLatch latch;

  private TaskGroup(
      ThreadFactory threadFactory,
      Policy<? super T> policy,
      Joiner<? super T, ? extends R> joiner) {
    this.threadFactory = threadFactory;
    this.policy = policy;
    this.joiner = joiner;
    this.owner = Thread.currentThread();
    this.threads = Sets.newConcurrentHashSet();
    this.latch = new IncrementableCountDownLatch(1);
    this.state = TaskGroupState.NEW;
    this.cancelled = new AtomicBoolean(false);
  }

  private void ensureOwner() {
    if (Thread.currentThread() != owner) {
      throw new IllegalStateException("Current thread not owner");
    }
  }

  private void ensureNotJoined() {
    if (state.compareTo(TaskGroupState.FORKED) > 0) {
      throw new IllegalStateException("Already joined or task group is closed");
    }
  }

  private void ensureJoinedIfOwner() {
    if (Thread.currentThread() == owner && state.compareTo(TaskGroupState.JOIN_STARTED) <= 0) {
      throw new IllegalStateException("join not called");
    }
  }

  @SuppressWarnings("AllowVirtualThreads")
  private static ThreadFactory defaultThreadFactory() {
    return Thread.ofVirtual().factory();
  }

  /**
   * Opens a new task group with the given policy and joiner. It should be used with
   * try-with-resources statement like:
   *
   * <pre>{@code
   * try (var group = TaskGroup.open(policy, joiner)) {
   *   ...
   * }
   * }</pre>
   *
   * <p>The calling thread becomes the task group's owner and is the only thread allowed to call
   * {@link #fork}, {@link #join} or {@link #close} on it.
   */
  public static <T, R> TaskGroup<T, R> open(
      Policy<? super T> policy, Joiner<? super T, ? extends R> joiner) {
    return new TaskGroup<>(defaultThreadFactory(), policy, joiner);
  }

  /**
   * Forks a subtask to be executed in a new thread. The new thread execute the subtasks
   * concurrently with the current thread.
   *
   * <p>If a new thread cannot be created, a {@link RejectedExecutionException} is thrown.
   *
   * <p>If the task completes successfully, the result is available through {@link Subtask#get}. If
   * the task fails, the exception is available through {@link Subtask#exception}. If the task group
   * is cancelled, the task is not started, neither method can be used to obtain the outcome.
   *
   * @throws IllegalStateException if not called from the owner thread, or if the task group is
   *     already joined.
   */
  @CanIgnoreReturnValue
  public <U extends T> Subtask<U> fork(Callable<? extends U> task) {
    ensureOwner();
    ensureNotJoined();

    var subtask = new SubtaskImpl<U>(this, task);

    if (!cancelled.get()) {
      var thread = threadFactory.newThread(subtask);
      if (thread == null) {
        throw new RejectedExecutionException("Rejected by thread factory");
      }
      latch.increment();
      thread.start();
    }

    state = TaskGroupState.FORKED;
    return subtask;
  }

  @CanIgnoreReturnValue
  public <U extends T> Subtask<U> fork(Runnable task) {
    return fork(
        () -> {
          task.run();
          return null;
        });
  }

  /**
   * Returns a result or throws per the {@link Joiner}, after waiting for subtasks to complete per
   * the {@link Policy}.
   *
   * <p>This method must be called if {@link #fork} has been called at least once. Once it returns
   * without interruption, it must not be called again.
   *
   * @throws IllegalStateException if called from a thread other than the owner
   * @throws InterruptedException if interrupted while waiting for subtasks to complete
   */
  @CanIgnoreReturnValue
  public R join() throws ExecutionException, InterruptedException {
    ensureOwner();
    ensureNotJoined();

    state = TaskGroupState.JOIN_STARTED;

    latch.countDown();
    // If the await is interrupted, the group will be cancelled inside {@link #close}.
    latch.await();

    state = TaskGroupState.JOIN_COMPLETED;

    try {
      return joiner.result();
    } catch (Throwable e) {
      throw new ExecutionException(e);
    }
  }

  /**
   * Similar to {@link #join}, but throws the checked exception from the subtasks instead of
   * wrapping them in an {@link ExecutionException}. If a subtask throws an exception that doesn't
   * match the given class, an {@link IllegalStateException} is thrown with the cause set to the
   * actual exception.
   */
  public <E extends Exception> R joinOrThrow(Class<E> exceptionClass)
      throws E, InterruptedException {
    return joinOrThrowInternal(exceptionClass, null, null);
  }

  /**
   * Similar to {@link #join}, but throws the checked exception from the subtasks instead of
   * wrapping them in an {@link ExecutionException}. If a subtask throws an exception that doesn't
   * match the given class, an {@link IllegalStateException} is thrown with the cause set to the
   * actual exception.
   */
  public <E1 extends Exception, E2 extends Exception> R joinOrThrow(
      Class<E1> exceptionClass1, Class<E2> exceptionClass2) throws E1, E2, InterruptedException {
    return joinOrThrowInternal(exceptionClass1, exceptionClass2, null);
  }

  /**
   * Similar to {@link #join}, but throws the checked exception from the subtasks instead of
   * wrapping them in an {@link ExecutionException}. If a subtask throws an exception that doesn't
   * match the given class, an {@link IllegalStateException} is thrown with the cause set to the
   * actual exception.
   */
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception> R joinOrThrow(
      Class<E1> exceptionClass1, Class<E2> exceptionClass2, Class<E3> exceptionClass3)
      throws E1, E2, E3, InterruptedException {
    return joinOrThrowInternal(exceptionClass1, exceptionClass2, exceptionClass3);
  }

  private <E1 extends Exception, E2 extends Exception, E3 extends Exception> R joinOrThrowInternal(
      @Nullable Class<E1> exceptionClass1,
      @Nullable Class<E2> exceptionClass2,
      @Nullable Class<E3> exceptionClass3)
      throws E1, E2, E3, InterruptedException {
    try {
      return join();
    } catch (ExecutionException e) {
      var cause = e.getCause();
      if (exceptionClass1 != null) {
        throwIfInstanceOf(cause, exceptionClass1);
      }
      if (exceptionClass2 != null) {
        throwIfInstanceOf(cause, exceptionClass2);
      }
      if (exceptionClass3 != null) {
        throwIfInstanceOf(cause, exceptionClass3);
      }
      throwIfUnchecked(cause);
      throw new IllegalStateException(cause);
    }
  }

  /** Returns whether the group is cancelled or in the process of being cancelled. */
  public boolean isCancelled() {
    return cancelled.get();
  }

  private void onComplete(Subtask<? extends T> subtask, Thread thread) {
    try {
      if (subtask.state() != Subtask.State.UNAVAILABLE) {
        // We want to call Joiner#onComplete first, so that if subtask failed and the policy decides
        // to cancel the group, the joiner can see the exception from this subtask first. Otherwise,
        // the exception from this subtask may race with the InterruptedException from other
        // subtasks that are cancelled. This will cause the joiner to sometimes throw
        // InterruptedException instead of the exception from this subtask, if the joiner only
        // throws one exception.
        joiner.onComplete(subtask);
        if (policy.onComplete(subtask)) {
          cancel();
        }
      }
    } finally {
      threads.remove(thread);
      latch.countDown();
    }
  }

  @SuppressWarnings("Interruption")
  private void interruptAll() {
    var currentThread = Thread.currentThread();
    for (var thread : ImmutableSet.copyOf(threads)) {
      if (thread != currentThread) {
        thread.interrupt();
      }
    }
  }

  /**
   * Cancels the task group if not already cancelled.
   *
   * <p>Cancellation will interrupt all subtask threads in the task group. No new subtasks can be
   * forked after cancellation.
   *
   * <p>This method can be called by any subtask threads.
   */
  private void cancel() {
    if (cancelled.compareAndSet(false, true)) {
      interruptAll();
    }
  }

  /**
   * @throws IllegalStateException if {@link #fork} was called at least once and {@link #join} was
   *     never called
   */
  @Override
  public void close() {
    ensureOwner();

    TaskGroupState s = state;
    switch (s) {
      case TaskGroupState.NEW -> {
        // If the group is new, the latch was never decremented. We need to decrement it here
        // because the latch is initialized with a count of 1.
        latch.countDown();
      }
      case TaskGroupState.FORKED -> {
        // throw if the owner didn't join after forking
        throw new IllegalStateException("Owner did not join after forking");
      }
      case TaskGroupState.JOIN_STARTED -> {
        // Cancel the group if join did not complete.
        cancel();
      }
      case TaskGroupState.JOIN_COMPLETED -> {}
      case TaskGroupState.CLOSED -> {
        return;
      }
    }

    try {
      latch.awaitUninterruptibly();
    } finally {
      state = TaskGroupState.CLOSED;
    }
  }

  @VisibleForTesting
  ImmutableSet<Thread> getThreads() {
    return ImmutableSet.copyOf(threads);
  }

  /** A subtask forked with {@link #fork}. */
  public interface Subtask<T> extends Supplier<T> {
    /** The state of the subtask. */
    enum State {
      UNAVAILABLE,
      FAILED,
      SUCCESS,
    }

    /** Returns the state of the subtask. */
    State state();

    /**
     * Returns the result of the subtask if it completed successfully.
     *
     * @throws IllegalStateException if the subtask has not completed, or did not complete
     *     successfully.
     */
    @Override
    T get();

    /**
     * Returns the exception thrown by the subtask if it failed.
     *
     * @throws IllegalStateException if the subtask has not completed, or did not fail.
     */
    Throwable exception();
  }

  private static final class SubtaskImpl<T> implements Subtask<T>, Runnable {
    private static final NullOrExceptionResult RESULT_NULL = new NullOrExceptionResult(null);

    private final TaskGroup<? super T, ?> taskGroup;
    private final Callable<? extends T> task;

    private volatile Object result;

    private SubtaskImpl(TaskGroup<? super T, ?> taskGroup, Callable<? extends T> task) {
      this.taskGroup = taskGroup;
      this.task = task;
    }

    @Override
    public void run() {
      Thread thread = Thread.currentThread();
      boolean added = taskGroup.threads.add(thread);
      checkState(added);
      try {
        if (taskGroup.cancelled.get()) {
          // If the task group was cancelled, skip the task. We must check the cancellation state
          // after adding the thread to the set to avoid a race with {@link #cancel}.
          return;
        }

        T result = null;
        Throwable ex = null;
        try {
          result = task.call();
        } catch (Throwable e) {
          ex = e;
        }

        if (ex == null) {
          this.result = result != null ? result : RESULT_NULL;
        } else {
          this.result = new NullOrExceptionResult(ex);
        }
      } finally {
        taskGroup.onComplete(this, thread);
      }
    }

    @Override
    public Subtask.State state() {
      Object result = this.result;
      if (result == null) {
        return State.UNAVAILABLE;
      } else if (result instanceof NullOrExceptionResult nullOrExceptionResult) {
        // null or failed
        return nullOrExceptionResult.exception() == null ? State.SUCCESS : State.FAILED;
      } else {
        return State.SUCCESS;
      }
    }

    @Override
    public T get() {
      taskGroup.ensureJoinedIfOwner();
      Object result = this.result;
      if (result instanceof NullOrExceptionResult nullOrExceptionResult) {
        if (nullOrExceptionResult.exception() == null) {
          return null;
        }
      } else if (result != null) {
        @SuppressWarnings("unchecked")
        T r = (T) result;
        return r;
      }
      throw new IllegalStateException(
          "Result is unavailable or subtask did not complete successfully");
    }

    @Override
    public Throwable exception() {
      taskGroup.ensureJoinedIfOwner();
      Object result = this.result;
      if (result instanceof NullOrExceptionResult nullOrExceptionResult) {
        if (nullOrExceptionResult.exception() != null) {
          return nullOrExceptionResult.exception();
        }
      }
      throw new IllegalStateException(
          "Result is unavailable or subtask did not complete with exception");
    }

    @Override
    public String toString() {
      String stateAsString =
          switch (state()) {
            case UNAVAILABLE -> "[Unavailable]";
            case SUCCESS -> "[Completed successfully]";
            case FAILED -> "[Failed: " + ((NullOrExceptionResult) result).exception() + "]";
          };
      return Objects.toIdentityString(this) + stateAsString;
    }

    /** A result of a subtask that is either null or an exception. */
    private record NullOrExceptionResult(@Nullable Throwable exception) {}
  }

  /** An object that can be used to cancel the task group depending on the subtask state. */
  public interface Policy<T> {
    /**
     * Called by the thread that started the subtask when it completes.
     *
     * @return true to cancel the task group.
     */
    default boolean onComplete(Subtask<? extends T> subtask) {
      return false;
    }
  }

  /** A collection of {@link Policy} implementations. */
  public static class Policies {
    private Policies() {}

    /** Returns a policy that cancels the task group if any subtask fails. */
    @SuppressWarnings("unchecked")
    public static <T> Policy<T> allSuccessful() {
      return (Policy<T>) ALL_SUCCESSFUL;
    }

    private static final Policy<Object> ALL_SUCCESSFUL =
        new Policy<Object>() {
          @Override
          public boolean onComplete(Subtask<? extends Object> subtask) {
            return subtask.state() == Subtask.State.FAILED;
          }
        };

    /** Returns a policy that cancels the task group if any subtask succeeds. */
    @SuppressWarnings("unchecked")
    public static <T> Policy<T> anySuccessful() {
      return (Policy<T>) ANY_SUCCESSFUL;
    }

    private static final Policy<Object> ANY_SUCCESSFUL =
        new Policy<Object>() {
          @Override
          public boolean onComplete(Subtask<? extends Object> subtask) {
            return subtask.state() == Subtask.State.SUCCESS;
          }
        };

    /** Returns a policy that waits for all subtasks to complete, no matter their state. */
    @SuppressWarnings("unchecked")
    public static <T> Policy<T> allCompleted() {
      return (Policy<T>) ALL_COMPLETED;
    }

    private static final Policy<Object> ALL_COMPLETED = new Policy<Object>() {};
  }

  /**
   * An object used to process the result of subtasks and produce the final result for the task
   * group.
   */
  public interface Joiner<T, R> {
    /** Called by the thread that started the subtask when it completes. */
    void onComplete(Subtask<? extends T> subtask);

    /**
     * Called by {@link #join} to get the final result after waiting for all subtasks to complete.
     * The result from this method is returned by {@link #join}. If this method throws, then {@link
     * #join} throws an {@link ExecutionException} which the exception thrown by this method as the
     * cause.
     */
    R result() throws Throwable;
  }

  /** A collection of {@link Joiner} implementations. */
  public static class Joiners {
    private Joiners() {}

    /**
     * Returns a joiner that returns the result of all subtasks that complete successfully.
     *
     * <p>If any subtask fails, the joiner causes {@link #join} to throw.
     *
     * <p>The order of the items in the returned list is undefined - it is not guaranteed to be the
     * same as the order in which the subtasks were forked.
     */
    public static <T> Joiner<T, List<T>> allSuccessfulOrThrow() {
      return new AllSuccessfulOrThrow<T>();
    }

    private static final class AllSuccessfulOrThrow<T> implements Joiner<T, List<T>> {
      private final ConcurrentLinkedDeque<T> results = new ConcurrentLinkedDeque<>();
      private volatile Throwable error;

      @Override
      public void onComplete(Subtask<? extends T> subtask) {
        Subtask.State state = subtask.state();
        if (state == Subtask.State.FAILED) {
          if (error == null) {
            // There might be a race here, but it doesn't matter which error got set.
            error = subtask.exception();
          }
        } else {
          results.add(subtask.get());
        }
      }

      @Override
      public ImmutableList<T> result() throws Throwable {
        Throwable e = error;
        if (e != null) {
          throw e;
        } else {
          return ImmutableList.copyOf(results);
        }
      }
    }

    /**
     * Returns a joiner that returns the result of an arbitrarily chosen subtask that completes
     * successfully.
     *
     * <p>If all subtasks fail, the joiner causes {@link #join} to throw {@link
     * NoSuchElementException}.
     */
    public static <T> Joiner<T, T> anySuccessfulOrThrow() {
      return new AnySuccessfulOrThrow<T>();
    }

    private static final class AnySuccessfulOrThrow<T> implements Joiner<T, T> {
      private final AtomicReference<Subtask<? extends T>> subtaskRef = new AtomicReference<>(null);

      @Override
      public void onComplete(Subtask<? extends T> subtask) {
        Subtask.State newState = subtask.state();
        Subtask<? extends T> oldSubtask;
        while (((oldSubtask = subtaskRef.get()) == null)
            || oldSubtask.state().compareTo(newState) < 0) {
          if (subtaskRef.compareAndSet(oldSubtask, subtask)) {
            return;
          }
        }
      }

      @Override
      public T result() throws Throwable {
        var subtask = this.subtaskRef.get();
        if (subtask == null) {
          throw new NoSuchElementException("No subtasks completed");
        }
        return switch (subtask.state()) {
          case SUCCESS -> subtask.get();
          case FAILED -> throw subtask.exception();
          default -> throw new IllegalStateException("Unexpected state: " + subtask.state());
        };
      }
    }

    /**
     * Returns a joiner that ignores the result of successful subtasks.
     *
     * <p>If any subtask fails, the joiner causes {@link #join} to throw.
     */
    public static <T> Joiner<T, Void> voidOrThrow() {
      return new VoidOrThrow<T>();
    }

    @VisibleForTesting
    static final class VoidOrThrow<T> implements Joiner<T, Void> {
      private volatile Throwable error;

      @Override
      public void onComplete(Subtask<? extends T> subtask) {
        Subtask.State state = subtask.state();
        if (state == Subtask.State.FAILED && error == null) {
          // There might be a race here, but it doesn't matter which error got set.
          error = subtask.exception();
        }
      }

      @Override
      public Void result() throws Throwable {
        Throwable e = error;
        if (e != null) {
          throw e;
        } else {
          return null;
        }
      }

      @VisibleForTesting
      Throwable getError() {
        return error;
      }
    }
  }
}
