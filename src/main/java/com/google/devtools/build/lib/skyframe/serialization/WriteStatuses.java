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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.util.concurrent.AbstractFuture;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.concurrent.QuiescingFuture;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Collection;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;

/**
 * Container for {@link WriteStatus} and its implementations.
 *
 * <p>The alternative of having {@link WriteStatus} as the top level type and its implementations as
 * inner classes, requires all the implementations to be public.
 */
public class WriteStatuses {


  /** Returns the stateless, immediately successful write status. */
  public static WriteStatus immediateWriteStatus() {
    return ImmediateWriteStatus.NOVEL;
  }

  /**
   * Returns a stateless, immediately successful write status with the given novelty.
   *
   * @param wasNovel true if new bytes were actually written; false if they already existed in the
   *     backend.
   */
  public static WriteStatus immediateWriteStatus(boolean wasNovel) {
    return wasNovel ? ImmediateWriteStatus.NOVEL : ImmediateWriteStatus.NOT_NOVEL;
  }

  /** Creates an immediately failed write status. */
  public static WriteStatus immediateFailedWriteStatus(Throwable cause) {
    return new ImmediateFailedWriteStatus(cause);
  }


  /** Combines {@code futures} into a single future (general purpose). */
  public static WriteStatus aggregateWriteStatuses(Collection<WriteStatus> writeStatuses) {
    if (writeStatuses.isEmpty()) {
      return immediateWriteStatus();
    }
    if (writeStatuses.size() == 1) {
      return writeStatuses.iterator().next();
    }
    return AggregateWriteStatus.create(writeStatuses);
  }

  /**
   * A general purpose, reference-count-based {@link WriteStatus} aggregator.
   *
   * <p>This class implements {@link WriteStatus} and thus extends {@link ListenableFuture<Boolean>}
   * (via {@link QuiescingFuture<Boolean>}) to track novelty.
   *
   * <p>Uses less memory in-flight than {@link Futures#whenAllSucceed} because it does not retain
   * the list of input futures and therefore also releases those futures earlier.
   *
   * <p>Preserves all callback edges.
   */
  private static final class AggregateWriteStatus extends QuiescingFuture<Boolean>
      implements WriteStatus, FutureCallback<Boolean> {
    private volatile boolean wasNovel = false;

    private static WriteStatus create(Iterable<WriteStatus> writeStatuses) {
      return new WriteStatusBuilder().addAll(writeStatuses).build();
    }

    private AggregateWriteStatus() {
      super(directExecutor());
    }

    @Override
    protected Boolean getValue() {
      return wasNovel;
    }

    /**
     * Implementation of {@link FutureCallback<Boolean>}.
     *
     * @deprecated only used by {@link #create} for callback processing
     */
    @Deprecated
    @Override
    public void onSuccess(Boolean novel) {
      if (novel) {
        var unused = WAS_NOVEL_HANDLE.compareAndSet(this, false, true);
      }
      decrement();
    }

    /**
     * Implementation of {@link FutureCallback<Boolean>}.
     *
     * @deprecated only used by {@link #create} for callback processing
     */
    @Deprecated
    @Override
    public void onFailure(Throwable t) {
      if (t instanceof CancellationException) {
        cancel(/* mayInterruptIfRunning= */ false); // nothing running
        return;
      }
      notifyException(t);
    }

    private void add(ListenableFuture<Boolean> status) {
      increment();
      Futures.addCallback(status, (FutureCallback<Boolean>) this, directExecutor());
    }

    private void clearPreincrement() {
      decrement();
    }

    private static final VarHandle WAS_NOVEL_HANDLE;

    static {
      try {
        WAS_NOVEL_HANDLE =
            MethodHandles.lookup()
                .findVarHandle(AggregateWriteStatus.class, "wasNovel", boolean.class);
      } catch (ReflectiveOperationException e) {
        throw new ExceptionInInitializerError(e);
      }
    }
  }

  /**
   * Builder for {@link WriteStatus}.
   *
   * <p>This builder is thread safe, and {@link #build} is idempotent. Neither {@link #add} nor
   * {@link #addAll} may be called after {@link #build}.
   */
  public static final class WriteStatusBuilder {
    private ListenableFuture<Boolean> first = null;
    private AggregateWriteStatus aggregate = null;
    private boolean built = false;

    /**
     * Adds a status to the aggregate.
     *
     * @throws IllegalStateException if called after {@link #build}
     */
    @CanIgnoreReturnValue
    public synchronized WriteStatusBuilder add(ListenableFuture<Boolean> status) {
      checkState(!built, "cannot add to WriteStatusBuilder after build()");
      if (first == null) {
        first = status;
      } else if (aggregate == null) {
        aggregate = new AggregateWriteStatus();
        aggregate.add(first);
        aggregate.add(status);
      } else {
        aggregate.add(status);
      }
      return this;
    }

    /**
     * Adds all statuses to the aggregate.
     *
     * @throws IllegalStateException if called after {@link #build}
     */
    @CanIgnoreReturnValue
    public synchronized WriteStatusBuilder addAll(
        Iterable<? extends ListenableFuture<Boolean>> statuses) {
      for (ListenableFuture<Boolean> status : statuses) {
        add(status);
      }
      return this;
    }

    /**
     * Builds and returns the aggregated {@link WriteStatus}.
     *
     * <p>This method is idempotent; subsequent calls return the same {@link WriteStatus}.
     */
    public synchronized WriteStatus build() {
      if (first == null) {
        built = true;
        // Zero dependency statuses.
        return immediateWriteStatus();
      }
      if (aggregate == null) {
        built = true;
        // One dependency status. Return it, possibly wrapping it with SettableWriteStatus.
        if (first instanceof WriteStatus) {
          return (WriteStatus) first;
        }
        SettableWriteStatus wrapper = new SettableWriteStatus();
        wrapper.completeWithFuture(first);
        first = wrapper;
        return wrapper;
      }
      if (!built) {
        aggregate.clearPreincrement();
        built = true;
      }
      return aggregate;
    }
  }

  /**
   * A settable {@link WriteStatus}, analogous to {@link
   * com.google.common.util.concurrent.SettableFuture}.
   */
  public static final class SettableWriteStatus extends AbstractFuture<Boolean>
      implements WriteStatus {
    /**
     * Signals the successful completion of the write operation with novelty information.
     *
     * @param wasNovel true if new bytes were actually written; false if they already existed in the
     *     backend.
     */
    public void markSuccess(boolean wasNovel) {
      checkState(set(wasNovel), "attempted to markSuccess already set %s", this);
    }

    /** Signals the successful completion of the write operation with novelty set to true. */
    public void markSuccess() {
      markSuccess(/* wasNovel= */ true);
    }

    public void failWith(Throwable cause) {
      if (cause instanceof CancellationException) {
        checkState(
            cancel(/* mayInterruptIfRunning= */ false),
            "attempted to failWith(%s) already set %s",
            cause,
            this);
        return;
      }
      checkState(setException(cause), "attempted to failWith(%s) already set %s", cause, this);
    }

    public void completeWith(WriteStatus future) {
      checkState(setFuture(future), "attempted to completeWith(%s) already set %s", future, this);
    }

    void completeWithFuture(ListenableFuture<Boolean> future) {
      checkState(
          setFuture(future), "attempted to completeWithFuture(%s) already set %s", future, this);
    }
  }

  private static final class ImmediateWriteStatus implements WriteStatus {
    private static final ImmediateWriteStatus NOVEL = new ImmediateWriteStatus(true);
    private static final ImmediateWriteStatus NOT_NOVEL = new ImmediateWriteStatus(false);

    private final boolean wasNovel;

    private ImmediateWriteStatus(boolean wasNovel) {
      this.wasNovel = wasNovel;
    }

    @Override
    public void addListener(Runnable listener, Executor executor) {
      executor.execute(listener); // Immediately executes listener.
    }

    @Override
    public boolean cancel(boolean mayInterruptIfRunning) {
      return false;
    }

    @Override
    public Boolean get() {
      return wasNovel;
    }

    @Override
    public Boolean get(long timeout, TimeUnit unit) {
      return wasNovel;
    }

    @Override
    public boolean isCancelled() {
      return false;
    }

    @Override
    public boolean isDone() {
      return true;
    }
  }

  private static final class ImmediateFailedWriteStatus implements WriteStatus {
    private final ExecutionException exception;

    private ImmediateFailedWriteStatus(Throwable cause) {
      this.exception = new ExecutionException(cause);
    }

    @Override
    public void addListener(Runnable listener, Executor executor) {
      executor.execute(listener); // Immediately executes listener.
    }

    @Override
    public boolean cancel(boolean mayInterruptIfRunning) {
      return false;
    }

    @Override
    public Boolean get() throws ExecutionException {
      throw exception;
    }

    @Override
    public Boolean get(long timeout, TimeUnit unit) throws ExecutionException {
      return get();
    }

    @Override
    public boolean isCancelled() {
      return false;
    }

    @Override
    public boolean isDone() {
      return true;
    }
  }

  private WriteStatuses() {} // namespace only
}
