// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.util.concurrent.AbstractFuture;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.errorprone.annotations.CheckReturnValue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Deduplicates concurrent tasks identified by unique keys. For any given key, only one task is
 * actively executed at a time.
 *
 * <p>Any futures returned by this class can be individually canceled without affecting other
 * callers. The shared task is only canceled if all callers have canceled their futures and the task
 * is interrupted if and only if all callers requested interruption.
 */
public final class TaskDeduplicator<K, V> {
  private final ConcurrentMap<K, RefcountedFuture<V>> inFlightTasks = new ConcurrentHashMap<>();

  /**
   * Returns a future representing either a new or already ongoing execution of the task.
   *
   * <p>The returned future must eventually be completed. The task is only canceled if the futures
   * returned to all callers for the same key have been canceled.
   *
   * <p>taskSupplier is called at most once per call to this method, and only if a new execution is
   * started, which makes it usable to detect that case. It runs while holding exclusive access to
   * {@code key}, so it must be short and must not block on other tasks.
   */
  @CheckReturnValue
  public ListenableFuture<V> executeIfNew(K key, Supplier<ListenableFuture<V>> taskSupplier) {
    while (true) {
      var isNewHolder = new boolean[1];
      var future =
          inFlightTasks.computeIfAbsent(
              key,
              unusedKey -> {
                isNewHolder[0] = true;
                return RefcountedFuture.wrap(taskSupplier.get());
              });
      if (isNewHolder[0]) {
        future.addListener(() -> inFlightTasks.remove(key, future), directExecutor());
      } else {
        // The shared future may have been canceled between the lookup and the call to retain().
        if (!future.retain()) {
          inFlightTasks.remove(key, future);
          continue;
        }
      }
      return IndividuallyCancelableFuture.wrap(future);
    }
  }

  /**
   * Returns a future representing either a new or already ongoing execution of the task that is
   * guaranteed to happen-after any executions started before the call of this method.
   *
   * <p>The returned future must eventually be completed. The task is only canceled if the futures
   * returned to all callers for the same key have been canceled.
   *
   * <p>taskSupplier is called at most once per call to this method, and only if a new execution is
   * started, which makes it usable to detect that case. It runs while holding exclusive access to
   * {@code key}, so it must be short and must not block on other tasks.
   */
  @CheckReturnValue
  public ListenableFuture<V> executeUnconditionally(
      K key, Supplier<ListenableFuture<V>> taskSupplier) {
    detach(key);
    return executeIfNew(key, taskSupplier);
  }

  /**
   * Detaches any ongoing execution of the task for the given key, so that a subsequent call to
   * {@link #executeIfNew} starts a new execution instead of joining it. The detached execution
   * keeps running for the callers that already joined it.
   *
   * <p>This method synchronizes with the registration of the detached execution: state written by
   * its registering thread before the registration is visible to the caller once this method
   * returns.
   */
  public void detach(K key) {
    inFlightTasks.remove(key);
  }

  /**
   * Returns the number of callers that are currently awaiting an ongoing execution of the task for
   * the given key, or 0 if there is none.
   */
  public int getActiveUseCount(K key) {
    var future = inFlightTasks.get(key);
    return future != null ? future.getActiveUseCount() : 0;
  }

  /**
   * Returns a future representing an already ongoing execution of the task or null if there is
   * none.
   *
   * <p>The returned future must eventually be completed. The task is only canceled if the futures
   * returned to all callers for the same key have been canceled.
   */
  @CheckReturnValue
  @Nullable
  public ListenableFuture<V> maybeJoinExecution(K key) {
    var future = inFlightTasks.get(key);
    if (future == null) {
      return null;
    }
    if (!future.retain()) {
      inFlightTasks.remove(key, future);
      return null;
    }
    return IndividuallyCancelableFuture.wrap(future);
  }

  /**
   * A future adapter that is canceled only when {@link #cancel} has been called one more time than
   * {@link #retain}.
   */
  private static final class RefcountedFuture<V> extends AbstractFuture<V> implements Runnable {
    private final ListenableFuture<V> delegate;
    // Initialized to 1 in the constructor and incremented via retain(). Once it drops to 0, it
    // can never return to 1 or higher (0 is a sticky state).
    private final AtomicInteger refcount = new AtomicInteger(1);
    private volatile boolean mayInterruptIfRunning = true;

    static <V> RefcountedFuture<V> wrap(ListenableFuture<V> delegate) {
      var wrappedFuture = new RefcountedFuture<>(delegate);
      delegate.addListener(wrappedFuture, directExecutor());
      return wrappedFuture;
    }

    RefcountedFuture(ListenableFuture<V> delegate) {
      this.delegate = delegate;
      setFuture(delegate);
    }

    @Override
    public void run() {}

    @Override
    public boolean cancel(boolean mayInterruptIfRunning) {
      if (!mayInterruptIfRunning) {
        this.mayInterruptIfRunning = false;
      }
      if (refcount.updateAndGet(oldCount -> oldCount >= 1 ? oldCount - 1 : 0) == 0) {
        return super.cancel(this.mayInterruptIfRunning);
      }
      return false;
    }

    @Nullable
    @Override
    protected String pendingToString() {
      return "delegate=[%s (%d active uses)]".formatted(delegate, refcount.get());
    }

    /** Retains the future, returning true if successful. */
    boolean retain() {
      return refcount.updateAndGet(oldCount -> oldCount >= 1 ? oldCount + 1 : 0) != 0;
    }

    int getActiveUseCount() {
      return refcount.get();
    }
  }

  /**
   * A future adapter that forwards cancellation requests to its delegate but cancels itself even if
   * the delegate doesn't.
   */
  private static final class IndividuallyCancelableFuture<V> extends AbstractFuture<V>
      implements Runnable {
    private final RefcountedFuture<V> delegate;

    static <V> ListenableFuture<V> wrap(RefcountedFuture<V> delegate) {
      var wrappedFuture = new IndividuallyCancelableFuture<>(delegate);
      delegate.addListener(wrappedFuture, directExecutor());
      return wrappedFuture;
    }

    IndividuallyCancelableFuture(RefcountedFuture<V> delegate) {
      this.delegate = delegate;
    }

    @Override
    public void run() {
      setFuture(delegate);
    }

    @Override
    public boolean cancel(boolean mayInterruptIfRunning) {
      var didCancel = super.cancel(mayInterruptIfRunning);
      if (didCancel) {
        delegate.cancel(mayInterruptIfRunning);
      }
      return didCancel;
    }

    @Nullable
    @Override
    protected String pendingToString() {
      return "delegate=[%s]".formatted(delegate);
    }
  }
}
