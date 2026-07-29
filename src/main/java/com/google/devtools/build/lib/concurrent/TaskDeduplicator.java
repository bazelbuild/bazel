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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.util.concurrent.AbstractFuture;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.errorprone.annotations.CheckReturnValue;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Deduplicates concurrent tasks identified by unique keys.
 *
 * <p>Every execution is started with caller-provided attributes of type {@code A} that describe
 * what it produces. A caller joins an ongoing execution for its key only if its own {@code canJoin}
 * predicate accepts that execution's attributes, which lets callers with different needs share work
 * asymmetrically: a caller that needs little can join an execution that produces more, while a
 * caller that needs more starts its own execution and takes over as the one that subsequent callers
 * see. Callers that all need the same thing pass a predicate that is always true.
 *
 * <p>At most one execution per key is joinable at a time, but an execution that has been taken over
 * keeps running for the callers that already joined it.
 *
 * <p>Any futures returned by this class can be individually canceled without affecting other
 * callers. The shared task is only canceled if all callers have canceled their futures and the task
 * is interrupted if and only if all callers requested interruption.
 *
 * <p>An execution is forgotten as soon as it completes, so a task that failed or was canceled is
 * retried by the next call for the same key. Callers that need results to be remembered across
 * calls have to layer their own cache on top.
 *
 * <p>The {@code taskSupplier} and {@code canJoin} callbacks run while a lock on an internal map is
 * held. They must therefore be cheap, must not block and must not call back into this deduplicator,
 * which would either deadlock or fail with an {@link IllegalStateException}. Supplying a task
 * typically means submitting it to an executor, which satisfies these requirements; running it
 * inline does not.
 */
public final class TaskDeduplicator<K, A, V> {
  private final ConcurrentMap<K, RefcountedFuture<A, V>> inFlightTasks = new ConcurrentHashMap<>();

  /**
   * Returns a future representing either the ongoing execution for the key, if {@code canJoin}
   * accepts its attributes, or a new execution started with the given attributes.
   *
   * <p>A new execution replaces the ongoing one as the execution that subsequent callers can join.
   * The replaced execution keeps running for the callers that already joined it.
   *
   * <p>The returned future must eventually be completed. The task is only canceled if the futures
   * returned to all callers for the same key have been canceled.
   *
   * <p>taskSupplier is called at most once and only if no ongoing execution is joined.
   */
  @CheckReturnValue
  public ListenableFuture<V> execute(
      K key,
      A attributes,
      Predicate<? super A> canJoin,
      Supplier<ListenableFuture<V>> taskSupplier) {
    var isNewHolder = new boolean[1];
    var future =
        inFlightTasks.compute(
            key,
            (unusedKey, ongoingExecution) -> {
              if (ongoingExecution != null
                  && canJoin.test(ongoingExecution.attributes())
                  && ongoingExecution.retain()) {
                return ongoingExecution;
              }
              isNewHolder[0] = true;
              return new RefcountedFuture<>(attributes, taskSupplier.get());
            });
    if (isNewHolder[0]) {
      future.addListener(() -> inFlightTasks.remove(key, future), directExecutor());
    }
    return IndividuallyCancelableFuture.wrap(future);
  }

  /**
   * Returns a future representing the ongoing execution for the key if {@code canJoin} accepts its
   * attributes, or null if there is no such execution.
   *
   * <p>A null return value doesn't necessarily mean that no task is running for the key: an
   * execution that is being canceled concurrently isn't joined either. Unlike {@link #execute},
   * this method never starts a new execution.
   *
   * <p>The returned future must eventually be completed. The task is only canceled if the futures
   * returned to all callers for the same key have been canceled.
   */
  @CheckReturnValue
  @Nullable
  public ListenableFuture<V> maybeJoinExecution(K key, Predicate<? super A> canJoin) {
    var future = inFlightTasks.get(key);
    if (future == null || !canJoin.test(future.attributes())) {
      return null;
    }
    if (!future.retain()) {
      inFlightTasks.remove(key, future);
      return null;
    }
    return IndividuallyCancelableFuture.wrap(future);
  }

  @VisibleForTesting
  @Nullable
  RefcountedFuture<A, V> ongoingExecutionForTesting(K key) {
    return inFlightTasks.get(key);
  }

  /**
   * A future adapter that is canceled only when {@link #cancel} has been called one more time than
   * {@link #retain}.
   */
  static final class RefcountedFuture<A, V> extends AbstractFuture<V> {
    private final A attributes;
    private final ListenableFuture<V> delegate;
    // Initialized to 1 in the constructor and incremented via retain(). Once it drops to 0, it
    // can never return to 1 or higher (0 is a sticky state).
    private final AtomicInteger refcount = new AtomicInteger(1);
    private volatile boolean mayInterruptIfRunning = true;

    RefcountedFuture(A attributes, ListenableFuture<V> delegate) {
      this.attributes = attributes;
      this.delegate = delegate;
      // Completes this future with the delegate's outcome and forwards a cancellation of this
      // future to the delegate.
      setFuture(delegate);
    }

    /** Returns the attributes this execution was started with. */
    A attributes() {
      return attributes;
    }

    @Override
    public boolean cancel(boolean mayInterruptIfRunning) {
      if (!mayInterruptIfRunning) {
        this.mayInterruptIfRunning = false;
      }
      // The write above happens-before the update below, which in turn happens-before the update
      // that drops the refcount to zero. The caller that observes zero thus sees the requests of
      // all other callers not to interrupt the task.
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

    /** Returns the number of callers that are still interested in this execution. */
    @VisibleForTesting
    int activeUses() {
      return refcount.get();
    }
  }

  /**
   * A future adapter that forwards cancellation requests to its delegate but cancels itself even if
   * the delegate doesn't.
   */
  private static final class IndividuallyCancelableFuture<V> extends AbstractFuture<V>
      implements Runnable {
    private final RefcountedFuture<?, V> delegate;

    static <V> ListenableFuture<V> wrap(RefcountedFuture<?, V> delegate) {
      var wrappedFuture = new IndividuallyCancelableFuture<>(delegate);
      delegate.addListener(wrappedFuture, directExecutor());
      return wrappedFuture;
    }

    IndividuallyCancelableFuture(RefcountedFuture<?, V> delegate) {
      this.delegate = delegate;
    }

    @Override
    public void run() {
      // The outcome is copied over manually rather than with setFuture: if this future has already
      // been canceled, setFuture would cancel the delegate a second time and thus decrement its
      // refcount twice on behalf of this single caller.
      try {
        set(Futures.getDone(delegate));
      } catch (ExecutionException e) {
        setException(e.getCause());
      } catch (CancellationException e) {
        // Either all callers canceled their futures, in which case this future is already canceled,
        // or the task completed as canceled on its own, e.g. because its executor was shut down.
        super.cancel(/* mayInterruptIfRunning= */ false);
      }
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
