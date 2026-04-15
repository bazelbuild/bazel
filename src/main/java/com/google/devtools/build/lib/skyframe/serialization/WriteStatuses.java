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
import com.google.errorprone.annotations.DoNotCall;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Collection;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Container for {@link WriteStatus} and its implementations.
 *
 * <p>The alternative of having {@link WriteStatus} as the top level type and its implementations as
 * inner classes, requires all the implementations to be public.
 */
public class WriteStatuses {
  /**
   * Represents future success or failure of a write operation.
   *
   * <p>This can act like an ordinary future, but has special case, memory saving handling for
   * aggregation.
   *
   * <p>The {@link Boolean} result of this future indicates the "novelty" of the write. A {@code
   * true} result means new bytes were actually written to the storage backend; {@code false} means
   * they were already present. Novelty tracking is used for metrics and defaults to {@code true} if
   * the backend configuration doesn't support it.
   *
   * <p>OR semantics are used for aggregation: an aggregate is novel if any of its components are
   * novel.
   */
  // The ImmediateWriteStatus class should be singleton, so it's cleaner to not derive it from
  // AbstractFuture.
  @SuppressWarnings("ShouldNotSubclass")
  public sealed interface WriteStatus extends ListenableFuture<Boolean>
      permits ImmediateWriteStatus,
          ImmediateFailedWriteStatus,
          SettableWriteStatus,
          SparseAggregateWriteStatus,
          AggregateWriteStatus {}

  /** Returns the stateless, immediately successful write status. */
  public static WriteStatus immediateWriteStatus() {
    return ImmediateWriteStatus.INSTANCE;
  }

  /** Creates an immediately failed write status. */
  public static WriteStatus immediateFailedWriteStatus(Throwable cause) {
    return new ImmediateFailedWriteStatus(cause);
  }

  /**
   * Combines {@code writeStatuses} into a single future using <i>sparse</i> aggregation.
   *
   * <p>NB: This is not a general purpose aggregation and must only be used under certain
   * conditions. See {@link SparseAggregateWriteStatus} for details.
   */
  public static WriteStatus sparselyAggregateWriteStatuses(Collection<WriteStatus> writeStatuses) {
    if (writeStatuses.isEmpty()) {
      return immediateWriteStatus();
    }
    if (writeStatuses.size() == 1) {
      return writeStatuses.iterator().next();
    }
    return SparseAggregateWriteStatus.create(writeStatuses);
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
   * A reference-count based aggregator for {@link WriteStatus}es.
   *
   * <p><b>Sparsity:</b> when {@link addToAggregator} is called, only the first invocation creates a
   * callback and the rest are ignored. This is appropriate when all {@link WriteStatus}es are
   * ultimately aggregated into a single top-level future, e.g., the {@link
   * com.google.devtools.build.lib.skyframe.serialization.analysis.FrontierSerializer}.
   *
   * <p>When a {@link com.google.common.util.concurrent.SettableFuture} with sparse edges is
   * desired, this class may be used by calling the methods {@link #notifyWriteSucceeded} and {@link
   * #notifyWriteFailed} appropriately. Since this class derives from {@link QuiescingFuture},
   * there's a pre-increment, so calling one of those two methods once is sufficient for setting the
   * value.
   */
  public static final class SparseAggregateWriteStatus extends QuiescingFuture<Boolean>
      implements WriteStatus, FutureCallback<Boolean> {
    private volatile SparseAggregateWriteStatus listeningAggregate = null;
    private volatile boolean wasNovel = false;

    public SparseAggregateWriteStatus() {
      super(directExecutor());
    }

    /** Creates an aggregate that depends on all the statuses in {@code writeStatuses}. */
    private static SparseAggregateWriteStatus create(
        Iterable<? extends ListenableFuture<Boolean>> writeStatuses) {
      return new SparseAggregateWriteStatusBuilder().addAll(writeStatuses).build();
    }

    /**
     * Signals the successful completion of an aggregate component.
     *
     * <p>Only clients using the aggregate as a settable future call this.
     */
    public void notifyWriteSucceeded() {
      notifyWriteSucceeded(/* novel= */ true);
    }

    /**
     * Signals the successful completion of an aggregate component with novelty information.
     *
     * <p>Only clients using the aggregate as a settable future call this.
     *
     * @param novel true if new bytes were actually written; false if they already existed in the
     *     backend.
     */
    public void notifyWriteSucceeded(boolean novel) {
      if (novel) {
        // "OR" semantics: if any component was novel, the aggregate is novel.
        var unused = WAS_NOVEL_HANDLE.compareAndSet(this, false, true);
      }
      decrement();
    }

    /**
     * Signals the failure of an aggregate component.
     *
     * <p>Only clients using the aggregate as a settable future (or {@link
     * SparseAggregateWriteStatusBuilder}) call this.
     */
    public void notifyWriteFailed(Throwable t) {
      if (t instanceof CancellationException) {
        cancel(/* mayInterruptIfRunning= */ false); // nothing running
        return;
      }
      notifyException(t);
    }

    @Override
    protected Boolean getValue() {
      return wasNovel;
    }

    /**
     * Prepares for the addition of a new write operation by incrementing the internal reference
     * count.
     *
     * <p>By incrementing *before* the write is actually added, we ensure that the reference count
     * accurately reflects the number of pending writes, even if some writes complete immediately.
     */
    private void prepareForAddingWrite() {
      increment();
    }

    private void addToAggregator(SparseAggregateWriteStatus aggregate) {
      // The CAS here accepts the first listener, and ignores any additional ones.
      if (LISTENING_AGGREGATOR_HANDLE.compareAndSet(this, null, aggregate)) {
        aggregate.prepareForAddingWrite();
        addListener(
            () -> {
              try {
                boolean result = Futures.getDone(this);
                listeningAggregate.notifyWriteSucceeded(result);
              } catch (ExecutionException e) {
                listeningAggregate.notifyWriteFailed(e);
              } catch (CancellationException e) {
                listeningAggregate.cancel(/* mayInterruptIfRunning= */ false); // nothing running
              }
            },
            directExecutor());
      }
    }

    private void clearPreincrement() {
      decrement();
    }

    /**
     * Implementation of {@link FutureCallback<Boolean>}.
     *
     * @deprecated only for use by {@link #create} callback processing.
     */
    @Deprecated
    @Override
    @DoNotCall
    @SuppressWarnings("InlineMeSuggester")
    public void onSuccess(Boolean novel) {
      notifyWriteSucceeded(novel);
    }

    /**
     * Implementation of {@link FutureCallback<Boolean>}.
     *
     * @deprecated only for use by {@link #create} callback processing.
     */
    @Deprecated
    @Override
    @DoNotCall
    @SuppressWarnings("InlineMeSuggester")
    public void onFailure(Throwable t) {
      if (t instanceof CancellationException) {
        cancel(/* mayInterruptIfRunning= */ false);
        return;
      }
      notifyWriteFailed(t);
    }

    private static final VarHandle LISTENING_AGGREGATOR_HANDLE;
    private static final VarHandle WAS_NOVEL_HANDLE;

    static {
      try {
        LISTENING_AGGREGATOR_HANDLE =
            MethodHandles.lookup()
                .findVarHandle(
                    SparseAggregateWriteStatus.class,
                    "listeningAggregate",
                    SparseAggregateWriteStatus.class);
        WAS_NOVEL_HANDLE =
            MethodHandles.lookup()
                .findVarHandle(SparseAggregateWriteStatus.class, "wasNovel", boolean.class);
      } catch (ReflectiveOperationException e) {
        throw new ExceptionInInitializerError(e);
      }
    }
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
   * <p>In contrast to {@link SparseAggregateWriteStatus} preserves all callback edges.
   */
  private static final class AggregateWriteStatus extends QuiescingFuture<Boolean>
      implements WriteStatus, FutureCallback<Boolean> {
    private volatile boolean wasNovel = false;

    private static AggregateWriteStatus create(Iterable<WriteStatus> writeStatuses) {
      return new AggregateWriteStatusBuilder().addAll(writeStatuses).build();
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

  /** Interface for building aggregated {@link WriteStatus}es. */
  public interface WriteStatusBuilder {
    /** Adds a status to the aggregate. */
    @CanIgnoreReturnValue
    WriteStatusBuilder add(ListenableFuture<Boolean> status);

    /** Adds all statuses to the aggregate. */
    @CanIgnoreReturnValue
    WriteStatusBuilder addAll(Iterable<? extends ListenableFuture<Boolean>> statuses);

    /**
     * Builds and returns the aggregated {@link WriteStatus}.
     *
     * <p>Should only be called once.
     */
    WriteStatus build();
  }

  /**
   * Builder for {@link AggregateWriteStatus}.
   *
   * <p>This builder is thread safe, but {@link #build} should only be called once.
   */
  static final class AggregateWriteStatusBuilder implements WriteStatusBuilder {
    private final AggregateWriteStatus aggregate = new AggregateWriteStatus();
    private final AtomicBoolean preincrementCleared = new AtomicBoolean(false);

    @CanIgnoreReturnValue
    @Override
    public AggregateWriteStatusBuilder add(ListenableFuture<Boolean> status) {
      aggregate.add(status);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public AggregateWriteStatusBuilder addAll(
        Iterable<? extends ListenableFuture<Boolean>> statuses) {
      for (ListenableFuture<Boolean> status : statuses) {
        aggregate.add(status);
      }
      return this;
    }

    /** Should only be called once. */
    @Override
    public AggregateWriteStatus build() {
      checkState(!preincrementCleared.getAndSet(true), "build must only be called once");
      aggregate.clearPreincrement();
      return aggregate;
    }
  }

  /**
   * Builder for {@link SparseAggregateWriteStatus}.
   *
   * <p>This builder is thread safe, but {@link #build} should only be called once.
   */
  public static final class SparseAggregateWriteStatusBuilder implements WriteStatusBuilder {
    private final SparseAggregateWriteStatus aggregate = new SparseAggregateWriteStatus();
    private final AtomicBoolean preincrementCleared = new AtomicBoolean(false);

    @CanIgnoreReturnValue
    @Override
    public SparseAggregateWriteStatusBuilder add(ListenableFuture<Boolean> status) {
      if (status.isDone()) {
        try {
          if (Futures.getDone(status)) {
            // notifyWriteSucceeded(true) updates the novelty bit and also decrements.
            // Increments the reference count to stay consistent.
            aggregate.prepareForAddingWrite();
            aggregate.notifyWriteSucceeded(true);
          }
        } catch (ExecutionException | CancellationException e) {
          // InternalFutureFailureAccess might be more efficient, but failures should be rare.
          //
          // Increments the reference count for consistency.
          aggregate.prepareForAddingWrite();
          aggregate.notifyWriteFailed(e);
        }
        return this;
      }

      switch (status) {
        case SparseAggregateWriteStatus sparse:
          // The addToAggregator logic ensures that each SparseAggregateWriteStatus has at most one
          // SparseAggregateWriteStatus parent.
          sparse.addToAggregator(aggregate);
          break;
        default:
          aggregate.prepareForAddingWrite();
          Futures.addCallback(status, (FutureCallback<Boolean>) aggregate, directExecutor());
          break;
      }
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public SparseAggregateWriteStatusBuilder addAll(
        Iterable<? extends ListenableFuture<Boolean>> statuses) {
      for (ListenableFuture<Boolean> status : statuses) {
        add(status);
      }
      return this;
    }

    @Override
    public SparseAggregateWriteStatus build() {
      checkState(!preincrementCleared.getAndSet(true), "build must only be called once");
      aggregate.clearPreincrement();
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
  }

  private static final class ImmediateWriteStatus implements WriteStatus {
    private static final ImmediateWriteStatus INSTANCE = new ImmediateWriteStatus();

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
      return true;
    }

    @Override
    public Boolean get(long timeout, TimeUnit unit) {
      return true;
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
