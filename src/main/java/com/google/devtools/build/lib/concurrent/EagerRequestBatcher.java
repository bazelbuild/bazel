// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.concurrent.RequestBatching.BatchExecutionStrategy;
import com.google.devtools.build.lib.concurrent.RequestBatching.CallbackMultiplexer;
import com.google.devtools.build.lib.concurrent.RequestBatching.FutureMultiplexer;
import com.google.devtools.build.lib.concurrent.RequestBatching.Multiplexer;
import com.google.devtools.build.lib.concurrent.RequestBatching.Operation;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.concurrent.Executor;
import javax.annotation.Nullable;

/**
 * A lock-free, asynchronous request batcher designed for eager execution.
 *
 * <p>This class batches unary requests and executes them together. It eagerly dispatches batches as
 * soon as they reach {@code maxBatchSize} or when the number of active concurrent requests is below
 * {@code targetConcurrentRequests}.
 *
 * <p>Submissions do not block waiting for batch execution or queue capacity. It uses lock-free
 * constructs (via {@link VarHandle}) to ensure thread-safe queue access and is safe to use with
 * virtual threads.
 */
public final class EagerRequestBatcher<RequestT, ResponseT> {

  private static final long PLUS_ONE_SIZE_PLUS_ONE_REFS = 0x0000_0001_0000_0001L;
  private static final long MINUS_ONE_REF = -1L;
  private static final long PLUS_ONE_SIZE_MINUS_ONE_REFS = 0x0000_0000_FFFF_FFFFL;

  private static final VarHandle STATE;
  private static final VarHandle SIZE_AND_REFS;

  private static int refs(long sizeAndRefs) {
    return (int) sizeAndRefs;
  }

  private static int size(long sizeAndRefs) {
    return (int) (sizeAndRefs >>> 32);
  }

  /**
   * Encapsulates the dynamic state.
   *
   * <p><b>CRITICAL CONCURRENCY INVARIANT (EFFECTIVE IMMUTABILITY):</b> The fields of this nested
   * class are intentionally mutable to enable in-place updates and buffer reuse during concurrent
   * CAS retry loops (eliminating GC allocation churn).
   *
   * <p>However, to prevent race conditions and state corruption, this object MUST be treated as
   * strictly <b>immutable post-publication</b>. Once a State reference is successfully installed
   * into the global 'state' volatile pointer, it is read-only.
   *
   * <p>NEVER write to or mutate the fields of the active state pointed to by
   * EagerRequestBatcher.state.
   */
  private static final class State {
    private int inFlightRequests;
    private Buffer buffer;

    private State(int inFlightRequests, Buffer buffer) {
      this.inFlightRequests = inFlightRequests;
      this.buffer = buffer;
    }
  }

  private static final class Buffer {
    private volatile long sizeAndRefs = 1L;
    private final Object[] elements;

    private Buffer(int maxBatchSize) {
      this.elements = new Object[maxBatchSize];
    }
  }

  private final int maxBatchSize;
  private final int targetConcurrentRequests;
  private final Executor executor;
  private final BatchExecutionStrategy<RequestT, ResponseT> batchExecutionStrategy;

  /**
   * Encapsulation of dynamic state of the batcher.
   *
   * <p>Fields of {@code state} must not be modified. Mutations are performed by swapping in new
   * instances using CAS.
   */
  private volatile State state;

  /**
   * Creates a batcher with standard Multiplexer.
   *
   * @param executor the executor used for distributing responses and triggering completion.
   *     <b>CRITICAL CONCURRENCY CONTRACT:</b> The injected executor MUST be unbounded (e.g., {@link
   *     ForkJoinPool#commonPool()} or an executor configured with an unbounded task queue). Using a
   *     bounded-queue executor will result in silent, permanent capacity slot leaks and a complete
   *     lockup of eager sending under task saturation, as asynchronous {@link
   *     RejectedExecutionException}s are silently swallowed inside the underlying {@link
   *     ListenableFuture#addListener} pipeline.
   */
  public static <RequestT, ResponseT> EagerRequestBatcher<RequestT, ResponseT> create(
      Multiplexer<RequestT, ResponseT> multiplexer,
      Executor responseDistributionExecutor,
      QueuePool<RequestT, ResponseT> pool,
      int targetConcurrentRequests,
      Executor executor) {
    return new EagerRequestBatcher<>(
        RequestBatching.createBatchExecutionStrategy(multiplexer, responseDistributionExecutor),
        pool,
        targetConcurrentRequests,
        executor);
  }

  /**
   * Creates a batcher with CallbackMultiplexer.
   *
   * @param executor the executor used for distributing responses and triggering completion.
   *     <b>CRITICAL CONCURRENCY CONTRACT:</b> The injected executor MUST be unbounded (e.g., {@link
   *     ForkJoinPool#commonPool()} or an executor configured with an unbounded task queue). Using a
   *     bounded-queue executor will result in silent, permanent capacity slot leaks and a complete
   *     lockup of eager sending under task saturation, as asynchronous {@link
   *     RejectedExecutionException}s are silently swallowed inside the underlying {@link
   *     ListenableFuture#addListener} pipeline.
   */
  public static <RequestT, ResponseT>
      EagerRequestBatcher<RequestT, ResponseT> createWithCallbackMultiplexer(
          CallbackMultiplexer<RequestT, ResponseT> multiplexer,
          QueuePool<RequestT, ResponseT> pool,
          int targetConcurrentRequests,
          Executor executor) {
    return new EagerRequestBatcher<>(
        RequestBatching.createCallbackBatchExecutionStrategy(multiplexer),
        pool,
        targetConcurrentRequests,
        executor);
  }

  /**
   * Creates a batcher with FutureMultiplexer.
   *
   * @param executor the executor used for distributing responses and triggering completion.
   *     <b>CRITICAL CONCURRENCY CONTRACT:</b> The injected executor MUST be unbounded (e.g., {@link
   *     ForkJoinPool#commonPool()} or an executor configured with an unbounded task queue). Using a
   *     bounded-queue executor will result in silent, permanent capacity slot leaks and a complete
   *     lockup of eager sending under task saturation, as asynchronous {@link
   *     RejectedExecutionException}s are silently swallowed inside the underlying {@link
   *     ListenableFuture#addListener} pipeline.
   */
  public static <RequestT, ResponseT>
      EagerRequestBatcher<RequestT, ResponseT> createWithFutureMultiplexer(
          FutureMultiplexer<RequestT, ResponseT> multiplexer,
          QueuePool<RequestT, ResponseT> pool,
          int targetConcurrentRequests,
          Executor executor) {
    return new EagerRequestBatcher<>(
        RequestBatching.createFutureBatchExecutionStrategy(multiplexer),
        pool,
        targetConcurrentRequests,
        executor);
  }

  // Package-private constructor for testing and internal use
  @VisibleForTesting
  EagerRequestBatcher(
      BatchExecutionStrategy<RequestT, ResponseT> batchExecutionStrategy,
      // Kept for API compatibility. To be removed in a follow up change.
      QueuePool<RequestT, ResponseT> pool,
      int targetConcurrentRequests,
      Executor executor) {
    this.batchExecutionStrategy = batchExecutionStrategy;
    this.maxBatchSize = pool.getMaxBatchSize();
    checkArgument(targetConcurrentRequests >= 1, "targetConcurrentRequests must be >= 1");
    this.targetConcurrentRequests = targetConcurrentRequests;
    this.executor = executor;
    this.state = new State(0, new Buffer(maxBatchSize));
  }

  public ListenableFuture<ResponseT> submit(RequestT request) {
    var operation = new Operation<RequestT, ResponseT>(request);
    State newState = null;
    State snapshot = state;
    Buffer buffer = snapshot.buffer;

    while (true) {
      long sizeAndRefsSnapshot = buffer.sizeAndRefs;
      int size = size(sizeAndRefsSnapshot);
      int refs = refs(sizeAndRefsSnapshot);

      // 1. Check if buffer is already retired/dispatched
      if (refs <= 0 || size >= maxBatchSize) {
        snapshot = state;
        buffer = snapshot.buffer;
        continue;
      }

      // 2. Rule R2: Buffer is about to reach maxBatchSize. Detach and write last slot.
      if (size >= maxBatchSize - 1) {
        if (newState == null) {
          newState = new State(snapshot.inFlightRequests + 1, new Buffer(maxBatchSize));
        } else {
          newState.inFlightRequests = snapshot.inFlightRequests + 1;
        }

        if (!STATE.compareAndSet(this, snapshot, newState)) {
          snapshot = state;
          buffer = snapshot.buffer;
          continue;
        }

        buffer.elements[maxBatchSize - 1] = operation;

        long preDecrement = (long) SIZE_AND_REFS.getAndAdd(buffer, PLUS_ONE_SIZE_MINUS_ONE_REFS);
        if (refs(preDecrement) == 1) {
          execute(buffer);
        }
        return operation;
      }

      // 3. Normal Insert Path: Try to reserve a slot
      long targetValue = sizeAndRefsSnapshot + PLUS_ONE_SIZE_PLUS_ONE_REFS;
      if (SIZE_AND_REFS.compareAndSet(buffer, sizeAndRefsSnapshot, targetValue)) {
        buffer.elements[size] = operation;

        long preDecrement = (long) SIZE_AND_REFS.getAndAdd(buffer, MINUS_ONE_REF);
        if (refs(preDecrement) == 1) {
          execute(buffer);
        } else {
          sendIfPossible(newState);
        }
        return operation;
      }
    }
  }

  /**
   * Attempts to send the current batch if the number of in-flight requests is below {@code
   * targetConcurrentRequests} and it is non-empty.
   *
   * <p>If the conditions are observed to be not met, returns without sending.
   */
  private void sendIfPossible(@Nullable State preAllocatedState) {
    State newState = preAllocatedState;

    while (true) {
      State snapshot = state;

      if (snapshot.inFlightRequests >= targetConcurrentRequests) {
        return;
      }

      Buffer buffer = snapshot.buffer;
      long sizeAndRefsSnapshot = buffer.sizeAndRefs;

      if (size(sizeAndRefsSnapshot) == 0) {
        return;
      }

      if (newState == null) {
        newState = new State(snapshot.inFlightRequests + 1, new Buffer(maxBatchSize));
      } else {
        newState.inFlightRequests = snapshot.inFlightRequests + 1;
      }

      if (!STATE.compareAndSet(this, snapshot, newState)) {
        continue;
      }

      long preDecrement = (long) SIZE_AND_REFS.getAndAdd(buffer, MINUS_ONE_REF);
      if (refs(preDecrement) == 1) {
        execute(buffer);
      }
      return;
    }
  }

  private void onBatchDone() {
    State newState = null;

    while (true) {
      State snapshot = state;

      if (newState == null) {
        newState = new State(snapshot.inFlightRequests - 1, snapshot.buffer);
      } else {
        newState.inFlightRequests = snapshot.inFlightRequests - 1;
        newState.buffer = snapshot.buffer;
      }

      if (STATE.compareAndSet(this, snapshot, newState)) {
        break;
      }
    }

    sendIfPossible(null);
  }

  private void execute(Buffer buffer) {
    ImmutableList<Operation<RequestT, ResponseT>> batch = copyElements(buffer);

    ListenableFuture<?> batchFuture;
    try {
      batchFuture =
          batchExecutionStrategy.executeBatch(Lists.transform(batch, Operation::request), batch);
    } catch (Throwable t) {
      handleSynchronousException(batch, t);
      return;
    }

    batchFuture.addListener(this::onBatchDone, executor);
  }

  private static <RequestT, ResponseT> ImmutableList<Operation<RequestT, ResponseT>> copyElements(
      Buffer buffer) {
    int size = size(buffer.sizeAndRefs);
    ImmutableList.Builder<Operation<RequestT, ResponseT>> builder =
        ImmutableList.builderWithExpectedSize(size);
    for (int i = 0; i < size; i++) {
      @SuppressWarnings("unchecked") // Java doesn't permit parameterized arrays.
      Operation<RequestT, ResponseT> element = (Operation<RequestT, ResponseT>) buffer.elements[i];
      if (element == null) {
        throw new IllegalStateException(
            "Null element found in buffer at index " + i + " with size " + size);
      }
      builder.add(element);
    }
    return builder.build();
  }

  private void handleSynchronousException(
      ImmutableList<Operation<RequestT, ResponseT>> operations, Throwable t) {
    onBatchDone();
    for (Operation<RequestT, ResponseT> operation : operations) {
      operation.acceptFailure(t);
    }
  }

  @VisibleForTesting
  int getInFlightCount() {
    return state.inFlightRequests;
  }

  @VisibleForTesting
  int getQueueSize() {
    return size(state.buffer.sizeAndRefs);
  }

  @VisibleForTesting
  static VarHandle getBufferSizeAndRefsVarHandleForTesting() {
    return SIZE_AND_REFS;
  }

  @VisibleForTesting
  Object getActiveBufferForTesting() {
    return state.buffer;
  }

  static {
    try {
      MethodHandles.Lookup l = MethodHandles.lookup();
      STATE = l.findVarHandle(EagerRequestBatcher.class, "state", State.class);
      SIZE_AND_REFS = l.findVarHandle(Buffer.class, "sizeAndRefs", long.class);
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }
}
