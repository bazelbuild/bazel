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
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;
import javax.annotation.concurrent.GuardedBy;

/**
 * A lock-based, asynchronous request batcher designed for eager execution.
 *
 * <p>This class batches unary requests and executes them together. It eagerly dispatches batches as
 * soon as they reach {@code maxBatchSize} or when the number of active concurrent requests is below
 * {@code targetConcurrentRequests}.
 *
 * <p>Submissions do not block waiting for batch execution or queue capacity. While the class uses
 * lightweight synchronization (synchronized) to ensure thread-safe queue access, the lock is held
 * only for brief in-memory updates, ensuring that calling threads never block on I/O or
 * backpressure.
 *
 * <p><b>Locking Efficiency:</b> While the class uses a lock to ensure thread-safe queue operations,
 * the lock hold time is extremely short. Under high load, the relatively more expensive {@link
 * ThreadLocal} pool lookup is amortized over the batch size (occurring only once per {@code
 * maxBatchSize} submissions), keeping the average lock hold time close to a few nanoseconds.
 *
 * <p><b>Virtual Thread Performance:</b> While this class will function correctly when called from
 * virtual threads, performance will likely be very poor. It relies on {@link ThreadLocal} via
 * {@link QueuePool} for optimization, which does not behave predictably or efficiently with
 * short-lived virtual threads. Additionally, it uses monitor-based locks (synchronized) which can
 * cause pinning of carrier threads.
 */
public final class EagerRequestBatcher<RequestT, ResponseT> {

  private final Object lock = new Object();

  @GuardedBy("lock")
  private List<Operation<RequestT, ResponseT>> queue;

  @GuardedBy("lock")
  private int inFlightCount = 0;

  private final int maxBatchSize;
  private final int targetConcurrentRequests;

  /** Executor used for batch completion work, which may include sending batches. */
  private final Executor executor;

  private final BatchExecutionStrategy<RequestT, ResponseT> batchExecutionStrategy;
  private final QueuePool<RequestT, ResponseT> pool;

  /** Creates a batcher with standard Multiplexer. */
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

  /** Creates a batcher with CallbackMultiplexer. */
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

  /** Creates a batcher with FutureMultiplexer. */
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
      QueuePool<RequestT, ResponseT> pool,
      int targetConcurrentRequests,
      Executor executor) {
    this.batchExecutionStrategy = batchExecutionStrategy;
    this.pool = pool;
    this.maxBatchSize = pool.getMaxBatchSize();
    checkArgument(targetConcurrentRequests >= 1, "targetConcurrentRequests must be >= 1");
    this.targetConcurrentRequests = targetConcurrentRequests;
    this.executor = executor;
    this.queue = new ArrayList<>(maxBatchSize);
  }

  public ListenableFuture<ResponseT> submit(RequestT request) {
    Operation<RequestT, ResponseT> operation = new Operation<>(request);
    List<Operation<RequestT, ResponseT>> batch = null;

    synchronized (lock) {
      queue.add(operation);

      // Rule 1 (Eager): Execute immediately if queue reaches maxBatchSize.
      // Rule 2 (Target Concurrency): Execute immediately if in-flight count is below target.
      if (queue.size() >= maxBatchSize || inFlightCount < targetConcurrentRequests) {
        batch = swapQueue();
        inFlightCount++;
      }
    }

    if (batch != null) {
      execute(copyAndRecycle(batch));
    }

    return operation;
  }

  private void onBatchComplete() {
    List<Operation<RequestT, ResponseT>> batch = null;
    synchronized (lock) {
      if (!queue.isEmpty() && inFlightCount <= targetConcurrentRequests) {
        batch = swapQueue();
        // A batch has just completed, but the queue contents will be sent immediately so
        // inFlightCount does not change.
      } else {
        inFlightCount--;
      }
    }

    if (batch != null) {
      execute(copyAndRecycle(batch));
    }
  }

  /**
   * Swaps the queue with a clean one from the pool.
   *
   * <p>IMPORTANT: after this swap, a batch must be recycled into {@link #pool} before any other
   * calls to {@link QueuePool#getQueue()} from this thread.
   *
   * @return the queue at the moment this method was called
   */
  @GuardedBy("lock")
  private List<Operation<RequestT, ResponseT>> swapQueue() {
    List<Operation<RequestT, ResponseT>> batch = queue;
    queue = pool.getQueue();
    return batch;
  }

  private ImmutableList<Operation<RequestT, ResponseT>> copyAndRecycle(
      List<Operation<RequestT, ResponseT>> batch) {
    ImmutableList<Operation<RequestT, ResponseT>> copy = ImmutableList.copyOf(batch);
    pool.recycleQueue(batch);
    return copy;
  }

  private void execute(ImmutableList<Operation<RequestT, ResponseT>> batch) {
    ListenableFuture<?> batchFuture;
    try {
      batchFuture =
          batchExecutionStrategy.executeBatch(Lists.transform(batch, Operation::request), batch);
    } catch (Throwable t) {
      handleSynchronousException(batch, t);
      return;
    }

    batchFuture.addListener(this::onBatchComplete, executor);
  }

  private void handleSynchronousException(
      ImmutableList<Operation<RequestT, ResponseT>> operations, Throwable t) {
    synchronized (lock) {
      inFlightCount--;
    }
    for (Operation<RequestT, ResponseT> operation : operations) {
      operation.acceptFailure(t);
    }
  }

  // Package-private for testing
  @VisibleForTesting
  int getInFlightCount() {
    synchronized (lock) {
      return inFlightCount;
    }
  }

  @VisibleForTesting
  int getQueueSize() {
    synchronized (lock) {
      return queue.size();
    }
  }
}
