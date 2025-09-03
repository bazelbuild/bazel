// Copyright 2024 The Bazel Authors. All rights reserved.
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
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.util.concurrent.Futures.addCallback;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.concurrent.PaddedAddresses.createPaddedBaseAddress;
import static com.google.devtools.build.lib.concurrent.PaddedAddresses.getAlignedAddress;
import static java.lang.Math.min;
import static java.util.concurrent.Executors.newFixedThreadPool;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.AbstractFuture;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.unsafe.UnsafeProvider;

import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.lang.ref.Cleaner;
import java.util.List;
import java.util.concurrent.Executor;
import javax.annotation.Nullable;
import sun.misc.Unsafe;

/**
 * Provides a unary request-response interface but implements batching.
 *
 * <p>Clients should provide a {@link Multiplexer} implementation that performs the actual batched
 * operations.
 *
 * <p>This class is thread-safe.
 *
 * <p>Non-final for mockability.
 */
@SuppressWarnings("SunApi") // TODO: b/359688989 - clean this up
public class RequestBatcher<RequestT, ResponseT> {
  /* This class employs concurrent workers that perform the following cycle:
   *
   *   1. Collect all available request-response pairs from the queue up to `maxBatchSize`.
   *   2. Execute the collected pairs as a batch.
   *
   * We guarantee that every submitted request is handled. The following traces all possible paths a
   * request-response pair can take through the batcher to demonstrate this guarantee.
   *
   * Possible Paths:
   *
   *   1. The pair is present in some `submit` call.
   *   2. The pair is enqueued, but not yet reflected in the request-responses count.
   *   3. The pair is enqueued, and request-responses count has been incremented.
   *
   * Step 1: Initial part of `submit`
   *
   * A. We check the active-workers count. If it's less than `maxConcurrentRequests`, a new worker
   *    is started and the pair is directly assigned to it.
   *
   * B. Otherwise, we enqueue the pair. When the queue is full, we sleep and try again until
   *    enqueuing succeeds. After enqueuing, we proceed to Step 2.
   *
   * Case A bypasses Step 2, and the pair is immediately assigned a worker.
   *
   * Step 2: Request-response Enqueued
   *
   * Step 2 is not atomic with Step 1, so the counters might have changed. We re-check
   * active-workers count.
   *
   * A. If it's already at `maxConcurrentRequests`, we attempt to increment request-responses count
   *    atomically, ensuring active-workers count remains unchanged during the increment. Success
   *    leads to Step 3.
   *
   * B. If active-workers count is below the target (due to concurrent activity), we start a new
   *    worker like in Step 1, and dequeue an arbitrary element to assign to it. This maintains
   *    consistency between queue size and request-responses count. The new worker guarantees
   *    processing of all enqueued request-responses (including the one we just added), even if that
   *    specific request ends up handled by a different worker.
   *
   * Step 3: Request-response Enqueued and request-responses count Incremented
   *
   * The atomic request-responses count increment only happens in Step 2 if active-workers count is
   * already at the target. Workers only stop if request-responses count is 0. Since
   * `maxConcurrentRequests` > 0, there's always at least one active worker to handle the
   * request-response.
   */

  /**
   * A common cleaner shared by all instances.
   *
   * <p>Used to free memory allocated by {@link PaddedAddresses}.
   */
  private static final Cleaner cleaner = Cleaner.create();

  private static final long QUEUE_FULL_SLEEP_MS = 100;

  /**
   * Executor dedicated to draining the queue, specifically the {@link
   * #continueToNextBatchOrBecomeIdle} method.
   *
   * <p><b>Purpose of Isolation:</b> This executor is isolated to prevent potential deadlocks. The
   * {@link #submit} method can block if the task queue is full. If all threads in the client's
   * executor become blocked waiting to submit tasks, only {@link #continueToNextBatchOrBecomeIdle}
   * can free up space in the queue. Scheduling this continuation logic on the same, potentially
   * blocked, client executor would lead to a deadlock.
   *
   * <p><b>Deadlock Avoidance:</b> As long as {@link #continueToNextBatchOrBecomeIdle} does not
   * contain blocking operations (which is true in the current implementation), using a separate
   * executor is sufficient to prevent this specific deadlock scenario.
   */
  private final Executor queueDrainingExecutor;

  private final BatchExecutionStrategy<RequestT, ResponseT> batchExecutionStrategy;

  /**
   * Reads this many at a time when constructing a batch.
   *
   * <p>Note that since {@link #populateBatch} always begins with 1 pair, the resulting batch size
   * is one more than this.
   */
  private final int maxBatchSize;

  /** Number of active workers to target. */
  private final int maxConcurrentRequests;

  /**
   * Address of an integer containing two counters.
   *
   * <p>Having two counters in the same integer enables simultaneous, atomic updates of both values.
   *
   * <ul>
   *   <li><b>request-responses count</b>: the lower 20-bits (occupying the bits of {@link
   *       #REQUEST_COUNT_MASK}) contain a lower bound of request-responses in {@link #queue}. This
   *       is incremented after successful enqueuing and decremented before dequeuing. This counter
   *       value is never more than the size of the queue so it can be used to guarantee that the
   *       number of calls to {@link ConcurrentFifo#take} do not exceed the number of successful
   *       {@link ConcurrentFifo#tryAppend} calls.
   *   <li><b>active-workers count</b>: the upper 12-bits (starting from {@link
   *       #ACTIVE_WORKERS_COUNT_BIT_OFFSET}) contain the number of active workers.
   * </ul>
   */
  private final long countersAddress;

  private final ConcurrentFifo<Operation<RequestT, ResponseT>> queue;

  /** Injectable batching logic. */

  /** Batching strategy where a single batch request returns a single batch future response. */
  public interface Multiplexer<RequestT, ResponseT> {
    /**
     * Evaluates {@code requests} as a batch.
     *
     * @return a future containing a list of responses, positionally aligned with {@code requests}
     */
    ListenableFuture<List<ResponseT>> execute(List<RequestT> requests);
  }

  /**
   * A callback for a single request within a batch, which must be completed exactly once.
   *
   * <p>Used with {@link CallbackMultiplexer}.
   */
  public interface ResponseSink<ResponseT> {
    /**
     * Fulfills the corresponding request with a successful response.
     *
     * @param response the result of the operation. A {@code null} value is permitted and will be
     *     forwarded to the original caller as a successful result.
     */
    void acceptResponse(@Nullable ResponseT response);

    /**
     * Fails the corresponding request with the given {@link Throwable}.
     *
     * <p>A sink should only be completed once. Subsequent calls to this method after the sink has
     * already been completed will be ignored.
     */
    void acceptFailure(Throwable t);
  }

  /**
   * A batching strategy where the implementation provides concrete response values asynchronously
   * via callbacks.
   */
  public interface CallbackMultiplexer<RequestT, ResponseT> {
    /**
     * Executes the batch of {@code requests}, pushing results directly to the corresponding {@link
     * ResponseSink} instances in the {@code sinks} list.
     *
     * <p>The supplied {@code sinks} list is co-indexed with the {@code requests} list. The
     * implementation of this method <strong>must</strong> ensure that for each request, the
     * corresponding sink is completed exactly once by calling either {@link
     * ResponseSink#acceptResponse} on success or {@link ResponseSink#acceptFailure} on failure.
     *
     * <p>The {@link RequestBatcher} internally monitors the completion of all sink operations for
     * the batch.
     *
     * @return A non-null {@link Runnable} that the {@code RequestBatcher} will execute on behalf of
     *     the client. The {@code RequestBatcher} guarantees it will run this callback after all
     *     sinks for this specific batch have been completed, but <strong>before</strong> this
     *     batch's concurrency slot is released. This provides a reliable mechanism for performing
     *     batch-specific resource cleanup. For instance, if recycling identifiers used in the
     *     requests, this guarantee ensures the identifiers are made available before a subsequent
     *     batch could possibly use them. The callback should be lightweight.
     */
    Runnable execute(
        List<RequestT> requests, ImmutableList<? extends ResponseSink<ResponseT>> sinks);
  }

  /**
   * Accepts a future response value.
   *
   * <p>Used with {@link PerResponseMultiplexer}.
   */
  public interface FutureResponseSink<ResponseT> {
    void acceptFutureResponse(ListenableFuture<ResponseT> futureResponse);
  }

  /** Batching strategy when a single batch request returns a response per future request. */
  public interface PerResponseMultiplexer<RequestT, ResponseT> {
    /** Executes {@code requests} in a batch and populates corresponding {@code responses}. */
    void execute(
        List<RequestT> requests, ImmutableList<? extends FutureResponseSink<ResponseT>> responses);
  }

  public static <RequestT, ResponseT>
      BatchExecutionStrategy<RequestT, ResponseT> createBatchExecutionStrategy(
          Multiplexer<RequestT, ResponseT> multiplexer, Executor responseDistributionExecutor) {
    return new MultiplexerAdapter<>(multiplexer, responseDistributionExecutor);
  }

  public static <RequestT, ResponseT>
      BatchExecutionStrategy<RequestT, ResponseT> createCallbackBatchExecutionStrategy(
          CallbackMultiplexer<RequestT, ResponseT> multiplexer) {
    return new CallbackMultiplexerAdapter<>(multiplexer);
  }

  public static <RequestT, ResponseT>
      BatchExecutionStrategy<RequestT, ResponseT> createPerResponseBatchExecutionStrategy(
          PerResponseMultiplexer<RequestT, ResponseT> multiplexer) {
    return new PerResponseMultiplexerAdapter<>(multiplexer);
  }

  private interface BatchExecutionStrategy<RequestT, ResponseT> {
    ListenableFuture<?> executeBatch(
        List<RequestT> requests, ImmutableList<Operation<RequestT, ResponseT>> operations);
  }

  public static <RequestT, ResponseT> RequestBatcher<RequestT, ResponseT> create(
      BatchExecutionStrategy<RequestT, ResponseT> batchExecutionStrategy,
      int maxBatchSize,
      int maxConcurrentRequests) {
    long baseAddress = createPaddedBaseAddress(4);
    long countersAddress = getAlignedAddress(baseAddress, /* offset= */ 0);

    var queue =
        new ConcurrentFifo<Operation<RequestT, ResponseT>>(
            Operation.class,
            /* sizeAddress= */ getAlignedAddress(baseAddress, /* offset= */ 1),
            /* appendIndexAddress= */ getAlignedAddress(baseAddress, /* offset= */ 2),
            /* takeIndexAddress= */ getAlignedAddress(baseAddress, /* offset= */ 3));

    var batcher =
        new RequestBatcher<RequestT, ResponseT>(
            // `maxConcurrentRequests` is the maximum level of invocation concurrency possible for
            // the `queueDrainingExecutor`. It is possible for this to overrun, but the work is
            // relatively lightweight and the batch round trip latency is expected to dominate.
            /* queueDrainingExecutor= */ newFixedThreadPool(maxConcurrentRequests),
            batchExecutionStrategy,
            maxBatchSize,
            maxConcurrentRequests,
            countersAddress,
            queue);

    cleaner.register(batcher, new AddressFreer(baseAddress));

    return batcher;
  }

  /**
   * Low-level constructor.
   *
   * <p>Caller owns memory addresses used by {@code queue} and cleanup of memory at {@code
   * countersAddress}.
   */
  // TODO: b/386384684 - remove Unsafe usage
  @VisibleForTesting
  RequestBatcher(
      Executor queueDrainingExecutor,
      BatchExecutionStrategy<RequestT, ResponseT> batchExecutionStrategy,
      int maxBatchSize,
      int maxConcurrentRequests,
      long countersAddress,
      ConcurrentFifo<Operation<RequestT, ResponseT>> queue) {
    checkArgument(maxConcurrentRequests > 0, "maxConcurrentRequests=%s < 1", maxConcurrentRequests);
    checkArgument(
        maxConcurrentRequests <= ACTIVE_WORKERS_COUNT_MAX,
        "maxConcurrentRequests=%s > %s",
        maxConcurrentRequests,
        ACTIVE_WORKERS_COUNT_MAX);
    checkArgument(maxBatchSize > 0);
    this.queueDrainingExecutor = queueDrainingExecutor;
    this.batchExecutionStrategy = batchExecutionStrategy;
    this.maxBatchSize = maxBatchSize;
    this.maxConcurrentRequests = maxConcurrentRequests;
    this.countersAddress = countersAddress;
    this.queue = queue;

    // Initializes memory at countersAddress.
    UNSAFE.putInt(null, countersAddress, 0);
  }

  /**
   * Submits a request, subject to batching.
   *
   * <p>This method <em>blocks</em> when the queue is full.
   *
   * <p>Callers should consider processing the response on a different executor if processing is
   * expensive to avoid delaying work pending other responses in the batch.
   */
  // TODO: b/386384684 - remove Unsafe usage
  public ListenableFuture<ResponseT> submit(RequestT request) {
    var requestResponse = new Operation<RequestT, ResponseT>(request);

    // Tries to start a worker as long as the active worker count is less than
    // `maxConcurrentRequests`.
    while (true) {
      int snapshot = UNSAFE.getIntVolatile(null, countersAddress);
      int activeWorkers = snapshot >>> ACTIVE_WORKERS_COUNT_BIT_OFFSET;
      if (activeWorkers >= maxConcurrentRequests) {
        break;
      }
      if (UNSAFE.compareAndSwapInt(null, countersAddress, snapshot, snapshot + ONE_ACTIVE_WORKER)) {
        // An active worker was reserved. Starts the worker by executing a batch.
        executeBatch(requestResponse);
        return requestResponse;
      }
    }

    while (!queue.tryAppend(requestResponse)) {
      // As of 09/11/2024, this class is only used for remote cache interactions (see
      // b/358347099#comment18). Here, the queue filling up is primarily caused by insufficient
      // network bandwidth. Experiments show that sleeping here improves overall system throughput,
      // even more than increasing the buffer size.
      try {
        Thread.sleep(QUEUE_FULL_SLEEP_MS);
      } catch (InterruptedException e) {
        return immediateFailedFuture(e);
      }
    }
    // Enqueuing succeeded.

    while (true) {
      int snapshot = UNSAFE.getIntVolatile(null, countersAddress); // pessimistic read
      int activeWorkers = snapshot >>> ACTIVE_WORKERS_COUNT_BIT_OFFSET;
      if (activeWorkers >= maxConcurrentRequests) {
        // Increments the request-responses count.
        if (UNSAFE.compareAndSwapInt(null, countersAddress, snapshot, snapshot + ONE_REQUEST)) {
          // This must not be reached if `activeWorkers` is 0. Guaranteed by the enclosing check.
          return requestResponse;
        }
      } else {
        // This is a less common case where the task was enqueued, but the number of active workers
        // immediately dipped below `targetWorkersCount`. Starts a worker.
        if (UNSAFE.compareAndSwapInt(
            null, countersAddress, snapshot, snapshot + ONE_ACTIVE_WORKER)) {
          // Usually, decrementing the request-responses count must precede taking from the queue.
          // Here, a request-response was just enqueued and the count has not yet been incremented.
          executeBatch(queue.take());
          return requestResponse;
        }
      }
    }
  }

  // TODO: b/386384684 - remove Unsafe usage
  @Override
  public String toString() {
    int snapshot = UNSAFE.getIntVolatile(null, countersAddress);
    return String.format(
        "activeWorkers=%d, requestCount=%d\nqueue=%s\n",
        snapshot >>> ACTIVE_WORKERS_COUNT_BIT_OFFSET, snapshot & REQUEST_COUNT_MASK, queue);
  }

  /**
   * Constructs a batch by polling elements from the queue until it is empty, then executes it.
   *
   * <p>After the batch is executed, arranges follow-up work by calling {@code
   * #continueToNextBatchOrBecomeIdle}.
   *
   * @param requestResponse a single element to be included in the batch. This ensures the batch is
   *     non-empty.
   */
  private void executeBatch(Operation<RequestT, ResponseT> requestResponse) {
    ImmutableList<Operation<RequestT, ResponseT>> batch = populateBatch(requestResponse);
    batchExecutionStrategy
        .executeBatch(Lists.transform(batch, Operation::request), batch)
        .addListener(this::continueToNextBatchOrBecomeIdle, queueDrainingExecutor);
  }

  /**
   * Polls at most {@link #maxBatchSize} elements from the {@link #queue} and creates a batch.
   *
   * @param requestResponse an element to add to the batch.
   */
  // TODO: b/386384684 - remove Unsafe usage
  private ImmutableList<Operation<RequestT, ResponseT>> populateBatch(
      Operation<RequestT, ResponseT> requestResponse) {
    var accumulator = ImmutableList.<Operation<RequestT, ResponseT>>builder().add(requestResponse);
    while (true) {
      int snapshot = UNSAFE.getIntVolatile(null, countersAddress);
      int requestCount = snapshot & REQUEST_COUNT_MASK;
      if (requestCount == 0) {
        break;
      }
      int toRead = min(maxBatchSize, requestCount);
      if (!UNSAFE.compareAndSwapInt(null, countersAddress, snapshot, snapshot - toRead)) {
        continue;
      }
      for (int i = 0; i < toRead; i++) {
        accumulator.add(queue.take());
      }
      break;
    }
    return accumulator.build();
  }

  /**
   * Either processes the next batch or releases the held token.
   *
   * <p>Tries to process the next batch if enqueued requests are available. Otherwise, stops working
   * and decrements the active worker count.
   */
  // TODO: b/386384684 - remove Unsafe usage
  private void continueToNextBatchOrBecomeIdle() {
    while (true) {
      int snapshot = UNSAFE.getIntVolatile(null, countersAddress);
      if ((snapshot & REQUEST_COUNT_MASK) == 0) {
        // There are no enqueued requests. Tries to become idle.
        if (UNSAFE.compareAndSwapInt(
            null, countersAddress, snapshot, snapshot - ONE_ACTIVE_WORKER)) {
          return;
        }
      } else {
        // Tries to reserve an enqueued request-response to begin another batch.
        if (UNSAFE.compareAndSwapInt(null, countersAddress, snapshot, snapshot - ONE_REQUEST)) {
          executeBatch(queue.take());
          return;
        }
      }
    }
  }

  @VisibleForTesting
  static final class Operation<RequestT, ResponseT> extends AbstractFuture<ResponseT>
      implements ResponseSink<ResponseT>, FutureResponseSink<ResponseT> {
    private final RequestT request;
    private boolean isFutureSet = false;

    private Operation(RequestT request) {
      this.request = request;
    }

    private RequestT request() {
      return request;
    }

    private void setResponse(@Nullable ResponseT response) {
      // It's possible for the future to be cancelled by an external event (e.g., an interrupt).
      // `set` will return false if the future has already been completed or cancelled.
      // If `set` fails, we verify that the future was cancelled. This distinguishes
      // graceful cancellation from a bug where we try to set the response more than once.
      if (!set(response)) {
        checkState(
            isCancelled(),
            "response already set for request=%s, %s while trying to set future response %s",
            request,
            this,
            response);
      }
    }

    @Override
    public void acceptResponse(@Nullable ResponseT response) {
      setResponse(response);
    }

    @Override
    public void acceptFailure(Throwable t) {
      setException(t);
    }

    @Override
    public void acceptFutureResponse(ListenableFuture<ResponseT> futureResponse) {
      setFuture(futureResponse);
      isFutureSet = true;
    }

    private void errorIfFutureUnset() {
      if (!isFutureSet) {
        setException(
            new IllegalStateException(
                String.format(
                    "Future for %s is unexpectedly not set. It should have been set by the"
                        + " PerResponseMultiplexer.execute implementation",
                    request)));
      }
    }

    @Override
    @CanIgnoreReturnValue
    protected boolean setException(Throwable t) {
      return super.setException(t);
    }
  }

  private static final class MultiplexerAdapter<RequestT, ResponseT>
      implements BatchExecutionStrategy<RequestT, ResponseT> {
    private final Multiplexer<RequestT, ResponseT> multiplexer;

    /**
     * Executor provided by the client to invoke callbacks for individual responses within a batched
     * response.
     *
     * <p><b>Important:</b> For each batch, response callbacks are executed sequentially on a single
     * thread. If a callback involves significant processing, the client should offload the work to
     * separate threads to prevent delays in processing subsequent responses.
     */
    private final Executor responseDistributionExecutor;

    private MultiplexerAdapter(
        Multiplexer<RequestT, ResponseT> multiplexer, Executor responseDistributionExecutor) {
      this.multiplexer = multiplexer;
      this.responseDistributionExecutor = responseDistributionExecutor;
    }

    @Override
    public ListenableFuture<?> executeBatch(
        List<RequestT> requests, ImmutableList<Operation<RequestT, ResponseT>> operations) {
      ListenableFuture<List<ResponseT>> futureResponses =
          multiplexer.execute(Lists.transform(operations, Operation::request));

      addCallback(
          futureResponses,
          new FutureCallback<List<ResponseT>>() {
            @Override
            public void onFailure(Throwable t) {
              for (Operation<RequestT, ResponseT> operation : operations) {
                operation.setException(t);
              }
            }

            @Override
            public void onSuccess(List<ResponseT> responses) {
              if (responses.size() != operations.size()) {
                onFailure(
                    new AssertionError(
                        "RequestBatcher expected operations.size()="
                            + operations.size()
                            + " responses, but responses.size()="
                            + responses.size()));
                return;
              }
              for (int i = 0; i < responses.size(); ++i) {
                operations.get(i).setResponse(responses.get(i));
              }
            }
          },
          responseDistributionExecutor);

      return futureResponses;
    }
  }

  private static final class CallbackMultiplexerAdapter<RequestT, ResponseT>
      implements BatchExecutionStrategy<RequestT, ResponseT> {
    private final CallbackMultiplexer<RequestT, ResponseT> multiplexer;

    private CallbackMultiplexerAdapter(CallbackMultiplexer<RequestT, ResponseT> multiplexer) {
      this.multiplexer = multiplexer;
    }

    @Override
    public ListenableFuture<?> executeBatch(
        List<RequestT> requests, ImmutableList<Operation<RequestT, ResponseT>> operations) {
      Runnable batchCompleteCallback =
          multiplexer.execute(Lists.transform(operations, Operation::request), operations);
      return Futures.whenAllComplete(operations).run(batchCompleteCallback, directExecutor());
    }
  }

  private static final class PerResponseMultiplexerAdapter<RequestT, ResponseT>
      implements BatchExecutionStrategy<RequestT, ResponseT> {
    private final PerResponseMultiplexer<RequestT, ResponseT> multiplexer;

    private PerResponseMultiplexerAdapter(PerResponseMultiplexer<RequestT, ResponseT> multiplexer) {
      this.multiplexer = multiplexer;
    }

    @Override
    public ListenableFuture<?> executeBatch(
        List<RequestT> requests, ImmutableList<Operation<RequestT, ResponseT>> operations) {
      multiplexer.execute(Lists.transform(operations, Operation::request), operations);
      for (Operation<RequestT, ResponseT> operation : operations) {
        operation.errorIfFutureUnset();
      }
      return Futures.whenAllComplete(operations).run(() -> {}, directExecutor());
    }
  }

  private static final int REQUEST_COUNT_MASK = 0x000F_FFFF;
  private static final int ONE_REQUEST = 1;

  private static final int ACTIVE_WORKERS_COUNT_BIT_OFFSET = 20;
  private static final int ONE_ACTIVE_WORKER = 1 << ACTIVE_WORKERS_COUNT_BIT_OFFSET;
  private static final int ACTIVE_WORKERS_COUNT_MAX = 0x0000_0FFF;

  static {
    checkState(
        REQUEST_COUNT_MASK == ConcurrentFifo.CAPACITY_MASK,
        "Request Count Constants inconsistent with ConcurrentFifo");
    checkState(
        ONE_REQUEST == (REQUEST_COUNT_MASK & -REQUEST_COUNT_MASK),
        "Inconsistent Request Count Constants");
  }

  private static final Unsafe UNSAFE = UnsafeProvider.unsafe();
}
