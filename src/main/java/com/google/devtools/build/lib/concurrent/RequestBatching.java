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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.util.concurrent.Futures.addCallback;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.AbstractFuture;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.List;
import java.util.concurrent.Executor;
import javax.annotation.Nullable;

/** Shared API and internal components for request batching. */
public final class RequestBatching {

  private RequestBatching() {}

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
  public interface ResponseSink<RequestT, ResponseT> {
    /** Returns the original request associated with this sink. */
    RequestT request();

    /** Returns true if the sink has been completed (success or failure). */
    boolean isDone();

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
        List<RequestT> requests, ImmutableList<? extends ResponseSink<RequestT, ResponseT>> sinks);
  }

  /**
   * Accepts a future response value.
   *
   * <p>Used with {@link FutureMultiplexer}.
   */
  public interface FutureSink<ResponseT> {
    void acceptFuture(ListenableFuture<ResponseT> future);
  }

  /** Batching strategy when a single batch request returns a response per future request. */
  public interface FutureMultiplexer<RequestT, ResponseT> {
    /** Executes {@code requests} in a batch and populates corresponding {@code responses}. */
    void execute(List<RequestT> requests, ImmutableList<? extends FutureSink<ResponseT>> responses);
  }

  static <RequestT, ResponseT>
      BatchExecutionStrategy<RequestT, ResponseT> createBatchExecutionStrategy(
          Multiplexer<RequestT, ResponseT> multiplexer, Executor responseDistributionExecutor) {
    return new MultiplexerAdapter<>(multiplexer, responseDistributionExecutor);
  }

  static <RequestT, ResponseT>
      BatchExecutionStrategy<RequestT, ResponseT> createCallbackBatchExecutionStrategy(
          CallbackMultiplexer<RequestT, ResponseT> multiplexer) {
    return new CallbackMultiplexerAdapter<>(multiplexer);
  }

  static <RequestT, ResponseT>
      BatchExecutionStrategy<RequestT, ResponseT> createFutureBatchExecutionStrategy(
          FutureMultiplexer<RequestT, ResponseT> multiplexer) {
    return new FutureMultiplexerAdapter<>(multiplexer);
  }

  interface BatchExecutionStrategy<RequestT, ResponseT> {
    ListenableFuture<?> executeBatch(
        List<RequestT> requests, ImmutableList<Operation<RequestT, ResponseT>> operations);
  }

  static final class Operation<RequestT, ResponseT> extends AbstractFuture<ResponseT>
      implements ResponseSink<RequestT, ResponseT>, FutureSink<ResponseT> {
    private final RequestT request;
    private boolean isFutureSet = false;

    Operation(RequestT request) {
      this.request = request;
    }

    @Override
    public RequestT request() {
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
    public void acceptFuture(ListenableFuture<ResponseT> future) {
      setFuture(future);
      isFutureSet = true;
    }

    void errorIfFutureUnset() {
      if (!isFutureSet) {
        setException(
            new IllegalStateException(
                String.format(
                    "Future for %s is unexpectedly not set. It should have been set by the"
                        + " FutureMultiplexer.execute implementation",
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

  private static final class FutureMultiplexerAdapter<RequestT, ResponseT>
      implements BatchExecutionStrategy<RequestT, ResponseT> {
    private final FutureMultiplexer<RequestT, ResponseT> multiplexer;

    private FutureMultiplexerAdapter(FutureMultiplexer<RequestT, ResponseT> multiplexer) {
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
}
