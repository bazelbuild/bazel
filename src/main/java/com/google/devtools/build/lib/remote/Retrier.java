// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote;

import static com.google.common.base.Strings.isNullOrEmpty;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static java.util.concurrent.TimeUnit.MILLISECONDS;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.util.concurrent.AsyncCallable;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.devtools.build.lib.remote.Retrier.CircuitBreaker.State;
import com.google.devtools.build.lib.remote.Retrier.ResultClassifier.Result;
import com.google.devtools.build.lib.remote.logging.RpcLogContext;
import io.grpc.Context;
import java.io.IOException;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.RejectedExecutionException;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import javax.annotation.concurrent.ThreadSafe;

/**
 * Supports retrying the execution of a {@link Callable} in case of failure.
 *
 * <p>The errors that are retried are configurable via a {@link ResultClassifier}. The delay between
 * executions is specified by a {@link Backoff}. Additionally, the retrier supports circuit breaking
 * to stop execution in case of high failure rates.
 */
@ThreadSafe
public class Retrier {

  /** A backoff strategy. */
  public interface Backoff {

    /**
     * Returns the next delay in milliseconds, or a value less than {@code 0} if we should stop
     * retrying.
     */
    long nextDelayMillis(Exception e);

    /**
     * Returns the number of calls to {@link #nextDelayMillis(Exception)} thus far, not counting any
     * calls that returned less than {@code 0}.
     */
    int getRetryAttempts();
  }

  /**
   * The circuit breaker allows to reject execution when failure rates are high.
   *
   * <p>The initial state of a circuit breaker is the {@link State#ACCEPT_CALLS}. Calls are executed
   * and retried in this state. However, if error rates are high a circuit breaker can choose to
   * transition into {@link State#REJECT_CALLS}. In this state any calls are rejected with a {@link
   * CircuitBreakerException} immediately. A circuit breaker in state {@link State#REJECT_CALLS} can
   * periodically return a {@code TRIAL_CALL} state, in which case a call will be executed once and
   * in case of success the circuit breaker may return to state {@code ACCEPT_CALLS}.
   *
   * <p>A circuit breaker implementation must be thread-safe.
   *
   * @see <a href = "https://martinfowler.com/bliki/CircuitBreaker.html">CircuitBreaker</a>
   */
  public interface CircuitBreaker {

    /** The state of the circuit breaker. */
    enum State {
      /**
       * Calls are executed and retried in case of failure.
       *
       * <p>The circuit breaker can transition into state {@link State#REJECT_CALLS}.
       */
      ACCEPT_CALLS,

      /**
       * A call is executed and not retried in case of failure.
       *
       * <p>The circuit breaker can transition into any state.
       */
      TRIAL_CALL,

      /**
       * All calls are rejected.
       *
       * <p>The circuit breaker can transition into state {@link State#TRIAL_CALL}.
       */
      REJECT_CALLS
    }

    /** Returns the current {@link State} of the circuit breaker. */
    State state();

    /**
     * Called after an execution failed.
     *
     * @param observedState the {@link State} returned by {@link #state()} when this call was
     *     dispatched, so the breaker can attribute the outcome (in particular, distinguish a {@link
     *     State#TRIAL_CALL} probe from an ordinary call).
     */
    void recordFailure(State observedState);

    /**
     * Called after an execution succeeded.
     *
     * @param observedState the {@link State} returned by {@link #state()} when this call was
     *     dispatched (see {@link #recordFailure(State)}).
     */
    void recordSuccess(State observedState);

    /**
     * Returns a human-readable description of the breaker's current failure statistics (for example
     * the observed failure rate and the configured threshold), for appending to the {@link
     * CircuitBreakerException} message when a call is rejected. Implementations that have no
     * details to report should return an empty string.
     */
    String failureDetails();
  }

  /** Thrown if the call was stopped by a circuit breaker. */
  public static class CircuitBreakerException extends IOException {
    private CircuitBreakerException(String failureDetails) {
      super(
          "Call not executed due to a high failure rate."
              + (isNullOrEmpty(failureDetails) ? "" : " " + failureDetails));
    }
  }

  /** Determines whether the result of a call is success, retriable failure or permanent failure. */
  @FunctionalInterface
  public interface ResultClassifier {

    /** The result of a call execution. */
    enum Result {
      /** A call is executed successfully. */
      SUCCESS,

      /** A call execution is failed with retriable error. */
      TRANSIENT_FAILURE,

      /** A call execution is failed with permanent error. */
      PERMANENT_FAILURE
    }

    /** Returns the {@link Result} of the call execution. */
    Result test(Exception e);
  }

  /**
   * {@link Sleeper#sleep(long)} is called to pause between synchronous retries ({@link
   * #execute(Callable)}.
   */
  public interface Sleeper {
    void sleep(long millis) throws InterruptedException;
  }

  /** Disables circuit breaking. */
  public static final CircuitBreaker ALLOW_ALL_CALLS =
      new CircuitBreaker() {
        @Override
        public State state() {
          return State.ACCEPT_CALLS;
        }

        @Override
        public void recordFailure(State observedState) {}

        @Override
        public void recordSuccess(State observedState) {}

        @Override
        public String failureDetails() {
          return "";
        }
      };

  /** Disables retries. */
  public static final Backoff RETRIES_DISABLED =
      new Backoff() {
        @Override
        public long nextDelayMillis(Exception e) {
          return -1;
        }

        @Override
        public int getRetryAttempts() {
          return 0;
        }
      };

  /** No backoff. */
  public static class ZeroBackoff implements Backoff {

    private final int maxRetries;
    private int retries;

    public ZeroBackoff(int maxRetries) {
      this.maxRetries = maxRetries;
    }

    @Override
    public long nextDelayMillis(Exception e) {
      if (retries >= maxRetries) {
        return -1;
      }
      retries++;
      return 0;
    }

    @Override
    public int getRetryAttempts() {
      return retries;
    }
  }

  private final Supplier<Backoff> backoffSupplier;
  private final ResultClassifier resultClassifier;
  private final CircuitBreaker circuitBreaker;
  private final ListeningScheduledExecutorService retryService;
  private final Sleeper sleeper;

  public Retrier(
      Supplier<Backoff> backoffSupplier,
      ResultClassifier resultClassifier,
      ListeningScheduledExecutorService retryScheduler,
      CircuitBreaker circuitBreaker) {
    this(backoffSupplier, resultClassifier, retryScheduler, circuitBreaker, MILLISECONDS::sleep);
  }

  @VisibleForTesting
  Retrier(
      Supplier<Backoff> backoffSupplier,
      ResultClassifier resultClassifier,
      ListeningScheduledExecutorService retryService,
      CircuitBreaker circuitBreaker,
      Sleeper sleeper) {
    this.backoffSupplier = backoffSupplier;
    this.resultClassifier = resultClassifier;
    this.retryService = retryService;
    this.circuitBreaker = circuitBreaker;
    this.sleeper = sleeper;
  }

  /**
   * Returns an id identifying a new logical call. When non-null it is propagated to the gRPC {@link
   * Context} (wrapped in a {@link RpcLogContext}, together with the attempt number) around each
   * attempt, so the logging interceptor can tag attempt entries with it. Returns {@code null} by
   * default (no propagation); subclasses may override to enable correlated gRPC logging.
   */
  @Nullable
  protected String newRpcId() {
    return null;
  }

  /**
   * Called once when a logical call gives up after exhausting its retries, immediately before the
   * final failure is propagated. Does nothing by default.
   *
   * <p>This fires only at the retries-exhausted seam: a retriable (transient) failure for which the
   * backoff has no further delay to give. It does <em>not</em> fire for non-retriable failures,
   * circuit-breaker rejections, or interruptions, which propagate without exhausting retries.
   *
   * @param e the exception that caused the final failure
   * @param backoff the backoff used for this logical call (for the retry-attempt count)
   * @param rpcId the id from {@link #newRpcId()} for this logical call, or {@code null}
   */
  protected void onRetriesExhausted(Exception e, Backoff backoff, @Nullable String rpcId) {}

  /** A {@link Callable} that can be retried in case of transient failure. */
  @FunctionalInterface
  public interface RetryableCallable<T, E extends Exception> extends Callable<T> {
    @Override
    T call() throws IOException, InterruptedException, E;
  }

  /**
   * Execute a {@link RetryableCallable}, retrying execution in case of transient failure and
   * returning the result in case of success.
   */
  public <T, E extends Exception> T execute(RetryableCallable<T, E> call)
      throws E, IOException, InterruptedException {
    return execute(call, newBackoff());
  }

  /**
   * Execute a {@link RetryableCallable}, retrying execution in case of transient failure and
   * returning the result in case of success with give {@link Backoff}.
   *
   * <p>{@link InterruptedException} is not retried.
   *
   * @param call the {@link Callable} to execute.
   * @throws E or {@link IOException} if the {@code call} didn't succeed within the framework
   *     specified by {@code backoffSupplier} and {@code resultClassifier}.
   * @throws CircuitBreakerException in case a call was rejected because the circuit breaker
   *     tripped.
   * @throws InterruptedException if the {@code call} throws an {@link InterruptedException} or the
   *     current thread's interrupted flag is set.
   */
  public <T, E extends Exception> T execute(RetryableCallable<T, E> call, Backoff backoff)
      throws E, IOException, InterruptedException {
    return execute(call, backoff, newRpcId());
  }

  private <T, E extends Exception> T execute(
      RetryableCallable<T, E> call, Backoff backoff, @Nullable String rpcId)
      throws E, IOException, InterruptedException {
    while (true) {
      State circuitState = circuitBreaker.state();
      if (State.REJECT_CALLS.equals(circuitState)) {
        throw new CircuitBreakerException(circuitBreaker.failureDetails());
      }
      try {
        if (Thread.interrupted()) {
          throw new InterruptedException();
        }
        T r = callWithContext(call, backoff, rpcId);
        circuitBreaker.recordSuccess(circuitState);
        return r;
      } catch (InterruptedException e) {
        throw e;
      } catch (Exception e) {
        Result r = resultClassifier.test(e);
        if (r.equals(Result.SUCCESS)) {
          circuitBreaker.recordSuccess(circuitState);
        } else {
          circuitBreaker.recordFailure(circuitState);
        }
        if (!r.equals(Result.TRANSIENT_FAILURE) || Objects.equals(circuitState, State.TRIAL_CALL)) {
          throw e;
        }
        final long delayMillis = backoff.nextDelayMillis(e);
        if (delayMillis < 0) {
          onRetriesExhausted(e, backoff, rpcId);
          throw e;
        }
        sleeper.sleep(delayMillis);
      }
    }
  }

  private <T, E extends Exception> T callWithContext(
      RetryableCallable<T, E> call, Backoff backoff, @Nullable String rpcId)
      throws E, IOException, InterruptedException {
    if (rpcId == null) {
      return call.call();
    }
    Context context =
        Context.current()
            .withValue(RpcLogContext.KEY, new RpcLogContext(rpcId, backoff.getRetryAttempts() + 1));
    Context previous = context.attach();
    try {
      return call.call();
    } finally {
      context.detach(previous);
    }
  }

  private <T> ListenableFuture<T> callWithContext(
      AsyncCallable<T> call, Backoff backoff, @Nullable String rpcId) throws Exception {
    if (rpcId == null) {
      return call.call();
    }
    Context context =
        Context.current()
            .withValue(RpcLogContext.KEY, new RpcLogContext(rpcId, backoff.getRetryAttempts() + 1));
    Context previous = context.attach();
    try {
      return call.call();
    } finally {
      context.detach(previous);
    }
  }

  /** Executes an {@link AsyncCallable}, retrying execution in case of transient failure. */
  public <T> ListenableFuture<T> executeAsync(AsyncCallable<T> call) {
    return executeAsync(call, newBackoff());
  }

  /**
   * Executes an {@link AsyncCallable}, retrying execution in case of transient failure with the
   * given backoff.
   */
  public <T> ListenableFuture<T> executeAsync(AsyncCallable<T> call, Backoff backoff) {
    return executeAsync(call, backoff, newRpcId());
  }

  private <T> ListenableFuture<T> executeAsync(
      AsyncCallable<T> call, Backoff backoff, @Nullable String rpcId) {
    final State circuitState = circuitBreaker.state();
    if (State.REJECT_CALLS.equals(circuitState)) {
      return immediateFailedFuture(new CircuitBreakerException(circuitBreaker.failureDetails()));
    }
    try {
      ListenableFuture<T> future =
          Futures.transformAsync(
              callWithContext(call, backoff, rpcId),
              (f) -> {
                circuitBreaker.recordSuccess(circuitState);
                return immediateFuture(f);
              },
              directExecutor());
      return Futures.catchingAsync(
          future,
          Exception.class,
          t -> onExecuteAsyncFailure(t, call, backoff, circuitState, rpcId),
          directExecutor());
    } catch (Exception e) {
      return onExecuteAsyncFailure(e, call, backoff, circuitState, rpcId);
    }
  }

  private <T> ListenableFuture<T> onExecuteAsyncFailure(
      Exception t,
      AsyncCallable<T> call,
      Backoff backoff,
      State circuitState,
      @Nullable String rpcId) {
    Result r = resultClassifier.test(t);
    if (r.equals(Result.TRANSIENT_FAILURE)) {
      circuitBreaker.recordFailure(circuitState);
      if (circuitState.equals(State.TRIAL_CALL)) {
        return immediateFailedFuture(t);
      }
      long waitMillis = backoff.nextDelayMillis(t);
      if (waitMillis >= 0) {
        try {
          return Futures.scheduleAsync(
              () -> executeAsync(call, backoff, rpcId), waitMillis, MILLISECONDS, retryService);
        } catch (RejectedExecutionException e) {
          // May be thrown by .scheduleAsync(...) if i.e. the executor is shutdown.
          return immediateFailedFuture(new IOException(e));
        }
      } else {
        onRetriesExhausted(t, backoff, rpcId);
        return immediateFailedFuture(t);
      }
    } else {
      if (r.equals(Result.SUCCESS)) {
        circuitBreaker.recordSuccess(circuitState);
      } else {
        circuitBreaker.recordFailure(circuitState);
      }
      return immediateFailedFuture(t);
    }
  }

  public Backoff newBackoff() {
    return backoffSupplier.get();
  }

  public boolean isRetriable(Exception e) {
    return resultClassifier.test(e).equals(Result.TRANSIENT_FAILURE);
  }

  CircuitBreaker getCircuitBreaker() {
    return this.circuitBreaker;
  }
}
