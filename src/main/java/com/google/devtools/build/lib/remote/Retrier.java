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

import static java.util.concurrent.TimeUnit.MILLISECONDS;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.util.concurrent.AsyncCallable;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.remote.Retrier.CircuitBreaker.State;
import com.google.devtools.build.lib.remote.Retrier.ResultClassifier.Result;
import java.io.IOException;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.RejectedExecutionException;
import java.util.function.Supplier;
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

    /** Called after an execution failed. */
    void recordFailure();

    /** Called after an execution succeeded. */
    void recordSuccess();
  }

  /** Thrown if the call was stopped by a circuit breaker. */
  public static class CircuitBreakerException extends IOException {
    private CircuitBreakerException() {
      super("Call not executed due to a high failure rate.");
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
        public void recordFailure() {}

        @Override
        public void recordSuccess() {}
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

  ListeningScheduledExecutorService getRetryService() {
    return retryService;
  }

  /**
   * Execute a {@link Callable}, retrying execution in case of transient failure and returning the
   * result in case of success.
   */
  public <T> T execute(Callable<T> call) throws Exception {
    return execute(call, newBackoff());
  }

  /**
   * Execute a {@link Callable}, retrying execution in case of transient failure and returning the
   * result in case of success with give {@link Backoff}.
   *
   * <p>{@link InterruptedException} is not retried.
   *
   * @param call the {@link Callable} to execute.
   * @throws Exception if the {@code call} didn't succeed within the framework specified by {@code
   *     backoffSupplier} and {@code resultClassifier}.
   * @throws CircuitBreakerException in case a call was rejected because the circuit breaker
   *     tripped.
   * @throws InterruptedException if the {@code call} throws an {@link InterruptedException} or the
   *     current thread's interrupted flag is set.
   */
  public <T> T execute(Callable<T> call, Backoff backoff) throws Exception {
    while (true) {
      State circuitState = circuitBreaker.state();
      if (State.REJECT_CALLS.equals(circuitState)) {
        throw new CircuitBreakerException();
      }
      try {
        if (Thread.interrupted()) {
          throw new InterruptedException();
        }
        T r = call.call();
        circuitBreaker.recordSuccess();
        return r;
      } catch (InterruptedException e) {
        throw e;
      } catch (Exception e) {
        Result r = resultClassifier.test(e);
        if (r.equals(Result.SUCCESS)) {
          circuitBreaker.recordSuccess();
        } else {
          circuitBreaker.recordFailure();
        }
        if (!r.equals(Result.TRANSIENT_FAILURE) || Objects.equals(circuitState, State.TRIAL_CALL)) {
          throw e;
        }
        final long delayMillis = backoff.nextDelayMillis(e);
        if (delayMillis < 0) {
          throw e;
        }
        sleeper.sleep(delayMillis);
      }
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
    final State circuitState = circuitBreaker.state();
    if (State.REJECT_CALLS.equals(circuitState)) {
      return Futures.immediateFailedFuture(new CircuitBreakerException());
    }
    try {
      ListenableFuture<T> future =
          Futures.transformAsync(
              call.call(),
              (f) -> {
                circuitBreaker.recordSuccess();
                return Futures.immediateFuture(f);
              },
              MoreExecutors.directExecutor());
      return Futures.catchingAsync(
          future,
          Exception.class,
          t -> onExecuteAsyncFailure(t, call, backoff, circuitState),
          MoreExecutors.directExecutor());
    } catch (Exception e) {
      return onExecuteAsyncFailure(e, call, backoff, circuitState);
    }
  }

  private <T> ListenableFuture<T> onExecuteAsyncFailure(
      Exception t, AsyncCallable<T> call, Backoff backoff, State circuitState) {
    Result r = resultClassifier.test(t);
    if (r.equals(Result.TRANSIENT_FAILURE)) {
      circuitBreaker.recordFailure();
      if (circuitState.equals(State.TRIAL_CALL)) {
        return Futures.immediateFailedFuture(t);
      }
      long waitMillis = backoff.nextDelayMillis(t);
      if (waitMillis >= 0) {
        try {
          return Futures.scheduleAsync(
              () -> executeAsync(call, backoff), waitMillis, MILLISECONDS, retryService);
        } catch (RejectedExecutionException e) {
          // May be thrown by .scheduleAsync(...) if i.e. the executor is shutdown.
          return Futures.immediateFailedFuture(new IOException(e));
        }
      } else {
        return Futures.immediateFailedFuture(t);
      }
    } else {
      if (r.equals(Result.SUCCESS)) {
        circuitBreaker.recordSuccess();
      } else {
        circuitBreaker.recordFailure();
      }
      return Futures.immediateFailedFuture(t);
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
