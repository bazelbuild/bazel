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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.AsyncCallable;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListenableScheduledFuture;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.remote.Retrier.CircuitBreaker.State;
import java.io.IOException;
import java.util.concurrent.Callable;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.function.Predicate;
import java.util.function.Supplier;
import javax.annotation.concurrent.ThreadSafe;

/**
 * Supports retrying the execution of a {@link Callable} in case of failure.
 *
 * <p>The errors that are retried are configurable via a {@link Predicate<? super Exception>}. The
 * delay between executions is specified by a {@link Backoff}. Additionally, the retrier supports
 * circuit breaking to stop execution in case of high failure rates.
 */
@ThreadSafe
public class Retrier {

  /** A backoff strategy. */
  public interface Backoff {

    /**
     * Returns the next delay in milliseconds, or a value less than {@code 0} if we should stop
     * retrying.
     */
    long nextDelayMillis();

    /**
     * Returns the number of calls to {@link #nextDelayMillis()} thus far, not counting any calls
     * that returned less than {@code 0}.
     */
    int getRetryAttempts();
  }

  /**
   * The circuit breaker allows to reject execution when failure rates are high.
   *
   * <p>The initial state of a circuit breaker is the {@link State#ACCEPT_CALLS}. Calls are executed
   * and retried in this state. However, if error rates are high a circuit breaker can choose to
   * transition into {@link State#REJECT_CALLS}. In this state any calls are rejected with a {@link
   * RetryException} immediately. A circuit breaker in state {@link State#REJECT_CALLS} can
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

  /**
   * {@link Sleeper#sleep(long)} is called to pause between synchronous retries ({@link
   * #execute(Callable)}.
   */
  public interface Sleeper {
    void sleep(long millis) throws InterruptedException;
  }

  /**
   * Wraps around the actual cause for the retry. Contains information about the number of retry
   * attempts.
   */
  public static class RetryException extends IOException {

    private final int attempts;

    public RetryException(String message, int numRetries, Exception cause) {
      super(message, cause);
      this.attempts = numRetries + 1;
    }

    protected RetryException(String message) {
      super(message);
      this.attempts = 0;
    }

    /**
     * Returns the number of times a {@link Callable} has been executed before this exception was
     * thrown.
     */
    public int getAttempts() {
      return attempts;
    }
  }

  /** Thrown if the call was stopped by a circuit breaker. */
  public static class CircuitBreakerException extends RetryException {

    private CircuitBreakerException(String message, int numRetries, Exception cause) {
      super(message, numRetries, cause);
    }

    private CircuitBreakerException() {
      super("Call not executed due to a high failure rate.");
    }
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
        public long nextDelayMillis() {
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
    public long nextDelayMillis() {
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
  private final Predicate<? super Exception> shouldRetry;
  private final CircuitBreaker circuitBreaker;
  private final ListeningScheduledExecutorService retryService;
  private final Sleeper sleeper;

  public Retrier(
      Supplier<Backoff> backoffSupplier,
      Predicate<? super Exception> shouldRetry,
      ListeningScheduledExecutorService retryScheduler,
      CircuitBreaker circuitBreaker) {
    this(
        backoffSupplier, shouldRetry, retryScheduler, circuitBreaker, TimeUnit.MILLISECONDS::sleep);
  }

  @VisibleForTesting
  Retrier(
      Supplier<Backoff> backoffSupplier,
      Predicate<? super Exception> shouldRetry,
      ListeningScheduledExecutorService retryService,
      CircuitBreaker circuitBreaker,
      Sleeper sleeper) {
    this.backoffSupplier = backoffSupplier;
    this.shouldRetry = shouldRetry;
    this.retryService = retryService;
    this.circuitBreaker = circuitBreaker;
    this.sleeper = sleeper;
  }

  /**
   * Execute a {@link Callable}, retrying execution in case of failure and returning the result in
   * case of success.
   *
   * <p>{@link InterruptedException} is not retried.
   *
   * @param call the {@link Callable} to execute.
   * @throws RetryException if the {@code call} didn't succeed within the framework specified by
   *     {@code backoffSupplier} and {@code shouldRetry}.
   * @throws CircuitBreakerException in case a call was rejected because the circuit breaker
   *     tripped.
   * @throws InterruptedException if the {@code call} throws an {@link InterruptedException} or the
   *     current thread's interrupted flag is set.
   */
  public <T> T execute(Callable<T> call) throws RetryException, InterruptedException {
    final Backoff backoff = newBackoff();
    while (true) {
      final State circuitState;
      circuitState = circuitBreaker.state();
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
        circuitBreaker.recordFailure();
        throw e;
      } catch (Exception e) {
        circuitBreaker.recordFailure();
        Exception orig = e;
        if (e instanceof RetryException) {
          // Support nested retry calls.
          e = (Exception) e.getCause();
        }
        if (State.TRIAL_CALL.equals(circuitState)) {
          throw new CircuitBreakerException(
              "Call failed in circuit breaker half open state.", 0, e);
        }
        int attempts = backoff.getRetryAttempts();
        if (!shouldRetry.test(e)) {
          throw new RetryException(
              "Call failed with not retriable error: " + orig.getMessage(), attempts, e);
        }
        final long delayMillis = backoff.nextDelayMillis();
        if (delayMillis < 0) {
          throw new RetryException(
              "Call failed after " + attempts + " retry attempts: " + orig.getMessage(),
              attempts,
              e);
        }
        sleeper.sleep(delayMillis);
      }
    }
  }

  /**
   * Executes an {@link AsyncCallable}, retrying execution in case of failure and returning a {@link
   * ListenableFuture} pointing to the result/error.
   */
  public <T> ListenableFuture<T> executeAsync(AsyncCallable<T> call) {
    SettableFuture<T> f = SettableFuture.create();
    executeAsync(call, f);
    return f;
  }

  /**
   * Executes an {@link AsyncCallable}, retrying execution in case of failure and uses the provided
   * {@code promise} to point to the result/error.
   */
  public <T> void executeAsync(AsyncCallable<T> call, SettableFuture<T> promise) {
    Preconditions.checkNotNull(call);
    Preconditions.checkNotNull(promise);
    Backoff backoff = newBackoff();
    executeAsync(call, promise, backoff);
  }

  private <T> void executeAsync(AsyncCallable<T> call, SettableFuture<T> outerF, Backoff backoff) {
    Preconditions.checkState(!outerF.isDone(), "outerF completed already.");
    try {
      Futures.addCallback(
          call.call(),
          new FutureCallback<T>() {
            @Override
            public void onSuccess(T t) {
              outerF.set(t);
            }

            @Override
            public void onFailure(Throwable t) {
              onExecuteAsyncFailure(t, call, outerF, backoff);
            }
          },
          MoreExecutors.directExecutor());
    } catch (Exception e) {
      onExecuteAsyncFailure(e, call, outerF, backoff);
    }
  }

  private <T> void onExecuteAsyncFailure(
      Throwable t, AsyncCallable<T> call, SettableFuture<T> outerF, Backoff backoff) {
    long waitMillis = backoff.nextDelayMillis();
    if (waitMillis >= 0 && t instanceof Exception && isRetriable((Exception) t)) {
      try {
        ListenableScheduledFuture<?> sf =
            retryService.schedule(
                () -> {
                  executeAsync(call, outerF, backoff);
                },
                waitMillis,
                TimeUnit.MILLISECONDS);
        Futures.addCallback(
            sf,
            new FutureCallback<Object>() {
              @Override
              public void onSuccess(Object o) {
                // Submitted successfully. Intentionally left empty.
              }

              @Override
              public void onFailure(Throwable t) {
                Exception e = t instanceof Exception ? (Exception) t : new Exception(t);
                outerF.setException(
                    new RetryException(
                        "Scheduled execution errored.", backoff.getRetryAttempts(), e));
              }
            },
            MoreExecutors.directExecutor());
      } catch (RejectedExecutionException e) {
        // May be thrown by .schedule(...) if i.e. the executor is shutdown.
        outerF.setException(
            new RetryException("Rejected by executor.", backoff.getRetryAttempts(), e));
      }
    } else {
      Exception e = t instanceof Exception ? (Exception) t : new Exception(t);
      String message =
          waitMillis >= 0
              ? "Status not retriable"
              : "Exhausted retry attempts (" + backoff.getRetryAttempts() + ")";
      if (!e.getMessage().isEmpty()) {
        message += ": " + e.getMessage();
      } else {
        message += ".";
      }
      RetryException error = new RetryException(message, backoff.getRetryAttempts(), e);
      outerF.setException(error);
    }
  }

  public Backoff newBackoff() {
    return backoffSupplier.get();
  }

  public boolean isRetriable(Exception e) {
    return shouldRetry.test(e);
  }
}
