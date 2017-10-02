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
import com.google.devtools.build.lib.remote.Retrier2.CircuitBreaker.State;
import java.io.IOException;
import java.util.concurrent.Callable;
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
// TODO(buchgr): Move to a different package and use it for BES code.
@ThreadSafe
class Retrier2 {

  /**
   * A backoff strategy.
   */
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
   * <p>The initial state of a circuit breaker is the {@link State#ACCEPT_CALLS}. Calls are
   * executed and retried in this state. However, if error rates are high a circuit breaker can
   * choose to transition into {@link State#REJECT_CALLS}. In this state any calls are rejected with
   * a {@link RetryException2} immediately. A circuit breaker in state {@link State#REJECT_CALLS}
   * can periodically return a {@code TRIAL_CALL} state, in which case a call will be executed once
   * and in case of success the circuit breaker may return to state {@code ACCEPT_CALLS}.
   *
   * <p>A circuit breaker implementation must be thread-safe.
   *
   * @see <a href = "https://martinfowler.com/bliki/CircuitBreaker.html">CircuitBreaker</a>
   */
  public interface CircuitBreaker {

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

    /**
     * Returns the current {@link State} of the circuit breaker.
     */
    State state();

    /**
     * Called after an execution failed.
     */
    void recordFailure();

    /**
     * Called after an execution succeeded.
     */
    void recordSuccess();
  }

  public interface Sleeper {
    void sleep(long millis) throws InterruptedException;
  }

  public static class RetryException2 extends IOException {

    private final int attempts;

    public RetryException2(String message, int numRetries, Exception cause) {
      super(message, cause);
      this.attempts = numRetries + 1;
    }

    protected RetryException2(String message) {
      super(message);
      this.attempts = 0;
    }

    /**
     * Returns the number of times a {@link Callable} has been executed before this exception
     * was thrown.
     */
    public int getAttempts() {
      return attempts;
    }
  }

  public static class CircuitBreakerException extends RetryException2 {

    private CircuitBreakerException(String message, int numRetries, Exception cause) {
      super(message, numRetries, cause);
    }

    private CircuitBreakerException() {
      super("Call not executed due to a high failure rate.");
    }
  }

  public static final CircuitBreaker ALLOW_ALL_CALLS = new CircuitBreaker() {
    @Override
    public State state() {
      return State.ACCEPT_CALLS;
    }

    @Override
    public void recordFailure() {
    }

    @Override
    public void recordSuccess() {
    }
  };

  public static final Backoff RETRIES_DISABLED = new Backoff() {
    @Override
    public long nextDelayMillis() {
      return -1;
    }

    @Override
    public int getRetryAttempts() {
      return 0;
    }
  };

  private final Supplier<Backoff> backoffSupplier;
  private final Predicate<? super Exception> shouldRetry;
  private final CircuitBreaker circuitBreaker;
  private final Sleeper sleeper;

  public Retrier2 (Supplier<Backoff> backoffSupplier, Predicate<? super Exception> shouldRetry,
      CircuitBreaker circuitBreaker) {
    this(backoffSupplier, shouldRetry, circuitBreaker, TimeUnit.MILLISECONDS::sleep);
  }

  @VisibleForTesting
  Retrier2 (Supplier<Backoff> backoffSupplier, Predicate<? super Exception> shouldRetry,
      CircuitBreaker circuitBreaker, Sleeper sleeper) {
    this.backoffSupplier = backoffSupplier;
    this.shouldRetry = shouldRetry;
    this.circuitBreaker = circuitBreaker;
    this.sleeper = sleeper;
  }

  /**
   * Execute a {@link Callable}, retrying execution in case of failure and returning the result in
   * case of success.
   *
   * <p>{@link InterruptedException} is not retried.
   *
   * @param call  the {@link Callable} to execute.
   * @throws RetryException2 if the {@code call} didn't succeed within the framework specified by
   *                        {@code backoffSupplier} and {@code shouldRetry}.
   * @throws CircuitBreakerException  in case a call was rejected because the circuit breaker
   *                                  tripped.
   * @throws InterruptedException if the {@code call} throws an {@link InterruptedException} or the
   *                              current thread's interrupted flag is set.
   */
  public <T> T execute(Callable<T> call) throws RetryException2, InterruptedException {
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
        if (e instanceof RetryException2) {
          // Support nested retry calls.
          e = (Exception) e.getCause();
        }
        if (State.TRIAL_CALL.equals(circuitState)) {
          throw new CircuitBreakerException("Call failed in circuit breaker half open state.", 0,
              e);
        }
        int attempts = backoff.getRetryAttempts();
        if (!shouldRetry.test(e)) {
          throw new RetryException2("Call failed with not retriable error.", attempts, e);
        }
        final long delayMillis = backoff.nextDelayMillis();
        if (delayMillis < 0) {
          throw new RetryException2(
              "Call failed after exhausting retry attempts: " + attempts, attempts, e);
        }
        sleeper.sleep(delayMillis);
      }
    }
  }

  //TODO(buchgr): Add executeAsync to be used by ByteStreamUploader
  // <T> ListenableFuture<T> executeAsync(AsyncCallable<T> call, ScheduledExecutorService executor)

  public Backoff newBackoff() {
    return backoffSupplier.get();
  }

  public boolean isRetriable(Exception e) {
    return shouldRetry.test(e);
  }
}
