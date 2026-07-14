// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.circuitbreaker;

import com.google.devtools.build.lib.remote.Retrier;
import java.util.Locale;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * The {@link FailureCircuitBreaker} implementation of the {@link Retrier.CircuitBreaker} prevents
 * further calls to a remote cache once the failures rate within a given window exceeds a specified
 * threshold for a build. In the context of Bazel, a new instance of {@link Retrier.CircuitBreaker}
 * is created for each build. Therefore, if the circuit breaker trips during a build, the remote
 * cache will by default be disabled for the remainder of that build. However, it will be enabled
 * again for the next build as a new instance of {@link Retrier.CircuitBreaker} will be created.
 *
 * <p>If a positive recovery delay is configured, a tripped breaker does not stay open for the whole
 * build: after the delay elapses it hands a single {@link State#TRIAL_CALL} probe to one in-flight
 * call. If that probe succeeds the breaker closes ({@link State#ACCEPT_CALLS}) and normal calls
 * resume; if it fails the breaker re-opens and schedules another probe. With a non-positive delay
 * (the default) recovery is disabled and the breaker stays open for the remainder of the build.
 */
public class FailureCircuitBreaker implements Retrier.CircuitBreaker {

  private final AtomicReference<State> state;
  private final AtomicInteger successes;
  private final AtomicInteger failures;
  private final int failureRateThreshold;
  private final int slidingWindowSize;
  private final int minCallCountToComputeFailureRate;
  private final int minFailCountToComputeFailureRate;
  private final int recoveryDelayMillis;
  private final ScheduledExecutorService scheduledExecutor;

  // Set true by the scheduled recovery task once the recovery delay has elapsed, signalling that a
  // single trial probe may be issued. Consumed by state() when it hands out a TRIAL_CALL.
  private volatile boolean trialEligible;

  // True while exactly one trial probe is outstanding, so concurrent callers keep seeing
  // REJECT_CALLS (avoiding a thundering herd) until the probe resolves. Because the CircuitBreaker
  // interface's record* methods carry no per-call identity, the probe's outcome is attributed via
  // this flag: whichever in-flight call resolves first wins. If an ACCEPT-era call is still running
  // when the probe is issued it may resolve the probe instead; this is self-correcting (a concurrent
  // success is a valid "remote recovered" signal, and a spurious re-open only costs one more
  // recovery delay), so we accept it rather than widen the interface.
  private final AtomicBoolean trialInFlight = new AtomicBoolean(false);

  /**
   * Creates a {@link FailureCircuitBreaker}.
   *
   * @param failureRateThreshold is used to set the min percentage of failure required to trip the
   *     circuit breaker in given time window.
   * @param slidingWindowSize the size of the sliding window in milliseconds to calculate the number
   *     of failures.
   * @param minCallCountToComputeFailureRate the minimum number of calls within the window before the
   *     failure rate is computed and the breaker may trip.
   * @param minFailCountToComputeFailureRate the minimum number of failures within the window before
   *     the failure rate is computed and the breaker may trip.
   * @param recoveryDelayMillis the delay in milliseconds after the breaker trips before it issues a
   *     trial probe. A non-positive value disables recovery, keeping a tripped breaker open for the
   *     remainder of the build.
   * @param scheduledExecutor executor for scheduling tasks to decrement success and failure counts
   *     and to re-enable trial probes.
   */
  public FailureCircuitBreaker(
      int failureRateThreshold,
      int slidingWindowSize,
      int minCallCountToComputeFailureRate,
      int minFailCountToComputeFailureRate,
      int recoveryDelayMillis,
      ScheduledExecutorService scheduledExecutor) {
    this.failures = new AtomicInteger(0);
    this.successes = new AtomicInteger(0);
    this.failureRateThreshold = failureRateThreshold;
    this.slidingWindowSize = slidingWindowSize;
    this.minCallCountToComputeFailureRate = minCallCountToComputeFailureRate;
    this.minFailCountToComputeFailureRate = minFailCountToComputeFailureRate;
    this.recoveryDelayMillis = recoveryDelayMillis;
    this.state = new AtomicReference<>(State.ACCEPT_CALLS);
    this.scheduledExecutor = scheduledExecutor;
  }

  @Override
  public State state() {
    State current = this.state.get();
    // When recovery is enabled and a probe is due, hand TRIAL_CALL to exactly one caller; everyone
    // else continues to see REJECT_CALLS until the probe resolves via recordSuccess/recordFailure.
    if (current == State.REJECT_CALLS
        && recoveryDelayMillis > 0
        && trialEligible
        && trialInFlight.compareAndSet(false, true)) {
      trialEligible = false;
      return State.TRIAL_CALL;
    }
    return current;
  }

  @Override
  public void recordFailure() {
    if (trialInFlight.get()) {
      // The trial probe failed: stay open and schedule another probe after the recovery delay.
      this.state.set(State.REJECT_CALLS);
      trialInFlight.set(false);
      scheduleTrial();
      return;
    }

    int failureCount = failures.incrementAndGet();
    int totalCallCount = successes.get() + failureCount;
    scheduleWindowDecrement(failures);

    if (totalCallCount < minCallCountToComputeFailureRate
        && failureCount < minFailCountToComputeFailureRate) {
      // The remote call count is below the threshold required to calculate the failure rate.
      return;
    }
    double failureRate = (failureCount * 100.0) / totalCallCount;

    // Only the thread that flips the state to REJECT_CALLS schedules the (single) recovery probe.
    if (failureRate > this.failureRateThreshold
        && this.state.compareAndSet(State.ACCEPT_CALLS, State.REJECT_CALLS)) {
      scheduleTrial();
    }
  }

  @Override
  public void recordSuccess() {
    if (trialInFlight.get()) {
      // The trial probe succeeded: reset the counters (window decrements scheduled before the reset
      // are floored at zero, see decrementToFloor) and close the breaker to resume normal calls.
      failures.set(0);
      successes.set(0);
      this.state.set(State.ACCEPT_CALLS);
      trialInFlight.set(false);
      return;
    }

    successes.incrementAndGet();
    scheduleWindowDecrement(successes);
  }

  private void scheduleWindowDecrement(AtomicInteger counter) {
    if (slidingWindowSize > 0) {
      var unused =
          scheduledExecutor.schedule(
              () -> decrementToFloor(counter), slidingWindowSize, TimeUnit.MILLISECONDS);
    }
  }

  /**
   * Decrements {@code counter} as its sliding window slides, flooring at zero. Flooring matters
   * after a successful trial probe resets the counters to zero: decrement tasks scheduled before the
   * reset would otherwise drive the counts negative and leave the breaker unable to trip again for a
   * long time.
   */
  private static void decrementToFloor(AtomicInteger counter) {
    counter.updateAndGet(value -> value > 0 ? value - 1 : 0);
  }

  private void scheduleTrial() {
    if (recoveryDelayMillis > 0 && scheduledExecutor != null) {
      var unused =
          scheduledExecutor.schedule(
              () -> {
                trialEligible = true;
              },
              recoveryDelayMillis,
              TimeUnit.MILLISECONDS);
    }
  }

  @Override
  public String failureDetails() {
    int failureCount = failures.get();
    int totalCallCount = successes.get() + failureCount;
    double failureRate = totalCallCount == 0 ? 0.0 : (failureCount * 100.0) / totalCallCount;
    String window =
        slidingWindowSize > 0
            ? String.format(Locale.US, "the last %dms", slidingWindowSize)
            : "the whole build";
    return String.format(
        Locale.US,
        "%d out of %d remote calls failed (%.2f%%) within %s, exceeding the %d%% failure rate"
            + " threshold.",
        failureCount,
        totalCallCount,
        failureRate,
        window,
        failureRateThreshold);
  }
}
