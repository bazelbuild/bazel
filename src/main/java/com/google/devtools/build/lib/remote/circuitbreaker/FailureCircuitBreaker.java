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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A {@link Retrier.CircuitBreaker} that stops calling a remote cache/executor once the failure rate
 * within a sliding window exceeds a configured threshold. In the context of Bazel a new instance is
 * created for each build, so a breaker that trips disables the remote cache for that build (by
 * default for the remainder of it) and is reset for the next build.
 *
 * <p>Recovery within a build is opt-in via a positive recovery delay. After the breaker trips it
 * moves through a small state machine: {@code OPEN} (all calls rejected) then, once the recovery
 * delay elapses, {@code RECOVERING} (the next caller is handed a single {@link State#TRIAL_CALL}
 * probe while everyone else keeps seeing {@link State#REJECT_CALLS}) then {@code PROBING} (the
 * probe is in flight). A successful probe closes the breaker ({@link State#ACCEPT_CALLS}) and
 * starts a fresh window; a failed probe reopens it and schedules the next probe. With a
 * non-positive delay (the default) the breaker never leaves {@code OPEN}, preserving the previous
 * permanent-open behaviour.
 *
 * <p>The outcome of a probe is attributed using the {@link State} each call observed when it was
 * dispatched (passed to {@link #recordSuccess(State)}/{@link #recordFailure(State)}): only the call
 * that observed {@link State#TRIAL_CALL} can resolve the probe, so a slow call that started before
 * the breaker tripped cannot hijack it.
 */
public class FailureCircuitBreaker implements Retrier.CircuitBreaker {

  /**
   * Internal health state. Maps to the interface {@link State} exposed to the {@link Retrier}:
   * {@code CLOSED} to {@link State#ACCEPT_CALLS}, everything else to {@link State#REJECT_CALLS}
   * except that {@code RECOVERING} hands a single {@link State#TRIAL_CALL} to one caller.
   */
  private enum Health {
    /** Normal operation; calls are accepted. */
    CLOSED,
    /** Tripped; all calls are rejected until the recovery delay elapses. */
    OPEN,
    /** The recovery delay has elapsed; the next caller may be given a single trial probe. */
    RECOVERING,
    /** A single trial probe is in flight; other callers are still rejected. */
    PROBING
  }

  private final AtomicReference<Health> health = new AtomicReference<>(Health.CLOSED);
  private final AtomicInteger successes = new AtomicInteger(0);
  private final AtomicInteger failures = new AtomicInteger(0);

  // Bumped whenever the counters are reset, so window-decrement tasks scheduled before the reset
  // become no-ops instead of decrementing the fresh counts.
  private final AtomicLong generation = new AtomicLong(0);

  private final int failureRateThreshold;
  private final int slidingWindowSize;
  private final int minCallCountToComputeFailureRate;
  private final int minFailCountToComputeFailureRate;
  private final int recoveryDelayMillis;
  private final ScheduledExecutorService scheduledExecutor;

  /**
   * Creates a {@link FailureCircuitBreaker}.
   *
   * @param failureRateThreshold is used to set the min percentage of failure required to trip the
   *     circuit breaker in given time window.
   * @param slidingWindowSize the size of the sliding window in milliseconds to calculate the number
   *     of failures.
   * @param minCallCountToComputeFailureRate the minimum number of calls within the window before
   *     the failure rate is computed and the breaker may trip.
   * @param minFailCountToComputeFailureRate the minimum number of failures within the window before
   *     the failure rate is computed and the breaker may trip.
   * @param recoveryDelayMillis the delay in milliseconds after the breaker trips before it issues a
   *     trial probe. A non-positive value disables recovery, keeping a tripped breaker open for the
   *     remainder of the build.
   * @param scheduledExecutor executor for scheduling tasks to decrement success and failure counts
   *     and to make trial probes available; may be {@code null} only when both the sliding window
   *     and recovery are disabled.
   */
  public FailureCircuitBreaker(
      int failureRateThreshold,
      int slidingWindowSize,
      int minCallCountToComputeFailureRate,
      int minFailCountToComputeFailureRate,
      int recoveryDelayMillis,
      ScheduledExecutorService scheduledExecutor) {
    this.failureRateThreshold = failureRateThreshold;
    this.slidingWindowSize = slidingWindowSize;
    this.minCallCountToComputeFailureRate = minCallCountToComputeFailureRate;
    this.minFailCountToComputeFailureRate = minFailCountToComputeFailureRate;
    this.recoveryDelayMillis = recoveryDelayMillis;
    this.scheduledExecutor = scheduledExecutor;
  }

  @Override
  public State state() {
    Health current = health.get();
    if (current == Health.RECOVERING) {
      // The recovery delay has elapsed: hand TRIAL_CALL to exactly one caller (whoever wins the
      // CAS); every other caller keeps seeing REJECT_CALLS until the probe resolves.
      if (health.compareAndSet(Health.RECOVERING, Health.PROBING)) {
        return State.TRIAL_CALL;
      }
      // Lost the race for the probe; re-read so the returned state reflects where we ended up.
      current = health.get();
    }
    return current == Health.CLOSED ? State.ACCEPT_CALLS : State.REJECT_CALLS;
  }

  @Override
  public void recordFailure(State observedState) {
    if (observedState == State.TRIAL_CALL) {
      // The trial probe failed: reopen and schedule the next probe. Only the probe observes
      // TRIAL_CALL, so a slow pre-trip call cannot consume the trial outcome here.
      if (health.compareAndSet(Health.PROBING, Health.OPEN)) {
        scheduleTrial();
      }
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

    // Only the thread that trips the breaker (wins the CAS) schedules the single recovery probe.
    if (failureRate > failureRateThreshold && health.compareAndSet(Health.CLOSED, Health.OPEN)) {
      scheduleTrial();
    }
  }

  @Override
  public void recordSuccess(State observedState) {
    if (observedState == State.TRIAL_CALL) {
      // The trial probe succeeded: close the breaker and start a fresh window.
      if (health.compareAndSet(Health.PROBING, Health.CLOSED)) {
        resetCounters();
      }
      return;
    }

    successes.incrementAndGet();
    scheduleWindowDecrement(successes);
  }

  private void resetCounters() {
    // Invalidate window-decrement tasks scheduled before the reset, then zero the counts, so stale
    // decrements cannot erase failures recorded after recovery.
    generation.incrementAndGet();
    failures.set(0);
    successes.set(0);
  }

  private void scheduleWindowDecrement(AtomicInteger counter) {
    if (slidingWindowSize > 0) {
      long scheduledGeneration = generation.get();
      var unused =
          scheduledExecutor.schedule(
              () -> {
                if (generation.get() == scheduledGeneration) {
                  var ignored = counter.decrementAndGet();
                }
              },
              slidingWindowSize,
              TimeUnit.MILLISECONDS);
    }
  }

  private void scheduleTrial() {
    if (recoveryDelayMillis > 0 && scheduledExecutor != null) {
      var unused =
          scheduledExecutor.schedule(
              () -> {
                var ignored = health.compareAndSet(Health.OPEN, Health.RECOVERING);
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
