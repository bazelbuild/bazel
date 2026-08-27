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

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.remote.Retrier.CircuitBreaker.State;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.common.options.Options;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class FailureCircuitBreakerTest {

  private final RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);

  /**
   * Returns a mock scheduler that records scheduled tasks by delay: tasks scheduled at {@code
   * recoveryDelayMillis} (the recovery probes) go into {@code trialTasks}, everything else (the
   * sliding-window decrements) into {@code windowTasks}. A test runs them manually to simulate the
   * recovery delay or the window elapsing. All tasks the breaker schedules are {@link Runnable}s.
   */
  private static ScheduledExecutorService newSchedulerCapturing(
      long recoveryDelayMillis, List<Runnable> trialTasks, List<Runnable> windowTasks) {
    ScheduledExecutorService mockScheduler = mock(ScheduledExecutorService.class);
    when(mockScheduler.schedule(any(Runnable.class), anyLong(), any(TimeUnit.class)))
        .thenAnswer(
            invocation -> {
              Runnable task = invocation.getArgument(0);
              long delayMillis = invocation.getArgument(1);
              if (delayMillis == recoveryDelayMillis) {
                trialTasks.add(task);
              } else {
                windowTasks.add(task);
              }
              return null;
            });
    return mockScheduler;
  }

  @Test
  public void testRecordFailure_circuitTrips() {
    final int failureRateThreshold = 10;
    final int windowInterval = 100;
    List<Runnable> windowTasks = Collections.synchronizedList(new ArrayList<>());
    FailureCircuitBreaker failureCircuitBreaker =
        new FailureCircuitBreaker(
            failureRateThreshold,
            windowInterval,
            remoteOptions.getRemoteMinCallCountToComputeFailureRate(),
            remoteOptions.getRemoteMinFailCountToComputeFailureRate(),
            /* recoveryDelayMillis= */ 0,
            newSchedulerCapturing(/* recoveryDelayMillis= */ 0, new ArrayList<>(), windowTasks));

    List<Runnable> listOfSuccessAndFailureCalls = new ArrayList<>();
    for (int index = 0; index < failureRateThreshold; index++) {
      listOfSuccessAndFailureCalls.add(
          () -> failureCircuitBreaker.recordFailure(State.ACCEPT_CALLS));
    }
    for (int index = 0; index < failureRateThreshold * 9; index++) {
      listOfSuccessAndFailureCalls.add(
          () -> failureCircuitBreaker.recordSuccess(State.ACCEPT_CALLS));
    }
    Collections.shuffle(listOfSuccessAndFailureCalls);

    // Make calls equal to the threshold number of not-ignored failure calls in parallel.
    listOfSuccessAndFailureCalls.stream().parallel().forEach(Runnable::run);
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.ACCEPT_CALLS);

    // Run all captured window-decrement tasks to simulate the window expiring.
    int expectedCalls = failureRateThreshold * 10;
    assertThat(windowTasks).hasSize(expectedCalls);
    windowTasks.forEach(Runnable::run);
    windowTasks.clear(); // Clear for the next round.

    listOfSuccessAndFailureCalls.stream().parallel().forEach(Runnable::run);
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.ACCEPT_CALLS);

    // We don't run the new scheduled tasks, simulating being within the window.
    failureCircuitBreaker.recordFailure(State.ACCEPT_CALLS);
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.REJECT_CALLS);
  }

  @Test
  public void testRecordFailure_minCallCriteriaNotMet() {
    final int failureRateThreshold = 0;
    final int windowInterval = 100;
    final int minCallToComputeFailure = remoteOptions.getRemoteMinCallCountToComputeFailureRate();
    FailureCircuitBreaker failureCircuitBreaker =
        new FailureCircuitBreaker(
            failureRateThreshold,
            windowInterval,
            remoteOptions.getRemoteMinCallCountToComputeFailureRate(),
            remoteOptions.getRemoteMinFailCountToComputeFailureRate(),
            /* recoveryDelayMillis= */ 0,
            mock(ScheduledExecutorService.class));

    // Make success calls, a failure call, and a total number of calls less than
    // minCallToComputeFailure.
    for (int index = 0; index < minCallToComputeFailure - 2; index++) {
      failureCircuitBreaker.recordSuccess(State.ACCEPT_CALLS);
    }
    failureCircuitBreaker.recordFailure(State.ACCEPT_CALLS);
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.ACCEPT_CALLS);

    // We don't run the scheduled tasks, simulating being within the window.
    failureCircuitBreaker.recordFailure(State.ACCEPT_CALLS);
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.REJECT_CALLS);
  }

  @Test
  public void testRecordFailure_minFailCriteriaNotMet() {
    final int failureRateThreshold = 10;
    final int windowInterval = 100;
    final int minFailToComputeFailure = remoteOptions.getRemoteMinFailCountToComputeFailureRate();
    FailureCircuitBreaker failureCircuitBreaker =
        new FailureCircuitBreaker(
            failureRateThreshold,
            windowInterval,
            remoteOptions.getRemoteMinCallCountToComputeFailureRate(),
            remoteOptions.getRemoteMinFailCountToComputeFailureRate(),
            /* recoveryDelayMillis= */ 0,
            mock(ScheduledExecutorService.class));

    // Make a number of failure calls less than minFailToComputeFailure.
    for (int index = 0; index < minFailToComputeFailure - 1; index++) {
      failureCircuitBreaker.recordFailure(State.ACCEPT_CALLS);
    }
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.ACCEPT_CALLS);

    // We don't run the scheduled tasks, simulating being within the window.
    failureCircuitBreaker.recordFailure(State.ACCEPT_CALLS);
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.REJECT_CALLS);
  }

  @Test
  public void testFailureDetails_reportsStats() {
    final int failureRateThreshold = 10;
    final int windowInterval = 100;
    FailureCircuitBreaker failureCircuitBreaker =
        new FailureCircuitBreaker(
            failureRateThreshold,
            windowInterval,
            /* minCallCountToComputeFailureRate= */ 0,
            /* minFailCountToComputeFailureRate= */ 0,
            /* recoveryDelayMillis= */ 0,
            mock(ScheduledExecutorService.class));

    failureCircuitBreaker.recordSuccess(State.ACCEPT_CALLS);
    failureCircuitBreaker.recordFailure(State.ACCEPT_CALLS);
    failureCircuitBreaker.recordFailure(State.ACCEPT_CALLS);
    failureCircuitBreaker.recordFailure(State.ACCEPT_CALLS);

    String details = failureCircuitBreaker.failureDetails();
    assertThat(details).contains("3 out of 4 remote calls failed");
    assertThat(details).contains("75.00%");
    assertThat(details).contains("the last 100ms");
    assertThat(details).contains("10% failure rate threshold");
  }

  @Test
  public void testRecovery_disabledByDefault_breakerStaysOpen() {
    List<Runnable> trialTasks = Collections.synchronizedList(new ArrayList<>());
    List<Runnable> windowTasks = Collections.synchronizedList(new ArrayList<>());
    FailureCircuitBreaker breaker =
        new FailureCircuitBreaker(
            /* failureRateThreshold= */ 0,
            /* slidingWindowSize= */ 100,
            /* minCallCountToComputeFailureRate= */ 1,
            /* minFailCountToComputeFailureRate= */ 1,
            /* recoveryDelayMillis= */ 0,
            newSchedulerCapturing(/* recoveryDelayMillis= */ 0, trialTasks, windowTasks));

    // A single failure trips the breaker (0% threshold, min counts of 1).
    breaker.recordFailure(State.ACCEPT_CALLS);
    assertThat(breaker.state()).isEqualTo(State.REJECT_CALLS);

    // Recovery is disabled: no probe is ever scheduled and the breaker never leaves REJECT_CALLS.
    assertThat(trialTasks).isEmpty();
    assertThat(breaker.state()).isEqualTo(State.REJECT_CALLS);
  }

  @Test
  public void testRecovery_trialProbeSuccessClosesBreaker() {
    List<Runnable> trialTasks = Collections.synchronizedList(new ArrayList<>());
    List<Runnable> windowTasks = Collections.synchronizedList(new ArrayList<>());
    FailureCircuitBreaker breaker =
        new FailureCircuitBreaker(
            /* failureRateThreshold= */ 0,
            /* slidingWindowSize= */ 100,
            /* minCallCountToComputeFailureRate= */ 1,
            /* minFailCountToComputeFailureRate= */ 1,
            /* recoveryDelayMillis= */ 1000,
            newSchedulerCapturing(/* recoveryDelayMillis= */ 1000, trialTasks, windowTasks));

    // Trip the breaker; a single recovery probe is scheduled for after the recovery delay.
    breaker.recordFailure(State.ACCEPT_CALLS);
    assertThat(breaker.state()).isEqualTo(State.REJECT_CALLS);
    assertThat(trialTasks).hasSize(1);

    // Before the delay elapses the breaker is still open.
    assertThat(breaker.state()).isEqualTo(State.REJECT_CALLS);

    // Simulate the recovery delay elapsing.
    trialTasks.get(0).run();

    // Exactly one caller gets the trial probe; concurrent callers keep seeing REJECT_CALLS.
    assertThat(breaker.state()).isEqualTo(State.TRIAL_CALL);
    assertThat(breaker.state()).isEqualTo(State.REJECT_CALLS);

    // The probe succeeds: the breaker closes and normal calls resume.
    breaker.recordSuccess(State.TRIAL_CALL);
    assertThat(breaker.state()).isEqualTo(State.ACCEPT_CALLS);
  }

  @Test
  public void testRecovery_trialProbeFailureReopensAndReschedules() {
    List<Runnable> trialTasks = Collections.synchronizedList(new ArrayList<>());
    List<Runnable> windowTasks = Collections.synchronizedList(new ArrayList<>());
    FailureCircuitBreaker breaker =
        new FailureCircuitBreaker(
            /* failureRateThreshold= */ 0,
            /* slidingWindowSize= */ 100,
            /* minCallCountToComputeFailureRate= */ 1,
            /* minFailCountToComputeFailureRate= */ 1,
            /* recoveryDelayMillis= */ 1000,
            newSchedulerCapturing(/* recoveryDelayMillis= */ 1000, trialTasks, windowTasks));

    // Trip the breaker and let the first recovery delay elapse.
    breaker.recordFailure(State.ACCEPT_CALLS);
    assertThat(trialTasks).hasSize(1);
    trialTasks.get(0).run();
    assertThat(breaker.state()).isEqualTo(State.TRIAL_CALL);

    // The probe fails: the breaker re-opens and schedules another probe.
    breaker.recordFailure(State.TRIAL_CALL);
    assertThat(breaker.state()).isEqualTo(State.REJECT_CALLS);
    assertThat(trialTasks).hasSize(2);

    // The next recovery delay elapses and another single probe is offered.
    trialTasks.get(1).run();
    assertThat(breaker.state()).isEqualTo(State.TRIAL_CALL);
  }

  @Test
  public void testRecovery_slowPreTripCallCannotHijackProbe() {
    List<Runnable> trialTasks = Collections.synchronizedList(new ArrayList<>());
    List<Runnable> windowTasks = Collections.synchronizedList(new ArrayList<>());
    FailureCircuitBreaker breaker =
        new FailureCircuitBreaker(
            /* failureRateThreshold= */ 0,
            /* slidingWindowSize= */ 100,
            /* minCallCountToComputeFailureRate= */ 1,
            /* minFailCountToComputeFailureRate= */ 1,
            /* recoveryDelayMillis= */ 1000,
            newSchedulerCapturing(/* recoveryDelayMillis= */ 1000, trialTasks, windowTasks));

    // Trip, let the recovery delay elapse, and hand out the probe.
    breaker.recordFailure(State.ACCEPT_CALLS);
    trialTasks.get(0).run();
    assertThat(breaker.state()).isEqualTo(State.TRIAL_CALL);

    // A call that started before the breaker tripped now completes with a failure. It observed
    // ACCEPT_CALLS, so it must not consume the in-flight probe, reopen the breaker, or schedule a
    // new probe.
    breaker.recordFailure(State.ACCEPT_CALLS);
    assertThat(breaker.state()).isEqualTo(State.REJECT_CALLS);
    assertThat(trialTasks).hasSize(1);

    // The actual probe then succeeds and still closes the breaker.
    breaker.recordSuccess(State.TRIAL_CALL);
    assertThat(breaker.state()).isEqualTo(State.ACCEPT_CALLS);
  }

  @Test
  public void testRecovery_successfulProbeInvalidatesStaleWindowDecrements() {
    List<Runnable> trialTasks = Collections.synchronizedList(new ArrayList<>());
    List<Runnable> windowTasks = Collections.synchronizedList(new ArrayList<>());
    FailureCircuitBreaker breaker =
        new FailureCircuitBreaker(
            /* failureRateThreshold= */ 0,
            /* slidingWindowSize= */ 100,
            /* minCallCountToComputeFailureRate= */ 2,
            /* minFailCountToComputeFailureRate= */ 2,
            /* recoveryDelayMillis= */ 1000,
            newSchedulerCapturing(/* recoveryDelayMillis= */ 1000, trialTasks, windowTasks));

    // Two failures trip the breaker (100% rate once the min counts of 2 are met); each schedules a
    // window-decrement task.
    breaker.recordFailure(State.ACCEPT_CALLS);
    breaker.recordFailure(State.ACCEPT_CALLS);
    assertThat(breaker.state()).isEqualTo(State.REJECT_CALLS);
    assertThat(windowTasks).hasSize(2);

    // A probe succeeds and closes the breaker, resetting the counters (and bumping the generation).
    trialTasks.get(0).run();
    assertThat(breaker.state()).isEqualTo(State.TRIAL_CALL);
    breaker.recordSuccess(State.TRIAL_CALL);
    assertThat(breaker.state()).isEqualTo(State.ACCEPT_CALLS);

    // A fresh failure is recorded after the reset.
    breaker.recordFailure(State.ACCEPT_CALLS);

    // The two stale (pre-reset) window decrements now fire. They must not erase the fresh failure.
    windowTasks.get(0).run();
    windowTasks.get(1).run();

    // One more failure reaches the min count of 2 at a 100% rate, so the breaker trips again --
    // which
    // only holds if the fresh failure survived the stale decrements.
    breaker.recordFailure(State.ACCEPT_CALLS);
    assertThat(breaker.state()).isEqualTo(State.REJECT_CALLS);
  }
}
