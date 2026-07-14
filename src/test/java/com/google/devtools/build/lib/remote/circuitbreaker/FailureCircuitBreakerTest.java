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
import java.util.concurrent.Callable;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class FailureCircuitBreakerTest {

  private final RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);

  @Test
  // Suppress unchecked warnings because any(Callable.class) uses a raw type,
  // which causes Javac to fail with -Werror.
  @SuppressWarnings("unchecked")
  public void testRecordFailure_circuitTrips() throws InterruptedException {
    final int failureRateThreshold = 10;
    final int windowInterval = 100;
    ScheduledExecutorService mockScheduler = mock(ScheduledExecutorService.class);
    List<Runnable> capturedRunnables = Collections.synchronizedList(new ArrayList<>());

    // Stub both schedule overloads to capture the scheduled tasks.
    // This allows us to simulate window expiration by running them manually.
    // We need to stub Callable because method references like failures::decrementAndGet
    // return a value and can be matched to Callable by the compiler.
    when(mockScheduler.schedule(any(Runnable.class), anyLong(), any(TimeUnit.class)))
        .thenAnswer(
            invocation -> {
              capturedRunnables.add(invocation.getArgument(0));
              return null;
            });
    when(mockScheduler.schedule(any(Callable.class), anyLong(), any(TimeUnit.class)))
        .thenAnswer(
            invocation -> {
              Callable<?> callable = invocation.getArgument(0);
              capturedRunnables.add(
                  () -> {
                    try {
                      callable.call();
                    } catch (Exception e) {
                      throw new RuntimeException(e);
                    }
                  });
              return null;
            });
    FailureCircuitBreaker failureCircuitBreaker =
        new FailureCircuitBreaker(
            failureRateThreshold,
            windowInterval,
            remoteOptions.getRemoteMinCallCountToComputeFailureRate(),
            remoteOptions.getRemoteMinFailCountToComputeFailureRate(),
            /* recoveryDelayMillis= */ 0,
            mockScheduler);

    List<Runnable> listOfSuccessAndFailureCalls = new ArrayList<>();
    for (int index = 0; index < failureRateThreshold; index++) {
      listOfSuccessAndFailureCalls.add(failureCircuitBreaker::recordFailure);
    }

    for (int index = 0; index < failureRateThreshold * 9; index++) {
      listOfSuccessAndFailureCalls.add(failureCircuitBreaker::recordSuccess);
    }

    Collections.shuffle(listOfSuccessAndFailureCalls);

    // make calls equals to threshold number of not ignored failure calls in parallel.
    listOfSuccessAndFailureCalls.stream().parallel().forEach(Runnable::run);
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.ACCEPT_CALLS);

    int expectedCalls = failureRateThreshold * 10;
    // Run all captured runnables to simulate window expiration.
    assertThat(capturedRunnables).hasSize(expectedCalls);
    capturedRunnables.forEach(Runnable::run);
    capturedRunnables.clear(); // Clear for the next round

    // make calls equals to threshold number of not ignored failure calls in parallel.
    listOfSuccessAndFailureCalls.stream().parallel().forEach(Runnable::run);
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.ACCEPT_CALLS);

    // We don't run the new scheduled tasks, simulating being within the window.
    failureCircuitBreaker.recordFailure();
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.REJECT_CALLS);
  }

  @Test
  public void testRecordFailure_minCallCriteriaNotMet() throws InterruptedException {
    final int failureRateThreshold = 0;
    final int windowInterval = 100;
    final int minCallToComputeFailure = remoteOptions.getRemoteMinCallCountToComputeFailureRate();
    ScheduledExecutorService mockScheduler = mock(ScheduledExecutorService.class);
    FailureCircuitBreaker failureCircuitBreaker =
        new FailureCircuitBreaker(
            failureRateThreshold,
            windowInterval,
            remoteOptions.getRemoteMinCallCountToComputeFailureRate(),
            remoteOptions.getRemoteMinFailCountToComputeFailureRate(),
            /* recoveryDelayMillis= */ 0,
            mockScheduler);

    // make success calls, failure call and number of total calls less than
    // minCallToComputeFailure.
    for (int index = 0; index < minCallToComputeFailure - 2; index++) {
      failureCircuitBreaker.recordSuccess();
    }
    failureCircuitBreaker.recordFailure();
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.ACCEPT_CALLS);

    // We don't run the scheduled tasks, simulating being within the window.
    failureCircuitBreaker.recordFailure();
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.REJECT_CALLS);
  }

  @Test
  public void testRecordFailure_minFailCriteriaNotMet() throws InterruptedException {
    final int failureRateThreshold = 10;
    final int windowInterval = 100;
    final int minFailToComputeFailure = remoteOptions.getRemoteMinFailCountToComputeFailureRate();
    ScheduledExecutorService mockScheduler = mock(ScheduledExecutorService.class);
    FailureCircuitBreaker failureCircuitBreaker =
        new FailureCircuitBreaker(
            failureRateThreshold,
            windowInterval,
            remoteOptions.getRemoteMinCallCountToComputeFailureRate(),
            remoteOptions.getRemoteMinFailCountToComputeFailureRate(),
            /* recoveryDelayMillis= */ 0,
            mockScheduler);

    // make number of failure calls less than minFailToComputeFailure.
    for (int index = 0; index < minFailToComputeFailure - 1; index++) {
      failureCircuitBreaker.recordFailure();
    }
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.ACCEPT_CALLS);

    // We don't run the scheduled tasks, simulating being within the window.
    failureCircuitBreaker.recordFailure();
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.REJECT_CALLS);
  }

  @Test
  public void testFailureDetails_reportsStats() {
    final int failureRateThreshold = 10;
    final int windowInterval = 100;
    ScheduledExecutorService mockScheduler = mock(ScheduledExecutorService.class);
    FailureCircuitBreaker failureCircuitBreaker =
        new FailureCircuitBreaker(
            failureRateThreshold,
            windowInterval,
            /* minCallCountToComputeFailureRate= */ 0,
            /* minFailCountToComputeFailureRate= */ 0,
            /* recoveryDelayMillis= */ 0,
            mockScheduler);

    failureCircuitBreaker.recordSuccess();
    failureCircuitBreaker.recordFailure();
    failureCircuitBreaker.recordFailure();
    failureCircuitBreaker.recordFailure();

    String details = failureCircuitBreaker.failureDetails();
    assertThat(details).contains("3 out of 4 remote calls failed");
    assertThat(details).contains("75.00%");
    assertThat(details).contains("the last 100ms");
    assertThat(details).contains("10% failure rate threshold");
  }

  /**
   * Returns a mock scheduler that records the tasks the breaker schedules, splitting them by delay:
   * recovery-probe tasks (scheduled at {@code recoveryDelayMillis}) go into {@code trialTasks} and
   * sliding-window decrement tasks (scheduled at the window size) go into {@code windowTasks}. A test
   * can then run them manually to simulate the recovery delay or the window elapsing.
   */
  private static ScheduledExecutorService newRecoveryScheduler(
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
  public void testRecovery_disabledByDefault_breakerStaysOpen() {
    List<Runnable> trialTasks = Collections.synchronizedList(new ArrayList<>());
    List<Runnable> windowTasks = Collections.synchronizedList(new ArrayList<>());
    FailureCircuitBreaker failureCircuitBreaker =
        new FailureCircuitBreaker(
            /* failureRateThreshold= */ 0,
            /* slidingWindowSize= */ 100,
            /* minCallCountToComputeFailureRate= */ 1,
            /* minFailCountToComputeFailureRate= */ 1,
            /* recoveryDelayMillis= */ 0,
            newRecoveryScheduler(/* recoveryDelayMillis= */ 0, trialTasks, windowTasks));

    // A single failure trips the breaker (0% threshold, min counts of 1).
    failureCircuitBreaker.recordFailure();
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.REJECT_CALLS);

    // With recovery disabled, no trial probe is ever scheduled and the breaker stays open.
    assertThat(trialTasks).isEmpty();
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.REJECT_CALLS);
  }

  @Test
  public void testRecovery_trialProbeSuccessClosesBreaker() {
    List<Runnable> trialTasks = Collections.synchronizedList(new ArrayList<>());
    List<Runnable> windowTasks = Collections.synchronizedList(new ArrayList<>());
    FailureCircuitBreaker failureCircuitBreaker =
        new FailureCircuitBreaker(
            /* failureRateThreshold= */ 0,
            /* slidingWindowSize= */ 100,
            /* minCallCountToComputeFailureRate= */ 1,
            /* minFailCountToComputeFailureRate= */ 1,
            /* recoveryDelayMillis= */ 1000,
            newRecoveryScheduler(/* recoveryDelayMillis= */ 1000, trialTasks, windowTasks));

    // Trip the breaker; a single recovery probe is scheduled for after the recovery delay.
    failureCircuitBreaker.recordFailure();
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.REJECT_CALLS);
    assertThat(trialTasks).hasSize(1);

    // Before the delay elapses the breaker is still open.
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.REJECT_CALLS);

    // Simulate the recovery delay elapsing.
    trialTasks.get(0).run();

    // Exactly one caller gets the trial probe; concurrent callers keep seeing REJECT_CALLS.
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.TRIAL_CALL);
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.REJECT_CALLS);

    // The probe succeeds: the breaker closes and normal calls resume.
    failureCircuitBreaker.recordSuccess();
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.ACCEPT_CALLS);
  }

  @Test
  public void testRecovery_trialProbeFailureReopensAndReschedules() {
    List<Runnable> trialTasks = Collections.synchronizedList(new ArrayList<>());
    List<Runnable> windowTasks = Collections.synchronizedList(new ArrayList<>());
    FailureCircuitBreaker failureCircuitBreaker =
        new FailureCircuitBreaker(
            /* failureRateThreshold= */ 0,
            /* slidingWindowSize= */ 100,
            /* minCallCountToComputeFailureRate= */ 1,
            /* minFailCountToComputeFailureRate= */ 1,
            /* recoveryDelayMillis= */ 1000,
            newRecoveryScheduler(/* recoveryDelayMillis= */ 1000, trialTasks, windowTasks));

    // Trip the breaker and let the first recovery delay elapse.
    failureCircuitBreaker.recordFailure();
    assertThat(trialTasks).hasSize(1);
    trialTasks.get(0).run();
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.TRIAL_CALL);

    // The probe fails: the breaker re-opens and schedules another probe.
    failureCircuitBreaker.recordFailure();
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.REJECT_CALLS);
    assertThat(trialTasks).hasSize(2);

    // The next recovery delay elapses and another single probe is offered.
    trialTasks.get(1).run();
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.TRIAL_CALL);
  }

  @Test
  public void testRecovery_successfulProbeFloorsStaleWindowDecrements() {
    List<Runnable> trialTasks = Collections.synchronizedList(new ArrayList<>());
    List<Runnable> windowTasks = Collections.synchronizedList(new ArrayList<>());
    FailureCircuitBreaker failureCircuitBreaker =
        new FailureCircuitBreaker(
            /* failureRateThreshold= */ 0,
            /* slidingWindowSize= */ 100,
            /* minCallCountToComputeFailureRate= */ 1,
            /* minFailCountToComputeFailureRate= */ 1,
            /* recoveryDelayMillis= */ 1000,
            newRecoveryScheduler(/* recoveryDelayMillis= */ 1000, trialTasks, windowTasks));

    // Accumulate several failures; the first trips the breaker and each schedules a window decrement.
    for (int index = 0; index < 5; index++) {
      failureCircuitBreaker.recordFailure();
    }
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.REJECT_CALLS);
    assertThat(windowTasks).hasSize(5);
    assertThat(trialTasks).hasSize(1);

    // A probe succeeds and closes the breaker, resetting the failure/success counters to zero.
    trialTasks.get(0).run();
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.TRIAL_CALL);
    failureCircuitBreaker.recordSuccess();
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.ACCEPT_CALLS);

    // The window decrements scheduled before the reset now fire; flooring keeps the counts at zero
    // instead of driving them negative...
    windowTasks.forEach(Runnable::run);

    // ...so a single fresh failure can still trip the breaker again.
    failureCircuitBreaker.recordFailure();
    assertThat(failureCircuitBreaker.state()).isEqualTo(State.REJECT_CALLS);
  }
}
