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
}
