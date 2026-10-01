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
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;

/** Factory for {@link Retrier.CircuitBreaker} */
public class CircuitBreakerFactory {

  private CircuitBreakerFactory() {}

  /**
   * Creates the instance of the {@link Retrier.CircuitBreaker} as per the strategy defined in
   * {@link RemoteOptions}. In case of undefined strategy defaults to {@link
   * Retrier.ALLOW_ALL_CALLS} implementation.
   *
   * @param remoteOptions The configuration for the CircuitBreaker implementation.
   * @return an instance of CircuitBreaker.
   */
  public static Retrier.CircuitBreaker createCircuitBreaker(final RemoteOptions remoteOptions) {
    if (remoteOptions.getCircuitBreakerStrategy() == RemoteOptions.CircuitBreakerStrategy.FAILURE) {
      int slidingWindowMillis = (int) remoteOptions.getRemoteFailureWindowInterval().toMillis();
      int recoveryDelayMillis =
          (int) remoteOptions.getRemoteCircuitBreakerRecoveryDelay().toMillis();
      boolean needsScheduler = slidingWindowMillis > 0 || recoveryDelayMillis > 0;
      return new FailureCircuitBreaker(
          remoteOptions.getRemoteFailureRateThreshold(),
          slidingWindowMillis,
          remoteOptions.getRemoteMinCallCountToComputeFailureRate(),
          remoteOptions.getRemoteMinFailCountToComputeFailureRate(),
          recoveryDelayMillis,
          needsScheduler ? newScheduler() : null);
    }
    return Retrier.ALLOW_ALL_CALLS;
  }

  /**
   * Returns a single-threaded scheduler for the breaker's window-decrement and recovery-probe
   * tasks. The worker is a daemon so a per-build breaker never keeps the server alive; it is a
   * platform thread because these are trivial, non-blocking tasks driven by a delay queue, which
   * virtual threads do not suit.
   */
  private static ScheduledExecutorService newScheduler() {
    return Executors.newSingleThreadScheduledExecutor(
        Thread.ofPlatform().name("remote-circuit-breaker").daemon().factory());
  }
}
