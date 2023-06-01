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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.remote.Retrier;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The {@link FailureCircuitBreaker} implementation of the {@link Retrier.CircuitBreaker} prevents
 * further calls to a remote cache once the number of failures within a given window exceeds a
 * specified threshold for a build. In the context of Bazel, a new instance of {@link
 * Retrier.CircuitBreaker} is created for each build. Therefore, if the circuit breaker trips during
 * a build, the remote cache will be disabled for that build. However, it will be enabled again for
 * the next build as a new instance of {@link Retrier.CircuitBreaker} will be created.
 */
public class FailureCircuitBreaker implements Retrier.CircuitBreaker {

  private State state;
  private final AtomicInteger failures;
  private final int failureThreshold;
  private final int slidingWindowSize;
  private final ScheduledExecutorService scheduledExecutor;
  private final ImmutableSet<Class<? extends Exception>> ignoredErrors;

  /**
   * Creates a {@link FailureCircuitBreaker}.
   *
   * @param failureThreshold is used to set the number of failures required to trip the circuit
   *     breaker in given time window.
   * @param slidingWindowSize the size of the sliding window in milliseconds to calculate the number
   *     of failures.
   */
  public FailureCircuitBreaker(int failureThreshold, int slidingWindowSize) {
    this.failureThreshold = failureThreshold;
    this.failures = new AtomicInteger(0);
    this.slidingWindowSize = slidingWindowSize;
    this.state = State.ACCEPT_CALLS;
    this.scheduledExecutor =
        slidingWindowSize > 0 ? Executors.newSingleThreadScheduledExecutor() : null;
    this.ignoredErrors = CircuitBreakerFactory.DEFAULT_IGNORED_ERRORS;
  }

  @Override
  public State state() {
    return this.state;
  }

  @Override
  public void recordFailure(Exception e) {
    if (!ignoredErrors.contains(e.getClass())) {
      int failureCount = failures.incrementAndGet();
      if (slidingWindowSize > 0) {
        var unused =
            scheduledExecutor.schedule(
                failures::decrementAndGet, slidingWindowSize, TimeUnit.MILLISECONDS);
      }
      // Since the state can only be changed to the open state, synchronization is not required.
      if (failureCount > this.failureThreshold) {
        this.state = State.REJECT_CALLS;
      }
    }
  }

  @Override
  public void recordSuccess() {
    // do nothing, implement if we need to set threshold on failure rate instead of count.
  }
}
