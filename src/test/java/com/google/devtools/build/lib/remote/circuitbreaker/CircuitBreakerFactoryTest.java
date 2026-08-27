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

import com.google.devtools.build.lib.remote.Retrier;
import com.google.devtools.build.lib.remote.Retrier.CircuitBreaker.State;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.options.RemoteOptions.CircuitBreakerStrategy;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CircuitBreakerFactory}. */
@RunWith(JUnit4.class)
public class CircuitBreakerFactoryTest {
  @Test
  public void testCreateCircuitBreaker_failureStrategy() {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    remoteOptions.setCircuitBreakerStrategy(CircuitBreakerStrategy.FAILURE);

    assertThat(CircuitBreakerFactory.createCircuitBreaker(remoteOptions))
        .isInstanceOf(FailureCircuitBreaker.class);
  }

  @Test
  public void testCreateCircuitBreaker_nullStrategy() {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    assertThat(CircuitBreakerFactory.createCircuitBreaker(remoteOptions))
        .isEqualTo(Retrier.ALLOW_ALL_CALLS);
  }

  @Test
  public void testCreateCircuitBreaker_minCallCountFromOptions() throws OptionsParsingException {
    RemoteOptions remoteOptions =
        Options.parse(
                RemoteOptions.class,
                "--experimental_circuit_breaker_strategy=failure",
                "--experimental_remote_failure_rate_threshold=10",
                "--experimental_remote_failure_window_interval=0",
                "--experimental_remote_min_call_count_to_compute_failure_rate=5",
                "--experimental_remote_min_fail_count_to_compute_failure_rate=1000")
            .getOptions();
    FailureCircuitBreaker circuitBreaker =
        (FailureCircuitBreaker) CircuitBreakerFactory.createCircuitBreaker(remoteOptions);

    // Four successes and one failure reach the configured min call count (5) with a 20% failure
    // rate, which exceeds the 10% threshold. This would not trip under the default min call count
    // of 100, proving the flag took effect. The high min fail count keeps the failure gate out of
    // the way, so tripping here also proves the two counts are not wired in the wrong order.
    for (int i = 0; i < 4; i++) {
      circuitBreaker.recordSuccess(State.ACCEPT_CALLS);
    }
    assertThat(circuitBreaker.state()).isEqualTo(State.ACCEPT_CALLS);
    circuitBreaker.recordFailure(State.ACCEPT_CALLS);
    assertThat(circuitBreaker.state()).isEqualTo(State.REJECT_CALLS);
  }

  @Test
  public void testCreateCircuitBreaker_minFailCountFromOptions() throws OptionsParsingException {
    RemoteOptions remoteOptions =
        Options.parse(
                RemoteOptions.class,
                "--experimental_circuit_breaker_strategy=failure",
                "--experimental_remote_failure_rate_threshold=10",
                "--experimental_remote_failure_window_interval=0",
                "--experimental_remote_min_call_count_to_compute_failure_rate=1000",
                "--experimental_remote_min_fail_count_to_compute_failure_rate=3")
            .getOptions();
    FailureCircuitBreaker circuitBreaker =
        (FailureCircuitBreaker) CircuitBreakerFactory.createCircuitBreaker(remoteOptions);

    // The failure rate is computed once the configured min fail count (3) is reached, even though
    // the total call count is far below the configured min call count (1000). Under the default min
    // fail count of 12 the breaker would still be accepting calls here, proving the flag took
    // effect.
    circuitBreaker.recordFailure(State.ACCEPT_CALLS);
    circuitBreaker.recordFailure(State.ACCEPT_CALLS);
    assertThat(circuitBreaker.state()).isEqualTo(State.ACCEPT_CALLS);
    circuitBreaker.recordFailure(State.ACCEPT_CALLS);
    assertThat(circuitBreaker.state()).isEqualTo(State.REJECT_CALLS);
  }
}
