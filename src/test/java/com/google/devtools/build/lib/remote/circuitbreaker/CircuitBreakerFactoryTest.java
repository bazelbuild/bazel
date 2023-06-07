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
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.options.RemoteOptions.CircuitBreakerStrategy;
import com.google.devtools.common.options.Options;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CircuitBreakerFactory}. */
@RunWith(JUnit4.class)
public class CircuitBreakerFactoryTest {
  @Test
  public void testCreateCircuitBreaker_failureStrategy() {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    remoteOptions.circuitBreakerStrategy = CircuitBreakerStrategy.FAILURE;

    assertThat(CircuitBreakerFactory.createCircuitBreaker(remoteOptions))
        .isInstanceOf(FailureCircuitBreaker.class);
  }

  @Test
  public void testCreateCircuitBreaker_nullStrategy() {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    assertThat(CircuitBreakerFactory.createCircuitBreaker(remoteOptions))
        .isEqualTo(Retrier.ALLOW_ALL_CALLS);
  }
}
