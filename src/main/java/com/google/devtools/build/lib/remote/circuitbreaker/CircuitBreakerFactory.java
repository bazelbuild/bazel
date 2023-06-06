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
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import io.grpc.Status;
import java.util.function.Predicate;

import static com.google.devtools.build.lib.remote.RemoteRetrier.fromException;


/** Factory for {@link Retrier.CircuitBreaker} */
public class CircuitBreakerFactory {
  public static final Predicate<? super Exception> DEFAULT_IGNORED_ERRORS =
      e -> {
        Status s = fromException(e);
        if (s == null) {
          return e.getClass() == CacheNotFoundException.class;
        }
        switch (s.getCode()) {
          case NOT_FOUND:
          case OUT_OF_RANGE:
            System.out.println("out of range");
            return true;
          default:
            return false;
        }
      };

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
    if (remoteOptions.circuitBreakerStrategy == RemoteOptions.CircuitBreakerStrategy.FAILURE) {
      return new FailureCircuitBreaker(
          remoteOptions.remoteFailureThreshold,
          (int) remoteOptions.remoteFailureWindowInterval.toMillis());
    }
    return Retrier.ALLOW_ALL_CALLS;
  }
}
