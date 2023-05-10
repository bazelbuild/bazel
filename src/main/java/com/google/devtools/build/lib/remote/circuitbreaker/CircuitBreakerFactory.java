package com.google.devtools.build.lib.remote.circuitbreaker;

import com.google.devtools.build.lib.remote.Retrier;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.options.RemoteOptions;

/**
 * Factory for {@link Retrier.CircuitBreaker}
 */
public class CircuitBreakerFactory {

  public static final ImmutableSet<Class<? extends Exception>> DEFAULT_IGNORED_ERRORS =
      ImmutableSet.of(CacheNotFoundException.class);

  private CircuitBreakerFactory() {
  }

  /**
   * Creates the instance of the {@link Retrier.CircuitBreaker} as per the strategy defined in {@link RemoteOptions}.
   * In case of undefined strategy defaults to {@link Retrier.ALLOW_ALL_CALLS} implementation.
   *
   * @param remoteOptions The configuration for the CircuitBreaker implementation.
   * @return an instance of CircuitBreaker.
   */
  public static Retrier.CircuitBreaker createCircuitBreaker(final RemoteOptions remoteOptions) {
    if (remoteOptions.circuitBreakerStrategy == RemoteOptions.CircuitBreakerStrategy.FAILURE) {
      return new FailureCircuitBreaker(remoteOptions.remoteFailureThreshold,
          (int) remoteOptions.remoteFailureWindowSize.toMillis());
    }
    return Retrier.ALLOW_ALL_CALLS;
  }
}