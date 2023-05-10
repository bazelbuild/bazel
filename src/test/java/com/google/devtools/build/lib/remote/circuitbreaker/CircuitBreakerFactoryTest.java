package com.google.devtools.build.lib.remote.circuitbreaker;

import com.google.devtools.build.lib.remote.Retrier;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.options.RemoteOptions.CircuitBreakerStrategy;
import com.google.devtools.common.options.Options;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static com.google.common.truth.Truth.assertThat;

/** Tests for {@link CircuitBreakerFactory}. */
@RunWith(JUnit4.class)
public class CircuitBreakerFactoryTest {
    @Test
    public void testCreateCircuitBreaker_FailureStrategy() {
        RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
        remoteOptions.circuitBreakerStrategy = CircuitBreakerStrategy.FAILURE;

        assertThat(CircuitBreakerFactory.createCircuitBreaker(remoteOptions)).isInstanceOf(FailureCircuitBreaker.class);
    }

    @Test
    public void testCreateCircuitBreaker_NullStrategy() {
        RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
        assertThat(CircuitBreakerFactory.createCircuitBreaker(remoteOptions)).isEqualTo(Retrier.ALLOW_ALL_CALLS);
    }
}
