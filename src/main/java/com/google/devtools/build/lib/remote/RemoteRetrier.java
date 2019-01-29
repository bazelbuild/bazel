// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import io.grpc.Status;
import io.grpc.StatusException;
import io.grpc.StatusRuntimeException;
import java.io.IOException;
import java.time.Duration;
import java.util.concurrent.Callable;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Predicate;
import java.util.function.Supplier;

/** Specific retry logic for remote execution/caching. */
public class RemoteRetrier extends Retrier {

  public static final Predicate<? super Exception> RETRIABLE_GRPC_ERRORS =
      e -> {
        if (!(e instanceof StatusException) && !(e instanceof StatusRuntimeException)) {
          return false;
        }
        Status s = Status.fromThrowable(e);
        switch (s.getCode()) {
          case CANCELLED:
            return !Thread.currentThread().isInterrupted();
          case UNKNOWN:
          case DEADLINE_EXCEEDED:
          case ABORTED:
          case INTERNAL:
          case UNAVAILABLE:
          case UNAUTHENTICATED:
          case RESOURCE_EXHAUSTED:
            return true;
          default:
            return false;
        }
      };

  public static final Predicate<? super Exception> RETRIABLE_GRPC_EXEC_ERRORS =
      e -> {
        if (RETRIABLE_GRPC_ERRORS.test(e)) {
          return true;
        }
        return RemoteRetrierUtils.causedByStatus(e, Status.Code.NOT_FOUND);
      };

  public RemoteRetrier(
      RemoteOptions options,
      Predicate<? super Exception> shouldRetry,
      ListeningScheduledExecutorService retryScheduler,
      CircuitBreaker circuitBreaker) {
    this(
        options.experimentalRemoteRetry
            ? () -> new ExponentialBackoff(options)
            : () -> RETRIES_DISABLED,
        shouldRetry,
        retryScheduler,
        circuitBreaker);
  }

  public RemoteRetrier(
      Supplier<Backoff> backoff,
      Predicate<? super Exception> shouldRetry,
      ListeningScheduledExecutorService retryScheduler,
      CircuitBreaker circuitBreaker) {
    super(backoff, shouldRetry, retryScheduler, circuitBreaker);
  }

  @VisibleForTesting
  RemoteRetrier(
      Supplier<Backoff> backoff,
      Predicate<? super Exception> shouldRetry,
      ListeningScheduledExecutorService retryScheduler,
      CircuitBreaker circuitBreaker,
      Sleeper sleeper) {
    super(backoff, shouldRetry, retryScheduler, circuitBreaker, sleeper);
  }

  /**
   * Execute a callable with retries. {@link IOException} and {@link InterruptedException} are
   * propagated directly to the caller. All other exceptions are wrapped in {@link RuntimeError}.
   */
  @Override
  public <T> T execute(Callable<T> call) throws IOException, InterruptedException {
    try {
      return super.execute(call);
    } catch (Exception e) {
      Throwables.propagateIfInstanceOf(e, IOException.class);
      Throwables.propagateIfInstanceOf(e, InterruptedException.class);
      throw Throwables.propagate(e);
    }
  }

  static class ExponentialBackoff implements Retrier.Backoff {

    private final long maxMillis;
    private long nextDelayMillis;
    private int attempts = 0;
    private final double multiplier;
    private final double jitter;
    private final int maxAttempts;

    /**
     * Creates a Backoff supplier for an optionally jittered exponential backoff. The supplier is
     * ThreadSafe (non-synchronized calls to get() are fine), but the returned Backoff is not.
     *
     * @param initial The initial backoff duration.
     * @param max The maximum backoff duration.
     * @param multiplier The amount the backoff should increase in each iteration. Must be >1.
     * @param jitter The amount the backoff should be randomly varied (0-1), with 0 providing no
     *     jitter, and 1 providing a duration that is 0-200% of the non-jittered duration.
     * @param maxAttempts Maximal times to attempt a retry 0 means no retries.
     */
    ExponentialBackoff(Duration initial, Duration max, double multiplier, double jitter,
        int maxAttempts) {
      Preconditions.checkArgument(multiplier > 1, "multipler must be > 1");
      Preconditions.checkArgument(jitter >= 0 && jitter <= 1, "jitter must be in the range (0, 1)");
      Preconditions.checkArgument(maxAttempts >= 0, "maxAttempts must be >= 0");
      nextDelayMillis = initial.toMillis();
      maxMillis = max.toMillis();
      this.multiplier = multiplier;
      this.jitter = jitter;
      this.maxAttempts = maxAttempts;
    }

    ExponentialBackoff(RemoteOptions options) {
      this(Duration.ofMillis(options.experimentalRemoteRetryStartDelayMillis),
          Duration.ofMillis(options.experimentalRemoteRetryMaxDelayMillis),
          options.experimentalRemoteRetryMultiplier,
          options.experimentalRemoteRetryJitter,
          options.experimentalRemoteRetryMaxAttempts);
    }

    @Override
    public long nextDelayMillis() {
      if (attempts == maxAttempts) {
        return -1;
      }
      attempts++;
      double jitterRatio = jitter * (ThreadLocalRandom.current().nextDouble(2.0) - 1);
      long result = (long) (nextDelayMillis * (1 + jitterRatio));
      // Advance current by the non-jittered result.
      nextDelayMillis = (long) (nextDelayMillis * multiplier);
      if (nextDelayMillis > maxMillis) {
        nextDelayMillis = maxMillis;
      }
      return result;
    }

    @Override
    public int getRetryAttempts() {
      return attempts;
    }
  }
}
