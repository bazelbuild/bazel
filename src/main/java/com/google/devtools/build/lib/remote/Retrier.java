// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.base.Supplier;
import com.google.common.base.Throwables;
import com.google.devtools.build.lib.util.Preconditions;
import io.grpc.Status;
import io.grpc.StatusException;
import io.grpc.StatusRuntimeException;
import java.io.IOException;
import java.time.Duration;
import java.util.concurrent.Callable;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

/**
 * Supports execution with retries on particular gRPC Statuses. The retrier is ThreadSafe.
 *
 * <p>Example usage: The simple use-case is to call retrier.execute, e.g:
 *
 * <pre>
 * foo = retrier.execute(
 *     new Callable<Foo>() {
 *       @Override
 *       public Foo call() {
 *         return grpcStub.getFoo(fooRequest);
 *       }
 *     });
 * </pre>
 */
public class Retrier {
  /** Wraps around a StatusRuntimeException to make it pass through a single layer of retries. */
  public static class PassThroughException extends Exception {
    public PassThroughException(StatusRuntimeException e) {
      super(e);
    }
  }

  /**
   * Backoff is a stateful object providing a sequence of durations that are used to time delays
   * between retries. It is not ThreadSafe. The reason that Backoff needs to be stateful, rather
   * than a static map of attempt number to delay, is to enable using the retrier via the manual
   * calling isRetriable and nextDelayMillis manually (see ByteStreamUploader example).
   */
  public interface Backoff {

    /** Indicates that no more retries should be made for use in {@link #nextDelayMillis()}. */
    static final long STOP = -1L;

    /** Returns the next delay in milliseconds, or < 0 if we should not continue retrying. */
    long nextDelayMillis();

    /**
     * Returns the number of calls to {@link #nextDelayMillis()} thus far, not counting any calls
     * that returned STOP.
     */
    int getRetryAttempts();

    /**
     * Creates a Backoff supplier for a Backoff which does not support any retries. Both the
     * Supplier and the Backoff are stateless and thread-safe.
     */
    static final Supplier<Backoff> NO_RETRIES =
        () ->
            new Backoff() {
              @Override
              public long nextDelayMillis() {
                return STOP;
              }

              @Override
              public int getRetryAttempts() {
                return 0;
              }
            };

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
    static Supplier<Backoff> exponential(
        Duration initial, Duration max, double multiplier, double jitter, int maxAttempts) {
      Preconditions.checkArgument(multiplier > 1, "multipler must be > 1");
      Preconditions.checkArgument(jitter >= 0 && jitter <= 1, "jitter must be in the range (0, 1)");
      Preconditions.checkArgument(maxAttempts >= 0, "maxAttempts must be >= 0");
      return () ->
          new Backoff() {
            private final long maxMillis = max.toMillis();
            private long nextDelayMillis = initial.toMillis();
            private int attempts = 0;

            @Override
            public long nextDelayMillis() {
              if (attempts == maxAttempts) {
                return STOP;
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
          };
    }
  }

  public static final Predicate<Status> DEFAULT_IS_RETRIABLE =
      st -> {
        switch (st.getCode()) {
          case CANCELLED:
            return !Thread.currentThread().isInterrupted();
          case UNKNOWN:
          case DEADLINE_EXCEEDED:
          case ABORTED:
          case INTERNAL:
          case UNAVAILABLE:
          case UNAUTHENTICATED:
            return true;
          default:
            return false;
        }
      };

  public static final Predicate<Status> RETRY_ALL = Predicates.alwaysTrue();
  public static final Predicate<Status> RETRY_NONE = Predicates.alwaysFalse();
  public static final Retrier NO_RETRIES = new Retrier(Backoff.NO_RETRIES, RETRY_NONE);

  private final Supplier<Backoff> backoffSupplier;
  private final Predicate<Status> isRetriable;

  @VisibleForTesting
  Retrier(Supplier<Backoff> backoffSupplier, Predicate<Status> isRetriable) {
    this.backoffSupplier = backoffSupplier;
    this.isRetriable = isRetriable;
  }

  public Retrier(RemoteOptions options) {
    this(
        options.experimentalRemoteRetry
            ? Backoff.exponential(
                Duration.ofMillis(options.experimentalRemoteRetryStartDelayMillis),
                Duration.ofMillis(options.experimentalRemoteRetryMaxDelayMillis),
                options.experimentalRemoteRetryMultiplier,
                options.experimentalRemoteRetryJitter,
                options.experimentalRemoteRetryMaxAttempts)
            : Backoff.NO_RETRIES,
        DEFAULT_IS_RETRIABLE);
  }

  /**
   * Returns {@code true} if the {@link Status} is retriable.
   */
  public boolean isRetriable(Status s) {
    return isRetriable.apply(s);
  }

  /**
   * Executes the given callable in a loop, retrying on retryable errors, as defined by the current
   * backoff/retry policy. Will raise the last encountered retriable error, or the first
   * non-retriable error.
   *
   * <p>This method never throws {@link StatusRuntimeException} even if the passed-in Callable does.
   *
   * @param c The callable to execute.
   */
  public <T> T execute(Callable<T> c) throws InterruptedException, IOException {
    Backoff backoff = backoffSupplier.get();
    while (true) {
      try {
        return c.call();
      } catch (PassThroughException e) {
        throw (StatusRuntimeException) e.getCause();
      } catch (RetryException e) {
        throw e;  // Nested retries are always pass-through.
      } catch (StatusException | StatusRuntimeException e) {
        Status st = Status.fromThrowable(e);
        long delay = backoff.nextDelayMillis();
        if (st.getCode() == Status.Code.CANCELLED && Thread.currentThread().isInterrupted()) {
          Thread.currentThread().interrupt();
          throw new InterruptedException();
        }
        if (delay < 0 || !isRetriable.apply(st)) {
          throw new RetryException(st.asRuntimeException(), backoff.getRetryAttempts());
        }
        sleep(delay);
      } catch (Exception e) {
        // Generic catch because Callable is declared to throw Exception, we rethrow any unchecked
        // exception as well as any exception we declared above.
        Throwables.throwIfUnchecked(e);
        Throwables.throwIfInstanceOf(e, IOException.class);
        Throwables.throwIfInstanceOf(e, InterruptedException.class);
        throw new RetryException(e, backoff.getRetryAttempts());
      }
    }
  }

  @VisibleForTesting
  void sleep(long timeMillis) throws InterruptedException {
    Preconditions.checkArgument(
        timeMillis >= 0L, "timeMillis must not be negative: %s", timeMillis);
    TimeUnit.MILLISECONDS.sleep(timeMillis);
  }

  public Backoff newBackoff() {
    return backoffSupplier.get();
  }
}
