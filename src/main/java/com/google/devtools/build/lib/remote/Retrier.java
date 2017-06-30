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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.util.Preconditions;
import io.grpc.Status;
import io.grpc.StatusException;
import io.grpc.StatusRuntimeException;
import java.time.Duration;
import java.util.List;
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
 *
 * <p>When you need to retry multiple asynchronous calls, you can do:
 *
 * <pre>
 * Retrier.Backoff backoff = retrier.newBackoff();
 * List<Status> errors = Collections.synchronizedList(new ArrayList<Status>());
 * while (true) {
 *   CountDownLatch finishLatch = new CountDownLatch(items.size());
 *   for (Item item : items) {
 *     requestObserver = myStub.asyncCall(
 *         request,
 *         new StreamObserver<Response>() {
 *            ...
 *
 *            @Override
 *            public void onError(Throwable t) {
 *              // Need to handle non Status errors here!
 *              errors.add(Status.fromThrowable(t));
 *              finishLatch.countDown();
 *            }
 *            @Override
 *            public void onCompleted() {
 *               finishLatch.countDown();
 *            }
 *         });
 *     requestObserver.onNext(i1);
 *     requestObserver.onNext(i2);
 *     ...
 *     requestObserver.onCompleted();
 *   }
 *   finishLatch.await(someTime, TimeUnit.SECONDS);
 *   if (errors.isEmpty()) {
 *     return;
 *   }
 *   retrier.onFailures(backoff, errors);  // Sleep once for the whole batch of failures.
 *   items = failingItems;  // this needs to be collected from the observers as well.
 * }
 * </pre>
 *
 * <p>This retries the multiple calls in bulk. Another way to do it is retry each call separately as
 * it occurs:
 *
 * <pre>
 * class RetryingObserver extends StreamObserver<Response> {
 *   private final CountDownLatch finishLatch;
 *   private final Backoff backoff;
 *   private final AtomicReference<RuntimeException> exception;
 *
 *   RetryingObserver(
 *       CountDownLatch finishLatch, Backoff backoff, AtomicReference<RuntimeException> exception) {
 *     this.finishLatch = finishLatch;
 *     this.backoff = backoff;
 *     this.exception = exception;
 *   }
 *
 *   @Override
 *   public void onError(Throwable t) {
 *     // Need to handle non Status errors here first!
 *     try {
 *       retrier.onFailure(backoff, Status.fromThrowable(t));
 *
 *       // This assumes you passed through the relevant info to recreate the original request:
 *       requestObserver = myStub.asyncCall(
 *           request,
 *           new RetryingObserver(finishLatch, backoff));  // Recursion!
 *       requestObserver.onNext(i1);
 *       requestObserver.onNext(i2);
 *       ...
 *       requestObserver.onCompleted();
 *
 *     } catch (RetryException e) {
 *       exception.compareAndSet(null, e);
 *       finishLatch.countDown();
 *     }
 *   }
 *   @Override
 *   public void onCompleted() {
 *     finishLatch.countDown();
 *   }
 * }
 *
 * Retrier.Backoff backoff = retrier.newBackoff();
 * List<Status> errors = Collections.synchronizedList(new ArrayList<Status>());
 * while (true) {
 *   CountDownLatch finishLatch = new CountDownLatch(items.size());
 *   for (Item item : items) {
 *     requestObserver = myStub.asyncCall(
 *         request,
 *         new RetryingObserver(finishLatch, backoff));
 *     requestObserver.onNext(i1);
 *     requestObserver.onNext(i2);
 *     ...
 *     requestObserver.onCompleted();
 *   }
 *   finishLatch.await(someTime, TimeUnit.SECONDS);
 *   if (exception.get() != null) {
 *     throw exception.get(); // Re-throw the first encountered exception.
 *   }
 * }
 * </pre>
 *
 * In both cases you need to instantiate and keep a Backoff object, and use onFailure(s) to retry.
 */
public class Retrier {
  /**
   * Backoff is a stateful object providing a sequence of durations that are used to time delays
   * between retries. It is not ThreadSafe. The reason that Backoff needs to be stateful, rather
   * than a static map of attempt number to delay, is to enable using the retrier via the manual
   * onFailure(backoff, e) method (see multiple async gRPC calls example above).
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
   * Executes the given callable in a loop, retrying on retryable errors, as defined by the current
   * backoff/retry policy. Will raise the last encountered retriable error, or the first
   * non-retriable error.
   *
   * @param c The callable to execute.
   */
  public <T> T execute(Callable<T> c) throws InterruptedException, RetryException {
    Backoff backoff = backoffSupplier.get();
    while (true) {
      try {
        return c.call();
      } catch (StatusException | StatusRuntimeException e) {
        onFailure(backoff, Status.fromThrowable(e));
      } catch (Exception e) {
        // Generic catch because Callable is declared to throw Exception.
        Throwables.throwIfUnchecked(e);
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

  public void onFailure(Backoff backoff, Status s) throws RetryException, InterruptedException {
    onFailures(backoff, ImmutableList.of(s));
  }

  /**
   * Handles failures according to the current backoff/retry policy. If any of the errors are not
   * retriable, the first such error is thrown. Otherwise, if backoff still allows, this sleeps for
   * the specified duration. Otherwise, the first error is thrown.
   *
   * @param backoff The backoff object to get delays from.
   * @param errors The errors that occurred (must be non-empty).
   */
  public void onFailures(Backoff backoff, List<Status> errors)
      throws InterruptedException, RetryException {
    Preconditions.checkArgument(!errors.isEmpty(), "errors must be non-empty");
    long delay = backoff.nextDelayMillis();
    for (Status st : errors) {
      if (st.getCode() == Status.Code.CANCELLED && Thread.currentThread().isInterrupted()) {
        Thread.currentThread().interrupt();
        throw new InterruptedException();
      }
      if (delay < 0 || !isRetriable.apply(st)) {
        throw new RetryException(st.asRuntimeException(), backoff.getRetryAttempts());
      }
    }
    sleep(delay);
  }
}
