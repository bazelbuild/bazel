// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.common.util.concurrent.AsyncCallable;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListenableScheduledFuture;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.function.Predicate;
import java.util.function.Supplier;

/**
 * Supports retrying the asynchronous execution of a {@link Callable} in case of failure.
 */
public class AsyncRetrier extends Retrier {
  private final ListeningScheduledExecutorService retryService;

  public AsyncRetrier(
      Retrier retrier, 
      ListeningScheduledExecutorService retryService) {
    super(retrier);
    this.retryService = retryService;
  }

  public AsyncRetrier(
      RemoteOptions options,
      Predicate<? super Exception> shouldRetry,
      ListeningScheduledExecutorService retryService,
      CircuitBreaker circuitBreaker) {
    this(
        options.experimentalRemoteRetry
            ? () -> new RemoteRetrier.ExponentialBackoff(options)
            : () -> RETRIES_DISABLED,
        shouldRetry,
        retryService,
        circuitBreaker);
  }

  public AsyncRetrier(
      Supplier<Backoff> backoffSupplier,
      Predicate<? super Exception> shouldRetry,
      ListeningScheduledExecutorService retryService,
      CircuitBreaker circuitBreaker) {
    super(backoffSupplier, shouldRetry, circuitBreaker);
    this.retryService = retryService;
  }

  @VisibleForTesting
  AsyncRetrier(
      Supplier<Backoff> backoffSupplier,
      Predicate<? super Exception> shouldRetry,
      ListeningScheduledExecutorService retryService,
      CircuitBreaker circuitBreaker,
      Sleeper sleeper) {
    super(backoffSupplier, shouldRetry, circuitBreaker, sleeper);
    this.retryService = retryService;
  }

  /**
   * Executes an {@link AsyncCallable}, retrying execution in case of failure and returning a {@link
   * ListenableFuture} pointing to the result/error.
   */
  public <T> ListenableFuture<T> executeAsync(AsyncCallable<T> call) {
    SettableFuture<T> f = SettableFuture.create();
    executeAsync(call, f);
    return f;
  }

  /**
   * Executes an {@link AsyncCallable}, retrying execution in case of failure and uses the provided
   * {@code promise} to point to the result/error.
   */
  public <T> void executeAsync(AsyncCallable<T> call, SettableFuture<T> promise) {
    Preconditions.checkNotNull(call);
    Preconditions.checkNotNull(promise);
    Backoff backoff = newBackoff();
    executeAsync(call, promise, backoff);
  }

  private <T> void executeAsync(AsyncCallable<T> call, SettableFuture<T> outerF, Backoff backoff) {
    Preconditions.checkState(!outerF.isDone(), "outerF completed already.");
    try {
      Futures.addCallback(
          call.call(),
          new FutureCallback<T>() {
            @Override
            public void onSuccess(T t) {
              outerF.set(t);
            }

            @Override
            public void onFailure(Throwable t) {
              onExecuteAsyncFailure(t, call, outerF, backoff);
            }
          },
          MoreExecutors.directExecutor());
    } catch (Exception e) {
      onExecuteAsyncFailure(e, call, outerF, backoff);
    }
  }

  private <T> void onExecuteAsyncFailure(
      Throwable t, AsyncCallable<T> call, SettableFuture<T> outerF, Backoff backoff) {
    long waitMillis = backoff.nextDelayMillis();
    if (waitMillis >= 0 && t instanceof Exception && isRetriable((Exception) t)) {
      try {
        ListenableScheduledFuture<?> sf =
            retryService.schedule(
                () -> executeAsync(call, outerF, backoff), waitMillis, TimeUnit.MILLISECONDS);
        Futures.addCallback(
            sf,
            new FutureCallback<Object>() {
              @Override
              public void onSuccess(Object o) {
                // Submitted successfully. Intentionally left empty.
              }

              @Override
              public void onFailure(Throwable t) {
                Exception e = t instanceof Exception ? (Exception) t : new Exception(t);
                outerF.setException(
                    new RetryException(
                        "Scheduled execution errored.", backoff.getRetryAttempts(), e));
              }
            },
            MoreExecutors.directExecutor());
      } catch (RejectedExecutionException e) {
        // May be thrown by .schedule(...) if i.e. the executor is shutdown.
        outerF.setException(
            new RetryException("Rejected by executor.", backoff.getRetryAttempts(), e));
      }
    } else {
      Exception e = t instanceof Exception ? (Exception) t : new Exception(t);
      String message =
          waitMillis >= 0
              ? "Status not retriable."
              : "Exhaused retry attempts (" + backoff.getRetryAttempts() + ")";
      RetryException error = new RetryException(message, backoff.getRetryAttempts(), e);
      outerF.setException(error);
    }
  }
}
