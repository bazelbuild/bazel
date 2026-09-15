// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.concurrent;

import static com.google.common.base.Throwables.throwIfInstanceOf;
import static com.google.common.base.Throwables.throwIfUnchecked;
import static com.google.common.util.concurrent.Futures.addCallback;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import javax.annotation.Nullable;

/** Utility class for working with futures. */
public class MoreFutures {

  private MoreFutures() {}

  /**
   * Creates a new {@code ListenableFuture} whose value is a list containing the values of all its
   * input futures, if all succeed. If any input fails, the returned future fails. If any of the
   * futures fails, it cancels all the other futures.
   *
   * <p>This method is similar to {@code Futures.allAsList} but additionally it cancels all the
   * futures in case any of them fails.
   */
  public static <V> ListenableFuture<List<V>> allAsListOrCancelAll(
      final Iterable<? extends ListenableFuture<? extends V>> futures) {
    ListenableFuture<List<V>> combinedFuture = Futures.allAsList(futures);
    addCallback(
        combinedFuture,
        new FutureCallback<List<V>>() {
          @Override
          public void onSuccess(@Nullable List<V> vs) {}

          /**
           * In case of a failure of any of the futures (that gets propagated to combinedFuture) we
           * cancel all the futures in the list.
           */
          @Override
          public void onFailure(Throwable ignore) {
            for (ListenableFuture<? extends V> future : futures) {
              future.cancel(true);
            }
          }
        },
        directExecutor());
    return combinedFuture;
  }

  /**
   * Returns the result of {@code future}. If it threw an {@link InterruptedException} (wrapped in
   * an {@link ExecutionException}), throws that underlying {@link InterruptedException}. Crashes on
   * all other exceptions.
   *
   * <p>If {@code cancelOnInterrupt} is true, the future is cancelled if it threw an {@link
   * InterruptedException}.
   */
  @CanIgnoreReturnValue
  public static <R> R waitForFutureAndGet(Future<R> future, boolean cancelOnInterrupt)
      throws InterruptedException {
    try {
      return future.get();
    } catch (ExecutionException e) {
      throwIfInstanceOf(e.getCause(), InterruptedException.class);
      throwIfUnchecked(e.getCause());
      throw new IllegalStateException(e);
    } catch (InterruptedException e) {
      if (cancelOnInterrupt) {
        future.cancel(/* mayInterruptIfRunning= */ true);
      }
      throw e;
    }
  }

  public static <R, E extends Exception> R waitForFutureAndGetWithCheckedException(
      Future<R> future, boolean cancelOnInterrupt, Class<E> exceptionClass)
      throws E, InterruptedException {
    return waitForFutureAndGetWithCheckedException(future, cancelOnInterrupt, exceptionClass, null);
  }

  public static <R, E1 extends Exception, E2 extends Exception>
      R waitForFutureAndGetWithCheckedException(
          Future<R> future,
          boolean cancelOnInterrupt,
          Class<E1> exceptionClass1,
          @Nullable Class<E2> exceptionClass2)
          throws E1, E2, InterruptedException {
    return waitForFutureAndGetWithCheckedException(
        future, cancelOnInterrupt, exceptionClass1, exceptionClass2, null);
  }

  public static <R, E1 extends Exception, E2 extends Exception, E3 extends Exception>
      R waitForFutureAndGetWithCheckedException(
          Future<R> future,
          boolean cancelOnInterrupt,
          Class<E1> exceptionClass1,
          @Nullable Class<E2> exceptionClass2,
          @Nullable Class<E3> exceptionClass3)
          throws E1, E2, E3, InterruptedException {
    try {
      return future.get();
    } catch (ExecutionException e) {
      throwIfInstanceOf(e.getCause(), exceptionClass1);
      if (exceptionClass2 != null) {
        throwIfInstanceOf(e.getCause(), exceptionClass2);
      }
      if (exceptionClass3 != null) {
        throwIfInstanceOf(e.getCause(), exceptionClass3);
      }
      throwIfUnchecked(e.getCause());
      throwIfInstanceOf(e.getCause(), InterruptedException.class);
      throw new IllegalStateException(e);
    } catch (InterruptedException e) {
      if (cancelOnInterrupt) {
        future.cancel(/* mayInterruptIfRunning= */ true);
      }
      throw e;
    }
  }
}
