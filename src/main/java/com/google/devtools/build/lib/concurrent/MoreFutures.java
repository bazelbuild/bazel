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

import static com.google.common.util.concurrent.Futures.addCallback;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.base.Throwables;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import javax.annotation.Nullable;

/** Utility class for working with futures. */
public class MoreFutures {

  private MoreFutures() {}

  /**
   * Waits for the first one of the following to occur:
   *
   * <ul>
   *   <li>All of the given futures complete successfully.
   *   <li>One of the given futures has an {@link ExecutionException}. This {@link
   *       ExecutionException} is propagated. (N.B. If multiple futures have {@link
   *       ExecutionExceptions}s, one will be selected non-deterministically.)
   *   <li>The calling thread is interrupted. The {@link InterruptedException} is propagated.
   * </ul>
   */
  public static <V> void waitForAllInterruptiblyFailFast(
      Iterable<? extends Future<? extends V>> futures)
      throws ExecutionException, InterruptedException {
    int numFutures = Iterables.size(futures);
    while (true) {
      int numCompletedFutures = 0;
      for (Future<? extends V> future : futures) {
        try {
          future.get(1, TimeUnit.MILLISECONDS);
        } catch (TimeoutException te) {
          continue;
        }
        numCompletedFutures++;
      }
      if (numCompletedFutures == numFutures) {
        return;
      }
    }
  }

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
   */
  public static <R> R waitForFutureAndGet(Future<R> future) throws InterruptedException {
    try {
      return future.get();
    } catch (ExecutionException e) {
      Throwables.propagateIfPossible(e.getCause(), InterruptedException.class);
      throw new IllegalStateException(e);
    }
  }

  public static <R, E extends Exception> R waitForFutureAndGetWithCheckedException(
      Future<R> future, Class<E> exceptionClass) throws E, InterruptedException {
    return waitForFutureAndGetWithCheckedException(future, exceptionClass, null);
  }

  public static <R, E1 extends Exception, E2 extends Exception>
      R waitForFutureAndGetWithCheckedException(
          Future<R> future, Class<E1> exceptionClass1, @Nullable Class<E2> exceptionClass2)
          throws E1, E2, InterruptedException {
    try {
      return future.get();
    } catch (ExecutionException e) {
      if (exceptionClass2 == null) {
        Throwables.propagateIfPossible(e.getCause(), exceptionClass1);
      } else {
        Throwables.propagateIfPossible(e.getCause(), exceptionClass1, exceptionClass2);
      }
      Throwables.throwIfInstanceOf(e.getCause(), InterruptedException.class);
      throw new IllegalStateException(e);
    }
  }
}
