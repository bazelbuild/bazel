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
package com.google.devtools.build.android;

import com.google.common.util.concurrent.ListenableFuture;

import com.android.ide.common.res2.MergingException;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutionException;

/**
 * Aggregates a list of failed {@link ListenableFuture} and throws it as an exception.
 */
class FailedFutureAggregator<T extends Throwable> {

  private ExceptionFactory<T> exceptionFactory;

  interface ExceptionFactory<T extends Throwable> {
    T create();
  }

  public static FailedFutureAggregator<IOException> forIOExceptionsWithMessage(
      final String message) {
    return new FailedFutureAggregator<IOException>(
        new ExceptionFactory<IOException>() {
          @Override
          public IOException create() {
            return new IOException(message);
          }
        });
  }

  public static FailedFutureAggregator<MergingException> createForMergingExceptionWithMessage(
      final String message) {
    return new FailedFutureAggregator<MergingException>(
        new ExceptionFactory<MergingException>() {
          @Override
          public MergingException create() {
            return new MergingException(message);
          }
        });
  }

  private FailedFutureAggregator(ExceptionFactory<T> exceptionFactory) {
    this.exceptionFactory = exceptionFactory;
  }

  /** Iterates throw a list of futures, throwing an Exception if any have failed. */
  public void aggregateAndMaybeThrow(List<ListenableFuture<Boolean>> tasks) throws T {
    // Retrieve all the exceptions and wrap them in an IOException.
    T exception = null;
    for (ListenableFuture<Boolean> task : tasks) {
      try {
        task.get();
      } catch (ExecutionException | InterruptedException e) {
        // Lazy exception creation to avoid creating an unused Exception.
        if (exception == null) {
          exception = exceptionFactory.create();
        }
        exception.addSuppressed(e.getCause() != null ? e.getCause() : e);
      }
    }
    if (exception != null) {
      throw exception;
    }
  }
}
