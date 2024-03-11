// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.common.util.concurrent.Uninterruptibles.getUninterruptibly;

import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutionException;

/** Helpers for serialization futures. */
public final class FutureHelpers {

  /**
   * Waits for {@code future} and returns the result.
   *
   * <p>Handles exceptions by converting them into {@link SerializationException}.
   */
  @CanIgnoreReturnValue // may be called for side effects
  public static <T> T waitForSerializationFuture(ListenableFuture<T> future)
      throws SerializationException {
    try {
      // TODO: b/297857068 - revisit whether this should handle to interrupts. As of 02/09/24,
      // serialization doesn't handle interrupts so introducing them here could lead to unforseen
      // problems.
      return getUninterruptibly(future);
    } catch (ExecutionException e) {
      throw asSerializationException(e.getCause());
    }
  }

  /** Combines a list of {@code Void} futures into a single future. */
  static ListenableFuture<Void> aggregateStatusFutures(List<ListenableFuture<Void>> futures) {
    if (futures.size() == 1) {
      return futures.get(0);
    }
    return Futures.whenAllSucceed(futures).call(() -> null, directExecutor());
  }

  private static SerializationException asSerializationException(Throwable cause) {
    if (cause instanceof SerializationException) {
      return new SerializationException(cause);
    }
    if (cause instanceof IOException) {
      return new SerializationException("serialization I/O error", cause);
    }
    return new SerializationException("unexpected serialization error", cause);
  }

  private FutureHelpers() {}
}
