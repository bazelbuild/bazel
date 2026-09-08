// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.util;

import static com.google.common.base.Throwables.throwIfInstanceOf;
import static com.google.common.base.Throwables.throwIfUnchecked;

import java.io.IOException;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

/** Utility methods for futures. */
public final class Futures {

  /**
   * Returns the result of a {@link Future} if successful, or throws any checked {@link Exception}
   * directly if it's an {@link IOException} or else wraps it in an {@link IOException}.
   *
   * <p>Cancel the future on {@link InterruptedException}
   */
  public static <T> T getFromFuture(Future<T> f) throws IOException, InterruptedException {
    return getFromFuture(f, /* cancelOnInterrupt */ true);
  }

  /**
   * Returns the result of a {@link Future} if successful, or throws any checked {@link Exception}
   * directly if it's an {@link IOException} or else wraps it in an {@link IOException}.
   *
   * @param cancelOnInterrupt cancel the future on {@link InterruptedException} if {@code true}.
   */
  public static <T> T getFromFuture(Future<T> f, boolean cancelOnInterrupt)
      throws IOException, InterruptedException {
    try {
      return f.get();
    } catch (CancellationException e) {
      throw new InterruptedException();
    } catch (ExecutionException e) {
      throwIfInstanceOf(e.getCause(), InterruptedException.class);
      throwIfInstanceOf(e.getCause(), IOException.class);
      throwIfUnchecked(e.getCause());
      throw new IOException(e.getCause());
    } catch (InterruptedException e) {
      if (cancelOnInterrupt) {
        f.cancel(true);
      }
      throw e;
    }
  }

  private Futures() {}
}
