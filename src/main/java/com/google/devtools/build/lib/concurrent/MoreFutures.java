// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;

import java.util.List;

import javax.annotation.Nullable;

/**
 * Utility class for working with futures.
 */
public class MoreFutures {

  private MoreFutures() {}

  /**
   * Creates a new {@code ListenableFuture} whose value is a list containing the
   * values of all its input futures, if all succeed. If any input fails, the
   * returned future fails. If any of the futures fails, it cancels all the other futures.
   *
   * <p> This method is similar to {@code Futures.allAsList} but additionally it cancels all the
   * futures in case any of them fails.
   */
  public static <V> ListenableFuture<List<V>> allAsListOrCancelAll(
      final Iterable<? extends ListenableFuture<? extends V>> futures) {
    ListenableFuture<List<V>> combinedFuture = Futures.allAsList(futures);
    Futures.addCallback(combinedFuture, new FutureCallback<List<V>>() {
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
    });
    return combinedFuture;
  }
}
