// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.rules.ninja.file;

import com.google.common.base.Throwables;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.List;
import java.util.concurrent.ExecutionException;

/**
 * Helper class for accumulating the scheduled {@link ListenableFuture} sub-tasks and gathering
 * their results.
 *
 * @param <T> the type that ListenableFuture returns
 * @param <E> the type of exception, that the computation task can throw
 */
public class CollectingListFuture<T, E extends Exception> {
  private final List<ListenableFuture<T>> futures;
  private final Class<E> exceptionClazz;

  /** @param exceptionClazz the class of exception, that the computation task can throw */
  public CollectingListFuture(Class<E> exceptionClazz) {
    this.exceptionClazz = exceptionClazz;
    futures = Lists.newArrayList();
  }

  /** Adds future to the list */
  public void add(ListenableFuture<T> future) {
    futures.add(future);
  }

  /** Returns the list of combined results of all futures, registered with {@link #add}. */
  public List<T> getResult() throws E, InterruptedException {
    try {
      return Futures.allAsList(futures).get();
    } catch (ExecutionException e) {
      Throwable causeOrSelf = e.getCause();
      if (causeOrSelf == null) {
        causeOrSelf = e;
      }
      Throwables.propagateIfPossible(causeOrSelf, exceptionClazz);
      throw new IllegalStateException(e);
    }
  }
}
