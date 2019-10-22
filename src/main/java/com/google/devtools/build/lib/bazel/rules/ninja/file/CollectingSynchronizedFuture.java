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

import com.google.common.collect.Lists;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.Collections;
import java.util.List;

public class CollectingSynchronizedFuture<T, E extends Exception> {
  private final List<ListenableFuture<T>> futures;
  private final Class<E> exceptionClazz;

  public CollectingSynchronizedFuture(Class<E> exceptionClazz) {
    futures = Collections.synchronizedList(Lists.newArrayList());
    this.exceptionClazz = exceptionClazz;
  }

  /** Adds future to the list */
  public void add(ListenableFuture<T> future) {
    futures.add(future);
  }

  public List<T> getResult() throws E, InterruptedException {
    List<T> result = Lists.newArrayList();
    while (!futures.isEmpty()) {
      List<ListenableFuture<T>> copy = Lists.newArrayList(futures);
      result.addAll(CollectingListFuture.getListOfFuturesResult(copy, exceptionClazz));
      futures.removeAll(copy);
    }
    return result;
  }
}
