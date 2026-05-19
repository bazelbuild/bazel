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
package com.google.devtools.build.lib.concurrent;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.devtools.build.lib.concurrent.RequestBatching.Operation;
import java.util.ArrayList;
import java.util.List;

/**
 * A shared, thread-local pool of {@link ArrayList} containing {@code Operation} instances.
 *
 * <p>This pool is designed to be shared across multiple {@link EagerRequestBatcher} instances. It
 * eliminates churn by reusing the same thread-local list allocations.
 */
public final class QueuePool<RequestT, ResponseT> {
  private final ThreadLocal<List<Operation<RequestT, ResponseT>>> pool;
  private final int maxBatchSize;

  public QueuePool(int maxBatchSize) {
    checkArgument(maxBatchSize >= 1, "maxBatchSize must be >= 1");
    this.maxBatchSize = maxBatchSize;
    this.pool =
        ThreadLocal.withInitial(() -> new ArrayList<Operation<RequestT, ResponseT>>(maxBatchSize));
  }

  /**
   * Gets a list from the pool for the current thread.
   *
   * <p>IMPORTANT: if the caller modifies or takes ownership of this list, it must recycle a
   * different, unowned, list. Otherwise, a later call to {@code getQueue} could return the same
   * list and cause an aliasing bug.
   */
  List<Operation<RequestT, ResponseT>> getQueue() {
    return pool.get();
  }

  /** Clears the list and returns it to the pool for the current thread. */
  void recycleQueue(List<Operation<RequestT, ResponseT>> queue) {
    queue.clear();
    pool.set(queue);
  }

  public int getMaxBatchSize() {
    return maxBatchSize;
  }
}
