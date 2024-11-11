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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A class to build shards (work queues) for a given task.
 *
 * <p>{@link #add}ed elements will be equally distributed among the shards.
 *
 * @param <T> the type of collection over which we're sharding
 */
public final class Sharder<T> implements Iterable<List<T>> {
  private final ImmutableList<List<T>> shards;
  private final AtomicInteger count = new AtomicInteger();

  public Sharder(int maxNumShards, int expectedTotalSize) {
    Preconditions.checkArgument(maxNumShards > 0);
    Preconditions.checkArgument(expectedTotalSize >= 0);
    this.shards = immutableListOfLists(maxNumShards, expectedTotalSize / maxNumShards);
  }

  /**
   * Adds an item to a shard.
   *
   * <p>May safely be called concurrently by multiple threads.
   */
  @ThreadSafe
  public void add(T item) {
    int nextShardIndex = count.incrementAndGet() % shards.size();
    List<T> shard = shards.get(nextShardIndex);
    synchronized (shard) {
      shard.add(item);
    }
  }

  /**
   * Returns an immutable list of mutable lists.
   *
   * @param numLists the number of top-level lists.
   * @param expectedSize the expected size of each mutable list.
   * @return a list of lists.
   */
  private static <T> ImmutableList<List<T>> immutableListOfLists(int numLists, int expectedSize) {
    var outerList = ImmutableList.<List<T>>builderWithExpectedSize(numLists);
    for (int i = 0; i < numLists; i++) {
      outerList.add(new ArrayList<>(expectedSize));
    }
    return outerList.build();
  }

  @Override
  public Iterator<List<T>> iterator() {
    return Iterables.filter(shards, list -> !list.isEmpty()).iterator();
  }
}
