// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.Iterables;
import com.google.common.primitives.Ints;
import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * A sharding visitor which uses a {@link AbstractQueueVisitor}. It shards pending-visit items and
 * aims at reaching maximum parallelism by ensuring all threads are utilized unless number of
 * pending items are fewer than number of threads.
 */
public abstract class AbstractShardedVisitor<T> {

  protected static final int DEFAULT_THREAD_COUNT = Runtime.getRuntime().availableProcessors();

  private static final int DEFAULT_MAX_BATCH_SIZE = 8192;

  /**
   * Default {@link ErrorClassifier} used by the visitor returned by {@link
   * #AbstractShardedVisitor(String)}.
   */
  public static final ErrorClassifier ERROR_CLASSIFIER =
      new ErrorClassifier() {
        @Override
        protected ErrorClassification classifyException(Exception e) {
          return e instanceof RuntimeException
              ? ErrorClassification.CRITICAL_AND_LOG
              : ErrorClassification.NOT_CRITICAL;
        }
      };

  private final AbstractQueueVisitor executor;
  private final LinkedBlockingQueue<T> remainingItemsToVisit;
  private final int numThreads;

  /**
   * Creates a visitor that uses a {@link ForkJoinQuiescingExecutor} backed by a {@link
   * ForkJoinPool} using {@code DEFAULT_THREAD_COUNT} threads.
   */
  protected AbstractShardedVisitor(String name) {
    this(
        ForkJoinQuiescingExecutor.newBuilder()
            .withOwnershipOf(NamedForkJoinPool.newNamedPool(name, DEFAULT_THREAD_COUNT))
            .setErrorClassifier(ERROR_CLASSIFIER)
            .build(),
        DEFAULT_THREAD_COUNT);
  }

  /**
   * Creates a visitor using an {@link AbstractQueueVisitor}, using up {@code numThreads} threads.
   */
  public AbstractShardedVisitor(AbstractQueueVisitor executor, int numThreads) {
    this.executor = executor;
    this.numThreads = numThreads;
    this.remainingItemsToVisit = new LinkedBlockingQueue<>();
  }

  /**
   * Starts parallel visitations of items. Waits until queue is drained and there are no more items
   * to visit.
   */
  public void scheduleVisitationsAndAwaitQuiescence(Collection<T> itemsToVisit)
      throws InterruptedException {
    remainingItemsToVisit.addAll(itemsToVisit);
    shardAndScheduleRemainingItems();
    executor.awaitQuiescence(/*interruptWorkers=*/ true);
  }

  /** Ensures no item is still pending visitation. */
  public void checkComplete() {
    if (remainingItemsToVisit.isEmpty()) {
      return;
    }
    int numUnvisitedItems = remainingItemsToVisit.size();
    ArrayList<T> unvisitedItems = new ArrayList<>(10);
    remainingItemsToVisit.drainTo(unvisitedItems, 10);
    throw new IllegalStateException(
        String.format(
            "There are %s item(s) enqueued for visiting but not visited before quiescence "
                + "(sample: %s)",
            numUnvisitedItems, Iterables.limit(unvisitedItems, 10)));
  }

  /** Gets max batch size in each shard. */
  protected int getMaxBatchSize() {
    return DEFAULT_MAX_BATCH_SIZE;
  }

  /** Shards the work of {@link #visit}ing {@code remainingItemsToVisit} across the free threads. */
  private void shardAndScheduleRemainingItems() {
    // Note that LinkedBlockingQueue#size() is a constant time operation.
    int numTasksExcludingThis = Ints.checkedCast(executor.getTaskCount()) - 1;
    int freeThreads = Math.max(numThreads - numTasksExcludingThis, 1);

    int itemsPerThread = (remainingItemsToVisit.size() / freeThreads) + 1;
    int batchSize = Math.min(itemsPerThread, getMaxBatchSize());

    for (int i = 0; i < freeThreads; i++) {
      ArrayList<T> items = new ArrayList<>(batchSize);
      remainingItemsToVisit.drainTo(items, batchSize);
      // We may be done because someone else stole our items or because freeThreads was greater than
      // remainingItemsToVisit.size() and we've finished dealing out the remaining items.
      if (items.isEmpty()) {
        break;
      }
      executor.execute(
          () -> {
            try {
              remainingItemsToVisit.addAll(visit(items));
              shardAndScheduleRemainingItems();
            } catch (InterruptedException e) {
              // The work thread may get interrupted only when the main thread is interrupted. Stop
              // doing further work.
              Thread.currentThread().interrupt();
            }
          });
    }
  }

  /** Visits {@code itemsToVisit} and returns the next batch of items to visit. */
  protected abstract Collection<T> visit(Collection<T> itemsToVisit) throws InterruptedException;
}
