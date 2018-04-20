// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.worker;

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableSet;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.concurrent.ThreadSafe;
import org.apache.commons.pool2.impl.GenericKeyedObjectPool;
import org.apache.commons.pool2.impl.GenericKeyedObjectPoolConfig;

/**
 * A worker pool that spawns multiple workers and delegates work to them.
 *
 * <p>This is useful when the worker cannot handle multiple parallel requests on its own and we need
 * to pre-fork a couple of them instead.
 */
@ThreadSafe
final class WorkerPool extends GenericKeyedObjectPool<WorkerKey, Worker> {
  private final AtomicInteger highPriorityWorkersInUse = new AtomicInteger(0);
  private final ImmutableSet<String> highPriorityWorkerMnemonics;

  /**
   * @param factory worker factory
   * @param config pool configuration
   * @param highPriorityWorkers mnemonics of high priority workers
   */
  public WorkerPool(
      WorkerFactory factory,
      GenericKeyedObjectPoolConfig config,
      Iterable<String> highPriorityWorkers) {
    super(factory, config);
    highPriorityWorkerMnemonics = ImmutableSet.copyOf(highPriorityWorkers);
  }

  /**
   * Gets a worker.
   *
   * @param key worker key
   * @return a worker
   */
  @Override
  public Worker borrowObject(WorkerKey key) throws IOException, InterruptedException {
    Worker result;
    try {
      result = super.borrowObject(key);
    } catch (Throwable t) {
      Throwables.propagateIfPossible(t, IOException.class, InterruptedException.class);
      throw new RuntimeException("unexpected", t);
    }

    if (highPriorityWorkerMnemonics.contains(key.getMnemonic())) {
      highPriorityWorkersInUse.incrementAndGet();
    } else {
      try {
        waitForHighPriorityWorkersToFinish();
      } catch (InterruptedException e) {
        returnObject(key, result);
        throw e;
      }
    }

    return result;
  }

  @Override
  public void returnObject(WorkerKey key, Worker obj) {
    if (highPriorityWorkerMnemonics.contains(key.getMnemonic())) {
      decrementHighPriorityWorkerCount();
    }
    super.returnObject(key, obj);
  }

  @Override
  public void invalidateObject(WorkerKey key, Worker obj) throws IOException, InterruptedException {
    if (highPriorityWorkerMnemonics.contains(key.getMnemonic())) {
      decrementHighPriorityWorkerCount();
    }
    try {
      super.invalidateObject(key, obj);
    } catch (Throwable t) {
      Throwables.propagateIfPossible(t, IOException.class, InterruptedException.class);
      throw new RuntimeException("unexpected", t);
    }
  }

  // Decrements the high-priority workers counts and pings waiting threads if appropriate.
  private void decrementHighPriorityWorkerCount() {
    if (highPriorityWorkersInUse.decrementAndGet() <= 1) {
      synchronized (highPriorityWorkersInUse) {
        highPriorityWorkersInUse.notifyAll();
      }
    }
  }

  // Returns once less than two high-priority workers are running.
  private void waitForHighPriorityWorkersToFinish() throws InterruptedException {
    // Fast path for the case where the high-priority workers feature is not in use.
    if (highPriorityWorkerMnemonics.isEmpty()) {
      return;
    }

    while (highPriorityWorkersInUse.get() > 1) {
      synchronized (highPriorityWorkersInUse) {
        highPriorityWorkersInUse.wait();
      }
    }
  }
}
