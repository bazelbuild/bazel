// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.worker.WorkerTestUtils.createWorkerKey;

import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.lib.worker.WorkerProcessStatus.Status;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.impl.DefaultPooledObject;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class SimpleWorkerPoolTest {

  TestWorkerFactory workerFactory;
  private static FileSystem fileSystem;

  private static class TestWorker extends SingleplexWorker {
    TestWorker(
        WorkerKey workerKey, int workerId, Path workDir, Path logFile, WorkerOptions options) {
      super(workerKey, workerId, workDir, logFile, options);
    }
  }

  private static class TestWorkerFactory extends WorkerFactory {
    private int workerIds = 1;

    public TestWorkerFactory(Path workerBaseDir, WorkerOptions workerOptions) {
      super(workerBaseDir, workerOptions);
    }

    @Override
    public PooledObject<Worker> makeObject(WorkerKey workerKey) {
      return new DefaultPooledObject<>(
          new TestWorker(
              workerKey,
              workerIds++,
              fileSystem.getPath("/workDir"),
              fileSystem.getPath("/logDir"),
              workerOptions));
    }

    @Override
    public boolean validateObject(WorkerKey key, PooledObject<Worker> p) {
      return p.getObject().getStatus().isValid();
    }
  }

  @Before
  public void setUp() throws Exception {
    fileSystem = new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);
    workerFactory = new TestWorkerFactory(null, new WorkerOptions());
  }

  @Test
  public void testReturn_shrinkPoolOnce() throws Exception {
    SimpleWorkerPool workerPool = new SimpleWorkerPool(workerFactory, 3);
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey);
    Worker worker2 = workerPool.borrowObject(workerKey);

    assertThat(workerPool.getMaxTotalPerKey(workerKey)).isEqualTo(3);

    worker1.getStatus().maybeUpdateStatus(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);

    workerPool.returnObject(workerKey, worker1);
    workerPool.returnObject(workerKey, worker2);

    assertThat(workerPool.getMaxTotalPerKey(workerKey)).isEqualTo(2);
    assertThat(workerPool.getNumIdle(workerKey)).isEqualTo(1);
  }

  @Test
  public void testReturn_whenAllWorkersDoomed_shrinksToOne() throws Exception {
    SimpleWorkerPool workerPool = new SimpleWorkerPool(workerFactory, 3);
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey);
    Worker worker2 = workerPool.borrowObject(workerKey);
    Worker worker3 = workerPool.borrowObject(workerKey);

    assertThat(workerPool.getMaxTotalPerKey(workerKey)).isEqualTo(3);

    worker1.getStatus().maybeUpdateStatus(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);
    worker2.getStatus().maybeUpdateStatus(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);
    worker3.getStatus().maybeUpdateStatus(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);

    workerPool.returnObject(workerKey, worker1);
    workerPool.returnObject(workerKey, worker2);
    workerPool.returnObject(workerKey, worker3);

    assertThat(workerPool.getMaxTotalPerKey(workerKey)).isEqualTo(1);
    assertThat(workerPool.getNumIdle(workerKey)).isEqualTo(0);
  }

  @Test
  public void testReturn_differentWorkerKeyDontAffectEachOtherPools() throws Exception {
    SimpleWorkerPool workerPool = new SimpleWorkerPool(workerFactory, 3);
    WorkerKey workerKey1 = createWorkerKey(fileSystem, "mnem1", false);
    WorkerKey workerKey2 = createWorkerKey(fileSystem, "mnem2", false);
    Worker worker1 = workerPool.borrowObject(workerKey1);
    Worker worker2 = workerPool.borrowObject(workerKey2);

    assertThat(workerPool.getMaxTotalPerKey(workerKey1)).isEqualTo(3);
    assertThat(workerPool.getMaxTotalPerKey(workerKey2)).isEqualTo(3);

    worker1.getStatus().maybeUpdateStatus(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);

    workerPool.returnObject(workerKey1, worker1);
    workerPool.returnObject(workerKey2, worker2);

    assertThat(workerPool.getMaxTotalPerKey(workerKey1)).isEqualTo(2);
    assertThat(workerPool.getMaxTotalPerKey(workerKey2)).isEqualTo(3);
    assertThat(workerPool.getNumIdle(workerKey1)).isEqualTo(0);
    assertThat(workerPool.getNumIdle(workerKey2)).isEqualTo(1);
  }
}
