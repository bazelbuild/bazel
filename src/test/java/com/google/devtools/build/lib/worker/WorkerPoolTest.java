// Copyright 2020 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.lib.worker.WorkerProcessStatus.Status;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;

/** Tests WorkerPool. */
@RunWith(Parameterized.class)
public class WorkerPoolTest {

  public static final FileSystem fileSystem =
      new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);

  private int workerIds = 1;

  @Parameter(0)
  public WorkerPoolSupplier workerPoolSupplier;

  @Parameter(1)
  public Supplier<WorkerFactory> workerFactorySupplier;

  private WorkerPool workerPool;
  private WorkerFactory factoryMock;

  private static final WorkerOptions options = new WorkerOptions();

  private static class TestWorker extends SingleplexWorker {
    TestWorker(
        WorkerKey workerKey, int workerId, Path workDir, Path logFile, WorkerOptions options) {
      super(workerKey, workerId, workDir, logFile, options);
    }
  }

  private interface WorkerPoolSupplier {
    WorkerPool get(WorkerFactory factory);
  }

  @Parameters
  public static List<Object[]> data() throws IOException {
    Supplier<WorkerFactory> workerFactorySupplier =
        () -> spy(new WorkerFactory(fileSystem.getPath("/outputbase/bazel-workers"), options));

    return Arrays.asList(
        new Object[][] {
          {
            (WorkerPoolSupplier)
                (factory) ->
                    new WorkerPoolImplLegacy(
                        new WorkerPoolConfig(
                            factory,
                            /* workerMaxInstances= */ ImmutableList.of(
                                Maps.immutableEntry("mnem", 2), Maps.immutableEntry("", 1)),
                            /* workerMaxMultiplexInstances= */ ImmutableList.of(
                                Maps.immutableEntry("mnem", 2), Maps.immutableEntry("", 1)))),
            workerFactorySupplier,
          },
          {
            (WorkerPoolSupplier)
                (factory) ->
                    new WorkerPoolImpl(
                        new WorkerPoolConfig(
                            factory,
                            /* workerMaxInstances= */ ImmutableList.of(
                                Maps.immutableEntry("mnem", 2)),
                            /* workerMaxMultiplexInstances= */ ImmutableList.of(
                                Maps.immutableEntry("mnem", 2)))),
            workerFactorySupplier,
          }
        });
  }

  @Before
  public void setUp() throws Exception {
    factoryMock = workerFactorySupplier.get();
    workerPool = workerPoolSupplier.get(factoryMock);
    doAnswer(
            arg ->
                new TestWorker(
                    arg.getArgument(0),
                    workerIds++,
                    fileSystem.getPath("/workDir"),
                    fileSystem.getPath("/logDir"),
                    options))
        .when(factoryMock)
        .create(any());
    doAnswer(
            args -> {
              Worker worker = args.getArgument(1);
              return worker.getStatus().isValid();
            })
        .when(factoryMock)
        .validateWorker(any(), any());
    doAnswer(
            args -> {
              Worker worker = args.getArgument(1);
              worker.destroy();
              return null;
            })
        .when(factoryMock)
        .destroyWorker(any(), any());
  }

  @Test
  public void testBorrow_createsWhenNeeded() throws Exception {
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey);
    Worker worker2 = workerPool.borrowObject(workerKey);
    assertThat(worker1.getWorkerId()).isEqualTo(1);
    assertThat(worker2.getWorkerId()).isEqualTo(2);
    verify(factoryMock, times(2)).create(workerKey);
  }

  @Test
  public void testBorrow_reusesWhenPossible() throws Exception {
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey);
    workerPool.returnObject(workerKey, worker1);
    Worker worker2 = workerPool.borrowObject(workerKey);
    assertThat(worker1).isSameInstanceAs(worker2);
    verify(factoryMock).create(workerKey);
  }

  @Test
  public void testBorrow_nonSpecifiedKey() throws Exception {
    WorkerKey workerKey1 = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey1);
    Worker worker1a = workerPool.borrowObject(workerKey1);
    assertThat(worker1.getWorkerId()).isEqualTo(1);
    assertThat(worker1a.getWorkerId()).isEqualTo(2);
    WorkerKey workerKey2 = createWorkerKey(fileSystem, "other", false);
    Worker worker2 = workerPool.borrowObject(workerKey2);
    assertThat(worker2.getWorkerId()).isEqualTo(3);
    verify(factoryMock, times(2)).create(workerKey1);
    verify(factoryMock).create(workerKey2);
  }

  @Test
  public void testBorrow_pooledByKey() throws Exception {
    WorkerKey workerKey1 = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey1);
    Worker worker1a = workerPool.borrowObject(workerKey1);
    assertThat(worker1.getWorkerId()).isEqualTo(1);
    assertThat(worker1a.getWorkerId()).isEqualTo(2);
    WorkerKey workerKey2 = createWorkerKey(fileSystem, "mnem", false, "arg1");
    Worker worker2 = workerPool.borrowObject(workerKey2);
    assertThat(worker2.getWorkerId()).isEqualTo(3);
    verify(factoryMock, times(2)).create(workerKey1);
    verify(factoryMock).create(workerKey2);
  }

  @Test
  public void testBorrow_separateMultiplexWorkers() throws Exception {
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey);
    assertThat(worker1.getWorkerId()).isEqualTo(1);
    workerPool.returnObject(workerKey, worker1);

    WorkerKey multiplexKey = createWorkerKey(fileSystem, "mnem", true);
    Worker multiplexWorker1 = workerPool.borrowObject(multiplexKey);
    Worker multiplexWorker2 = workerPool.borrowObject(multiplexKey);
    Worker worker1a = workerPool.borrowObject(workerKey);

    assertThat(multiplexWorker1.getWorkerId()).isEqualTo(2);
    assertThat(multiplexWorker2.getWorkerId()).isEqualTo(3);
    assertThat(worker1a.getWorkerId()).isEqualTo(1);

    verify(factoryMock).create(workerKey);
    verify(factoryMock, times(2)).create(multiplexKey);
  }

  @Test
  public void testBorrow_doomedWorkers() throws Exception {
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey);
    Worker worker2 = workerPool.borrowObject(workerKey);

    worker1.getStatus().maybeUpdateStatus(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);

    assertThat(worker1.getStatus().isKilled()).isFalse();
    assertThat(worker2.getStatus().isKilled()).isFalse();

    workerPool.returnObject(workerKey, worker1);

    assertThat(worker1.getStatus().isKilled()).isTrue();
    assertThat(worker2.getStatus().isKilled()).isFalse();
  }

  @Test
  public void testBorrow_blocksWhenUnavailable() throws Exception {
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    Worker unused1 = workerPool.borrowObject(workerKey);
    Worker unused2 = workerPool.borrowObject(workerKey);
    TestThread blockedBorrowThread =
        new TestThread(
            () -> {
              Worker unused = workerPool.borrowObject(workerKey);
            });
    blockedBorrowThread.start();

    AssertionError e =
        assertThrows(AssertionError.class, () -> blockedBorrowThread.joinAndAssertState(1000));
    assertThat(e).hasCauseThat().hasMessageThat().contains("is still alive");
    assertThat(workerPool.getNumActive(workerKey)).isEqualTo(2);
  }

  @Test
  public void testBorrow_blockedThread_getsReturnedWorker() throws Exception {
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey);
    Worker unused2 = workerPool.borrowObject(workerKey);
    TestThread blockedBorrowThread =
        new TestThread(
            () -> {
              // This blocks until worker1 returns its object.
              Worker worker = workerPool.borrowObject(workerKey);
              assertThat(worker).isSameInstanceAs(worker1);
            });
    blockedBorrowThread.start();

    // We want to 3rd borrow to be blocked for some time.
    Thread.sleep(500);
    workerPool.returnObject(worker1.getWorkerKey(), worker1);

    blockedBorrowThread.joinAndAssertState(10000);
    assertThat(workerPool.getNumActive(workerKey)).isEqualTo(2);
  }

  @Test
  public void testBorrow_blockedThread_createsWorkerWhenInvalidated() throws Exception {
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey);
    Worker unused2 = workerPool.borrowObject(workerKey);
    TestThread blockedBorrowThread =
        new TestThread(
            () -> {
              Worker worker = workerPool.borrowObject(workerKey);
              // Create a new worker instead.
              assertThat(worker.getWorkerId()).isEqualTo(3);
            });
    blockedBorrowThread.start();

    // We want to 3rd borrow to be blocked for some time.
    Thread.sleep(500);
    workerPool.invalidateObject(worker1.getWorkerKey(), worker1);

    blockedBorrowThread.joinAndAssertState(10000);
    assertThat(workerPool.getNumActive(workerKey)).isEqualTo(2);
  }

  @Test
  public void testBorrow_blockedThread_remainsBlockedWhenInvalidatedAndShrunk() throws Exception {
    // This is meant solely for WorkerPoolImpl; WorkerPoolImplLegacy will block only after some
    // time because GenericKeyedObjectPool only implements some
    assumeTrue(workerPool instanceof WorkerPoolImpl);
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey);
    Worker unused2 = workerPool.borrowObject(workerKey);
    TestThread blockedBorrowThread =
        new TestThread(
            () -> {
              Worker unused = workerPool.borrowObject(workerKey);
            });
    blockedBorrowThread.start();

    // There's no need to wait here as it doesn't matter whether #invalidateObject gets called
    // before or after the 3rd #borrowObject, the pool would not have the quota and borrowing will
    // still get blocked.
    worker1.getStatus().maybeUpdateStatus(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);
    workerPool.invalidateObject(worker1.getWorkerKey(), worker1);

    AssertionError e =
        assertThrows(AssertionError.class, () -> blockedBorrowThread.joinAndAssertState(1000));
    assertThat(e).hasCauseThat().hasMessageThat().contains("is still alive");
    assertThat(workerPool.getNumActive(workerKey)).isEqualTo(1);
  }

  @Test
  public void testEvict_evictsIdleWorkers() throws Exception {
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey);
    Worker worker2 = workerPool.borrowObject(workerKey);
    workerPool.returnObject(workerKey, worker1);
    workerPool.returnObject(workerKey, worker2);
    ImmutableSet<Integer> evicted =
        workerPool.evictWorkers(ImmutableSet.of(worker1.getWorkerId(), worker2.getWorkerId()));
    assertThat(evicted).containsExactly(worker1.getWorkerId(), worker2.getWorkerId());
    assertThat(workerPool.getNumActive(workerKey)).isEqualTo(0);
  }

  @Test
  public void testEvict_doesNotEvictActiveWorkers() throws Exception {
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey);
    Worker worker2 = workerPool.borrowObject(workerKey);
    workerPool.returnObject(workerKey, worker1);
    ImmutableSet<Integer> evicted =
        workerPool.evictWorkers(ImmutableSet.of(worker1.getWorkerId(), worker2.getWorkerId()));
    // Worker2 does not get evicted because it is still active.
    assertThat(evicted).containsExactly(worker1.getWorkerId());
    assertThat(workerPool.getNumActive(workerKey)).isEqualTo(1);
  }

  @Test
  public void testGetIdleWorkers() throws Exception {
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey);
    Worker worker2 = workerPool.borrowObject(workerKey);

    assertThat(workerPool.getIdleWorkers()).isEmpty();
    workerPool.returnObject(workerKey, worker1);
    workerPool.returnObject(workerKey, worker2);

    assertThat(workerPool.getIdleWorkers())
        .containsExactly(worker1.getWorkerId(), worker2.getWorkerId());
    assertThat(workerPool.getNumActive(workerKey)).isEqualTo(0);
  }

  @Test
  public void testShrinkingPool_doesNotShrinkBelowOneWorker() throws Exception {
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    assertThat(workerPool.getMaxTotalPerKey(workerKey)).isEqualTo(2);

    Worker worker1 = workerPool.borrowObject(workerKey);
    // Shrink the worker pool by 1.
    worker1.getStatus().maybeUpdateStatus(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);
    workerPool.returnObject(workerKey, worker1);
    assertThat(workerPool.getMaxTotalPerKey(workerKey)).isEqualTo(1);

    Worker worker2 = workerPool.borrowObject(workerKey);
    // Attempt to shrink the pool again.
    worker2.getStatus().maybeUpdateStatus(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);
    workerPool.returnObject(workerKey, worker2);
    // It should not be shrunk below 1.
    assertThat(workerPool.getMaxTotalPerKey(workerKey)).isEqualTo(1);
    assertThat(workerPool.getNumActive(workerKey)).isEqualTo(0);
  }

  @Test
  public void testGetNumActive() throws Exception {
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    assertThat(workerPool.getNumActive(workerKey)).isEqualTo(0);
    Worker worker1 = workerPool.borrowObject(workerKey);
    Worker worker2 = workerPool.borrowObject(workerKey);
    assertThat(workerPool.getNumActive(workerKey)).isEqualTo(2);
    workerPool.returnObject(workerKey, worker1);
    workerPool.returnObject(workerKey, worker2);
    assertThat(workerPool.getNumActive(workerKey)).isEqualTo(0);
  }

  @Test
  public void testReset_removesPreviouslyShrunkValues() throws Exception {
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    assertThat(workerPool.getMaxTotalPerKey(workerKey)).isEqualTo(2);

    Worker worker1 = workerPool.borrowObject(workerKey);
    // Shrink the worker pool by 1.
    worker1.getStatus().maybeUpdateStatus(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);
    workerPool.returnObject(workerKey, worker1);
    assertThat(workerPool.getMaxTotalPerKey(workerKey)).isEqualTo(1);

    workerPool.reset();
    assertThat(workerPool.getMaxTotalPerKey(workerKey)).isEqualTo(2);
  }
}
