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
import static com.google.devtools.build.lib.worker.TestUtils.createWorkerKey;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.lib.worker.WorkerPoolImpl.WorkerPoolConfig;
import java.util.Map.Entry;
import org.apache.commons.pool2.impl.DefaultPooledObject;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/** Tests WorkerPool. */
@RunWith(JUnit4.class)
public class WorkerPoolTest {
  @Rule public final MockitoRule mockito = MockitoJUnit.rule();
  @Mock WorkerFactory factoryMock;
  private FileSystem fileSystem;
  private int workerIds = 1;

  private static class TestWorker extends SingleplexWorker {
    TestWorker(WorkerKey workerKey, int workerId, Path workDir, Path logFile) {
      super(workerKey, workerId, workDir, logFile);
    }
  }

  @Before
  public void setUp() throws Exception {
    fileSystem = new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);
    doAnswer(
            arg ->
                new DefaultPooledObject<>(
                    new TestWorker(
                        arg.getArgument(0),
                        workerIds++,
                        fileSystem.getPath("/workDir"),
                        fileSystem.getPath("/logDir"))))
        .when(factoryMock)
        .makeObject(any());
    when(factoryMock.validateObject(any(), any())).thenReturn(true);
  }

  @Test
  public void testBorrow_createsWhenNeeded() throws Exception {
    WorkerPool workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList("mnem", 2, "", 1), entryList()));
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey);
    Worker worker2 = workerPool.borrowObject(workerKey);
    assertThat(worker1.getWorkerId()).isEqualTo(1);
    assertThat(worker2.getWorkerId()).isEqualTo(2);
    verify(factoryMock, times(2)).makeObject(workerKey);
  }

  @Test
  public void testBorrow_reusesWhenPossible() throws Exception {
    WorkerPool workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList("mnem", 2, "", 1), entryList()));
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey);
    workerPool.returnObject(workerKey, worker1);
    Worker worker2 = workerPool.borrowObject(workerKey);
    assertThat(worker1).isSameInstanceAs(worker2);
    verify(factoryMock, times(1)).makeObject(workerKey);
  }

  @Test
  public void testBorrow_usesDefault() throws Exception {
    WorkerPool workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList("mnem", 2, "", 1), entryList()));
    WorkerKey workerKey1 = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey1);
    Worker worker1a = workerPool.borrowObject(workerKey1);
    assertThat(worker1.getWorkerId()).isEqualTo(1);
    assertThat(worker1a.getWorkerId()).isEqualTo(2);
    WorkerKey workerKey2 = createWorkerKey(fileSystem, "other", false);
    Worker worker2 = workerPool.borrowObject(workerKey2);
    assertThat(worker2.getWorkerId()).isEqualTo(3);
    verify(factoryMock, times(2)).makeObject(workerKey1);
    verify(factoryMock, times(1)).makeObject(workerKey2);
  }

  @Test
  public void testBorrow_pooledByKey() throws Exception {
    WorkerPool workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList("mnem", 2, "", 1), entryList()));
    WorkerKey workerKey1 = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey1);
    Worker worker1a = workerPool.borrowObject(workerKey1);
    assertThat(worker1.getWorkerId()).isEqualTo(1);
    assertThat(worker1a.getWorkerId()).isEqualTo(2);
    WorkerKey workerKey2 = createWorkerKey(fileSystem, "mnem", false, "arg1");
    Worker worker2 = workerPool.borrowObject(workerKey2);
    assertThat(worker2.getWorkerId()).isEqualTo(3);
    verify(factoryMock, times(2)).makeObject(workerKey1);
    verify(factoryMock, times(1)).makeObject(workerKey2);
  }

  @Test
  public void testBorrow_separateMultiplexWorkers() throws Exception {
    WorkerPool workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(
                factoryMock, entryList("mnem", 1, "", 1), entryList("mnem", 2, "", 1)));
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

    verify(factoryMock, times(1)).makeObject(workerKey);
    verify(factoryMock, times(2)).makeObject(multiplexKey);
  }

  @Test
  public void testBorrow_doomedWorkers() throws Exception {
    WorkerPool workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList("mnem", 2, "", 1), entryList()));
    WorkerKey workerKey = createWorkerKey(fileSystem, "mnem", false);
    Worker worker1 = workerPool.borrowObject(workerKey);
    Worker worker2 = workerPool.borrowObject(workerKey);

    workerPool.setDoomedWorkers(ImmutableSet.of(worker1.getWorkerId()));

    assertThat(worker1.isDoomed()).isFalse();
    assertThat(worker2.isDoomed()).isFalse();

    workerPool.returnObject(workerKey, worker1);

    assertThat(worker1.isDoomed()).isTrue();
    assertThat(worker2.isDoomed()).isFalse();
  }

  private static ImmutableList<Entry<String, Integer>> entryList() {
    return ImmutableList.of();
  }

  private static ImmutableList<Entry<String, Integer>> entryList(
      String key1, int value1, String key2, int value2) {
    return ImmutableList.of(Maps.immutableEntry(key1, value1), Maps.immutableEntry(key2, value2));
  }

}
