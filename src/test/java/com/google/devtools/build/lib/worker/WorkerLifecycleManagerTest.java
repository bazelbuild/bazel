// Copyright 2022 The Bazel Authors. All rights reserved.
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
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.lib.worker.WorkerPoolImpl.WorkerPoolConfig;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
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

@RunWith(JUnit4.class)
public final class WorkerLifecycleManagerTest {

  @Rule public final MockitoRule mockito = MockitoJUnit.rule();
  private final EventBus eventBus = new EventBus();

  @Mock WorkerFactory factoryMock;
  private FileSystem fileSystem;
  private int workerIds = 1;
  private EventRecorder eventRecorder = new EventRecorder();

  private static class EventRecorder {
    public List<WorkerEvictedEvent> workerEvictedEvents;

    public EventRecorder() {
      workerEvictedEvents = new ArrayList<>();
    }

    @Subscribe
    public synchronized void setXcodeConfigInfo(WorkerEvictedEvent workerEvictedEvent) {
      this.workerEvictedEvents.add(workerEvictedEvent);
    }
  }

  @Before
  public void setUp() throws Exception {
    fileSystem = new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);
    doAnswer(
            arg ->
                new DefaultPooledObject<>(
                    new SingleplexWorker(
                        arg.getArgument(0),
                        workerIds++,
                        fileSystem.getPath("/workDir"),
                        fileSystem.getPath("/logDir"))))
        .when(factoryMock)
        .makeObject(any());
    when(factoryMock.validateObject(any(), any())).thenReturn(true);
    eventRecorder = new EventRecorder();
    eventBus.register(eventRecorder);
  }

  @Test
  public void testEvictWorkers_doNothing_lowMemoryUsage() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList("dummy", 1), emptyEntryList()));
    WorkerKey key = createWorkerKey("dummy", fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);

    ImmutableList<WorkerMetric> workerMetrics =
        ImmutableList.of(
            WorkerMetric.create(
                createWorkerProperties(w1.getWorkerId(), 1L, "dummy"),
                createWorkerStat(1024),
                true));
    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 1024 * 100;

    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);
    manager.setEventBus(eventBus);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
    assertThat(eventRecorder.workerEvictedEvents).isEmpty();
  }

  @Test
  public void testEvictWorkers_doNothing_zeroThreshold() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList("dummy", 1), emptyEntryList()));
    WorkerKey key = createWorkerKey("dummy", fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);

    ImmutableList<WorkerMetric> workerMetrics =
        ImmutableList.of(
            WorkerMetric.create(
                createWorkerProperties(w1.getWorkerId(), 1L, "dummy"),
                createWorkerStat(1024),
                true));
    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 0;

    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
  }

  @Test
  public void testEvictWorkers_doNothing_emptyMetrics() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList("dummy", 1), emptyEntryList()));
    WorkerKey key = createWorkerKey("dummy", fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);

    ImmutableList<WorkerMetric> workerMetrics = ImmutableList.of();
    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 1;

    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
  }

  @Test
  public void testGetEvictionCandidates_selectOnlyWorker() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList("dummy", 1), emptyEntryList()));
    WorkerKey key = createWorkerKey("dummy", fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);
    ImmutableList<WorkerMetric> workerMetrics =
        ImmutableList.of(
            WorkerMetric.create(
                createWorkerProperties(w1.getWorkerId(), 1L, "dummy", w1.getWorkerKey().hashCode()),
                createWorkerStat(2000),
                true));
    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 1;
    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);
    manager.setEventBus(eventBus);
    WorkerEvictedEvent event =
        new WorkerEvictedEvent(w1.getWorkerKey().hashCode(), w1.getWorkerKey().getMnemonic());

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
    assertThat(eventRecorder.workerEvictedEvents).containsExactly(event);
  }

  @Test
  public void testGetEvictionCandidates_evictLargestWorkers() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList("dummy", 3), emptyEntryList()));
    WorkerKey key = createWorkerKey("dummy", fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    Worker w2 = workerPool.borrowObject(key);
    Worker w3 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);
    workerPool.returnObject(key, w2);
    workerPool.returnObject(key, w3);

    ImmutableList<WorkerMetric> workerMetrics =
        ImmutableList.of(
            WorkerMetric.create(
                createWorkerProperties(w1.getWorkerId(), 1L, "dummy"),
                createWorkerStat(2000),
                true),
            WorkerMetric.create(
                createWorkerProperties(w2.getWorkerId(), 2L, "dummy"),
                createWorkerStat(1000),
                true),
            WorkerMetric.create(
                createWorkerProperties(w3.getWorkerId(), 3L, "dummy"),
                createWorkerStat(4000),
                true));

    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 2;
    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(3);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
    assertThat(workerPool.borrowObject(key).getWorkerId()).isEqualTo(w2.getWorkerId());
  }

  @Test
  public void testGetEvictionCandidates_numberOfWorkersIsMoreThanDeafaultNumTests()
      throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList("dummy", 4), emptyEntryList()));
    WorkerKey key = createWorkerKey("dummy", fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    Worker w2 = workerPool.borrowObject(key);
    Worker w3 = workerPool.borrowObject(key);
    Worker w4 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);
    workerPool.returnObject(key, w2);
    workerPool.returnObject(key, w3);
    workerPool.returnObject(key, w4);

    ImmutableList<WorkerMetric> workerMetrics =
        ImmutableList.of(
            WorkerMetric.create(
                createWorkerProperties(w1.getWorkerId(), 1L, "dummy"),
                createWorkerStat(2000),
                true),
            WorkerMetric.create(
                createWorkerProperties(w2.getWorkerId(), 2L, "dummy"),
                createWorkerStat(2000),
                true),
            WorkerMetric.create(
                createWorkerProperties(w3.getWorkerId(), 3L, "dummy"),
                createWorkerStat(4000),
                true),
            WorkerMetric.create(
                createWorkerProperties(w4.getWorkerId(), 4L, "dummy"),
                createWorkerStat(4000),
                true));

    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 1;
    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(4);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
  }

  @Test
  public void testGetEvictionCandidates_evictWorkerWithSameMenmonicButDifferentKeys()
      throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList("dummy", 3), emptyEntryList()));
    WorkerKey key1 = createWorkerKey("dummy", fileSystem);
    WorkerKey key2 = createWorkerKey("dummy", fileSystem, true);

    Worker w1 = workerPool.borrowObject(key1);
    Worker w2 = workerPool.borrowObject(key2);
    Worker w3 = workerPool.borrowObject(key2);
    workerPool.returnObject(key1, w1);
    workerPool.returnObject(key2, w2);
    workerPool.returnObject(key2, w3);

    ImmutableList<WorkerMetric> workerMetrics =
        ImmutableList.of(
            WorkerMetric.create(
                createWorkerProperties(w1.getWorkerId(), 1L, "dummy"),
                createWorkerStat(3000),
                true),
            WorkerMetric.create(
                createWorkerProperties(w2.getWorkerId(), 2L, "dummy"),
                createWorkerStat(3000),
                true),
            WorkerMetric.create(
                createWorkerProperties(w3.getWorkerId(), 3L, "dummy"),
                createWorkerStat(1000),
                true));

    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 2;
    options.workerVerbose = true;
    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);

    assertThat(workerPool.getNumIdlePerKey(key1)).isEqualTo(1);
    assertThat(workerPool.getNumIdlePerKey(key2)).isEqualTo(2);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key1)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key1)).isEqualTo(0);

    assertThat(workerPool.getNumIdlePerKey(key2)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key2)).isEqualTo(0);
  }

  @Test
  public void testGetEvictionCandidates_evictOnlyIdleWorkers() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList("dummy", 3), emptyEntryList()));
    WorkerKey key = createWorkerKey("dummy", fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    Worker w2 = workerPool.borrowObject(key);
    Worker w3 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);
    workerPool.returnObject(key, w2);

    ImmutableList<WorkerMetric> workerMetrics =
        ImmutableList.of(
            WorkerMetric.create(
                createWorkerProperties(w1.getWorkerId(), 1L, "dummy"),
                createWorkerStat(2000),
                true),
            WorkerMetric.create(
                createWorkerProperties(w2.getWorkerId(), 2L, "dummy"),
                createWorkerStat(1000),
                true),
            WorkerMetric.create(
                createWorkerProperties(w3.getWorkerId(), 3L, "dummy"),
                createWorkerStat(4000),
                true));

    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 2;

    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(2);
    assertThat(workerPool.getNumActive(key)).isEqualTo(1);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key)).isEqualTo(1);
  }

  @Test
  public void testGetEvictionCandidates_evictDifferentWorkerKeys() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList("dummy", 2, "smart", 2), emptyEntryList()));
    WorkerKey key1 = createWorkerKey("dummy", fileSystem);
    WorkerKey key2 = createWorkerKey("smart", fileSystem);
    Worker w1 = workerPool.borrowObject(key1);
    Worker w2 = workerPool.borrowObject(key1);
    Worker w3 = workerPool.borrowObject(key2);
    Worker w4 = workerPool.borrowObject(key2);
    workerPool.returnObject(key1, w1);
    workerPool.returnObject(key1, w2);
    workerPool.returnObject(key2, w3);
    workerPool.returnObject(key2, w4);

    ImmutableList<WorkerMetric> workerMetrics =
        ImmutableList.of(
            WorkerMetric.create(
                createWorkerProperties(w1.getWorkerId(), 1L, "dummy"),
                createWorkerStat(1000),
                true),
            WorkerMetric.create(
                createWorkerProperties(w2.getWorkerId(), 2L, "dummy"),
                createWorkerStat(4000),
                true),
            WorkerMetric.create(
                createWorkerProperties(w3.getWorkerId(), 3L, "smart"),
                createWorkerStat(3000),
                true),
            WorkerMetric.create(
                createWorkerProperties(w4.getWorkerId(), 4L, "smart"),
                createWorkerStat(1000),
                true));

    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 2;

    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);

    assertThat(workerPool.getNumIdlePerKey(key1)).isEqualTo(2);
    assertThat(workerPool.getNumActive(key1)).isEqualTo(0);
    assertThat(workerPool.getNumIdlePerKey(key2)).isEqualTo(2);
    assertThat(workerPool.getNumActive(key2)).isEqualTo(0);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key1)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key1)).isEqualTo(0);
    assertThat(workerPool.borrowObject(key1).getWorkerId()).isEqualTo(w1.getWorkerId());

    assertThat(workerPool.getNumIdlePerKey(key2)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key2)).isEqualTo(0);
    assertThat(workerPool.borrowObject(key2).getWorkerId()).isEqualTo(w4.getWorkerId());
  }

  @Test
  public void testGetEvictionCandidates_testDoomedWorkers() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList("dummy", 2), emptyEntryList()));
    WorkerKey key = createWorkerKey("dummy", fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    Worker w2 = workerPool.borrowObject(key);

    ImmutableList<WorkerMetric> workerMetrics =
        ImmutableList.of(
            WorkerMetric.create(
                createWorkerProperties(w1.getWorkerId(), 1L, "dummy"),
                createWorkerStat(2000),
                true),
            WorkerMetric.create(
                createWorkerProperties(w2.getWorkerId(), 2L, "dummy"),
                createWorkerStat(2000),
                true));

    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 1;
    options.shrinkWorkerPool = true;

    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key)).isEqualTo(2);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key)).isEqualTo(2);
    assertThat(workerPool.getDoomedWorkers())
        .isEqualTo(Sets.newHashSet(w1.getWorkerId(), w2.getWorkerId()));
  }

  @Test
  public void testGetEvictionCandidates_testDoomedAndIdleWorkers() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList("dummy", 5), emptyEntryList()));
    WorkerKey key = createWorkerKey("dummy", fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    Worker w2 = workerPool.borrowObject(key);
    Worker w3 = workerPool.borrowObject(key);
    Worker w4 = workerPool.borrowObject(key);
    Worker w5 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);
    workerPool.returnObject(key, w2);

    ImmutableList<WorkerMetric> workerMetrics =
        ImmutableList.of(
            WorkerMetric.create(
                createWorkerProperties(w1.getWorkerId(), 1L, "dummy"),
                createWorkerStat(2000),
                true),
            WorkerMetric.create(
                createWorkerProperties(w2.getWorkerId(), 2L, "dummy"),
                createWorkerStat(1000),
                true),
            WorkerMetric.create(
                createWorkerProperties(w3.getWorkerId(), 3L, "dummy"),
                createWorkerStat(4000),
                true),
            WorkerMetric.create(
                createWorkerProperties(w4.getWorkerId(), 4L, "dummy"),
                createWorkerStat(5000),
                true),
            WorkerMetric.create(
                createWorkerProperties(w5.getWorkerId(), 5L, "dummy"),
                createWorkerStat(1000),
                true));

    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 2;
    options.shrinkWorkerPool = true;

    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(2);
    assertThat(workerPool.getNumActive(key)).isEqualTo(3);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key)).isEqualTo(3);
    assertThat(workerPool.getDoomedWorkers())
        .isEqualTo(Sets.newHashSet(w3.getWorkerId(), w4.getWorkerId()));
  }

  @Test
  public void evictWorkers_testMultiplexWorkers() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, emptyEntryList(), entryList("dummy", 2)));
    WorkerKey key =
        createWorkerKey("dummy", fileSystem, /* multiplex= */ true, /* sandboxed= */ false);
    Worker w1 = workerPool.borrowObject(key);
    Worker w2 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);
    workerPool.returnObject(key, w2);
    ImmutableList<WorkerMetric> workerMetrics =
        ImmutableList.of(
            WorkerMetric.create(
                createMultiplexWorkerProperties(
                    ImmutableList.of(w1.getWorkerId(), w2.getWorkerId()), 1L, "dummy"),
                createWorkerStat(4000),
                true));
    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 1;
    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
  }

  @Test
  public void evictWorkers_doomMultiplexWorker() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, emptyEntryList(), entryList("dummy", 2)));
    WorkerKey key =
        createWorkerKey("dummy", fileSystem, /* multiplex= */ true, /* sandboxed= */ false);
    Worker w1 = workerPool.borrowObject(key);
    Worker w2 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);
    ImmutableList<WorkerMetric> workerMetrics =
        ImmutableList.of(
            WorkerMetric.create(
                createMultiplexWorkerProperties(
                    ImmutableList.of(w1.getWorkerId(), w2.getWorkerId()), 1L, "dummy"),
                createWorkerStat(4000),
                true));
    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 1;
    options.shrinkWorkerPool = true;
    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key)).isEqualTo(1);
    assertThat(workerPool.getDoomedWorkers())
        .isEqualTo(Sets.newHashSet(w1.getWorkerId(), w2.getWorkerId()));
  }

  private static WorkerMetric.WorkerProperties createWorkerProperties(
      int workerId, long processId, String mnemonic, int workerKeyHash) {
    return WorkerMetric.WorkerProperties.create(
        ImmutableList.of(workerId),
        processId,
        mnemonic,
        /* isMultiplex= */ false,
        /* isSandboxed= */ false,
        workerKeyHash);
  }

  private static WorkerMetric.WorkerProperties createWorkerProperties(
      int workerId, long processId, String mnemonic) {
    return createWorkerProperties(workerId, processId, mnemonic, /* workerKeyHash= */ 0);
  }

  private static WorkerMetric.WorkerProperties createMultiplexWorkerProperties(
      ImmutableList<Integer> workerIds, long processId, String mnemonic) {
    return WorkerMetric.WorkerProperties.create(
        workerIds, processId, mnemonic, true, /* isSandboxed= */ false, /* workerKeyHash= */ 0);
  }

  private static WorkerMetric.WorkerStat createWorkerStat(int memoryUsage) {
    return WorkerMetric.WorkerStat.create(
        memoryUsage, /*lastCallTimestamp */ Instant.now(), /* timestamp*/ Instant.now());
  }

  private static ImmutableList<Entry<String, Integer>> emptyEntryList() {
    return ImmutableList.of();
  }

  private static ImmutableList<Entry<String, Integer>> entryList(String key1, int value1) {
    return ImmutableList.of(Maps.immutableEntry(key1, value1));
  }

  private static ImmutableList<Entry<String, Integer>> entryList(
      String key1, int value1, String key2, int value2) {
    return ImmutableList.of(Maps.immutableEntry(key1, value1), Maps.immutableEntry(key2, value2));
  }
}
