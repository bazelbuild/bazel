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

import static com.google.common.collect.ImmutableList.toImmutableList;
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

  private static final String DUMMY_MNEMONIC = "dummy";
  private static final long PROCESS_ID_1 = 1L;
  private static final long PROCESS_ID_2 = 2L;
  private static final long PROCESS_ID_3 = 3L;
  private static final long PROCESS_ID_4 = 4L;
  private static final long PROCESS_ID_5 = 5L;

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
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 1), emptyEntryList()));
    WorkerKey key = createWorkerKey(DUMMY_MNEMONIC, fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);
    ImmutableList<WorkerProcessMetrics> workerMetrics =
        ImmutableList.of(createWorkerMetric(w1, PROCESS_ID_1, /* memoryInKb= */ 1024));
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
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 1), emptyEntryList()));
    WorkerKey key = createWorkerKey(DUMMY_MNEMONIC, fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);

    ImmutableList<WorkerProcessMetrics> workerMetrics =
        ImmutableList.of(createWorkerMetric(w1, PROCESS_ID_1, /* memoryInKb= */ 1024));
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
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 1), emptyEntryList()));
    WorkerKey key = createWorkerKey(DUMMY_MNEMONIC, fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);

    ImmutableList<WorkerProcessMetrics> workerMetrics = ImmutableList.of();
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
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 1), emptyEntryList()));
    WorkerKey key = createWorkerKey(DUMMY_MNEMONIC, fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);
    ImmutableList<WorkerProcessMetrics> workerMetrics =
        ImmutableList.of(createWorkerMetric(w1, PROCESS_ID_1, /* memoryInKb= */ 2000));
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
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 3), emptyEntryList()));
    WorkerKey key = createWorkerKey(DUMMY_MNEMONIC, fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    Worker w2 = workerPool.borrowObject(key);
    Worker w3 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);
    workerPool.returnObject(key, w2);
    workerPool.returnObject(key, w3);

    ImmutableList<WorkerProcessMetrics> workerMetrics =
        ImmutableList.of(
            createWorkerMetric(w1, PROCESS_ID_1, /* memoryInKb= */ 2000),
            createWorkerMetric(w2, PROCESS_ID_2, /* memoryInKb= */ 1000),
            createWorkerMetric(w3, PROCESS_ID_3, /* memoryInKb= */ 4000));

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
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 4), emptyEntryList()));
    WorkerKey key = createWorkerKey(DUMMY_MNEMONIC, fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    Worker w2 = workerPool.borrowObject(key);
    Worker w3 = workerPool.borrowObject(key);
    Worker w4 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);
    workerPool.returnObject(key, w2);
    workerPool.returnObject(key, w3);
    workerPool.returnObject(key, w4);

    ImmutableList<WorkerProcessMetrics> workerMetrics =
        ImmutableList.of(
            createWorkerMetric(w1, PROCESS_ID_1, /* memoryInKb= */ 2000),
            createWorkerMetric(w2, PROCESS_ID_2, /* memoryInKb= */ 2000),
            createWorkerMetric(w3, PROCESS_ID_3, /* memoryInKb= */ 4000),
            createWorkerMetric(w4, PROCESS_ID_4, /* memoryInKb= */ 4000));

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
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 3), emptyEntryList()));
    WorkerKey key1 = createWorkerKey(DUMMY_MNEMONIC, fileSystem);
    WorkerKey key2 = createWorkerKey(DUMMY_MNEMONIC, fileSystem, true);

    Worker w1 = workerPool.borrowObject(key1);
    Worker w2 = workerPool.borrowObject(key2);
    Worker w3 = workerPool.borrowObject(key2);
    workerPool.returnObject(key1, w1);
    workerPool.returnObject(key2, w2);
    workerPool.returnObject(key2, w3);

    ImmutableList<WorkerProcessMetrics> workerMetrics =
        ImmutableList.of(
            createWorkerMetric(w1, PROCESS_ID_1, /* memoryInKb= */ 3000),
            createWorkerMetric(w2, PROCESS_ID_2, /* memoryInKb= */ 3000),
            createWorkerMetric(w3, PROCESS_ID_3, /* memoryInKb= */ 1000));

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
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 3), emptyEntryList()));
    WorkerKey key = createWorkerKey(DUMMY_MNEMONIC, fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    Worker w2 = workerPool.borrowObject(key);
    Worker w3 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);
    workerPool.returnObject(key, w2);

    ImmutableList<WorkerProcessMetrics> workerMetrics =
        ImmutableList.of(
            createWorkerMetric(w1, PROCESS_ID_1, /* memoryInKb= */ 2000),
            createWorkerMetric(w2, PROCESS_ID_2, /* memoryInKb= */ 1000),
            createWorkerMetric(w3, PROCESS_ID_3, /* memoryInKb= */ 4000));

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
            new WorkerPoolConfig(
                factoryMock, entryList(DUMMY_MNEMONIC, 2, "smart", 2), emptyEntryList()));
    WorkerKey key1 = createWorkerKey(DUMMY_MNEMONIC, fileSystem);
    WorkerKey key2 = createWorkerKey("smart", fileSystem);
    Worker w1 = workerPool.borrowObject(key1);
    Worker w2 = workerPool.borrowObject(key1);
    Worker w3 = workerPool.borrowObject(key2);
    Worker w4 = workerPool.borrowObject(key2);
    workerPool.returnObject(key1, w1);
    workerPool.returnObject(key1, w2);
    workerPool.returnObject(key2, w3);
    workerPool.returnObject(key2, w4);

    ImmutableList<WorkerProcessMetrics> workerMetrics =
        ImmutableList.of(
            createWorkerMetric(w1, PROCESS_ID_1, /* memoryInKb= */ 1000),
            createWorkerMetric(w2, PROCESS_ID_2, /* memoryInKb= */ 4000),
            createWorkerMetric(w3, PROCESS_ID_3, /* memoryInKb= */ 3000),
            createWorkerMetric(w4, PROCESS_ID_4, /* memoryInKb= */ 1000));

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
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 2), emptyEntryList()));
    WorkerKey key = createWorkerKey(DUMMY_MNEMONIC, fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    Worker w2 = workerPool.borrowObject(key);

    ImmutableList<WorkerProcessMetrics> workerMetrics =
        ImmutableList.of(
            createWorkerMetric(w1, PROCESS_ID_1, /* memoryInKb= */ 2000),
            createWorkerMetric(w2, PROCESS_ID_2, /* memoryInKb= */ 2000));

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
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 5), emptyEntryList()));
    WorkerKey key = createWorkerKey(DUMMY_MNEMONIC, fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    Worker w2 = workerPool.borrowObject(key);
    Worker w3 = workerPool.borrowObject(key);
    Worker w4 = workerPool.borrowObject(key);
    Worker w5 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);
    workerPool.returnObject(key, w2);

    ImmutableList<WorkerProcessMetrics> workerMetrics =
        ImmutableList.of(
            createWorkerMetric(w1, PROCESS_ID_1, /* memoryInKb= */ 2000),
            createWorkerMetric(w2, PROCESS_ID_2, /* memoryInKb= */ 1000),
            createWorkerMetric(w3, PROCESS_ID_3, /* memoryInKb= */ 4000),
            createWorkerMetric(w4, PROCESS_ID_4, /* memoryInKb= */ 5000),
            createWorkerMetric(w5, PROCESS_ID_5, /* memoryInKb= */ 1000));

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
            new WorkerPoolConfig(factoryMock, emptyEntryList(), entryList(DUMMY_MNEMONIC, 2)));
    WorkerKey key =
        createWorkerKey(DUMMY_MNEMONIC, fileSystem, /* multiplex= */ true, /* sandboxed= */ false);
    Worker w1 = workerPool.borrowObject(key);
    Worker w2 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);
    workerPool.returnObject(key, w2);
    ImmutableList<WorkerProcessMetrics> workerMetrics =
        ImmutableList.of(
            createMultiplexWorkerMetric(
                ImmutableList.of(w1, w2), PROCESS_ID_1, /* memoryInKb= */ 4000));
    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 1;
    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
  }

  @Test
  public void evictWorkers_doomMultiplexWorker() throws Exception {
    String dummyMnemonic = DUMMY_MNEMONIC;
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, emptyEntryList(), entryList(dummyMnemonic, 2)));
    WorkerKey key =
        createWorkerKey(dummyMnemonic, fileSystem, /* multiplex= */ true, /* sandboxed= */ false);
    Worker w1 = workerPool.borrowObject(key);
    Worker w2 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);
    ImmutableList<WorkerProcessMetrics> workerMetrics =
        ImmutableList.of(
            createMultiplexWorkerMetric(
                ImmutableList.of(w1, w2), PROCESS_ID_1, /* memoryInKb= */ 4000));
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

  private static final Instant DEFAULT_INSTANT = BlazeClock.instance().now();

  private static WorkerProcessMetrics createWorkerMetric(
      Worker worker, long processId, int memoryInKb) {
    // We need to override the processId.
    WorkerProcessMetrics wm =
        new WorkerProcessMetrics(
            worker.getWorkerId(),
            processId,
            worker.getWorkerKey().getMnemonic(),
            worker.getWorkerKey().isMultiplex(),
            worker.getWorkerKey().isSandboxed(),
            worker.getWorkerKey().hashCode());
    wm.addCollectedMetrics(
        memoryInKb, /* isMeasurable= */ true, /* collectionTime= */ DEFAULT_INSTANT);
    return wm;
  }

  private static WorkerProcessMetrics createMultiplexWorkerMetric(
      ImmutableList<Worker> workers, long processId, int memoryInKb) {
    WorkerProcessMetrics workerProcessMetrics =
        new WorkerProcessMetrics(
            workers.stream().map(Worker::getWorkerId).collect(toImmutableList()),
            processId,
            workers.get(0).getWorkerKey().getMnemonic(),
            workers.get(0).getWorkerKey().isMultiplex(),
            workers.get(0).getWorkerKey().isSandboxed(),
            workers.get(0).getWorkerKey().hashCode());
    workerProcessMetrics.addCollectedMetrics(
        memoryInKb, /* isMeasurable= */ true, /* collectionTime= */ DEFAULT_INSTANT);
    return workerProcessMetrics;
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
