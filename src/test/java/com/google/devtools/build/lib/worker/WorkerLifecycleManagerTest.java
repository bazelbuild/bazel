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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.lib.worker.WorkerPoolImpl.WorkerPoolConfig;
import com.google.devtools.build.lib.worker.WorkerProcessStatus.Status;
import java.time.Instant;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Set;
import org.apache.commons.pool2.PooledObject;
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

  private final EventBus eventBus = new EventBus();

  @Rule public final MockitoRule mockito = MockitoJUnit.rule();
  @Mock WorkerFactory factoryMock;
  private FileSystem fileSystem;
  private EventRecorder eventRecorder = new EventRecorder();

  private static class EventRecorder {
    public final Set<WorkerEvictedEvent> workerEvictedEvents;
    public final Set<WorkerDestroyedEvent> workerDestroyedEvents;

    public EventRecorder() {
      workerEvictedEvents = new HashSet<>();
      workerDestroyedEvents = new HashSet<>();
    }

    @Subscribe
    public synchronized void handleWorkerEvictedEvent(WorkerEvictedEvent event) {
      // Check to ensure that there should never be duplicate WorkerEvictedEvent(s) posted by the
      // WorkerLifecycleManager as that would break the pool stats counts in the MetricsCollector.
      assertThat(workerEvictedEvents).doesNotContain(event);
      this.workerEvictedEvents.add(event);
    }

    @Subscribe
    public synchronized void handleWorkerDestroyedEvent(WorkerDestroyedEvent event) {
      // Check to ensure that there should never be duplicate WorkerDestroyedEvent(s) posted by the
      // WorkerLifecycleManager as that would break the pool stats counts in the MetricsCollector.
      assertThat(workerDestroyedEvents).doesNotContain(event);
      this.workerDestroyedEvents.add(event);
    }

    public void assertContainsEvictedEventsOfWorkers(ImmutableList<Worker> workers) {
      assertThat(workerEvictedEvents)
          .containsExactlyElementsIn(
              workers.stream()
                  .map(
                      w ->
                          new WorkerEvictedEvent(
                              w.getWorkerId(),
                              w.getWorkerKey().hashCode(),
                              w.getWorkerKey().getMnemonic()))
                  .collect(toImmutableList()));
    }

    public void assertContainsDestroyedEventsOfWorkers(ImmutableList<Worker> workers) {
      assertThat(workerDestroyedEvents)
          .containsExactlyElementsIn(
              workers.stream()
                  .map(
                      w ->
                          new WorkerDestroyedEvent(
                              w.getWorkerId(),
                              w.getWorkerKey().hashCode(),
                              w.getWorkerKey().getMnemonic(),
                              w.getStatus()))
                  .collect(toImmutableList()));
    }
  }

  private static final String DUMMY_MNEMONIC = "dummy";
  private static final long PROCESS_ID_1 = 1L;
  private static final long PROCESS_ID_2 = 2L;
  private static final long PROCESS_ID_3 = 3L;
  private static final long PROCESS_ID_4 = 4L;
  private static final long PROCESS_ID_5 = 5L;

  private int workerIds = 1;

  @Before
  public void setUp() throws Exception {
    fileSystem = new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);
    eventRecorder = new EventRecorder();
    eventBus.register(eventRecorder);

    doAnswer(
            args -> {
              WorkerKey key = args.getArgument(0);
              if (key.isMultiplex()) {
                WorkerMultiplexer multiplexer =
                    WorkerMultiplexerManager.getInstance(key, fileSystem.getPath("/logDir"));
                return new DefaultPooledObject<>(
                    new WorkerProxy(
                        key,
                        workerIds++,
                        multiplexer.getLogFile(),
                        multiplexer,
                        key.getExecRoot()));
              }
              return new DefaultPooledObject<>(
                  new SingleplexWorker(
                      key,
                      workerIds++,
                      fileSystem.getPath("/workDir"),
                      fileSystem.getPath("/logDir")));
            })
        .when(factoryMock)
        .makeObject(any());
    doAnswer(
            args -> {
              PooledObject<Worker> obj = args.getArgument(1);
              return obj.getObject().getStatus().isValid();
            })
        .when(factoryMock)
        .validateObject(any(), any());
    doAnswer(
            args -> {
              PooledObject<Worker> obj = args.getArgument(1);
              Worker worker = obj.getObject();
              worker.destroy();
              eventBus.post(
                  new WorkerDestroyedEvent(
                      worker.getWorkerId(),
                      worker.getWorkerKey().hashCode(),
                      worker.getWorkerKey().getMnemonic(),
                      worker.getStatus()));
              return null;
            })
        .when(factoryMock)
        .destroyObject(any(), any());
  }

  @Test
  public void testEvictWorkers_doNothing_lowMemoryUsage() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 1), emptyEntryList()));
    workerPool.setEventBus(eventBus);
    WorkerKey key = createWorkerKey(DUMMY_MNEMONIC, fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);
    ImmutableList<WorkerProcessMetrics> workerMetrics =
        ImmutableList.of(createWorkerMetric(w1, PROCESS_ID_1, /* memoryInKb= */ 1000));
    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 1000 * 100;

    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);
    manager.setEventBus(eventBus);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
    // It should still have a valid status since it was not killed / marked to be killed.
    assertThat(w1.getStatus().isValid()).isTrue();
    assertThat(eventRecorder.workerEvictedEvents).isEmpty();
    assertThat(eventRecorder.workerDestroyedEvents).isEmpty();
  }

  @Test
  public void testEvictWorkers_doNothing_zeroThreshold() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 1), emptyEntryList()));
    workerPool.setEventBus(eventBus);
    WorkerKey key = createWorkerKey(DUMMY_MNEMONIC, fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);

    ImmutableList<WorkerProcessMetrics> workerMetrics =
        ImmutableList.of(createWorkerMetric(w1, PROCESS_ID_1, /* memoryInKb= */ 1000));
    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 0;

    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);
    manager.setEventBus(eventBus);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
    // It should still have a valid status since it was not killed / marked to be killed.
    assertThat(w1.getStatus().isValid()).isTrue();
    assertThat(eventRecorder.workerEvictedEvents).isEmpty();
    assertThat(eventRecorder.workerDestroyedEvents).isEmpty();
  }

  @Test
  public void testEvictWorkers_doNothing_emptyMetrics() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 1), emptyEntryList()));
    workerPool.setEventBus(eventBus);
    WorkerKey key = createWorkerKey(DUMMY_MNEMONIC, fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);

    ImmutableList<WorkerProcessMetrics> workerMetrics = ImmutableList.of();
    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 1;

    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);
    manager.setEventBus(eventBus);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
    // It should still have a valid status since it was not killed / marked to be killed.
    assertThat(w1.getStatus().isValid()).isTrue();
    assertThat(eventRecorder.workerEvictedEvents).isEmpty();
    assertThat(eventRecorder.workerDestroyedEvents).isEmpty();
  }

  @Test
  public void testGetEvictionCandidates_selectOnlyWorker() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 1), emptyEntryList()));
    workerPool.setEventBus(eventBus);
    WorkerKey key = createWorkerKey(DUMMY_MNEMONIC, fileSystem);
    Worker w1 = workerPool.borrowObject(key);
    workerPool.returnObject(key, w1);
    ImmutableList<WorkerProcessMetrics> workerMetrics =
        ImmutableList.of(createWorkerMetric(w1, PROCESS_ID_1, /* memoryInKb= */ 2000));
    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 1;
    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);
    manager.setEventBus(eventBus);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
    // It should still have a valid status since it was not killed / marked to be killed.
    assertThat(w1.getStatus().isValid()).isTrue();

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
    // Directly killed since it is already returned.
    assertThat(w1.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    eventRecorder.assertContainsEvictedEventsOfWorkers(ImmutableList.of(w1));
    eventRecorder.assertContainsDestroyedEventsOfWorkers(ImmutableList.of(w1));
  }

  @Test
  public void testGetEvictionCandidates_evictLargestWorkers() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 3), emptyEntryList()));
    workerPool.setEventBus(eventBus);
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
    manager.setEventBus(eventBus);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(3);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
    assertThat(w1.getStatus().isValid()).isTrue();
    assertThat(w2.getStatus().isValid()).isTrue();
    assertThat(w3.getStatus().isValid()).isTrue();

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
    // Only w1 and w3 should have been killed.
    assertThat(w1.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    assertThat(w2.getStatus().isValid()).isTrue();
    assertThat(w3.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    eventRecorder.assertContainsEvictedEventsOfWorkers(ImmutableList.of(w1, w3));
    eventRecorder.assertContainsDestroyedEventsOfWorkers(ImmutableList.of(w1, w3));
  }

  @Test
  public void testGetEvictionCandidates_numberOfWorkersIsMoreThanDefaultNumTests()
      throws Exception {
    // The default number of tests is 3.
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 4), emptyEntryList()));
    workerPool.setEventBus(eventBus);
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
    manager.setEventBus(eventBus);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(4);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
    assertThat(w1.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    assertThat(w2.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    assertThat(w3.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    assertThat(w4.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    eventRecorder.assertContainsEvictedEventsOfWorkers(ImmutableList.of(w1, w2, w3, w4));
    eventRecorder.assertContainsDestroyedEventsOfWorkers(ImmutableList.of(w1, w2, w3, w4));
  }

  @Test
  public void testGetEvictionCandidates_evictWorkerWithSameMenmonicButDifferentKeys()
      throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 3), emptyEntryList()));
    workerPool.setEventBus(eventBus);
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
    manager.setEventBus(eventBus);

    assertThat(workerPool.getNumIdlePerKey(key1)).isEqualTo(1);
    assertThat(workerPool.getNumIdlePerKey(key2)).isEqualTo(2);
    assertThat(w1.getStatus().isValid()).isTrue();
    assertThat(w2.getStatus().isValid()).isTrue();
    assertThat(w3.getStatus().isValid()).isTrue();

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key1)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key1)).isEqualTo(0);

    assertThat(workerPool.getNumIdlePerKey(key2)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key2)).isEqualTo(0);
    // Only w3 shouldn't be killed.
    assertThat(w1.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    assertThat(w2.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    assertThat(w3.getStatus().isValid()).isTrue();
    eventRecorder.assertContainsEvictedEventsOfWorkers(ImmutableList.of(w1, w2));
    eventRecorder.assertContainsDestroyedEventsOfWorkers(ImmutableList.of(w1, w2));
  }

  @Test
  public void testGetEvictionCandidates_evictOnlyIdleWorkers() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 3), emptyEntryList()));
    workerPool.setEventBus(eventBus);
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
    manager.setEventBus(eventBus);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(2);
    assertThat(workerPool.getNumActive(key)).isEqualTo(1);
    assertThat(w1.getStatus().isValid()).isTrue();
    assertThat(w2.getStatus().isValid()).isTrue();
    assertThat(w3.getStatus().isValid()).isTrue();

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key)).isEqualTo(1);
    assertThat(w1.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    assertThat(w2.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    // w3 is not killed because we're not shrinking the worker pool.
    assertThat(w3.getStatus().isValid()).isTrue();
    eventRecorder.assertContainsEvictedEventsOfWorkers(ImmutableList.of(w1, w2));
    eventRecorder.assertContainsDestroyedEventsOfWorkers(ImmutableList.of(w1, w2));
  }

  @Test
  public void testGetEvictionCandidates_evictDifferentWorkerKeys() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(
                factoryMock, entryList(DUMMY_MNEMONIC, 2, "smart", 2), emptyEntryList()));
    workerPool.setEventBus(eventBus);
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
    manager.setEventBus(eventBus);

    assertThat(workerPool.getNumIdlePerKey(key1)).isEqualTo(2);
    assertThat(workerPool.getNumActive(key1)).isEqualTo(0);
    assertThat(workerPool.getNumIdlePerKey(key2)).isEqualTo(2);
    assertThat(workerPool.getNumActive(key2)).isEqualTo(0);
    assertThat(w1.getStatus().isValid()).isTrue();
    assertThat(w2.getStatus().isValid()).isTrue();
    assertThat(w3.getStatus().isValid()).isTrue();
    assertThat(w4.getStatus().isValid()).isTrue();

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key1)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key1)).isEqualTo(0);
    assertThat(workerPool.borrowObject(key1).getWorkerId()).isEqualTo(w1.getWorkerId());

    assertThat(workerPool.getNumIdlePerKey(key2)).isEqualTo(1);
    assertThat(workerPool.getNumActive(key2)).isEqualTo(0);
    assertThat(workerPool.borrowObject(key2).getWorkerId()).isEqualTo(w4.getWorkerId());

    assertThat(w1.getStatus().isValid()).isTrue();
    assertThat(w2.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    assertThat(w3.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    assertThat(w4.getStatus().isValid()).isTrue();
    eventRecorder.assertContainsEvictedEventsOfWorkers(ImmutableList.of(w2, w3));
    eventRecorder.assertContainsDestroyedEventsOfWorkers(ImmutableList.of(w2, w3));
  }

  @Test
  public void testGetEvictionCandidates_testDoomedWorkers() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 2), emptyEntryList()));
    workerPool.setEventBus(eventBus);
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
    manager.setEventBus(eventBus);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key)).isEqualTo(2);
    assertThat(w1.getStatus().isValid()).isTrue();
    assertThat(w2.getStatus().isValid()).isTrue();

    manager.evictWorkers(workerMetrics);

    assertThat(w1.getStatus().get()).isEqualTo(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);
    assertThat(w2.getStatus().get()).isEqualTo(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);

    // Return only one worker.
    workerPool.returnObject(key, w1);

    // w1 gets destroyed when it is returned, so there are 0 idle workers.
    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key)).isEqualTo(1);
    // Since w1 is already returned, it is killed on return.
    assertThat(w1.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    // Since w2 is still active, it is marked to be killed which will happen when it is returned.
    assertThat(w2.getStatus().get()).isEqualTo(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);
    eventRecorder.assertContainsEvictedEventsOfWorkers(ImmutableList.of(w1));
    eventRecorder.assertContainsDestroyedEventsOfWorkers(ImmutableList.of(w1));

    // Return the remaining worker.
    workerPool.returnObject(key, w2);
    assertThat(w2.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    eventRecorder.assertContainsEvictedEventsOfWorkers(ImmutableList.of(w1, w2));
    eventRecorder.assertContainsDestroyedEventsOfWorkers(ImmutableList.of(w1, w2));
  }

  @Test
  public void testGetEvictionCandidates_testDoomedAndIdleWorkers() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, entryList(DUMMY_MNEMONIC, 5), emptyEntryList()));
    workerPool.setEventBus(eventBus);
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
    manager.setEventBus(eventBus);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(2);
    assertThat(workerPool.getNumActive(key)).isEqualTo(3);
    assertThat(w1.getStatus().isValid()).isTrue();
    assertThat(w2.getStatus().isValid()).isTrue();
    assertThat(w3.getStatus().isValid()).isTrue();
    assertThat(w4.getStatus().isValid()).isTrue();
    assertThat(w5.getStatus().isValid()).isTrue();

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key)).isEqualTo(3);
    // w1 and w2 are killed immediately.
    assertThat(w1.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    assertThat(w2.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    // w3 and w4 are killed only when returned.
    assertThat(w3.getStatus().get()).isEqualTo(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);
    assertThat(w4.getStatus().get()).isEqualTo(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);
    assertThat(w5.getStatus().isValid()).isTrue();
    eventRecorder.assertContainsEvictedEventsOfWorkers(ImmutableList.of(w1, w2));
    eventRecorder.assertContainsDestroyedEventsOfWorkers(ImmutableList.of(w1, w2));
  }

  @Test
  public void evictWorkers_testMultiplexWorkers() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, emptyEntryList(), entryList(DUMMY_MNEMONIC, 2)));
    workerPool.setEventBus(eventBus);
    WorkerKey key =
        createWorkerKey(DUMMY_MNEMONIC, fileSystem, /* multiplex= */ true, /* sandboxed= */ false);
    Worker w1 = workerPool.borrowObject(key);
    Worker w2 = workerPool.borrowObject(key);

    // Multiplex workers should share the same status instance.
    assertThat(w1.getStatus()).isSameInstanceAs(w2.getStatus());

    workerPool.returnObject(key, w1);
    workerPool.returnObject(key, w2);
    ImmutableList<WorkerProcessMetrics> workerMetrics =
        ImmutableList.of(
            createMultiplexWorkerMetric(
                ImmutableList.of(w1, w2), PROCESS_ID_1, /* memoryInKb= */ 4000));
    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 1;
    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);
    manager.setEventBus(eventBus);

    manager.evictWorkers(workerMetrics);

    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
    // Since both w1 and w2 have been returned, it is killed.
    assertThat(w1.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    eventRecorder.assertContainsEvictedEventsOfWorkers(ImmutableList.of(w1, w2));
    eventRecorder.assertContainsDestroyedEventsOfWorkers(ImmutableList.of(w1, w2));
  }

  @Test
  public void evictWorkers_doomMultiplexWorker() throws Exception {
    WorkerPoolImpl workerPool =
        new WorkerPoolImpl(
            new WorkerPoolConfig(factoryMock, emptyEntryList(), entryList(DUMMY_MNEMONIC, 2)));
    workerPool.setEventBus(eventBus);
    WorkerKey key =
        createWorkerKey(DUMMY_MNEMONIC, fileSystem, /* multiplex= */ true, /* sandboxed= */ false);
    Worker w1 = workerPool.borrowObject(key);
    Worker w2 = workerPool.borrowObject(key);

    // Multiplex workers should share the same status instance.
    assertThat(w1.getStatus()).isSameInstanceAs(w2.getStatus());

    workerPool.returnObject(key, w1);
    ImmutableList<WorkerProcessMetrics> workerMetrics =
        ImmutableList.of(
            createMultiplexWorkerMetric(
                ImmutableList.of(w1, w2), PROCESS_ID_1, /* memoryInKb= */ 4000));
    WorkerOptions options = new WorkerOptions();
    options.totalWorkerMemoryLimitMb = 1;
    options.shrinkWorkerPool = true;
    WorkerLifecycleManager manager = new WorkerLifecycleManager(workerPool, options);
    manager.setEventBus(eventBus);

    manager.evictWorkers(workerMetrics);

    // w1 should have been evicted already.
    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key)).isEqualTo(1);
    // Not yet killed because w2 is still alive (and both share a WorkerProcessStatus).
    assertThat(w1.getStatus().get()).isEqualTo(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);
    eventRecorder.assertContainsEvictedEventsOfWorkers(ImmutableList.of(w1));
    eventRecorder.assertContainsDestroyedEventsOfWorkers(ImmutableList.of(w1));

    workerPool.returnObject(key, w2);
    assertThat(workerPool.getNumIdlePerKey(key)).isEqualTo(0);
    assertThat(workerPool.getNumActive(key)).isEqualTo(0);
    // Status is only set to killed after the last worker proxy is destroyed.
    assertThat(w2.getStatus().get()).isEqualTo(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    eventRecorder.assertContainsEvictedEventsOfWorkers(ImmutableList.of(w1, w2));
    eventRecorder.assertContainsDestroyedEventsOfWorkers(ImmutableList.of(w1, w2));
  }

  private static final Instant DEFAULT_INSTANT = BlazeClock.instance().now();

  private static WorkerProcessMetrics createWorkerMetric(
      Worker worker, long processId, int memoryInKb) {
    // We need to override the processId.
    WorkerProcessMetrics wm =
        new WorkerProcessMetrics(
            worker.getWorkerId(),
            processId,
            worker.getStatus(),
            worker.getWorkerKey().getMnemonic(),
            worker.getWorkerKey().isMultiplex(),
            worker.getWorkerKey().isSandboxed(),
            worker.getWorkerKey().hashCode());
    wm.addCollectedMetrics(memoryInKb, /* collectionTime= */ DEFAULT_INSTANT);
    return wm;
  }

  private static WorkerProcessMetrics createMultiplexWorkerMetric(
      ImmutableList<Worker> workers, long processId, int memoryInKb) {
    WorkerProcessMetrics workerProcessMetrics =
        new WorkerProcessMetrics(
            workers.stream().map(Worker::getWorkerId).collect(toImmutableList()),
            processId,
            workers.get(0).getStatus(),
            workers.get(0).getWorkerKey().getMnemonic(),
            workers.get(0).getWorkerKey().isMultiplex(),
            workers.get(0).getWorkerKey().isSandboxed(),
            workers.get(0).getWorkerKey().hashCode());
    workerProcessMetrics.addCollectedMetrics(memoryInKb, /* collectionTime= */ DEFAULT_INSTANT);
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
