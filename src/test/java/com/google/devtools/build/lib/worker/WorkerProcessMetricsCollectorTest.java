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

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.WorkerMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.WorkerMetrics.WorkerStatus;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.metrics.CgroupsInfoCollector;
import com.google.devtools.build.lib.metrics.PsInfoCollector;
import com.google.devtools.build.lib.metrics.ResourceSnapshot;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.worker.WorkerProcessMetricsCollector.WorkerMetricsPublishComparator;
import com.google.devtools.build.lib.worker.WorkerProcessStatus.Status;
import java.time.Instant;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for the WorkerSpawnRunner. */
@RunWith(JUnit4.class)
public class WorkerProcessMetricsCollectorTest {

  private PsInfoCollector psInfoCollector;
  private CgroupsInfoCollector cgroupsInfoCollector;

  private WorkerProcessMetricsCollector spyCollector;
  ManualClock clock = new ManualClock();

  @Before
  public void setUp() {
    psInfoCollector = mock(PsInfoCollector.class);
    cgroupsInfoCollector = mock(CgroupsInfoCollector.class);
    spyCollector = spy(new WorkerProcessMetricsCollector(psInfoCollector, cgroupsInfoCollector));
    spyCollector.clear();
    spyCollector.setClock(clock);
  }

  private static final int WORKER_ID_1 = 1;
  private static final int WORKER_ID_2 = 2;
  private static final int WORKER_ID_3 = 3;
  private static final int WORKER_ID_4 = 4;
  private static final int WORKER_ID_5 = 5;
  private static final long PROCESS_ID_1 = 100L;
  private static final long PROCESS_ID_2 = 200L;
  private static final long PROCESS_ID_3 = 300L;
  private static final long PROCESS_ID_4 = 400L;
  private static final long PROCESS_ID_5 = 500L;
  private static final int WORKER_KEY_HASH_1 = 1;
  private static final int WORKER_KEY_HASH_2 = 2;
  private static final int WORKER_KEY_HASH_3 = 3;
  private static final int WORKER_KEY_HASH_4 = 4;
  private static final int WORKER_KEY_HASH_5 = 5;
  private static final String JAVAC_MNEMONIC = "Javac";
  private static final String CPP_COMPILE_MNEMONIC = "CppCompile";
  private static final String PROTO_MNEMONIC = "Proto";

  private void assertWorkerMetricContains(
      WorkerProcessMetrics workerMetric,
      ImmutableList<Integer> expectedWorkerIds,
      Long expectedProcessId,
      String expectedMnemonic,
      boolean expectedIsMultiplex,
      boolean expectedIsSandboxed,
      int expectedWorkerKeyHash,
      int expectedActionsExecuted,
      boolean expectedIsMeasurable,
      Instant expectedLastCallTime,
      Instant expectedCollectedTime) {
    assertThat(workerMetric).isNotNull();
    assertThat(workerMetric.getWorkerIds()).containsExactlyElementsIn(expectedWorkerIds);
    assertThat(workerMetric.getProcessId()).isEqualTo(expectedProcessId);
    assertThat(workerMetric.getMnemonic()).isEqualTo(expectedMnemonic);
    assertThat(workerMetric.isMultiplex()).isEqualTo(expectedIsMultiplex);
    assertThat(workerMetric.isSandboxed()).isEqualTo(expectedIsSandboxed);
    assertThat(workerMetric.getWorkerKeyHash()).isEqualTo(expectedWorkerKeyHash);
    assertThat(workerMetric.getActionsExecuted()).isEqualTo(expectedActionsExecuted);
    assertThat(workerMetric.isMeasurable()).isEqualTo(expectedIsMeasurable);
    assertThat(workerMetric.getLastCallTime().get()).isEqualTo(expectedLastCallTime);
    if (expectedCollectedTime == null) {
      assertThat(workerMetric.getLastCollectedTime().isEmpty()).isTrue();
    } else {
      assertThat(workerMetric.getLastCollectedTime().isPresent()).isTrue();
      assertThat(workerMetric.getLastCollectedTime().get()).isEqualTo(expectedCollectedTime);
    }
  }

  @Test
  public void testRegisterWorker_insertDifferent() throws Exception {
    spyCollector.registerWorker(
        WORKER_ID_1,
        PROCESS_ID_1,
        new WorkerProcessStatus(),
        JAVAC_MNEMONIC,
        /* isMultiplex= */ true,
        /* isSandboxed= */ false,
        WORKER_KEY_HASH_1,
        /* cgroup= */ null);
    assertThat(spyCollector.getPidToWorkerProcessMetrics().keySet()).containsExactly(PROCESS_ID_1);
    spyCollector.registerWorker(
        WORKER_ID_2,
        PROCESS_ID_2,
        new WorkerProcessStatus(),
        CPP_COMPILE_MNEMONIC,
        /* isMultiplex= */ false,
        /* isSandboxed= */ true,
        WORKER_KEY_HASH_2,
        /* cgroup= */ null);
    assertThat(spyCollector.getPidToWorkerProcessMetrics().keySet())
        .containsExactly(PROCESS_ID_1, PROCESS_ID_2);
    assertWorkerMetricContains(
        spyCollector.getPidToWorkerProcessMetrics().get(PROCESS_ID_1),
        ImmutableList.of(WORKER_ID_1),
        PROCESS_ID_1,
        JAVAC_MNEMONIC,
        /* expectedIsMultiplex= */ true,
        /* expectedIsSandboxed= */ false,
        WORKER_KEY_HASH_1,
        /* expectedActionsExecuted= */ 0,
        /* expectedIsMeasurable= */ false,
        /* expectedLastCallTime= */ DEFAULT_CLOCK_START_INSTANT,
        /* expectedCollectedTime= */ null);
    assertWorkerMetricContains(
        spyCollector.getPidToWorkerProcessMetrics().get(PROCESS_ID_2),
        ImmutableList.of(WORKER_ID_2),
        PROCESS_ID_2,
        CPP_COMPILE_MNEMONIC,
        /* expectedIsMultiplex= */ false,
        /* expectedIsSandboxed= */ true,
        WORKER_KEY_HASH_2,
        /* expectedActionsExecuted= */ 0,
        /* expectedIsMeasurable= */ false,
        /* expectedLastCallTime= */ DEFAULT_CLOCK_START_INSTANT,
        /* expectedCollectedTime= */ null);
  }

  @Test
  public void testRegisterWorker_insertMultiplex() throws Exception {
    spyCollector.registerWorker(
        WORKER_ID_1,
        PROCESS_ID_1,
        new WorkerProcessStatus(),
        JAVAC_MNEMONIC,
        /* isMultiplex= */ true,
        /* isSandboxed= */ true,
        WORKER_KEY_HASH_1,
        /* cgroup= */ null);
    assertThat(spyCollector.getPidToWorkerProcessMetrics().keySet()).containsExactly(PROCESS_ID_1);
    assertWorkerMetricContains(
        spyCollector.getPidToWorkerProcessMetrics().get(PROCESS_ID_1),
        ImmutableList.of(WORKER_ID_1),
        PROCESS_ID_1,
        JAVAC_MNEMONIC,
        /* expectedIsMultiplex= */ true,
        /* expectedIsSandboxed= */ true,
        WORKER_KEY_HASH_1,
        /* expectedActionsExecuted= */ 0,
        /* expectedIsMeasurable= */ false,
        /* expectedLastCallTime= */ DEFAULT_CLOCK_START_INSTANT,
        /* expectedCollectedTime= */ null);

    Instant secondTime = DEFAULT_CLOCK_START_INSTANT.plusSeconds(10);
    clock.setTime(secondTime.toEpochMilli());

    spyCollector.registerWorker(
        WORKER_ID_2,
        PROCESS_ID_1,
        new WorkerProcessStatus(),
        JAVAC_MNEMONIC,
        /* isMultiplex= */ true,
        /* isSandboxed= */ true,
        WORKER_KEY_HASH_1,
        /* cgroup= */ null);
    assertThat(spyCollector.getPidToWorkerProcessMetrics().keySet()).containsExactly(PROCESS_ID_1);
    assertWorkerMetricContains(
        spyCollector.getPidToWorkerProcessMetrics().get(PROCESS_ID_1),
        ImmutableList.of(WORKER_ID_1, WORKER_ID_2),
        PROCESS_ID_1,
        JAVAC_MNEMONIC,
        /* expectedIsMultiplex= */ true,
        /* expectedIsSandboxed= */ true,
        WORKER_KEY_HASH_1,
        /* expectedActionsExecuted= */ 0,
        /* expectedIsMeasurable= */ false,
        /* expectedLastCallTime= */ secondTime,
        /* expectedCollectedTime= */ null);
  }

  @Test
  public void testRegisterWorker_insertSame() throws Exception {
    spyCollector.registerWorker(
        WORKER_ID_1,
        PROCESS_ID_1,
        new WorkerProcessStatus(),
        JAVAC_MNEMONIC,
        /* isMultiplex= */ true,
        /* isSandboxed= */ true,
        WORKER_KEY_HASH_1,
        /* cgroup= */ null);
    assertThat(spyCollector.getPidToWorkerProcessMetrics().keySet()).containsExactly(PROCESS_ID_1);
    assertWorkerMetricContains(
        spyCollector.getPidToWorkerProcessMetrics().get(PROCESS_ID_1),
        ImmutableList.of(WORKER_ID_1),
        PROCESS_ID_1,
        JAVAC_MNEMONIC,
        /* expectedIsMultiplex= */ true,
        /* expectedIsSandboxed= */ true,
        WORKER_KEY_HASH_1,
        /* expectedActionsExecuted= */ 0,
        /* expectedIsMeasurable= */ false,
        /* expectedLastCallTime= */ DEFAULT_CLOCK_START_INSTANT,
        /* expectedCollectedTime= */ null);

    Instant secondTime = DEFAULT_CLOCK_START_INSTANT.plusSeconds(10);
    clock.setTime(secondTime.toEpochMilli());

    // When it is the same worker, it should only update the last call time.
    spyCollector.registerWorker(
        WORKER_ID_1,
        PROCESS_ID_1,
        new WorkerProcessStatus(),
        JAVAC_MNEMONIC,
        /* isMultiplex= */ true,
        /* isSandboxed= */ true,
        WORKER_KEY_HASH_1,
        /* cgroup= */ null);
    assertThat(spyCollector.getPidToWorkerProcessMetrics().keySet()).containsExactly(PROCESS_ID_1);
    assertWorkerMetricContains(
        spyCollector.getPidToWorkerProcessMetrics().get(PROCESS_ID_1),
        ImmutableList.of(WORKER_ID_1),
        PROCESS_ID_1,
        JAVAC_MNEMONIC,
        /* expectedIsMultiplex= */ true,
        /* expectedIsSandboxed= */ true,
        WORKER_KEY_HASH_1,
        /* expectedActionsExecuted= */ 0,
        /* expectedIsMeasurable= */ false,
        /* expectedLastCallTime= */ secondTime,
        /* expectedCollectedTime= */ null);
  }

  @Test
  public void testCollectMetrics() throws Exception {
    // Worker 1 simulates a measurable worker processes has executed some actions.
    spyCollector.registerWorker(
        WORKER_ID_1,
        PROCESS_ID_1,
        new WorkerProcessStatus(),
        JAVAC_MNEMONIC,
        /* isMultiplex= */ true,
        /* isSandboxed= */ false,
        WORKER_KEY_HASH_1,
        /* cgroup= */ null);
    spyCollector.onWorkerFinishExecution(PROCESS_ID_1);
    // Worker 2 simulates a measurable worker process that has not yet completed execution of any
    // actions.
    spyCollector.registerWorker(
        WORKER_ID_2,
        PROCESS_ID_2,
        new WorkerProcessStatus(),
        CPP_COMPILE_MNEMONIC,
        /* isMultiplex= */ false,
        /* isSandboxed= */ true,
        WORKER_KEY_HASH_2,
        /* cgroup= */ null);
    // Worker 3 simulates a non-measurable worker that has not executed any actions.
    spyCollector.registerWorker(
        WORKER_ID_3,
        PROCESS_ID_3,
        new WorkerProcessStatus(),
        PROTO_MNEMONIC,
        /* isMultiplex= */ true,
        /* isSandboxed= */ true,
        WORKER_KEY_HASH_3,
        /* cgroup= */ null);
    // Worker 4 simulates a non-measurable worker that has executed an action and was killed.
    WorkerProcessStatus s4 = new WorkerProcessStatus();
    spyCollector.registerWorker(
        WORKER_ID_4,
        PROCESS_ID_4,
        /* status= */ s4,
        PROTO_MNEMONIC,
        /* isMultiplex= */ true,
        /* isSandboxed= */ true,
        WORKER_KEY_HASH_4,
        /* cgroup= */ null);
    spyCollector.onWorkerFinishExecution(PROCESS_ID_4);
    s4.maybeUpdateStatus(Status.KILLED_DUE_TO_MEMORY_PRESSURE);
    // Worker 5 simulates a non-measurable worker that has executed an action, but was not killed.
    WorkerProcessStatus s5 = new WorkerProcessStatus();
    spyCollector.registerWorker(
        WORKER_ID_5,
        PROCESS_ID_5,
        /* status= */ s5,
        PROTO_MNEMONIC,
        /* isMultiplex= */ true,
        /* isSandboxed= */ true,
        WORKER_KEY_HASH_5,
        /* cgroup= */ null);
    spyCollector.onWorkerFinishExecution(PROCESS_ID_5);

    ImmutableMap<Long, Integer> memoryUsageMap =
        ImmutableMap.of(
            PROCESS_ID_1, 1234,
            PROCESS_ID_2, 2345);
    Instant collectionTime = DEFAULT_CLOCK_START_INSTANT.plusSeconds(10);
    ResourceSnapshot resourceSnapshot = ResourceSnapshot.create(memoryUsageMap, collectionTime);
    doReturn(resourceSnapshot).when(spyCollector).collectResourceUsage();
    clock.setTime(collectionTime.toEpochMilli());

    ImmutableList<WorkerProcessMetrics> metrics = spyCollector.collectMetrics();

    // All workers measurable or non-measurable should be reported.
    assertThat(metrics.stream().flatMap(m -> m.getWorkerIds().stream()).collect(toImmutableSet()))
        .containsExactly(WORKER_ID_1, WORKER_ID_2, WORKER_ID_3, WORKER_ID_4, WORKER_ID_5);
    assertWorkerMetricContains(
        getWorkerProcessMetricsFromList(WORKER_ID_1, metrics),
        ImmutableList.of(WORKER_ID_1),
        PROCESS_ID_1,
        JAVAC_MNEMONIC,
        /* expectedIsMultiplex= */ true,
        /* expectedIsSandboxed= */ false,
        WORKER_KEY_HASH_1,
        /* expectedActionsExecuted= */ 1,
        /* expectedIsMeasurable= */ true,
        /* expectedLastCallTime= */ DEFAULT_CLOCK_START_INSTANT,
        /* expectedCollectedTime= */ collectionTime);
    assertWorkerMetricContains(
        getWorkerProcessMetricsFromList(WORKER_ID_2, metrics),
        ImmutableList.of(WORKER_ID_2),
        PROCESS_ID_2,
        CPP_COMPILE_MNEMONIC,
        /* expectedIsMultiplex= */ false,
        /* expectedIsSandboxed= */ true,
        WORKER_KEY_HASH_2,
        /* expectedActionsExecuted= */ 0,
        /* expectedIsMeasurable= */ true,
        /* expectedLastCallTime= */ DEFAULT_CLOCK_START_INSTANT,
        /* expectedCollectedTime= */ collectionTime);
    // Worker 3's metrics should not be included since it is both non-measurable and did not execute
    // any actions. Its status shouldn't be unknown because it is possible that
    assertWorkerMetricContains(
        getWorkerProcessMetricsFromList(WORKER_ID_3, metrics),
        ImmutableList.of(WORKER_ID_3),
        PROCESS_ID_3,
        PROTO_MNEMONIC,
        /* expectedIsMultiplex= */ true,
        /* expectedIsSandboxed= */ true,
        WORKER_KEY_HASH_3,
        /* expectedActionsExecuted= */ 0,
        /* expectedIsMeasurable= */ false,
        /* expectedLastCallTime= */ DEFAULT_CLOCK_START_INSTANT,
        /* expectedCollectedTime= */ null);
    assertWorkerMetricContains(
        getWorkerProcessMetricsFromList(WORKER_ID_4, metrics),
        ImmutableList.of(WORKER_ID_4),
        PROCESS_ID_4,
        PROTO_MNEMONIC,
        /* expectedIsMultiplex= */ true,
        /* expectedIsSandboxed= */ true,
        WORKER_KEY_HASH_4,
        /* expectedActionsExecuted= */ 1,
        /* expectedIsMeasurable= */ false,
        /* expectedLastCallTime= */ DEFAULT_CLOCK_START_INSTANT,
        /* expectedCollectedTime= */ null);
    assertWorkerMetricContains(
        getWorkerProcessMetricsFromList(WORKER_ID_5, metrics),
        ImmutableList.of(WORKER_ID_5),
        PROCESS_ID_5,
        PROTO_MNEMONIC,
        /* expectedIsMultiplex= */ true,
        /* expectedIsSandboxed= */ true,
        WORKER_KEY_HASH_5,
        /* expectedActionsExecuted= */ 1,
        /* expectedIsMeasurable= */ false,
        /* expectedLastCallTime= */ DEFAULT_CLOCK_START_INSTANT,
        /* expectedCollectedTime= */ null);
    // Worker 5's status should have been updated to killed_unknown, because it had executed actions
    // but is now non-measurable.
    assertThat(s5.get()).isEqualTo(Status.KILLED_UNKNOWN);
  }

  @Test
  public void testCollectResourceUsage_windows() {
    Instant collectionTime = DEFAULT_CLOCK_START_INSTANT.plusSeconds(10);
    clock.setTime(collectionTime.toEpochMilli());
    when(psInfoCollector.collectResourceUsage(any(), any()))
        .thenReturn(ResourceSnapshot.create(ImmutableMap.of(PROCESS_ID_1, 1000), collectionTime));
    when(cgroupsInfoCollector.collectResourceUsage(any(), any()))
        .thenReturn(ResourceSnapshot.create(ImmutableMap.of(PROCESS_ID_1, 2000), collectionTime));

    ResourceSnapshot snapshot =
        spyCollector.collectResourceUsage(OS.WINDOWS, ImmutableSet.of(PROCESS_ID_1));

    // On non-linux and non-darwin, it should always return an empty snapshot.
    assertThat(snapshot).isEqualTo(ResourceSnapshot.create(ImmutableMap.of(), collectionTime));
  }

  @Test
  public void testCollectResourceUsage_darwin_usingPs() {
    Instant collectionTime = DEFAULT_CLOCK_START_INSTANT.plusSeconds(10);
    clock.setTime(collectionTime.toEpochMilli());
    when(psInfoCollector.collectResourceUsage(any(), any()))
        .thenReturn(ResourceSnapshot.create(ImmutableMap.of(PROCESS_ID_1, 1000), collectionTime));
    when(cgroupsInfoCollector.collectResourceUsage(any(), any()))
        .thenReturn(ResourceSnapshot.create(ImmutableMap.of(PROCESS_ID_1, 2000), collectionTime));

    ResourceSnapshot snapshot =
        spyCollector.collectResourceUsage(OS.DARWIN, ImmutableSet.of(PROCESS_ID_1));

    // Should return the cgroup information rather than the ps information.
    assertThat(snapshot)
        .isEqualTo(ResourceSnapshot.create(ImmutableMap.of(PROCESS_ID_1, 1000), collectionTime));
  }

  @Test
  public void testCollectResourceUsage_linux_usingPs() {
    spyCollector.setUseCgroupsOnLinux(/* useCgroupsOnLinux= */ false);
    Instant collectionTime = DEFAULT_CLOCK_START_INSTANT.plusSeconds(10);
    clock.setTime(collectionTime.toEpochMilli());
    when(psInfoCollector.collectResourceUsage(any(), any()))
        .thenReturn(ResourceSnapshot.create(ImmutableMap.of(PROCESS_ID_1, 1000), collectionTime));
    when(cgroupsInfoCollector.collectResourceUsage(any(), any()))
        .thenReturn(ResourceSnapshot.create(ImmutableMap.of(PROCESS_ID_1, 2000), collectionTime));

    ResourceSnapshot snapshot =
        spyCollector.collectResourceUsage(OS.LINUX, ImmutableSet.of(PROCESS_ID_1));

    // Should return the ps information rather than the cgroup information.
    assertThat(snapshot)
        .isEqualTo(ResourceSnapshot.create(ImmutableMap.of(PROCESS_ID_1, 1000), collectionTime));
  }

  @Test
  public void testCollectResourceUsage_linux_usingCgroups() {

    spyCollector.setUseCgroupsOnLinux(/* useCgroupsOnLinux= */ true);
    Instant collectionTime = DEFAULT_CLOCK_START_INSTANT.plusSeconds(10);
    clock.setTime(collectionTime.toEpochMilli());
    when(psInfoCollector.collectResourceUsage(any(), any()))
        .thenReturn(ResourceSnapshot.create(ImmutableMap.of(PROCESS_ID_1, 1000), collectionTime));
    when(cgroupsInfoCollector.collectResourceUsage(any(), any()))
        .thenReturn(ResourceSnapshot.create(ImmutableMap.of(PROCESS_ID_1, 2000), collectionTime));

    ResourceSnapshot snapshot =
        spyCollector.collectResourceUsage(OS.LINUX, ImmutableSet.of(PROCESS_ID_1));

    // Should return the cgroup information rather than the ps information.
    assertThat(snapshot)
        .isEqualTo(ResourceSnapshot.create(ImmutableMap.of(PROCESS_ID_1, 2000), collectionTime));
  }

  @Test
  public void testWorkerMetricsPublishComparator_compare() {
    WorkerMetrics alive1 = newWorkerMetrics(1, WorkerStatus.ALIVE, 100);
    WorkerMetrics alive2 = newWorkerMetrics(2, WorkerStatus.ALIVE, 200);
    WorkerMetrics evicted1 = newWorkerMetrics(3, WorkerStatus.KILLED_DUE_TO_MEMORY_PRESSURE, 100);
    WorkerMetrics evicted2 = newWorkerMetrics(4, WorkerStatus.KILLED_DUE_TO_MEMORY_PRESSURE, 200);
    WorkerMetrics others1 = newWorkerMetrics(5, WorkerStatus.KILLED_UNKNOWN, 100);
    WorkerMetrics others2 =
        newWorkerMetrics(6, WorkerStatus.KILLED_DUE_TO_USER_EXEC_EXCEPTION, 200);

    WorkerMetricsPublishComparator comparator = new WorkerMetricsPublishComparator();
    // WorkerMetrics of the same status priority should be compared by their memory usage (higher
    // gets prioritized).
    assertThat(comparator.compare(alive1, alive2)).isEqualTo(1);
    assertThat(comparator.compare(evicted1, evicted2)).isEqualTo(1);
    assertThat(comparator.compare(others1, others2)).isEqualTo(1);

    // WorkerMetrics should be first compared by their status priorities rather than their memory
    // usage.
    assertThat(comparator.compare(alive1, evicted2)).isEqualTo(-1);
    assertThat(comparator.compare(evicted1, others2)).isEqualTo(-1);
    assertThat(comparator.compare(others2, alive1)).isEqualTo(1);
  }

  @Test
  public void testLimitWorkerMetricsToPublish() {
    WorkerMetrics alive1 = newWorkerMetrics(1, WorkerStatus.ALIVE, 200);
    WorkerMetrics alive2 = newWorkerMetrics(2, WorkerStatus.ALIVE, 100);
    WorkerMetrics evicted3 = newWorkerMetrics(3, WorkerStatus.KILLED_DUE_TO_MEMORY_PRESSURE, 100);
    WorkerMetrics evicted4 = newWorkerMetrics(4, WorkerStatus.KILLED_DUE_TO_MEMORY_PRESSURE, 200);
    WorkerMetrics others5 = newWorkerMetrics(5, WorkerStatus.KILLED_UNKNOWN, 200);
    WorkerMetrics others6 =
        newWorkerMetrics(6, WorkerStatus.KILLED_DUE_TO_USER_EXEC_EXCEPTION, 100);

    // Based on prioritization and then sorted by worker id.
    assertThat(
            WorkerProcessMetricsCollector.limitWorkerMetricsToPublish(
                ImmutableList.of(alive1, alive2, evicted3, evicted4, others5, others6), 3))
        .containsExactly(alive1, alive2, evicted4);
    assertThat(
            WorkerProcessMetricsCollector.limitWorkerMetricsToPublish(
                ImmutableList.of(alive1, evicted4, others5, others6), 3))
        .containsExactly(alive1, evicted4, others5);
    // If under the limit, it should just report everything.
    assertThat(
            WorkerProcessMetricsCollector.limitWorkerMetricsToPublish(
                ImmutableList.of(alive1, alive2, evicted4, others6), 10))
        .containsExactly(alive1, alive2, evicted4, others6);
  }

  private WorkerMetrics newWorkerMetrics(int id, WorkerStatus status, int memoryInKb) {
    return WorkerMetrics.newBuilder()
        .addWorkerIds(id)
        .setWorkerStatus(status)
        .addWorkerStats(
            WorkerMetrics.WorkerStats.newBuilder().setWorkerMemoryInKb(memoryInKb).build())
        .build();
  }

  private WorkerProcessMetrics getWorkerProcessMetricsFromList(
      int workerId, ImmutableList<WorkerProcessMetrics> metrics) {
    return metrics.stream().filter(wm -> wm.getWorkerIds().contains(workerId)).findFirst().get();
  }

  private static final long DEFAULT_CLOCK_START_TIME = 0L;
  private static final Instant DEFAULT_CLOCK_START_INSTANT =
      Instant.ofEpochMilli(DEFAULT_CLOCK_START_TIME);

  private static class ManualClock implements Clock {
    private long currentTime = DEFAULT_CLOCK_START_TIME;

    ManualClock() {}

    @Override
    public long nanoTime() {
      throw new AssertionError("unexpected method call");
    }

    @Override
    public long currentTimeMillis() {
      return currentTime;
    }

    void setTime(long currentTime) {
      this.currentTime = currentTime;
    }
  }
}
