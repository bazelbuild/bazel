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
import static com.google.common.truth.Truth8.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.spy;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.metrics.PsInfoCollector;
import com.google.devtools.build.lib.worker.WorkerProcessStatus.Status;
import java.time.Instant;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for the WorkerSpawnRunner. */
@RunWith(JUnit4.class)
public class WorkerProcessMetricsCollectorTest {

  private final WorkerProcessMetricsCollector spyCollector =
      spy(WorkerProcessMetricsCollector.instance());
  ManualClock clock = new ManualClock();

  @Before
  public void setUp() {
    spyCollector.clear();
    spyCollector.setClock(clock);
  }

  private static final int WORKER_ID_1 = 1;
  private static final int WORKER_ID_2 = 2;
  private static final int WORKER_ID_3 = 3;
  private static final int WORKER_ID_4 = 4;
  private static final long PROCESS_ID_1 = 100L;
  private static final long PROCESS_ID_2 = 200L;
  private static final long PROCESS_ID_3 = 300L;
  private static final long PROCESS_ID_4 = 400L;
  private static final int WORKER_KEY_HASH_1 = 1;
  private static final int WORKER_KEY_HASH_2 = 2;
  private static final int WORKER_KEY_HASH_3 = 3;
  private static final int WORKER_KEY_HASH_4 = 4;
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
        WORKER_KEY_HASH_1);
    assertThat(spyCollector.getProcessIdToWorkerProcessMetrics().keySet())
        .containsExactly(PROCESS_ID_1);
    spyCollector.registerWorker(
        WORKER_ID_2,
        PROCESS_ID_2,
        new WorkerProcessStatus(),
        CPP_COMPILE_MNEMONIC,
        /* isMultiplex= */ false,
        /* isSandboxed= */ true,
        WORKER_KEY_HASH_2);
    assertThat(spyCollector.getProcessIdToWorkerProcessMetrics().keySet())
        .containsExactly(PROCESS_ID_1, PROCESS_ID_2);
    assertWorkerMetricContains(
        spyCollector.getProcessIdToWorkerProcessMetrics().get(PROCESS_ID_1),
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
        spyCollector.getProcessIdToWorkerProcessMetrics().get(PROCESS_ID_2),
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
        WORKER_KEY_HASH_1);
    assertThat(spyCollector.getProcessIdToWorkerProcessMetrics().keySet())
        .containsExactly(PROCESS_ID_1);
    assertWorkerMetricContains(
        spyCollector.getProcessIdToWorkerProcessMetrics().get(PROCESS_ID_1),
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
        WORKER_KEY_HASH_1);
    assertThat(spyCollector.getProcessIdToWorkerProcessMetrics().keySet())
        .containsExactly(PROCESS_ID_1);
    assertWorkerMetricContains(
        spyCollector.getProcessIdToWorkerProcessMetrics().get(PROCESS_ID_1),
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
        WORKER_KEY_HASH_1);
    assertThat(spyCollector.getProcessIdToWorkerProcessMetrics().keySet())
        .containsExactly(PROCESS_ID_1);
    assertWorkerMetricContains(
        spyCollector.getProcessIdToWorkerProcessMetrics().get(PROCESS_ID_1),
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
        WORKER_KEY_HASH_1);
    assertThat(spyCollector.getProcessIdToWorkerProcessMetrics().keySet())
        .containsExactly(PROCESS_ID_1);
    assertWorkerMetricContains(
        spyCollector.getProcessIdToWorkerProcessMetrics().get(PROCESS_ID_1),
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
        WORKER_KEY_HASH_1);
    WorkerProcessMetricsCollector.instance().onWorkerFinishExecution(PROCESS_ID_1);
    // Worker 2 simulates a measurable worker process that has not yet completed execution of any
    // actions.
    spyCollector.registerWorker(
        WORKER_ID_2,
        PROCESS_ID_2,
        new WorkerProcessStatus(),
        CPP_COMPILE_MNEMONIC,
        /* isMultiplex= */ false,
        /* isSandboxed= */ true,
        WORKER_KEY_HASH_2);
    // Worker 3 simulates a non-measurable worker that has not executed any actions.
    spyCollector.registerWorker(
        WORKER_ID_3,
        PROCESS_ID_3,
        new WorkerProcessStatus(),
        PROTO_MNEMONIC,
        /* isMultiplex= */ true,
        /* isSandboxed= */ true,
        WORKER_KEY_HASH_3);
    // Worker 4 simulates a non-measurable worker that has executed an action and was killed.
    WorkerProcessStatus s4 = new WorkerProcessStatus();
    spyCollector.registerWorker(
        WORKER_ID_4,
        PROCESS_ID_4,
        s4,
        PROTO_MNEMONIC,
        /* isMultiplex= */ true,
        /* isSandboxed= */ true,
        WORKER_KEY_HASH_4);
    WorkerProcessMetricsCollector.instance().onWorkerFinishExecution(PROCESS_ID_4);
    s4.maybeUpdateStatus(Status.KILLED_DUE_TO_MEMORY_PRESSURE);

    ImmutableMap<Long, Integer> memoryUsageMap =
        ImmutableMap.of(
            PROCESS_ID_1, 1234,
            PROCESS_ID_2, 2345);
    ImmutableSet<Long> expectedPids =
        ImmutableSet.of(PROCESS_ID_1, PROCESS_ID_2, PROCESS_ID_3, PROCESS_ID_4);
    Instant collectionTime = DEFAULT_CLOCK_START_INSTANT.plusSeconds(10);
    PsInfoCollector.ResourceSnapshot resourceSnapshot =
        PsInfoCollector.ResourceSnapshot.create(memoryUsageMap, collectionTime);
    doReturn(resourceSnapshot).when(spyCollector).collectMemoryUsageByPid(any(), eq(expectedPids));
    clock.setTime(collectionTime.toEpochMilli());

    ImmutableList<WorkerProcessMetrics> metrics = spyCollector.collectMetrics();

    // Since pid 3 is the only non-measurable process with no actions executed, we only expect
    // WorkerProcessMetrics from pid 1, 2 and 4.
    assertThat(metrics.stream().flatMap(m -> m.getWorkerIds().stream()).collect(toImmutableSet()))
        .containsExactly(WORKER_ID_1, WORKER_ID_2, WORKER_ID_4);
    assertWorkerMetricContains(
        metrics.stream().filter(wm -> wm.getWorkerIds().contains(WORKER_ID_1)).findFirst().get(),
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
        metrics.stream().filter(wm -> wm.getWorkerIds().contains(WORKER_ID_2)).findFirst().get(),
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
    // any actions.
    assertThat(metrics.stream().filter(wm -> wm.getWorkerIds().contains(WORKER_ID_3)).findFirst())
        .isEmpty();
    assertWorkerMetricContains(
        metrics.stream().filter(wm -> wm.getWorkerIds().contains(WORKER_ID_4)).findFirst().get(),
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
