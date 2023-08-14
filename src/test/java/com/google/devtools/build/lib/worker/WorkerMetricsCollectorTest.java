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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.spy;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.metrics.PsInfoCollector;
import java.time.Instant;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for the WorkerSpawnRunner. */
@RunWith(JUnit4.class)
public class WorkerMetricsCollectorTest {

  private final WorkerMetricsCollector spyCollector = spy(WorkerMetricsCollector.instance());
  ManualClock clock = new ManualClock();

  @Before
  public void setUp() {
    spyCollector.clear();
    spyCollector.setClock(clock);
  }

  @Test
  public void testRegisterWorker_insertDifferent() throws Exception {
    WorkerMetric.WorkerProperties props1 =
        WorkerMetric.WorkerProperties.create(
            /* workerIds= */ ImmutableList.of(1),
            /* processId= */ 100,
            /* mnemonic= */ "Javac",
            /* isMultiplex= */ true,
            /* isSandboxed= */ false,
            /* workerKeyHash= */ 1);
    WorkerMetric.WorkerProperties props2 =
        WorkerMetric.WorkerProperties.create(
            /* workerIds= */ ImmutableList.of(2),
            /* processId= */ 200,
            /* mnemonic= */ "CppCompile",
            /* isMultiplex= */ false,
            /* isSandboxed= */ true,
            /* workerKeyHash= */ 2);
    ImmutableMap<Long, WorkerMetric.WorkerProperties> map =
        ImmutableMap.of(100L, props1, 200L, props2);

    spyCollector.registerWorker(
        props1.getWorkerIds().get(0),
        props1.getProcessId(),
        props1.getMnemonic(),
        props1.isMultiplex(),
        props1.isSandboxed(),
        props1.getWorkerKeyHash());
    assertThat(spyCollector.getProcessIdToWorkerProperties()).hasSize(1);
    spyCollector.registerWorker(
        props2.getWorkerIds().get(0),
        props2.getProcessId(),
        props2.getMnemonic(),
        props2.isMultiplex(),
        props2.isSandboxed(),
        props2.getWorkerKeyHash());
    assertThat(spyCollector.getProcessIdToWorkerProperties()).hasSize(2);
    assertThat(spyCollector.getProcessIdToWorkerProperties()).isEqualTo(map);
  }

  @Test
  public void testRegisterWorker_insertMultiplex() throws Exception {
    WorkerMetric.WorkerProperties props1 =
        WorkerMetric.WorkerProperties.create(
            /* workerIds= */ ImmutableList.of(1),
            /* processId= */ 100L,
            /* mnemonic= */ "Javac",
            /* isMultiplex= */ true,
            /* isSandboxed= */ true,
            /* workerKeyHash= */ 1);
    WorkerMetric.WorkerProperties props2 =
        WorkerMetric.WorkerProperties.create(
            /* workerIds= */ ImmutableList.of(2),
            /* processId= */ 100L,
            /* mnemonic= */ "Javac",
            /* isMultiplex= */ true,
            /* isSandboxed= */ true,
            /* workerKeyHash= */ 1);
    Instant registrationTime1 = Instant.ofEpochSecond(1000);
    Instant registrationTime2 = registrationTime1.plusSeconds(10);
    ImmutableMap<Long, WorkerMetric.WorkerProperties> map =
        ImmutableMap.of(
            100L,
            WorkerMetric.WorkerProperties.create(
                /* workerIds= */ ImmutableList.of(1, 2),
                /* processId= */ 100L,
                /* mnemonic= */ "Javac",
                /* isMultiplex= */ true,
                /* isSandboxed= */ true,
                /* workerKeyHash= */ 1));
    ImmutableMap<Long, Instant> lastCallMap1 = ImmutableMap.of(100L, registrationTime1);
    ImmutableMap<Long, Instant> lastCallMap2 = ImmutableMap.of(100L, registrationTime2);

    clock.setTime(registrationTime1.toEpochMilli());
    registerWorker(spyCollector, props1);

    assertThat(spyCollector.getProcessIdToWorkerProperties()).hasSize(1);
    assertThat(spyCollector.getWorkerLastCallTime()).isEqualTo(lastCallMap1);

    clock.setTime(registrationTime2.toEpochMilli());
    registerWorker(spyCollector, props2);

    assertThat(spyCollector.getProcessIdToWorkerProperties()).hasSize(1);
    assertThat(spyCollector.getProcessIdToWorkerProperties()).isEqualTo(map);
    assertThat(spyCollector.getWorkerLastCallTime()).isEqualTo(lastCallMap2);
  }

  @Test
  public void testRegisterWorker_insertSame() throws Exception {
    WorkerMetric.WorkerProperties props1 =
        WorkerMetric.WorkerProperties.create(
            /* workerIds= */ ImmutableList.of(1),
            /* processId= */ 100,
            /* mnemonic= */ "Javac",
            /* isMultiplex= */ true,
            /* isSandboxed= */ false,
            /* workerKeyHash= */ 1);
    WorkerMetric.WorkerProperties props2 =
        WorkerMetric.WorkerProperties.create(
            /* workerIds= */ ImmutableList.of(1),
            /* processId= */ 100,
            /* mnemonic= */ "Javac",
            /* isMultiplex= */ true,
            /* isSandboxed= */ false,
            /* workerKeyHash= */ 2);
    Instant registrationTime1 = Instant.ofEpochSecond(1000);
    Instant registrationTime2 = registrationTime1.plusSeconds(10);
    ImmutableMap<Long, WorkerMetric.WorkerProperties> propertiesMap = ImmutableMap.of(100L, props1);
    ImmutableMap<Long, Instant> lastCallMap1 = ImmutableMap.of(100L, registrationTime1);
    ImmutableMap<Long, Instant> lastCallMap2 = ImmutableMap.of(100L, registrationTime2);

    clock.setTime(registrationTime1.toEpochMilli());
    registerWorker(spyCollector, props1);
    assertThat(spyCollector.getProcessIdToWorkerProperties()).isEqualTo(propertiesMap);
    assertThat(spyCollector.getWorkerLastCallTime()).isEqualTo(lastCallMap1);

    clock.setTime(registrationTime2.toEpochMilli());
    registerWorker(spyCollector, props2);
    assertThat(spyCollector.getProcessIdToWorkerProperties()).isEqualTo(propertiesMap);
    assertThat(spyCollector.getWorkerLastCallTime()).isEqualTo(lastCallMap2);
  }

  @Test
  public void testcollectMetrics() throws Exception {
    WorkerMetric.WorkerProperties props1 =
        WorkerMetric.WorkerProperties.create(
            /* workerIds= */ ImmutableList.of(1),
            /* processId= */ 100,
            /* mnemonic= */ "Javac",
            /* isMultiplex= */ true,
            /* isSandboxed= */ false,
            /* workerKeyHash= */ 1);
    WorkerMetric.WorkerProperties props2 =
        WorkerMetric.WorkerProperties.create(
            /* workerIds= */ ImmutableList.of(2),
            /* processId= */ 200,
            /* mnemonic= */ "CppCompile",
            /* isMultiplex= */ false,
            /* isSandboxed= */ true,
            /* workerKeyHash= */ 2);
    WorkerMetric.WorkerProperties props3 =
        WorkerMetric.WorkerProperties.create(
            /* workerIds= */ ImmutableList.of(3),
            /* processId= */ 300,
            /* mnemonic= */ "Proto",
            /* isMultiplex= */ true,
            /* isSandboxed= */ true,
            /* workerKeyHash= */ 3);
    Instant registrationTime = Instant.ofEpochSecond(1000);
    Instant collectionTime = registrationTime.plusSeconds(10);
    WorkerMetric.WorkerStat stat1 =
        WorkerMetric.WorkerStat.create(1234, registrationTime, collectionTime);
    WorkerMetric.WorkerStat stat2 =
        WorkerMetric.WorkerStat.create(2345, registrationTime, collectionTime);
    WorkerMetric.WorkerStat stat3 =
        WorkerMetric.WorkerStat.create(0, registrationTime, collectionTime);
    WorkerMetric workerMetric1 = WorkerMetric.create(props1, stat1, true);
    WorkerMetric workerMetric2 = WorkerMetric.create(props2, stat2, true);
    WorkerMetric workerMetric3 = WorkerMetric.create(props3, stat3, false);
    ImmutableSet<Long> expectedPids = ImmutableSet.of(100L, 200L, 300L);
    ImmutableMap<Long, WorkerMetric.WorkerProperties> propsMap =
        ImmutableMap.of(
            100L, props1,
            200L, props2);
    ImmutableMap<Long, Integer> memoryUsageMap =
        ImmutableMap.of(
            100L, stat1.getUsedMemoryInKB(),
            200L, stat2.getUsedMemoryInKB());

    PsInfoCollector.ResourceSnapshot resourceSnapshot =
        PsInfoCollector.ResourceSnapshot.create(memoryUsageMap, collectionTime);
    ImmutableList<WorkerMetric> expectedMetrics =
        ImmutableList.of(workerMetric1, workerMetric2, workerMetric3);

    doReturn(resourceSnapshot).when(spyCollector).collectMemoryUsageByPid(any(), eq(expectedPids));

    clock.setTime(registrationTime.toEpochMilli());

    registerWorker(spyCollector, props1);
    registerWorker(spyCollector, props2);
    registerWorker(spyCollector, props3);

    ImmutableList<WorkerMetric> metrics = spyCollector.collectMetrics();

    assertThat(metrics).containsExactlyElementsIn(expectedMetrics);
    assertThat(spyCollector.getProcessIdToWorkerProperties()).isEqualTo(propsMap);
  }

  private static void registerWorker(
      WorkerMetricsCollector collector, WorkerMetric.WorkerProperties props) {
    collector.registerWorker(
        props.getWorkerIds().get(0),
        props.getProcessId(),
        props.getMnemonic(),
        props.isMultiplex(),
        props.isSandboxed(),
        props.getWorkerKeyHash());
  }

  private static class ManualClock implements Clock {
    private long currentTime = 0L;

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
