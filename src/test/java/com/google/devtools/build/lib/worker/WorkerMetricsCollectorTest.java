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
import com.google.devtools.build.lib.util.PsInfoCollector;
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
  public void testCollectStats_mutipleSubprocesses() throws Exception {
    // pstree of these processes
    // 0-+-1---3-+-7
    //   |       `-8
    //   |-2-+-4
    //   |   `-9
    //   |-5
    //   `-10

    // ps command results:
    // PID PPID RSS
    // 1   0    3216
    // 2   0    4232
    // 3   1    1234
    // 4   2    1001
    // 5   0    40000
    // 7   3    2345
    // 8   3    3456
    // 9   2    1032
    // 10  0    1024
    ImmutableMap<Long, PsInfoCollector.PsInfo> psInfos =
        ImmutableMap.of(
            1L, PsInfoCollector.PsInfo.create(1, 0, 3216),
            2L, PsInfoCollector.PsInfo.create(2, 0, 4232),
            3L, PsInfoCollector.PsInfo.create(3, 1, 1234),
            4L, PsInfoCollector.PsInfo.create(4, 2, 1001),
            5L, PsInfoCollector.PsInfo.create(5, 0, 40000),
            7L, PsInfoCollector.PsInfo.create(7, 3, 2345),
            8L, PsInfoCollector.PsInfo.create(8, 3, 3456),
            9L, PsInfoCollector.PsInfo.create(9, 2, 1032),
            10L, PsInfoCollector.PsInfo.create(10, 0, 1024));

    ImmutableSet<Long> pids = ImmutableSet.of(1L, 2L, 5L, 6L);
    ImmutableMap<Long, Integer> expectedMemoryUsageByPid =
        ImmutableMap.of(1L, 3216 + 1234 + 2345 + 3456, 2L, 4232 + 1001 + 1032, 5L, 40000);

    ImmutableMap<Long, Integer> memoryUsageByPid =
        spyCollector.summarizeDescendantsMemory(psInfos, pids);
    assertThat(memoryUsageByPid).isEqualTo(expectedMemoryUsageByPid);
  }

  @Test
  public void testRegisterWorker_insertDifferent() throws Exception {
    WorkerMetric.WorkerProperties props1 =
        WorkerMetric.WorkerProperties.create(
            /* workerIds= */ ImmutableList.of(1),
            /* processId= */ 100,
            /* mnemonic= */ "Javac",
            /* isMultiplex= */ true,
            /* isSandboxed= */ false);
    WorkerMetric.WorkerProperties props2 =
        WorkerMetric.WorkerProperties.create(
            /* workerIds= */ ImmutableList.of(2),
            /* processId= */ 200,
            /* mnemonic= */ "CppCompile",
            /* isMultiplex= */ false,
            /* isSandboxed= */ true);
    ImmutableMap<Long, WorkerMetric.WorkerProperties> map =
        ImmutableMap.of(100L, props1, 200L, props2);

    spyCollector.registerWorker(
        props1.getWorkerIds().get(0),
        props1.getProcessId(),
        props1.getMnemonic(),
        props1.isMultiplex(),
        props1.isSandboxed());
    assertThat(spyCollector.getProcessIdToWorkerProperties()).hasSize(1);
    spyCollector.registerWorker(
        props2.getWorkerIds().get(0),
        props2.getProcessId(),
        props2.getMnemonic(),
        props2.isMultiplex(),
        props2.isSandboxed());
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
            /* isSandboxed= */ true);
    WorkerMetric.WorkerProperties props2 =
        WorkerMetric.WorkerProperties.create(
            /* workerIds= */ ImmutableList.of(2),
            /* processId= */ 100L,
            /* mnemonic= */ "Javac",
            /* isMultiplex= */ true,
            /* isSandboxed= */ true);
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
                /* isSandboxed= */ true));
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
            /* isSandboxed= */ false);
    WorkerMetric.WorkerProperties props2 =
        WorkerMetric.WorkerProperties.create(
            /* workerIds= */ ImmutableList.of(1),
            /* processId= */ 100,
            /* mnemonic= */ "Javac",
            /* isMultiplex= */ true,
            /* isSandboxed= */ false);
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
            /* isSandboxed= */ false);
    WorkerMetric.WorkerProperties props2 =
        WorkerMetric.WorkerProperties.create(
            /* workerIds= */ ImmutableList.of(2),
            /* processId= */ 200,
            /* mnemonic= */ "CppCompile",
            /* isMultiplex= */ false,
            /* isSandboxed= */ true);
    WorkerMetric.WorkerProperties props3 =
        WorkerMetric.WorkerProperties.create(
            /* workerIds= */ ImmutableList.of(3),
            /* processId= */ 300,
            /* mnemonic= */ "Proto",
            /* isMultiplex= */ true,
            /* isSandboxed= */ true);
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
    WorkerMetricsCollector.MemoryCollectionResult memoryCollectionResult =
        new WorkerMetricsCollector.MemoryCollectionResult(memoryUsageMap, collectionTime);
    ImmutableList<WorkerMetric> expectedMetrics =
        ImmutableList.of(workerMetric1, workerMetric2, workerMetric3);

    doReturn(memoryCollectionResult)
        .when(spyCollector)
        .collectMemoryUsageByPid(any(), eq(expectedPids));

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
        props.isSandboxed());
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
