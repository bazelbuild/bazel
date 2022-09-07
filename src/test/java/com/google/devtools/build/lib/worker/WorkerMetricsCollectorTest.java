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
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.util.OS;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.time.Instant;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/** Unit tests for the WorkerSpawnRunner. */
@RunWith(JUnit4.class)
public class WorkerMetricsCollectorTest {

  @Rule public final MockitoRule mockito = MockitoJUnit.rule();
  private final WorkerMetricsCollector spyCollector = spy(WorkerMetricsCollector.instance());
  @Captor ArgumentCaptor<ImmutableSet<Long>> pidsCaptor;
  ManualClock clock = new ManualClock();

  @Before
  public void setUp() {
    spyCollector.clear();
    spyCollector.setClock(clock);
  }

  @Test
  public void testCollectStats_ignoreSpaces() throws Exception {
    String psOutput = "    PID  \t  RSS\n   1  3216 \t\n  \t 2 \t 4096 \t";
    ImmutableMap<Long, Long> subprocessesMap =
        ImmutableMap.of(
            1L, 1L,
            2L, 2L);
    ImmutableSet<Long> pids = ImmutableSet.of(1L, 2L);
    InputStream psStream = new ByteArrayInputStream(psOutput.getBytes(UTF_8));
    Process process = mock(Process.class);

    when(spyCollector.getSubprocesses(pids)).thenReturn(subprocessesMap);
    when(spyCollector.buildPsProcess(subprocessesMap.keySet())).thenReturn(process);
    when(process.getInputStream()).thenReturn(psStream);

    ImmutableMap<Long, Integer> memoryUsageByPid =
        spyCollector.collectMemoryUsageByPid(OS.LINUX, pids).pidToMemoryInKb;

    ImmutableMap<Long, Integer> expectedMemoryUsageByPid = ImmutableMap.of(1L, 3216, 2L, 4096);
    assertThat(memoryUsageByPid).isEqualTo(expectedMemoryUsageByPid);
  }

  @Test
  public void testCollectStats_mutipleSubprocesses() throws Exception {
    String psOutput = "PID  RSS  \n 1  3216\n 2  4232\n 3 1234 \n 4 1001 \n 5 40000";
    ImmutableMap<Long, Long> subprocessesMap =
        ImmutableMap.of(
            1L, 1L,
            2L, 2L,
            3L, 1L,
            4L, 2L,
            5L, 5L,
            6L, 1L);
    ImmutableSet<Long> pids = ImmutableSet.of(1L, 2L, 5L);
    InputStream psStream = new ByteArrayInputStream(psOutput.getBytes(UTF_8));
    Process process = mock(Process.class);

    when(spyCollector.getSubprocesses(pids)).thenReturn(subprocessesMap);
    when(spyCollector.buildPsProcess(subprocessesMap.keySet())).thenReturn(process);
    when(process.getInputStream()).thenReturn(psStream);

    ImmutableMap<Long, Integer> memoryUsageByPid =
        spyCollector.collectMemoryUsageByPid(OS.LINUX, pids).pidToMemoryInKb;

    ImmutableMap<Long, Integer> expectedMemoryUsageByPid =
        ImmutableMap.of(1L, 3216 + 1234, 2L, 4232 + 1001, 5L, 40000);
    assertThat(memoryUsageByPid).isEqualTo(expectedMemoryUsageByPid);
  }

  @Test
  public void testRegisterWorker_insertDifferent() throws Exception {
    WorkerMetric.WorkerProperties props1 =
        WorkerMetric.WorkerProperties.create(
            /* workerId= */ 1,
            /* processId= */ 100,
            /* mnemonic= */ "Javac",
            /* isMultiplex= */ true,
            /* isSandboxed= */ false);
    WorkerMetric.WorkerProperties props2 =
        WorkerMetric.WorkerProperties.create(
            /* workerId= */ 2,
            /* processId= */ 200,
            /* mnemonic= */ "CppCompile",
            /* isMultiplex= */ false,
            /* isSandboxed= */ true);
    ImmutableMap<Integer, WorkerMetric.WorkerProperties> map =
        ImmutableMap.of(1, props1, 2, props2);

    spyCollector.registerWorker(props1);
    assertThat(spyCollector.getWorkerIdToWorkerProperties()).hasSize(1);
    spyCollector.registerWorker(props2);
    assertThat(spyCollector.getWorkerIdToWorkerProperties()).hasSize(2);
    assertThat(spyCollector.getWorkerIdToWorkerProperties()).isEqualTo(map);
  }

  @Test
  public void testRegisterWorker_insertSame() throws Exception {
    WorkerMetric.WorkerProperties props1 =
        WorkerMetric.WorkerProperties.create(
            /*workerId= */ 1,
            /*processId= */ 100,
            /*mnemonic= */ "Javac",
            /*isMultiplex= */ true,
            /*isSandboxed= */ false);
    WorkerMetric.WorkerProperties props2 =
        WorkerMetric.WorkerProperties.create(
            /*workerId= */ 1,
            /*processId= */ 100,
            /*mnemonic= */ "Javac",
            /*isMultiplex= */ true,
            /*isSandboxed= */ false);
    Instant registrationTime1 = Instant.ofEpochSecond(1000);
    Instant registrationTime2 = registrationTime1.plusSeconds(10);
    ImmutableMap<Integer, WorkerMetric.WorkerProperties> propertiesMap = ImmutableMap.of(1, props1);
    ImmutableMap<Integer, Instant> lastCallMap1 = ImmutableMap.of(1, registrationTime1);
    ImmutableMap<Integer, Instant> lastCallMap2 = ImmutableMap.of(1, registrationTime2);

    clock.setTime(registrationTime1.toEpochMilli());
    spyCollector.registerWorker(props1);
    assertThat(spyCollector.getWorkerIdToWorkerProperties()).isEqualTo(propertiesMap);
    assertThat(spyCollector.getWorkerLastCallTime()).isEqualTo(lastCallMap1);

    clock.setTime(registrationTime2.toEpochMilli());
    spyCollector.registerWorker(props2);
    assertThat(spyCollector.getWorkerIdToWorkerProperties()).isEqualTo(propertiesMap);
    assertThat(spyCollector.getWorkerLastCallTime()).isEqualTo(lastCallMap2);
  }

  @Test
  public void testcollectMetrics() throws Exception {
    WorkerMetric.WorkerProperties props1 =
        WorkerMetric.WorkerProperties.create(
            /*workerId= */ 1,
            /*processId= */ 100,
            /*mnemonic= */ "Javac",
            /*isMultiplex= */ true,
            /*isSandboxed= */ false);
    WorkerMetric.WorkerProperties props2 =
        WorkerMetric.WorkerProperties.create(
            /*workerId= */ 2,
            /*processId= */ 200,
            /*mnemonic= */ "CppCompile",
            /*isMultiplex= */ false,
            /*isSandboxed= */ true);
    WorkerMetric.WorkerProperties props3 =
        WorkerMetric.WorkerProperties.create(
            /*workerId= */ 3,
            /*processId= */ 300,
            /*mnemonic= */ "Proto",
            /*isMultiplex= */ true,
            /*isSandboxed= */ true);
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
    ImmutableMap<Integer, WorkerMetric.WorkerProperties> propsMap =
        ImmutableMap.of(
            1, props1,
            2, props2);
    ImmutableMap<Long, Integer> memoryUsageMap =
        ImmutableMap.of(
            100L, stat1.getUsedMemoryInKB(),
            200L, stat2.getUsedMemoryInKB());
    WorkerMetricsCollector.MemoryCollectionResult memoryCollectionResult =
        new WorkerMetricsCollector.MemoryCollectionResult(memoryUsageMap, collectionTime);
    ImmutableList<WorkerMetric> expectedMetrics =
        ImmutableList.of(workerMetric1, workerMetric2, workerMetric3);

    when(spyCollector.collectMemoryUsageByPid(any(), pidsCaptor.capture()))
        .thenReturn(memoryCollectionResult);
    clock.setTime(registrationTime.toEpochMilli());

    spyCollector.registerWorker(props1);
    spyCollector.registerWorker(props2);
    spyCollector.registerWorker(props3);

    ImmutableList<WorkerMetric> metrics = spyCollector.collectMetrics();

    assertThat(pidsCaptor.getValue()).containsExactlyElementsIn(expectedPids);
    assertThat(metrics).containsExactlyElementsIn(expectedMetrics);
    assertThat(spyCollector.getWorkerIdToWorkerProperties()).isEqualTo(propsMap);
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
