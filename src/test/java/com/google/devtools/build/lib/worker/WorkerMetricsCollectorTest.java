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
import com.google.devtools.build.lib.util.OS;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.time.Instant;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
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
  @Captor ArgumentCaptor<List<Long>> pidsCaptor;

  @Before
  public void setUp() {
    WorkerMetricsCollector.instance().clear();
  }

  @Test
  public void testCollectStats_ignoreSpaces() throws Exception {
    String psOutput = "    PID  \t  RSS\n   1  3216 \t\n  \t 2 \t 4096 \t";
    ImmutableMap<Long, Long> subprocessesMap =
        ImmutableMap.of(
            1L, 1L,
            2L, 2L);
    List<Long> pids = Arrays.asList(1L, 2L);
    InputStream psStream = new ByteArrayInputStream(psOutput.getBytes(UTF_8));
    Process process = mock(Process.class);

    when(spyCollector.getSubprocesses(pids)).thenReturn(subprocessesMap);
    when(spyCollector.buildPsProcess(subprocessesMap.keySet())).thenReturn(process);
    when(process.getInputStream()).thenReturn(psStream);

    Map<Long, WorkerMetric.WorkerStat> pidResults = spyCollector.collectStats(OS.LINUX, pids);

    assertThat(pidResults).hasSize(2);
    assertThat(pidResults.get(1L).getUsedMemoryInKB()).isEqualTo(3216);
    assertThat(pidResults.get(2L).getUsedMemoryInKB()).isEqualTo(4096);
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
    List<Long> pids = Arrays.asList(1L, 2L, 5L);
    InputStream psStream = new ByteArrayInputStream(psOutput.getBytes(UTF_8));
    Process process = mock(Process.class);

    when(spyCollector.getSubprocesses(pids)).thenReturn(subprocessesMap);
    when(spyCollector.buildPsProcess(subprocessesMap.keySet())).thenReturn(process);
    when(process.getInputStream()).thenReturn(psStream);

    Map<Long, WorkerMetric.WorkerStat> pidResults = spyCollector.collectStats(OS.LINUX, pids);

    assertThat(pidResults).hasSize(3);
    assertThat(pidResults.get(1L).getUsedMemoryInKB()).isEqualTo(3216 + 1234);
    assertThat(pidResults.get(2L).getUsedMemoryInKB()).isEqualTo(4232 + 1001);
    assertThat(pidResults.get(5L).getUsedMemoryInKB()).isEqualTo(40000);
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
    ImmutableMap<Integer, WorkerMetric.WorkerProperties> map = ImmutableMap.of(1, props1);

    spyCollector.registerWorker(props1);
    assertThat(spyCollector.getWorkerIdToWorkerProperties()).hasSize(1);
    spyCollector.registerWorker(props2);
    assertThat(spyCollector.getWorkerIdToWorkerProperties()).hasSize(1);
    assertThat(spyCollector.getWorkerIdToWorkerProperties()).isEqualTo(map);
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
    WorkerMetric.WorkerStat stat1 = WorkerMetric.WorkerStat.create(1234, Instant.now());
    WorkerMetric.WorkerStat stat2 = WorkerMetric.WorkerStat.create(2345, Instant.now());
    WorkerMetric workerMetric1 = WorkerMetric.create(props1, stat1, true);
    WorkerMetric workerMetric2 = WorkerMetric.create(props2, stat2, true);
    WorkerMetric workerMetric3 = WorkerMetric.create(props3, null, false);
    ImmutableList<Long> expectedPids = ImmutableList.of(100L, 200L, 300L);
    ImmutableMap<Integer, WorkerMetric.WorkerProperties> propsMap =
        ImmutableMap.of(
            1, props1,
            2, props2);
    ImmutableMap<Long, WorkerMetric.WorkerStat> statsMap =
        ImmutableMap.of(
            100L, stat1,
            200L, stat2);
    ImmutableList<WorkerMetric> expectedMetrics =
        ImmutableList.of(workerMetric1, workerMetric2, workerMetric3);

    when(spyCollector.collectStats(any(), pidsCaptor.capture())).thenReturn(statsMap);

    spyCollector.registerWorker(props1);
    spyCollector.registerWorker(props2);
    spyCollector.registerWorker(props3);

    ImmutableList<WorkerMetric> metrics = spyCollector.collectMetrics();

    assertThat(pidsCaptor.getValue()).containsExactlyElementsIn(expectedPids);
    assertThat(metrics).containsExactlyElementsIn(expectedMetrics);
    assertThat(spyCollector.getWorkerIdToWorkerProperties()).isEqualTo(propsMap);
  }
}
