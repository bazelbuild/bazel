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
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.util.OS;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/** Unit tests for the WorkerSpawnRunner. */
@RunWith(JUnit4.class)
public class WorkerMetricsCollectorTest {
  @Rule public final MockitoRule mockito = MockitoJUnit.rule();
  @Mock ExtendedEventHandler reporter;
  @Mock EventBus eventBus;

  @Before
  public void setUp() {
    doNothing().when(eventBus).register(any());
  }

  @Test
  public void testCollectStats_ignoreSpaces() throws Exception {
    WorkerMetricsCollector collector = new WorkerMetricsCollector(reporter, eventBus);
    WorkerMetricsCollector spyCollector = spy(collector);

    String psOutput = "    PID  \t  RSS\n   1  3216 \t\n  \t 2 \t 4096 \t";
    List<Long> pids = Arrays.asList(1L, 2L);
    InputStream psStream = new ByteArrayInputStream(psOutput.getBytes(UTF_8));
    Process process = mock(Process.class);

    when(spyCollector.buildPsProcess(pids)).thenReturn(process);
    when(process.getInputStream()).thenReturn(psStream);

    Map<Long, WorkerMetric.WorkerStat> pidResults = spyCollector.collectStats(OS.LINUX, pids);

    assertThat(pidResults).hasSize(2);
    assertThat(pidResults.get(1L).getUsedMemoryInKB()).isEqualTo(3);
    assertThat(pidResults.get(2L).getUsedMemoryInKB()).isEqualTo(4);
  }

  @Test
  public void testCollectStats_filterInvalidPids() throws Exception {
    WorkerMetricsCollector collector = new WorkerMetricsCollector(reporter, eventBus);
    WorkerMetricsCollector spyCollector = spy(collector);

    String psOutput = "PID  RSS  \n 1  3216";
    List<Long> pids = Arrays.asList(1L, 0L);

    InputStream psStream = new ByteArrayInputStream(psOutput.getBytes(UTF_8));
    Process process = mock(Process.class);

    when(spyCollector.buildPsProcess(pids)).thenReturn(process);
    when(process.getInputStream()).thenReturn(psStream);

    Map<Long, WorkerMetric.WorkerStat> pidResults = spyCollector.collectStats(OS.LINUX, pids);

    assertThat(pidResults).hasSize(1);
    assertThat(pidResults.get(1L).getUsedMemoryInKB()).isEqualTo(3);
  }
}
