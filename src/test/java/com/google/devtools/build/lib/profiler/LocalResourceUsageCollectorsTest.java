// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.profiler;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.profiler.LocalResourceUsageCollectors.LocalMemoryUsageCollector;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link LocalResourceUsageCollectors}. */
@RunWith(JUnit4.class)
public class LocalResourceUsageCollectorsTest {

  private static final long MB = 1024 * 1024;

  private final MemoryMXBean memoryBean = Mockito.mock(MemoryMXBean.class);
  private final BugReporter bugReporter = Mockito.mock(BugReporter.class);

  @Before
  public void resetPeak() {
    // The peak is tracked in static state, so reset it before each test for isolation.
    LocalResourceUsageCollectors.resetPeakBazelMemoryUsage();
  }

  @Test
  public void peakUsedMemory_isZeroBeforeAnySample() {
    assertThat(LocalResourceUsageCollectors.getPeakBazelMemoryUsageBytes()).isEqualTo(0L);
  }

  @Test
  public void localMemoryUsageCollector_recordsPeakOfHeapPlusNonHeap() {
    LocalMemoryUsageCollector collector = new LocalMemoryUsageCollector(memoryBean, bugReporter);

    // First sample: 100 MB heap + 20 MB non-heap = 120 MB.
    setUsedMemory(100 * MB, 20 * MB);
    collector.collect(/* deltaNanos= */ 1.0, (task, value) -> {});
    assertThat(LocalResourceUsageCollectors.getPeakBazelMemoryUsageBytes()).isEqualTo(120 * MB);

    // A larger sample raises the peak: 200 MB heap + 30 MB non-heap = 230 MB.
    setUsedMemory(200 * MB, 30 * MB);
    collector.collect(1.0, (task, value) -> {});
    assertThat(LocalResourceUsageCollectors.getPeakBazelMemoryUsageBytes()).isEqualTo(230 * MB);

    // A smaller sample does not lower the peak.
    setUsedMemory(50 * MB, 10 * MB);
    collector.collect(1.0, (task, value) -> {});
    assertThat(LocalResourceUsageCollectors.getPeakBazelMemoryUsageBytes()).isEqualTo(230 * MB);
  }

  @Test
  public void localMemoryUsageCollector_stillReportsSampleToProfile() {
    LocalMemoryUsageCollector collector = new LocalMemoryUsageCollector(memoryBean, bugReporter);
    setUsedMemory(100 * MB, 20 * MB);

    long[] reported = {-1};
    // The value written to the JSON trace profile is in megabytes.
    collector.collect(1.0, (task, value) -> reported[0] = value.longValue());

    assertThat(reported[0]).isEqualTo(120);
  }

  @Test
  public void resetPeakBazelMemoryUsage_clearsPeak() {
    LocalMemoryUsageCollector collector = new LocalMemoryUsageCollector(memoryBean, bugReporter);
    setUsedMemory(100 * MB, 20 * MB);
    collector.collect(1.0, (task, value) -> {});
    assertThat(LocalResourceUsageCollectors.getPeakBazelMemoryUsageBytes()).isGreaterThan(0L);

    LocalResourceUsageCollectors.resetPeakBazelMemoryUsage();

    assertThat(LocalResourceUsageCollectors.getPeakBazelMemoryUsageBytes()).isEqualTo(0L);
  }

  private void setUsedMemory(long heapUsedBytes, long nonHeapUsedBytes) {
    when(memoryBean.getHeapMemoryUsage())
        .thenReturn(new MemoryUsage(/* init= */ -1, heapUsedBytes, heapUsedBytes, /* max= */ -1));
    when(memoryBean.getNonHeapMemoryUsage())
        .thenReturn(
            new MemoryUsage(/* init= */ -1, nonHeapUsedBytes, nonHeapUsedBytes, /* max= */ -1));
  }
}
