// Copyright 2018 The Bazel Authors. All rights reserved.
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

import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CpuUsageTimeSeries}. */
@RunWith(JUnit4.class)
public final class CpuUsageTimeSeriesTest {
  @Test
  public void testAddRange() {
    CpuUsageTimeSeries cpuUsageTimeSeries = new CpuUsageTimeSeries(42, 100);
    cpuUsageTimeSeries.addRange(42, 142);
    cpuUsageTimeSeries.addRange(442, 542);
    double[] values = cpuUsageTimeSeries.toDoubleArray(5);
    assertThat(values).usingTolerance(1.0e-10).containsExactly(1, 0, 0, 0, 1).inOrder();
  }

  @Test
  public void testAddRangeWithValue() {
    CpuUsageTimeSeries cpuUsageTimeSeries = new CpuUsageTimeSeries(42, 100);
    cpuUsageTimeSeries.addRange(42, 242, 3);
    cpuUsageTimeSeries.addRange(442, 542, 0.5);
    double[] values = cpuUsageTimeSeries.toDoubleArray(5);
    assertThat(values).usingTolerance(1.0e-10).containsExactly(3, 3, 0, 0, .5).inOrder();
  }

  @Test
  public void testAddRangeOverlappingWithValue() {
    CpuUsageTimeSeries cpuUsageTimeSeries = new CpuUsageTimeSeries(42, 100);
    cpuUsageTimeSeries.addRange(42, 242, 3);
    cpuUsageTimeSeries.addRange(142, 442, 0.5);
    double[] values = cpuUsageTimeSeries.toDoubleArray(5);
    assertThat(values).usingTolerance(1.0e-10).containsExactly(3, 3.5, 0.5, 0.5, 0).inOrder();
  }

  @Test
  public void testAddRangeFractions() {
    CpuUsageTimeSeries cpuUsageTimeSeries = new CpuUsageTimeSeries(42, 100);
    cpuUsageTimeSeries.addRange(92, 267);
    double[] values = cpuUsageTimeSeries.toDoubleArray(5);
    assertThat(values).usingTolerance(1.0e-10).containsExactly(0.5, 1, 0.25, 0, 0).inOrder();
  }

  @Test
  public void testAddRangeWithValueFractions() {
    CpuUsageTimeSeries cpuUsageTimeSeries = new CpuUsageTimeSeries(42, 100);
    cpuUsageTimeSeries.addRange(92, 267, 3);
    double[] values = cpuUsageTimeSeries.toDoubleArray(5);
    assertThat(values).usingTolerance(1.0e-10).containsExactly(1.5, 3, 0.75, 0, 0).inOrder();
  }

  @Test
  public void testResize() {
    CpuUsageTimeSeries cpuUsageTimeSeries = new CpuUsageTimeSeries(0, 100);
    cpuUsageTimeSeries.addRange(0, 100 * 100 + 1, 42);
    double[] values = cpuUsageTimeSeries.toDoubleArray(101);
    double[] expected = new double[101];
    Arrays.fill(expected, 0, expected.length - 1, 42);
    expected[expected.length - 1] = 0.42;
    assertThat(values).usingTolerance(1.0e-10).containsExactly(expected).inOrder();
  }
}
