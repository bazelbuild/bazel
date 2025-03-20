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

import com.google.devtools.build.lib.testutil.TestThread;
import java.time.Duration;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TimeSeries}. */
@RunWith(JUnit4.class)
public final class TimeSeriesTest {
  @Test
  public void testAddRange() {
    TimeSeries timeSeries = new TimeSeries(Duration.ofMillis(42), Duration.ofMillis(100));
    timeSeries.addRange(Duration.ofMillis(42), Duration.ofMillis(142));
    timeSeries.addRange(Duration.ofMillis(442), Duration.ofMillis(542));
    double[] values = timeSeries.toDoubleArray(5);
    assertThat(values).usingTolerance(1.0e-10).containsExactly(1, 0, 0, 0, 1).inOrder();
  }

  @Test
  public void testAddRangeWithValue() {
    TimeSeries timeSeries = new TimeSeries(Duration.ofMillis(42), Duration.ofMillis(100));
    timeSeries.addRange(Duration.ofMillis(42), Duration.ofMillis(242), 3);
    timeSeries.addRange(Duration.ofMillis(442), Duration.ofMillis(542), 0.5);
    double[] values = timeSeries.toDoubleArray(5);
    assertThat(values).usingTolerance(1.0e-10).containsExactly(3, 3, 0, 0, .5).inOrder();
  }

  @Test
  public void testAddRangeOverlappingWithValue() {
    TimeSeries timeSeries = new TimeSeries(Duration.ofMillis(42), Duration.ofMillis(100));
    timeSeries.addRange(Duration.ofMillis(42), Duration.ofMillis(242), 3);
    timeSeries.addRange(Duration.ofMillis(142), Duration.ofMillis(442), 0.5);
    double[] values = timeSeries.toDoubleArray(5);
    assertThat(values).usingTolerance(1.0e-10).containsExactly(3, 3.5, 0.5, 0.5, 0).inOrder();
  }

  @Test
  public void testAddRangeFractions() {
    TimeSeries timeSeries = new TimeSeries(Duration.ofMillis(42), Duration.ofMillis(100));
    timeSeries.addRange(Duration.ofMillis(92), Duration.ofMillis(267));
    double[] values = timeSeries.toDoubleArray(5);
    assertThat(values).usingTolerance(1.0e-10).containsExactly(0.5, 1, 0.25, 0, 0).inOrder();
  }

  @Test
  public void testAddRangeWithValueFractions() {
    TimeSeries timeSeries = new TimeSeries(Duration.ofMillis(42), Duration.ofMillis(100));
    timeSeries.addRange(Duration.ofMillis(92), Duration.ofMillis(267), 3);
    double[] values = timeSeries.toDoubleArray(5);
    assertThat(values).usingTolerance(1.0e-10).containsExactly(1.5, 3, 0.75, 0, 0).inOrder();
  }

  @Test
  public void testResize() {
    TimeSeries timeSeries = new TimeSeries(Duration.ZERO, Duration.ofMillis(100));
    timeSeries.addRange(Duration.ZERO, Duration.ofMillis(100 * 100 + 1), 42);
    double[] values = timeSeries.toDoubleArray(101);
    double[] expected = new double[101];
    Arrays.fill(expected, 0, expected.length - 1, 42);
    expected[expected.length - 1] = 0.42;
    assertThat(values).usingTolerance(1.0e-10).containsExactly(expected).inOrder();
  }

  @Test
  public void testParallelism() throws Exception {
    // Define two threads. One is writing 1 on odd places, and another writes 2 on even places.
    TimeSeries timeSeries = new TimeSeries(Duration.ZERO, Duration.ofMillis(100));
    CountDownLatch latch = new CountDownLatch(2);
    TestThread thread1 =
        new TestThread(
            () -> {
              latch.countDown();
              latch.await();
              for (int i = 0; i < 50; i++) {
                timeSeries.addRange(
                    Duration.ofMillis(2 * i * 100), Duration.ofMillis((2 * i + 1) * 100), 1);
              }
            });
    TestThread thread2 =
        new TestThread(
            () -> {
              latch.countDown();
              latch.await();
              for (int i = 0; i < 50; i++) {
                timeSeries.addRange(
                    Duration.ofMillis((2 * i + 1) * 100), Duration.ofMillis((2 * i + 2) * 100), 2);
              }
            });
    double[] expected = new double[100];
    for (int i = 0; i < 100; i++) {
      if (i % 2 == 0) {
        expected[i] = 1;
      } else {
        expected[i] = 2;
      }
    }

    thread1.start();
    thread2.start();

    thread1.joinAndAssertState(10000);
    thread2.joinAndAssertState(10000);
    assertThat(timeSeries.toDoubleArray(100))
        .usingTolerance(1.0e-10)
        .containsExactly(expected)
        .inOrder();
  }
}
