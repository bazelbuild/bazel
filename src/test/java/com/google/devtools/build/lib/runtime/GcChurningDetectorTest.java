// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;

import com.google.devtools.build.lib.runtime.MemoryPressure.MemoryPressureStats;
import com.google.devtools.build.lib.runtime.MemoryPressure.MemoryPressureStats.FullGcFractionPoint;
import com.google.devtools.build.lib.testutil.ManualClock;
import java.time.Duration;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class GcChurningDetectorTest {
  @Test
  public void populateStats() {
    ManualClock fakeClock = new ManualClock();

    GcChurningDetector underTest = new GcChurningDetector(fakeClock);

    fakeClock.advance(Duration.ofMillis(50L));
    underTest.handle(fullGcEvent(Duration.ofMillis(10L)));

    fakeClock.advance(Duration.ofMillis(50L));
    underTest.handle(fullGcEvent(Duration.ofMillis(40L)));

    MemoryPressureStats.Builder actualBuilder = MemoryPressureStats.newBuilder();
    underTest.populateStats(actualBuilder);

    assertThat(actualBuilder.build())
        .isEqualTo(
            MemoryPressureStats.newBuilder()
                .addFullGcFractionPoint(
                    FullGcFractionPoint.newBuilder()
                        .setInvocationWallTimeSoFarMs(50)
                        .setFullGcFractionSoFar(0.2)
                        .build())
                .addFullGcFractionPoint(
                    FullGcFractionPoint.newBuilder()
                        .setInvocationWallTimeSoFarMs(100)
                        .setFullGcFractionSoFar(0.5)
                        .build())
                .build());
  }

  @Test
  public void doesNotRecordDataPointIfInvocationWallTimeSoFarIsLessThanOneMillisecond() {
    ManualClock fakeClock = new ManualClock();

    GcChurningDetector underTest = new GcChurningDetector(fakeClock);

    fakeClock.advance(Duration.ofNanos(456L));
    underTest.handle(fullGcEvent(Duration.ofNanos(123L)));

    fakeClock.advance(Duration.ofMillis(2L));
    underTest.handle(fullGcEvent(Duration.ofMillis(1L)));

    MemoryPressureStats.Builder actualBuilder = MemoryPressureStats.newBuilder();
    underTest.populateStats(actualBuilder);

    assertThat(actualBuilder.build())
        .isEqualTo(
            MemoryPressureStats.newBuilder()
                .addFullGcFractionPoint(
                    FullGcFractionPoint.newBuilder()
                        .setInvocationWallTimeSoFarMs(2)
                        .setFullGcFractionSoFar(0.5)
                        .build())
                .build());
  }

  private static MemoryPressureEvent fullGcEvent(Duration duration) {
    return MemoryPressureEvent.newBuilder()
        .setWasFullGc(true)
        .setTenuredSpaceUsedBytes(1234L)
        .setTenuredSpaceMaxBytes(5678L)
        .setDuration(duration)
        .build();
  }
}
