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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verifyNoMoreInteractions;

import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.bugreport.Crash;
import com.google.devtools.build.lib.runtime.MemoryPressure.MemoryPressureStats;
import com.google.devtools.build.lib.runtime.MemoryPressure.MemoryPressureStats.FullGcFractionPoint;
import com.google.devtools.build.lib.server.FailureDetails.Crash.OomCauseCategory;
import com.google.devtools.build.lib.testutil.ManualClock;
import java.time.Duration;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;

@RunWith(JUnit4.class)
public class GcChurningDetectorTest {
  private final BugReporter mockBugReporter = mock(BugReporter.class);

  @Test
  public void populateStats() {
    ManualClock fakeClock = new ManualClock();

    GcChurningDetector underTest =
        new GcChurningDetector(
            /* thresholdPercentage= */ 100,
            /* thresholdPercentageIfMultipleTopLevelTargets= */ 100,
            fakeClock,
            mockBugReporter);

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

    verifyNoOom();
  }

  @Test
  public void doesNotRecordDataPointIfInvocationWallTimeSoFarIsLessThanOneMillisecond() {
    ManualClock fakeClock = new ManualClock();

    GcChurningDetector underTest =
        new GcChurningDetector(
            /* thresholdPercentage= */ 100,
            /* thresholdPercentageIfMultipleTopLevelTargets= */ 100,
            fakeClock,
            mockBugReporter);

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

  @Test
  public void oom() {
    ManualClock fakeClock = new ManualClock();

    GcChurningDetector underTest =
        new GcChurningDetector(
            /* thresholdPercentage= */ 50,
            /* thresholdPercentageIfMultipleTopLevelTargets= */ 50,
            fakeClock,
            mockBugReporter);

    fakeClock.advance(Duration.ofMinutes(3L));
    underTest.handle(fullGcEvent(Duration.ofMinutes(1L)));
    verifyNoOom();

    fakeClock.advance(Duration.ofMinutes(1L));
    underTest.handle(fullGcEvent(Duration.ofMinutes(1L)));
    verifyOom();
  }

  @Test
  public void minInvocationWallTimeDuration() {
    ManualClock fakeClock = new ManualClock();

    GcChurningDetector underTest =
        new GcChurningDetector(
            /* thresholdPercentage= */ 50,
            /* thresholdPercentageIfMultipleTopLevelTargets= */ 50,
            fakeClock,
            mockBugReporter);

    fakeClock.advance(Duration.ofSeconds(30L));
    underTest.handle(fullGcEvent(Duration.ofSeconds(15L)));
    verifyNoOom();

    fakeClock.advance(Duration.ofSeconds(29L));
    underTest.handle(fullGcEvent(Duration.ofSeconds(14L)));
    verifyNoOom();

    fakeClock.advance(Duration.ofSeconds(1L));
    underTest.handle(fullGcEvent(Duration.ofSeconds(1L)));
    verifyOom();
  }

  @Test
  public void thresholdPercentageIfMultipleTopLevelTargets_onlySingleTarget() {
    ManualClock fakeClock = new ManualClock();

    GcChurningDetector underTest =
        new GcChurningDetector(
            /* thresholdPercentage= */ 100,
            /* thresholdPercentageIfMultipleTopLevelTargets= */ 50,
            fakeClock,
            mockBugReporter);

    fakeClock.advance(Duration.ofSeconds(60L));
    underTest.handle(fullGcEvent(Duration.ofSeconds(30L)));
    verifyNoOom();

    underTest.targetParsingComplete(1);
    fakeClock.advance(Duration.ofSeconds(30L));
    underTest.handle(fullGcEvent(Duration.ofSeconds(20L)));
    verifyNoOom();
  }

  @Test
  public void thresholdPercentageIfMultipleTopLevelTargets() {
    ManualClock fakeClock = new ManualClock();

    GcChurningDetector underTest =
        new GcChurningDetector(
            /* thresholdPercentage= */ 100,
            /* thresholdPercentageIfMultipleTopLevelTargets= */ 50,
            fakeClock,
            mockBugReporter);

    fakeClock.advance(Duration.ofSeconds(60L));
    underTest.handle(fullGcEvent(Duration.ofSeconds(40L)));
    verifyNoOom();

    underTest.targetParsingComplete(2);
    fakeClock.advance(Duration.ofSeconds(30L));
    underTest.handle(fullGcEvent(Duration.ofSeconds(20L)));
    verifyOom();
  }

  private void verifyNoOom() {
    verifyNoInteractions(mockBugReporter);
  }

  private void verifyOom() {
    ArgumentCaptor<Crash> crashArgument = ArgumentCaptor.forClass(Crash.class);
    verify(mockBugReporter).handleCrash(crashArgument.capture(), any());
    Crash crash = crashArgument.getValue();
    Throwable oom = crash.getThrowable();
    assertThat(oom).isInstanceOf(OutOfMemoryError.class);
    assertThat(crash.getDetailedExitCode().getFailureDetail().getCrash().getOomCauseCategory())
        .isEqualTo(OomCauseCategory.GC_CHURNING);
  }

  @After
  public void verifyNoMoreBugReports() {
    verifyNoMoreInteractions(mockBugReporter);
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
