// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verifyNoMoreInteractions;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.bugreport.Crash;
import com.google.devtools.build.lib.runtime.GcThrashingDetector.Limit;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.time.Duration;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.ArgumentCaptor;

/** Tests for {@link GcThrashingDetector}. */
@RunWith(TestParameterInjector.class)
public final class GcThrashingDetectorTest {

  private final BugReporter bugReporter = mock(BugReporter.class);
  private final ManualClock clock = new ManualClock();

  private enum GcType {
    ORGANIC_FULL,
    MINOR,
    MANUAL
  }

  @Before
  public void setClock() {
    clock.advanceMillis(100000);
  }

  @After
  public void verifyNoMoreBugReports() {
    verifyNoMoreInteractions(bugReporter);
  }

  @Test
  public void limitViolated_oom() {
    GcThrashingDetector detector =
        createDetector(/* threshold= */ 90, Limit.of(Duration.ofSeconds(10), 2));

    detector.handle(percentUsedAfterGc(91, GcType.ORGANIC_FULL));
    clock.advance(Duration.ofSeconds(5));
    detector.handle(percentUsedAfterGc(91, GcType.ORGANIC_FULL));

    verifyOom();
  }

  @Test
  public void underThreshold_noOom() {
    GcThrashingDetector detector =
        createDetector(/* threshold= */ 90, Limit.of(Duration.ofSeconds(10), 2));

    detector.handle(percentUsedAfterGc(89, GcType.ORGANIC_FULL));
    clock.advance(Duration.ofSeconds(5));
    detector.handle(percentUsedAfterGc(89, GcType.ORGANIC_FULL));

    verifyNoOom();
  }

  @Test
  public void limitViolatedAfterUnderThreshold_oom() {
    GcThrashingDetector detector =
        createDetector(/* threshold= */ 90, Limit.of(Duration.ofSeconds(10), 2));

    detector.handle(percentUsedAfterGc(89, GcType.ORGANIC_FULL));
    clock.advance(Duration.ofSeconds(1));
    detector.handle(percentUsedAfterGc(91, GcType.ORGANIC_FULL));
    clock.advance(Duration.ofSeconds(1));
    detector.handle(percentUsedAfterGc(91, GcType.ORGANIC_FULL));

    verifyOom();
  }

  @Test
  public void outsideOfPeriod_noOom() {
    GcThrashingDetector detector =
        createDetector(/* threshold= */ 90, Limit.of(Duration.ofSeconds(10), 2));

    detector.handle(percentUsedAfterGc(91, GcType.ORGANIC_FULL));
    clock.advance(Duration.ofSeconds(11));
    detector.handle(percentUsedAfterGc(91, GcType.ORGANIC_FULL));

    verifyNoOom();
  }

  @Test
  public void backUnderThreshold_noOom(@TestParameter GcType type) {
    GcThrashingDetector detector =
        createDetector(/* threshold= */ 90, Limit.of(Duration.ofSeconds(10), 2));

    detector.handle(percentUsedAfterGc(91, GcType.ORGANIC_FULL));
    clock.advance(Duration.ofSeconds(1));
    detector.handle(percentUsedAfterGc(89, type));
    clock.advance(Duration.ofSeconds(1));
    detector.handle(percentUsedAfterGc(91, GcType.ORGANIC_FULL));

    verifyNoOom();
  }

  @Test
  public void notOrganicFullGc_noOom(@TestParameter({"MINOR", "MANUAL"}) GcType type) {
    GcThrashingDetector detector =
        createDetector(/* threshold= */ 90, Limit.of(Duration.ofSeconds(10), 2));

    detector.handle(percentUsedAfterGc(91, type));
    clock.advance(Duration.ofSeconds(5));
    detector.handle(percentUsedAfterGc(91, GcType.ORGANIC_FULL));

    verifyNoOom();
  }

  @Test
  public void multipleLimits_noneViolated_noOom() {
    GcThrashingDetector detector =
        createDetector(
            /* threshold= */ 90,
            Limit.of(Duration.ofSeconds(10), 2),
            Limit.of(Duration.ofMinutes(1), 3));

    detector.handle(percentUsedAfterGc(91, GcType.ORGANIC_FULL));
    clock.advance(Duration.ofSeconds(11));
    detector.handle(percentUsedAfterGc(91, GcType.ORGANIC_FULL));
    clock.advance(Duration.ofSeconds(50));
    detector.handle(percentUsedAfterGc(91, GcType.ORGANIC_FULL));

    verifyNoOom();
  }

  @Test
  public void multipleLimits_firstViolated_oom() {
    GcThrashingDetector detector =
        createDetector(
            /* threshold= */ 90,
            Limit.of(Duration.ofSeconds(10), 2),
            Limit.of(Duration.ofMinutes(1), 3));

    detector.handle(percentUsedAfterGc(91, GcType.ORGANIC_FULL));
    clock.advance(Duration.ofSeconds(5));
    detector.handle(percentUsedAfterGc(91, GcType.ORGANIC_FULL));

    verifyOomWithMessage("2 consecutive full GCs within the past 10 seconds");
  }

  @Test
  public void multipleLimits_secondViolated_oom() {
    GcThrashingDetector detector =
        createDetector(
            /* threshold= */ 90,
            Limit.of(Duration.ofSeconds(10), 2),
            Limit.of(Duration.ofMinutes(1), 3));

    detector.handle(percentUsedAfterGc(91, GcType.ORGANIC_FULL));
    clock.advance(Duration.ofSeconds(11));
    detector.handle(percentUsedAfterGc(91, GcType.ORGANIC_FULL));
    clock.advance(Duration.ofSeconds(11));
    detector.handle(percentUsedAfterGc(91, GcType.ORGANIC_FULL));

    verifyOomWithMessage("3 consecutive full GCs within the past 60 seconds");
  }

  private GcThrashingDetector createDetector(int threshold, Limit... limits) {
    return new GcThrashingDetector(threshold, ImmutableList.copyOf(limits), clock, bugReporter);
  }

  @CanIgnoreReturnValue
  private OutOfMemoryError verifyOom() {
    ArgumentCaptor<Crash> crashArgument = ArgumentCaptor.forClass(Crash.class);
    verify(bugReporter).handleCrash(crashArgument.capture(), any());
    Throwable oom = crashArgument.getValue().getThrowable();
    assertThat(oom).isInstanceOf(OutOfMemoryError.class);
    return (OutOfMemoryError) oom;
  }

  private void verifyOomWithMessage(String message) {
    OutOfMemoryError oom = verifyOom();
    assertThat(oom).hasMessageThat().contains(message);
  }

  private void verifyNoOom() {
    verifyNoInteractions(bugReporter);
  }

  private static MemoryPressureEvent percentUsedAfterGc(int percentUsed, GcType type) {
    checkArgument(percentUsed >= 0, percentUsed);
    MemoryPressureEvent.Builder event =
        MemoryPressureEvent.newBuilder()
            .setTenuredSpaceUsedBytes(percentUsed)
            .setTenuredSpaceMaxBytes(100L);
    switch (type) {
      case ORGANIC_FULL:
        event.setWasFullGc(true);
        break;
      case MINOR:
        break;
      case MANUAL:
        event.setWasManualGc(true).setWasFullGc(true);
        break;
    }
    return event.build();
  }
}
