// Copyright 2019 The Bazel Authors. All rights reserved.
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
import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verifyNoMoreInteractions;

import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.bugreport.Crash;
import com.google.devtools.build.lib.runtime.MemoryPressure.MemoryPressureStats;
import com.google.devtools.build.lib.server.FailureDetails.MemoryOptions.Code;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.common.options.Options;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.lang.ref.WeakReference;
import java.time.Duration;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.ArgumentCaptor;
import org.mockito.ArgumentMatchers;

/** Tests for {@link RetainedHeapLimiter}. */
@RunWith(TestParameterInjector.class)
public final class RetainedHeapLimiterTest {

  private final BugReporter bugReporter = mock(BugReporter.class);
  private final ManualClock clock = new ManualClock();
  private final MemoryPressureOptions options = Options.getDefaults(MemoryPressureOptions.class);

  private final RetainedHeapLimiter underTest =
      RetainedHeapLimiter.createForTest(bugReporter, clock);

  @Before
  public void setClock() {
    clock.advanceMillis(100000);
  }

  @After
  public void verifyNoMoreBugReports() {
    verifyNoMoreInteractions(bugReporter);
  }

  @Test
  public void underThreshold_noOom() throws Exception {
    options.oomMoreEagerlyThreshold = 99;
    underTest.setOptions(options);

    underTest.handle(percentUsedAfterOtherGc(100));
    underTest.handle(percentUsedAfterForcedGc(89));

    verifyNoInteractions(bugReporter);
    assertStats(MemoryPressureStats.newBuilder().setManuallyTriggeredGcs(1));
  }

  @Test
  public void overThreshold_oom() throws Exception {
    options.oomMoreEagerlyThreshold = 90;
    underTest.setOptions(options);

    // Triggers GC, and tells RetainedHeapLimiter to OOM if too much memory used next time.
    underTest.handle(percentUsedAfterOtherGc(91));

    underTest.handle(percentUsedAfterForcedGc(91));

    ArgumentCaptor<Crash> crashArgument = ArgumentCaptor.forClass(Crash.class);
    verify(bugReporter).handleCrash(crashArgument.capture(), ArgumentMatchers.any());
    OutOfMemoryError oom = (OutOfMemoryError) crashArgument.getValue().getThrowable();
    assertThat(oom).hasMessageThat().contains("forcing exit due to GC thrashing");
    assertThat(oom).hasMessageThat().contains("tenured space is more than 90% occupied");

    assertStats(MemoryPressureStats.newBuilder().setManuallyTriggeredGcs(1));
  }

  @Test
  public void inactiveAfterOom() throws Exception {
    options.oomMoreEagerlyThreshold = 90;
    options.minTimeBetweenTriggeredGc = Duration.ZERO;
    underTest.setOptions(options);

    underTest.handle(percentUsedAfterOtherGc(91));
    underTest.handle(percentUsedAfterForcedGc(91));
    verify(bugReporter).handleCrash(any(), any());

    // No more GC or bug reports even if notifications come in after an OOM is in progress.
    WeakReference<?> ref = new WeakReference<>(new Object());
    clock.advanceMillis(Duration.ofMinutes(1).toMillis());
    underTest.handle(percentUsedAfterOtherGc(91));
    underTest.handle(percentUsedAfterForcedGc(91));
    assertThat(ref.get()).isNotNull();
    verifyNoMoreBugReports();

    assertStats(MemoryPressureStats.newBuilder().setManuallyTriggeredGcs(1));
  }

  @Test
  public void externalGcNoTrigger() throws Exception {
    options.oomMoreEagerlyThreshold = 90;
    underTest.setOptions(options);

    // No trigger because cause was "System.gc()".
    underTest.handle(percentUsedAfterForcedGc(91));

    // Proof: no OOM.
    underTest.handle(percentUsedAfterForcedGc(91));
    verifyNoInteractions(bugReporter);

    assertStats(MemoryPressureStats.newBuilder().setManuallyTriggeredGcs(0));
  }

  @Test
  public void triggerReset() throws Exception {
    options.oomMoreEagerlyThreshold = 90;
    underTest.setOptions(options);

    underTest.handle(percentUsedAfterOtherGc(91));

    // Got under the threshold, so no OOM.
    underTest.handle(percentUsedAfterForcedGc(89));

    // No OOM this time since wasn't triggered.
    underTest.handle(percentUsedAfterForcedGc(91));
    verifyNoInteractions(bugReporter);
  }

  @Test
  public void triggerRaceWithOtherGc() throws Exception {
    options.oomMoreEagerlyThreshold = 90;
    underTest.setOptions(options);

    underTest.handle(percentUsedAfterOtherGc(91));
    underTest.handle(percentUsedAfterOtherGc(91));
    underTest.handle(percentUsedAfterForcedGc(91));

    ArgumentCaptor<Crash> crashArgument = ArgumentCaptor.forClass(Crash.class);
    verify(bugReporter).handleCrash(crashArgument.capture(), ArgumentMatchers.any());
    assertThat(crashArgument.getValue().getThrowable()).isInstanceOf(OutOfMemoryError.class);
  }

  @Test
  public void minTimeBetweenGc_lessThan_noGc() throws Exception {
    options.oomMoreEagerlyThreshold = 90;
    options.minTimeBetweenTriggeredGc = Duration.ofMinutes(1);
    underTest.setOptions(options);
    WeakReference<?> ref = new WeakReference<>(new Object());

    underTest.handle(percentUsedAfterOtherGc(91));
    assertThat(ref.get()).isNull();
    underTest.handle(percentUsedAfterForcedGc(89));

    ref = new WeakReference<>(new Object());
    clock.advanceMillis(Duration.ofSeconds(59).toMillis());
    underTest.handle(percentUsedAfterOtherGc(91));
    assertThat(ref.get()).isNotNull();

    assertStats(
        MemoryPressureStats.newBuilder()
            .setManuallyTriggeredGcs(1)
            .setMaxConsecutiveIgnoredGcsOverThreshold(1));
  }

  @Test
  public void minTimeBetweenGc_greaterThan_gc() throws Exception {
    options.oomMoreEagerlyThreshold = 90;
    options.minTimeBetweenTriggeredGc = Duration.ofMinutes(1);
    underTest.setOptions(options);
    WeakReference<?> ref = new WeakReference<>(new Object());

    underTest.handle(percentUsedAfterOtherGc(91));
    assertThat(ref.get()).isNull();
    underTest.handle(percentUsedAfterForcedGc(89));

    ref = new WeakReference<>(new Object());
    clock.advanceMillis(Duration.ofSeconds(61).toMillis());
    underTest.handle(percentUsedAfterOtherGc(91));
    assertThat(ref.get()).isNull();

    assertStats(
        MemoryPressureStats.newBuilder()
            .setManuallyTriggeredGcs(2)
            .setMaxConsecutiveIgnoredGcsOverThreshold(0));
  }

  @Test
  public void reportsMaxConsecutiveIgnored() throws Exception {
    options.oomMoreEagerlyThreshold = 90;
    options.minTimeBetweenTriggeredGc = Duration.ofMinutes(1);
    underTest.setOptions(options);

    underTest.handle(percentUsedAfterOtherGc(91));
    underTest.handle(percentUsedAfterForcedGc(89));
    for (int i = 0; i < 6; i++) {
      underTest.handle(percentUsedAfterOtherGc(91));
    }

    clock.advanceMillis(Duration.ofMinutes(2).toMillis());

    underTest.handle(percentUsedAfterOtherGc(91));
    underTest.handle(percentUsedAfterForcedGc(89));
    for (int i = 0; i < 8; i++) {
      underTest.handle(percentUsedAfterOtherGc(91));
    }
    underTest.handle(percentUsedAfterOtherGc(89)); // Breaks the streak of over threshold events.
    underTest.handle(percentUsedAfterOtherGc(91));

    clock.advanceMillis(Duration.ofMinutes(2).toMillis());

    underTest.handle(percentUsedAfterOtherGc(91));
    underTest.handle(percentUsedAfterOtherGc(89));
    for (int i = 0; i < 7; i++) {
      underTest.handle(percentUsedAfterOtherGc(91));
    }

    assertStats(
        MemoryPressureStats.newBuilder()
            .setManuallyTriggeredGcs(3)
            .setMaxConsecutiveIgnoredGcsOverThreshold(8));
  }

  @Test
  public void threshold100_noGcTriggeredEvenWithNonsenseStats() throws Exception {
    options.oomMoreEagerlyThreshold = 100;
    underTest.setOptions(options);
    WeakReference<?> ref = new WeakReference<>(new Object());

    underTest.handle(percentUsedAfterOtherGc(101));
    assertThat(ref.get()).isNotNull();

    assertStats(MemoryPressureStats.newBuilder().setManuallyTriggeredGcs(0));
  }

  @Test
  public void worksWithoutSettingOptions() {
    underTest.handle(percentUsedAfterOtherGc(95));
    assertStats(MemoryPressureStats.newBuilder().setManuallyTriggeredGcs(0));
  }

  @Test
  public void statsReset() throws Exception {
    options.oomMoreEagerlyThreshold = 90;
    underTest.setOptions(options);

    underTest.handle(percentUsedAfterOtherGc(91));

    assertStats(MemoryPressureStats.newBuilder().setManuallyTriggeredGcs(1));
    assertStats(MemoryPressureStats.newBuilder().setManuallyTriggeredGcs(0));
  }

  @Test
  public void invalidThreshold_throws(@TestParameter({"-1", "101"}) int threshold) {
    options.oomMoreEagerlyThreshold = threshold;
    AbruptExitException e =
        assertThrows(AbruptExitException.class, () -> underTest.setOptions(options));
    assertThat(e.getDetailedExitCode().getFailureDetail().getMemoryOptions().getCode())
        .isEqualTo(Code.EXPERIMENTAL_OOM_MORE_EAGERLY_THRESHOLD_INVALID_VALUE);
  }

  private static MemoryPressureEvent percentUsedAfterForcedGc(int percentUsed) {
    return percentUsedAfterGc(/* wasManualGc= */ true, percentUsed);
  }

  private static MemoryPressureEvent percentUsedAfterOtherGc(int percentUsed) {
    return percentUsedAfterGc(/* wasManualGc= */ false, percentUsed);
  }

  private static MemoryPressureEvent percentUsedAfterGc(boolean wasManualGc, int percentUsed) {
    checkArgument(percentUsed >= 0, percentUsed);
    return MemoryPressureEvent.newBuilder()
        .setWasManualGc(wasManualGc)
        .setWasFullGc(true)
        .setTenuredSpaceUsedBytes(percentUsed)
        .setTenuredSpaceMaxBytes(100L)
        .build();
  }

  private void assertStats(MemoryPressureStats.Builder expected) {
    MemoryPressureStats.Builder stats = MemoryPressureStats.newBuilder();
    underTest.addStatsAndReset(stats);
    assertThat(stats.build()).isEqualTo(expected.build());
  }
}
