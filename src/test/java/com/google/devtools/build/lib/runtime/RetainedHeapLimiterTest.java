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
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.runtime.MemoryPressureHandler.Event;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.util.AbruptExitException;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RetainedHeapLimiter}. */
@RunWith(JUnit4.class)
public final class RetainedHeapLimiterTest {
  @After
  public void cleanUp() {
    BugReport.maybePropagateUnprocessedThrowableIfInTest();
  }

  @Test
  public void noTenuredSpaceFound() throws AbruptExitException {
    RetainedHeapLimiter underTest = RetainedHeapLimiter.create(BugReporter.defaultInstance());

    AbruptExitException e =
        assertThrows(
            AbruptExitException.class, () -> underTest.setThreshold(/*listening=*/ false, 80));
    FailureDetails.FailureDetail failureDetail = e.getDetailedExitCode().getFailureDetail();
    assertThat(failureDetail.getMessage())
        .contains("unable to watch for GC events to exit JVM when 80% of heap is used");
    assertThat(failureDetail.getMemoryOptions().getCode())
        .isEqualTo(
            FailureDetails.MemoryOptions.Code
                .EXPERIMENTAL_OOM_MORE_EAGERLY_NO_TENURED_COLLECTORS_FOUND);
  }

  @Test
  public void underThreshold_noOom() throws Exception {
    RetainedHeapLimiter underTest = RetainedHeapLimiter.create(BugReporter.defaultInstance());

    underTest.setThreshold(/*listening=*/ true, 99);

    underTest.handle(percentUsedAfterOtherGc(100));
    underTest.handle(percentUsedAfterForcedGc(89));
  }

  @Test
  public void overThreshold_oom() throws Exception {
    RetainedHeapLimiter underTest = RetainedHeapLimiter.create(BugReporter.defaultInstance());

    underTest.setThreshold(/*listening=*/ true, 90);

    // Triggers GC, and tells RetainedHeapLimiter to OOM if too much memory used next time.
    underTest.handle(percentUsedAfterOtherGc(91));

    assertThrows(
        SecurityException.class, // From attempt to halt jvm in test.
        () -> underTest.handle(percentUsedAfterForcedGc(91)));
    OutOfMemoryError oom =
        assertThrows(OutOfMemoryError.class, BugReport::maybePropagateUnprocessedThrowableIfInTest);

    assertThat(oom).hasMessageThat().contains("forcing exit due to GC thrashing");
    assertThat(oom).hasMessageThat().contains("tenured space is more than 90% occupied");
  }

  @Test
  public void externalGcNoTrigger() throws Exception {
    RetainedHeapLimiter underTest = RetainedHeapLimiter.create(BugReporter.defaultInstance());

    underTest.setThreshold(/*listening=*/ true, 90);

    // No trigger because cause was "System.gc()".
    underTest.handle(percentUsedAfterForcedGc(91));

    // Proof: no OOM.
    underTest.handle(percentUsedAfterForcedGc(91));
  }

  @Test
  public void triggerReset() throws Exception {
    RetainedHeapLimiter underTest = RetainedHeapLimiter.create(BugReporter.defaultInstance());

    underTest.setThreshold(/*listening=*/ true, 90);

    underTest.handle(percentUsedAfterOtherGc(91));

    // Got under the threshold, so no OOM.
    underTest.handle(percentUsedAfterForcedGc(89));

    // No OOM this time since wasn't triggered.
    underTest.handle(percentUsedAfterForcedGc(91));
  }

  @Test
  public void triggerRaceWithOtherGc() throws Exception {
    RetainedHeapLimiter underTest = RetainedHeapLimiter.create(BugReporter.defaultInstance());

    underTest.setThreshold(/*listening=*/ true, 90);

    underTest.handle(percentUsedAfterOtherGc(91));
    underTest.handle(percentUsedAfterOtherGc(91));
    assertThrows(
        SecurityException.class, // From attempt to halt jvm in test.
        () -> underTest.handle(percentUsedAfterForcedGc(91)));
    assertThrows(OutOfMemoryError.class, BugReport::maybePropagateUnprocessedThrowableIfInTest);
  }

  private static Event percentUsedAfterForcedGc(int percentUsed) {
    return percentUsedAfterGc(/*wasManualGc=*/ true, percentUsed);
  }

  private static Event percentUsedAfterOtherGc(int percentUsed) {
    return percentUsedAfterGc(/*wasManualGc=*/ false, percentUsed);
  }

  private static Event percentUsedAfterGc(boolean wasManualGc, int percentUsed) {
    checkArgument(percentUsed >= 0 && percentUsed <= 100, percentUsed);
    return Event.newBuilder()
        .setWasManualGc(wasManualGc)
        .setTenuredSpaceUsedBytes(percentUsed)
        .setTenuredSpaceMaxBytes(100L)
        .build();
  }
}
