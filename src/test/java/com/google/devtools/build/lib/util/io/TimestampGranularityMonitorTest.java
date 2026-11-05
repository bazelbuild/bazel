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

package com.google.devtools.build.lib.util.io;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.testutil.ManualClock;
import java.time.Duration;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TimestampGranularityMonitor}. */
@RunWith(JUnit4.class)
public class TimestampGranularityMonitorTest {
  private static final long SECOND_MILLIS = 1_700_000_000_000L;
  private static final long NOW_MILLIS = SECOND_MILLIS + 123;

  private final ManualClock clock = new ManualClock();
  private final TimestampGranularityMonitor tsgm = new TimestampGranularityMonitor(clock);

  @Before
  public void setClock() {
    clock.advanceMillis(NOW_MILLIS);
  }

  private static long nanos(long millis) {
    return millis * 1_000_000;
  }

  @Test
  public void noTimestamps_noWait() {
    assertThat(tsgm.getRemainingWait()).isEqualTo(Duration.ZERO);
  }

  @Test
  public void subMillisecondTimestamp_noWait() {
    tsgm.notifyDependenceOnFileTime(null, nanos(NOW_MILLIS) - 456_789);

    assertThat(tsgm.getRemainingWait()).isEqualTo(Duration.ZERO);
  }

  @Test
  public void wholeMillisecondTimestamp_waitsForNextMillisecond() {
    tsgm.notifyDependenceOnFileTime(null, nanos(NOW_MILLIS));

    assertThat(tsgm.getRemainingWait()).isEqualTo(Duration.ofMillis(1));
    clock.advanceMillis(1);
    assertThat(tsgm.getRemainingWait()).isEqualTo(Duration.ZERO);
  }

  @Test
  public void wholeSecondTimestamp_waitsForNextSecondPlusSlack() {
    tsgm.notifyDependenceOnFileTime(null, nanos(SECOND_MILLIS));

    assertThat(tsgm.getRemainingWait()).isEqualTo(Duration.ofMillis(1050 - 123));
    clock.advanceMillis(1050 - 123 - 1);
    assertThat(tsgm.getRemainingWait()).isEqualTo(Duration.ofMillis(1));
    clock.advanceMillis(1);
    assertThat(tsgm.getRemainingWait()).isEqualTo(Duration.ZERO);
  }

  @Test
  public void oldTimestamps_noWait() {
    tsgm.notifyDependenceOnFileTime(null, nanos(SECOND_MILLIS - 2000));
    tsgm.notifyDependenceOnFileTime(null, nanos(NOW_MILLIS - 1));

    assertThat(tsgm.getRemainingWait()).isEqualTo(Duration.ZERO);
  }

  @Test
  public void timestampFromClockAheadOfOurs_ignored() {
    tsgm.notifyDependenceOnFileTime(null, nanos(SECOND_MILLIS + 1000));
    tsgm.notifyDependenceOnFileTime(null, nanos(NOW_MILLIS + 51));

    assertThat(tsgm.getRemainingWait()).isEqualTo(Duration.ZERO);
  }

  @Test
  public void timestampSlightlyAheadOfOurs_waitsForIt() {
    tsgm.notifyDependenceOnFileTime(null, nanos(NOW_MILLIS + 10));

    assertThat(tsgm.getRemainingWait()).isEqualTo(Duration.ofMillis(11));
  }

  @Test
  public void multipleTimestamps_waitsForLatest() {
    tsgm.notifyDependenceOnFileTime(null, nanos(NOW_MILLIS));
    tsgm.notifyDependenceOnFileTime(null, nanos(SECOND_MILLIS));
    tsgm.notifyDependenceOnFileTime(null, nanos(NOW_MILLIS - 5));

    assertThat(tsgm.getRemainingWait()).isEqualTo(Duration.ofMillis(1050 - 123));
  }

  @Test
  public void waitForTimestampGranularity_waitsUntilClockAdvances() {
    TimestampGranularityMonitor tsgm = new TimestampGranularityMonitor(BlazeClock.instance());
    long ctimeMillis = BlazeClock.instance().currentTimeMillis();
    tsgm.notifyDependenceOnFileTime(null, nanos(ctimeMillis));

    tsgm.waitForTimestampGranularity(OutErr.SYSTEM_OUT_ERR);

    assertThat(BlazeClock.instance().currentTimeMillis()).isGreaterThan(ctimeMillis);
    assertThat(tsgm.getRemainingWait()).isEqualTo(Duration.ZERO);
  }
}
