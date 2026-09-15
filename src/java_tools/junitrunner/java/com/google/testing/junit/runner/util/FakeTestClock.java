// Copyright 2010 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.util;

import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.time.Duration;
import java.time.Instant;

/**
 * A Ticker whose value can be advanced programmatically in test.
 *
 * <p>The ticker can be configured so that the time is incremented whenever {@link #now()} is
 * called.
 *
 * <p>This class is thread-safe.
 */
public class FakeTestClock extends TestClock {

  private Instant wallTimeOffset = Instant.EPOCH;
  private Duration monotonic = Duration.ZERO;
  private Duration autoIncrementStep = Duration.ZERO;

  /** Advances the ticker value by {@code time} in {@code timeUnit}. */
  @CanIgnoreReturnValue
  public synchronized FakeTestClock advance(Duration duration) {
    monotonic = monotonic.plus(duration);
    return this;
  }

  /**
   * Sets the wall time offset to the specified value. That is the offset between the wall time and
   * the monotonic advance set either via {@link #setAutoIncrementStep(Duration)} or {@link
   * #advance(Duration)}.
   *
   * <p>The default behavior is to have an offset of zero, which means that the monotonic timestamp
   * has the same value as the wall time (relative to EPOCH).
   */
  public void setWallTimeOffset(Instant wallTimeOffset) {
    this.wallTimeOffset = wallTimeOffset;
  }

  @Override
  Duration monotonicTime() {
    return monotonic;
  }

  @Override
  Instant wallTime() {
    return wallTimeOffset.plus(monotonic);
  }

  @Override
  public synchronized TestInstant now() {
    advance(autoIncrementStep);
    return super.now();
  }
}

