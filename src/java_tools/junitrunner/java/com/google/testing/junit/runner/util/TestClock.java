// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

import java.time.Duration;
import java.time.Instant;

/**
 * A time source used to obtain:
 * <li>a monotonic timestamp with no relation to a wall time;
 * <li>a timestamp that can be used to obtain wall time but is not guaranteed to be monotonic.
 */
public abstract class TestClock {
  /** Constructor for use by subclasses. */
  protected TestClock() {}

  /**
   * Returns an immutable value type that contains both a monotonic timestamp (used to measure
   * relative time but unrelated to wall time) and an EPOCH relative timestamp.
   */
  public TestInstant now() {
    return new TestInstant(wallTime(), monotonicTime());
  }

  /**
   * Returns a monotonic timestamp that can only be used to compute relative time.
   *
   * <p><b>Warning:</b> the returned timestamp can only be used to measure elapsed time, not wall
   * time.
   */
  abstract Duration monotonicTime();

  /**
   * A timestamp that may be used to obtain wall time, but is not guaranteed to be monotonic.
   *
   * <p><b>Warning:</b> the returned timestamp is not guaranteed to be monotonic, and it may appear
   * to go back in time in certain cases (e.g. daylight saving time).
   */
  abstract Instant wallTime();

  /**
   * A time source that produces an epoch timestamp using {@link System#currentTimeMillis} and a
   * monotonic timestamp using {@link System#nanoTime}.
   */
  public static TestClock systemClock() {
    return SYSTEM_TEST_CLOCK;
  }

  private static final TestClock SYSTEM_TEST_CLOCK =
      new TestClock() {
        @Override
        public Duration monotonicTime() {
          return Duration.ofNanos(System.nanoTime());
        }

        @Override
        public Instant wallTime() {
          return Instant.ofEpochMilli(System.currentTimeMillis());
        }
      };

  /**
   * An immutable value type that contains both a monotonic timestamp (used to measure relative time
   * but unrelated to wall time) and an EPOCH timestamp.
   */
  public static class TestInstant {
    public static final TestInstant UNKNOWN = new TestInstant(Instant.EPOCH, Duration.ZERO);

    private final Instant wallTime;
    private final Duration monotonicTime;

    public TestInstant(Instant wallTime, Duration monotonicTime) {
      this.wallTime = wallTime;
      this.monotonicTime = monotonicTime;
    }

    /**
     * A timestamp that may be used to obtain wall time, but is not guaranteed to be monotonic.
     *
     * <p><b>Warning:</b> the returned timestamp is not guaranteed to be monotonic, and it may
     * appear to go back in time in certain cases (e.g. daylight saving time).
     */
    public Instant wallTime() {
      return wallTime;
    }

    /**
     * Returns a monotonic timestamp that can only be used to compute relative time.
     *
     * <p><b>Warning:</b> the returned timestamp can only be used to measure elapsed time, not wall
     * time.
     */
    public Duration monotonicTime() {
      return monotonicTime;
    }
  }
}
