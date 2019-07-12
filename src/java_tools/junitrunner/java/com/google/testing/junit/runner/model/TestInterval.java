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

package com.google.testing.junit.runner.model;

import com.google.testing.junit.runner.util.TestClock.TestInstant;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;

/**
 * Implementation of an immutable time interval, representing a period of time between two instants.
 *
 * <p>This class is thread-safe and immutable.
 */
public final class TestInterval {

  private static final DateTimeFormatter ISO8601_WITH_MILLIS_FORMATTER =
      new DateTimeFormatterBuilder().appendInstant(3).toFormatter();
  private final TestInstant startInstant;
  private final TestInstant endInstant;

  public TestInterval(TestInstant startInstant, TestInstant endInstant) {
    if (startInstant.monotonicTime().compareTo(endInstant.monotonicTime()) > 0) {
      throw new IllegalArgumentException("Start must be before end");
    }
    this.startInstant = startInstant;
    this.endInstant = endInstant;
  }

  public long getStartMillis() {
    return startInstant.wallTime().toEpochMilli();
  }

  public long getEndMillis() {
    return endInstant.wallTime().toEpochMilli();
  }

  public long toDurationMillis() {
    return endInstant.monotonicTime().minus(startInstant.monotonicTime()).toMillis();
  }

  public TestInterval withEndMillis(TestInstant now) {
    return new TestInterval(startInstant, now);
  }

  public String startInstantToString() {
    // Format as ISO8601 with 3 fractional digits on seconds
    // This format is not affected by timezones and locale which improves interoperability
    return ISO8601_WITH_MILLIS_FORMATTER.format(startInstant.wallTime());
  }

  /** Returns a TestInterval that contains both TestIntervals passed as parameter. */
  public static TestInterval around(TestInterval a, TestInterval b) {
    TestInstant start =
        a.startInstant.monotonicTime().compareTo(b.startInstant.monotonicTime()) < 0
            ? a.startInstant
            : b.startInstant;
    TestInstant end =
        a.endInstant.monotonicTime().compareTo(b.endInstant.monotonicTime()) > 0
            ? a.endInstant
            : b.endInstant;
    return new TestInterval(start, end);
  }
}
