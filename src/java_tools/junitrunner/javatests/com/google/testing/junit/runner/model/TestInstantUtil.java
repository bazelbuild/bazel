// Copyright 2015 The Bazel Authors. All Rights Reserved.
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
import java.time.Duration;
import java.time.Instant;

/** Utility class used for quick creation of TestInstant for testing. */
public class TestInstantUtil {

  // Added to the monotonic nano timestamp to assert it is not used as absolute time
  private static final long INITIAL_RELATIVE_TIMESTAMP = 111111L;

  /** Creates a TestInstant with a monotonic timestamp that is offset from the wall time. */
  public static TestInstant testInstant(Instant wallTime) {
    return new TestInstant(
        wallTime, Duration.ofMillis(INITIAL_RELATIVE_TIMESTAMP + wallTime.toEpochMilli()));
  }

  /** Returns a TestInstant advanced in time by the specified duration. */
  public static TestInstant advance(TestInstant instant, Duration duration) {
    return new TestInstant(
        instant.wallTime().plus(duration), instant.monotonicTime().plus(duration));
  }
}
