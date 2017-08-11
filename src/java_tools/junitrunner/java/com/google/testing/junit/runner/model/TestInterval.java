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

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.TimeZone;

/**
 * Implementation of an immutable time interval, representing a period of time between two instants.
 *
 * <p>This class is thread-safe and immutable.
 */
public final class TestInterval {
  private final long startInstant;
  private final long endInstant;

  public TestInterval(long startInstant, long endInstant) {
    if (startInstant > endInstant) {
      throw new IllegalArgumentException("Start must be before end");
    }
    this.startInstant = startInstant;
    this.endInstant = endInstant;
  }

  public long getStartMillis() {
    return startInstant;
  }

  public long getEndMillis() {
    return endInstant;
  }

  public long toDurationMillis() {
    return endInstant - startInstant;
  }

  public TestInterval withEndMillis(long millis) {
    return new TestInterval(startInstant, millis);
  }

  public String startInstantToString() {
    // Format as ISO8601 string
    return startInstantToString(TimeZone.getDefault());
  }

  /** Exposed for testing because java Date does not allow setting of timezones. */
  // VisibleForTesting
  String startInstantToString(TimeZone tz) {
    DateFormat format = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSXXX");
    format.setTimeZone(tz);
    return format.format(new Date(startInstant));
  }
}
