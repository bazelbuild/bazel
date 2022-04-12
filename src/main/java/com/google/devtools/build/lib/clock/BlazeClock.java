// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.clock;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.util.concurrent.TimeUnit;

/**
 * Provides the clock implementation used by Blaze, which is {@link JavaClock}
 * by default, but can be overridden at runtime. Note that if you set this
 * clock, you also have to set the clock used by the Profiler.
 */
@ThreadSafe
public abstract class BlazeClock {

  private BlazeClock() {
  }

  private static volatile Clock instance = new JavaClock();

  /**
   * Returns singleton instance of the clock
   */
  public static Clock instance() {
    return instance;
  }

  /**
   * Overrides default clock instance.
   */
  public static synchronized void setClock(Clock clock) {
    instance = clock;
  }

  public static long nanoTime() {
    return instance().nanoTime();
  }

  /**
   * Converts from nanos to millis since the epoch. In particular, note that {@link System#nanoTime}
   * does not specify any particular time reference but only notes that returned values are only
   * meaningful when taking in relation to each other.
   */
  public interface NanosToMillisSinceEpochConverter {
    /** Converts from nanos to millis since the epoch. */
    long toEpochMillis(long timeNanos);
  }

  /**
   * Creates a {@link NanosToMillisSinceEpochConverter} from the current BlazeClock instance by
   * taking the current time in millis and the current time in nanos to compute the appropriate
   * offset.
   */
  public static NanosToMillisSinceEpochConverter createNanosToMillisSinceEpochConverter() {
    return createNanosToMillisSinceEpochConverter(instance);
  }

  /**
   * Creates a {@link NanosToMillisSinceEpochConverter} from clock by taking the current time in
   * millis and the current time in nanos to compute the appropriate offset.
   */
  @VisibleForTesting
  public static NanosToMillisSinceEpochConverter createNanosToMillisSinceEpochConverter(
      Clock clock) {
    long nowInMillis = clock.currentTimeMillis();
    long nowInNanos = clock.nanoTime();
    return (timeNanos) -> nowInMillis - TimeUnit.NANOSECONDS.toMillis((nowInNanos - timeNanos));
  }
}
