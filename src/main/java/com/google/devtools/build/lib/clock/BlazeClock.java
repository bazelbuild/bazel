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
 * Provides the clock implementation used by Blaze, which is {@link JavaClock} by default, but can
 * be overridden at runtime. If you set this clock, you also have to set the clock used by the
 * Profiler.
 *
 * <p>Note that clock readings are relative to an unspecified reference time, so returned values are
 * only meaningful when compared to each other. A {@link NanosToMillisSinceEpochConverter} or {@link
 * MillisSinceEpochToNanosConverter} may be used to convert clock readings into milliseconds since
 * the epoch or vice-versa.
 */
@ThreadSafe
public abstract class BlazeClock {

  private BlazeClock() {}

  private static volatile Clock instance = new JavaClock();

  /** Returns singleton instance of the clock */
  public static Clock instance() {
    return instance;
  }

  /** Overrides default clock instance. */
  public static synchronized void setClock(Clock clock) {
    instance = clock;
  }

  public static long nanoTime() {
    return instance().nanoTime();
  }

  /** Converts from nanos to millis since the epoch. */
  public interface NanosToMillisSinceEpochConverter {

    /** Converts from nanos to millis since the epoch. */
    long toEpochMillis(long timeNanos);
  }

  /**
   * Creates a {@link NanosToMillisSinceEpochConverter} from the current {@link BlazeClock}
   * instance.
   */
  public static NanosToMillisSinceEpochConverter createNanosToMillisSinceEpochConverter() {
    return createNanosToMillisSinceEpochConverter(instance);
  }

  /** Creates a {@link NanosToMillisSinceEpochConverter} from the given {@link Clock}. */
  @VisibleForTesting
  public static NanosToMillisSinceEpochConverter createNanosToMillisSinceEpochConverter(
      Clock clock) {
    long nowInMillis = clock.currentTimeMillis();
    long nowInNanos = clock.nanoTime();
    return (timeNanos) -> nowInMillis - TimeUnit.NANOSECONDS.toMillis(nowInNanos - timeNanos);
  }

  /** Converts from millis since the epoch to nanos. */
  public interface MillisSinceEpochToNanosConverter {

    /** Converts from millis since the epoch to nanos. */
    long toNanos(long timeMillis);
  }

  /**
   * Creates a {@link NanosToMillisSinceEpochConverter} from the current {@link BlazeClock}
   * instance.
   */
  public static MillisSinceEpochToNanosConverter createMillisSinceEpochToNanosConverter() {
    return createMillisSinceEpochToNanosConverter(instance);
  }

  /** Creates a {@link MillisSinceEpochToNanosConverter} from the given {@link Clock}. */
  @VisibleForTesting
  public static MillisSinceEpochToNanosConverter createMillisSinceEpochToNanosConverter(
      Clock clock) {
    long nowInMillis = clock.currentTimeMillis();
    long nowInNanos = clock.nanoTime();
    return (timeMillis) -> nowInNanos - TimeUnit.MILLISECONDS.toNanos(nowInMillis - timeMillis);
  }
}
