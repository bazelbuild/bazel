/*
 * Copyright 2019 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.runtime;

import android.os.SystemClock;
import j$.time.ZoneId;

/**
 * Conversion from JDK-type-compatible {@link java.time.Clock} returned from Android APIs to core
 * library desugared {@link j$.time.Clock}.
 */
@SuppressWarnings("AndroidApiChecker") // Non-API code: Only compiled bytecode for type conversion.
public final class ClockWrapper extends j$.time.Clock {

  private final java.time.Clock clock;
  private final j$.time.ZoneId overridingZoneId;

  private ClockWrapper(java.time.Clock clock) {
    this(clock, null);
  }

  private ClockWrapper(java.time.Clock clock, j$.time.ZoneId overridingZoneId) {
    this.clock = clock;
    this.overridingZoneId = overridingZoneId;
  }

  @Override
  public j$.time.ZoneId getZone() {
    return overridingZoneId == null ? fromZoneId(clock.getZone()) : overridingZoneId;
  }

  @Override
  public j$.time.Clock withZone(j$.time.ZoneId zoneId) {
    return new ClockWrapper(clock, zoneId);
  }

  @Override
  public long millis() {
    return clock.millis();
  }

  @Override
  public j$.time.Instant instant() {
    return fromInstant(clock.instant());
  }

  public static j$.time.Clock currentGnssTimeClock() {
    return ClockWrapper.fromClock(SystemClock.currentGnssTimeClock());
  }

  static j$.time.Clock fromClock(java.time.Clock clock) {
    return new ClockWrapper(clock);
  }

  static ZoneId fromZoneId(java.time.ZoneId zoneId) {
    return ZoneId.of(zoneId.getId());
  }

  static j$.time.Instant fromInstant(java.time.Instant instant) {
    return j$.time.Instant.ofEpochSecond(instant.getEpochSecond(), instant.getNano());
  }
}
