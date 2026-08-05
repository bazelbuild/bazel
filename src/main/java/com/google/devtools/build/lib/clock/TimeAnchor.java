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

package com.google.devtools.build.lib.clock;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.TimeUnit;

/**
 * A fixed correspondence between a reading of {@link Clock#nanoTime} and a reading of {@link
 * Clock#currentTimeMillis}, used to place monotonic clock readings onto the wall-clock timeline.
 *
 * <p>{@link Clock#nanoTime} and {@link Clock#currentTimeMillis} are backed by different operating
 * system clocks whose offset is not constant: the wall clock is stepped by NTP and, on a virtual
 * machine, by the hypervisor after boot or after a snapshot is resumed, while the monotonic clock
 * never steps. Code that needs both a duration and an absolute timestamp must therefore not read
 * the two clocks independently; it must measure durations on {@link Clock#nanoTime} and map them
 * onto the wall clock through a single anchor.
 *
 * <p>Exactly one anchor is created per command, at the earliest point at which the server sees the
 * command, and is available from {@code CommandEnvironment#getTimeAnchor}. Using one anchor for the
 * whole command guarantees that all timestamps it produces are mutually consistent and that no
 * duration derived from them is distorted by a wall-clock step. Do not create ad-hoc anchors in
 * order to convert timestamps that were recorded much earlier: the resulting values would be
 * shifted by whatever drift the wall clock accumulated in between.
 *
 * @param clock the clock this anchor was taken from
 * @param epochMillis the wall-clock time at which this anchor was taken, in millis since the epoch
 * @param nanos the {@link Clock#nanoTime} reading at which this anchor was taken
 */
@Immutable
public record TimeAnchor(Clock clock, long epochMillis, long nanos) {

  /**
   * Prefer {@link #create}. An anchor is only meaningful if both readings were taken at the same
   * moment, which this constructor cannot verify.
   */
  @VisibleForTesting
  public TimeAnchor {}

  /** Creates an anchor from a reading of the given clock taken now. */
  public static TimeAnchor create(Clock clock) {
    long epochMillis = clock.currentTimeMillis();
    long nanos = clock.nanoTime();
    return new TimeAnchor(clock, epochMillis, nanos);
  }

  /** Creates an anchor from a reading of {@link BlazeClock#instance} taken now. */
  public static TimeAnchor create() {
    return create(BlazeClock.instance());
  }

  /**
   * Returns the current reading of the clock this anchor was taken from.
   *
   * <p>Prefer this over reading a clock directly wherever the reading will later be converted, so
   * that the reading and the anchor cannot come from two different clocks.
   */
  public long nanoTime() {
    return clock.nanoTime();
  }

  /**
   * Returns the current wall-clock time, derived from this anchor.
   *
   * <p>This is deliberately not a direct read of the wall clock: the result stays consistent with
   * the timestamps this anchor produces for other readings even if the wall clock is stepped while
   * the command runs.
   */
  public Instant now() {
    return toInstant(nanoTime());
  }

  /** Converts a {@link Clock#nanoTime} reading to millis since the epoch. */
  public long toEpochMillis(long timeNanos) {
    return epochMillis + TimeUnit.NANOSECONDS.toMillis(timeNanos - nanos);
  }

  /** Converts a {@link Clock#nanoTime} reading to an {@link Instant}. */
  public Instant toInstant(long timeNanos) {
    return Instant.ofEpochMilli(toEpochMillis(timeNanos));
  }

  /** Converts millis since the epoch to a {@link Clock#nanoTime} reading. */
  public long toNanos(long timeMillis) {
    return nanos + TimeUnit.MILLISECONDS.toNanos(timeMillis - epochMillis);
  }

  /** Returns the {@link Clock#nanoTime} reading lying the given duration before this anchor. */
  public long nanosBefore(Duration duration) {
    return nanos - duration.toNanos();
  }

  /** Returns the time between two {@link Clock#nanoTime} readings. */
  public static Duration between(long fromNanos, long toNanos) {
    return Duration.ofNanos(toNanos - fromNanos);
  }

  /** Returns the number of seconds between two {@link Clock#nanoTime} readings. */
  public static double secondsBetween(long fromNanos, long toNanos) {
    return (toNanos - fromNanos) / 1_000_000_000.0;
  }
}
