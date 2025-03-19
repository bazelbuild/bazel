// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.runtime.MemoryPressure.MemoryPressureStats;
import com.google.devtools.build.lib.runtime.MemoryPressure.MemoryPressureStats.FullGcFractionPoint;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;

/**
 * Per-invocation handler of {@link MemoryPressureEvent} to detect GC churning.
 *
 * <p>"GC churning" is the situation when the time spent doing full GCs is a big fraction of the
 * overall invocation wall time. See {@link GcThrashingDetector} for "GC thrashing". GC churning and
 * GC thrashing can sometimes, but not necessarily, coincide. Consider a situation where Blaze does
 * many full GCs all of which are fruitful. By definition that cannot be GC thrashing, but if the
 * full GCs are numerous and long enough it could be GC churning.
 */
class GcChurningDetector {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private Duration cumulativeFullGcDuration = Duration.ZERO;
  private final Clock clock;
  private final Instant start;

  private final ArrayList<FullGcFractionPoint> fullGcFractionPoints = new ArrayList<>();

  @VisibleForTesting
  GcChurningDetector(Clock clock) {
    this.clock = clock;
    this.start = clock.now();
  }

  static GcChurningDetector createForCommand() {
    return new GcChurningDetector(BlazeClock.instance());
  }

  // This is called from MemoryPressureListener on a single memory-pressure-listener-0 thread, so it
  // should never be called concurrently, but mark it synchronized for good measure.
  synchronized void handle(MemoryPressureEvent event) {
    if (!event.wasFullGc() || event.wasManualGc()) {
      return;
    }

    cumulativeFullGcDuration = cumulativeFullGcDuration.plus(event.duration());
    Duration invocationWallTimeDuration = Duration.between(start, clock.now());
    double gcFraction =
        cumulativeFullGcDuration.toMillis() * 1.0 / invocationWallTimeDuration.toMillis();
    fullGcFractionPoints.add(
        FullGcFractionPoint.newBuilder()
            // This narrowing conversion is fine in practice since MAX_INT ms is almost 25 days, and
            // we don't care about supporting an invocation running for that long.
            .setInvocationWallTimeSoFarMs((int) invocationWallTimeDuration.toMillis())
            .setFullGcFractionSoFar(gcFraction)
            .build());
    logger.atInfo().log(
        "cumulativeFullGcDuration=%s invocationWallTimeDuration=%s gcFraction=%.3f",
        cumulativeFullGcDuration, invocationWallTimeDuration, gcFraction);

    // TODO: b/389784555 - Crash Blaze when there has been too much GC churn.
  }

  void populateStats(MemoryPressureStats.Builder memoryPressureStatsBuilder) {
    memoryPressureStatsBuilder.addAllFullGcFractionPoint(fullGcFractionPoints);
  }
}
