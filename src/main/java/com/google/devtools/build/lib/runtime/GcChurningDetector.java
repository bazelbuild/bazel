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
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.bugreport.Crash;
import com.google.devtools.build.lib.bugreport.CrashContext;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.runtime.MemoryPressure.MemoryPressureStats;
import com.google.devtools.build.lib.runtime.MemoryPressure.MemoryPressureStats.FullGcFractionPoint;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.Crash.Code;
import com.google.devtools.build.lib.server.FailureDetails.Crash.OomCauseCategory;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.DetailedExitCode;
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
  private static final Duration MIN_INVOCATION_WALL_TIME_DURATION = Duration.ofMinutes(1);

  private volatile int thresholdPercentage;
  private final int thresholdPercentageIfMultipleTopLevelTargets;
  private Duration cumulativeFullGcDuration = Duration.ZERO;
  private final Clock clock;
  private final Instant start;
  private final ArrayList<FullGcFractionPoint> fullGcFractionPoints = new ArrayList<>();
  private final BugReporter bugReporter;

  @VisibleForTesting
  GcChurningDetector(
      int thresholdPercentage,
      int thresholdPercentageIfMultipleTopLevelTargets,
      Clock clock,
      BugReporter bugReporter) {
    this.thresholdPercentage = thresholdPercentage;
    this.thresholdPercentageIfMultipleTopLevelTargets =
        thresholdPercentageIfMultipleTopLevelTargets;
    this.clock = clock;
    this.start = clock.now();
    this.bugReporter = bugReporter;
  }

  static GcChurningDetector createForCommand(MemoryPressureOptions options) {
    return new GcChurningDetector(
        options.gcChurningThreshold,
        options.gcChurningThresholdIfMultipleTopLevelTargets.orElse(options.gcChurningThreshold),
        BlazeClock.instance(),
        BugReporter.defaultInstance());
  }

  void targetParsingComplete(int numTopLevelTargets) {
    if (numTopLevelTargets > 1) {
      thresholdPercentage = thresholdPercentageIfMultipleTopLevelTargets;
      logger.atInfo().log(
          "Switched to thresholdPercentage of %s because there were %s top-level targets",
          thresholdPercentage, numTopLevelTargets);
    }
  }

  // This is called from MemoryPressureListener on a single memory-pressure-listener-0 thread, so it
  // should never be called concurrently, but mark it synchronized for good measure.
  synchronized void handle(MemoryPressureEvent event) {
    if (!event.wasFullGc() || event.wasManualGc()) {
      return;
    }

    cumulativeFullGcDuration = cumulativeFullGcDuration.plus(event.duration());
    Duration invocationWallTimeDuration = Duration.between(start, clock.now());
    // This narrowing conversion is fine in practice since MAX_INT ms is almost 25 days, and
    // we don't care about supporting an invocation running for that long.
    int invocationWallTimeSoFarMs = (int) invocationWallTimeDuration.toMillis();
    if (invocationWallTimeSoFarMs == 0) {
      // Given that our data points have millisecond resolution, don't bother recording a data point
      // if it's been less than a full millisecond so far.
      return;
    }
    double gcFraction = cumulativeFullGcDuration.toMillis() * 1.0 / invocationWallTimeSoFarMs;
    fullGcFractionPoints.add(
        FullGcFractionPoint.newBuilder()
            .setInvocationWallTimeSoFarMs(invocationWallTimeSoFarMs)
            .setFullGcFractionSoFar(gcFraction)
            .build());
    logger.atInfo().log(
        "cumulativeFullGcDuration=%s invocationWallTimeDuration=%s gcFraction=%.3f",
        cumulativeFullGcDuration, invocationWallTimeDuration, gcFraction);

    double gcFractionPercentage = gcFraction * 100;
    if (gcFractionPercentage >= thresholdPercentage
        && invocationWallTimeDuration.compareTo(MIN_INVOCATION_WALL_TIME_DURATION) >= 0) {
      OutOfMemoryError oom =
          new OutOfMemoryError(
              String.format(
                  "GcChurningDetector forcing exit: %.1f%% of the invocation's wall time so far"
                      + " (%ss) has been spent doing full GCs",
                  gcFractionPercentage, invocationWallTimeDuration.toSeconds()));
      logger.atInfo().log("Calling handleCrash");
      bugReporter.handleCrash(
          Crash.from(
              oom,
              DetailedExitCode.of(
                  FailureDetail.newBuilder()
                      .setMessage(oom.getMessage())
                      .setCrash(
                          FailureDetails.Crash.newBuilder()
                              .setCode(Code.CRASH_OOM)
                              .setOomCauseCategory(OomCauseCategory.GC_CHURNING))
                      .build())),
          CrashContext.halt());
    }
  }

  void populateStats(MemoryPressureStats.Builder memoryPressureStatsBuilder) {
    memoryPressureStatsBuilder.addAllFullGcFractionPoint(fullGcFractionPoints);
  }
}
