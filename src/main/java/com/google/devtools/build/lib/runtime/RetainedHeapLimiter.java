// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.bugreport.Crash;
import com.google.devtools.build.lib.bugreport.CrashContext;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.runtime.MemoryPressure.MemoryPressureStats;
import com.google.devtools.common.options.Options;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Monitors the size of the retained heap and exit promptly if it grows too large.
 *
 * <p>Specifically, checks the size of the tenured space after each major GC; if it exceeds {@link
 * MemoryPressureOptions#oomMoreEagerlyThreshold}%, call {@link System#gc()} to trigger a
 * stop-the-world collection; if it's still more than {@link
 * MemoryPressureOptions#oomMoreEagerlyThreshold}% full, exit with an {@link OutOfMemoryError}.
 */
final class RetainedHeapLimiter implements MemoryPressureStatCollector {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final BugReporter bugReporter;
  private final Clock clock;

  private volatile MemoryPressureOptions options = inactiveOptions();

  private final AtomicBoolean throwingOom = new AtomicBoolean(false);
  private final AtomicBoolean heapLimiterTriggeredGc = new AtomicBoolean(false);
  private final AtomicInteger consecutiveIgnoredFullGcsOverThreshold = new AtomicInteger(0);
  private final AtomicBoolean loggedIgnoreWarningSinceLastGc = new AtomicBoolean(false);
  private final AtomicLong lastTriggeredGcMillis = new AtomicLong();
  private final AtomicInteger gcsTriggered = new AtomicInteger(0);
  private final AtomicInteger maxConsecutiveIgnoredFullGcsOverThreshold = new AtomicInteger(0);

  static RetainedHeapLimiter create(BugReporter bugReporter) {
    return new RetainedHeapLimiter(bugReporter, BlazeClock.instance());
  }

  @VisibleForTesting
  static RetainedHeapLimiter createForTest(BugReporter bugReporter, Clock clock) {
    return new RetainedHeapLimiter(bugReporter, clock);
  }

  private RetainedHeapLimiter(BugReporter bugReporter, Clock clock) {
    this.bugReporter = checkNotNull(bugReporter);
    this.clock = checkNotNull(clock);
  }

  @ThreadSafety.ThreadCompatible // Can only be called on the logical main Bazel thread.
  void setOptions(MemoryPressureOptions options) {
    if (options.gcThrashingLimitsRetainedHeapLimiterMutuallyExclusive
        && !options.gcThrashingLimits.isEmpty()) {
      this.options = inactiveOptions();
    } else {
      this.options = options;
    }
  }

  // Can be called concurrently, handles concurrent calls with #setThreshold gracefully.
  @ThreadSafety.ThreadSafe
  public void handle(MemoryPressureEvent event) {
    if (throwingOom.get()) {
      return; // Do nothing if a crash is already in progress.
    }

    boolean wasHeapLimiterTriggeredGc = false;
    boolean wasGcLockerDeferredHeapLimiterTriggeredGc = false;
    if (event.wasManualGc()) {
      wasHeapLimiterTriggeredGc = heapLimiterTriggeredGc.getAndSet(false);
      if (!wasHeapLimiterTriggeredGc) {
        // This was a manually triggered GC, but not from us earlier: short-circuit.
        logger.atInfo().log("Ignoring manual GC from other source");
        return;
      }
    } else if (event.wasGcLockerInitiatedGc() && heapLimiterTriggeredGc.getAndSet(false)) {
      // If System.gc() is called was while there are JNI thread(s) in the critical region, GCLocker
      // defers the GC until those threads exit the critical region. However, all GCLocker initiated
      // GCs are minor evacuation pauses, so we won't get the full GC we requested. Cancel the
      // timeout so we can attempt System.gc() again if we're still over the threshold. See full
      // explanation in b/263405096#comment14.
      logger.atWarning().log(
          "Observed a GCLocker initiated GC without observing a manual GC since the last call to"
              + " System.gc(), cancelling timeout to permit a retry");
      wasGcLockerDeferredHeapLimiterTriggeredGc = true;
      lastTriggeredGcMillis.set(0);
    }

    // Get a local reference to guard against concurrent modifications.
    MemoryPressureOptions options = this.options;
    int threshold = options.oomMoreEagerlyThreshold;

    if (threshold == 100) {
      return; // Inactive.
    }

    int actual = event.percentTenuredSpaceUsed();
    if (actual < threshold) {
      if (wasHeapLimiterTriggeredGc || wasGcLockerDeferredHeapLimiterTriggeredGc) {
        logger.atInfo().log("Back under threshold (%s%% of tenured space)", actual);
      }
      consecutiveIgnoredFullGcsOverThreshold.set(0);
      return;
    }

    if (wasHeapLimiterTriggeredGc) {
      if (!throwingOom.getAndSet(true)) {
        // We got here from a GC initiated by the other branch.
        OutOfMemoryError oom =
            new OutOfMemoryError(
                String.format(
                    "RetainedHeapLimiter forcing exit due to GC thrashing: After back-to-back full"
                        + " GCs, the tenured space is more than %s%% occupied (%s out of a tenured"
                        + " space size of %s).",
                    threshold, event.tenuredSpaceUsedBytes(), event.tenuredSpaceMaxBytes()));
        logger.atInfo().log("Calling handleCrash");
        // Exits the runtime.
        bugReporter.handleCrash(Crash.from(oom), CrashContext.halt());
      }
    } else if (clock.currentTimeMillis() - lastTriggeredGcMillis.get()
        > options.minTimeBetweenTriggeredGc.toMillis()) {
      logger.atInfo().log(
          "Triggering a full GC (%s%% of tenured space after %s GC)",
          actual, event.wasFullGc() ? "full" : "minor");
      heapLimiterTriggeredGc.set(true);
      gcsTriggered.incrementAndGet();
      // Force a full stop-the-world GC and see if it can get us below the threshold.
      System.gc();
      lastTriggeredGcMillis.set(clock.currentTimeMillis());
      consecutiveIgnoredFullGcsOverThreshold.set(0);
      loggedIgnoreWarningSinceLastGc.set(false);
    } else if (event.wasFullGc()) {
      int consecutiveIgnored = consecutiveIgnoredFullGcsOverThreshold.incrementAndGet();
      maxConsecutiveIgnoredFullGcsOverThreshold.accumulateAndGet(consecutiveIgnored, Math::max);
      logger.atWarning().log(
          "Ignoring possible GC thrashing x%s (%s%% of tenured space after full GC) because of"
              + " recently triggered GC",
          consecutiveIgnored, actual);
    } else if (!loggedIgnoreWarningSinceLastGc.getAndSet(true)) {
      logger.atWarning().log(
          "Ignoring possible GC thrashing (%s%% of tenured space after minor GC) because of"
              + " recently triggered GC",
          actual);
    }
  }

  @Override
  public void addStatsAndReset(MemoryPressureStats.Builder stats) {
    stats
        .setManuallyTriggeredGcs(gcsTriggered.getAndSet(0))
        .setMaxConsecutiveIgnoredGcsOverThreshold(
            maxConsecutiveIgnoredFullGcsOverThreshold.getAndSet(0));
    consecutiveIgnoredFullGcsOverThreshold.set(0);
  }

  private static MemoryPressureOptions inactiveOptions() {
    var options = Options.getDefaults(MemoryPressureOptions.class);
    options.oomMoreEagerlyThreshold = 100;
    return options;
  }
}
