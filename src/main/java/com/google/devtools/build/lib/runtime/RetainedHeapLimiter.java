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

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.bugreport.Crash;
import com.google.devtools.build.lib.bugreport.CrashContext;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.MemoryOptions;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Monitors the size of the retained heap and exit promptly if it grows too large.
 *
 * <p>Specifically, checks the size of the tenured space after each major GC; if it exceeds {@link
 * #occupiedHeapPercentageThreshold}%, call {@link System#gc()} to trigger a stop-the-world
 * collection; if it's still more than {@link #occupiedHeapPercentageThreshold}% full, exit with an
 * {@link OutOfMemoryError}.
 */
final class RetainedHeapLimiter {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static final long MIN_TIME_BETWEEN_TRIGGERED_GC_MILLISECONDS = 60000;

  private final AtomicBoolean throwingOom = new AtomicBoolean(false);
  private final AtomicBoolean heapLimiterTriggeredGc = new AtomicBoolean(false);
  private volatile int occupiedHeapPercentageThreshold = 100;
  private final AtomicLong lastTriggeredGcInMilliseconds = new AtomicLong();
  private final BugReporter bugReporter;

  static RetainedHeapLimiter create(BugReporter bugReporter) {
    return new RetainedHeapLimiter(bugReporter);
  }

  private RetainedHeapLimiter(BugReporter bugReporter) {
    this.bugReporter = checkNotNull(bugReporter);
  }

  @ThreadSafety.ThreadCompatible // Can only be called on the logical main Bazel thread.
  void setThreshold(boolean listening, int oomMoreEagerlyThreshold) throws AbruptExitException {
    if (oomMoreEagerlyThreshold < 0 || oomMoreEagerlyThreshold > 100) {
      throw createExitException(
          "--experimental_oom_more_eagerly_threshold must be a percent between 0 and 100 but was "
              + oomMoreEagerlyThreshold,
          MemoryOptions.Code.EXPERIMENTAL_OOM_MORE_EAGERLY_THRESHOLD_INVALID_VALUE);
    }
    if (!listening && oomMoreEagerlyThreshold != 100) {
      throw createExitException(
          "No tenured GC collectors were found: unable to watch for GC events to exit JVM when "
              + oomMoreEagerlyThreshold
              + "% of heap is used",
          MemoryOptions.Code.EXPERIMENTAL_OOM_MORE_EAGERLY_NO_TENURED_COLLECTORS_FOUND);
    }
    this.occupiedHeapPercentageThreshold = oomMoreEagerlyThreshold;
  }

  private static AbruptExitException createExitException(String message, MemoryOptions.Code code) {
    return new AbruptExitException(
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage(message)
                .setMemoryOptions(MemoryOptions.newBuilder().setCode(code))
                .build()));
  }

  // Can be called concurrently, handles concurrent calls with #setThreshold gracefully.
  @ThreadSafety.ThreadSafe
  public void handle(MemoryPressureEvent event) {
    if (event.wasManualGc() && !heapLimiterTriggeredGc.getAndSet(false)) {
      // This was a manually triggered GC, but not from us earlier: short-circuit.
      return;
    }

    // Get a local reference to guard against concurrent modifications.
    int threshold = this.occupiedHeapPercentageThreshold;

    int actual = (int) ((event.tenuredSpaceUsedBytes() * 100L) / event.tenuredSpaceMaxBytes());
    if (actual < threshold) {
      return;
    }

    if (event.wasManualGc()) {
      if (!throwingOom.getAndSet(true)) {
        // We got here from a GC initiated by the other branch.
        OutOfMemoryError oom =
            new OutOfMemoryError(
                String.format(
                    "RetainedHeapLimiter forcing exit due to GC thrashing: After back-to-back full"
                        + " GCs, the tenured space is more than %s%% occupied (%s out of a tenured"
                        + " space size of %s).",
                    threshold, event.tenuredSpaceUsedBytes(), event.tenuredSpaceMaxBytes()));
        // Exits the runtime.
        bugReporter.handleCrash(Crash.from(oom), CrashContext.halt());
      }
    } else if (System.currentTimeMillis() - lastTriggeredGcInMilliseconds.get()
        > MIN_TIME_BETWEEN_TRIGGERED_GC_MILLISECONDS) {
      logger.atInfo().log(
          "Triggering a full GC with %s tenured space used out of a tenured space size of %s",
          event.tenuredSpaceUsedBytes(), event.tenuredSpaceMaxBytes());
      heapLimiterTriggeredGc.set(true);
      // Force a full stop-the-world GC and see if it can get us below the threshold.
      System.gc();
      lastTriggeredGcInMilliseconds.set(System.currentTimeMillis());
    }
  }
}
