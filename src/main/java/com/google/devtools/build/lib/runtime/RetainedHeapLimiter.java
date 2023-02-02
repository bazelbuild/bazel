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
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.MemoryOptions;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.common.options.Options;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Monitors the size of the retained heap and exit promptly if it grows too large.
 *
 * <p>Specifically, checks the size of the tenured space after each major GC; if it exceeds {@link
 * MemoryPressureOptions#oomMoreEagerlyThreshold}%, call {@link System#gc()} to trigger a
 * stop-the-world collection; if it's still more than {@link
 * MemoryPressureOptions#oomMoreEagerlyThreshold}% full, exit with an {@link OutOfMemoryError}.
 */
final class RetainedHeapLimiter {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final BugReporter bugReporter;
  private final Clock clock;

  private volatile MemoryPressureOptions options = Options.getDefaults(MemoryPressureOptions.class);

  private final AtomicBoolean throwingOom = new AtomicBoolean(false);
  private final AtomicBoolean heapLimiterTriggeredGc = new AtomicBoolean(false);
  private final AtomicBoolean loggedIgnoreWarningSinceLastGc = new AtomicBoolean(false);
  private final AtomicLong lastTriggeredGcMillis = new AtomicLong();

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
  void setOptions(MemoryPressureOptions options) throws AbruptExitException {
    if (options.oomMoreEagerlyThreshold < 0 || options.oomMoreEagerlyThreshold > 100) {
      throw createExitException(
          "--experimental_oom_more_eagerly_threshold must be a percent between 0 and 100 but was "
              + options.oomMoreEagerlyThreshold,
          MemoryOptions.Code.EXPERIMENTAL_OOM_MORE_EAGERLY_THRESHOLD_INVALID_VALUE);
    }
    this.options = options;
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
    if (throwingOom.get()) {
      return; // Do nothing if a crash is already in progress.
    }

    if (event.wasManualGc() && !heapLimiterTriggeredGc.getAndSet(false)) {
      // This was a manually triggered GC, but not from us earlier: short-circuit.
      return;
    }

    // Get a local reference to guard against concurrent modifications.
    MemoryPressureOptions options = this.options;
    int threshold = options.oomMoreEagerlyThreshold;

    if (threshold == 100) {
      return; // Inactive.
    }

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
        logger.atInfo().log("Calling handleCrash");
        // Exits the runtime.
        bugReporter.handleCrash(Crash.from(oom), CrashContext.halt());
      }
    } else if (clock.currentTimeMillis() - lastTriggeredGcMillis.get()
        > options.minTimeBetweenTriggeredGc.toMillis()) {
      logger.atInfo().log(
          "Triggering a full GC with %s tenured space used out of a tenured space size of %s",
          event.tenuredSpaceUsedBytes(), event.tenuredSpaceMaxBytes());
      heapLimiterTriggeredGc.set(true);
      // Force a full stop-the-world GC and see if it can get us below the threshold.
      System.gc();
      lastTriggeredGcMillis.set(clock.currentTimeMillis());
      loggedIgnoreWarningSinceLastGc.set(false);
    } else if (!loggedIgnoreWarningSinceLastGc.getAndSet(true)) {
      logger.atWarning().log(
          "Ignoring possible GC thrashing (%s%% of tenured space) because of recently triggered GC",
          actual);
    }
  }
}
