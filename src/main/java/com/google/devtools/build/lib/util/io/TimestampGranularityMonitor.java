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

package com.google.devtools.build.lib.util.io;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/**
 * A utility class for dealing with filesystem timestamp granularity issues.
 *
 * <p>Consider a sequence of commands such as
 *
 * <pre>
 *     echo ... &gt; foo/bar
 *     blaze query ...
 *     echo ... &gt; foo/bar
 *     blaze query ...
 * </pre>
 *
 * For performance, we assume that files whose size and timestamps haven't changed can safely be
 * cached without reexamining their contents. If the file system only records timestamps in whole
 * seconds and these commands all run within the same second, foo/bar has the same timestamps after
 * the second write as after the first, which violates this assumption.
 *
 * <p>To address this, whenever we check the timestamp of a file whose contents we don't track
 * otherwise, we record until when a change to the file could still be given the same timestamp. At
 * the end of the command, we wait until that time has passed, so that any file modification
 * performed after the command exits results in a different file timestamp.
 *
 * <p>File systems don't report the granularity of their timestamps, so we infer it from the change
 * time itself: a change time that is a whole number of seconds or milliseconds is assumed to come
 * from a file system with that granularity. Finer timestamps don't require waiting: either the file
 * system gives a change made after a stat a later timestamp (e.g. APFS, or Linux 6.13+ with
 * multigrain timestamps), or it takes timestamps from a clock that advances in ticks of a few
 * milliseconds, which is less than the time between two changes made before and after a command.
 *
 * <p>This class assumes that the file system's clock agrees with {@link Clock#currentTimeMillis()}
 * up to a small slack, which may not hold for network file systems. Waiting can't make up for a
 * file system clock that is ahead of ours, so such timestamps are ignored.
 */
@ThreadCompatible
public class TimestampGranularityMonitor {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final long NANOS_PER_MILLI = 1_000_000;
  private static final long NANOS_PER_SECOND = 1_000_000_000;

  /**
   * How far the file system's clock may deviate from ours, e.g. because it only advances once per
   * scheduler tick.
   */
  private static final long SLACK_NANOS = 50 * NANOS_PER_MILLI;

  /** The time until which to wait at the end of the command and the file that requires it. */
  private record Deadline(long nanos, @Nullable PathFragment path) {}

  private final AtomicReference<Deadline> deadline =
      new AtomicReference<>(new Deadline(Long.MIN_VALUE, null));

  private final Clock clock;

  public TimestampGranularityMonitor(Clock clock) {
    this.clock = clock;
  }

  /**
   * Record that the output of this Blaze command depended on the contents of a build file or source
   * file with the specified change time.
   */
  @ThreadSafe
  public void notifyDependenceOnFileTime(@Nullable PathFragment path, long ctimeNanos) {
    long granularityNanos;
    if (ctimeNanos % NANOS_PER_SECOND == 0) {
      granularityNanos = NANOS_PER_SECOND + SLACK_NANOS;
    } else if (ctimeNanos % NANOS_PER_MILLI == 0) {
      granularityNanos = NANOS_PER_MILLI;
    } else {
      return;
    }
    long deadlineNanos = ctimeNanos + granularityNanos;
    if (deadlineNanos <= deadline.get().nanos()) {
      return;
    }
    if (ctimeNanos > currentTimeNanos() + SLACK_NANOS) {
      // The file system's clock is ahead of ours, so waiting on ours wouldn't help.
      return;
    }
    deadline.accumulateAndGet(
        new Deadline(deadlineNanos, path), (a, b) -> a.nanos() >= b.nanos() ? a : b);
  }

  /**
   * If needed, wait until the next "tick" of the filesystem timestamp clock. This is done to ensure
   * that files changed after the current Blaze command finishes will have timestamps different than
   * the ones it observed. Otherwise a sequence of commands such as
   *
   * <pre>
   *     echo ... &gt; foo/BUILD
   *     blaze query ...
   *     echo ... &gt; foo/BUILD
   *     blaze query ...
   * </pre>
   *
   * could return wrong results, due to the contents of package foo being cached even though
   * foo/BUILD changed.
   */
  public void waitForTimestampGranularity(OutErr outErr) {
    Duration remainingWait = getRemainingWait();
    if (remainingWait.isZero()) {
      return;
    }
    Deadline deadline = this.deadline.get();
    if (deadline.path() != null) {
      logger.atInfo().log(
          "Waiting for %dms on completion because of %s",
          remainingWait.toMillis(),
          deadline.path());
    }
    long before = clock.currentTimeMillis();
    long startedWaiting = Profiler.instance().nanoTimeMaybe();
    boolean interrupted = false;

    while (!remainingWait.isZero()) {
      try {
        Thread.sleep(remainingWait);
      } catch (InterruptedException e) {
        if (!interrupted) {
          outErr.printErrLn("INFO: Hang on a second...");
          interrupted = true;
        }
      }
      remainingWait = getRemainingWait();
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }

    Profiler.instance().logSimpleTask(startedWaiting, ProfilerTask.WAIT, "Timestamp granularity");
    logger.atInfo().log(
        "Waited for %dms for file system to catch up", clock.currentTimeMillis() - before);
  }

  /** Returns how much longer {@link #waitForTimestampGranularity} has to wait. */
  @VisibleForTesting
  Duration getRemainingWait() {
    long deadlineNanos = deadline.get().nanos();
    long nowNanos = currentTimeNanos();
    return deadlineNanos > nowNanos ? Duration.ofNanos(deadlineNanos - nowNanos) : Duration.ZERO;
  }

  private long currentTimeNanos() {
    return clock.currentTimeMillis() * NANOS_PER_MILLI;
  }
}
