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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.logging.Logger;

/**
 * A utility class for dealing with filesystem timestamp granularity issues.
 *
 * <p>
 * Consider a sequence of commands such as
 * <pre>
 *     echo ... &gt; foo/bar
 *     blaze query ...
 *     echo ... &gt; foo/bar
 *     blaze query ...
 * </pre>
 *
 * If these commands all run very fast, it is possible that the timestamp
 * on foo/bar is not changed by the second command, even though some time has
 * passed, because the times are the same when rounded to the file system
 * timestamp granularity (often 1 second).
 * For performance, we assume that files whose
 * timestamps haven't changed can safely be cached without reexamining their contents.
 * But this assumption would be violated in the above scenario.
 *
 * <p>
 * To address this, we record the current time at the start of executing
 * a Blaze command, and whenever we check the timestamp of a source file
 * or BUILD file, we check if the timestamp of that source file matches
 * the current time.  If so, we set a flag.  At the end of the command,
 * if the flag was set, then we wait until the clock has advanced, so
 * that any file modifications performed after the command exits will
 * result in a different file timestamp.
 *
 * <p>
 * This class implicitly assumes that each filesystem's clock
 * is the same as either System.currentTimeMillis() or
 * System.currentTimeMillis() rounded down to the nearest second.
 * That is not strictly correct; there might be clock skew between
 * the cpu clock and the file system clocks (e.g. for NFS file systems),
 * and some file systems might have different granularity (e.g. the old
 * DOS FAT filesystem has TWO-second granularity timestamps).
 * Clock skew can be addressed using NTP.
 * Other granularities could be addressed by small changes to this class,
 * if it turns out to be needed.
 *
 * <p>
 * Another alternative design that we considered was to write to a file and
 * read its timestamp.  But doing that is a little tricky because we don't have
 * a FileSystem or Path handy.  Also, if we were going to do this, the stamp
 * file that is used should be in the same file system as the input files.
 * But the input file system(s) might not be writeable, and even if it is,
 * it's hard for Blaze to find a writable file on the same filesystem as the
 * input files.
 */
@ThreadCompatible
public class TimestampGranularityMonitor {
  private static final Logger logger =
      Logger.getLogger(TimestampGranularityMonitor.class.getName());

  /**
   * The time of the start of the current Blaze command,
   * in milliseconds.
   */
  private long commandStartTimeMillis;

  /**
   * The time of the start of the current Blaze command,
   * in milliseconds, rounded to one second granularity.
   */
  private long commandStartTimeMillisRounded;

  /**
   * True iff we detected a source file or BUILD file whose (unrounded)
   * timestamp matched the time at the start of the current Blaze command
   * rounded to the nearest second.
   */
  private volatile boolean waitASecond;

  /**
   * True iff we detected a source file or BUILD file whose timestamp
   * exactly matched the time at the start of the current Blaze command
   * (measuring both in integral numbers of milliseconds).
   */
  private volatile boolean waitAMillisecond;

  private final Clock clock;

  public TimestampGranularityMonitor(Clock clock) {
    this.clock = clock;
  }

  /**
   * Record the time at which the Blaze command started.
   * This is needed for use by waitForTimestampGranularity().
   */
  public void setCommandStartTime() {
    this.commandStartTimeMillis = clock.currentTimeMillis();
    this.commandStartTimeMillisRounded = roundDown(this.commandStartTimeMillis);
    this.waitASecond = false;
    this.waitAMillisecond = false;
  }

  /**
   * Record that the output of this Blaze command depended on the contents
   * of a build file or source file with the specified time stamp.
   */
  @ThreadSafe
  public void notifyDependenceOnFileTime(PathFragment path, long mtime) {
    if (mtime == this.commandStartTimeMillis) {
      logger.info("Will have to wait for a millisecond on completion because of " + path);
      this.waitAMillisecond = true;
    }
    if (mtime == this.commandStartTimeMillisRounded) {
      logger.info("Will have to wait for a second on completion because of " + path);
      this.waitASecond = true;
    }
  }

  /**
   * If needed, wait until the next "tick" of the filesystem timestamp clock.
   * This is done to ensure that files created after the current Blaze command
   * finishes will have timestamps different than files created before the
   * current Blaze command started.  Otherwise a sequence of commands
   * such as
   * <pre>
   *     echo ... &gt; foo/BUILD
   *     blaze query ...
   *     echo ... &gt; foo/BUILD
   *     blaze query ...
   * </pre>
   * could return wrong results, due to the contents of package foo
   * being cached even though foo/BUILD changed.
   */
  public void waitForTimestampGranularity(OutErr outErr) {
    if (this.waitASecond || this.waitAMillisecond) {
      long before = clock.currentTimeMillis();
      long startedWaiting = Profiler.nanoTimeMaybe();
      boolean interrupted = false;

      if (waitASecond) {
        // 50ms slack after the whole-second boundary
        while (clock.currentTimeMillis() < commandStartTimeMillisRounded + 1050) {
          try {
            Thread.sleep(50 /* milliseconds */);
          } catch (InterruptedException e) {
            if (!interrupted) {
              outErr.printErrLn("INFO: Hang on a second...");
              interrupted = true;
            }
          }
        }
      } else {
        while (clock.currentTimeMillis() == commandStartTimeMillis) {
          try {
            Thread.sleep(1 /* milliseconds */);
          } catch (InterruptedException e) {
            if (!interrupted) {
              outErr.printErrLn("INFO: Hang on a millisecond...");
              interrupted = true;
            }
          }
        }
      }
      if (interrupted) {
        Thread.currentThread().interrupt();
      }

      Profiler.instance().logSimpleTask(startedWaiting, ProfilerTask.WAIT,
                                        "Timestamp granularity");
      logger.info(
          "Waited for "
              + (clock.currentTimeMillis() - before)
              + "ms for file system"
              + " to catch up");
    }
  }

  /**
   * Rounds the specified time, in milliseconds, down to the nearest second,
   * and returns the result in milliseconds.
   */
  private static long roundDown(long millis) {
    return millis / 1000 * 1000;
  }

}
