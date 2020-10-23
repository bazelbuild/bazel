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

package com.google.devtools.build.lib.shell;

import com.google.common.annotations.VisibleForTesting;
import java.time.Duration;
import java.util.Optional;

/**
 * Represents the termination status of a command.  {@link Process#waitFor} is
 * not very precisely specified, so this class encapsulates the interpretation
 * of values returned by it.
 *
 * Caveat: due to the lossy encoding, it's not always possible to accurately
 * distinguish signal and exit cases.  In particular, processes that exit with
 * a value within the interval [129, 191] will be mistaken for having been
 * terminated by a signal.
 *
 * Instances are immutable.
 */
public final class TerminationStatus {

  private final int waitResult;
  private final boolean timedOut;
  private final Optional<Duration> wallExecutionTime;
  private final Optional<Duration> userExecutionTime;
  private final Optional<Duration> systemExecutionTime;

  /**
   * Values taken from the glibc strsignal(3) function.
   */
  private static final String[] SIGNAL_STRINGS = {
    null,
    "Hangup",
    "Interrupt",
    "Quit",
    "Illegal instruction",
    "Trace/breakpoint trap",
    "Aborted",
    "Bus error",
    "Floating point exception",
    "Killed",
    "User defined signal 1",
    "Segmentation fault",
    "User defined signal 2",
    "Broken pipe",
    "Alarm clock",
    "Terminated",
    "Stack fault",
    "Child exited",
    "Continued",
    "Stopped (signal)",
    "Stopped",
    "Stopped (tty input)",
    "Stopped (tty output)",
    "Urgent I/O condition",
    "CPU time limit exceeded",
    "File size limit exceeded",
    "Virtual timer expired",
    "Profiling timer expired",
    "Window changed",
    "I/O possible",
    "Power failure",
    "Bad system call",
  };

  private static String getSignalString(int signum) {
    return signum > 0 && signum < SIGNAL_STRINGS.length
        ? SIGNAL_STRINGS[signum]
        : "Signal " + signum;
  }

  /**
   * Construct a TerminationStatus instance from a Process waitFor code.
   *
   * @param waitResult the value returned by {@link java.lang.Process#waitFor}.
   * @param timedOut whether the execution timed out
   */
  public TerminationStatus(int waitResult, boolean timedOut) {
    this(waitResult, timedOut, Optional.empty(), Optional.empty(), Optional.empty());
  }

  /**
   * Construct a TerminationStatus instance from a Process waitFor code.
   *
   * <p>TerminationStatus objects are considered equal if they have the same waitResult.
   *
   * @param waitResult the value returned by {@link java.lang.Process#waitFor}.
   * @param timedOut whether the execution timed out
   * @param wallExecutionTime the wall execution time of the command, if available
   * @param userExecutionTime the user execution time of the command, if available
   * @param systemExecutionTime the system execution time of the command, if available
   */
  public TerminationStatus(
      int waitResult,
      boolean timedOut,
      Optional<Duration> wallExecutionTime,
      Optional<Duration> userExecutionTime,
      Optional<Duration> systemExecutionTime) {
    this.waitResult = waitResult;
    this.timedOut = timedOut;
    this.wallExecutionTime = wallExecutionTime;
    this.userExecutionTime = userExecutionTime;
    this.systemExecutionTime = systemExecutionTime;
  }

  /**
   * Returns the exit code returned by the subprocess.
   */
  public int getRawExitCode() {
    return waitResult;
  }

  /**
   * Returns true iff the process exited with code 0.
   */
  public boolean success() {
    return exited() && getExitCode() == 0;
  }

  // We're relying on undocumented behaviour of Process.waitFor, specifically
  // that waitResult is the exit status when the process returns normally, or
  // 128+signalnumber when the process is terminated by a signal.  We further
  // assume that value signal numbers fall in the interval [1, 63].
  @VisibleForTesting static final int SIGNAL_1 = 128 + 1;
  @VisibleForTesting static final int SIGNAL_63 = 128 + 63;
  @VisibleForTesting static final int SIGNAL_SIGABRT = 128 + 6;
  @VisibleForTesting static final int SIGNAL_SIGKILL = 128 + 9;
  @VisibleForTesting static final int SIGNAL_SIGBUS = 128 + 10;
  @VisibleForTesting static final int SIGNAL_SIGTERM = 128 + 15;

  /**
   * Returns true if the given exit code represents a crash.
   *
   * <p>This is a static function that processes a raw exit status because that's all the
   * information that we have around in the single use case of this function. Propagating a {@link
   * TerminationStatus} object to that point would be costly. If this function is needed for
   * anything else, then this should be reevaluated.
   */
  public static boolean crashed(int rawStatus) {
    return rawStatus == SIGNAL_SIGABRT || rawStatus == SIGNAL_SIGBUS;
  }

  /** Returns true iff the process exited normally. */
  public boolean exited() {
    return !timedOut && (waitResult < SIGNAL_1 || waitResult > SIGNAL_63);
  }

  /** Returns true if the process timed out. */
  public boolean timedOut() {
    return timedOut;
  }

  /**
   * Returns the exit code of the subprocess.  Undefined if exited() is false.
   */
  public int getExitCode() {
    if (!exited()) {
      throw new IllegalStateException("getExitCode() not defined");
    }
    return waitResult;
  }

  /**
   * Returns the number of the signal that terminated the process.  Undefined
   * if exited() returns true.
   */
  public int getTerminatingSignal() {
    if (exited() || timedOut) {
      throw new IllegalStateException("getTerminatingSignal() not defined");
    }
    return waitResult - SIGNAL_1 + 1;
  }

  /**
   * Returns the wall execution time.
   *
   * @return the measurement, or empty in case of execution errors or when the measurement is not
   *     implemented for the current platform
   */
  public Optional<Duration> getWallExecutionTime() {
    return wallExecutionTime;
  }

  /**
   * Returns the user execution time.
   *
   * @return the measurement, or empty in case of execution errors or when the measurement is not
   *     implemented for the current platform
   */
  public Optional<Duration> getUserExecutionTime() {
    return userExecutionTime;
  }

  /**
   * Returns the system execution time.
   *
   * @return the measurement, or empty in case of execution errors or when the measurement is not
   *     implemented for the current platform
   */
  public Optional<Duration> getSystemExecutionTime() {
    return systemExecutionTime;
  }

  /**
   * Returns a short string describing the termination status.
   * e.g. "Exit 1" or "Hangup".
   */
  public String toShortString() {
    return exited()
        ? "Exit " + getExitCode()
        : (timedOut ? "Timeout" : getSignalString(getTerminatingSignal()));
  }

  @Override
  public String toString() {
    if (exited()) {
      return "Process exited with status " + getExitCode();
    } else if (timedOut) {
      return "Timed out";
    } else {
      return "Process terminated by signal " + getTerminatingSignal();
    }
  }

  @Override
  public int hashCode() {
    return waitResult;
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof TerminationStatus
        && ((TerminationStatus) other).waitResult == this.waitResult;
  }

  /** Returns a new {@link TerminationStatus.Builder}. */
  public static Builder builder() {
    return new TerminationStatus.Builder();
  }

  /** Builder for {@link TerminationStatus} objects. */
  public static class Builder {
    // We use nullness here instead of Optional to avoid confusion between fields that may not
    // yet have been set from fields that can legitimately hold an Optional value in the built
    // object.

    private Integer waitResponse = null;
    private Boolean timedOut = null;

    private Optional<Duration> wallExecutionTime = Optional.empty();
    private Optional<Duration> userExecutionTime = Optional.empty();
    private Optional<Duration> systemExecutionTime = Optional.empty();

    /** Sets the value returned by {@link java.lang.Process#waitFor}. */
    public Builder setWaitResponse(int waitResponse) {
      this.waitResponse = waitResponse;
      return this;
    }

    /** Sets whether the action timed out or not. */
    public Builder setTimedOut(boolean timedOut) {
      this.timedOut = timedOut;
      return this;
    }

    /** Sets the wall execution time. */
    public Builder setWallExecutionTime(Duration wallExecutionTime) {
      this.wallExecutionTime = Optional.of(wallExecutionTime);
      return this;
    }

    /** Sets or clears the wall execution time. */
    public Builder setWallExecutionTime(Optional<Duration> wallExecutionTime) {
      this.wallExecutionTime = wallExecutionTime;
      return this;
    }

    /** Sets the user execution time. */
    public Builder setUserExecutionTime(Duration userExecutionTime) {
      this.userExecutionTime = Optional.of(userExecutionTime);
      return this;
    }

    /** Sets or clears the user execution time. */
    public Builder setUserExecutionTime(Optional<Duration> userExecutionTime) {
      this.userExecutionTime = userExecutionTime;
      return this;
    }

    /** Sets the system execution time. */
    public Builder setSystemExecutionTime(Duration systemExecutionTime) {
      this.systemExecutionTime = Optional.of(systemExecutionTime);
      return this;
    }

    /** Sets or clears the system execution time. */
    public Builder setSystemExecutionTime(Optional<Duration> systemExecutionTime) {
      this.systemExecutionTime = systemExecutionTime;
      return this;
    }

    /** Builds a {@link TerminationStatus} object. */
    public TerminationStatus build() {
      if (waitResponse == null) {
        throw new IllegalStateException("waitResponse never set");
      }
      if (timedOut == null) {
        throw new IllegalStateException("timedOut never set");
      }
      return new TerminationStatus(
          waitResponse, timedOut, wallExecutionTime, userExecutionTime, systemExecutionTime);
    }
  }
}
