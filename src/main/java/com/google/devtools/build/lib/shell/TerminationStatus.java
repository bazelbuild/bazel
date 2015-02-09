// Copyright 2014 Google Inc. All rights reserved.
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
   */
  public TerminationStatus(int waitResult) {
    this.waitResult = waitResult;
  }

  /**
   * Returns the "raw" result returned by Process.waitFor. This value is not
   * precisely defined, and is not to be confused with the value passed to
   * exit(2) by the subprocess.  Use getTerminationStatus() instead.
   */
  int getRawResult() {
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
  private static final int SIGNAL_1  = 128 + 1;
  private static final int SIGNAL_63 = 128 + 63;

  /**
   * Returns true iff the process exited normally.
   */
  public boolean exited() {
    return waitResult < SIGNAL_1 || waitResult > SIGNAL_63;
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
    if (exited()) {
      throw new IllegalStateException("getTerminatingSignal() not defined");
    }
    return waitResult - SIGNAL_1 + 1;
  }

  /**
   * Returns a short string describing the termination status.
   * e.g. "Exit 1" or "Hangup".
   */
  public String toShortString() {
    return exited()
      ? ("Exit " + getExitCode())
      : (getSignalString(getTerminatingSignal()));
  }

  @Override
  public String toString() {
    return exited()
      ? ("Process exited with status " + getExitCode())
      : ("Process terminated by signal " + getTerminatingSignal());
  }

  @Override
  public int hashCode() {
    return waitResult;
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof TerminationStatus &&
      ((TerminationStatus) other).waitResult == this.waitResult;
  }
}
