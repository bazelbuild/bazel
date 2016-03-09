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
package com.google.devtools.build.lib.runtime;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.OutErr;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Utility methods for sending bug reports.
 *
 * <p> Note, code in this class must be extremely robust.  There's nothing
 * worse than a crash-handler that itself crashes!
 */
public abstract class BugReport {

  private BugReport() {}

  private static Logger LOG = Logger.getLogger(BugReport.class.getName());

  private static BlazeVersionInfo versionInfo = BlazeVersionInfo.instance();

  private static BlazeRuntime runtime = null;

  private static AtomicBoolean alreadyHandlingCrash = new AtomicBoolean(false);

  public static void setRuntime(BlazeRuntime newRuntime) {
    Preconditions.checkNotNull(newRuntime);
    Preconditions.checkState(runtime == null, "runtime already set: %s, %s", runtime, newRuntime);
    runtime = newRuntime;
  }

  /**
   * Logs the unhandled exception with a special prefix signifying that this was a crash.
   *
   * @param exception the unhandled exception to display.
   * @param args additional values to record in the message.
   * @param values Additional string values to clarify the exception.
   */
  public static void sendBugReport(Throwable exception, List<String> args, String... values) {
    if (!versionInfo.isReleasedBlaze()) {
      LOG.info("(Not a released binary; not logged.)");
      return;
    }

    logException(exception, filterClientEnv(args), values);
  }

  private static void logCrash(Throwable throwable, String... args) {
    BugReport.sendBugReport(throwable, Arrays.asList(args));
    BugReport.printBug(OutErr.SYSTEM_OUT_ERR, throwable);
    System.err.println("Blaze crash in async thread:");
    throwable.printStackTrace();
  }

  /**
   * Print and send a bug report, and exit with the proper Blaze code. Does not exit if called a
   * second time.
   */
  public static void handleCrash(Throwable throwable, String... args) {
    if (alreadyHandlingCrash.compareAndSet(false, true)) {
      int exitCode = getExitCodeForThrowable(throwable);
      try {
        logCrash(throwable, args);
        if (runtime != null) {
          runtime.notifyCommandComplete(exitCode);
          // We don't call runtime#shutDown() here because all it does is shut down the modules, and
          // who knows if they can be trusted.
        }
      } finally {
        // Avoid shutdown deadlock issues: If an application shutdown hook crashes, it will trigger
        // our Blaze crash handler (this method). Calling System#exit() here, would therefore induce
        // a deadlock. This call would block on the shutdown sequence completing, but the shutdown
        // sequence would in turn be blocked on this thread finishing. Instead, exit fast via
        // halt().
        Runtime.getRuntime().halt(exitCode);
      }
    } else {
      logCrash(throwable, args);
    }
  }

  /** Get exit code corresponding to throwable. */
  public static int getExitCodeForThrowable(Throwable throwable) {
    return (throwable instanceof OutOfMemoryError)
        ? ExitCode.OOM_ERROR.getNumericExitCode()
        : ExitCode.BLAZE_INTERNAL_ERROR.getNumericExitCode();
  }

  private static void printThrowableTo(OutErr outErr, Throwable e) {
    PrintStream err = new PrintStream(outErr.getErrorStream());
    e.printStackTrace(err);
    err.flush();
    LOG.log(Level.SEVERE, "Blaze crashed", e);
  }

  /**
   * Print user-helpful information about the bug/crash to the output.
   *
   * @param outErr where to write the output
   * @param e the exception thrown
   */
  public static void printBug(OutErr outErr, Throwable e) {
    if (e instanceof OutOfMemoryError) {
      outErr.printErr(e.getMessage() + "\n\n" +
          "Blaze ran out of memory and crashed.\n");
    } else {
      printThrowableTo(outErr, e);
    }
  }

  /**
   * Filters {@code args} by removing any item that starts with "--client_env",
   * then returns this as an immutable list.
   *
   * <p>The client's environment variables may contain sensitive data, so we filter it out.
   */
  private static List<String> filterClientEnv(Iterable<String> args) {
    if (args == null) {
      return null;
    }

    ImmutableList.Builder<String> filteredArgs = ImmutableList.builder();
    for (String arg : args) {
      if (arg != null && !arg.startsWith("--client_env")) {
        filteredArgs.add(arg);
      }
    }
    return filteredArgs.build();
  }

  // Log the exception.  Because this method is only called in a blaze release,
  // this will result in a report being sent to a remote logging service.
  private static void logException(Throwable exception, List<String> args, String... values) {
    // The preamble is used in the crash watcher, so don't change it
    // unless you know what you're doing.
    String preamble = exception instanceof OutOfMemoryError
        ? "Blaze OOMError: "
        : "Blaze crashed with args: ";

    LoggingUtil.logToRemote(Level.SEVERE, preamble + Joiner.on(' ').join(args), exception,
        values);
  }
}
