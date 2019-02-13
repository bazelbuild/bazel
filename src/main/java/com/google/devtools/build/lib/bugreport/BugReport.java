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
package com.google.devtools.build.lib.bugreport;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.util.CustomExitCodePublisher;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.io.OutErr;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * Utility methods for handling crashes: we log the crash, optionally send a bug report, and then
 * terminate the jvm.
 *
 * <p> Note, code in this class must be extremely robust. There's nothing worse than a crash-handler
 * that itself crashes!
 */
public abstract class BugReport {

  private BugReport() {}

  private static final Logger logger = Logger.getLogger(BugReport.class.getName());

  private static BlazeVersionInfo versionInfo = BlazeVersionInfo.instance();

  private static BlazeRuntimeInterface runtime = null;

  @Nullable private static volatile Throwable unprocessedThrowableInTest = null;
  private static final Object LOCK = new Object();

  private static final boolean IN_TEST = System.getenv("TEST_TMPDIR") != null;

  private static final boolean SHOULD_NOT_SEND_BUG_REPORT_BECAUSE_IN_TEST =
      IN_TEST && System.getenv("ENABLE_BUG_REPORT_LOGGING_IN_TEST") == null;

  /**
   * This is a narrow interface for {@link BugReport}'s usage of BlazeRuntime. It lives in this
   * file, for the sake of avoiding a build-time cycle.
   */
  public interface BlazeRuntimeInterface {
    String getProductName();
    void notifyCommandComplete(int exitCode);
    void shutdownOnCrash();
  }

  public static void setRuntime(BlazeRuntimeInterface newRuntime) {
    Preconditions.checkNotNull(newRuntime);
    Preconditions.checkState(runtime == null, "runtime already set: %s, %s", runtime, newRuntime);
    runtime = newRuntime;
  }

  private static String getProductName() {
    return runtime != null ? runtime.getProductName() : "<unknown>";
  }

  /**
   * In tests, Runtime#halt is disabled. Thus, the main thread should call this method whenever it
   * is about to block on thread completion that might hang because of a failed halt below.
   */
  public static void maybePropagateUnprocessedThrowableIfInTest() {
    if (IN_TEST) {
      // Instead of the jvm having been halted, we might have a saved Throwable.
      synchronized (LOCK) {
        Throwable lastUnprocessedThrowableInTest = unprocessedThrowableInTest;
        unprocessedThrowableInTest = null;
        if (lastUnprocessedThrowableInTest != null) {
          Throwables.throwIfUnchecked(lastUnprocessedThrowableInTest);
          throw new RuntimeException(lastUnprocessedThrowableInTest);
        }
      }
    }
  }

  /**
   * Logs the unhandled exception with a special prefix signifying that this was a crash.
   *
   * @param exception the unhandled exception to display.
   * @param args additional values to record in the message.
   * @param values Additional string values to clarify the exception.
   */
  public static void sendBugReport(Throwable exception, List<String> args, String... values) {
    if (SHOULD_NOT_SEND_BUG_REPORT_BECAUSE_IN_TEST) {
      Throwables.throwIfUnchecked(exception);
      throw new IllegalStateException(
          "Bug reports in tests should crash: " + args + ", " + Arrays.toString(values), exception);
    }
    if (!versionInfo.isReleasedBlaze()) {
      logger.info("(Not a released binary; not logged.)");
      return;
    }

    logException(exception, filterClientEnv(args), values);
  }

  private static void logCrash(Throwable throwable, boolean sendBugReport, String... args) {
    logger.severe("Crash: " + Throwables.getStackTraceAsString(throwable));
    if (sendBugReport) {
      BugReport.sendBugReport(throwable, Arrays.asList(args));
    }
    BugReport.printBug(OutErr.SYSTEM_OUT_ERR, throwable);
    System.err.println("ERROR: " + getProductName() + " crash in async thread:");
    throwable.printStackTrace();
  }

  /**
   * Print, log, send a bug report, and then cause the current Blaze command to fail with the
   * specified exit code, and then cause the jvm to terminate.
   *
   * <p>Has no effect if another crash has already been handled by {@link BugReport}.
   */
  public static void handleCrashWithoutSendingBugReport(
      Throwable throwable, int exitCode, String... args) {
    handleCrash(throwable, /*sendBugReport=*/ false, Integer.valueOf(exitCode), args);
  }

  /**
   * Print, log, send a bug report, and then cause the current Blaze command to fail with the
   * specified exit code, and then cause the jvm to terminate.
   *
   * <p>Has no effect if another crash has already been handled by {@link BugReport}.
   */
  public static void handleCrash(Throwable throwable, int exitCode, String... args) {
    handleCrash(throwable, /*sendBugReport=*/ true, Integer.valueOf(exitCode), args);
  }

  /**
   * Print, log, and send a bug report, and then cause the current Blaze command to fail with an
   * exit code inferred from the given {@link Throwable}, and then cause the jvm to terminate.
   *
   * <p>Has no effect if another crash has already been handled by {@link BugReport}.
   */
  public static void handleCrash(Throwable throwable, String... args) {
    handleCrash(throwable, /*sendBugReport=*/ true, /*exitCode=*/ null, args);
  }

  private static void handleCrash(
      Throwable throwable, boolean sendBugReport, @Nullable Integer exitCode, String... args) {
    int exitCodeToUse = exitCode == null
        ? getExitCodeForThrowable(throwable).getNumericExitCode()
        : exitCode.intValue();
    try {
      synchronized (LOCK) {
        if (IN_TEST) {
          unprocessedThrowableInTest = throwable;
        }
        logCrash(throwable, sendBugReport, args);
        try {
          if (runtime != null) {
            runtime.notifyCommandComplete(exitCodeToUse);
            // We don't call runtime#shutDown() here because all it does is shut down the modules,
            // and who knows if they can be trusted. Instead, we call runtime#shutdownOnCrash()
            // which attempts to cleanly shutdown those modules that might have something pending
            // to do as a best-effort operation.
            runtime.shutdownOnCrash();
          }
          CustomExitCodePublisher.maybeWriteExitStatusFile(exitCodeToUse);
        } finally {
          // Avoid shutdown deadlock issues: If an application shutdown hook crashes, it will
          // trigger our Blaze crash handler (this method). Calling System#exit() here, would
          // therefore induce a deadlock. This call would block on the shutdown sequence completing,
          // but the shutdown sequence would in turn be blocked on this thread finishing. Instead,
          // exit fast via halt().
          Runtime.getRuntime().halt(exitCodeToUse);
        }
      }
    } catch (Throwable t) {
      System.err.println(
          "ERROR: A crash occurred while "
              + getProductName()
              + " was trying to handle a crash! Please file a bug against "
              + getProductName()
              + " and include the information below.");

      System.err.println("Original uncaught exception:");
      throwable.printStackTrace(System.err);

      System.err.println("Exception encountered during BugReport#handleCrash:");
      t.printStackTrace(System.err);

      Runtime.getRuntime().halt(exitCodeToUse);
    }
  }

  /** Get exit code corresponding to throwable. */
  public static ExitCode getExitCodeForThrowable(Throwable throwable) {
    return (Throwables.getRootCause(throwable) instanceof OutOfMemoryError)
        ? ExitCode.OOM_ERROR
        : ExitCode.BLAZE_INTERNAL_ERROR;
  }

  private static void printThrowableTo(OutErr outErr, Throwable e) {
    PrintStream err = new PrintStream(outErr.getErrorStream());
    e.printStackTrace(err);
    err.flush();
    logger.log(Level.SEVERE, getProductName() + " crashed", e);
  }

  /**
   * Print user-helpful information about the bug/crash to the output.
   *
   * @param outErr where to write the output
   * @param e the exception thrown
   */
  public static void printBug(OutErr outErr, Throwable e) {
    if (e instanceof OutOfMemoryError) {
      outErr.printErr(
          e.getMessage() + "\n\nERROR: " + getProductName() + " ran out of memory and crashed.\n");
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
    logger.severe("Exception: " + Throwables.getStackTraceAsString(exception));
    // The preamble is used in the crash watcher, so don't change it
    // unless you know what you're doing.
    String preamble = getProductName()
        + (exception instanceof OutOfMemoryError ? " OOMError: " : " crashed with args: ");

    LoggingUtil.logToRemote(Level.SEVERE, preamble + Joiner.on(' ').join(args), exception,
        values);
  }
}
