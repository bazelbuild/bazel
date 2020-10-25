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

import static com.google.common.base.Strings.isNullOrEmpty;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.util.CrashFailureDetails;
import com.google.devtools.build.lib.util.CustomExitCodePublisher;
import com.google.devtools.build.lib.util.CustomFailureDetailPublisher;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.TestType;
import com.google.devtools.build.lib.util.io.OutErr;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import javax.annotation.Nullable;

/**
 * Utility methods for handling crashes: we log the crash, optionally send a bug report, and then
 * terminate the jvm.
 *
 * <p>Note, code in this class must be extremely robust. There's nothing worse than a crash-handler
 * that itself crashes!
 */
public final class BugReport {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  static final BugReporter REPORTER_INSTANCE = new DefaultBugReporter();

  private static final BlazeVersionInfo VERSION_INFO = BlazeVersionInfo.instance();

  private static BlazeRuntimeInterface runtime = null;

  @Nullable private static volatile Throwable unprocessedThrowableInTest = null;
  private static final Object LOCK = new Object();

  private static final boolean SHOULD_NOT_SEND_BUG_REPORT_BECAUSE_IN_TEST =
      TestType.isInTest() && System.getenv("ENABLE_BUG_REPORT_LOGGING_IN_TEST") == null;

  private BugReport() {}

  /**
   * This is a narrow interface for {@link BugReport}'s usage of BlazeRuntime. It lives in this
   * file, for the sake of avoiding a build-time cycle.
   */
  public interface BlazeRuntimeInterface {
    String getProductName();
    /**
     * Perform all possible clean-up before crashing, posting events etc. so long as crashing isn't
     * significantly delayed or another crash isn't triggered.
     */
    void cleanUpForCrash(DetailedExitCode exitCode);
  }

  public static void setRuntime(BlazeRuntimeInterface newRuntime) {
    Preconditions.checkNotNull(newRuntime);
    Preconditions.checkState(
        runtime == null || TestType.isInTest(), "runtime already set: %s, %s", runtime, newRuntime);
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
    if (TestType.isInTest()) {
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
   * Convenience method for {@linkplain #sendBugReport(Throwable, List, String...) sending a bug
   * report} without additional arguments.
   */
  public static void sendBugReport(Throwable exception) {
    sendBugReport(exception, /*args=*/ ImmutableList.of());
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
    if (!VERSION_INFO.isReleasedBlaze()) {
      logger.atInfo().log("(Not a released binary; not logged.)");
      return;
    }

    logException(exception, filterArgs(args), values);
  }

  private static void logCrash(Throwable throwable, boolean sendBugReport, String... args) {
    logger.atSevere().withCause(throwable).log("Crash");
    if (sendBugReport) {
      BugReport.sendBugReport(throwable, Arrays.asList(args));
    }
    logThrowableToConsole(throwable);
  }

  private static void logThrowableToConsole(Throwable throwable) {
    BugReport.printBug(OutErr.SYSTEM_OUT_ERR, throwable, /*oomMessage=*/ null);
    System.err.println("ERROR: " + getProductName() + " crash in async thread:");
    throwable.printStackTrace();
  }

  /**
   * Print, log, and then cause the current Blaze command to fail with the specified exit code, and
   * then cause the jvm to terminate.
   *
   * <p>Has no effect if another crash has already been handled by {@link BugReport}.
   */
  public static RuntimeException handleCrashWithoutSendingBugReport(
      Throwable throwable, ExitCode exitCode, String... args) {
    throw handleCrash(
        throwable,
        /*sendBugReport=*/ false,
        DetailedExitCode.of(exitCode, CrashFailureDetails.forThrowable(throwable)),
        args);
  }

  /**
   * Print, log, send a bug report, and then cause the current Blaze command to fail with the
   * specified exit code, and then cause the jvm to terminate.
   *
   * <p>Has no effect if another crash has already been handled by {@link BugReport}.
   */
  public static RuntimeException handleCrash(
      Throwable throwable, ExitCode exitCode, String... args) {
    throw handleCrash(
        throwable,
        /*sendBugReport=*/ true,
        DetailedExitCode.of(exitCode, CrashFailureDetails.forThrowable(throwable)),
        args);
  }

  /**
   * Print, log, and send a bug report, and then cause the current Blaze command to fail with an
   * exit code inferred from the given {@link Throwable}, and then cause the jvm to terminate.
   *
   * <p>Has no effect if another crash has already been handled by {@link BugReport}.
   */
  public static RuntimeException handleCrash(Throwable throwable, String... args) {
    throw handleCrash(
        throwable,
        /*sendBugReport=*/ true,
        CrashFailureDetails.detailedExitCodeForThrowable(throwable),
        args);
  }

  private static RuntimeException handleCrash(
      Throwable throwable,
      boolean sendBugReport,
      DetailedExitCode detailedExitCode,
      String... args) {
    int numericExitCode = detailedExitCode.getExitCode().getNumericExitCode();
    try {
      synchronized (LOCK) {
        if (TestType.isInTest()) {
          unprocessedThrowableInTest = throwable;
        }
        // Don't try to send a bug report during a crash in a test, it will throw itself.
        if (!TestType.isInTest() || !sendBugReport) {
          logCrash(throwable, sendBugReport, args);
        } else {
          logThrowableToConsole(throwable);
        }
        // TODO(b/167592709): remove verbose logging when bug resolved.
        logger.atInfo().log("Finished logging crash, runtime: %s", runtime);
        try {
          if (runtime != null) {
            runtime.cleanUpForCrash(detailedExitCode);
          }
          logger.atInfo().log("Finished runtime cleanup");
          CustomExitCodePublisher.maybeWriteExitStatusFile(numericExitCode);
          logger.atInfo().log("Wrote exit status file");
          CustomFailureDetailPublisher.maybeWriteFailureDetailFile(
              detailedExitCode.getFailureDetail());
          logger.atInfo().log("Wrote failure detail file");
        } finally {
          logger.atInfo().log("Entered inner finally block");
          // Avoid shutdown deadlock issues: If an application shutdown hook crashes, it will
          // trigger our Blaze crash handler (this method). Calling System#exit() here, would
          // therefore induce a deadlock. This call would block on the shutdown sequence completing,
          // but the shutdown sequence would in turn be blocked on this thread finishing. Instead,
          // exit fast via halt().
          Runtime.getRuntime().halt(numericExitCode);
          logger.atSevere().log("Failed to halt (inner block)!");
        }
      }
    } catch (Throwable t) {
      logger.atSevere().withCause(t).log("Threw while crashing");
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
    } finally {
      logger.atInfo().log("Entered outer finally block");
      Runtime.getRuntime().halt(numericExitCode);
      logger.atSevere().log("Failed to halt (outer block)!");
    }
    logger.atSevere().log("Failed to crash in handleCrash");
    throw new IllegalStateException("never get here", throwable);
  }

  private static void printThrowableTo(OutErr outErr, Throwable e) {
    PrintStream err = new PrintStream(outErr.getErrorStream());
    e.printStackTrace(err);
    err.flush();
    logger.atSevere().withCause(e).log("%s crashed", getProductName());
  }

  /**
   * Print user-helpful information about the bug/crash to the output.
   *
   * @param outErr where to write the output
   * @param e the exception thrown
   */
  public static void printBug(OutErr outErr, Throwable e, @Nullable String oomMessage) {
    if (e instanceof OutOfMemoryError) {
      outErr.printErr("ERROR: " + constructOomExitMessage(oomMessage));
    } else {
      printThrowableTo(outErr, e);
    }
  }

  public static String constructOomExitMessage(@Nullable String extraInfo) {
    String msg = getProductName() + " ran out of memory and crashed.";
    return isNullOrEmpty(extraInfo) ? msg : msg + " " + extraInfo;
  }

  /**
   * Filters {@code args} by removing superfluous items:
   *
   * <ul>
   *   <li>The client's environment variables may contain sensitive data, so we filter it out.
   *   <li>{@code --default_override} is spammy.
   * </ul>
   */
  private static List<String> filterArgs(Iterable<String> args) {
    if (args == null) {
      return null;
    }

    ImmutableList.Builder<String> filteredArgs = ImmutableList.builder();
    for (String arg : args) {
      if (arg != null
          && !arg.startsWith("--client_env=")
          && !arg.startsWith("--default_override=")) {
        filteredArgs.add(arg);
      }
    }
    return filteredArgs.build();
  }

  // Log the exception.  Because this method is only called in a blaze release,
  // this will result in a report being sent to a remote logging service.
  private static void logException(Throwable exception, List<String> args, String... values) {
    logger.atSevere().withCause(exception).log("Exception");
    // The preamble is used in the crash watcher, so don't change it
    // unless you know what you're doing.
    String preamble =
        getProductName()
            + (exception instanceof OutOfMemoryError ? " OOMError: " : " crashed with args: ");

    LoggingUtil.logToRemote(Level.SEVERE, preamble + Joiner.on(' ').join(args), exception, values);
  }

  private static class DefaultBugReporter implements BugReporter {
    @Override
    public void sendBugReport(Throwable exception) {
      BugReport.sendBugReport(exception);
    }

    @Override
    public void sendBugReport(Throwable exception, List<String> args, String... values) {
      BugReport.sendBugReport(exception, args, values);
    }

    @Override
    public RuntimeException handleCrash(Throwable throwable, String... args) {
      throw BugReport.handleCrash(throwable, args);
    }
  }
}
