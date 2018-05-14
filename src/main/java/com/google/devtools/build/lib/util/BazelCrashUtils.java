// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.util;

import com.google.common.base.Throwables;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/** Utilities for crashing the jvm. */
public class BazelCrashUtils {
  private static final Logger logger = Logger.getLogger(BazelCrashUtils.class.getName());

  private static final Object HALT_LOCK = new Object();

  @Nullable private static volatile Throwable unprocessedThrowableInTest = null;

  private BazelCrashUtils() {}

  /**
   * In tests, System.exit is disabled. Thus, the main thread should call this method whenever it is
   * about to block on thread completion that might hang because of a failed halt below.
   */
  public static void maybePropagateUnprocessedThrowableIfInTest() {
    synchronized (HALT_LOCK) {
      Throwable lastUnprocessedThrowableInTest = unprocessedThrowableInTest;
      unprocessedThrowableInTest = null;
      if (lastUnprocessedThrowableInTest != null) {
        Throwables.throwIfUnchecked(lastUnprocessedThrowableInTest);
        throw new RuntimeException(lastUnprocessedThrowableInTest);
      }
    }
  }

  /** Terminates the jvm with the exit code {@link ExitCode#BLAZE_INTERNAL_ERROR}. */
  public static RuntimeException halt(Throwable cause) {
    return halt(cause, ExitCode.BLAZE_INTERNAL_ERROR.getNumericExitCode());
  }

  /**
   * Terminates the jvm with the given {@code exitCode}. Note that, in server mode, the jvm's exit
   * code has nothing to do with the exit code picked by the blaze client, except as communicated by
   * the 'custom_exit_code_on_abrupt_exit' behavior. If this fails, the exit code picked by the
   * client will default to INTERNAL_ERROR.
   */
  public static RuntimeException halt(Throwable crash, int exitCode) {
    synchronized (HALT_LOCK) {
      try {
        logCrashDetails(crash);
        unprocessedThrowableInTest = crash;
        CustomExitCodePublisher.maybeWriteExitStatusFile(exitCode);
      } catch (OutOfMemoryError oom) {
        // In case we tried to halt with a non-OOM, but OOMed during the lead up to the halt.
        Runtime.getRuntime().halt(exitCode);
      } finally {
        doDie(crash, exitCode);
      }
      throw new IllegalStateException("impossible");
    }
  }

  private static void logCrashDetails(Throwable crash) {
    logger.severe(
        "About to call System#exit. Halt requested due to crash. See stderr for details ("
            + crash.getMessage()
            + ")");
    System.err.println("About to call System#exit. Halt requested due to crash");
    crash.printStackTrace();
    System.err.println("... called from");
    new Exception().printStackTrace();
  }

  private static void doDie(Throwable crash, int exitCode) {
    try {
      // Bypass shutdown hooks if we encountered an OOM, they're likely to take way too long to be
      // sensible to run in such a hostile environment.
      Throwable cause = crash;
      while (cause != null) {
        if (cause instanceof OutOfMemoryError) {
          Runtime.getRuntime().halt(exitCode);
        }
        cause = cause.getCause();
      }
    } finally {
      System.exit(exitCode);
    }
  }
}
