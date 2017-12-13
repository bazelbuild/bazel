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

package com.google.devtools.build.lib.util;

import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/**
 * Logging utilities for sending log messages to a remote service. Log messages
 * will not be output anywhere else, including the terminal and blaze clients.
 */
@ThreadSafety.ThreadSafe
public final class LoggingUtil {
  // TODO(bazel-team): this class is a thin wrapper around Logger and could probably be discarded.
  private static Future<Logger> remoteLogger;

  /**
   * Installs the remote logger.
   *
   * <p>This can only be called once, and the caller should not keep the
   * reference to the logger.
   *
   * @param logger The logger future. Must have already started.
   */
  public static synchronized void installRemoteLogger(Future<Logger> logger) {
    Preconditions.checkState(remoteLogger == null);
    remoteLogger = logger;
  }

  /**
   * Installs the remote logger. Same as {@link #installRemoteLogger}, but since multiple tests will
   * run in the same JVM, does not assert that this is the first time the logger is being installed.
   */
  public static synchronized void installRemoteLoggerForTesting(Future<Logger> logger) {
    remoteLogger = logger;
  }

  /** Returns the installed logger, or null if none is installed. */
  public static synchronized Logger getRemoteLogger() {
    try {
      return (remoteLogger == null) ? null : Uninterruptibles.getUninterruptibly(remoteLogger);
    } catch (ExecutionException e) {
      throw new RuntimeException("Unexpected error initializing remote logging", e);
    }
  }

  /**
   * @see #logToRemote(Level, String, Throwable, String...).
   */
  public static void logToRemote(Level level, String msg, Throwable trace) {
    Logger logger = getRemoteLogger();
    if (logger != null) {
      logger.log(level, msg, trace);
    }
  }

  /**
   * Log a message to the remote backend.  This is done out of thread, so this
   * method is non-blocking.
   *
   * @param level The severity level. Non null.
   * @param msg The log message. Non null.
   * @param trace The stack trace.  May be null.
   * @param values Additional values to upload.
   */
  public static void logToRemote(Level level, String msg, Throwable trace,
      String... values) {
    Logger logger = getRemoteLogger();
    if (logger != null) {
      LogRecord logRecord = new LogRecord(level, msg);
      logRecord.setThrown(trace);
      logRecord.setParameters(values);
      logger.log(logRecord);
    }
  }
}
