// Copyright 2020 The Bazel Authors. All rights reserved.
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

import java.io.IOException;
import java.nio.file.Path;
import java.util.Optional;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Utility to handle low-level interactions with debug ("info") logging. While actual logging is
 * done with the {@code GoogleLogger} class, getting at internals is easier with the native {@link
 * Logger} object.
 */
public class DebugLoggerConfigurator {
  private static final Logger logger = Logger.getLogger(DebugLoggerConfigurator.class.getName());
  // Make sure we keep a strong reference to this logger, so that the
  // configuration isn't lost when the gc kicks in.
  private static final Logger templateLogger = Logger.getLogger("com.google.devtools.build");
  private static Level currentVerbosityLevel = null;

  private DebugLoggerConfigurator() {}

  /**
   * Returns the path to the Blaze/Bazel server INFO log.
   *
   * @return the path to the log or empty if the log is not yet open
   * @throws IOException if the log location cannot be determined
   */
  public static Optional<String> getServerLogPath() throws IOException {
    return LogHandlerQuerier.getConfiguredInstance()
        .getLoggerFilePath(logger)
        .map(Path::toAbsolutePath)
        .map(Object::toString);
  }

  /** Configures "com.google.devtools.build.*" loggers to the given {@code level}. */
  public static void setupLogging(Level level) {
    if (!level.equals(currentVerbosityLevel)) {
      templateLogger.setLevel(level);
      templateLogger.info("Log level: " + templateLogger.getLevel());
      currentVerbosityLevel = level;
    }
  }

  /** Flushes all loggers at com.google.devtools.build.* or higher. */
  public static void flushServerLog() {
    for (Logger logger = templateLogger; logger != null; logger = logger.getParent()) {
      for (Handler handler : logger.getHandlers()) {
        if (handler != null) {
          handler.flush();
        }
      }
    }
  }
}
