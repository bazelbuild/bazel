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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Supplier;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;
import java.util.logging.FileHandler;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogManager;
import java.util.logging.LogRecord;

/**
 * A {@link LogHandlerQuerier} for working with {@link java.util.logging.FileHandler} instances.
 *
 * <p>This querier is intended for situations where the logging handler is configured on the JVM
 * command line to be {@link java.util.logging.FileHandler}, but where the code which needs to query
 * the handler does not know the handler's class or cannot import it. The command line then should
 * in addition specify {@code
 * -Dcom.google.devtools.build.lib.util.LogHandlerQuerier.class=com.google.devtools.build.lib.util.FileHandlerQuerier}
 * and an instance of FileHandlerQuerier class can then be obtained from {@code
 * LogHandlerQuerier.getInstance()}.
 *
 * <p>Due to limitations of java.util.logging API, this querier only supports obtaining the log file
 * path when it's specified in java.util.logging.config with no % variables.
 *
 * <p>TODO: is intended that this class be removed once Bazel is no longer using
 * {@link java.util.logging.FileHandler}.
 */
public class FileHandlerQuerier extends LogHandlerQuerier {
  /** Wrapper around LogManager.getLogManager() for testing. */
  private final Supplier<LogManager> logManagerSupplier;

  @VisibleForTesting
  FileHandlerQuerier(Supplier<LogManager> logManagerSupplier) {
    this.logManagerSupplier = logManagerSupplier;
  }

  public FileHandlerQuerier() {
    this(() -> LogManager.getLogManager());
  }

  @Override
  protected boolean canQuery(Handler handler) {
    return handler instanceof FileHandler;
  }

  @Override
  protected Optional<Path> getLogHandlerFilePath(Handler handler) {
    // Hack: java.util.logging.FileHandler has no API for getting the current file path. Instead, we
    // try to parse the configured path and check that it has no % variables.
    String pattern = logManagerSupplier.get().getProperty("java.util.logging.FileHandler.pattern");
    if (pattern == null) {
      throw new IllegalStateException(
          "java.util.logging.config property java.util.logging.FileHandler.pattern is undefined");
    }
    if (pattern.matches(".*%[thgu].*")) {
      throw new IllegalStateException(
          "resolving %-coded variables in java.util.logging.config property "
              + "java.util.logging.FileHandler.pattern is not supported");
    }
    Path path = Paths.get(pattern.trim());

    // Hack: java.util.logging.FileHandler has no API for checking if a log file is currently open.
    // Instead, we try to query whether the handler can log a SEVERE level record - which for
    // expected configurations should be true iff a log file is open.
    if (!handler.isLoggable(new LogRecord(Level.SEVERE, ""))) {
      return Optional.empty();
    }
    return Optional.of(path);
  }
}
