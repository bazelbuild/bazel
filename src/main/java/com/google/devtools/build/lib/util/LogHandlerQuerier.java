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

import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.errorprone.annotations.ForOverride;
import java.nio.file.Path;
import java.util.Optional;
import java.util.logging.Handler;
import java.util.logging.Logger;

/**
 * A retriever for logging handler properties, e.g. the log file path.
 *
 * <p>A querier is intended for situations where a logging handler is configured on the JVM command
 * line, and where the code which needs to query the handler does not know the handler's class or
 * cannot import it. The command line then should in addition specify an appropriate child class of
 * {@link LogHandlerQuerier} via the {@code
 * -Dcom.google.devtools.build.lib.util.LogHandlerQuerier.class} flag, and an instance of that
 * appropriate child class can be obtained from {@code LogHandlerQuerier.getInstance()}.
 */
public abstract class LogHandlerQuerier {
  private static final Supplier<LogHandlerQuerier> configuredInstanceSupplier =
      Suppliers.memoize(LogHandlerQuerier::makeConfiguredInstance);

  private static LogHandlerQuerier makeConfiguredInstance() {
    String propertyName = LogHandlerQuerier.class.getName() + ".class";
    String subclassName = System.getProperty(propertyName);
    checkState(subclassName != null, "System property %s is not defined", propertyName);
    try {
      return (LogHandlerQuerier)
          ClassLoader.getSystemClassLoader()
              .loadClass(subclassName)
              .getDeclaredConstructor()
              .newInstance();
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("System property " + propertyName + " value is invalid", e);
    }
  }

  /**
   * Returns the singleton instance of the LogHandlerQuerier child class which was configured as a
   * system property on the JVM command line via the {@code
   * -Dcom.google.devtools.build.lib.util.LogHandlerQuerier.class} flag.
   *
   * <p>This method is thread-safe.
   *
   * @throws IllegalStateException if the JVM property was not defined or if an instance of the
   *     class named by the property could not be constructed
   */
  public static LogHandlerQuerier getConfiguredInstance() {
    return configuredInstanceSupplier.get();
  }

  /**
   * Returns a logger's handler's log file path, iterating through all handlers and the logger's
   * ancestors' handlers as necessary.
   *
   * <p>The method will stop iterating at the first log handler that it can query, returning the log
   * path if it is available for that log handler, or an empty {@link Optional} if the log file for
   * that handler is currently unavailable.
   *
   * @param logger a logger whose handlers, and ancestors' handlers if necessary, will be queried
   * @throws IllegalArgumentException if the {@link LogHandlerQuerier} cannot query any {@link
   *     Handler} for this logger or its ancestors
   */
  public Optional<Path> getLoggerFilePath(Logger logger) {
    for (; logger != null; logger = logger.getParent()) {
      for (Handler handler : logger.getHandlers()) {
        if (canQuery(handler)) {
          return getLogHandlerFilePath(handler);
        }
      }
    }
    throw new IllegalArgumentException("Failed to find a queryable logging handler");
  }

  /** Checks if this {@link LogHandlerQuerier} can query the given handler. */
  @ForOverride
  protected abstract boolean canQuery(Handler handler);

  /**
   * Returns a logging handler's log file path.
   *
   * @param handler logging handler to query
   * @return the log handler's log file path if the log file is currently available
   */
  @ForOverride
  protected abstract Optional<Path> getLogHandlerFilePath(Handler handler);
}
