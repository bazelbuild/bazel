// Copyright 2015 Google Inc. All rights reserved.
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

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

/**
 * A convenient way to actively get access to timing information (e.g. for logging purposes) with
 * minimal boilerplate. Contrast with {@link Profiler}, which records profiling data for offline
 * consumption.
 *
 * <p>The intended usage is:
 *
 * <pre>
 * {@code
 * try (AutoProfiler profiler = AutoProfiler.createLoggingProfiler("Your stuff")) {
 *   // Your code here.
 * }
 * }
 * </pre>
 *
 * <p>but if the try-with-resources pattern is too cumbersome, you can also do
 *
 * <pre>
 * {@code
 * AutoProfiler profiler = AutoProfiler.createLoggingProfiler("Your stuff");
 * // Your code here.
 * profiler.close();
 * }
 * </pre>
 */
public class AutoProfiler implements AutoCloseable {
  private static final Logger LOG = Logger.getLogger(AutoProfiler.class.getName());

  private final ElapsedTimeReceiver elapsedTimeReceiver;
  private final long startTimeNanos;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  private AutoProfiler(ElapsedTimeReceiver elapsedTimeReceiver, long startTimeNanos) {
    this.elapsedTimeReceiver = elapsedTimeReceiver;
    this.startTimeNanos = startTimeNanos;
  }

  /** A opaque receiver of elapsed time information. */
  public interface ElapsedTimeReceiver {
    /**
     * Receives the elapsed time of the lifetime of an {@link AutoProfiler} instance.
     *
     * <p>Note that System#nanoTime isn't guaranteed to be non-decreasing, so implementations should
     * check for non-positive {@code elapsedTimeNanos} if they care about this sort of thing.
     */
    void accept(long elapsedTimeNanos);
  }

  /**
   * Returns an {@link AutoProfiler} that, when closed, logs the elapsed time in milliseconds to a
   * default logger.
   *
   * <p>The returned {@link AutoProfiler} is thread-safe.
   */
  public static AutoProfiler create(String taskDescription) {
    return create(taskDescription, LOG);
  }

  /**
   * Returns an {@link AutoProfiler} that, when closed, logs the elapsed time in milliseconds to
   * the given {@link Logger}.
   *
   * <p>The returned {@link AutoProfiler} is thread-safe.
   */
  public static AutoProfiler create(String taskDescription, Logger logger) {
    return create(taskDescription, logger, TimeUnit.MILLISECONDS);
  }

  /**
   * Returns an {@link AutoProfiler} that, when closed, logs the elapsed time the given
   * {@link TimeUnit} to the given {@link Logger}.
   *
   * <p>The returned {@link AutoProfiler} is thread-safe.
   */
  public static AutoProfiler create(String taskDescription, Logger logger, TimeUnit timeUnit) {
    return create(new LoggingElapsedTimeReceiver(taskDescription, logger, timeUnit));
  }

  /**
   * Returns an {@link AutoProfiler} that, when closed, invokes the given
   * {@link ElapsedTimeReceiver}.
   *
   * <p>The returned {@link AutoProfiler} is as thread-safe as the given
   * {@link ElapsedTimeReceiver} is.
   */
  public static AutoProfiler create(ElapsedTimeReceiver elapsedTimeReceiver) {
    return new AutoProfiler(elapsedTimeReceiver, BlazeClock.nanoTime());
  }

  /** Can be called at most once. */
  @Override
  public void close() {
    long elapsedTimeNanos = BlazeClock.nanoTime() - startTimeNanos;
    Preconditions.checkState(closed.compareAndSet(false, true));
    elapsedTimeReceiver.accept(elapsedTimeNanos);
  }

  private static class LoggingElapsedTimeReceiver implements ElapsedTimeReceiver {
    private final String taskDescription;
    private final Logger logger;
    private final TimeUnit timeUnit;

    private LoggingElapsedTimeReceiver(String taskDescription, Logger logger, TimeUnit timeUnit) {
      this.taskDescription = taskDescription;
      this.logger = logger;
      this.timeUnit = timeUnit;
    }

    @Override
    public void accept(long elapsedTimeNanos) {
      if (elapsedTimeNanos > 0) {
        logger.info(String.format("Spent %d %s doing %s",
            timeUnit.convert(elapsedTimeNanos, TimeUnit.NANOSECONDS),
            timeUnit.toString(),
            taskDescription));
      }
    }
  }
}

