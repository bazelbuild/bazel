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

package com.google.devtools.build.lib.profiler;

import com.google.common.base.Ascii;
import com.google.common.flogger.GoogleLogger;
import com.google.common.flogger.LogSites;
import com.google.devtools.build.lib.profiler.AutoProfiler.ElapsedTimeReceiver;
import com.google.devtools.build.lib.profiler.AutoProfiler.FilteringElapsedTimeReceiver;
import java.time.Duration;
import java.util.concurrent.TimeUnit;

/** Utility for creating {@link AutoProfiler} instances from {@link GoogleLogger} instances. */
public class GoogleAutoProfilerUtils {
  private static final GoogleLogger selfLogger = GoogleLogger.forEnclosingClass();
  private static final String MILLISECONDS_STRING =
      Ascii.toLowerCase(TimeUnit.MILLISECONDS.toString());

  private GoogleAutoProfilerUtils() {}

  /**
   * Use {@link #logged(String, Duration)} instead. This should only be called when caller has a
   * specially configured {@code logger} and cannot use this class's one.
   */
  public static AutoProfiler logged(
      String description, GoogleLogger logger, long minTimeForLoggingInMilliseconds) {
    return AutoProfiler.create(makeReceiver(description, logger, minTimeForLoggingInMilliseconds));
  }

  public static AutoProfiler logged(String description, Duration duration) {
    return logged(description, selfLogger, duration.toMillis());
  }

  public static AutoProfiler logged(String description) {
    return AutoProfiler.create(
        elapsedTimeNanos ->
            selfLogger
                .atInfo()
                .withInjectedLogSite(LogSites.callerOf(AutoProfiler.class))
                .log(
                    AutoProfiler.LOGGING_MESSAGE_TEMPLATE,
                    TimeUnit.MILLISECONDS.convert(elapsedTimeNanos, TimeUnit.NANOSECONDS),
                    MILLISECONDS_STRING,
                    description));
  }

  private static ElapsedTimeReceiver makeReceiver(
      String description, GoogleLogger logger, long minTimeForLoggingInMilliseconds) {
    return new FloggerElapsedTimeReceiver(
        description, logger, minTimeForLoggingInMilliseconds, TimeUnit.MILLISECONDS);
  }

  /** {@link FilteringElapsedTimeReceiver} that logs messages to a {@link GoogleLogger}. */
  private static class FloggerElapsedTimeReceiver extends FilteringElapsedTimeReceiver {
    // Some classes in Google-internal Blaze use a specially configured logger. When those classes
    // record elapsed time using this library, they pass their logger in here, which we use instead
    // of this library's default selfLogger.
    private final GoogleLogger logger;

    FloggerElapsedTimeReceiver(
        String taskDescription, GoogleLogger logger, long minTimeForLogging, TimeUnit timeUnit) {
      super(taskDescription, minTimeForLogging, timeUnit);
      this.logger = logger;
    }

    @Override
    protected void log(String logMessage) {
      logger.atInfo().withInjectedLogSite(LogSites.callerOf(AutoProfiler.class)).log(logMessage);
    }
  }
}
