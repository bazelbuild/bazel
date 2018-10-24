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

import com.google.common.flogger.GoogleLogger;
import com.google.common.flogger.LogSite;
import com.google.devtools.build.lib.profiler.AutoProfiler.ElapsedTimeReceiver;
import com.google.devtools.build.lib.profiler.AutoProfiler.FilteringElapsedTimeReceiver;
import java.util.concurrent.TimeUnit;

/** Utility for creating {@link AutoProfiler} instances from {@link GoogleLogger} instances. */
public class GoogleAutoProfilerUtils {
  private GoogleAutoProfilerUtils() {}

  public static AutoProfiler logged(
      String description, GoogleLogger logger, long minTimeForLoggingInMilliseconds) {
    return AutoProfiler.create(makeReceiver(description, logger, minTimeForLoggingInMilliseconds));
  }

  public static AutoProfiler logged(
      String description,
      GoogleLogger logger,
      long minTimeForLoggingInMilliseconds,
      LogSite logSite) {
    return AutoProfiler.create(
        makeReceiver(description, logger, minTimeForLoggingInMilliseconds, logSite));
  }

  private static ElapsedTimeReceiver makeReceiver(
      String description, GoogleLogger logger, long minTimeForLoggingInMilliseconds) {
    return new FilteringElapsedTimeReceiver(
        description, minTimeForLoggingInMilliseconds, TimeUnit.MILLISECONDS) {
      @Override
      protected void log(String logMessage) {
        logger.atInfo().log(logMessage);
      }
    };
  }

  private static ElapsedTimeReceiver makeReceiver(
      String description,
      GoogleLogger logger,
      long minTimeForLoggingInMilliseconds,
      LogSite logSite) {
    return new FilteringElapsedTimeReceiver(
        description, minTimeForLoggingInMilliseconds, TimeUnit.MILLISECONDS) {
      @Override
      protected void log(String logMessage) {
        logger.atInfo().withInjectedLogSite(logSite).log(logMessage);
      }
    };
  }
}
