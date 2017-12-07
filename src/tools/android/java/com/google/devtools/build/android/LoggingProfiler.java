// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android;

import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/** Logs timing data to the logger. */
public class LoggingProfiler implements Profiler {

  private final Logger logger = Logger.getLogger(LoggingProfiler.class.getName());
  private final Map<String, Stopwatch> tasks = new HashMap<>();

  private LoggingProfiler() {}

  public static Profiler createAndStart(String taskName) {
    final LoggingProfiler profiler = new LoggingProfiler();
    return profiler.startTask(taskName);
  }

  @Override
  public Profiler startTask(String taskName) {
    Preconditions.checkArgument(!tasks.containsKey(taskName));
    tasks.put(taskName, Stopwatch.createStarted());
    return this;
  }

  @Override
  public Profiler recordEndOf(String taskName) {
    Preconditions.checkArgument(tasks.containsKey(taskName));
    final Stopwatch task = tasks.remove(taskName);
    logger.finer(String.format("%s in %ss", taskName, task.elapsed(TimeUnit.MILLISECONDS) / 1000f));
    return this;
  }
}
