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

import static java.util.stream.Collectors.joining;

import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Logs timing data to an in memory list that can be dumped. */
public class InMemoryProfiler implements Profiler {

  private final Map<String, Stopwatch> tasks = new HashMap<>();
  private final List<Timing> report = new ArrayList<>();

  private InMemoryProfiler() {}

  public static InMemoryProfiler createAndStart(String taskName) {
    final InMemoryProfiler profiler = new InMemoryProfiler();
    return profiler.startTask(taskName);
  }

  @Override
  public InMemoryProfiler startTask(String taskName) {
    Preconditions.checkArgument(!tasks.containsKey(taskName));
    tasks.put(taskName, Stopwatch.createStarted());
    return this;
  }

  @Override
  public InMemoryProfiler recordEndOf(String taskName) {
    Preconditions.checkArgument(tasks.containsKey(taskName));
    final Stopwatch task = tasks.remove(taskName);
    report.add(new Timing(taskName, task.elapsed().toMillis() / 1000f));
    return this;
  }

  public String asTimingReport() {
    return report.stream().map(Timing::toString).collect(joining("\n"));
  }

  static final class Timing {

    private final String taskName;
    private final float seconds;

    Timing(String taskName, float seconds) {
      this.taskName = taskName;
      this.seconds = seconds;
    }

    @Override
    public String toString() {
      return String.format("%s in %,.2fs", taskName, seconds);
    }
  }
}
