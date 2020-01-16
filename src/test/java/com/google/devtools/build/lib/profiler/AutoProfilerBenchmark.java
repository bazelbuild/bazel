// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.caliper.BeforeExperiment;
import com.google.caliper.Benchmark;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.UUID;

/**
 * Microbenchmarks for the overhead of {@link AutoProfiler} over using {@link Profiler} manually.
 */
public class AutoProfilerBenchmark {
  private final ProfilerTask profilerTaskType = ProfilerTask.INFO;

  @BeforeExperiment
  void startProfiler() throws Exception {
    Profiler.instance()
        .start(
            ImmutableSet.copyOf(ProfilerTask.values()),
            new InMemoryFileSystem().getPath("/out.dat").getOutputStream(),
            Profiler.Format.JSON_TRACE_FILE_FORMAT,
            "dummy_output_base",
            UUID.randomUUID(),
            false,
            BlazeClock.instance(),
            BlazeClock.instance().nanoTime(),
            /* enabledCpuUsageProfiling= */ false,
            /* slimProfile= */ false,
            /* enableActionCountProfile= */ false);
  }

  @BeforeExperiment
  void stopProfiler() throws Exception {
    Profiler.instance().stop();
  }

  @Benchmark
  void profiledWithAutoProfiler(int reps) {
    for (int i = 0; i < reps; i++) {
      try (AutoProfiler p = AutoProfiler.profiled("profiling", profilerTaskType)) {}
    }
  }

  @Benchmark
  void profiledManually(int reps) {
    for (int i = 0; i < reps; i++) {
      long startTime = Profiler.nanoTimeMaybe();
      Profiler.instance().logSimpleTask(startTime, profilerTaskType, "description");
    }
  }
}
