// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkState;
import static java.util.Map.entry;

import com.google.common.collect.ImmutableMap;
import javax.annotation.Nullable;

/** Describes counter series to be logged into profile. */
public record CounterSeriesTask(String laneName, @Nullable String colorName, String seriesName) {
  public static CounterSeriesTask ofProfilerTask(ProfilerTask profilerTask) {
    checkState(
        COUNTER_TASK_TO_SERIES_NAME.containsKey(profilerTask),
        "COUNTER_TASK_TO_SERIES_NAME does not contain %s",
        profilerTask);

    var laneName = profilerTask.description;
    var seriesName = COUNTER_TASK_TO_SERIES_NAME.get(profilerTask);
    var colorName = COUNTER_TASK_TO_COLOR.get(profilerTask);
    return new CounterSeriesTask(laneName, colorName, seriesName);
  }

  private static final ImmutableMap<ProfilerTask, String> COUNTER_TASK_TO_SERIES_NAME =
      ImmutableMap.ofEntries(
          entry(ProfilerTask.ACTION_COUNTS, "action"),
          entry(ProfilerTask.ACTION_CACHE_COUNTS, "local action cache"),
          entry(ProfilerTask.LOCAL_ACTION_COUNTS, "local action"),
          entry(ProfilerTask.LOCAL_CPU_USAGE, "cpu"),
          entry(ProfilerTask.SYSTEM_CPU_USAGE, "system cpu"),
          entry(ProfilerTask.LOCAL_MEMORY_USAGE, "memory"),
          entry(ProfilerTask.SYSTEM_MEMORY_USAGE, "system memory"),
          entry(ProfilerTask.SYSTEM_NETWORK_UP_USAGE, "system network up (Mbps)"),
          entry(ProfilerTask.SYSTEM_NETWORK_DOWN_USAGE, "system network down (Mbps)"),
          entry(ProfilerTask.WORKERS_MEMORY_USAGE, "workers memory"),
          entry(ProfilerTask.SYSTEM_LOAD_AVERAGE, "load"),
          entry(ProfilerTask.MEMORY_USAGE_ESTIMATION, "estimated memory"),
          entry(ProfilerTask.CPU_USAGE_ESTIMATION, "estimated cpu"),
          entry(ProfilerTask.PRESSURE_STALL_FULL_IO, "i/o pressure (full)"),
          entry(ProfilerTask.PRESSURE_STALL_FULL_MEMORY, "memory pressure (full)"),
          entry(ProfilerTask.PRESSURE_STALL_SOME_IO, "i/o pressure (some)"),
          entry(ProfilerTask.PRESSURE_STALL_SOME_MEMORY, "memory pressure (some)"),
          entry(ProfilerTask.PRESSURE_STALL_SOME_CPU, "cpu pressure (some)"),
          entry(ProfilerTask.ACTION_EXECUTION_SKYFUNCTION, "action execution (total)"),
          entry(ProfilerTask.ACTION_EXECUTION_SKYFUNCTION_DONE, "action execution (done)"),
          entry(ProfilerTask.CONFIGURED_TARGET_SKYFUNCTION, "configured target (total)"),
          entry(ProfilerTask.CONFIGURED_TARGET_SKYFUNCTION_DONE, "configured target (done)"),
          entry(ProfilerTask.ASPECT_SKYFUNCTION, "aspect (total)"),
          entry(ProfilerTask.ASPECT_SKYFUNCTION_DONE, "aspect (done)"),
          entry(ProfilerTask.PACKAGE_SKYFUNCTION, "package (total)"),
          entry(ProfilerTask.PACKAGE_SKYFUNCTION_DONE, "package (done)"),
          entry(ProfilerTask.BZL_LOAD_SKYFUNCTION, "bzl_load (total)"),
          entry(ProfilerTask.BZL_LOAD_SKYFUNCTION_DONE, "bzl_load (done)"),
          entry(ProfilerTask.GLOB_SKYFUNCTION, "glob (total)"),
          entry(ProfilerTask.GLOB_SKYFUNCTION_DONE, "glob (done)"),
          entry(ProfilerTask.GLOBS_SKYFUNCTION, "globs (total)"),
          entry(ProfilerTask.GLOBS_SKYFUNCTION_DONE, "globs (done)"));

  // Pick acceptable counter colors manually, unfortunately we have to pick from these
  // weird reserved names from
  // https://github.com/catapult-project/catapult/blob/master/tracing/tracing/base/color_scheme.html
  private static final ImmutableMap<ProfilerTask, String> COUNTER_TASK_TO_COLOR =
      ImmutableMap.ofEntries(
          entry(ProfilerTask.LOCAL_ACTION_COUNTS, "detailed_memory_dump"),
          entry(ProfilerTask.LOCAL_CPU_USAGE, "good"),
          entry(ProfilerTask.SYSTEM_CPU_USAGE, "rail_load"),
          entry(ProfilerTask.LOCAL_MEMORY_USAGE, "olive"),
          entry(ProfilerTask.SYSTEM_MEMORY_USAGE, "bad"),
          entry(ProfilerTask.SYSTEM_NETWORK_UP_USAGE, "rail_response"),
          entry(ProfilerTask.SYSTEM_NETWORK_DOWN_USAGE, "rail_response"),
          entry(ProfilerTask.WORKERS_MEMORY_USAGE, "rail_animation"),
          entry(ProfilerTask.SYSTEM_LOAD_AVERAGE, "generic_work"),
          entry(ProfilerTask.MEMORY_USAGE_ESTIMATION, "rail_idle"),
          entry(ProfilerTask.CPU_USAGE_ESTIMATION, "cq_build_attempt_passed"),
          entry(ProfilerTask.PRESSURE_STALL_FULL_IO, "rail_animation"),
          entry(ProfilerTask.PRESSURE_STALL_SOME_IO, "cq_build_attempt_failed"),
          entry(ProfilerTask.PRESSURE_STALL_FULL_MEMORY, "thread_state_unknown"),
          entry(ProfilerTask.PRESSURE_STALL_SOME_MEMORY, "rail_idle"),
          entry(ProfilerTask.PRESSURE_STALL_SOME_CPU, "thread_state_running"));
}
