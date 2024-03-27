// Copyright 2022 The Bazel Authors. All rights reserved.
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

import static java.util.Map.entry;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;

/**
 * Used to inject a whole counter series (or even multiple) in one go in the JSON trace profile;
 * these could be used to represent CPU or memory usages over the course of an invocation (as
 * opposed to individual tasks such as executing an action).
 */
final class CounterSeriesTraceData implements TraceData {
  @VisibleForTesting static final long PROCESS_ID = 1;
  private final Map<ProfilerTask, double[]> counterSeriesMap;
  private final Duration profileStart;
  private final Duration bucketDuration;
  private final int len;
  private final long threadId;
  private String displayName;

  @Nullable private String colorName;

  /**
   * If multiple series are passed: - they will be rendered as stacked chart; - we assume they all
   * have the same length; - display name and color are picked from the first profile task in the
   * map. However, colors the remaining series are picked arbitrarily by the Trace renderer.
   */
  CounterSeriesTraceData(
      Map<ProfilerTask, double[]> counterSeriesMap,
      Duration profileStart,
      Duration bucketDuration) {
    Integer len = null;
    for (ProfilerTask profilerTask : counterSeriesMap.keySet()) {
      Preconditions.checkState(COUNTER_TASK_TO_SERIES_NAME.containsKey(profilerTask));
      if (len == null) {
        len = counterSeriesMap.get(profilerTask).length;

        this.displayName = profilerTask.description;

        // Pick acceptable counter colors manually, unfortunately we have to pick from these
        // weird reserved names from
        // https://github.com/catapult-project/catapult/blob/master/tracing/tracing/base/color_scheme.html
        this.colorName = COUNTER_TASK_TO_COLOR.get(profilerTask);
      } else {
        Preconditions.checkState(len.equals(counterSeriesMap.get(profilerTask).length));
      }
    }
    this.len = len;
    this.threadId = Thread.currentThread().getId();
    this.counterSeriesMap = counterSeriesMap;
    this.profileStart = profileStart;
    this.bucketDuration = bucketDuration;

  }

  // Pick acceptable counter colors manually, unfortunately we have to pick from these
  // weird reserved names from
  // https://github.com/catapult-project/catapult/blob/master/tracing/tracing/base/color_scheme.html
  private static final ImmutableMap<ProfilerTask, String> COUNTER_TASK_TO_COLOR =
      ImmutableMap.ofEntries(
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

  private static final ImmutableMap<ProfilerTask, String> COUNTER_TASK_TO_SERIES_NAME =
      ImmutableMap.ofEntries(
          entry(ProfilerTask.ACTION_COUNTS, "action"),
          entry(ProfilerTask.ACTION_CACHE_COUNTS, "local action cache"),
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
          entry(ProfilerTask.PRESSURE_STALL_SOME_CPU, "cpu pressure (some)"));

  @Override
  public void writeTraceData(JsonWriter jsonWriter, long profileStartTimeNanos) throws IOException {
    // See
    // https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#heading=h.msg3086636uq
    // for how counter series are represented in the Chrome Trace Event format.
    boolean recorded = false;
    for (int i = 0; i < len; i++) {
      long timeNanos = profileStart.plus(bucketDuration.multipliedBy(i)).toNanos();
      jsonWriter.setIndent("  ");
      jsonWriter.beginObject();
      jsonWriter.setIndent("");
      jsonWriter.name("name").value(displayName);
      jsonWriter.name("pid").value(PROCESS_ID);
      jsonWriter.name("tid").value(threadId);
      if (colorName != null) {
        jsonWriter.name("cname").value(colorName);
      }
      jsonWriter.name("ph").value("C");
      jsonWriter.name("ts").value(TimeUnit.NANOSECONDS.toMicros(timeNanos - profileStartTimeNanos));
      jsonWriter.name("args");

      jsonWriter.beginObject();
      for (ProfilerTask profilerTask : counterSeriesMap.keySet()) {
        double value = counterSeriesMap.get(profilerTask)[i];
        // Skip counts equal to zero. They will show up as a thin line in the profile. Once we
        // record the profile task we need to post it until the end.
        if (Math.abs(value) > 0.00001 || recorded) {
          jsonWriter.name(COUNTER_TASK_TO_SERIES_NAME.get(profilerTask)).value(value);
          recorded = true;
        }
      }
      jsonWriter.endObject();

      jsonWriter.endObject();
    }
  }
}
