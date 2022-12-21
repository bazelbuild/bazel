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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.time.Duration;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;

/**
 * Used to inject a whole counter series in one go in the JSON trace profile; these could be used to
 * represent CPU or memory usages over the course of an invocations (as opposed to individual tasks
 * such as executing an action).
 */
final class CounterSeriesTraceData implements TraceData {
  private final String shortName;
  private final String readableName;
  private final double[] counterValues;
  private final Duration profileStart;
  private final Duration bucketDuration;
  @Nullable private final String colorName;

  CounterSeriesTraceData(
      ProfilerTask profilerTask,
      double[] counterValues,
      Duration profileStart,
      Duration bucketDuration) {
    Preconditions.checkArgument(COUNTER_TASK_TO_SERIES_NAME.containsKey(profilerTask));
    this.shortName = profilerTask.description;
    this.readableName = COUNTER_TASK_TO_SERIES_NAME.get(profilerTask);
    this.counterValues = counterValues;
    this.profileStart = profileStart;
    this.bucketDuration = bucketDuration;

    // Pick acceptable counter colors manually, unfortunately we have to pick from these
    // weird reserved names from
    // https://github.com/catapult-project/catapult/blob/master/tracing/tracing/base/color_scheme.html
    this.colorName = COUNTER_TASK_TO_COLOR.get(profilerTask);
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
          entry(ProfilerTask.CPU_USAGE_ESTIMATION, "cq_build_attempt_passed"));

  private static final ImmutableMap<ProfilerTask, String> COUNTER_TASK_TO_SERIES_NAME =
      ImmutableMap.ofEntries(
          entry(ProfilerTask.ACTION_COUNTS, "action"),
          entry(ProfilerTask.LOCAL_CPU_USAGE, "cpu"),
          entry(ProfilerTask.SYSTEM_CPU_USAGE, "system cpu"),
          entry(ProfilerTask.LOCAL_MEMORY_USAGE, "memory"),
          entry(ProfilerTask.SYSTEM_MEMORY_USAGE, "system memory"),
          entry(ProfilerTask.SYSTEM_NETWORK_UP_USAGE, "system network up (Mbps)"),
          entry(ProfilerTask.SYSTEM_NETWORK_DOWN_USAGE, "system network down (Mbps)"),
          entry(ProfilerTask.WORKERS_MEMORY_USAGE, "workers memory"),
          entry(ProfilerTask.SYSTEM_LOAD_AVERAGE, "load"),
          entry(ProfilerTask.MEMORY_USAGE_ESTIMATION, "estimated memory"),
          entry(ProfilerTask.CPU_USAGE_ESTIMATION, "estimated cpu"));

  @Override
  public void writeTraceData(JsonWriter jsonWriter, long profileStartTimeNanos) throws IOException {
    // See https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#heading=h.msg3086636uq
    // for how counter series are represented in the Chrome Trace Event format.
    for (int i = 0; i < counterValues.length; i++) {
      long timeNanos = profileStart.plus(bucketDuration.multipliedBy(i)).toNanos();
      // Skip counts equal to zero. They will show up as a thin line in the profile.
      if (Math.abs(counterValues[i]) <= 0.00001) {
        continue;
      }
      jsonWriter.setIndent("  ");
      jsonWriter.beginObject();
      jsonWriter.setIndent("");
      jsonWriter.name("name").value(shortName);
      if (colorName != null) {
        jsonWriter.name("cname").value(colorName);
      }
      jsonWriter.name("ph").value("C");
      jsonWriter.name("ts").value(TimeUnit.NANOSECONDS.toMicros(timeNanos - profileStartTimeNanos));
      jsonWriter.name("args");

      jsonWriter.beginObject();
      jsonWriter.name(readableName).value(counterValues[i]);
      jsonWriter.endObject();

      jsonWriter.endObject();
    }
  }
}
