// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;

/**
 * Container for the single task record.
 *
 * <p>Class itself is not thread safe, but all access to it from Profiler methods is.
 */
@ThreadCompatible
class TaskData implements TraceData {
  final long threadId;
  final long startTimeNanos;
  final ProfilerTask type;
  final String description;

  long durationNanos;

  TaskData(
      long threadId,
      long startTimeNanos,
      long durationNanos,
      ProfilerTask eventType,
      String description) {
    this.threadId = threadId;
    this.startTimeNanos = startTimeNanos;
    this.durationNanos = durationNanos;
    this.type = eventType;
    this.description = checkNotNull(description);
  }

  TaskData(long threadId, long startTimeNanos, ProfilerTask eventType, String description) {
    this(threadId, startTimeNanos, /* durationNanos= */ -1, eventType, description);
  }

  TaskData(long threadId, long startTimeNanos, long durationNanos, String description) {
    this.type = ProfilerTask.UNKNOWN;
    this.threadId = threadId;
    this.startTimeNanos = startTimeNanos;
    this.durationNanos = durationNanos;
    this.description = description;
  }

  @Override
  public String toString() {
    return "Thread " + threadId + ", type " + type + ", " + description;
  }

  @Override
  public void writeTraceData(JsonWriter jsonWriter, long profileStartTimeNanos) throws IOException {
    String eventType = durationNanos == 0 ? "i" : "X";
    jsonWriter.setIndent("  ");
    jsonWriter.beginObject();
    jsonWriter.setIndent("");
    if (type == null) {
      jsonWriter.setIndent("    ");
    } else {
      jsonWriter.name("cat").value(type.description);
    }
    jsonWriter.name("name").value(description);
    jsonWriter.name("ph").value(eventType);
    jsonWriter
        .name("ts")
        .value(TimeUnit.NANOSECONDS.toMicros(startTimeNanos - profileStartTimeNanos));
    if (durationNanos != 0) {
      jsonWriter.name("dur").value(TimeUnit.NANOSECONDS.toMicros(durationNanos));
    }
    jsonWriter.name("pid").value(1);

    if (this instanceof ActionTaskData actionTaskData) {
      if (actionTaskData.primaryOutputPath != null) {
        // Primary outputs are non-mergeable, thus incompatible with slim profiles.
        jsonWriter.name("out").value(actionTaskData.primaryOutputPath);
      }
      if (actionTaskData.targetLabel != null
          || actionTaskData.mnemonic != null
          || actionTaskData.configuration != null) {
        jsonWriter.name("args");
        jsonWriter.beginObject();
        if (actionTaskData.targetLabel != null) {
          jsonWriter.name("target").value(actionTaskData.targetLabel);
        }
        if (actionTaskData.mnemonic != null) {
          jsonWriter.name("mnemonic").value(actionTaskData.mnemonic);
        }
        if (actionTaskData.configuration != null) {
          jsonWriter.name("configuration").value(actionTaskData.configuration);
        }
        jsonWriter.endObject();
      }
    }
    if (type == ProfilerTask.CRITICAL_PATH_COMPONENT) {
      jsonWriter.name("args");
      jsonWriter.beginObject();
      jsonWriter.name("tid").value(threadId);
      jsonWriter.endObject();
    }
    jsonWriter
        .name("tid")
        .value(
            type == ProfilerTask.CRITICAL_PATH_COMPONENT
                ? ThreadMetadata.CRITICAL_PATH_THREAD_ID
                : threadId);
    jsonWriter.endObject();
  }

  /**
   * Similar to TaskData, specific for profiled actions. Depending on options, adds additional
   * action specific information such as primary output path and target label. This is only meant to
   * be used for ProfilerTask.ACTION.
   */
  static final class ActionTaskData extends TaskData {
    @Nullable final String primaryOutputPath;
    @Nullable final String targetLabel;
    @Nullable final String mnemonic;
    @Nullable final String configuration;

    ActionTaskData(
        long threadId,
        long startTimeNanos,
        long durationNanos,
        ProfilerTask eventType,
        @Nullable String mnemonic,
        String description,
        @Nullable String primaryOutputPath,
        @Nullable String targetLabel,
        @Nullable String configuration) {
      super(threadId, startTimeNanos, durationNanos, eventType, description);
      this.primaryOutputPath = primaryOutputPath;
      this.targetLabel = targetLabel;
      this.mnemonic = mnemonic;
      this.configuration = configuration;
    }
  }
}
