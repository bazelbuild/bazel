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

import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** This class is used to insert metadata about threads into the JSON trace profile. */
class ThreadMetadata implements TraceData {
  private final String readableName;
  private final long threadId;
  private final long sortIndex;

  public ThreadMetadata(String readableName, long threadId, long sortIndex) {
    this.readableName = readableName;
    this.threadId = threadId;
    this.sortIndex = sortIndex;
  }

  public ThreadMetadata() {
    this.readableName = getReadableName(Thread.currentThread().getName());
    this.threadId = Thread.currentThread().getId();
    this.sortIndex = getSortIndex(Thread.currentThread().getName());
  }

  // The JVM never returns 0 as thread ID, so we use that as fake thread ID for the critical path.
  static final long CRITICAL_PATH_THREAD_ID = 0;

  // In order to display the actions on the critical path in a nice way, we fake a thread for them.
  public static ThreadMetadata createFakeThreadMetadataForCriticalPath() {
    return new ThreadMetadata("Critical Path", CRITICAL_PATH_THREAD_ID, CRITICAL_PATH_SORT_INDEX);
  }

  private static boolean isMainThread(String threadName) {
    return threadName.startsWith("grpc-command");
  }

  private static boolean isGCThread(String threadName) {
    return threadName.equals("Notification Thread");
  }

  private static String getReadableName(String threadName) {
    if (isMainThread(threadName)) {
      return "Main Thread";
    }

    if (isGCThread(threadName)) {
      return "Garbage Collector";
    }

    return threadName;
  }

  /**
   * These constants describe ranges of threads. We suppose that there are no more than 10_000
   * threads of each kind, otherwise the profile becomes unreadable anyway. So the sort index of
   * skyframe threads is in range [10_000..20_000) for example.
   */
  private static final long SKYFRAME_EVALUATOR_SHIFT = 10_000;

  private static final long SKYFRAME_EVALUATOR_CPU_HEAVY_SHIFT = 20_000;
  private static final long SKYFRAME_EVALUATOR_EXECUTION_SHIFT = 30_000;
  private static final long SKYFRAME_EVALUATOR_MEMOIZING_SHIFT = 40_000;

  private static final long DYNAMIC_EXECUTION_SHIFT = 100_000;
  private static final long INCLUDE_SCANNER_SHIFT = 200_000;

  private static final long CRITICAL_PATH_SORT_INDEX = 0;
  private static final long MAIN_THREAD_SORT_INDEX = 1;
  private static final long GC_THREAD_SORT_INDEX = 2;
  private static final long MAX_SORT_INDEX = 1_000_000;

  private static final Pattern NUMBER_PATTERN = Pattern.compile("\\d+");

  private static long getSortIndex(String threadName) {
    if (isMainThread(threadName)) {
      return MAIN_THREAD_SORT_INDEX;
    }

    if (isGCThread(threadName)) {
      return GC_THREAD_SORT_INDEX;
    }

    Matcher numberMatcher = NUMBER_PATTERN.matcher(threadName);
    if (!numberMatcher.find()) {
      return MAX_SORT_INDEX;
    }

    long extractedNumber;
    try {
      extractedNumber = Long.parseLong(numberMatcher.group());
    } catch (NumberFormatException e) {
      // If the number cannot be parsed, e.g. is larger than a long, the actual position is not
      // really relevant.
      return MAX_SORT_INDEX;
    }

    if (threadName.startsWith("skyframe-evaluator")) {
      if (threadName.startsWith("skyframe-evaluator-cpu-heavy")) {
        return SKYFRAME_EVALUATOR_CPU_HEAVY_SHIFT + extractedNumber;
      }
      if (threadName.startsWith("skyframe-evaluator-execution")) {
        return SKYFRAME_EVALUATOR_EXECUTION_SHIFT + extractedNumber;
      }
      if (threadName.startsWith("skyframe-evaluator-memoizing")) {
        return SKYFRAME_EVALUATOR_MEMOIZING_SHIFT + extractedNumber;
      }
      return SKYFRAME_EVALUATOR_SHIFT + extractedNumber;
    }

    if (threadName.startsWith("dynamic-execution")) {
      return DYNAMIC_EXECUTION_SHIFT + extractedNumber;
    }

    if (threadName.startsWith("Include scanner")) {
      return INCLUDE_SCANNER_SHIFT + extractedNumber;
    }

    return MAX_SORT_INDEX;
  }

  @Override
  public void writeTraceData(JsonWriter jsonWriter, long profileStartTimeNanos) throws IOException {
    jsonWriter.setIndent("  ");
    jsonWriter.beginObject();
    jsonWriter.setIndent("");
    jsonWriter.name("name").value("thread_name");
    jsonWriter.name("ph").value("M");
    jsonWriter.name("pid").value(1);
    jsonWriter.name("tid").value(threadId);
    jsonWriter.name("args");

    jsonWriter.beginObject();
    jsonWriter.name("name").value(readableName);
    jsonWriter.endObject();

    jsonWriter.endObject();

    jsonWriter.setIndent("  ");
    jsonWriter.beginObject();
    jsonWriter.setIndent("");
    jsonWriter.name("name").value("thread_sort_index");
    jsonWriter.name("ph").value("M");
    jsonWriter.name("pid").value(1);
    jsonWriter.name("tid").value(threadId);
    jsonWriter.name("args");

    jsonWriter.beginObject();
    jsonWriter.name("sort_index").value(sortIndex);
    jsonWriter.endObject();

    jsonWriter.endObject();
  }
}
