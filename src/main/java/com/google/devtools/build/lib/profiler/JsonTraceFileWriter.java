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
import com.google.devtools.build.lib.profiler.Profiler.ActionTaskData;
import com.google.devtools.build.lib.profiler.Profiler.CounterData;
import com.google.devtools.build.lib.profiler.Profiler.TaskData;
import com.google.gson.stream.JsonWriter;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Date;
import java.util.HashMap;
import java.util.UUID;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/** Writes the profile in Json Trace file format. */
class JsonTraceFileWriter implements Runnable {
  protected final BlockingQueue<TaskData> queue;
  protected final Thread thread;
  protected IOException savedException;

  private final OutputStream outStream;
  private final long profileStartTimeNanos;
  private final ThreadLocal<Boolean> metadataPosted = ThreadLocal.withInitial(() -> Boolean.FALSE);
  private final boolean slimProfile;
  private final UUID buildID;
  private final String outputBase;

  // The JVM never returns 0 as thread ID, so we use that as fake thread ID for the critical path.
  private static final long CRITICAL_PATH_THREAD_ID = 0;

  private static final long SLIM_PROFILE_EVENT_THRESHOLD = 10_000;
  private static final long SLIM_PROFILE_MAXIMAL_PAUSE_NS = Duration.ofMillis(100).toNanos();
  private static final long SLIM_PROFILE_MAXIMAL_DURATION_NS = Duration.ofMillis(250).toNanos();
  private static final Pattern NUMBER_PATTERN = Pattern.compile("\\d+");

  private static final TaskData POISON_PILL = new TaskData(0, 0, null, "poison pill");

  /**
   * These constants describe ranges of threads. We suppose that there are no more than 10_000
   * threads of each kind, otherwise the profile becomes unreadable anyway. So the sort index of
   * skyframe threads is in range [10_000..20_000) for example.
   */
  private static final long SKYFRAME_EVALUATOR_SHIFT = 10_000;

  private static final long DYNAMIC_EXECUTION_SHIFT = 20_000;
  private static final long INCLUDE_SCANNER_SHIFT = 30_000;

  private static final long CRITICAL_PATH_SORT_INDEX = 0;
  private static final long MAIN_THREAD_SORT_INDEX = 1;
  private static final long GC_THREAD_SORT_INDEX = 2;
  private static final long MAX_SORT_INDEX = 1_000_000;

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

  JsonTraceFileWriter(
      OutputStream outStream,
      long profileStartTimeNanos,
      boolean slimProfile,
      String outputBase,
      UUID buildID) {
    this.queue = new LinkedBlockingQueue<>();
    this.thread = new Thread(this, "profile-writer-thread");
    this.outStream = outStream;
    this.profileStartTimeNanos = profileStartTimeNanos;
    this.slimProfile = slimProfile;
    this.buildID = buildID;
    this.outputBase = outputBase;
  }

  public void shutdown() throws IOException {
    // Add poison pill to queue and then wait for writer thread to shut down.
    queue.add(POISON_PILL);
    try {
      thread.join();
    } catch (InterruptedException e) {
      thread.interrupt();
      Thread.currentThread().interrupt();
    }
    if (savedException != null) {
      throw savedException;
    }
  }

  public void start() {
    thread.start();
  }

  public void enqueue(TaskData data) {
    if (!metadataPosted.get()) {
      metadataPosted.set(Boolean.TRUE);
      // Create a TaskData object that is special-cased below.
      queue.add(
          new TaskData(
              /* id= */ 0,
              /* startTimeNanos= */ -1,
              ProfilerTask.THREAD_NAME,
              Thread.currentThread().getName()));
      queue.add(
          new TaskData(
              /* id= */ 0,
              /* startTimeNanos= */ -1,
              ProfilerTask.THREAD_SORT_INDEX,
              String.valueOf(getSortIndex(Thread.currentThread().getName()))));
    }
    queue.add(data);
  }

  private static final class MergedEvent {
    int count = 0;
    long startTimeNanos;
    long endTimeNanos;
    TaskData data;

    /*
     * Tries to merge an additional event, i.e. if the event is close enough to the already merged
     * event.
     *
     * Returns null, if merging was possible.
     * If not mergeable, returns the TaskData of the previously merged events and clears the
     * internal data structures.
     */
    @Nullable
    TaskData maybeMerge(TaskData data) {
      long startTimeNanos = data.startTimeNanos;
      long endTimeNanos = startTimeNanos + data.duration;
      if (count > 0 && startTimeNanos >= this.startTimeNanos && endTimeNanos <= this.endTimeNanos) {
        // Skips child tasks.
        return null;
      }
      if (count == 0) {
        this.data = data;
        this.startTimeNanos = startTimeNanos;
        this.endTimeNanos = endTimeNanos;
        count++;
        return null;
      } else if (startTimeNanos <= this.endTimeNanos + SLIM_PROFILE_MAXIMAL_PAUSE_NS) {
        this.endTimeNanos = endTimeNanos;
        count++;
        return null;
      } else {
        TaskData ret = getAndReset();
        this.startTimeNanos = startTimeNanos;
        this.endTimeNanos = endTimeNanos;
        this.data = data;
        count = 1;
        return ret;
      }
    }

    // Returns a TaskData object representing the merged data and clears internal data structures.
    TaskData getAndReset() {
      TaskData ret;
      if (data == null || count <= 1) {
        ret = data;
      } else {
        ret =
            new TaskData(
                data.threadId,
                this.startTimeNanos,
                this.endTimeNanos - this.startTimeNanos,
                "merged " + count + " events");
      }
      count = 0;
      data = null;
      return ret;
    }
  }

  private void writeTask(JsonWriter writer, TaskData data) throws IOException {
    Preconditions.checkNotNull(data);
    String eventType = data.duration == 0 ? "i" : "X";
    writer.setIndent("  ");
    writer.beginObject();
    writer.setIndent("");
    if (data.type == null) {
      writer.setIndent("    ");
    } else {
      writer.name("cat").value(data.type.description);
    }
    writer.name("name").value(data.description);
    writer.name("ph").value(eventType);
    writer
        .name("ts")
        .value(TimeUnit.NANOSECONDS.toMicros(data.startTimeNanos - profileStartTimeNanos));
    if (data.duration != 0) {
      writer.name("dur").value(TimeUnit.NANOSECONDS.toMicros(data.duration));
    }
    writer.name("pid").value(1);

    if (data instanceof ActionTaskData) {
      ActionTaskData actionTaskData = (ActionTaskData) data;
      if (actionTaskData.primaryOutputPath != null) {
        // Primary outputs are non-mergeable, thus incompatible with slim profiles.
        writer.name("out").value(actionTaskData.primaryOutputPath);
      }
      if (actionTaskData.targetLabel != null) {
        writer.name("args");
        writer.beginObject();
        writer.name("target").value(actionTaskData.targetLabel);
        if (data.mnemonic.hasBeenSet()) {
          writer.name("mnemonic").value(data.mnemonic.getValueForJson());
        }
        writer.endObject();
      }
      if (data.mnemonic.hasBeenSet()) {
        writer.name("args");
        writer.beginObject();
        writer.name("mnemonic").value(data.mnemonic.getValueForJson());
        writer.endObject();
      }
    }
    if (data.type == ProfilerTask.CRITICAL_PATH_COMPONENT) {
      writer.name("args");
      writer.beginObject();
      writer.name("tid").value(data.threadId);
      writer.endObject();
    }
    long threadId =
        data.type == ProfilerTask.CRITICAL_PATH_COMPONENT ? CRITICAL_PATH_THREAD_ID : data.threadId;
    writer.name("tid").value(threadId);
    writer.endObject();
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

  private static boolean isMainThread(String threadName) {
    return threadName.startsWith("grpc-command");
  }

  private static boolean isGCThread(String threadName) {
    return threadName.equals("Service Thread");
  }

  /**
   * Saves all gathered information from taskQueue queue to the file. Method is invoked internally
   * by the Timer-based thread and at the end of profiling session.
   */
  @Override
  public void run() {
    try {
      boolean receivedPoisonPill = false;
      try (JsonWriter writer =
          new JsonWriter(
              // The buffer size of 262144 is chosen at random.
              new OutputStreamWriter(
                  new BufferedOutputStream(outStream, 262144), StandardCharsets.UTF_8))) {
        writer.beginObject();
        writer.name("otherData");
        writer.beginObject();
        writer.name("build_id").value(buildID.toString());
        writer.name("output_base").value(outputBase);
        writer.name("date").value(new Date().toString());
        writer.endObject();
        writer.name("traceEvents");
        writer.beginArray();
        TaskData data;

        // Generate metadata event for the critical path as thread 0 in disguise.
        writer.setIndent("  ");
        writer.beginObject();
        writer.setIndent("");
        writer.name("name").value("thread_name");
        writer.name("ph").value("M");
        writer.name("pid").value(1);
        writer.name("tid").value(CRITICAL_PATH_THREAD_ID);
        writer.name("args");
        writer.beginObject();
        writer.name("name").value("Critical Path");
        writer.endObject();
        writer.endObject();

        writer.setIndent("  ");
        writer.beginObject();
        writer.setIndent("");
        writer.name("name").value("thread_sort_index");
        writer.name("ph").value("M");
        writer.name("pid").value(1);
        writer.name("tid").value(CRITICAL_PATH_THREAD_ID);
        writer.name("args");
        writer.beginObject();
        writer.name("sort_index").value(String.valueOf(CRITICAL_PATH_SORT_INDEX));
        writer.endObject();
        writer.endObject();

        HashMap<Long, MergedEvent> eventsPerThread = new HashMap<>();
        int eventCount = 0;
        while ((data = queue.take()) != POISON_PILL) {
          Preconditions.checkNotNull(data);
          eventCount++;
          if (data.type == ProfilerTask.THREAD_NAME) {
            writer.setIndent("  ");
            writer.beginObject();
            writer.setIndent("");
            writer.name("name").value("thread_name");
            writer.name("ph").value("M");
            writer.name("pid").value(1);
            writer.name("tid").value(data.threadId);
            writer.name("args");

            writer.beginObject();
            writer.name("name").value(getReadableName(data.description));
            writer.endObject();

            writer.endObject();
            continue;
          }

          if (data.type == ProfilerTask.THREAD_SORT_INDEX) {
            writer.setIndent("  ");
            writer.beginObject();
            writer.setIndent("");
            writer.name("name").value("thread_sort_index");
            writer.name("ph").value("M");
            writer.name("pid").value(1);
            writer.name("tid").value(data.threadId);
            writer.name("args");

            writer.beginObject();
            writer.name("sort_index").value(data.description);
            writer.endObject();

            writer.endObject();
            continue;
          }

          if (COUNTER_TASK_TO_SERIES_NAME.containsKey(data.type)) {
            Preconditions.checkArgument(data instanceof CounterData);
            CounterData counterData = (CounterData) data;
            // Skip counts equal to zero. They will show up as a thin line in the profile.
            if (Math.abs(counterData.getCounterValue()) <= 0.00001) {
              continue;
            }
            writer.setIndent("  ");
            writer.beginObject();
            writer.setIndent("");
            writer.name("name").value(data.type.description);

            // Pick acceptable counter colors manually, unfortunately we have to pick from these
            // weird reserved names from
            // https://github.com/catapult-project/catapult/blob/master/tracing/tracing/base/color_scheme.html
            if (COUNTER_TASK_TO_COLOR.containsKey(data.type)) {
              writer.name("cname").value(COUNTER_TASK_TO_COLOR.get(data.type));
            }
            writer.name("ph").value("C");
            writer
                .name("ts")
                .value(TimeUnit.NANOSECONDS.toMicros(data.startTimeNanos - profileStartTimeNanos));
            writer.name("pid").value(1);
            writer.name("tid").value(data.threadId);
            writer.name("args");

            writer.beginObject();
            writer
                .name(COUNTER_TASK_TO_SERIES_NAME.get(data.type))
                .value(counterData.getCounterValue());
            writer.endObject();

            writer.endObject();
            continue;
          }
          if (slimProfile
              && eventCount > SLIM_PROFILE_EVENT_THRESHOLD
              && data.duration > 0
              && data.duration < SLIM_PROFILE_MAXIMAL_DURATION_NS
              && data.type != ProfilerTask.CRITICAL_PATH_COMPONENT) {
            eventsPerThread.putIfAbsent(data.threadId, new MergedEvent());
            TaskData taskData = eventsPerThread.get(data.threadId).maybeMerge(data);
            if (taskData != null) {
              writeTask(writer, taskData);
            }
          } else {
            writeTask(writer, data);
          }
        }
        for (JsonTraceFileWriter.MergedEvent value : eventsPerThread.values()) {
          TaskData taskData = value.getAndReset();
          if (taskData != null) {
            writeTask(writer, taskData);
          }
        }
        receivedPoisonPill = true;
        writer.setIndent("  ");
        writer.endArray();
        writer.endObject();
      } catch (IOException e) {
        this.savedException = e;
        if (!receivedPoisonPill) {
          while (queue.take() != POISON_PILL) {
            // We keep emptying the queue, but we can't write anything.
          }
        }
      }
    } catch (InterruptedException e) {
      // Exit silently.
    }
  }
}
