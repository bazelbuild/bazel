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

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.runtime.BlazeService;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.io.OutputStream;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.function.Supplier;

/**
 * Interface for the Blaze internal profiler. Provides facility to report various Blaze tasks and
 * store them (asynchronously) in the file for future analysis.
 *
 * <p>Implemented as singleton so any caller should use Profiler.instance() to obtain reference.
 *
 * <p>Internally, profiler uses two data structures - ThreadLocal task stack to track nested tasks
 * and single ConcurrentLinkedQueue to gather all completed tasks.
 *
 * <p>Also, due to the nature of the provided functionality (instrumentation of all Blaze
 * components), build.lib.profiler package will be used by almost every other Blaze package, so
 * special attention should be paid to avoid any dependencies on the rest of the Blaze code,
 * including build.lib.util and build.lib.vfs. This is important because build.lib.util and
 * build.lib.vfs contain Profiler invocations and any dependency on those two packages would create
 * circular relationship.
 *
 * @see ProfilerTask enum for recognized task types.
 */
@SuppressWarnings("GoodTime") // This code is very performance sensitive.
public interface TraceProfilerService extends BlazeService {

  /** File format enum. */
  enum Format {
    JSON_TRACE_FILE_FORMAT,
    JSON_TRACE_FILE_COMPRESSED_FORMAT
  }

  /** Returns the nanoTime of the current profiler instance, or -1 if not active. */
  long nanoTimeMaybe();

  /** Returns true iff profiling is currently enabled. */
  boolean isActive();

  /**
   * Records the beginning of a task as specified, and returns a {@link SilentCloseable} instance
   * that ends the task. This lets the system do the work of ending the task, with the compiler
   * giving a warning if the returned instance is not closed.
   *
   * <p>Use of this method allows to support nested task monitoring. For tasks that are known to not
   * have any subtasks, logSimpleTask() should be used instead.
   *
   * <p>Use like this:
   *
   * <pre>{@code
   * try (SilentCloseable c = Profiler.instance().profile(type, "description")) {
   *   // Your code here.
   * }
   * }</pre>
   *
   * @param type predefined task type - see ProfilerTask for available types.
   * @param description task description. May be stored until the end of the build.
   */
  SilentCloseable profile(ProfilerTask type, String description);

  /**
   * Version of {@link #profile(ProfilerTask, String)} that avoids creating string unless actually
   * profiling.
   */
  SilentCloseable profile(ProfilerTask type, Supplier<String> description);

  /**
   * Records the beginning of a task as specified, and returns a {@link SilentCloseable} instance
   * that ends the task. This lets the system do the work of ending the task, with the compiler
   * giving a warning if the returned instance is not closed.
   *
   * <p>Use of this method allows to support nested task monitoring. For tasks that are known to not
   * have any subtasks, logSimpleTask() should be used instead.
   *
   * <p>This is a convenience method that uses {@link ProfilerTask#INFO}.
   *
   * <p>Use like this:
   *
   * <pre>{@code
   * try (SilentCloseable c = Profiler.instance().profile("description")) {
   *   // Your code here.
   * }
   * }</pre>
   *
   * @param description task description. May be stored until the end of the build.
   */
  SilentCloseable profile(String description);

  /**
   * Used externally to submit simple task (one that does not have any subtasks). Depending on the
   * minDuration attribute of the task type, task may be just aggregated into the parent task and
   * not stored directly.
   *
   * @param startTimeNanos task start time (obtained through {@link Profiler#nanoTimeMaybe()})
   * @param type task type
   * @param description task description. May be stored until the end of the build.
   */
  void logSimpleTask(long startTimeNanos, ProfilerTask type, String description);

  /**
   * Used externally to submit simple task (one that does not have any subtasks). Depending on the
   * minDuration attribute of the task type, task may be just aggregated into the parent task and
   * not stored directly.
   *
   * <p>Note that start and stop time must both be acquired from the same clock instance.
   *
   * @param startTimeNanos task start time
   * @param stopTimeNanos task stop time
   * @param type task type
   * @param description task description. May be stored until the end of the build.
   */
  void logSimpleTask(
      long startTimeNanos, long stopTimeNanos, ProfilerTask type, String description);

  /**
   * Used externally to submit simple task (one that does not have any subtasks). Depending on the
   * minDuration attribute of the task type, task may be just aggregated into the parent task and
   * not stored directly.
   *
   * @param startTimeNanos task start time (obtained through {@link Profiler#nanoTimeMaybe()})
   * @param duration the duration of the task
   * @param type task type
   * @param description task description. May be stored until the end of the build.
   */
  void logSimpleTaskDuration(
      long startTimeNanos, Duration duration, ProfilerTask type, String description);

  /** Used to log "events" happening at a specific time - tasks with zero duration. */
  void logEventAtTime(long atTimeNanos, ProfilerTask type, String description);

  /** Used to log "events" - tasks with zero duration. */
  void logEvent(ProfilerTask type, String description);

  /**
   * Enable profiling.
   *
   * <p>Subsequent calls to beginTask/endTask will be recorded in the provided output stream. Please
   * note that stream performance is extremely important and buffered streams should be utilized.
   *
   * @param profiledTasks which of {@link ProfilerTask}s to track
   * @param stream output stream to store profile data. Note: passing unbuffered stream object
   *     reference may result in significant performance penalties
   * @param recordAllDurations iff true, record all tasks regardless of their duration; otherwise
   *     some tasks may get aggregated if they finished quick enough
   * @param clock a {@code BlazeClock.instance()}
   * @param execStartTimeNanos execution start time in nanos obtained from {@code clock.nanoTime()}
   */
  void start(
      Set<ProfilerTask> profiledTasks,
      OutputStream stream,
      Format format,
      String outputBase,
      UUID buildID,
      boolean recordAllDurations,
      Clock clock,
      long execStartTimeNanos,
      boolean slimProfile,
      boolean includePrimaryOutput,
      boolean includeTargetLabel,
      boolean includeConfiguration,
      boolean collectTaskHistograms)
      throws IOException;

  /**
   * Disable profiling and complete profile file creation. Subsequent calls to beginTask/endTask
   * will no longer be recorded in the profile.
   */
  void stop() throws IOException;

  /**
   * Clears the records the profiler instance keeps.
   *
   * <p>Should always be called between a {@link #stop()} and a subsequent {@link #start}.
   */
  void clear();

  /**
   * Returns task histograms. This must be called between calls to {@link #start} and {@link #stop},
   * or the returned recorders are all empty. Note that the returned recorders may still be modified
   * concurrently (but at least they are thread-safe, so that's good).
   *
   * <p>The stat recorders are indexed by {@code ProfilerTask#ordinal}. //TODO(b/458037154): Maybe
   * make the enums stable.
   */
  List<StatRecorder> getTasksHistograms();

  /**
   * Returns task histograms. This must be called between calls to {@link #start} and {@link #stop},
   * or the returned list is empty.
   */
  Iterable<SlowTask> getSlowestTasks();

  boolean isProfiling(ProfilerTask type);

  /** Convenience method to log phase marker tasks. */
  void markPhase(ProfilePhase phase) throws InterruptedException;

  /**
   * Similar to {@link #profile}, but specific to action-related events. Takes an extra argument:
   * primaryOutput.
   */
  SilentCloseable profileAction(
      ProfilerTask type,
      String mnemonic,
      String description,
      String primaryOutput,
      String targetLabel,
      String configuration);

  /**
   * Records the end of a task as specified.
   *
   * @param startTimeNanos task start time
   * @param type task type
   * @param description task description
   */
  void completeTask(long startTimeNanos, ProfilerTask type, String description);

  void registerCounterSeriesCollector(CounterSeriesCollector collector);

  void unregisterCounterSeriesCollector(CounterSeriesCollector collector);

  /** Adds a whole action count series to the writer bypassing histogram and subtask creation. */
  void logCounters(
      Map<CounterSeriesTask, double[]> counterSeriesMap,
      Duration profileStart,
      Duration bucketDuration);

  Duration getProfileElapsedTime();

  Duration getServerProcessCpuTime();

  /**
   * Profiles a future.
   *
   * @param future the future to profile
   * @param prefix the prefix of the virtual lanes. Similar to the thread name prefix.
   * @param type task type
   * @param description task description. May be stored until the end of the build.
   */
  @CanIgnoreReturnValue
  <T> ListenableFuture<T> profileFuture(
      ListenableFuture<T> future, String prefix, ProfilerTask type, String description);

  /**
   * Creates a profiler that can be used to profile async operations of a task.
   *
   * @param prefix the prefix of the virtual lanes. Similar to the thread name prefix.
   * @param description the description of task.
   */
  AsyncProfiler profileAsync(String prefix, String description);

  /** Creates a time series with the given start time and bucket duration. */
  TimeSeries createTimeSeries(Duration startTime, Duration bucketDuration);
}
