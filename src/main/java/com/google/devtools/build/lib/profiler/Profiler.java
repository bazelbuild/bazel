// Copyright 2014 The Bazel Authors. All rights reserved.
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


import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ResourceEstimator;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.collect.Extrema;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.profiler.PredicateBasedStatRecorder.RecorderAndPredicate;
import com.google.devtools.build.lib.profiler.StatRecorder.VfsHeuristics;
import com.google.devtools.build.lib.worker.WorkerMetricsCollector;
import com.google.gson.stream.JsonWriter;
import com.sun.management.OperatingSystemMXBean;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.management.ManagementFactory;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;
import java.util.zip.GZIPOutputStream;
import javax.annotation.Nullable;

/**
 * Blaze internal profiler. Provides facility to report various Blaze tasks and store them
 * (asynchronously) in the file for future analysis.
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
 * <p>
 *
 * @see ProfilerTask enum for recognized task types.
 */
@ThreadSafe
public final class Profiler {
  /** The profiler (a static singleton instance). Inactive by default. */
  private static final Profiler instance = new Profiler();

  private static final int HISTOGRAM_BUCKETS = 20;

  private static final Duration ACTION_COUNT_BUCKET_DURATION = Duration.ofMillis(200);

  /** File format enum. */
  public enum Format {
    JSON_TRACE_FILE_FORMAT,
    JSON_TRACE_FILE_COMPRESSED_FORMAT
  }

  /** A task that was very slow. */
  public static final class SlowTask implements Comparable<SlowTask> {
    final long durationNanos;
    final String description;
    final ProfilerTask type;

    private SlowTask(TaskData taskData) {
      this.durationNanos = taskData.duration;
      this.description = taskData.description;
      this.type = taskData.type;
    }

    @Override
    public int compareTo(SlowTask other) {
      long delta = durationNanos - other.durationNanos;
      if (delta < 0) { // Very clumsy
        return -1;
      } else if (delta > 0) {
        return 1;
      } else {
        return 0;
      }
    }

    public long getDurationNanos() {
      return durationNanos;
    }

    public String getDescription() {
      return description;
    }

    public ProfilerTask getType() {
      return type;
    }
  }

  /**
   * Container for the single task record.
   *
   * <p>Class itself is not thread safe, but all access to it from Profiler methods is.
   */
  @ThreadCompatible
  static class TaskData implements TraceData {
    final long threadId;
    final long startTimeNanos;
    final int id;
    final ProfilerTask type;
    final MnemonicData mnemonic;
    final String description;

    long duration;

    TaskData(
        int id,
        long startTimeNanos,
        ProfilerTask eventType,
        MnemonicData mnemonic,
        String description) {
      this.id = id;
      this.threadId = Thread.currentThread().getId();
      this.startTimeNanos = startTimeNanos;
      this.type = eventType;
      this.mnemonic = mnemonic;
      this.description = Preconditions.checkNotNull(description);
    }

    TaskData(int id, long startTimeNanos, ProfilerTask eventType, String description) {
      this(id, startTimeNanos, eventType, MnemonicData.getEmptyMnemonic(), description);
    }

    TaskData(long threadId, long startTimeNanos, long duration, String description) {
      this.id = -1;
      this.type = ProfilerTask.UNKNOWN;
      this.mnemonic = MnemonicData.getEmptyMnemonic();
      this.threadId = threadId;
      this.startTimeNanos = startTimeNanos;
      this.duration = duration;
      this.description = description;
    }

    @Override
    public String toString() {
      return "Thread " + threadId + ", task " + id + ", type " + type + ", " + description;
    }

    @Override
    public void writeTraceData(JsonWriter jsonWriter, long profileStartTimeNanos)
        throws IOException {
      String eventType = duration == 0 ? "i" : "X";
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
      if (duration != 0) {
        jsonWriter.name("dur").value(TimeUnit.NANOSECONDS.toMicros(duration));
      }
      jsonWriter.name("pid").value(1);

      if (this instanceof ActionTaskData) {
        ActionTaskData actionTaskData = (ActionTaskData) this;
        if (actionTaskData.primaryOutputPath != null) {
          // Primary outputs are non-mergeable, thus incompatible with slim profiles.
          jsonWriter.name("out").value(actionTaskData.primaryOutputPath);
        }
        if (actionTaskData.targetLabel != null) {
          jsonWriter.name("args");
          jsonWriter.beginObject();
          jsonWriter.name("target").value(actionTaskData.targetLabel);
          if (mnemonic.hasBeenSet()) {
            jsonWriter.name("mnemonic").value(mnemonic.getValueForJson());
          }
          jsonWriter.endObject();
        }
        if (mnemonic.hasBeenSet()) {
          jsonWriter.name("args");
          jsonWriter.beginObject();
          jsonWriter.name("mnemonic").value(mnemonic.getValueForJson());
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
  }

  /**
   * Similar to TaskData, specific for profiled actions. Depending on options, adds additional
   * action specific information such as primary output path and target label. This is only meant to
   * be used for ProfilerTask.ACTION.
   */
  static final class ActionTaskData extends TaskData {
    @Nullable final String primaryOutputPath;
    @Nullable final String targetLabel;

    ActionTaskData(
        int id,
        long startTimeNanos,
        ProfilerTask eventType,
        MnemonicData mnemonic,
        String description,
        @Nullable String primaryOutputPath,
        @Nullable String targetLabel) {
      super(id, startTimeNanos, eventType, mnemonic, description);
      this.primaryOutputPath = primaryOutputPath;
      this.targetLabel = targetLabel;
    }
  }

  /**
   * Aggregator class that keeps track of the slowest tasks of the specified type.
   *
   * <p><code>extremaAggregators</p> is sharded so that all threads need not compete for the same
   * lock if they do the same operation at the same time. Access to an individual {@link Extrema}
   * is synchronized on the {@link Extrema} instance itself.
   */
  private static final class SlowestTaskAggregator {
    private static final int SHARDS = 16;
    private static final int SIZE = 30;

    @SuppressWarnings({"unchecked", "rawtypes"})
    private final Extrema<SlowTask>[] extremaAggregators = new Extrema[SHARDS];

    SlowestTaskAggregator() {
      for (int i = 0; i < SHARDS; i++) {
        extremaAggregators[i] = Extrema.max(SIZE);
      }
    }

    // @ThreadSafe
    void add(TaskData taskData) {
      Extrema<SlowTask> extrema =
          extremaAggregators[(int) (Thread.currentThread().getId() % SHARDS)];
      synchronized (extrema) {
        extrema.aggregate(new SlowTask(taskData));
      }
    }

    // @ThreadSafe
    void clear() {
      for (int i = 0; i < SHARDS; i++) {
        Extrema<SlowTask> extrema = extremaAggregators[i];
        synchronized (extrema) {
          extrema.clear();
        }
      }
    }

    // @ThreadSafe
    Iterable<SlowTask> getSlowestTasks() {
      // This is slow, but since it only happens during the end of the invocation, it's OK.
      Extrema<SlowTask> mergedExtrema = Extrema.max(SIZE);
      for (int i = 0; i < SHARDS; i++) {
        Extrema<SlowTask> extrema = extremaAggregators[i];
        synchronized (extrema) {
          for (SlowTask task : extrema.getExtremeElements()) {
            mergedExtrema.aggregate(task);
          }
        }
      }
      return mergedExtrema.getExtremeElements();
    }
  }

  static final class CounterData extends TaskData {
    private final double counterValue;

    public CounterData(long timeNanos, ProfilerTask type, double counterValue) {
      super(/* id= */ -1, timeNanos, type, String.valueOf(counterValue));
      this.counterValue = counterValue;
    }

    public double getCounterValue() {
      return counterValue;
    }
  }

  private Clock clock;
  private ImmutableSet<ProfilerTask> profiledTasks;
  private volatile long profileStartTime;
  private volatile boolean recordAllDurations = false;
  private Duration profileCpuStartTime;

  /** This counter provides a unique id for every task, used to provide a parent/child relation. */
  private AtomicInteger taskId = new AtomicInteger();

  /**
   * The reference to the current writer, if any. If the referenced writer is null, then disk writes
   * are disabled. This can happen when slowest task recording is enabled.
   */
  private final AtomicReference<JsonTraceFileWriter> writerRef = new AtomicReference<>();

  private final SlowestTaskAggregator[] slowestTasks =
      new SlowestTaskAggregator[ProfilerTask.values().length];

  @VisibleForTesting
  final StatRecorder[] tasksHistograms = new StatRecorder[ProfilerTask.values().length];

  /** Thread that collects local cpu usage data (if enabled). */
  private CollectLocalResourceUsage resourceUsageThread;

  private TimeSeries actionCountTimeSeries;
  private TimeSeries actionCacheCountTimeSeries;
  private Duration actionCountStartTime;
  private boolean collectTaskHistograms;
  private boolean includePrimaryOutput;
  private boolean includeTargetLabel;

  private Profiler() {
    initHistograms();
    for (ProfilerTask task : ProfilerTask.values()) {
      if (task.collectsSlowestInstances) {
        slowestTasks[task.ordinal()] = new SlowestTaskAggregator();
      }
    }
  }

  private void initHistograms() {
    for (ProfilerTask task : ProfilerTask.values()) {
      if (task.isVfs()) {
        Map<String, ? extends Predicate<? super String>> vfsHeuristics =
            VfsHeuristics.vfsTypeHeuristics;
        List<RecorderAndPredicate> recorders = new ArrayList<>(vfsHeuristics.size());
        for (Map.Entry<String, ? extends Predicate<? super String>> e : vfsHeuristics.entrySet()) {
          recorders.add(
              new RecorderAndPredicate(
                  new SingleStatRecorder(task + " " + e.getKey(), HISTOGRAM_BUCKETS),
                  e.getValue()));
        }
        tasksHistograms[task.ordinal()] = new PredicateBasedStatRecorder(recorders);
      } else {
        tasksHistograms[task.ordinal()] = new SingleStatRecorder(task, HISTOGRAM_BUCKETS);
      }
    }
  }

  /**
   * Returns task histograms. This must be called between calls to {@link #start} and {@link #stop},
   * or the returned recorders are all empty. Note that the returned recorders may still be modified
   * concurrently (but at least they are thread-safe, so that's good).
   *
   * <p>The stat recorders are indexed by {@code ProfilerTask#ordinal}.
   */
  // TODO(ulfjack): This returns incomplete data by design. Maybe we should return the histograms on
  // stop instead? However, this is currently only called from one location in a module, and that
  // can't call stop itself. What to do?
  public synchronized ImmutableList<StatRecorder> getTasksHistograms() {
    return isActive() ? ImmutableList.copyOf(tasksHistograms) : ImmutableList.of();
  }

  public static Profiler instance() {
    return instance;
  }

  /**
   * Returns the nanoTime of the current profiler instance, or an arbitrary constant if not active.
   */
  public static long nanoTimeMaybe() {
    if (instance.isActive()) {
      return instance.clock.nanoTime();
    }
    return -1;
  }

  // Returns the elapsed wall clock time since the profile has been started or null if inactive.
  @Nullable
  public static Duration elapsedTimeMaybe() {
    if (instance.isActive()) {
      return Duration.ofNanos(instance.clock.nanoTime())
          .minus(Duration.ofNanos(instance.profileStartTime));
    }
    return null;
  }

  private static Duration getProcessCpuTime() {
    OperatingSystemMXBean bean =
        (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
    return Duration.ofNanos(bean.getProcessCpuTime());
  }

  // Returns the CPU time since the profile has been started or null if inactive.
  @Nullable
  public static Duration getProcessCpuTimeMaybe() {
    if (instance().isActive()) {
      return getProcessCpuTime().minus(instance().profileCpuStartTime);
    }
    return null;
  }

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
   * @param collectLoadAverage If true, collects system load average (as seen in uptime(1))
   */
  public synchronized void start(
      ImmutableSet<ProfilerTask> profiledTasks,
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
      boolean collectTaskHistograms,
      boolean collectWorkerDataInProfiler,
      boolean collectLoadAverage,
      boolean collectSystemNetworkUsage,
      boolean collectResourceEstimation,
      ResourceEstimator resourceEstimator,
      WorkerMetricsCollector workerMetricsCollector,
      BugReporter bugReporter)
      throws IOException {
    Preconditions.checkState(!isActive(), "Profiler already active");
    initHistograms();

    this.profiledTasks = profiledTasks;
    this.clock = clock;
    this.actionCountStartTime = Duration.ofNanos(clock.nanoTime());
    this.actionCountTimeSeries = new TimeSeries(actionCountStartTime, ACTION_COUNT_BUCKET_DURATION);
    this.actionCacheCountTimeSeries =
        new TimeSeries(actionCountStartTime, ACTION_COUNT_BUCKET_DURATION);
    this.collectTaskHistograms = collectTaskHistograms;
    this.includePrimaryOutput = includePrimaryOutput;
    this.includeTargetLabel = includeTargetLabel;

    // Reset state for the new profiling session.
    taskId.set(0);
    this.recordAllDurations = recordAllDurations;
    JsonTraceFileWriter writer = null;
    if (stream != null && format != null) {
      switch (format) {
        case JSON_TRACE_FILE_FORMAT:
          writer =
              new JsonTraceFileWriter(stream, execStartTimeNanos, slimProfile, outputBase, buildID);
          break;
        case JSON_TRACE_FILE_COMPRESSED_FORMAT:
          writer =
              new JsonTraceFileWriter(
                  new GZIPOutputStream(stream),
                  execStartTimeNanos,
                  slimProfile,
                  outputBase,
                  buildID);
      }
      writer.start();
    }
    this.writerRef.set(writer);

    // Activate profiler.
    profileStartTime = execStartTimeNanos;
    profileCpuStartTime = getProcessCpuTime();

    // Start collecting Bazel and system-wide CPU metric collection.
    resourceUsageThread =
        new CollectLocalResourceUsage(
            bugReporter,
            workerMetricsCollector,
            resourceEstimator,
            collectWorkerDataInProfiler,
            collectLoadAverage,
            collectSystemNetworkUsage,
            collectResourceEstimation);
    resourceUsageThread.setDaemon(true);
    resourceUsageThread.start();
  }

  /**
   * Returns task histograms. This must be called between calls to {@link #start} and {@link #stop},
   * or the returned list is empty.
   */
  // TODO(ulfjack): This returns incomplete data by design. Also see getTasksHistograms.
  public synchronized Iterable<SlowTask> getSlowestTasks() {
    List<Iterable<SlowTask>> slowestTasksByType = new ArrayList<>();

    for (SlowestTaskAggregator aggregator : slowestTasks) {
      if (aggregator != null) {
        slowestTasksByType.add(aggregator.getSlowestTasks());
      }
    }

    return Iterables.concat(slowestTasksByType);
  }

  private void collectActionCounts() {
    Duration endTime = Duration.ofNanos(clock.nanoTime());
    int len = (int) endTime.minus(actionCountStartTime).dividedBy(ACTION_COUNT_BUCKET_DURATION) + 1;
    Map<ProfilerTask, double[]> counterSeriesMap = new LinkedHashMap<>();
    if (actionCountTimeSeries != null) {
      double[] actionCountValues = actionCountTimeSeries.toDoubleArray(len);
      actionCountTimeSeries = null;
      counterSeriesMap.put(ProfilerTask.ACTION_COUNTS, actionCountValues);
    }
    if (actionCacheCountTimeSeries != null) {
      double[] actionCacheCountValues = actionCacheCountTimeSeries.toDoubleArray(len);
      actionCacheCountTimeSeries = null;
      counterSeriesMap.put(ProfilerTask.ACTION_CACHE_COUNTS, actionCacheCountValues);
    }
    if (!counterSeriesMap.isEmpty()) {
      instance.logCounters(counterSeriesMap, actionCountStartTime, ACTION_COUNT_BUCKET_DURATION);
    }
  }

  /**
   * Disable profiling and complete profile file creation. Subsequent calls to beginTask/endTask
   * will no longer be recorded in the profile.
   */
  public synchronized void stop() throws IOException {
    if (!isActive()) {
      return;
    }

    collectActionCounts();

    if (resourceUsageThread != null) {
      resourceUsageThread.stopCollecting();
      try {
        resourceUsageThread.join();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
      resourceUsageThread.logCollectedData();
      resourceUsageThread = null;
    }

    // Log a final event to update the duration of ProfilePhase.FINISH.
    logEvent(ProfilerTask.INFO, "Finishing");
    JsonTraceFileWriter writer = writerRef.getAndSet(null);
    if (writer != null) {
      writer.shutdown();
      writer = null;
    }
    Arrays.fill(tasksHistograms, null);
    profileStartTime = 0L;
    profileCpuStartTime = null;

    for (SlowestTaskAggregator aggregator : slowestTasks) {
      if (aggregator != null) {
        aggregator.clear();
      }
    }
  }

  /** Returns true iff profiling is currently enabled. */
  public boolean isActive() {
    return profileStartTime != 0L;
  }

  public boolean isProfiling(ProfilerTask type) {
    return profiledTasks.contains(type);
  }

  /**
   * Unless --record_full_profiler_data is given we drop small tasks and add their time to the
   * parents duration.
   */
  private boolean wasTaskSlowEnoughToRecord(ProfilerTask type, long duration) {
    return (recordAllDurations || duration >= type.minDuration);
  }

  /** Adds a whole action count series to the writer bypassing histogram and subtask creation. */
  public void logCounters(
      Map<ProfilerTask, double[]> counterSeriesMap,
      Duration profileStart,
      Duration bucketDuration) {
    JsonTraceFileWriter currentWriter = writerRef.get();
    if (isActive() && currentWriter != null) {
      CounterSeriesTraceData counterSeriesTraceData =
          new CounterSeriesTraceData(counterSeriesMap, profileStart, bucketDuration);
      currentWriter.enqueue(counterSeriesTraceData);
    }
  }

  /**
   * Adds task directly to the main queue bypassing task stack. Used for simple tasks that are known
   * to not have any subtasks.
   *
   * @param startTimeNanos task start time (obtained through {@link Profiler#nanoTimeMaybe()})
   * @param duration task duration
   * @param type task type
   * @param description task description. May be stored until end of build.
   */
  private void logTask(long startTimeNanos, long duration, ProfilerTask type, String description) {
    Preconditions.checkNotNull(description);
    Preconditions.checkState(!"".equals(description), "No description -> not helpful");
    if (duration < 0) {
      // See note in Clock#nanoTime, which is used by Profiler#nanoTimeMaybe.
      duration = 0;
    }

    StatRecorder statRecorder = tasksHistograms[type.ordinal()];
    if (collectTaskHistograms && statRecorder != null) {
      statRecorder.addStat((int) Duration.ofNanos(duration).toMillis(), description);
    }

    if (isActive() && startTimeNanos >= 0 && isProfiling(type)) {
      // Store instance fields as local variables so they are not nulled out from under us by
      // #clear.
      JsonTraceFileWriter currentWriter = writerRef.get();
      if (wasTaskSlowEnoughToRecord(type, duration)) {
        TaskData data = new TaskData(taskId.incrementAndGet(), startTimeNanos, type, description);
        data.duration = duration;
        if (currentWriter != null) {
          currentWriter.enqueue(data);
        }

        SlowestTaskAggregator aggregator = slowestTasks[type.ordinal()];

        if (aggregator != null) {
          aggregator.add(data);
        }
      }
    }
  }

  /**
   * Used externally to submit simple task (one that does not have any subtasks). Depending on the
   * minDuration attribute of the task type, task may be just aggregated into the parent task and
   * not stored directly.
   *
   * @param startTimeNanos task start time (obtained through {@link Profiler#nanoTimeMaybe()})
   * @param type task type
   * @param description task description. May be stored until the end of the build.
   */
  public void logSimpleTask(long startTimeNanos, ProfilerTask type, String description) {
    if (clock != null) {
      logTask(startTimeNanos, clock.nanoTime() - startTimeNanos, type, description);
    }
  }

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
  public void logSimpleTask(
      long startTimeNanos, long stopTimeNanos, ProfilerTask type, String description) {
    logTask(startTimeNanos, stopTimeNanos - startTimeNanos, type, description);
  }

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
  public void logSimpleTaskDuration(
      long startTimeNanos, Duration duration, ProfilerTask type, String description) {
    logTask(startTimeNanos, duration.toNanos(), type, description);
  }

  /** Used to log "events" happening at a specific time - tasks with zero duration. */
  public void logEventAtTime(long atTimeNanos, ProfilerTask type, String description) {
    logTask(atTimeNanos, 0, type, description);
  }

  /** Used to log "events" - tasks with zero duration. */
  @VisibleForTesting
  void logEvent(ProfilerTask type, String description) {
    logEventAtTime(clock.nanoTime(), type, description);
  }

  private SilentCloseable reallyProfile(ProfilerTask type, String description) {
    // ProfilerInfo.allTasksById is supposed to be an id -> Task map, but it is in fact a List,
    // which means that we cannot drop tasks to which we had already assigned ids. Therefore,
    // non-leaf tasks must not have a minimum duration. However, we don't quite consistently
    // enforce this, and Blaze only works because we happen not to add child tasks to those parent
    // tasks that have a minimum duration.
    TaskData taskData = new TaskData(taskId.incrementAndGet(), clock.nanoTime(), type, description);
    return () -> completeTask(taskData);
  }

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
  public SilentCloseable profile(ProfilerTask type, String description) {
    Preconditions.checkNotNull(description);
    return (isActive() && isProfiling(type)) ? reallyProfile(type, description) : NOP;
  }

  /**
   * Version of {@link #profile(ProfilerTask, String)} that avoids creating string unless actually
   * profiling.
   */
  public SilentCloseable profile(ProfilerTask type, Supplier<String> description) {
    return (isActive() && isProfiling(type))
        ? reallyProfile(type, Preconditions.checkNotNull(description.get()))
        : NOP;
  }

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
  public SilentCloseable profile(String description) {
    return profile(ProfilerTask.INFO, description);
  }

  /**
   * Similar to {@link #profile}, but specific to action-related events. Takes an extra argument:
   * primaryOutput.
   */
  public SilentCloseable profileAction(
      ProfilerTask type,
      String mnemonic,
      String description,
      String primaryOutput,
      String targetLabel) {
    Preconditions.checkNotNull(description);
    if (isActive() && isProfiling(type)) {
      TaskData taskData =
          new ActionTaskData(
              taskId.incrementAndGet(),
              clock.nanoTime(),
              type,
              new MnemonicData(mnemonic),
              description,
              includePrimaryOutput ? primaryOutput : null,
              includeTargetLabel ? targetLabel : null);
      return () -> completeTask(taskData);
    } else {
      return NOP;
    }
  }

  public SilentCloseable profileAction(
      ProfilerTask type, String description, String primaryOutput, String targetLabel) {
    return profileAction(type, null, description, primaryOutput, targetLabel);
  }

  private static final SilentCloseable NOP = () -> {};

  private boolean countAction(ProfilerTask type, TaskData taskData) {
    return type == ProfilerTask.ACTION
        || (type == ProfilerTask.INFO && "discoverInputs".equals(taskData.description));
  }

  /** Records the end of the task. */
  private void completeTask(TaskData data) {
    if (isActive()) {
      long endTime = clock.nanoTime();
      data.duration = endTime - data.startTimeNanos;
      boolean shouldRecordTask = wasTaskSlowEnoughToRecord(data.type, data.duration);
      JsonTraceFileWriter writer = writerRef.get();
      if (shouldRecordTask) {
        if (writer != null) {
          writer.enqueue(data);
        }
        if (actionCountTimeSeries != null && countAction(data.type, data)) {
          synchronized (this) {
            actionCountTimeSeries.addRange(
                Duration.ofNanos(data.startTimeNanos), Duration.ofNanos(endTime));
          }
        }
        if (actionCacheCountTimeSeries != null && data.type == ProfilerTask.ACTION_CHECK) {
          synchronized (this) {
            actionCacheCountTimeSeries.addRange(
                Duration.ofNanos(data.startTimeNanos), Duration.ofNanos(endTime));
          }
        }
        SlowestTaskAggregator aggregator = slowestTasks[data.type.ordinal()];
        if (aggregator != null) {
          aggregator.add(data);
        }
      }
    }
  }

  /** Convenience method to log phase marker tasks. */
  public void markPhase(ProfilePhase phase) throws InterruptedException {
    MemoryProfiler.instance().markPhase(phase);
    if (isActive() && isProfiling(ProfilerTask.PHASE)) {
      logEvent(ProfilerTask.PHASE, phase.description);
    }
  }
}
