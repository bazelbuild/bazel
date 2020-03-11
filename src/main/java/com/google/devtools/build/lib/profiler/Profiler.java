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

import static com.google.devtools.build.lib.profiler.ProfilerTask.TASK_COUNT;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.collect.Extrema;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.profiler.PredicateBasedStatRecorder.RecorderAndPredicate;
import com.google.devtools.build.lib.profiler.StatRecorder.VfsHeuristics;
import com.google.gson.stream.JsonWriter;
import com.sun.management.OperatingSystemMXBean;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;
import java.util.zip.GZIPOutputStream;

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
  private static final Logger logger = Logger.getLogger(Profiler.class.getName());

  /** The profiler (a static singleton instance). Inactive by default. */
  private static final Profiler instance = new Profiler();

  private static final int HISTOGRAM_BUCKETS = 20;

  private static final TaskData POISON_PILL = new TaskData(0, 0, null, null, "poison pill");

  private static final long ACTION_COUNT_BUCKET_MS = 200;

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
      if (delta < 0) {  // Very clumsy
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
   * Container for the single task record. Should never be instantiated directly - use
   * TaskStack.create() instead.
   *
   * <p>Class itself is not thread safe, but all access to it from Profiler methods is.
   */
  @ThreadCompatible
  private static class TaskData {
    final long threadId;
    final long startTimeNanos;
    final int id;
    final int parentId;
    final ProfilerTask type;
    final String description;

    long duration;
    int[] counts; // number of invocations per ProfilerTask type
    long[] durations; // time spend in the task per ProfilerTask type

    TaskData(
        int id, long startTimeNanos, TaskData parent, ProfilerTask eventType, String description) {
      this.id = id;
      this.threadId = Thread.currentThread().getId();
      this.parentId = (parent == null  ? 0 : parent.id);
      this.startTimeNanos = startTimeNanos;
      this.type = eventType;
      this.description = Preconditions.checkNotNull(description);
    }

    TaskData(long threadId, long startTimeNanos, long duration, String description) {
      this.id = -1;
      this.parentId = 0;
      this.type = ProfilerTask.UNKNOWN;
      this.threadId = threadId;
      this.startTimeNanos = startTimeNanos;
      this.duration = duration;
      this.description = description;
    }

    /** Aggregates information about an *immediate* subtask. */
    public void aggregateChild(ProfilerTask type, long duration) {
      int index = type.ordinal();
      if (counts == null) {
        // one entry for each ProfilerTask type
        counts = new int[TASK_COUNT];
        durations = new long[TASK_COUNT];
      }
      counts[index]++;
      durations[index] += duration;
    }

    @Override
    public String toString() {
      return "Thread " + threadId + ", task " + id + ", type " + type + ", " + description;
    }
  }

  private static final class ActionTaskData extends TaskData {
    final String primaryOutputPath;

    ActionTaskData(
        int id,
        long startTimeNanos,
        TaskData parent,
        ProfilerTask eventType,
        String description,
        String primaryOutputPath) {
      super(id, startTimeNanos, parent, eventType, description);
      this.primaryOutputPath = primaryOutputPath;
    }
  }

  /**
   * Tracks nested tasks for each thread.
   *
   * java.util.ArrayDeque is the most efficient stack implementation in the
   * Java Collections Framework (java.util.Stack class is older synchronized
   * alternative). It is, however, used here strictly for LIFO operations.
   * However, ArrayDeque is 1.6 only. For 1.5 best approach would be to utilize
   * ArrayList and emulate stack using it.
   */
  @ThreadSafe
  private final class TaskStack extends ThreadLocal<List<TaskData>> {
    @Override
    public List<TaskData> initialValue() {
      return new ArrayList<>();
    }

    public TaskData peek() {
      List<TaskData> list = get();
      if (list.isEmpty()) {
        return null;
      }
      return list.get(list.size() - 1);
    }

    public TaskData pop() {
      List<TaskData> list = get();
      return list.remove(list.size() - 1);
    }

    public boolean isEmpty() {
      return get().isEmpty();
    }

    public void push(ProfilerTask eventType, String description) {
      get().add(create(clock.nanoTime(), eventType, description));
    }

    public TaskData create(long startTimeNanos, ProfilerTask eventType, String description) {
      return new TaskData(taskId.incrementAndGet(), startTimeNanos, peek(), eventType, description);
    }

    public void pushActionTask(ProfilerTask eventType, String description, String primaryOutput) {
      get().add(createActionTask(clock.nanoTime(), eventType, description, primaryOutput));
    }

    public ActionTaskData createActionTask(
        long startTimeNanos, ProfilerTask eventType, String description, String primaryOutput) {
      return new ActionTaskData(
          taskId.incrementAndGet(), startTimeNanos, peek(), eventType, description, primaryOutput);
    }

    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder(
          "Current task stack for thread " + Thread.currentThread().getName() + ":\n");
      List<TaskData> list = get();
      for (int i = list.size() - 1; i >= 0; i--) {
        builder.append(list.get(i));
        builder.append("\n");
      }
      return builder.toString();
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
    private final int size;

    @SuppressWarnings({"unchecked", "rawtypes"})
    private final Extrema<SlowTask>[] extremaAggregators = new Extrema[SHARDS];

    SlowestTaskAggregator(int size) {
      this.size = size;

      for (int i = 0; i < SHARDS; i++) {
        extremaAggregators[i] = Extrema.max(size);
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
      // This is slow, but since it only happens during the end of the invocation, it's OK
      Extrema<SlowTask> mergedExtrema = Extrema.max(size);
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
  private AtomicReference<FileWriter> writerRef = new AtomicReference<>();

  /**
   * This is a per-thread data structure that's used to track the current stack of open tasks, the
   * purpose of which is to track the parent id of every task. This is also used to ensure that
   * {@link #profile} and {@link #completeTask} calls always occur in pairs.
   */
  // TODO(ulfjack): We can infer the parent/child relationship after the fact instead of tracking it
  // at runtime. That would allow us to remove this data structure entirely.
  private TaskStack taskStack;

  private final SlowestTaskAggregator[] slowestTasks =
      new SlowestTaskAggregator[ProfilerTask.values().length];

  private final StatRecorder[] tasksHistograms = new StatRecorder[ProfilerTask.values().length];

  /** Thread that collects local cpu usage data (if enabled). */
  private CollectLocalCpuUsage cpuUsageThread;

  private TimeSeries actionCountTimeSeries;
  private long actionCountStartTime;

  private Profiler() {
    initHistograms();
    for (ProfilerTask task : ProfilerTask.values()) {
      if (task.slowestInstancesCount != 0) {
        slowestTasks[task.ordinal()] = new SlowestTaskAggregator(task.slowestInstancesCount);
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
          recorders.add(new RecorderAndPredicate(
              new SingleStatRecorder(task + " " + e.getKey(), HISTOGRAM_BUCKETS), e.getValue()));
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
  public ImmutableList<StatRecorder> getTasksHistograms() {
    return ImmutableList.copyOf(tasksHistograms);
  }

  public static Profiler instance() {
    return instance;
  }

  /**
   * Returns the nanoTime of the current profiler instance, or an arbitrary
   * constant if not active.
   */
  public static long nanoTimeMaybe() {
    if (instance.isActive()) {
      return instance.clock.nanoTime();
    }
    return -1;
  }

  // Returns the elapsed wall clock time since the profile has been started or null if inactive.
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
      boolean enabledCpuUsageProfiling,
      boolean slimProfile,
      boolean enableActionCountProfile,
      boolean includePrimaryOutput)
      throws IOException {
    Preconditions.checkState(!isActive(), "Profiler already active");
    initHistograms();

    this.profiledTasks = profiledTasks;
    this.clock = clock;
    if (enableActionCountProfile) {
      this.actionCountStartTime = clock.nanoTime();
      this.actionCountTimeSeries =
          new TimeSeries(Duration.ofNanos(actionCountStartTime).toMillis(), ACTION_COUNT_BUCKET_MS);
    }

    // sanity check for current limitation on the number of supported types due
    // to using enum.ordinal() to store them instead of EnumSet for performance reasons.
    Preconditions.checkState(TASK_COUNT < 256,
        "The profiler implementation supports only up to 255 different ProfilerTask values.");

    // reset state for the new profiling session
    taskId.set(0);
    this.recordAllDurations = recordAllDurations;
    this.taskStack = new TaskStack();
    FileWriter writer = null;
    if (stream != null && format != null) {
      switch (format) {
        case JSON_TRACE_FILE_FORMAT:
          writer =
              new JsonTraceFileWriter(
                  stream,
                  execStartTimeNanos,
                  slimProfile,
                  outputBase,
                  buildID,
                  includePrimaryOutput);
          break;
        case JSON_TRACE_FILE_COMPRESSED_FORMAT:
          writer =
              new JsonTraceFileWriter(
                  new GZIPOutputStream(stream),
                  execStartTimeNanos,
                  slimProfile,
                  outputBase,
                  buildID,
                  includePrimaryOutput);
      }
      writer.start();
    }
    this.writerRef.set(writer);

    // activate profiler
    profileStartTime = execStartTimeNanos;
    profileCpuStartTime = getProcessCpuTime();

    if (enabledCpuUsageProfiling) {
      cpuUsageThread = new CollectLocalCpuUsage();
      cpuUsageThread.setDaemon(true);
      cpuUsageThread.start();
    }
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
    if (actionCountTimeSeries != null) {
      long endTimeMillis = Duration.ofNanos(clock.nanoTime()).toMillis();
      long profileStartMillis = Duration.ofNanos(actionCountStartTime).toMillis();
      int len = (int) ((endTimeMillis - profileStartMillis) / ACTION_COUNT_BUCKET_MS) + 1;
      double[] actionCountValues = actionCountTimeSeries.toDoubleArray(len);
      Profiler profiler = Profiler.instance();
      for (int i = 0; i < len; i++) {
        long timeMillis = profileStartMillis + i * ACTION_COUNT_BUCKET_MS;
        long timeNanos = TimeUnit.MILLISECONDS.toNanos(timeMillis);
        profiler.logEventAtTime(
            timeNanos, ProfilerTask.ACTION_COUNTS, String.valueOf(actionCountValues[i]));
      }
      actionCountTimeSeries = null;
    }
  }

  /**
   * Disable profiling and complete profile file creation.
   * Subsequent calls to beginTask/endTask will no longer
   * be recorded in the profile.
   */
  public synchronized void stop() throws IOException {
    if (!isActive()) {
      return;
    }

    collectActionCounts();

    if (cpuUsageThread != null) {
      cpuUsageThread.stopCollecting();
      try {
        cpuUsageThread.join();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
      cpuUsageThread.logCollectedData();
      cpuUsageThread = null;
    }

    // Log a final event to update the duration of ProfilePhase.FINISH.
    logEvent(ProfilerTask.INFO, "Finishing");
    FileWriter writer = writerRef.getAndSet(null);
    if (writer != null) {
      writer.shutdown();
      writer = null;
    }
    taskStack = null;
    initHistograms();
    profileStartTime = 0L;
    profileCpuStartTime = null;

    for (SlowestTaskAggregator aggregator : slowestTasks) {
      if (aggregator != null) {
        aggregator.clear();
      }
    }
  }

  /**
   *  Returns true iff profiling is currently enabled.
   */
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
    Preconditions.checkState(startTimeNanos > 0, "startTime was %s", startTimeNanos);
    Preconditions.checkState(!"".equals(description), "No description -> not helpful");
    if (duration < 0) {
      // See note in Clock#nanoTime, which is used by Profiler#nanoTimeMaybe.
      duration = 0;
    }

    tasksHistograms[type.ordinal()].addStat(
        (int) TimeUnit.NANOSECONDS.toMillis(duration), description);
    // Store instance fields as local variables so they are not nulled out from under us by #clear.
    TaskStack localStack = taskStack;
    FileWriter currentWriter = writerRef.get();
    if (localStack == null) {
      // Variables have been nulled out by #clear in between the check the caller made and this
      // point in the code. Probably due to an asynchronous crash.
      logger.severe("Variables null in profiler for " + type + ", probably due to async crash");
      return;
    }
    TaskData parent = localStack.peek();
    if (parent != null) {
      parent.aggregateChild(type, duration);
    }
    if (wasTaskSlowEnoughToRecord(type, duration)) {
      TaskData data = localStack.create(startTimeNanos, type, description);
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

  private boolean shouldProfile(long startTime, ProfilerTask type) {
    return isActive() && startTime > 0 && isProfiling(type);
  }

  /**
   * Used externally to submit simple task (one that does not have any subtasks). Depending on the
   * minDuration attribute of the task type, task may be just aggregated into the parent task and
   * not stored directly.
   *
   * @param startTime task start time (obtained through {@link Profiler#nanoTimeMaybe()})
   * @param type task type
   * @param description task description. May be stored until the end of the build.
   */
  public void logSimpleTask(long startTime, ProfilerTask type, String description) {
    if (shouldProfile(startTime, type)) {
      logTask(startTime, clock.nanoTime() - startTime, type, description);
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
    if (shouldProfile(startTimeNanos, type)) {
      logTask(startTimeNanos, stopTimeNanos - startTimeNanos, type, description);
    }
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
    if (shouldProfile(startTimeNanos, type)) {
      logTask(startTimeNanos, duration.toNanos(), type, description);
    }
  }

  /** Used to log "events" happening at a specific time - tasks with zero duration. */
  public void logEventAtTime(long atTimeNanos, ProfilerTask type, String description) {
    if (isActive() && isProfiling(type)) {
      logTask(atTimeNanos, 0, type, description);
    }
  }

  /** Used to log "events" - tasks with zero duration. */
  @VisibleForTesting
  void logEvent(ProfilerTask type, String description) {
    logEventAtTime(clock.nanoTime(), type, description);
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
   * <pre>
   * {@code
   * try (SilentCloseable c = Profiler.instance().profile(type, "description")) {
   *   // Your code here.
   * }
   * }
   * </pre>
   *
   * @param type predefined task type - see ProfilerTask for available types.
   * @param description task description. May be stored until the end of the build.
   */
  public SilentCloseable profile(ProfilerTask type, String description) {
    // ProfilerInfo.allTasksById is supposed to be an id -> Task map, but it is in fact a List,
    // which means that we cannot drop tasks to which we had already assigned ids. Therefore,
    // non-leaf tasks must not have a minimum duration. However, we don't quite consistently
    // enforce this, and Blaze only works because we happen not to add child tasks to those parent
    // tasks that have a minimum duration.
    Preconditions.checkNotNull(description);
    if (isActive() && isProfiling(type)) {
      taskStack.push(type, description);
      return () -> completeTask(type);
    } else {
      return () -> {};
    }
  }

  /**
   * Similar to {@link #profile}, but specific to action-related events. Takes an extra argument:
   * primaryOutput.
   */
  public SilentCloseable profileAction(
      ProfilerTask type, String description, String primaryOutput) {

    Preconditions.checkNotNull(description);
    if (isActive() && isProfiling(type)) {
      taskStack.pushActionTask(type, description, primaryOutput);
      return () -> completeTask(type);
    } else {
      return () -> {};
    }
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
   * <pre>
   * {@code
   * try (SilentCloseable c = Profiler.instance().profile("description")) {
   *   // Your code here.
   * }
   * }
   * </pre>
   *
   * @param description task description. May be stored until the end of the build.
   */
  public SilentCloseable profile(String description) {
    return profile(ProfilerTask.INFO, description);
  }

  private boolean countAction(ProfilerTask type, TaskData taskData) {
    return type == ProfilerTask.ACTION
        || (type == ProfilerTask.INFO && "discoverInputs".equals(taskData.description));
  }

  /**
   * Records the end of the task and moves tasks from the thread-local stack to
   * the main queue. Will validate that given task type matches task at the top
   * of the stack.
   *
   * @param type task type.
   */
  private void completeTask(ProfilerTask type) {
    if (isActive() && isProfiling(type)) {
      long endTime = clock.nanoTime();
      TaskData data = taskStack.pop();
      Preconditions.checkState(
          data.type == type,
          "Inconsistent Profiler.completeTask() call: should have been %s but got %s (%s, %s)",
          data.type,
          type,
          data,
          taskStack);
      data.duration = endTime - data.startTimeNanos;
      if (data.parentId > 0) {
        taskStack.peek().aggregateChild(data.type, data.duration);
      }
      boolean shouldRecordTask = wasTaskSlowEnoughToRecord(type, data.duration);
      FileWriter writer = writerRef.get();
      if ((shouldRecordTask || data.counts != null) && writer != null) {
        writer.enqueue(data);
      }

      if (shouldRecordTask) {
        if (actionCountTimeSeries != null && countAction(type, data)) {
          actionCountTimeSeries.addRange(
              Duration.ofNanos(data.startTimeNanos).toMillis(),
              Duration.ofNanos(endTime).toMillis());
        }
        SlowestTaskAggregator aggregator = slowestTasks[type.ordinal()];
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
      Preconditions.checkState(taskStack.isEmpty(), "Phase tasks must not be nested");
      logEvent(ProfilerTask.PHASE, phase.description);
    }
  }

  private abstract static class FileWriter implements Runnable {
    protected final BlockingQueue<TaskData> queue;
    protected final Thread thread;
    protected IOException savedException;

    FileWriter() {
      this.queue = new LinkedBlockingQueue<>();
      this.thread = new Thread(this, "profile-writer-thread");
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
      queue.add(data);
    }
  }

  /** Writes the profile in Json Trace file format. */
  private static class JsonTraceFileWriter extends FileWriter {
    private final OutputStream outStream;
    private final long profileStartTimeNanos;
    private final ThreadLocal<Boolean> metadataPosted =
        ThreadLocal.withInitial(() -> Boolean.FALSE);
    private final boolean slimProfile;
    private final boolean includePrimaryOutput;
    private final UUID buildID;
    private final String outputBase;

    // The JDK never returns 0 as thread id so we use that as fake thread id for the critical path.
    private static final long CRITICAL_PATH_THREAD_ID = 0;

    private static final long SLIM_PROFILE_EVENT_THRESHOLD = 10_000;
    private static final long SLIM_PROFILE_MAXIMAL_PAUSE_NS = Duration.ofMillis(100).toNanos();
    private static final long SLIM_PROFILE_MAXIMAL_DURATION_NS = Duration.ofMillis(250).toNanos();

    JsonTraceFileWriter(
        OutputStream outStream,
        long profileStartTimeNanos,
        boolean slimProfile,
        String outputBase,
        UUID buildID,
        boolean includePrimaryOutput) {
      this.outStream = outStream;
      this.profileStartTimeNanos = profileStartTimeNanos;
      this.slimProfile = slimProfile;
      this.buildID = buildID;
      this.outputBase = outputBase;
      this.includePrimaryOutput = includePrimaryOutput;
    }

    @Override
    public void enqueue(TaskData data) {
      if (!metadataPosted.get()) {
        metadataPosted.set(Boolean.TRUE);
        // Create a TaskData object that is special-cased below.
        queue.add(
            new TaskData(
                /* id= */ 0,
                /* startTimeNanos= */ -1,
                /* parent= */ null,
                ProfilerTask.THREAD_NAME,
                Thread.currentThread().getName()));
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
      TaskData maybeMerge(TaskData data) {
        long startTimeNanos = data.startTimeNanos;
        long endTimeNanos = startTimeNanos + data.duration;
        if (count > 0
            && startTimeNanos >= this.startTimeNanos
            && endTimeNanos <= this.endTimeNanos) {
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
        if (count <= 1) {
          ret = data;
        } else {
          if (data == null) {
            ret = data;
          }
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
      String eventType = data.duration == 0 ? "i" : "X";
      writer.setIndent("  ");
      writer.beginObject();
      writer.setIndent("");
      if (data == null || data.type == null) {
        writer.setIndent("    ");
      }
      writer.name("cat").value(data.type.description);
      writer.name("name").value(data.description);
      writer.name("ph").value(eventType);
      writer
          .name("ts")
          .value(TimeUnit.NANOSECONDS.toMicros(data.startTimeNanos - profileStartTimeNanos));
      if (data.duration != 0) {
        writer.name("dur").value(TimeUnit.NANOSECONDS.toMicros(data.duration));
      }
      writer.name("pid").value(1);

      // Primary outputs are non-mergeable, thus incompatible with slim profiles.
      if (includePrimaryOutput && data instanceof ActionTaskData) {
        writer.name("out").value(((ActionTaskData) data).primaryOutputPath);
      }
      long threadId =
          data.type == ProfilerTask.CRITICAL_PATH_COMPONENT
              ? CRITICAL_PATH_THREAD_ID
              : data.threadId;
      writer.name("tid").value(threadId);
      writer.endObject();
    }

    /**
     * Saves all gathered information from taskQueue queue to the file.
     * Method is invoked internally by the Timer-based thread and at the end of
     * profiling session.
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

          HashMap<Long, MergedEvent> eventsPerThread = new HashMap<>();
          int eventCount = 0;
          while ((data = queue.take()) != POISON_PILL) {
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
              writer.name("name").value(data.description);
              writer.endObject();

              writer.endObject();
              continue;
            }

            if (data.type == ProfilerTask.LOCAL_CPU_USAGE
                || data.type == ProfilerTask.ACTION_COUNTS) {
              // Skip counts equal to zero. They will show up as a thin line in the profile.
              if ("0.0".equals(data.description)) {
                continue;
              }
              writer.setIndent("  ");
              writer.beginObject();
              writer.setIndent("");
              writer.name("name").value(data.type.description);
              writer.name("ph").value("C");
              writer
                  .name("ts")
                  .value(
                      TimeUnit.NANOSECONDS.toMicros(data.startTimeNanos - profileStartTimeNanos));
              writer.name("pid").value(1);
              writer.name("tid").value(data.threadId);
              writer.name("args");

              writer.beginObject();
              writer.name("cpu").value(data.description);
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
          for (Profiler.JsonTraceFileWriter.MergedEvent value : eventsPerThread.values()) {
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
}
