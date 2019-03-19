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
import com.google.devtools.build.lib.util.VarInt;
import com.google.gson.stream.JsonWriter;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;
import java.util.zip.Deflater;
import java.util.zip.DeflaterOutputStream;
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
 * <p>All gathered instrumentation data will be stored in the file. Please, note, that while file
 * format is described here it is considered internal and can change at any time. For scripting,
 * using blaze analyze-profile --dump=raw would be more robust and stable solution.
 *
 * <p>
 *
 * <pre>
 * Profiler file consists of the deflated stream with following overall structure:
 *   HEADER
 *   TASK_TYPE_TABLE
 *   TASK_RECORD...
 *   EOF_MARKER
 *
 * HEADER:
 *   int32: magic token (Profiler.MAGIC)
 *   int32: version format (Profiler.VERSION)
 *   string: file comment
 *
 * TASK_TYPE_TABLE:
 *   int32: number of type names below
 *   string... : type names. Each of the type names is assigned id according to
 *               their position in this table starting from 0.
 *
 * TASK_RECORD:
 *   int32 size: size of the encoded task record
 *   byte[size] encoded_task_record:
 *     varint64: thread id - as was returned by Thread.getId()
 *     varint32: task id - starting from 1.
 *     varint32: parent task id for subtasks or 0 for root tasks
 *     varint64: start time in ns, relative to the Profiler.start() invocation
 *     varint64: task duration in ns
 *     byte:     task type id (see TASK_TYPE_TABLE)
 *     varint32: description string index incremented by 1 (>0) or 0 this is
 *               a first occurrence of the description string
 *     AGGREGATED_STAT...: remainder of the field (if present) represents
 *                         aggregated stats for that task
 *   string: *optional* description string, will appear only if description
 *           string index above was 0. In that case this string will be
 *           assigned next sequential id so every unique description string
 *           will appear in the file only once - after that it will be
 *           referenced by id.
 *
 * AGGREGATE_STAT:
 *   byte:     stat type
 *   varint32: total number of subtask invocations
 *   varint64: cumulative duration of subtask invocations in ns.
 *
 * EOF_MARKER:
 *   int64: -1 - please note that this corresponds to the thread id in the
 *               TASK_RECORD which is always > 0
 * </pre>
 *
 * @see ProfilerTask enum for recognized task types.
 */
@ThreadSafe
public final class Profiler {
  private static final Logger logger = Logger.getLogger(Profiler.class.getName());

  public static final int MAGIC = 0x11223344;

  // File version number. Note that merely adding new record types in
  // the ProfilerTask does not require bumping version number as long as original
  // enum values are not renamed or deleted.
  public static final int VERSION = 0x03;

  // EOF marker. Must be < 0.
  public static final int EOF_MARKER = -1;

  /** The profiler (a static singleton instance). Inactive by default. */
  private static final Profiler instance = new Profiler();

  private static final int HISTOGRAM_BUCKETS = 20;

  private static final TaskData POISON_PILL = new TaskData(0, 0, null, null, "poison pill");

  /** File format enum. */
  public enum Format {
    BINARY_BAZEL_FORMAT,
    JSON_TRACE_FILE_FORMAT,
    JSON_TRACE_FILE_COMPRESSED_FORMAT;
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
   * Container for the single task record.
   * Should never be instantiated directly - use TaskStack.create() instead.
   *
   * Class itself is not thread safe, but all access to it from Profiler
   * methods is.
   */
  @ThreadCompatible
  private static final class TaskData {
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
   * Implements datastore for object description indices. Intended to be used only by the
   * Profiler.save() method.
   */
  @ThreadCompatible
  private static final class ObjectDescriber {
    private Map<Object, Integer> descMap = new IdentityHashMap<>(2000);
    private int indexCounter = 0;

    ObjectDescriber() { }

    int getDescriptionIndex(String description) {
      Integer index = descMap.get(description);
      return (index != null) ? index : -1;
    }

    String memoizeDescription(String description) {
      Integer oldIndex = descMap.put(description, indexCounter++);
      // Do not use Preconditions class below due to the rather expensive
      // toString() calls used in the message.
      if (oldIndex != null) {
        throw new IllegalStateException(
            description
                + "' @ "
                + System.identityHashCode(description)
                + " already had description index "
                + oldIndex
                + " while assigning index "
                + descMap.get(description));
      } else if (description.length() > 20000) {
        // Note size 64k byte limitation in DataOutputStream#writeUTF().
        description = description.substring(0, 20000);
      }
      return description;
    }

    boolean isUnassigned(int index) {
      return (index < 0);
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

  /**
   * Enable profiling.
   *
   * <p>Subsequent calls to beginTask/endTask will be recorded in the provided output stream. Please
   * note that stream performance is extremely important and buffered streams should be utilized.
   *
   * @param profiledTasks which of {@link ProfilerTask}s to track
   * @param stream output stream to store profile data. Note: passing unbuffered stream object
   *     reference may result in significant performance penalties
   * @param comment a comment to insert in the profile data
   * @param recordAllDurations iff true, record all tasks regardless of their duration; otherwise
   *     some tasks may get aggregated if they finished quick enough
   * @param clock a {@code BlazeClock.instance()}
   * @param execStartTimeNanos execution start time in nanos obtained from {@code clock.nanoTime()}
   */
  public synchronized void start(
      ImmutableSet<ProfilerTask> profiledTasks,
      OutputStream stream,
      Format format,
      String comment,
      boolean recordAllDurations,
      Clock clock,
      long execStartTimeNanos,
      boolean enabledCpuUsageProfiling)
      throws IOException {
    Preconditions.checkState(!isActive(), "Profiler already active");
    initHistograms();

    this.profiledTasks = profiledTasks;
    this.clock = clock;

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
        case BINARY_BAZEL_FORMAT:
          writer = new BinaryFormatWriter(stream, execStartTimeNanos, comment);
          break;
        case JSON_TRACE_FILE_FORMAT:
          writer = new JsonTraceFileWriter(stream, execStartTimeNanos);
          break;
        case JSON_TRACE_FILE_COMPRESSED_FORMAT:
          writer = new JsonTraceFileWriter(new GZIPOutputStream(stream), execStartTimeNanos);
      }
      writer.start();
    }
    this.writerRef.set(writer);

    // activate profiler
    profileStartTime = execStartTimeNanos;

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

  /**
   * Disable profiling and complete profile file creation.
   * Subsequent calls to beginTask/endTask will no longer
   * be recorded in the profile.
   */
  public synchronized void stop() throws IOException {
    if (!isActive()) {
      return;
    }

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
      this.queue = new LinkedBlockingDeque<>();
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

  /** Writes the profile in the binary Bazel profile format. */
  private static class BinaryFormatWriter extends FileWriter {
    private final OutputStream outStream;
    private final long profileStartTime;
    private final String comment;

    BinaryFormatWriter(OutputStream outStream, long profileStartTime, String comment) {
      // Wrapping deflater stream in the buffered stream proved to reduce CPU consumption caused by
      // the write() method. Values for buffer sizes were chosen by running small amount of tests
      // and identifying point of diminishing returns - but I have not really tried to optimize
      // them.
      this.outStream = outStream;
      this.profileStartTime = profileStartTime;
      this.comment = comment;
    }

    private static void writeHeader(DataOutputStream out, String comment) throws IOException {
      out.writeInt(MAGIC); // magic
      out.writeInt(VERSION); // protocol_version
      out.writeUTF(comment);
      // ProfileTask.values() method sorts enums using their ordinal() value, so
      // there there is no need to store ordinal() value for each entry.
      out.writeInt(TASK_COUNT);
      for (ProfilerTask type : ProfilerTask.values()) {
        out.writeUTF(type.toString());
      }
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
        try (DataOutputStream out =
            new DataOutputStream(
                new BufferedOutputStream(
                    new DeflaterOutputStream(
                        // the DeflaterOutputStream has its own output buffer of 65k, chosen at
                        // random
                        outStream, new Deflater(Deflater.BEST_SPEED, false), 65536),
                    // buffer size, basically chosen at random
                    262144))) {
          writeHeader(out, comment);
          // Allocate the sink once to avoid GC
          ByteBuffer sink = ByteBuffer.allocate(1024);
          ObjectDescriber describer = new ObjectDescriber();
          TaskData data;
          while ((data = queue.take()) != POISON_PILL) {
            ((Buffer) sink).clear();

            VarInt.putVarLong(data.threadId, sink);
            VarInt.putVarInt(data.id, sink);
            VarInt.putVarInt(data.parentId, sink);
            VarInt.putVarLong(data.startTimeNanos - profileStartTime, sink);
            VarInt.putVarLong(data.duration, sink);

            // To save space (and improve performance), convert all description
            // strings to the canonical object and use IdentityHashMap to assign
            // unique numbers for each string.
            int descIndex = describer.getDescriptionIndex(data.description);
            VarInt.putVarInt(descIndex + 1, sink); // Add 1 to avoid encoding negative values.

            // Save types using their ordinal() value
            sink.put((byte) data.type.ordinal());

            // Save aggregated data stats.
            if (data.counts != null) {
              for (int i = 0; i < TASK_COUNT; i++) {
                if (data.counts[i] > 0) {
                  sink.put((byte) i); // aggregated type ordinal value
                  VarInt.putVarInt(data.counts[i], sink);
                  VarInt.putVarLong(data.durations[i], sink);
                }
              }
            }

            out.writeInt(sink.position());
            out.write(sink.array(), 0, sink.position());
            if (describer.isUnassigned(descIndex)) {
              out.writeUTF(describer.memoizeDescription(data.description));
            }
          }
          receivedPoisonPill = true;
          out.writeInt(EOF_MARKER);
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

  /** Writes the profile in Json Trace file format. */
  private static class JsonTraceFileWriter extends FileWriter {
    private final OutputStream outStream;
    private final long profileStartTimeNanos;
    private final ThreadLocal<Boolean> metadataPosted =
        ThreadLocal.withInitial(() -> Boolean.FALSE);
    // The JDK never returns 0 as thread id so we use that as fake thread id for the critical path.
    private static final long CRITICAL_PATH_THREAD_ID = 0;

    JsonTraceFileWriter(OutputStream outStream, long profileStartTimeNanos) {
      this.outStream = outStream;
      this.profileStartTimeNanos = profileStartTimeNanos;
    }

    @Override
    public void enqueue(TaskData data) {
      if (!metadataPosted.get().booleanValue()) {
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

          while ((data = queue.take()) != POISON_PILL) {
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
            if (data.type == ProfilerTask.LOCAL_CPU_USAGE) {
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
            String eventType = data.duration == 0 ? "i" : "X";
            writer.setIndent("  ");
            writer.beginObject();
            writer.setIndent("");
            writer.name("cat").value(data.type.description);
            writer.name("name").value(data.description);
            writer.name("ph").value(eventType);
            writer.name("ts")
                .value(TimeUnit.NANOSECONDS.toMicros(data.startTimeNanos - profileStartTimeNanos));
            if (data.duration != 0) {
              writer.name("dur").value(TimeUnit.NANOSECONDS.toMicros(data.duration));
            }
            writer.name("pid").value(1);
            long threadId =
                data.type == ProfilerTask.CRITICAL_PATH_COMPONENT
                    ? CRITICAL_PATH_THREAD_ID
                    : data.threadId;
            writer.name("tid").value(threadId);
            writer.endObject();
          }
          receivedPoisonPill = true;
          writer.setIndent("  ");
          writer.endArray();
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
