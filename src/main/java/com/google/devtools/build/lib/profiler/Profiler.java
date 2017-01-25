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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.profiler.PredicateBasedStatRecorder.RecorderAndPredicate;
import com.google.devtools.build.lib.profiler.StatRecorder.VfsHeuristics;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.VarInt;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;
import java.util.zip.Deflater;
import java.util.zip.DeflaterOutputStream;

/**
 * Blaze internal profiler. Provides facility to report various Blaze tasks and
 * store them (asynchronously) in the file for future analysis.
 * <p>
 * Implemented as singleton so any caller should use Profiler.instance() to
 * obtain reference.
 * <p>
 * Internally, profiler uses two data structures - ThreadLocal task stack to track
 * nested tasks and single ConcurrentLinkedQueue to gather all completed tasks.
 * <p>
 * Also, due to the nature of the provided functionality (instrumentation of all
 * Blaze components), build.lib.profiler package will be used by almost every
 * other Blaze package, so special attention should be paid to avoid any
 * dependencies on the rest of the Blaze code, including build.lib.util and
 * build.lib.vfs. This is important because build.lib.util and build.lib.vfs
 * contain Profiler invocations and any dependency on those two packages would
 * create circular relationship.
 * <p>
 * All gathered instrumentation data will be stored in the file. Please, note,
 * that while file format is described here it is considered internal and can
 * change at any time. For scripting, using blaze analyze-profile --dump=raw
 * would be more robust and stable solution.
 * <p>
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
//@ThreadSafe - commented out to avoid cyclic dependency with lib.util package
public final class Profiler {
  private static final Logger LOG = Logger.getLogger(Profiler.class.getName());

  static final int MAGIC = 0x11223344;

  // File version number. Note that merely adding new record types in
  // the ProfilerTask does not require bumping version number as long as original
  // enum values are not renamed or deleted.
  static final int VERSION = 0x03;

  // EOF marker. Must be < 0.
  static final int EOF_MARKER = -1;

  // Profiler will check for gathered data and persist all of it in the
  // separate thread every SAVE_DELAY ms.
  private static final int SAVE_DELAY = 2000; // ms

  /**
   * The profiler (a static singleton instance). Inactive by default.
   */
  private static final Profiler instance = new Profiler();

  private static final int HISTOGRAM_BUCKETS = 20;

  /**
   *
   * A task that was very slow.
   */
  public final class SlowTask implements Comparable<SlowTask> {
    final long durationNanos;
    final Object object;
    ProfilerTask type;

    private SlowTask(TaskData taskData) {
      this.durationNanos = taskData.duration;
      this.object = taskData.object;
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
      return toDescription(object);
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
  //@ThreadCompatible - commented out to avoid cyclic dependency with lib.util.
  private final class TaskData {
    final long threadId;
    final long startTime;
    long duration = 0L;
    final int id;
    final int parentId;
    int[] counts; // number of invocations per ProfilerTask type
    long[] durations; // time spend in the task per ProfilerTask type
    final ProfilerTask type;
    final Object object;

    TaskData(long startTime, TaskData parent,
             ProfilerTask eventType, Object object) {
      threadId = Thread.currentThread().getId();
      counts = null;
      durations = null;
      id = taskId.incrementAndGet();
      parentId = (parent == null  ? 0 : parent.id);
      this.startTime = startTime;
      this.type = eventType;
      this.object = Preconditions.checkNotNull(object);
    }

    /**
     * Aggregates information about an *immediate* subtask.
     */
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
      return "Thread " + threadId + ", task " + id + ", type " + type + ", " + object;
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
  //@ThreadSafe - commented out to avoid cyclic dependency with lib.util.
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

    public void push(ProfilerTask eventType, Object object) {
      get().add(create(clock.nanoTime(), eventType, object));
    }

    public TaskData create(long startTime, ProfilerTask eventType, Object object) {
      return new TaskData(startTime, peek(), eventType, object);
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

  private static String toDescription(Object object) {
    return (object instanceof Describable)
        ? ((Describable) object).describe()
        : object.toString();
  }

  /**
   * Implements datastore for object description indices. Intended to be used
   * only by the Profiler.save() method.
   */
  //@ThreadCompatible - commented out to avoid cyclic dependency with lib.util.
  private final class ObjectDescriber {
    private Map<Object, Integer> descMap = new IdentityHashMap<>(2000);
    private int indexCounter = 0;

    ObjectDescriber() { }

    int getDescriptionIndex(Object object) {
      Integer index = descMap.get(object);
      return (index != null) ? index : -1;
    }

    String getDescription(Object object) {
      String description = toDescription(object);

      Integer oldIndex = descMap.put(object, indexCounter++);
      // Do not use Preconditions class below due to the rather expensive
      // toString() calls used in the message.
      if (oldIndex != null) {
        throw new IllegalStateException(" Object '" + description + "' @ "
            + System.identityHashCode(object) + " already had description index "
            + oldIndex + " while assigning index " + descMap.get(object));
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
   * <p><code>priorityQueues</p> is sharded so that all threads need not compete for the same
   * lock if they do the same operation at the same time. Access to the individual queues is
   * synchronized on the queue objects themselves.
   */
  private final class SlowestTaskAggregator {
    private static final int SHARDS = 16;
    private final int size;

    @SuppressWarnings({"unchecked", "rawtypes"})
    private final PriorityQueue<SlowTask>[] priorityQueues = new PriorityQueue[SHARDS];

    SlowestTaskAggregator(int size) {
      this.size = size;

      for (int i = 0; i < SHARDS; i++) {
        priorityQueues[i] = new PriorityQueue<>(size + 1);
      }
    }

    // @ThreadSafe
    void add(TaskData taskData) {
      PriorityQueue<SlowTask> queue =
          priorityQueues[(int) (Thread.currentThread().getId() % SHARDS)];
      synchronized (queue) {
        if (queue.size() == size) {
          // Optimization: check if we are faster than the fastest element. If we are, we would
          // be the ones to fall off the end of the queue, therefore, we can safely return early.
          if (queue.peek().getDurationNanos() > taskData.duration) {
            return;
          }

          queue.add(new SlowTask(taskData));
          queue.remove();
        } else {
          queue.add(new SlowTask(taskData));
        }
      }
    }

    // @ThreadSafe
    void clear() {
      for (int i = 0; i < SHARDS; i++) {
        PriorityQueue<SlowTask> queue = priorityQueues[i];
        synchronized (queue) {
          queue.clear();
        }
      }
    }

    // @ThreadSafe
    Iterable<SlowTask> getSlowestTasks() {
      // This is slow, but since it only happens during the end of the invocation, it's OK
      PriorityQueue<SlowTask> merged = new PriorityQueue<>(size * SHARDS);
      for (int i = 0; i < SHARDS; i++) {
        PriorityQueue<SlowTask> queue = priorityQueues[i];
        synchronized (queue) {
          merged.addAll(queue);
        }
      }

      while (merged.size() > size) {
        merged.remove();
      }

      return merged;
    }
  }

  /**
   * Which {@link ProfilerTask}s are profiled.
   */
  public enum ProfiledTaskKinds {
    /**
     * Do not profile anything.
     *
     * <p>Performance is best with this case, but we lose critical path analysis and slowest
     * operation tracking.
     */
    NONE {
      @Override
      boolean isProfiling(ProfilerTask type) {
        return false;
      }
    },

    /**
     * Profile on a few, known-to-be-slow tasks.
     *
     * <p>Performance is somewhat decreased in comparison to {@link #NONE}, but we still track the
     * slowest operations (VFS).
     */
    SLOWEST {
      @Override
      boolean isProfiling(ProfilerTask type) {
        return type.collectsSlowestInstances();
      }
    },

    /**
     * Profile all tasks.
     *
     * <p>This is in use when {@code --profile} is specified.
     */
    ALL {
      @Override
      boolean isProfiling(ProfilerTask type) {
        return true;
      }
    };

    /** Whether the Profiler collects data for the given task type. */
    abstract boolean isProfiling(ProfilerTask type);
  }

  private Clock clock;
  private ProfiledTaskKinds profiledTaskKinds;
  private volatile long profileStartTime = 0L;
  private volatile boolean recordAllDurations = false;
  private AtomicInteger taskId = new AtomicInteger();

  private TaskStack taskStack;
  private Queue<TaskData> taskQueue;
  private DataOutputStream out;
  private Timer timer;
  private IOException saveException;
  private ObjectDescriber describer;
  @SuppressWarnings("unchecked")
  private final SlowestTaskAggregator[] slowestTasks =
  new SlowestTaskAggregator[ProfilerTask.values().length];

  private final StatRecorder[] tasksHistograms = new StatRecorder[ProfilerTask.values().length];

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
        for (Entry<String, ? extends Predicate<? super String>> e : vfsHeuristics.entrySet()) {
          recorders.add(new RecorderAndPredicate(
              new SingleStatRecorder(task + " " + e.getKey(), HISTOGRAM_BUCKETS), e.getValue()));
        }
        tasksHistograms[task.ordinal()] = new PredicateBasedStatRecorder(recorders);
      } else {
        tasksHistograms[task.ordinal()] = new SingleStatRecorder(task, HISTOGRAM_BUCKETS);
      }
    }
  }

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
   * <p>Subsequent calls to beginTask/endTask will be recorded
   * in the provided output stream. Please note that stream performance is
   * extremely important and buffered streams should be utilized.
   *
   * @param profiledTaskKinds which kinds of {@link ProfilerTask}s to track
   * @param stream output stream to store profile data. Note: passing unbuffered stream object
   *     reference may result in significant performance penalties
   * @param comment a comment to insert in the profile data
   * @param recordAllDurations iff true, record all tasks regardless of their duration; otherwise
   *     some tasks may get aggregated if they finished quick enough
   * @param clock a {@code BlazeClock.instance()}
   * @param execStartTimeNanos execution start time in nanos obtained from {@code clock.nanoTime()}
   */
  public synchronized void start(ProfiledTaskKinds profiledTaskKinds, OutputStream stream,
      String comment, boolean recordAllDurations, Clock clock, long execStartTimeNanos)
      throws IOException {
    Preconditions.checkState(!isActive(), "Profiler already active");
    taskStack = new TaskStack();
    taskQueue = new ConcurrentLinkedQueue<>();
    describer = new ObjectDescriber();

    this.profiledTaskKinds = profiledTaskKinds;
    this.clock = clock;

    // sanity check for current limitation on the number of supported types due
    // to using enum.ordinal() to store them instead of EnumSet for performance reasons.
    Preconditions.checkState(TASK_COUNT < 256,
        "The profiler implementation supports only up to 255 different ProfilerTask values.");

    // reset state for the new profiling session
    taskId.set(0);
    this.recordAllDurations = recordAllDurations;
    this.saveException = null;
    if (stream != null) {
      this.timer = new Timer("ProfilerTimer", true);
      // Wrapping deflater stream in the buffered stream proved to reduce CPU consumption caused by
      // the save() method. Values for buffer sizes were chosen by running small amount of tests
      // and identifying point of diminishing returns - but I have not really tried to optimize
      // them.
      this.out = new DataOutputStream(new BufferedOutputStream(new DeflaterOutputStream(
          stream, new Deflater(Deflater.BEST_SPEED, false), 65536), 262144));

      this.out.writeInt(MAGIC); // magic
      this.out.writeInt(VERSION); // protocol_version
      this.out.writeUTF(comment);
      // ProfileTask.values() method sorts enums using their ordinal() value, so
      // there there is no need to store ordinal() value for each entry.
      this.out.writeInt(TASK_COUNT);
      for (ProfilerTask type : ProfilerTask.values()) {
        this.out.writeUTF(type.toString());
      }

      // Start save thread
      timer.schedule(new TimerTask() {
        @Override public void run() { save(); }
      }, SAVE_DELAY, SAVE_DELAY);
    } else {
      this.out = null;
    }

    // activate profiler
    profileStartTime = execStartTimeNanos;
  }

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
    if (saveException != null) {
      throw saveException;
    }
    if (!isActive()) {
      return;
    }
    // Log a final event to update the duration of ProfilePhase.FINISH.
    logEvent(ProfilerTask.INFO, "Finishing");
    save();
    clear();

    for (SlowestTaskAggregator aggregator : slowestTasks) {
      if (aggregator != null) {
        aggregator.clear();
      }
    }

    if (saveException != null) {
      throw saveException;
    }
    if (out != null) {
      out.writeInt(EOF_MARKER);
      out.close();
      out = null;
    }
  }

  /**
   *  Returns true iff profiling is currently enabled.
   */
  public boolean isActive() {
    return profileStartTime != 0L;
  }

  public boolean isProfiling(ProfilerTask type) {
    return profiledTaskKinds.isProfiling(type);
  }

  /**
   * Saves all gathered information from taskQueue queue to the file.
   * Method is invoked internally by the Timer-based thread and at the end of
   * profiling session.
   */
  private synchronized void save() {
    if (out == null) {
      return;
    }
    try {
      // Allocate the sink once to avoid GC
      ByteBuffer sink = ByteBuffer.allocate(1024);
      while (!taskQueue.isEmpty()) {
        sink.clear();
        TaskData data = taskQueue.poll();

        VarInt.putVarLong(data.threadId, sink);
        VarInt.putVarInt(data.id, sink);
        VarInt.putVarInt(data.parentId, sink);
        VarInt.putVarLong(data.startTime - profileStartTime, sink);
        VarInt.putVarLong(data.duration, sink);

        // To save space (and improve performance), convert all description
        // strings to the canonical object and use IdentityHashMap to assign
        // unique numbers for each string.
        int descIndex = describer.getDescriptionIndex(data.object);
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

        this.out.writeInt(sink.position());
        this.out.write(sink.array(), 0, sink.position());
        if (describer.isUnassigned(descIndex)) {
          this.out.writeUTF(describer.getDescription(data.object));
        }
      }
      this.out.flush();
    } catch (IOException e) {
      saveException = e;
      clear();
      try {
        out.close();
      } catch (IOException e2) {
        // ignore it
      }
    }
  }

  private synchronized void clear() {
    initHistograms();
    profileStartTime = 0L;
    if (timer != null) {
      timer.cancel();
      timer = null;
    }
    taskStack = null;
    taskQueue = null;
    describer = null;

    // Note that slowest task aggregator are not cleared here because clearing happens
    // periodically over the course of a command invocation.
  }

  /**
   * Unless --record_full_profiler_data is given we drop small tasks and add their time to the
   * parents duration.
   */
  private boolean wasTaskSlowEnoughToRecord(ProfilerTask type, long duration) {
    return (recordAllDurations || duration >= type.minDuration);
  }

  /**
   * Adds task directly to the main queue bypassing task stack. Used for simple
   * tasks that are known to not have any subtasks.
   *
   * @param startTime task start time (obtained through {@link Profiler#nanoTimeMaybe()})
   * @param duration task duration
   * @param type task type
   * @param object object associated with that task. Can be String object that
   *               describes it.
   */
  private void logTask(long startTime, long duration, ProfilerTask type, Object object) {
    Preconditions.checkNotNull(object);
    Preconditions.checkState(startTime > 0, "startTime was %s", startTime);
    if (duration < 0) {
      // See note in Clock#nanoTime, which is used by Profiler#nanoTimeMaybe.
      duration = 0;
    }

    tasksHistograms[type.ordinal()].addStat((int) TimeUnit.NANOSECONDS.toMillis(duration), object);
    // Store instance fields as local variables so they are not nulled out from under us by #clear.
    TaskStack localStack = taskStack;
    Queue<TaskData> localQueue = taskQueue;
    if (localStack == null || localQueue == null) {
      // Variables have been nulled out by #clear in between the check the caller made and this
      // point in the code. Probably due to an asynchronous crash.
      LOG.severe("Variables null in profiler for " + type + ", probably due to async crash");
      return;
    }
    TaskData parent = localStack.peek();
    if (parent != null) {
      parent.aggregateChild(type, duration);
    }
    if (wasTaskSlowEnoughToRecord(type, duration)) {
      TaskData data = localStack.create(startTime, type, object);
      data.duration = duration;
      if (out != null) {
        localQueue.add(data);
      }

      SlowestTaskAggregator aggregator = slowestTasks[type.ordinal()];

      if (aggregator != null) {
        aggregator.add(data);
      }
    }
  }

  /**
   * Used externally to submit simple task (one that does not have any subtasks).
   * Depending on the minDuration attribute of the task type, task may be
   * just aggregated into the parent task and not stored directly.
   *
   * @param startTime task start time (obtained through {@link
   *        Profiler#nanoTimeMaybe()})
   * @param type task type
   * @param object object associated with that task. Can be String object that
   *               describes it.
   */
  public void logSimpleTask(long startTime, ProfilerTask type, Object object) {
    if (isActive() && isProfiling(type)) {
      logTask(startTime, clock.nanoTime() - startTime, type, object);
    }
  }

  /**
   * Used externally to submit simple task (one that does not have any
   * subtasks). Depending on the minDuration attribute of the task type, task
   * may be just aggregated into the parent task and not stored directly.
   *
   * <p>Note that start and stop time must both be acquired from the same clock
   * instance.
   *
   * @param startTime task start time
   * @param stopTime task stop time
   * @param type task type
   * @param object object associated with that task. Can be String object that
   *               describes it.
   */
  public void logSimpleTask(long startTime, long stopTime, ProfilerTask type, Object object) {
    if (isActive() && isProfiling(type)) {
      logTask(startTime, stopTime - startTime, type, object);
    }
  }

  /**
   * Used externally to submit simple task (one that does not have any
   * subtasks). Depending on the minDuration attribute of the task type, task
   * may be just aggregated into the parent task and not stored directly.
   *
   * @param startTime task start time (obtained through {@link
   *        Profiler#nanoTimeMaybe()})
   * @param duration the duration of the task
   * @param type task type
   * @param object object associated with that task. Can be String object that
   *               describes it.
   */
  public void logSimpleTaskDuration(long startTime, long duration, ProfilerTask type,
                                    Object object) {
    if (isActive() && isProfiling(type)) {
      logTask(startTime, duration, type, object);
    }
  }

  /**
   * Used to log "events" - tasks with zero duration.
   */
  public void logEvent(ProfilerTask type, Object object) {
    if (isActive() && isProfiling(type)) {
      logTask(clock.nanoTime(), 0, type, object);
    }
  }

  /**
   * Records the beginning of the task specified by the parameters. This method
   * should always be followed by completeTask() invocation to mark the end of
   * task execution (usually ensured by try {} finally {} block). Failure to do
   * so will result in task stack corruption.
   *
   * Use of this method allows to support nested task monitoring. For tasks that
   * are known to not have any subtasks, logSimpleTask() should be used instead.
   *
   * @param type predefined task type - see ProfilerTask for available types.
   * @param object object associated with that task. Can be String object that
   *               describes it.
   */
  public void startTask(ProfilerTask type, Object object) {
    // ProfilerInfo.allTasksById is supposed to be an id -> Task map, but it is in fact a List,
    // which means that we cannot drop tasks to which we had already assigned ids. Therefore,
    // non-leaf tasks must not have a minimum duration. However, we don't quite consistently
    // enforce this, and Blaze only works because we happen not to add child tasks to those parent
    // tasks that have a minimum duration.
    Preconditions.checkNotNull(object);
    if (isActive() && isProfiling(type)) {
      taskStack.push(type, object);
    }
  }

  /**
   * Records the end of the task and moves tasks from the thread-local stack to
   * the main queue. Will validate that given task type matches task at the top
   * of the stack.
   *
   * @param type task type.
   */
  public void completeTask(ProfilerTask type) {
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
      data.duration = endTime - data.startTime;
      if (data.parentId > 0) {
        taskStack.peek().aggregateChild(data.type, data.duration);
      }
      boolean shouldRecordTask = wasTaskSlowEnoughToRecord(type, data.duration);
      if (out != null && (shouldRecordTask || data.counts != null)) {
        taskQueue.add(data);
      }

      if (shouldRecordTask) {
        SlowestTaskAggregator aggregator = slowestTasks[type.ordinal()];

        if (aggregator != null) {
          aggregator.add(data);
        }
      }
    }
  }

  /**
   * Convenience method to log phase marker tasks.
   */
  public void markPhase(ProfilePhase phase) {
    MemoryProfiler.instance().markPhase(phase);
    if (isActive() && isProfiling(ProfilerTask.PHASE)) {
      Preconditions.checkState(taskStack.isEmpty(), "Phase tasks must not be nested");
      logEvent(ProfilerTask.PHASE, phase.description);
    }
  }

  /**
   * Convenience method to log spawn tasks.
   *
   * TODO(bazel-team): Right now method expects single string of the spawn action
   * as task description (usually either argv[0] or a name of the main executable
   * in case of complex shell commands). Maybe it should accept Command object
   * and create more user friendly description.
   */
  public void logSpawn(long startTime, String arg0) {
    if (isActive() && isProfiling(ProfilerTask.SPAWN)) {
      logTask(startTime, clock.nanoTime() - startTime, ProfilerTask.SPAWN, arg0);
    }
  }

}
