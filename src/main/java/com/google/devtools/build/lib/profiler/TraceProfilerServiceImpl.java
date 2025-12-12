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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.collect.Extrema;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.profiler.PredicateBasedStatRecorder.RecorderAndPredicate;
import com.google.devtools.build.lib.profiler.StatRecorder.VfsHeuristics;
import com.google.devtools.build.lib.profiler.TaskData.ActionTaskData;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.sun.management.OperatingSystemMXBean;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.management.ManagementFactory;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.zip.GZIPOutputStream;
import javax.annotation.Nullable;

/** Blaze internal profiler implementation. */
@ThreadSafe
@SuppressWarnings("GoodTime") // This code is very performance sensitive.
public final class TraceProfilerServiceImpl implements TraceProfilerService {
  private static final int HISTOGRAM_BUCKETS = 20;

  private static final Duration ACTION_COUNT_BUCKET_DURATION = Duration.ofMillis(200);


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
      Extrema<SlowTask> extrema = extremaAggregators[(int) (taskData.threadId % SHARDS)];
      synchronized (extrema) {
        extrema.aggregate(
            new SlowTask(taskData.durationNanos, taskData.description, taskData.type));
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
    ImmutableList<SlowTask> getSlowestTasks() {
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

  private Clock clock;
  private Set<ProfilerTask> profiledTasks;
  private volatile boolean active = false;
  private volatile boolean recordAllDurations = false;
  private Duration profileCpuStartTime = Duration.ZERO;
  private Duration profileCpuEndTime = Duration.ZERO;
  private Duration profileStartTime = Duration.ZERO;
  private Duration profileEndTime = Duration.ZERO;

  /**
   * The reference to the current writer, if any. If the referenced writer is null, then disk writes
   * are disabled. This can happen when slowest task recording is enabled.
   */
  private final AtomicReference<JsonTraceFileWriter> writerRef = new AtomicReference<>();

  private final SlowestTaskAggregator[] slowestTasks =
      new SlowestTaskAggregator[ProfilerTask.values().length];

  @VisibleForTesting
  final StatRecorder[] tasksHistograms = new StatRecorder[ProfilerTask.values().length];

  /** Collects local cpu usage data (if enabled). */
  private final ResourceCollector resourceCollector;

  private final AtomicReference<TimeSeries> actionCountTimeSeriesRef;
  private final AtomicReference<TimeSeries> actionCacheCountTimeSeriesRef;
  private final AtomicReference<TimeSeries> localActionCountTimeSeriesRef;
  private final AtomicReference<Map<String, TimeSeries>> inflightRpcTimeSeriesMapRef;

  private Duration actionCountStartTime;
  private boolean collectTaskHistograms;
  private boolean includePrimaryOutput;
  private boolean includeTargetLabel;
  private boolean includeConfiguration;

  public TraceProfilerServiceImpl() {
    actionCountTimeSeriesRef = new AtomicReference<>();
    actionCacheCountTimeSeriesRef = new AtomicReference<>();
    localActionCountTimeSeriesRef = new AtomicReference<>();
    inflightRpcTimeSeriesMapRef = new AtomicReference<>();
    initHistograms();
    for (ProfilerTask task : ProfilerTask.values()) {
      if (task.collectsSlowestInstances) {
        slowestTasks[task.ordinal()] = new SlowestTaskAggregator();
      }
    }

    resourceCollector = new ResourceCollector();
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

  @Override
  public void globalInit() {
    // This is to ensure that the profiler is available as early as possible during the server
    // startup.
    Profiler.setTraceProfilerService(this);
  }

  // TODO(ulfjack): This returns incomplete data by design. Maybe we should return the histograms on
  // stop instead? However, this is currently only called from one location in a module, and that
  // can't call stop itself. What to do?
  @Override
  public synchronized ImmutableList<StatRecorder> getTasksHistograms() {
    Preconditions.checkState(isActive());
    return ImmutableList.copyOf(tasksHistograms);
  }

  @Override
  public long nanoTimeMaybe() {
    return isActive() ? clock.nanoTime() : -1;
  }

  @Override
  public Duration getProfileElapsedTime() {
    Duration endTime = isActive() ? Duration.ofNanos(clock.nanoTime()) : profileEndTime;

    return endTime.minus(profileStartTime);
  }

  private static Duration getProcessCpuTime() {
    OperatingSystemMXBean bean =
        (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
    return Duration.ofNanos(bean.getProcessCpuTime());
  }

  @Override
  public Duration getServerProcessCpuTime() {
    Duration cpuEndTime = isActive() ? getProcessCpuTime() : profileCpuEndTime;
    return cpuEndTime.minus(profileCpuStartTime);
  }

  @Override
  public synchronized void start(
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
      throws IOException {
    checkState(!active, "Profiler already active");

    initHistograms();

    this.profiledTasks = profiledTasks.isEmpty() ? profiledTasks : EnumSet.copyOf(profiledTasks);
    this.clock = clock;
    this.actionCountStartTime = Duration.ofNanos(clock.nanoTime());
    this.actionCountTimeSeriesRef.set(
        createTimeSeries(actionCountStartTime, ACTION_COUNT_BUCKET_DURATION));
    this.actionCacheCountTimeSeriesRef.set(
        createTimeSeries(actionCountStartTime, ACTION_COUNT_BUCKET_DURATION));
    this.localActionCountTimeSeriesRef.set(
        createTimeSeries(actionCountStartTime, ACTION_COUNT_BUCKET_DURATION));
    this.inflightRpcTimeSeriesMapRef.set(new ConcurrentHashMap<>());
    this.collectTaskHistograms = collectTaskHistograms;
    this.includePrimaryOutput = includePrimaryOutput;
    this.includeTargetLabel = includeTargetLabel;
    this.includeConfiguration = includeConfiguration;
    this.recordAllDurations = recordAllDurations;

    JsonTraceFileWriter writer = null;
    if (stream != null && format != null) {
      writer =
          switch (format) {
            case JSON_TRACE_FILE_FORMAT ->
                new JsonTraceFileWriter(
                    stream, execStartTimeNanos, slimProfile, outputBase, buildID);
            case JSON_TRACE_FILE_COMPRESSED_FORMAT ->
                new JsonTraceFileWriter(
                    new GZIPOutputStream(stream),
                    execStartTimeNanos,
                    slimProfile,
                    outputBase,
                    buildID);
          };
      writer.start();
    }
    this.writerRef.set(writer);

    // Activate profiler.
    profileStartTime = Duration.ofNanos(execStartTimeNanos);
    profileCpuStartTime = getProcessCpuTime();
    active = true;

    // Start collecting Bazel and system-wide CPU metric collection.
    this.resourceCollector.start();
  }

  // TODO(ulfjack): This returns incomplete data by design. Also see getTasksHistograms.
  @Override
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
    Map<CounterSeriesTask, double[]> counterSeriesMap = new LinkedHashMap<>();
    TimeSeries actionCountTimeSeries = actionCountTimeSeriesRef.get();
    if (actionCountTimeSeries != null) {
      double[] actionCountValues = actionCountTimeSeries.toDoubleArray(len);
      actionCountTimeSeriesRef.set(null);
      counterSeriesMap.put(
          new CounterSeriesTask("action count", "action", /* color= */ null), actionCountValues);
    }
    TimeSeries actionCacheCountTimeSeries = actionCacheCountTimeSeriesRef.get();
    if (actionCacheCountTimeSeries != null) {
      double[] actionCacheCountValues = actionCacheCountTimeSeries.toDoubleArray(len);
      actionCacheCountTimeSeriesRef.set(null);
      counterSeriesMap.put(
          new CounterSeriesTask("action cache count", "local action cache", /* color= */ null),
          actionCacheCountValues);
    }
    if (!counterSeriesMap.isEmpty()) {
      logCounters(counterSeriesMap, actionCountStartTime, ACTION_COUNT_BUCKET_DURATION);
    }

    Map<CounterSeriesTask, double[]> localCounterSeriesMap = new LinkedHashMap<>();
    TimeSeries localActionCountTimeSeries = localActionCountTimeSeriesRef.get();
    if (localActionCountTimeSeries != null) {
      double[] localActionCountValues = localActionCountTimeSeries.toDoubleArray(len);
      localActionCountTimeSeriesRef.set(null);
      localCounterSeriesMap.put(
          new CounterSeriesTask(
              "action count (local)", "local action", CounterSeriesTask.Color.DETAILED_MEMORY_DUMP),
          localActionCountValues);
    }
    if (hasNonZeroValues(localCounterSeriesMap)) {
      logCounters(localCounterSeriesMap, actionCountStartTime, ACTION_COUNT_BUCKET_DURATION);
    }

    var inflightRpcTimeSeriesMap = inflightRpcTimeSeriesMapRef.getAndSet(null);
    if (inflightRpcTimeSeriesMap != null) {
      for (var entry : inflightRpcTimeSeriesMap.entrySet()) {
        Map<CounterSeriesTask, double[]> inflightRpcCounterSeriesMap = new LinkedHashMap<>();
        var name = entry.getKey();
        var timeSeries = entry.getValue();
        double[] values = timeSeries.toDoubleArray(len);
        inflightRpcCounterSeriesMap.put(
            new CounterSeriesTask("Inflight RPCs - " + name, name, /* color= */ null), values);
        logCounters(
            inflightRpcCounterSeriesMap, actionCountStartTime, ACTION_COUNT_BUCKET_DURATION);
      }
    }
  }

  private boolean hasNonZeroValues(Map<CounterSeriesTask, double[]> countersSeriesMap) {
    return countersSeriesMap.values().stream()
        .flatMapToDouble(Arrays::stream)
        .anyMatch(v -> v != 0);
  }

  @Override
  public synchronized void stop() throws IOException {
    if (!active) {
      return;
    }
    collectActionCounts();
    resourceCollector.stop();
    // Log a final event to update the duration of ProfilePhase.FINISH.
    logEvent(ProfilerTask.INFO, "Finishing");
    try {
      JsonTraceFileWriter writer = writerRef.getAndSet(null);
      if (writer != null) {
        writer.shutdown();
        writer = null;
      }
    } finally {
      profileCpuEndTime = getProcessCpuTime();
      profileEndTime = Duration.ofNanos(clock.nanoTime());
      active = false;
    }
  }

  @Override
  public synchronized void clear() {
    Preconditions.checkState(!active);
    Arrays.fill(tasksHistograms, null);
    profileStartTime = Duration.ZERO;
    profileEndTime = Duration.ZERO;
    profileCpuStartTime = Duration.ZERO;
    profileCpuEndTime = Duration.ZERO;
    for (SlowestTaskAggregator aggregator : slowestTasks) {
      if (aggregator != null) {
        aggregator.clear();
      }
    }
    multiLaneGenerator.reset();
  }

  @Override
  public boolean isActive() {
    return active;
  }

  @Override
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

  @Override
  public void registerCounterSeriesCollector(CounterSeriesCollector collector) {
    resourceCollector.registerCounterSeriesCollector(collector);
  }

  @Override
  public void unregisterCounterSeriesCollector(CounterSeriesCollector collector) {
    resourceCollector.unregisterCounterSeriesCollector(collector);
  }

  @Override
  public void logCounters(
      Map<CounterSeriesTask, double[]> counterSeriesMap,
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
    var lane = borrowLane();
    try {
      checkNotNull(description);
      checkState(!description.isEmpty(), "No description -> not helpful");
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
          TaskData data = new TaskData(getLaneId(lane), startTimeNanos, type, description);
          data.durationNanos = duration;
          if (currentWriter != null) {
            currentWriter.enqueue(data);
          }

          SlowestTaskAggregator aggregator = slowestTasks[type.ordinal()];

          if (aggregator != null) {
            aggregator.add(data);
          }
        }
      }
    } finally {
      releaseLane(lane);
    }
  }

  @Override
  public void logSimpleTask(long startTimeNanos, ProfilerTask type, String description) {
    if (clock != null) {
      logTask(startTimeNanos, clock.nanoTime() - startTimeNanos, type, description);
    }
  }

  @Override
  public void logSimpleTask(
      long startTimeNanos, long stopTimeNanos, ProfilerTask type, String description) {
    logTask(startTimeNanos, stopTimeNanos - startTimeNanos, type, description);
  }

  @Override
  public void logSimpleTaskDuration(
      long startTimeNanos, Duration duration, ProfilerTask type, String description) {
    logTask(startTimeNanos, duration.toNanos(), type, description);
  }

  @Override
  public void logEventAtTime(long atTimeNanos, ProfilerTask type, String description) {
    logTask(atTimeNanos, 0, type, description);
  }

  @Override
  public void logEvent(ProfilerTask type, String description) {
    logEventAtTime(clock.nanoTime(), type, description);
  }

  private SilentCloseable reallyProfile(ProfilerTask type, String description) {
    final long startTimeNanos = clock.nanoTime();
    var lane = borrowLane();
    return () -> {
      try {
        completeTask(getLaneId(lane), startTimeNanos, type, description);
      } finally {
        releaseLane(lane);
      }
    };
  }

  @Override
  public SilentCloseable profile(ProfilerTask type, String description) {
    return (isActive() && isProfiling(type)) ? reallyProfile(type, description) : NOP;
  }

  @Override
  public SilentCloseable profile(ProfilerTask type, Supplier<String> description) {
    return (isActive() && isProfiling(type)) ? reallyProfile(type, description.get()) : NOP;
  }

  @Override
  public SilentCloseable profile(String description) {
    return profile(ProfilerTask.INFO, description);
  }

  @Override
  public SilentCloseable profileAction(
      ProfilerTask type,
      String mnemonic,
      String description,
      String primaryOutput,
      String targetLabel,
      String configuration) {
    checkNotNull(description);
    if (isActive() && isProfiling(type)) {
      final long startTimeNanos = clock.nanoTime();
      var lane = borrowLane();
      return () -> {
        try {
          completeAction(
              getLaneId(lane),
              startTimeNanos,
              type,
              description,
              mnemonic,
              includePrimaryOutput ? primaryOutput : null,
              includeTargetLabel ? targetLabel : null,
              includeConfiguration ? configuration : null);
        } finally {
          releaseLane(lane);
        }
      };
    } else {
      return NOP;
    }
  }

  private static final SilentCloseable NOP = () -> {};

  private boolean countAction(ProfilerTask type) {
    return type == ProfilerTask.ACTION || type == ProfilerTask.DISCOVER_INPUTS;
  }

  @Override
  public void completeTask(long startTimeNanos, ProfilerTask type, String description) {
    var lane = borrowLane();
    try {
      completeTask(getLaneId(lane), startTimeNanos, type, description);
    } finally {
      releaseLane(lane);
    }
  }

  private void completeTask(
      long laneId, long startTimeNanos, ProfilerTask type, String description) {
    if (isActive()) {
      long endTimeNanos = clock.nanoTime();
      long duration = endTimeNanos - startTimeNanos;
      if (wasTaskSlowEnoughToRecord(type, duration)) {
        recordTask(new TaskData(laneId, startTimeNanos, duration, type, description));
      }

      if (type == ProfilerTask.RPC) {
        var inflightRpcTimeSerieMap = inflightRpcTimeSeriesMapRef.get();
        if (inflightRpcTimeSerieMap != null) {
          var timeSeries =
              inflightRpcTimeSerieMap.computeIfAbsent(
                  description,
                  (unused) -> createTimeSeries(actionCountStartTime, ACTION_COUNT_BUCKET_DURATION));
          timeSeries.addRange(Duration.ofNanos(startTimeNanos), Duration.ofNanos(endTimeNanos));
        }
      }
    }
  }

  private void completeAction(
      long threadId,
      long startTimeNanos,
      ProfilerTask type,
      String description,
      String mnemonic,
      @Nullable String primaryOutput,
      @Nullable String targetLabel,
      @Nullable String configuration) {
    if (isActive()) {
      long endTimeNanos = clock.nanoTime();
      long duration = endTimeNanos - startTimeNanos;
      boolean shouldRecordTask = wasTaskSlowEnoughToRecord(type, duration);
      if (shouldRecordTask) {
        recordTask(
            new ActionTaskData(
                threadId,
                startTimeNanos,
                duration,
                type,
                mnemonic,
                description,
                primaryOutput,
                targetLabel,
                configuration));
      }
    }
  }

  private void recordTask(TaskData data) {
    JsonTraceFileWriter writer = writerRef.get();
    if (writer != null) {
      writer.enqueue(data);
    }
    long endTimeNanos = data.startTimeNanos + data.durationNanos;
    TimeSeries actionCountTimeSeries = actionCountTimeSeriesRef.get();
    TimeSeries actionCacheCountTimeSeries = actionCacheCountTimeSeriesRef.get();
    TimeSeries localActionCountTimeSeries = localActionCountTimeSeriesRef.get();
    if (actionCountTimeSeries != null && countAction(data.type)) {
      actionCountTimeSeries.addRange(
          Duration.ofNanos(data.startTimeNanos), Duration.ofNanos(endTimeNanos));
    }
    if (actionCacheCountTimeSeries != null && data.type == ProfilerTask.ACTION_CHECK) {
      actionCacheCountTimeSeries.addRange(
          Duration.ofNanos(data.startTimeNanos), Duration.ofNanos(endTimeNanos));
    }

    if (localActionCountTimeSeries != null && data.type == ProfilerTask.LOCAL_ACTION_COUNTS) {
      localActionCountTimeSeries.addRange(
          Duration.ofNanos(data.startTimeNanos), Duration.ofNanos(endTimeNanos));
    }
    SlowestTaskAggregator aggregator = slowestTasks[data.type.ordinal()];
    if (aggregator != null) {
      aggregator.add(data);
    }
  }

  @Override
  public void markPhase(ProfilePhase phase) throws InterruptedException {
    MemoryProfiler.instance().markPhase(phase);
    if (isActive() && isProfiling(ProfilerTask.PHASE)) {
      logEvent(ProfilerTask.PHASE, phase.description);
    }
  }

  private final AtomicLong nextLaneId = new AtomicLong(1_000_000);
  private final MultiLaneGenerator multiLaneGenerator = new MultiLaneGenerator();

  private class MultiLaneGenerator {
    private final Map<String, LaneGenerator> laneGenerators = Maps.newConcurrentMap();

    /**
     * @return the lane if it's active, otherwise null.
     */
    @Nullable
    private Lane acquire(String prefix) {
      if (!isActive()) {
        return null;
      }
      var laneGenerator =
          laneGenerators.computeIfAbsent(prefix, unused -> new LaneGenerator(prefix));
      return laneGenerator.acquire();
    }

    private void release(Lane lane) {
      var laneGenerator = lane.laneGenerator;
      laneGenerator.release(lane);
    }

    private void reset() {
      multiLaneGenerator.laneGenerators.clear();
    }
  }

  private static class Lane implements Comparable<Lane> {
    private final LaneGenerator laneGenerator;
    private final long id;
    private int refCount;

    private Lane(LaneGenerator laneGenerator, long id) {
      this.laneGenerator = laneGenerator;
      this.id = id;
    }

    @Override
    public int compareTo(Lane o) {
      return Long.compare(id, o.id);
    }
  }

  private class LaneGenerator {
    private final String prefix;
    private final Queue<Lane> availableLanes = new ConcurrentLinkedQueue<>();
    private final AtomicInteger count = new AtomicInteger(0);

    private LaneGenerator(String prefix) {
      this.prefix = prefix;
    }

    public Lane acquire() {
      var lane = availableLanes.poll();
      // It might create more virtual lanes, but it's fine for our purpose.
      if (lane == null) {
        lane = new Lane(this, nextLaneId.getAndIncrement());
        int newLaneIndex = count.getAndIncrement();
        String newLaneName =
            prefix.endsWith("-")
                ? prefix + newLaneIndex + " (Virtual)"
                : prefix + "-" + newLaneIndex + " (Virtual)";
        var threadMetadata = new ThreadMetadata(newLaneName, lane.id);
        var writer = TraceProfilerServiceImpl.this.writerRef.get();
        if (writer != null) {
          writer.enqueue(threadMetadata);
        }
      }
      return lane;
    }

    public void release(Lane lane) {
      availableLanes.offer(lane);
    }
  }

  private final ThreadLocal<String> virtualThreadPrefix =
      ThreadLocal.withInitial(this::guessThreadPrefix);
  private final ThreadLocal<Lane> borrowedLane =
      ThreadLocal.withInitial(
          () -> {
            var prefix = virtualThreadPrefix.get();
            var lane = multiLaneGenerator.acquire(prefix);
            if (lane == null) {
              return null;
            }
            checkState(lane.refCount == 0);
            return lane;
          });

  @Nullable
  private Lane borrowLane() {
    if (!Thread.currentThread().isVirtual() || !isActive()) {
      return null;
    }
    var lane = borrowedLane.get();
    if (lane == null) {
      return null;
    }
    lane.refCount += 1;
    return lane;
  }

  private long getLaneId(@Nullable Lane lane) {
    if (lane == null) {
      return Thread.currentThread().threadId();
    }
    return lane.id;
  }

  private void releaseLane(@Nullable Lane lane) {
    if (lane == null) {
      return;
    }
    lane.refCount -= 1;
    if (lane.refCount == 0) {
      borrowedLane.remove();
      multiLaneGenerator.release(lane);
    }
  }

  private String guessThreadPrefix() {
    var currentThread = Thread.currentThread();
    checkState(currentThread.isVirtual());
    var threadName = currentThread.getName();

    // Assume the thread name has format "prefix%d"
    for (int i = threadName.length() - 1; i > 0; i--) {
      var ch = threadName.charAt(i);
      if (ch < '0' || ch > '9') {
        if (i < threadName.length() - 1) {
          return threadName.substring(0, i + 1);
        }
      }
    }
    return "Other";
  }

  @Override
  @CanIgnoreReturnValue
  public <T> ListenableFuture<T> profileFuture(
      ListenableFuture<T> future, String prefix, ProfilerTask type, String description) {
    Lane lane = multiLaneGenerator.acquire(prefix);
    if (lane == null) {
      return future;
    }

    long startTimeNanos = clock.nanoTime();
    future.addListener(
        () -> {
          try {
            completeTask(lane.id, startTimeNanos, type, description);
          } finally {
            multiLaneGenerator.release(lane);
          }
        },
        directExecutor());
    return future;
  }

  /**
   * Implementation of {@link AsyncProfiler}.
   *
   * <p>This class is thread-compatible but not thread-safe. You should create one profiler per
   * task.
   */
  public class AsyncProfilerImpl implements AsyncProfiler {
    @Nullable private final Lane lane;
    private final long startTimeNanos;
    private final String description;

    private AsyncProfilerImpl(String prefix, String description) {
      this.lane = multiLaneGenerator.acquire(prefix);
      this.startTimeNanos = clock.nanoTime();
      this.description = description;
    }

    @Override
    public SilentCloseable profile(ProfilerTask type, String description) {
      if (!(lane != null && isProfiling(type))) {
        return NOP;
      }
      long startTimeNanos = clock.nanoTime();
      return () -> completeTask(lane.id, startTimeNanos, type, description);
    }

    @Override
    public SilentCloseable profile(String description) {
      return profile(ProfilerTask.INFO, description);
    }

    @Override
    public <T> ListenableFuture<T> profileFuture(ListenableFuture<T> future, String description) {
      return profileFuture(future, ProfilerTask.INFO, description);
    }

    @Override
    @CanIgnoreReturnValue
    public <T> ListenableFuture<T> profileFuture(
        ListenableFuture<T> future, ProfilerTask type, String description) {
      var s = profile(type, description);
      future.addListener(s::close, directExecutor());
      return future;
    }

    @Override
    public Runnable profileCallback(Runnable runnable, String description) {
      return profileCallback(runnable, ProfilerTask.INFO, description);
    }

    @Override
    public Runnable profileCallback(Runnable runnable, ProfilerTask type, String description) {
      var s = profile(type, description);
      return () -> {
        s.close();
        runnable.run();
      };
    }

    @Override
    public <T> Consumer<T> profileCallback(Consumer<T> consumer, String description) {
      return profileCallback(consumer, ProfilerTask.INFO, description);
    }

    @Override
    public <T> Consumer<T> profileCallback(
        Consumer<T> consumer, ProfilerTask type, String description) {
      var s = profile(type, description);
      return t -> {
        s.close();
        consumer.accept(t);
      };
    }

    @Override
    public void close() {
      completeTask(lane.id, startTimeNanos, ProfilerTask.INFO, description);
      multiLaneGenerator.release(lane);
    }
  }

  @Override
  public AsyncProfiler profileAsync(String prefix, String description) {
    return new AsyncProfilerImpl(prefix, description);
  }

  @Override
  public TimeSeries createTimeSeries(Duration startTime, Duration bucketDuration) {
    return new TimeSeriesImpl(startTime, bucketDuration);
  }
}
