// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.metrics;


import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionResultReceivedEvent;
import com.google.devtools.build.lib.actions.AnalysisGraphStatsEvent;
import com.google.devtools.build.lib.actions.TotalAndConfiguredTargetOnlyMetric;
import com.google.devtools.build.lib.actions.cache.Protos.ActionCacheStatistics;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.analysis.AnalysisPhaseStartedEvent;
import com.google.devtools.build.lib.analysis.NoBuildRequestFinishedEvent;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.ActionSummary;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.ActionSummary.ActionData;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.ActionSummary.RunnerCount;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.ArtifactMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.BuildGraphMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.CumulativeMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.MemoryMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.MemoryMetrics.GarbageMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.NetworkMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.PackageMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.TargetMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.TimingMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.WorkerPoolMetrics;
import com.google.devtools.build.lib.buildtool.BuildPrecompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionStartingEvent;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.clock.BlazeClock.NanosToMillisSinceEpochConverter;
import com.google.devtools.build.lib.metrics.MetricsModule.Options;
import com.google.devtools.build.lib.metrics.PostGCMemoryUseRecorder.PeakHeap;
import com.google.devtools.build.lib.packages.metrics.ExtremaPackageMetricsRecorder;
import com.google.devtools.build.lib.packages.metrics.PackageLoadMetrics;
import com.google.devtools.build.lib.packages.metrics.PackageMetricsPackageLoadingListener;
import com.google.devtools.build.lib.packages.metrics.PackageMetricsRecorder;
import com.google.devtools.build.lib.profiler.MemoryProfiler;
import com.google.devtools.build.lib.profiler.NetworkMetricsCollector;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.SpawnStats;
import com.google.devtools.build.lib.skyframe.ExecutionFinishedEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelTargetPendingExecutionEvent;
import com.google.devtools.build.lib.worker.WorkerCreatedEvent;
import com.google.devtools.build.lib.worker.WorkerDestroyedEvent;
import com.google.devtools.build.lib.worker.WorkerEvictedEvent;
import com.google.devtools.build.lib.worker.WorkerMetricsCollector;
import com.google.devtools.build.skyframe.SkyframeGraphStatsEvent;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.protobuf.util.Durations;
import java.time.Duration;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAccumulator;
import java.util.stream.Stream;

class MetricsCollector {

  private final CommandEnvironment env;
  private final boolean recordMetricsForAllMnemonics;
  // For ActionSummary.
  private final ConcurrentHashMap<String, ActionStats> actionStatsMap = new ConcurrentHashMap<>();
  // Mapping from worker pool hash, to statistics which we collect during a build.
  private final HashMap<Integer, WorkerPoolStats> workerPoolStats = new HashMap<>();

  // For CumulativeMetrics.
  private final AtomicInteger numAnalyses;
  private final AtomicInteger numBuilds;

  private final ActionSummary.Builder actionSummary = ActionSummary.newBuilder();
  private final TargetMetrics.Builder targetMetrics = TargetMetrics.newBuilder();
  private final PackageMetrics.Builder packageMetrics = PackageMetrics.newBuilder();
  private final TimingMetrics.Builder timingMetrics = TimingMetrics.newBuilder();
  private final ArtifactMetrics.Builder artifactMetrics = ArtifactMetrics.newBuilder();
  private final BuildGraphMetrics.Builder buildGraphMetrics = BuildGraphMetrics.newBuilder();
  private final SpawnStats spawnStats = new SpawnStats();
  // Skymeld-specific: we don't have an ExecutionStartingEvent for skymeld, so we have to use
  // TopLevelTargetExecutionStartedEvent. This AtomicBoolean is so that we only account for the
  // build once.
  private final AtomicBoolean buildAccountedFor;
  private long executionStartMillis;

  @CanIgnoreReturnValue
  private MetricsCollector(
      CommandEnvironment env, AtomicInteger numAnalyses, AtomicInteger numBuilds) {
    this.env = env;
    Options options = env.getOptions().getOptions(Options.class);
    this.recordMetricsForAllMnemonics = options != null && options.recordMetricsForAllMnemonics;
    this.numAnalyses = numAnalyses;
    this.numBuilds = numBuilds;
    env.getEventBus().register(this);
    WorkerMetricsCollector.instance().setClock(env.getClock());
    this.buildAccountedFor = new AtomicBoolean();
  }

  static void installInEnv(
      CommandEnvironment env, AtomicInteger numAnalyses, AtomicInteger numBuilds) {
    new MetricsCollector(env, numAnalyses, numBuilds);
  }

  @SuppressWarnings("unused")
  @Subscribe
  public synchronized void logAnalysisStartingEvent(AnalysisPhaseStartedEvent event) {
    numAnalyses.getAndIncrement();
  }

  @SuppressWarnings("unused")
  @Subscribe
  public void onAnalysisPhaseComplete(AnalysisPhaseCompleteEvent event) {
    TotalAndConfiguredTargetOnlyMetric actionsConstructed = event.getActionsConstructed();
    actionSummary
        .setActionsCreated(actionsConstructed.total())
        .setActionsCreatedNotIncludingAspects(actionsConstructed.configuredTargetsOnly());
    TotalAndConfiguredTargetOnlyMetric targetsConfigured = event.getTargetsConfigured();
    targetMetrics
        .setTargetsConfigured(targetsConfigured.total())
        .setTargetsConfiguredNotIncludingAspects(targetsConfigured.configuredTargetsOnly());
    timingMetrics.setAnalysisPhaseTimeInMs(event.getTimeInMs());

    packageMetrics.setPackagesLoaded(event.getPkgManagerStats().getPackagesSuccessfullyLoaded());

    if (PackageMetricsPackageLoadingListener.getInstance().getPublishPackageMetricsInBep()) {
      PackageMetricsRecorder recorder =
          PackageMetricsPackageLoadingListener.getInstance().getPackageMetricsRecorder();
      if (recorder != null) {
        Stream<PackageLoadMetrics> metrics = recorder.getPackageLoadMetrics().stream();

        if (recorder.getRecorderType() == PackageMetricsRecorder.Type.ONLY_EXTREMES) {
          ExtremaPackageMetricsRecorder extremaPackageMetricsRecorder =
              (ExtremaPackageMetricsRecorder) recorder;
          // Safeguard: we have 5 metrics, so print at most 5 times the number of packages as being
          // tracked per metric.
          metrics = metrics.limit(5L * extremaPackageMetricsRecorder.getNumPackagesToTrack());
        }
        metrics.forEach(packageMetrics::addPackageLoadMetrics);
      }
    }
  }

  private void markExecutionPhaseStarted() {
    executionStartMillis = BlazeClock.instance().currentTimeMillis();
  }

  @SuppressWarnings("unused")
  @Subscribe
  public synchronized void logAnalysisGraphStats(AnalysisGraphStatsEvent event) {
    // Check only one event per build. No proto3 check for presence, so check for not-default value.
    if (buildGraphMetrics.getActionLookupValueCount() > 0) {
      BugReport.sendBugReport(
          new IllegalStateException(
              "Already initialized build graph metrics builder: "
                  + buildGraphMetrics
                  + ", "
                  + event.getBuildGraphMetrics()));
    }
    buildGraphMetrics.mergeFrom(event.getBuildGraphMetrics());
  }

  @SuppressWarnings("unused")
  @Subscribe
  public synchronized void logExecutionStartingEvent(ExecutionStartingEvent event) {
    markExecutionPhaseStarted();
    numBuilds.getAndIncrement();
  }

  // Skymeld-specific: we don't have an ExecutionStartingEvent for skymeld, so we have to use
  // TopLevelTargetExecutionStartedEvent
  @Subscribe
  public synchronized void handleExecutionPhaseStart(
      @SuppressWarnings("unused") TopLevelTargetPendingExecutionEvent event) {
    if (buildAccountedFor.compareAndSet(/*expectedValue=*/ false, /*newValue=*/ true)) {
      markExecutionPhaseStarted();
      numBuilds.getAndIncrement();
    }
  }

  @Subscribe
  public void onWorkerDestroyed(WorkerDestroyedEvent event) {
    synchronized (this) {
      WorkerPoolStats stats =
          getWorkerPoolStatsOrInsert(event.getWorkerPoolHash(), event.getMnemonic());

      stats.incrementDestroyedCount();
    }
  }

  @Subscribe
  public void onWorkerCreated(WorkerCreatedEvent event) {
    synchronized (this) {
      WorkerPoolStats stats =
          getWorkerPoolStatsOrInsert(event.getWorkerPoolHash(), event.getMnemonic());

      stats.incrementCreatedCount();
    }
  }

  @Subscribe
  public void onWorkerEvicted(WorkerEvictedEvent event) {
    synchronized (this) {
      WorkerPoolStats stats =
          getWorkerPoolStatsOrInsert(event.getWorkerPoolHash(), event.getMnemonic());

      stats.incrementEvictedCount();
    }
  }

  private WorkerPoolStats getWorkerPoolStatsOrInsert(int workerPoolHash, String mnemonic) {
    WorkerPoolStats stats =
        workerPoolStats.computeIfAbsent(
            workerPoolHash, (Integer k) -> new WorkerPoolStats(mnemonic));

    return stats;
  }

  @SuppressWarnings("unused")
  @Subscribe
  @AllowConcurrentEvents
  public void onActionComplete(ActionCompletionEvent event) {
    ActionStats actionStats =
        actionStatsMap.computeIfAbsent(event.getAction().getMnemonic(), ActionStats::new);
    actionStats.numActions.incrementAndGet();
    actionStats.firstStarted.accumulate(event.getRelativeActionStartTimeNanos());
    actionStats.lastEnded.accumulate(BlazeClock.nanoTime());
    spawnStats.incrementActionCount();
  }

  @Subscribe
  @AllowConcurrentEvents
  public void actionResultReceived(ActionResultReceivedEvent event) {
    spawnStats.countActionResult(event.getActionResult());
    ActionStats actionStats =
        actionStatsMap.computeIfAbsent(event.getAction().getMnemonic(), ActionStats::new);
    int systemTime = event.getActionResult().cumulativeCommandExecutionSystemTimeInMs();
    if (systemTime > 0) {
      actionStats.systemTime.addAndGet(systemTime);
    }
    int userTime = event.getActionResult().cumulativeCommandExecutionUserTimeInMs();
    if (userTime > 0) {
      actionStats.userTime.addAndGet(userTime);
    }
  }

  @SuppressWarnings("unused")
  @Subscribe
  public void onExecutionComplete(ExecutionFinishedEvent event) {
    timingMetrics.setExecutionPhaseTimeInMs(
        BlazeClock.instance().currentTimeMillis() - executionStartMillis);
    artifactMetrics
        .setSourceArtifactsRead(event.sourceArtifactsRead())
        .setOutputArtifactsSeen(event.outputArtifactsSeen())
        .setOutputArtifactsFromActionCache(event.outputArtifactsFromActionCache())
        .setTopLevelArtifacts(event.topLevelArtifacts());
  }

  @SuppressWarnings("unused")
  @Subscribe
  public void onSkyframeGraphStats(SkyframeGraphStatsEvent event) {
    buildGraphMetrics.setPostInvocationSkyframeNodeCount(event.getGraphSize());
  }

  @SuppressWarnings("unused")
  @Subscribe
  public void onBuildComplete(BuildPrecompleteEvent event) {
    postBuildMetricsEvent();
  }

  @SuppressWarnings("unused") // Used reflectively
  @Subscribe
  public void onNoBuildRequestFinishedEvent(NoBuildRequestFinishedEvent event) {
    postBuildMetricsEvent();
  }

  private void postBuildMetricsEvent() {
    env.getEventBus().post(new BuildMetricsEvent(createBuildMetrics()));
  }

  @SuppressWarnings("unused")
  @Subscribe
  private void logActionCacheStatistics(ActionCacheStatistics stats) {
    actionSummary.setActionCacheStatistics(stats);
  }

  private BuildMetrics createBuildMetrics() {
    BuildMetrics.Builder buildMetrics =
        BuildMetrics.newBuilder()
            .setActionSummary(finishActionSummary())
            .setMemoryMetrics(createMemoryMetrics())
            .setTargetMetrics(targetMetrics.build())
            .setPackageMetrics(packageMetrics.build())
            .setTimingMetrics(finishTimingMetrics())
            .setCumulativeMetrics(createCumulativeMetrics())
            .setArtifactMetrics(artifactMetrics.build())
            .setBuildGraphMetrics(buildGraphMetrics.build())
            .addAllWorkerMetrics(WorkerMetricsCollector.instance().createWorkerMetricsProto())
            .setWorkerPoolMetrics(createWorkerPoolMetrics());

    NetworkMetrics networkMetrics = NetworkMetricsCollector.instance().collectMetrics();
    if (networkMetrics != null) {
      buildMetrics.setNetworkMetrics(networkMetrics);
    }

    return buildMetrics.build();
  }

  private ActionData buildActionData(ActionStats actionStats) {
    NanosToMillisSinceEpochConverter nanosToMillisSinceEpochConverter =
        BlazeClock.createNanosToMillisSinceEpochConverter();
    ActionData.Builder builder =
        ActionData.newBuilder()
            .setMnemonic(actionStats.mnemonic)
            .setFirstStartedMs(
                nanosToMillisSinceEpochConverter.toEpochMillis(
                    actionStats.firstStarted.longValue()))
            .setLastEndedMs(
                nanosToMillisSinceEpochConverter.toEpochMillis(actionStats.lastEnded.longValue()))
            .setActionsExecuted(actionStats.numActions.get());
    long systemTime = actionStats.systemTime.get();
    if (systemTime > 0) {
      builder.setSystemTime(Durations.fromMillis(systemTime));
    }
    long userTime = actionStats.userTime.get();
    if (userTime > 0) {
      builder.setUserTime(Durations.fromMillis(userTime));
    }
    return builder.build();
  }

  private static final int MAX_ACTION_DATA = 20;

  private ActionSummary finishActionSummary() {
    Stream<ActionStats> actionStatsStream = actionStatsMap.values().stream();
    if (!recordMetricsForAllMnemonics) {
      actionStatsStream =
          actionStatsStream
              .sorted(Comparator.comparingLong(a -> -a.numActions.get()))
              .limit(MAX_ACTION_DATA);
    }
    actionStatsStream.forEach(action -> actionSummary.addActionData(buildActionData(action)));

    ImmutableMap<String, Integer> spawnSummary = spawnStats.getSummary();
    actionSummary.setActionsExecuted(spawnSummary.getOrDefault("total", 0));
    spawnSummary
        .entrySet()
        .forEach(
            e -> {
              RunnerCount.Builder builder = RunnerCount.newBuilder();
              builder.setName(e.getKey()).setCount(e.getValue());
              String execKind = spawnStats.getExecKindFor(e.getKey());
              if (execKind != null) {
                builder.setExecKind(execKind);
              }
              actionSummary.addRunnerCount(builder.build());
            });
    return actionSummary.build();
  }

  private MemoryMetrics createMemoryMetrics() {
    MemoryMetrics.Builder memoryMetrics = MemoryMetrics.newBuilder();
    if (MemoryProfiler.instance().getHeapUsedMemoryAtFinish() > 0) {
      memoryMetrics.setUsedHeapSizePostBuild(MemoryProfiler.instance().getHeapUsedMemoryAtFinish());
    }
    PostGCMemoryUseRecorder.get()
        .getPeakPostGcHeap()
        .map(PeakHeap::bytes)
        .ifPresent(memoryMetrics::setPeakPostGcHeapSize);

    if (memoryMetrics.getPeakPostGcHeapSize() < memoryMetrics.getUsedHeapSizePostBuild()) {
      // If we just did a GC and computed the heap size, update the one we got from the GC
      // notification (which may arrive too late for this specific GC).
      memoryMetrics.setPeakPostGcHeapSize(memoryMetrics.getUsedHeapSizePostBuild());
    }

    PostGCMemoryUseRecorder.get()
        .getPeakPostGcHeapTenuredSpace()
        .map(PeakHeap::bytes)
        .ifPresent(memoryMetrics::setPeakPostGcTenuredSpaceHeapSize);

    Map<String, Long> garbageStats = PostGCMemoryUseRecorder.get().getGarbageStats();
    for (Map.Entry<String, Long> garbageEntry : garbageStats.entrySet()) {
      GarbageMetrics.Builder garbageMetrics = GarbageMetrics.newBuilder();
      garbageMetrics.setType(garbageEntry.getKey()).setGarbageCollected(garbageEntry.getValue());
      memoryMetrics.addGarbageMetrics(garbageMetrics.build());
    }

    return memoryMetrics.build();
  }

  private CumulativeMetrics createCumulativeMetrics() {
    return CumulativeMetrics.newBuilder()
        .setNumAnalyses(numAnalyses.get())
        .setNumBuilds(numBuilds.get())
        .build();
  }

  private TimingMetrics finishTimingMetrics() {
    Duration elapsedWallTime = Profiler.elapsedTimeMaybe();
    if (elapsedWallTime != null) {
      timingMetrics.setWallTimeInMs(elapsedWallTime.toMillis());
    }
    Duration cpuTime = Profiler.getProcessCpuTimeMaybe();
    if (cpuTime != null) {
      timingMetrics.setCpuTimeInMs(cpuTime.toMillis());
    }
    return timingMetrics.build();
  }

  private WorkerPoolMetrics createWorkerPoolMetrics() {
    WorkerPoolMetrics.Builder metricsBuilder = WorkerPoolMetrics.newBuilder();

    workerPoolStats.forEach(
        (workerPoolHash, workerStats) ->
            metricsBuilder.addWorkerPoolStats(
                WorkerPoolMetrics.WorkerPoolStats.newBuilder()
                    .setHash(workerPoolHash)
                    .setMnemonic(workerStats.getMnemonic())
                    .setCreatedCount(workerStats.getCreatedCount())
                    .setDestroyedCount(workerStats.getDestroyedCount())
                    .setEvictedCount(workerStats.getEvictedCount())
                    .build()));

    return metricsBuilder.build();
  }

  private static class WorkerPoolStats {
    private int createdCount;
    private int destroyedCount;
    private int evictedCount;
    private final String mnemonic;

    WorkerPoolStats(String mnemonic) {
      this.mnemonic = mnemonic;
    }

    void incrementCreatedCount() {
      createdCount++;
    }

    void incrementDestroyedCount() {
      destroyedCount++;
    }

    void incrementEvictedCount() {
      evictedCount++;
    }

    public int getCreatedCount() {
      return createdCount;
    }

    public int getDestroyedCount() {
      return destroyedCount;
    }

    public String getMnemonic() {
      return mnemonic;
    }

    public int getEvictedCount() {
      return evictedCount;
    }
  }

  private static class ActionStats {

    final LongAccumulator firstStarted;
    final LongAccumulator lastEnded;
    final AtomicLong numActions;
    final String mnemonic;
    final AtomicLong systemTime;
    final AtomicLong userTime;

    ActionStats(String mnemonic) {
      this.mnemonic = mnemonic;
      firstStarted = new LongAccumulator(Math::min, Long.MAX_VALUE);
      lastEnded = new LongAccumulator(Math::max, 0);
      numActions = new AtomicLong();
      systemTime = new AtomicLong();
      userTime = new AtomicLong();
    }
  }
}
