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

import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.AnalysisGraphStatsEvent;
import com.google.devtools.build.lib.actions.TotalAndConfiguredTargetOnlyMetric;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.analysis.AnalysisPhaseStartedEvent;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.ActionSummary;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.ActionSummary.ActionData;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.ArtifactMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.BuildGraphMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.CumulativeMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.MemoryMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.PackageMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.TargetMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.TimingMetrics;
import com.google.devtools.build.lib.buildtool.BuildPrecompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionStartingEvent;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.clock.BlazeClock.NanosToMillisSinceEpochConverter;
import com.google.devtools.build.lib.metrics.MetricsModule.Options;
import com.google.devtools.build.lib.metrics.PostGCMemoryUseRecorder.PeakHeap;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.ExecutionFinishedEvent;
import com.google.devtools.build.skyframe.SkyframeGraphStatsEvent;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.time.Duration;
import java.util.Comparator;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAccumulator;
import java.util.stream.Stream;

class ActionStats {
  LongAccumulator firstStarted;
  LongAccumulator lastEnded;
  AtomicLong numActions;
  String mnemonic;

  ActionStats(String mnemonic) {
    this.mnemonic = mnemonic;
    firstStarted = new LongAccumulator(Math::min, Long.MAX_VALUE);
    lastEnded = new LongAccumulator(Math::max, 0);
    numActions = new AtomicLong();
  }
}

class MetricsCollector {
  private final CommandEnvironment env;
  private final boolean bepPublishUsedHeapSizePostBuild;
  private final boolean recordMetricsForAllMnemonics;
  // For ActionSummary.
  private final AtomicLong executedActionCount = new AtomicLong();
  private final ConcurrentHashMap<String, ActionStats> actionStatsMap = new ConcurrentHashMap<>();

  // For CumulativeMetrics.
  private final AtomicInteger numAnalyses;
  private final AtomicInteger numBuilds;

  private final ActionSummary.Builder actionSummary = ActionSummary.newBuilder();
  private final TargetMetrics.Builder targetMetrics = TargetMetrics.newBuilder();
  private final PackageMetrics.Builder packageMetrics = PackageMetrics.newBuilder();
  private final TimingMetrics.Builder timingMetrics = TimingMetrics.newBuilder();
  private final ArtifactMetrics.Builder artifactMetrics = ArtifactMetrics.newBuilder();
  private final BuildGraphMetrics.Builder buildGraphMetrics = BuildGraphMetrics.newBuilder();

  private MetricsCollector(
      CommandEnvironment env, AtomicInteger numAnalyses, AtomicInteger numBuilds) {
    this.env = env;
    Options options = env.getOptions().getOptions(Options.class);
    this.bepPublishUsedHeapSizePostBuild =
        options != null && options.bepPublishUsedHeapSizePostBuild;
    this.recordMetricsForAllMnemonics = options != null && options.recordMetricsForAllMnemonics;
    this.numAnalyses = numAnalyses;
    this.numBuilds = numBuilds;
    env.getEventBus().register(this);
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
    packageMetrics.setPackagesLoaded(event.getPkgManagerStats().getPackagesLoaded());
    timingMetrics.setAnalysisPhaseTimeInMs(event.getTimeInMs());
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
    numBuilds.getAndIncrement();
  }

  @SuppressWarnings("unused")
  @Subscribe
  @AllowConcurrentEvents
  public void onActionComplete(ActionCompletionEvent event) {
    executedActionCount.incrementAndGet();
    ActionStats actionStats =
        actionStatsMap.computeIfAbsent(event.getAction().getMnemonic(), ActionStats::new);
    actionStats.numActions.incrementAndGet();
    actionStats.firstStarted.accumulate(event.getRelativeActionStartTime());
    actionStats.lastEnded.accumulate(BlazeClock.nanoTime());
  }

  @SuppressWarnings("unused")
  @Subscribe
  public void onExecutionComplete(ExecutionFinishedEvent event) {
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
    env.getEventBus().post(new BuildMetricsEvent(createBuildMetrics()));
  }

  private BuildMetrics createBuildMetrics() {
    return BuildMetrics.newBuilder()
        .setActionSummary(finishActionSummary())
        .setMemoryMetrics(createMemoryMetrics())
        .setTargetMetrics(targetMetrics.build())
        .setPackageMetrics(packageMetrics.build())
        .setTimingMetrics(finishTimingMetrics())
        .setCumulativeMetrics(createCumulativeMetrics())
        .setArtifactMetrics(artifactMetrics.build())
        .setBuildGraphMetrics(buildGraphMetrics.build())
        .build();
  }

  private static final int MAX_ACTION_DATA = 20;

  private ActionSummary finishActionSummary() {
    NanosToMillisSinceEpochConverter nanosToMillisSinceEpochConverter =
        BlazeClock.createNanosToMillisSinceEpochConverter();
    Stream<ActionStats> actionStatsStream = actionStatsMap.values().stream();
    if (!recordMetricsForAllMnemonics) {
      actionStatsStream =
          actionStatsStream
              .sorted(Comparator.comparingLong(a -> -a.numActions.get()))
              .limit(MAX_ACTION_DATA);
    }
    actionStatsStream.forEach(
        action ->
            actionSummary.addActionData(
                ActionData.newBuilder()
                    .setMnemonic(action.mnemonic)
                    .setFirstStartedMs(
                        nanosToMillisSinceEpochConverter.toEpochMillis(
                            action.firstStarted.longValue()))
                    .setLastEndedMs(
                        nanosToMillisSinceEpochConverter.toEpochMillis(
                            action.lastEnded.longValue()))
                    .setActionsExecuted(action.numActions.get())
                    .build()));
    return actionSummary.setActionsExecuted(executedActionCount.get()).build();
  }

  private MemoryMetrics createMemoryMetrics() {
    MemoryMetrics.Builder memoryMetrics = MemoryMetrics.newBuilder();
    long usedHeapSizePostBuild = 0;
    if (bepPublishUsedHeapSizePostBuild) {
      System.gc();
      MemoryMXBean memBean = ManagementFactory.getMemoryMXBean();
      usedHeapSizePostBuild = memBean.getHeapMemoryUsage().getUsed();
      memoryMetrics.setUsedHeapSizePostBuild(usedHeapSizePostBuild);
    }
    PostGCMemoryUseRecorder.get()
        .getPeakPostGcHeap()
        .map(PeakHeap::bytes)
        .ifPresent(memoryMetrics::setPeakPostGcHeapSize);

    if (memoryMetrics.getPeakPostGcHeapSize() < usedHeapSizePostBuild) {
      // If we just did a GC and computed the heap size, update the one we got from the GC
      // notification (which may arrive too late for this specific GC).
      memoryMetrics.setPeakPostGcHeapSize(usedHeapSizePostBuild);
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
}
