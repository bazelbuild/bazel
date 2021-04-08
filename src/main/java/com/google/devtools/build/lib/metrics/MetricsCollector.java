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
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.ActionSummary;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.ArtifactMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.BuildGraphMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.CumulativeMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.MemoryMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.PackageMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.TargetMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.TimingMetrics;
import com.google.devtools.build.lib.buildtool.BuildPrecompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionStartingEvent;
import com.google.devtools.build.lib.metrics.MetricsModule.Options;
import com.google.devtools.build.lib.metrics.PostGCMemoryUseRecorder.PeakHeap;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.ExecutionFinishedEvent;
import com.google.devtools.build.skyframe.SkyframeGraphStatsEvent;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

class MetricsCollector {
  private final CommandEnvironment env;
  private final boolean bepPublishUsedHeapSizePostBuild;
  // For ActionSummary.
  private final AtomicLong executedActionCount = new AtomicLong();

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
    TotalAndConfiguredTargetOnlyMetric actionLookupValueCount = event.getActionLookupValueCount();
    TotalAndConfiguredTargetOnlyMetric actionCount = event.getActionCount();
    buildGraphMetrics
        .setActionLookupValueCount(actionLookupValueCount.total())
        .setActionLookupValueCountNotIncludingAspects(
            actionLookupValueCount.configuredTargetsOnly())
        .setActionCount(actionCount.total())
        .setActionCountNotIncludingAspects(actionCount.configuredTargetsOnly())
        .setOutputArtifactCount(event.getOutputArtifactCount());
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

  private ActionSummary finishActionSummary() {
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
