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
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.ActionSummary;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.MemoryMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.PackageMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.TargetMetrics;
import com.google.devtools.build.lib.buildtool.BuildPrecompleteEvent;
import com.google.devtools.build.lib.metrics.MetricsModule.Options;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.util.concurrent.atomic.AtomicLong;

class MetricsCollector {

  private final CommandEnvironment env;
  private final boolean bepPublishUsedHeapSizePostBuild;
  private final AtomicLong executedActionCount = new AtomicLong();

  private int actionsConstructed;
  private int targetsLoaded;
  private int targetsConfigured;
  private int packagesLoaded;

  MetricsCollector(CommandEnvironment env) {
    this.env = env;
    Options options = env.getOptions().getOptions(Options.class);
    this.bepPublishUsedHeapSizePostBuild =
        options != null && options.bepPublishUsedHeapSizePostBuild;
    env.getEventBus().register(this);
  }

  static void installInEnv(CommandEnvironment env) {
    new MetricsCollector(env);
  }

  @Subscribe
  public void onAnalysisPhaseComplete(AnalysisPhaseCompleteEvent event) {
    actionsConstructed = event.getActionsConstructed();
    targetsLoaded = event.getTargetsLoaded();
    targetsConfigured = event.getTargetsConfigured();
    packagesLoaded = event.getPkgManagerStats().getPackagesLoaded();
  }

  @Subscribe
  @AllowConcurrentEvents
  public void onActionComplete(ActionCompletionEvent event) {
    executedActionCount.incrementAndGet();
  }

  @Subscribe
  public void onBuildComplete(BuildPrecompleteEvent event) {
    env.getEventBus().post(new BuildMetricsEvent(createBuildMetrics()));
  }

  private BuildMetrics createBuildMetrics() {
    BuildMetrics.Builder metrics = BuildMetrics.newBuilder();
    metrics.setActionSummary(createActionSummary());
    metrics.setMemoryMetrics(createMemoryMetrics());
    metrics.setTargetMetrics(createTargetMetrics());
    metrics.setPackageMetrics(createPackageMetrics());
    return metrics.build();
  }

  private ActionSummary createActionSummary() {
    return ActionSummary.newBuilder()
        .setActionsCreated(actionsConstructed)
        .setActionsExecuted(executedActionCount.get())
        .build();
  }

  private MemoryMetrics createMemoryMetrics() {
    MemoryMetrics.Builder memoryMetrics = MemoryMetrics.newBuilder();
    if (bepPublishUsedHeapSizePostBuild) {
      System.gc();
      MemoryMXBean memBean = ManagementFactory.getMemoryMXBean();
      memoryMetrics.setUsedHeapSizePostBuild(memBean.getHeapMemoryUsage().getUsed());
    }
    return memoryMetrics.build();
  }

  private TargetMetrics createTargetMetrics() {
    return TargetMetrics.newBuilder()
        .setTargetsLoaded(targetsLoaded)
        .setTargetsConfigured(targetsConfigured)
        .build();
  }

  private PackageMetrics createPackageMetrics() {
    return PackageMetrics.newBuilder().setPackagesLoaded(packagesLoaded).build();
  }
}
