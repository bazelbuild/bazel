// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionExecutedEvent;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.SpawnActionExecutionException;
import com.google.devtools.build.lib.analysis.AnalysisFailureEvent;
import com.google.devtools.build.lib.analysis.AspectCompleteEvent;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.AspectAnalyzedEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.AspectBuiltEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TestAnalyzedEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelTargetAnalyzedEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelTargetBuiltEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelTargetSkippedEvent;
import com.google.errorprone.annotations.concurrent.GuardedBy;
import java.util.Comparator;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/**
 * Listens to the various status events of the top level targets/aspects.
 *
 * <p>WARNING: For consistency, the getter methods should only be used after the execution phase is
 * finished.
 */
@ThreadSafety.ThreadSafe
public class BuildResultListener {
  // Also includes test targets.
  private final Set<ConfiguredTarget> analyzedTargets = ConcurrentHashMap.newKeySet();
  private final Set<ConfiguredTarget> analyzedTests = ConcurrentHashMap.newKeySet();
  private final Map<AspectKey, ConfiguredAspect> analyzedAspects = new ConcurrentHashMap<>();
  // Also includes test targets.
  private final Set<ConfiguredTarget> skippedTargets = ConcurrentHashMap.newKeySet();
  // Also includes test targets.
  private final Set<ConfiguredTargetKey> builtTargets = ConcurrentHashMap.newKeySet();
  private final Set<AspectKey> builtAspects = ConcurrentHashMap.newKeySet();
  private final Map<ConfiguredTargetKey, NestedSet<Cause>> targetRootCauses =
      new ConcurrentHashMap<>();
  private final Map<AspectKey, NestedSet<Cause>> aspectRootCauses = new ConcurrentHashMap<>();
  private final AtomicBoolean hasSandboxedActionFailures = new AtomicBoolean(false);

  @GuardedBy("this")
  @Nullable
  private Stopwatch analysisTimer;

  @GuardedBy("this")
  @Nullable
  private Stopwatch executionTimer;

  @Subscribe
  @AllowConcurrentEvents
  public void addAnalyzedTarget(TopLevelTargetAnalyzedEvent event) {
    analyzedTargets.add(event.configuredTarget());
  }

  @Subscribe
  @AllowConcurrentEvents
  public void addAnalyzedTest(TestAnalyzedEvent event) {
    analyzedTests.add(event.configuredTarget());
  }

  @Subscribe
  @AllowConcurrentEvents
  public void addAnalyzedAspect(AspectAnalyzedEvent event) {
    analyzedAspects.put(event.aspectKey(), event.configuredAspect());
  }

  @Subscribe
  @AllowConcurrentEvents
  public void addSkippedTarget(TopLevelTargetSkippedEvent event) {
    skippedTargets.add(event.configuredTarget());
  }

  @Subscribe
  @AllowConcurrentEvents
  public void addBuiltTarget(TopLevelTargetBuiltEvent event) {
    builtTargets.add(event.configuredTargetKey());
  }

  @Subscribe
  @AllowConcurrentEvents
  public void addBuiltAspect(AspectBuiltEvent event) {
    builtAspects.add(event.aspectKey());
  }

  public ImmutableSet<ConfiguredTarget> getAnalyzedTargets() {
    return sortedCopyOf(ConfiguredTarget.ORDERING, analyzedTargets);
  }

  public ImmutableSet<ConfiguredTarget> getAnalyzedTests() {
    return sortedCopyOf(ConfiguredTarget.ORDERING, analyzedTests);
  }

  public ImmutableMap<AspectKey, ConfiguredAspect> getAnalyzedAspects() {
    return analyzedAspects.entrySet().stream()
        .sorted(Map.Entry.comparingByKey(AspectKey.ORDERING))
        .collect(toImmutableMap(Map.Entry::getKey, Map.Entry::getValue));
  }

  public ImmutableSet<ConfiguredTarget> getSkippedTargets() {
    return sortedCopyOf(ConfiguredTarget.ORDERING, skippedTargets);
  }

  public ImmutableSet<ConfiguredTargetKey> getBuiltTargets() {
    return sortedCopyOf(ConfiguredTargetKey.ORDERING, builtTargets);
  }

  public ImmutableSet<AspectKey> getBuiltAspects() {
    return sortedCopyOf(AspectKey.ORDERING, builtAspects);
  }

  private static <T> ImmutableSet<T> sortedCopyOf(Comparator<T> comparator, Set<T> set) {
    return ImmutableSet.copyOf(ImmutableList.sortedCopyOf(comparator, set));
  }

  @Subscribe
  @AllowConcurrentEvents
  public void targetComplete(TargetCompleteEvent event) {
    if (event.failed()) {
      targetRootCauses.put(event.getConfiguredTargetKey(), event.getRootCauses());
    }
  }

  @Subscribe
  @AllowConcurrentEvents
  public void aspectComplete(AspectCompleteEvent event) {
    if (event.failed()) {
      aspectRootCauses.put(event.getAspectKey(), event.getRootCauses());
    }
  }

  @Subscribe
  @AllowConcurrentEvents
  public void analysisFailure(AnalysisFailureEvent event) {
    if (event.getFailedAspect() != null) {
      aspectRootCauses.put(event.getFailedAspect(), event.getRootCauses());
    } else {
      targetRootCauses.put(event.getFailedTarget(), event.getRootCauses());
    }
  }

  public ImmutableMap<ConfiguredTargetKey, NestedSet<Cause>> getTargetRootCauses() {
    return ImmutableMap.copyOf(targetRootCauses);
  }

  public ImmutableMap<AspectKey, NestedSet<Cause>> getAspectRootCauses() {
    return ImmutableMap.copyOf(aspectRootCauses);
  }

  public synchronized void setAnalysisTimer(Stopwatch timer) {
    this.analysisTimer = timer;
  }

  public synchronized void stopAnalysisTimer() {
    if (analysisTimer != null && analysisTimer.isRunning()) {
      analysisTimer.stop();
    }
  }

  @SuppressWarnings("GoodTime") // logged as a long
  public synchronized long getAnalysisPhaseTimeInMillis() {
    return analysisTimer != null ? analysisTimer.elapsed().toMillis() : 0;
  }

  public synchronized void setExecutionTimer(Stopwatch timer) {
    this.executionTimer = timer;
  }

  public synchronized void stopExecutionTimer() {
    if (executionTimer != null && executionTimer.isRunning()) {
      executionTimer.stop();
    }
  }

  @SuppressWarnings("GoodTime") // logged as a long
  public synchronized long getExecutionPhaseTimeInMillis() {
    return executionTimer != null ? executionTimer.elapsed().toMillis() : 0;
  }

  @Subscribe
  @AllowConcurrentEvents
  public void actionExecuted(ActionExecutedEvent event) {
    ActionExecutionException exception = event.getException();
    if (exception != null && isSandboxFailure(exception)) {
      hasSandboxedActionFailures.set(true);
    }
  }

  private static boolean isSandboxFailure(ActionExecutionException exception) {
    if (exception instanceof SpawnActionExecutionException spawnException
        && isSandboxedRunner(spawnException.getSpawnResult().getRunnerName())) {
      return true;
    }
    FailureDetail failureDetail = exception.getDetailedExitCode().getFailureDetail();
    return failureDetail != null && failureDetail.hasSandbox();
  }

  public boolean hasSandboxedActionFailures() {
    return hasSandboxedActionFailures.get();
  }

  public static boolean isSandboxedRunner(@Nullable String runnerName) {
    return runnerName != null && (runnerName.endsWith("-sandbox") || runnerName.equals("docker"));
  }
}
