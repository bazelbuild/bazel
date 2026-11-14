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
import java.util.function.IntSupplier;
import java.util.function.Supplier;
import java.util.function.ToIntFunction;
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

  private final SortedCopy<ImmutableSet<ConfiguredTarget>> sortedAnalyzedTargets =
      new SortedCopy<>(
          analyzedTargets::size,
          () -> sortedCopyOf(ConfiguredTarget.ORDERING, analyzedTargets),
          ImmutableSet::size);
  private final SortedCopy<ImmutableSet<ConfiguredTarget>> sortedAnalyzedTests =
      new SortedCopy<>(
          analyzedTests::size,
          () -> sortedCopyOf(ConfiguredTarget.ORDERING, analyzedTests),
          ImmutableSet::size);
  private final SortedCopy<ImmutableMap<AspectKey, ConfiguredAspect>> sortedAnalyzedAspects =
      new SortedCopy<>(
          analyzedAspects::size,
          () ->
              analyzedAspects.entrySet().stream()
                  .sorted(Map.Entry.comparingByKey(AspectKey.ORDERING))
                  .collect(toImmutableMap(Map.Entry::getKey, Map.Entry::getValue)),
          ImmutableMap::size);
  private final SortedCopy<ImmutableSet<ConfiguredTarget>> sortedSkippedTargets =
      new SortedCopy<>(
          skippedTargets::size,
          () -> sortedCopyOf(ConfiguredTarget.ORDERING, skippedTargets),
          ImmutableSet::size);
  private final SortedCopy<ImmutableSet<ConfiguredTargetKey>> sortedBuiltTargets =
      new SortedCopy<>(
          builtTargets::size,
          () -> sortedCopyOf(ConfiguredTargetKey.ORDERING, builtTargets),
          ImmutableSet::size);
  private final SortedCopy<ImmutableSet<AspectKey>> sortedBuiltAspects =
      new SortedCopy<>(
          builtAspects::size,
          () -> sortedCopyOf(AspectKey.ORDERING, builtAspects),
          ImmutableSet::size);

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
    return sortedAnalyzedTargets.get();
  }

  public ImmutableSet<ConfiguredTarget> getAnalyzedTests() {
    return sortedAnalyzedTests.get();
  }

  public ImmutableMap<AspectKey, ConfiguredAspect> getAnalyzedAspects() {
    return sortedAnalyzedAspects.get();
  }

  public ImmutableSet<ConfiguredTarget> getSkippedTargets() {
    return sortedSkippedTargets.get();
  }

  public ImmutableSet<ConfiguredTargetKey> getBuiltTargets() {
    return sortedBuiltTargets.get();
  }

  public ImmutableSet<AspectKey> getBuiltAspects() {
    return sortedBuiltAspects.get();
  }

  private static <T> ImmutableSet<T> sortedCopyOf(Comparator<T> comparator, Set<T> set) {
    return ImmutableSet.copyOf(ImmutableList.sortedCopyOf(comparator, set));
  }

  /**
   * A sorted copy of a collection that only ever grows, made again only after it has grown.
   *
   * <p>The getters of this class are called several times each once a build is over, and sorting
   * the targets of a build of several hundred thousand of them takes seconds every time.
   */
  private static final class SortedCopy<C> {
    private final IntSupplier currentSize;
    private final Supplier<C> sort;
    private final ToIntFunction<C> sizeOf;

    @GuardedBy("this")
    @Nullable
    private C copy;

    SortedCopy(IntSupplier currentSize, Supplier<C> sort, ToIntFunction<C> sizeOf) {
      this.currentSize = currentSize;
      this.sort = sort;
      this.sizeOf = sizeOf;
    }

    synchronized C get() {
      // Nothing is ever removed, so a copy that has as many elements as the collection has now
      // holds the same elements.
      if (copy == null || sizeOf.applyAsInt(copy) != currentSize.getAsInt()) {
        copy = sort.get();
      }
      return copy;
    }
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
