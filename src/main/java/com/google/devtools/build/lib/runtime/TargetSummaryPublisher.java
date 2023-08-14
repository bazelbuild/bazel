// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.Supplier;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimaps;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.AspectCompleteEvent;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.buildtool.buildevent.TestFilteringCompleteEvent;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelTargetPendingExecutionEvent;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.errorprone.annotations.concurrent.GuardedBy;
import java.util.Collection;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;

/** Aggregates and reports target-wide final statuses in real-time. */
@ThreadSafety.ThreadSafe
public final class TargetSummaryPublisher {

  private final EventBus eventBus;
  private final Supplier<Boolean> mergedSkyframeAnalysisExecution;

  /** Number of top-level aspects populated from {@link BuildStartingEvent}. */
  private final AtomicInteger aspectCount = new AtomicInteger(-1);

  private final ConcurrentHashMap<ConfiguredTargetKey, TargetSummaryAggregator> aggregators =
      new ConcurrentHashMap<>();
  private final ListMultimap<ConfiguredTargetKey, AspectCompleteEvent> queuedAspectCompleteEvents =
      Multimaps.synchronizedListMultimap(ArrayListMultimap.create());

  public TargetSummaryPublisher(
      EventBus eventBus, Supplier<Boolean> mergedSkyframeAnalysisExecution) {
    this.eventBus = eventBus;
    this.mergedSkyframeAnalysisExecution = mergedSkyframeAnalysisExecution;
  }

  /**
   * Extracts how many aspect completions per target to expect. This must happen before {@link
   * #populateTargets}.
   */
  @Subscribe
  public void buildStarting(BuildStartingEvent event) {
    int count = event.request().getAspects().size();
    checkState(
        aspectCount.compareAndSet(/*expectedValue=*/ -1, count),
        "Duplicate BuildStartingEvent with %s aspects but already have %s",
        count,
        aspectCount);
  }

  /**
   * Populates the target summary map as soon as test filtering is complete. This is the earliest at
   * which the final set of targets to build and test is known. This must happen after {@link
   * #buildStarting}.
   */
  @Subscribe
  public void populateTargets(TestFilteringCompleteEvent event) {
    ImmutableSet<ConfiguredTarget> testTargets =
        event.getTestTargets() != null
            ? ImmutableSet.copyOf(event.getTestTargets())
            : ImmutableSet.of();
    ImmutableSet<ConfiguredTarget> skippedTests = ImmutableSet.copyOf(event.getSkippedTests());
    for (ConfiguredTarget target : event.getTargets()) {
      if (skippedTests.contains(target)) {
        // Skipped tests aren't built, and won't receive completion events, so we ignore them. Note
        // we'll still get (and ignore) a TestSummary event, but that event isn't published to BEP.
        continue;
      }

      ConfiguredTargetKey configuredTargetKey = asKey(target);
      TargetSummaryAggregator newAggregator =
          createAggregatorForTarget(/*isTest=*/ testTargets.contains(target), target);
      TargetSummaryAggregator oldAggregator =
          aggregators.putIfAbsent(configuredTargetKey, newAggregator);
      checkState(
          oldAggregator == null, "target: %s, values: %s %s", target, oldAggregator, newAggregator);
    }
  }

  /**
   * Populates the aggregator for a particular top level target, including test targets.
   *
   * <p>With skymeld, the corresponding AspectCompleteEvents may arrive before the aggregator is set
   * up. We therefore need to put those events in a queue and resolve them when the aggregator
   * becomes available.
   */
  @Subscribe
  @AllowConcurrentEvents
  public void populateTarget(TopLevelTargetPendingExecutionEvent event) {
    ConfiguredTargetKey configuredTargetKey = asKey(event.configuredTarget());
    synchronized (aggregators) {
      TargetSummaryAggregator aggregator =
          aggregators.computeIfAbsent(
              configuredTargetKey,
              (ConfiguredTargetKey k) ->
                  createAggregatorForTarget(event.isTest(), event.configuredTarget()));

      if (queuedAspectCompleteEvents.containsKey(configuredTargetKey)) {
        queuedAspectCompleteEvents
            .get(configuredTargetKey)
            .forEach(e -> aggregator.addCompletionEvent(!e.failed()));
        queuedAspectCompleteEvents.removeAll(configuredTargetKey);
      }
    }
  }

  /**
   * Creates a TargetSummaryAggregator for the given target.
   *
   * @return the created aggregator.
   */
  private TargetSummaryAggregator createAggregatorForTarget(
      boolean isTest, ConfiguredTarget target) {
    int expectedCompletions = aspectCount.get() + 1; // + 1 for target itself
    checkState(expectedCompletions > 0, "Haven't received BuildStartingEvent");
    // We want target summaries for alias targets, but note they don't receive test summaries.
    return new TargetSummaryAggregator(
        target, expectedCompletions, isTest && !AliasProvider.isAlias(target));
  }

  @Subscribe
  public void buildCompleteEvent(BuildCompleteEvent event) {
    BuildResult result = event.getResult();
    Collection<ConfiguredTarget> actualTargets = result.getActualTargets();
    Collection<ConfiguredTarget> successfulTargets = result.getSuccessfulTargets();
    if (actualTargets == null || successfulTargets == null) {
      return;
    }

    // Count out how many aspects have succeeded for each target
    ImmutableMap<ConfiguredTargetKey, Integer> aspectSuccesses =
        result.getSuccessfulAspects().stream()
            .collect(
                ImmutableMap.toImmutableMap(
                    AspectKey::getBaseConfiguredTargetKey, unused -> 1, Integer::sum));

    // Now go through all targets and set overall build success. This is a backstop against missing
    // {Target|Aspect}Completed events (e.g., due to interruption or failing fast after failures).
    int expectedAspectSuccesses = aspectCount.get();
    ImmutableSet<ConfiguredTarget> builtTargets = ImmutableSet.copyOf(successfulTargets);
    for (ConfiguredTarget target : actualTargets) {
      ConfiguredTargetKey targetKey = asKey(target);
      TargetSummaryAggregator aggregator = aggregators.get(targetKey);
      if (aggregator != null && !aggregator.published.get()) {
        // Overall success means all aspects were successful and the target didn't fail to build
        int successfulAspectCount = aspectSuccesses.getOrDefault(targetKey, 0);
        checkState(successfulAspectCount <= expectedAspectSuccesses);
        aggregator.setOverallBuildSuccess(
            builtTargets.contains(target) && successfulAspectCount == expectedAspectSuccesses);
      }
    }
  }

  @Subscribe
  @AllowConcurrentEvents
  public void targetComplete(TargetCompleteEvent event) {
    TargetSummaryAggregator aggregator = aggregators.get(event.getConfiguredTargetKey());
    if (aggregator != null && !aggregator.published.get()) {
      aggregator.addCompletionEvent(!event.failed());
    }
  }

  @Subscribe
  @AllowConcurrentEvents
  public void aspectComplete(AspectCompleteEvent event) {
    TargetSummaryAggregator aggregator;
    // Prevent a race condition where #populateTarget finishes checking the
    // queuedAspectCompleteEvents before the entries are added by this method:
    // aspectComplete: (sees aggregator == null)                                  (adds to queue)
    // populateTarget:                         (creates aggregator) (checks queue)
    synchronized (aggregators) {
      aggregator = aggregators.get(event.getAspectKey().getBaseConfiguredTargetKey());

      // With skymeld, the corresponding AspectCompleteEvents may arrive before the aggregator is
      // set up. We therefore need to put those events in a queue and resolve them when the
      // aggregator becomes available.
      if (mergedSkyframeAnalysisExecution.get() && aggregator == null) {
        queuedAspectCompleteEvents.put(event.getAspectKey().getBaseConfiguredTargetKey(), event);
        return;
      }
    }

    if (aggregator != null && !aggregator.published.get()) {
      aggregator.addCompletionEvent(!event.failed());
    }
  }

  @Subscribe
  @AllowConcurrentEvents
  public void testSummaryEvent(TestSummary event) {
    TargetSummaryAggregator aggregator = aggregators.get(asKey(event.getTarget()));
    if (aggregator != null && !aggregator.published.get()) {
      aggregator.setTestSummary(event.getStatus());
    }
  }

  private static ConfiguredTargetKey asKey(ConfiguredTarget target) {
    // checkArgument(!isAlias(target));
    return ConfiguredTargetKey.builder()
        .setLabel(AliasProvider.getDependencyLabel(target))
        .setConfigurationKey(target.getConfigurationKey())
        .build();
  }

  private class TargetSummaryAggregator {
    private final ConfiguredTarget target;
    private final boolean expectTestSummary;

    /**
     * Whether a TargetSummary for {@link #target} has been published. Users of this class can avoid
     * unnecessary synchronization by not calling synchronized methods if this flag is {@code true}.
     */
    private final AtomicBoolean published = new AtomicBoolean(false);

    /** Completion events we're still waiting on (always 0 if {@link #hasBuildFailure}). */
    @GuardedBy("this")
    private int remainingCompletions;

    @GuardedBy("this")
    private boolean hasBuildFailure;

    @Nullable
    @GuardedBy("this")
    private BlazeTestStatus testStatus;

    TargetSummaryAggregator(
        ConfiguredTarget target, int expectedCompletions, boolean expectTestSummary) {
      checkArgument(expectedCompletions > 0);
      this.target = target;
      this.expectTestSummary = expectTestSummary;
      remainingCompletions = expectedCompletions;
    }

    synchronized void addCompletionEvent(boolean success) {
      if (remainingCompletions <= 0) {
        return; // already published or still waiting on test summary
      }
      if (success) {
        --remainingCompletions;
      } else {
        remainingCompletions = 0; // short-circuit: no need to wait for any other events
        hasBuildFailure = true;
      }
      publishOnceWhenReady();
    }

    synchronized void setTestSummary(BlazeTestStatus status) {
      if (remainingCompletions <= 0 && (!expectTestSummary || testStatus != null)) {
        return; // already published
      }
      testStatus = checkNotNull(status);
      publishOnceWhenReady();
    }

    synchronized void setOverallBuildSuccess(boolean success) {
      if (remainingCompletions <= 0) {
        return; // already published or still waiting on test summary
      }
      remainingCompletions = 0;
      hasBuildFailure = !success;
      publishOnceWhenReady();
    }

    /**
     * Publishes {@link TargetSummaryEvent} for {@link #target} if {@link #hasBuildFailure} or when
     * we have any test status as well as all completions ({@link #remainingCompletions} == 0).
     */
    @GuardedBy("this")
    private void publishOnceWhenReady() {
      boolean alreadyPublished = published.get();
      if (remainingCompletions > 0
          || (!hasBuildFailure && expectTestSummary && testStatus == null)) {
        checkState(!alreadyPublished, "Shouldn't have published yet: %s", target);
        return;
      }
      if (alreadyPublished) {
        return;
      }
      TargetSummaryEvent event =
          TargetSummaryEvent.create(target, !hasBuildFailure, expectTestSummary, testStatus);
      eventBus.post(event);

      published.set(true);
    }
  }
}
