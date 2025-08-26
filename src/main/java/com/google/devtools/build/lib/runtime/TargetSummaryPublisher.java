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
import com.google.devtools.build.lib.skyframe.ToplevelAspectsIdentifiedEvent;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.errorprone.annotations.concurrent.GuardedBy;
import java.util.Collection;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/** Aggregates and reports target-wide final statuses in real-time. */
@ThreadSafety.ThreadSafe
public final class TargetSummaryPublisher {

  private final EventBus eventBus;
  private final Supplier<Boolean> mergedSkyframeAnalysisExecution;

  /** Whether or not toplevel aspects are present, from {@link BuildStartingEvent}. */
  private final AtomicBoolean hasAspects = new AtomicBoolean(false);

  private final ConcurrentHashMap<ConfiguredTargetKey, TargetSummaryAggregator> aggregators =
      new ConcurrentHashMap<>();
  private final ConcurrentHashMap<ConfiguredTargetKey, Integer> aspectCountPerTarget =
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
   *
   * <p>This excludes --exec_aspects.
   */
  @Subscribe
  public void buildStarting(BuildStartingEvent event) {
    hasAspects.set(!event.request().getAspects().isEmpty());
  }

  /**
   * Reports the correct number of top-level aspects that were analyzed for a given configured
   * target.
   *
   * <p>If analysis is successful, this event will be posted for all targets with any aspects, both
   * with and without skymeld (merged analysis+execution).
   *
   * <p>When skymeld is <em>disabled</em>, all of these events are posted strictly before the {@link
   * TestFilteringCompleteEvent}.
   *
   * <p>When skymeld is <em>enabled</em>, all of these events are posted strictly before any {@link
   * AspectCompleteEvent} is posted for the same configured target. They are posted concurrently
   * with the {@link TopLevelTargetPendingExecutionEvent} that is posted for the same target.
   */
  @Subscribe
  public void toplevelAspectsIdentified(ToplevelAspectsIdentifiedEvent event) {
    ConfiguredTargetKey targetKey = event.baseConfiguredTargetKey();
    int numTopLevelAspects = event.numTopLevelAspects();
    aspectCountPerTarget.put(targetKey, numTopLevelAspects);
    if (aggregators.containsKey(targetKey)) {
      // We may have already set the expected aspect completions if this method is racing with
      // #populateTarget(). This is safe because we guarantee that we set the same value, so that
      // no aspect completions have happened when we double-set the expected aspect completions.
      aggregators.get(targetKey).setExpectAspectCompletions(numTopLevelAspects);
    }
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
        // Skipped tests aren't built, and won't receive completion events, so we ignore them.
        // Note we'll still get (and ignore) a TestSummary event, but that event isn't published to
        // BEP.
        continue;
      }

      ConfiguredTargetKey configuredTargetKey = asKey(target);
      TargetSummaryAggregator newAggregator =
          createAggregatorForTarget(/* isTest= */ testTargets.contains(target), target);
      if (aspectCountPerTarget.containsKey(configuredTargetKey)) {
        newAggregator.setExpectAspectCompletions(aspectCountPerTarget.get(configuredTargetKey));
      }
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
      TargetSummaryAggregator newAggregator =
          createAggregatorForTarget(event.isTest(), event.configuredTarget());
      if (aspectCountPerTarget.containsKey(configuredTargetKey)) {
        newAggregator.setExpectAspectCompletions(aspectCountPerTarget.get(configuredTargetKey));
      }
      TargetSummaryAggregator oldAggregator =
          aggregators.putIfAbsent(configuredTargetKey, newAggregator);
      checkState(
          oldAggregator == null,
          "target: %s, values: %s %s",
          configuredTargetKey,
          oldAggregator,
          newAggregator);
      if (queuedAspectCompleteEvents.containsKey(configuredTargetKey)) {
        queuedAspectCompleteEvents
            .get(configuredTargetKey)
            .forEach(e -> newAggregator.addAspectCompletionEvent(!e.failed()));
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
    // We want target summaries for alias targets, but note they don't receive test summaries.
    return new TargetSummaryAggregator(
        target, isTest && !AliasProvider.isAlias(target), hasAspects.get());
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
    ImmutableSet<ConfiguredTarget> builtTargets = ImmutableSet.copyOf(successfulTargets);
    for (ConfiguredTarget target : actualTargets) {
      ConfiguredTargetKey targetKey = asKey(target);
      // If we have not seen the ToplevelAspectsIdentifiedEvent for a target, and we are expecting
      // aspects, then we know the target failed to build all its aspects as we never even analyzed
      // any aspects. Set expectedAspectSuccesses to Integer.MAX_VALUE to make clear: we failed.
      int expectedAspectSuccesses =
          aspectCountPerTarget.getOrDefault(targetKey, hasAspects.get() ? Integer.MAX_VALUE : 0);
      TargetSummaryAggregator aggregator = aggregators.get(targetKey);
      if (aggregator != null && !aggregator.published.get()) {
        // Overall success means all aspects were successful and the target didn't fail to build
        int successfulAspectCount = aspectSuccesses.getOrDefault(targetKey, 0);
        checkState(
            successfulAspectCount <= expectedAspectSuccesses,
            "for target %s got %s successful aspects, expected at most %s",
            targetKey,
            successfulAspectCount,
            expectedAspectSuccesses);
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
      aggregator.addAspectCompletionEvent(!event.failed());
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
    private final boolean expectAspectCompletions;
    private final boolean expectTestSummary;

    /**
     * Whether a TargetSummary for {@link #target} has been published. Users of this class can avoid
     * unnecessary synchronization by not calling synchronized methods if this flag is {@code true}.
     */
    private final AtomicBoolean published = new AtomicBoolean(false);

    /** Whether or not the target has completed being built, or if there was any build failure. */
    @GuardedBy("this")
    private boolean targetCompleted;

    /** Aspect completion events we're still waiting on (always 0 if {@link #hasBuildFailure}). */
    @GuardedBy("this")
    private int remainingAspectCompletions;

    @GuardedBy("this")
    private boolean hasBuildFailure;

    @Nullable
    @GuardedBy("this")
    private BlazeTestStatus testStatus;

    TargetSummaryAggregator(
        ConfiguredTarget target, boolean expectTestSummary, boolean expectAspectCompletions) {
      this.target = target;
      this.expectTestSummary = expectTestSummary;
      this.expectAspectCompletions = expectAspectCompletions;
      targetCompleted = false;
      remainingAspectCompletions = -1;
    }

    synchronized void setExpectAspectCompletions(int newRemainingAspectCompletions) {
      checkState(this.expectAspectCompletions, "Cannot track aspects unless --aspects is set.");
      checkState(
          remainingAspectCompletions < 0
              || remainingAspectCompletions == newRemainingAspectCompletions
              || hasBuildFailure,
          "Cannot call setExpectAspectCompletions() twice on a single target after aspect"
              + " completions have begun. Was %s, got %s.",
          remainingAspectCompletions,
          newRemainingAspectCompletions);
      // If we have already had a build failure (because the target failed) then we have set
      // remainingAspectCompletions = 0 and it should stay at zero.
      if (hasBuildFailure) {
        return;
      }

      remainingAspectCompletions = newRemainingAspectCompletions;
    }

    synchronized void addCompletionEvent(boolean success) {
      if (targetCompleted) {
        return; // already published or still waiting on aspects or test summary
      }
      targetCompleted = true;
      if (!success) {
        remainingAspectCompletions = 0;
        hasBuildFailure = true;
      }
      publishOnceWhenReady();
    }

    synchronized void addAspectCompletionEvent(boolean success) {
      if (remainingAspectCompletions <= 0) {
        return; // already published or still waiting on target or test summary
      }
      if (success) {
        --remainingAspectCompletions;
      } else {
        targetCompleted = true;
        remainingAspectCompletions = 0;
        hasBuildFailure = true;
      }
      publishOnceWhenReady();
    }

    synchronized void setTestSummary(BlazeTestStatus status) {
      if (remainingAspectCompletions <= 0
          && targetCompleted
          && (!expectTestSummary || testStatus != null)) {
        return; // already published
      }
      testStatus = checkNotNull(status);
      publishOnceWhenReady();
    }

    synchronized void setOverallBuildSuccess(boolean success) {
      if (remainingAspectCompletions <= 0 && targetCompleted) {
        return; // already published or still waiting on test summary
      }
      targetCompleted = true;
      remainingAspectCompletions = 0;
      hasBuildFailure = !success;
      publishOnceWhenReady();
    }

    /**
     * Publishes {@link TargetSummaryEvent} for {@link #target} if {@link #hasBuildFailure} or when
     * we have any test status as well as all completions ({@link #targetCompleted} and {@link
     * #remainingAspectCompletions} == 0).
     */
    @GuardedBy("this")
    private void publishOnceWhenReady() {
      boolean alreadyPublished = published.get();
      boolean waitingForTargetCompletion = !targetCompleted;
      boolean waitingForAspectCompletions =
          remainingAspectCompletions > 0
              || (expectAspectCompletions && remainingAspectCompletions == -1);
      boolean waitingForTestStatus = !hasBuildFailure && expectTestSummary && testStatus == null;
      if (waitingForTargetCompletion || waitingForAspectCompletions || waitingForTestStatus) {
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
