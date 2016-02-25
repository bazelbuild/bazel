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

package com.google.devtools.build.lib.runtime;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.MapMaker;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisFailureEvent;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.LabelAndConfiguration;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildInterruptedEvent;
import com.google.devtools.build.lib.buildtool.buildevent.TestFilteringCompleteEvent;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.events.ExceptionListener;
import com.google.devtools.build.lib.rules.test.TestProvider;
import com.google.devtools.build.lib.rules.test.TestResult;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;

import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentMap;

/**
 * This class aggregates and reports target-wide test statuses in real-time.
 * It must be public for EventBus invocation.
 */
@ThreadSafety.ThreadSafe
public class AggregatingTestListener {
  private final ConcurrentMap<Artifact, TestResult> statusMap = new MapMaker().makeMap();

  private final TestResultAnalyzer analyzer;
  private final EventBus eventBus;
  private final EventHandlerPreconditions preconditionHelper;
  private volatile boolean blazeHalted = false;
  private volatile boolean skippedTestsBecauseOfEarlierFailure;

  // summaryLock guards concurrent access to these two collections, which should be kept
  // synchronized with each other.
  private final Map<LabelAndConfiguration, TestSummary.Builder> summaries;
  private final Multimap<LabelAndConfiguration, Artifact> remainingRuns;
  private final Object summaryLock = new Object();

  public AggregatingTestListener(TestResultAnalyzer analyzer,
                                 EventBus eventBus,
                                 ExceptionListener listener) {
    this.analyzer = analyzer;
    this.eventBus = eventBus;
    this.preconditionHelper = new EventHandlerPreconditions(listener);

    this.summaries = Maps.newHashMap();
    this.remainingRuns = HashMultimap.create();
  }

  /**
   * @return An unmodifiable copy of the map of test results.
   */
  public Map<Artifact, TestResult> getStatusMap() {
    return ImmutableMap.copyOf(statusMap);
  }

  /**
   * Populates the test summary map as soon as test filtering is complete.
   * This is the earliest at which the final set of targets to test is known.
   */
  @Subscribe
  @AllowConcurrentEvents
  public void populateTests(TestFilteringCompleteEvent event) {
    // Add all target runs to the map, assuming 1:1 status artifact <-> result.
    synchronized (summaryLock) {
      for (ConfiguredTarget target : event.getTestTargets()) {
        Iterable<Artifact> statusArtifacts =
            target.getProvider(TestProvider.class).getTestParams().getTestStatusArtifacts();
        preconditionHelper.checkState(remainingRuns.putAll(asKey(target), statusArtifacts));

        // And create an empty summary suitable for incremental analysis.
        // Also has the nice side effect of mapping labels to RuleConfiguredTargets.
        TestSummary.Builder summary = TestSummary.newBuilder()
            .setTarget(target)
            .setStatus(BlazeTestStatus.NO_STATUS);
        preconditionHelper.checkState(summaries.put(asKey(target), summary) == null);
      }
    }
  }

  /**
   * Records a new test run result and incrementally updates the target status.
   * This event is sent upon completion of executed test runs.
   */
  @Subscribe
  @AllowConcurrentEvents
  public void testEvent(TestResult result) {
    Preconditions.checkState(
        statusMap.put(result.getTestStatusArtifact(), result) == null,
        "Duplicate result reported for an individual test shard");

    ActionOwner testOwner = result.getTestAction().getOwner();
    LabelAndConfiguration targetLabel = LabelAndConfiguration.of(
        testOwner.getLabel(), result.getTestAction().getConfiguration());

    TestSummary finalTestSummary = null;
    synchronized (summaryLock) {
      TestSummary.Builder summary = summaries.get(targetLabel);
      preconditionHelper.checkNotNull(summary);
      if (!remainingRuns.remove(targetLabel, result.getTestStatusArtifact())) {
        // This can happen if a buildCompleteEvent() was processed before this event reached us.
        // This situation is likely to happen if --notest_keep_going is set with multiple targets.
        return;
      }
     
      summary = analyzer.incrementalAnalyze(summary, result);

      // If all runs are processed, the target is finished and ready to report.
      if (!remainingRuns.containsKey(targetLabel)) {
        finalTestSummary = summary.build();
      }
    }

    // Report finished targets.
    if (finalTestSummary != null) {
      eventBus.post(finalTestSummary);
    }
  }

  private void targetFailure(LabelAndConfiguration label) {
    TestSummary finalSummary;
    synchronized (summaryLock) {
      if (!remainingRuns.containsKey(label)) {
        // Blaze does not guarantee that BuildResult.getSuccessfulTargets() and posted TestResult
        // events are in sync. Thus, it is possible that a test event was posted, but the target is
        // not present in the set of successful targets.
        return;
      }

      TestSummary.Builder summary = summaries.get(label);
      if (summary == null) {
        // Not a test target; nothing to do.
        return;
      }
      finalSummary =
          analyzer
              .markUnbuilt(summary, blazeHalted, skippedTestsBecauseOfEarlierFailure)
              .build();

      // These are never going to run; removing them marks the target complete.
      remainingRuns.removeAll(label);
    }
    eventBus.post(finalSummary);
  }

  @VisibleForTesting
  void buildComplete(
      Collection<ConfiguredTarget> actualTargets, Collection<ConfiguredTarget> successfulTargets) {
    if (actualTargets == null || successfulTargets == null) {
      return;
    }

    for (ConfiguredTarget target: Sets.difference(
        ImmutableSet.copyOf(actualTargets), ImmutableSet.copyOf(successfulTargets))) {
      targetFailure(asKey(target));
    }
  }

  @Subscribe
  public void buildCompleteEvent(BuildCompleteEvent event) {
    BuildResult result = event.getResult();
    if (result.wasCatastrophe()) {
      blazeHalted = true;
    } else if (result.skippedTargetsBecauseOfEarlierFailure()) {
      skippedTestsBecauseOfEarlierFailure = true;
    }
    buildComplete(result.getActualTargets(), result.getSuccessfulTargets());
  }

  @Subscribe
  public void analysisFailure(AnalysisFailureEvent event) {
    targetFailure(event.getFailedTarget());
  }

  @Subscribe
  @AllowConcurrentEvents
  public void buildInterrupted(BuildInterruptedEvent event) {
    blazeHalted = true;
  }

  /**
   * Called when a build action is not executed (e.g. because a dependency failed to build). We want
   * to catch such events in order to determine when a test target has failed to build.
   */
  @Subscribe
  @AllowConcurrentEvents
  public void targetComplete(TargetCompleteEvent event) {
    if (event.failed()) {
      targetFailure(new LabelAndConfiguration(event.getTarget()));
    }
  }

  /**
   * Returns the known aggregate results for the given target at the current moment.
   */
  public TestSummary.Builder getCurrentSummary(ConfiguredTarget target) {
    synchronized (summaryLock) {
      return summaries.get(asKey(target));
    }
  }

  /**
   * Returns all test status artifacts associated with a given target
   * whose runs have yet to finish.
   */
  public Collection<Artifact> getIncompleteRuns(ConfiguredTarget target) {
    synchronized (summaryLock) {
      return Collections.unmodifiableCollection(remainingRuns.get(asKey(target)));
    }
  }

  /**
   * Returns true iff all runs of the target are accounted for.
   */
  public boolean targetReported(ConfiguredTarget target) {
    synchronized (summaryLock) {
      return summaries.containsKey(asKey(target)) && !remainingRuns.containsKey(asKey(target));
    }
  }

  /**
   * Returns the {@link TestResultAnalyzer} associated with this listener.
   */
  public TestResultAnalyzer getAnalyzer() {
    return analyzer;
  }

  private LabelAndConfiguration asKey(ConfiguredTarget target) {
    return new LabelAndConfiguration(target);
  }
}
