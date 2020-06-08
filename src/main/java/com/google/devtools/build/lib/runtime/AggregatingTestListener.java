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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.AnalysisFailureEvent;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.analysis.test.TestResult;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildInterruptedEvent;
import com.google.devtools.build.lib.buildtool.buildevent.TestFilteringCompleteEvent;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.rules.AliasConfiguredTarget;
import com.google.devtools.build.lib.runtime.TerminalTestResultNotifier.TestSummaryOptions;
import com.google.devtools.build.lib.runtime.TestResultAggregator.AggregationPolicy;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * This class aggregates and reports target-wide test statuses in real-time.
 * It must be public for EventBus invocation.
 */
@ThreadSafety.ThreadSafe
public class AggregatingTestListener {
  private final TestSummaryOptions summaryOptions;
  private final ExecutionOptions executionOptions;
  private final EventBus eventBus;
  private volatile boolean blazeHalted = false;

  // Store information about potential failures in the presence of --nokeep_going or
  // --notest_keep_going.
  private boolean skipTargetsOnFailure;

  private final ConcurrentHashMap<ConfiguredTargetKey, TestResultAggregator> aggregators;

  public AggregatingTestListener(
      TestSummaryOptions summaryOptions, ExecutionOptions executionOptions, EventBus eventBus) {
    this.summaryOptions = summaryOptions;
    this.executionOptions = executionOptions;
    this.eventBus = eventBus;

    this.aggregators = new ConcurrentHashMap<>();
  }

  /** Returns an unmodifiable copy of the map of test results. */
  public Map<Artifact, TestResult> getStatusMapForTesting() {
    Map<Artifact, TestResult> result = new HashMap<>();
    for (TestResultAggregator aggregator : aggregators.values()) {
      result.putAll(aggregator.getStatusMapForTesting());
    }
    return result;
  }

  /** Returns the known aggregate results for the given target at the current moment. */
  public TestSummary.Builder getCurrentSummaryForTesting(ConfiguredTarget target) {
    return aggregators.get(asKey(target)).getCurrentSummaryForTesting();
  }

  /**
   * Returns all test status artifacts associated with a given target whose runs have yet to finish.
   */
  public Collection<Artifact> getIncompleteRunsForTesting(ConfiguredTarget target) {
    return aggregators.get(asKey(target)).getIncompleteRunsForTesting();
  }

  /**
   * Populates the test summary map as soon as test filtering is complete.
   * This is the earliest at which the final set of targets to test is known.
   */
  @Subscribe
  @AllowConcurrentEvents
  public void populateTests(TestFilteringCompleteEvent event) {
    AggregationPolicy policy =
        new AggregationPolicy(
            eventBus,
            executionOptions.testCheckUpToDate,
            summaryOptions.testVerboseTimeoutWarnings);
    // Add all target runs to the map, assuming 1:1 status artifact <-> result.
    for (ConfiguredTarget target : event.getTestTargets()) {
      if (isAlias(target)) {
        continue;
      }
      TestResultAggregator aggregator =
          new TestResultAggregator(target, event.getConfigurationForTarget(target), policy);
      TestResultAggregator oldAggregator = aggregators.put(asKey(target), aggregator);
      Preconditions.checkState(
          oldAggregator == null, "target: %s, values: %s %s", target, oldAggregator, aggregator);
    }
  }

  /**
   * Records a new test run result and incrementally updates the target status.
   * This event is sent upon completion of executed test runs.
   */
  @Subscribe
  @AllowConcurrentEvents
  public void testEvent(TestResult result) {
    ActionOwner testOwner = result.getTestAction().getOwner();
    ConfiguredTargetKey configuredTargetKey =
        ConfiguredTargetKey.builder()
            .setLabel(testOwner.getLabel())
            .setConfiguration(result.getTestAction().getConfiguration())
            .build();
    aggregators.get(configuredTargetKey).testEvent(result);
  }

  private void targetFailure(ConfiguredTargetKey configuredTargetKey) {
    TestResultAggregator aggregator = aggregators.get(configuredTargetKey);
    if (aggregator != null) {
      aggregator.targetFailure(blazeHalted, skipTargetsOnFailure);
    }
  }

  @VisibleForTesting
  void buildComplete(
      Collection<ConfiguredTarget> actualTargets, Collection<ConfiguredTarget> successfulTargets) {
    if (actualTargets == null || successfulTargets == null) {
      return;
    }

    for (ConfiguredTarget target :
        Sets.difference(
            ImmutableSet.copyOf(actualTargets), ImmutableSet.copyOf(successfulTargets))) {
      if (isAlias(target)) {
        continue;
      }
      targetFailure(asKey(target));
    }
  }

  @Subscribe
  public void buildCompleteEvent(BuildCompleteEvent event) {
    BuildResult result = event.getResult();
    if (result.wasCatastrophe()) {
      blazeHalted = true;
    }
    skipTargetsOnFailure = result.getStopOnFirstFailure();
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
      targetFailure(event.getConfiguredTargetKey());
    }
  }

  /**
   * Prints out the results of the given tests, and returns true if they all passed. Posts any
   * targets which weren't already completed by the listener to the EventBus. Reports all targets on
   * the console via the given notifier. Run at the end of the build, run only once.
   *
   * @param testTargets The list of targets being run
   * @param notifier A console notifier to echo results to.
   * @return true if all the tests passed, else false
   */
  public boolean differentialAnalyzeAndReport(
      Collection<ConfiguredTarget> testTargets,
      Collection<ConfiguredTarget> skippedTargets,
      TestResultNotifier notifier) {
    Preconditions.checkNotNull(testTargets);
    Preconditions.checkNotNull(notifier);

    // The natural ordering of the summaries defines their output order.
    Set<TestSummary> summaries = Sets.newTreeSet();

    int totalRun = 0; // Number of targets running at least one non-cached test.
    int passCount = 0;

    for (ConfiguredTarget testTarget : testTargets) {
      TestSummary summary;
      if (isAlias(testTarget)) {
        ConfiguredTargetKey actualKey =
            ConfiguredTargetKey.builder()
                .setLabel(testTarget.getLabel())
                .setConfigurationKey(testTarget.getConfigurationKey())
                .build();
        TestResultAggregator aggregator = aggregators.get(actualKey);
        TestSummary.Builder summaryBuilder = TestSummary.newBuilder();
        summaryBuilder.mergeFrom(aggregator.aggregateAndReportSummary(skipTargetsOnFailure));
        summaryBuilder.setTarget(testTarget);
        summary = summaryBuilder.build();
      } else {
        TestResultAggregator aggregator = aggregators.get(asKey(testTarget));
        summary = aggregator.aggregateAndReportSummary(skipTargetsOnFailure);
      }
      summaries.add(summary);

      // Finished aggregating; build the final console output.
      if (summary.actionRan()) {
        totalRun++;
      }

      if (TestResult.isBlazeTestStatusPassed(summary.getStatus())) {
        passCount++;
      }
    }

    int summarySize = summaries.size();
    int testTargetsSize = testTargets.size();
    Preconditions.checkState(
        summarySize == testTargetsSize,
        "Unequal sizes: %s vs %s (%s and %s)",
        summarySize,
        testTargetsSize,
        summaries,
        testTargets);

    notifier.notify(summaries, totalRun);
    // skipped targets are not in passCount since they have NO_STATUS
    Set<ConfiguredTarget> testTargetsSet = new HashSet<>(testTargets);
    Set<ConfiguredTarget> skippedTargetsSet = new HashSet<>(skippedTargets);
    return passCount == Sets.difference(testTargetsSet, skippedTargetsSet).size();
  }

  private static boolean isAlias(ConfiguredTarget target) {
    // I expect this to be consistent with target.getProvider(AliasProvider.class) != null.
    return target instanceof AliasConfiguredTarget;
  }

  private static ConfiguredTargetKey asKey(ConfiguredTarget target) {
    Preconditions.checkArgument(!isAlias(target));
    return ConfiguredTargetKey.builder()
        .setLabel(AliasProvider.getDependencyLabel(target))
        .setConfigurationKey(target.getConfigurationKey())
        .build();
  }
}
