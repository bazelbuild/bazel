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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.AnalysisFailureEvent;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.analysis.test.TestResult;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildInterruptedEvent;
import com.google.devtools.build.lib.buildtool.buildevent.TestFilteringCompleteEvent;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.runtime.TestResultAggregator.AggregationPolicy;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.TestCommand;
import com.google.devtools.build.lib.server.FailureDetails.TestCommand.Code;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TestAnalyzedEvent;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.DetailedExitCode.DetailedExitCodeComparator;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/** Aggregates and reports target-wide test statuses in real-time. */
@ThreadSafety.ThreadSafe
public final class AggregatingTestListener {

  private static final DetailedExitCode TESTS_FAILED_DETAILED_CODE =
      DetailedExitCode.of(
          FailureDetail.newBuilder()
              .setMessage("tests failed")
              .setTestCommand(TestCommand.newBuilder().setCode(Code.TESTS_FAILED))
              .build());

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

  /**
   * Populates the test summary map as soon as test filtering is complete. This is the earliest at
   * which the final set of targets to test is known.
   *
   * <p>This is used in the non-Skymeld case.
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
      if (AliasProvider.isAlias(target)) {
        // It is safe to skip aliases because the actual target will be in event.getTestTargets().
        continue;
      }
      TestResultAggregator aggregator =
          new TestResultAggregator(
              target,
              event.getConfigurationForTarget(target),
              policy,
              event.getSkippedTests().contains(target));
      TestResultAggregator oldAggregator = aggregators.put(asKey(target), aggregator);
      Preconditions.checkState(
          oldAggregator == null, "target: %s, values: %s %s", target, oldAggregator, aggregator);
    }
  }

  /**
   * Creates the {@link TestResultAggregator} for the analyzed test target.
   *
   * <p>Since the event is fired from within a SkyFunction, it is possible to receive duplicate
   * events. In case of duplication, simply return without creating any new aggregator.
   *
   * <p>This is used in the Skymeld case.
   */
  @Subscribe
  @AllowConcurrentEvents
  public void populateTest(TestAnalyzedEvent event) {
    ConfiguredTarget target = event.configuredTarget();
    // Even if target is an alias, we still need to ensure that there's an aggregator present.
    // Nothing guarantees that the actual target's TestAnalyzedEvent is posted before the alias
    // completes the test (b/419325593). Using computeIfAbsent ensures that we have a single
    // aggregator, as this method can be called concurrently for an alias and its actual target.
    aggregators.computeIfAbsent(
        asKey(target),
        k ->
            new TestResultAggregator(
                target.getActual(), // In case target is an alias.
                event.buildConfigurationValue(),
                new AggregationPolicy(
                    eventBus,
                    executionOptions.testCheckUpToDate,
                    summaryOptions.testVerboseTimeoutWarnings),
                event.isSkipped()));
  }

  /**
   * Records a new test run result and incrementally updates the target status. This event is sent
   * upon completion of executed test runs.
   */
  @Subscribe
  @AllowConcurrentEvents
  public void testEvent(TestResult result) {
    TestRunnerAction testAction = result.getTestAction();
    ConfiguredTargetKey key =
        ConfiguredTargetKey.builder()
            .setLabel(testAction.getOwner().getLabel())
            .setConfiguration(testAction.getConfiguration())
            .build();
    TestResultAggregator aggregator =
        checkNotNull(aggregators.get(key), "Missing aggregator for %s", key);
    aggregator.testEvent(result);
  }

  private void targetFailure(ConfiguredTargetKey configuredTargetKey) {
    TestResultAggregator aggregator = aggregators.get(configuredTargetKey);
    if (aggregator != null) {
      aggregator.targetFailure(blazeHalted, skipTargetsOnFailure);
    }
  }

  private void targetSkipped(ConfiguredTargetKey configuredTargetKey) {
    TestResultAggregator aggregator = aggregators.get(configuredTargetKey);
    if (aggregator != null) {
      aggregator.targetSkipped();
    }
  }

  @VisibleForTesting
  void buildComplete(
      Collection<ConfiguredTarget> actualTargets,
      Collection<ConfiguredTarget> skippedTargets,
      Collection<ConfiguredTarget> successfulTargets) {
    if (actualTargets == null || successfulTargets == null) {
      return;
    }

    ImmutableSet<ConfiguredTarget> nonSuccessfulTargets =
        Sets.difference(ImmutableSet.copyOf(actualTargets), ImmutableSet.copyOf(successfulTargets))
            .immutableCopy();
    for (ConfiguredTarget target :
        Sets.difference(
            ImmutableSet.copyOf(nonSuccessfulTargets), ImmutableSet.copyOf(skippedTargets))) {
      if (AliasProvider.isAlias(target)) {
        continue;
      }
      targetFailure(asKey(target));
    }

    for (ConfiguredTarget target : skippedTargets) {
      if (AliasProvider.isAlias(target)) {
        continue;
      }
      targetSkipped(asKey(target));
    }
  }

  @Subscribe
  public void buildCompleteEvent(BuildCompleteEvent event) {
    BuildResult result = event.getResult();
    if (result.wasCatastrophe()) {
      blazeHalted = true;
    }
    skipTargetsOnFailure = result.getStopOnFirstFailure();
    buildComplete(
        result.getActualTargets(), result.getSkippedTargets(), result.getSuccessfulTargets());
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
   * Prints out the results of the given tests, and returns a {@link DetailedExitCode} summarizing
   * those test results. Posts any targets which weren't already completed by the listener to the
   * EventBus. Reports all targets on the console via the given notifier. Run at the end of the
   * build, run only once.
   *
   * @param testTargets The list of targets being run
   * @param validatedTargets targets with ValidateTarget aspect success or null if aspect not used
   * @param notifier A console notifier to echo results to.
   * @return true if all the tests passed, else false
   */
  public DetailedExitCode differentialAnalyzeAndReport(
      Collection<ConfiguredTarget> testTargets,
      Collection<ConfiguredTarget> skippedTargets,
      @Nullable ImmutableSet<ConfiguredTargetKey> validatedTargets,
      TestResultNotifier notifier) {
    Preconditions.checkNotNull(testTargets);
    Preconditions.checkNotNull(notifier);

    // The natural ordering of the summaries defines their output order.
    Set<TestSummary> summaries = Sets.newTreeSet();

    int totalRun = 0; // Number of targets running at least one non-cached test.
    int passCount = 0;

    DetailedExitCode systemFailure = null;
    for (ConfiguredTarget testTarget : testTargets) {
      ConfiguredTargetKey key = asKey(testTarget);
      TestResultAggregator aggregator =
          Preconditions.checkNotNull(
              aggregators.get(key), "Missing aggregator (key=%s, testTarget=%s)", key, testTarget);
      TestSummary summary;
      if (AliasProvider.isAlias(testTarget)) {
        TestSummary.Builder summaryBuilder = TestSummary.newBuilder(testTarget);
        summaryBuilder.mergeFrom(aggregator.aggregateAndReportSummary(skipTargetsOnFailure));
        summary = summaryBuilder.build();
      } else {
        summary = aggregator.aggregateAndReportSummary(skipTargetsOnFailure);
      }

      if (validatedTargets != null
          && summary.getStatus() != BlazeTestStatus.NO_STATUS
          && !validatedTargets.contains(key)) {
        // Approximate what targetFailure() would do for test targets that failed validation for
        // the purposes of printing test results to console only. Note that absent -k,
        // targetFailure() ends up marking one test as FAILED_TO_BUILD before buildComplete() marks
        // the remaining targets NO_STATUS. While we could approximate that, for simplicity, we
        // just use NO_STATUS for all tests with failed validations for simplicity here (absent -k).
        // Events published on BEP are not affected by this, but validation failures are published
        // as separate events and are additionally accounted in TargetSummary BEP messages.
        TestSummary.Builder summaryBuilder = TestSummary.newBuilder(summary.getTarget());
        summaryBuilder.mergeFrom(summary);
        summaryBuilder.setStatus(
            skipTargetsOnFailure
                ? BlazeTestStatus.NO_STATUS
                : TestResultAggregator.aggregateStatus(
                    summary.getStatus(), BlazeTestStatus.FAILED_TO_BUILD));
        summary = summaryBuilder.build();
      }

      summaries.add(summary);

      // Finished aggregating; build the final console output.
      if (summary.actionRan()) {
        totalRun++;
      }

      if (TestResult.isBlazeTestStatusPassed(summary.getStatus())) {
        passCount++;
      }

      systemFailure =
          DetailedExitCodeComparator.chooseMoreImportantWithFirstIfTie(
              systemFailure, summary.getSystemFailure());
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

    if (systemFailure != null) {
      return systemFailure;
    }

    // skipped targets are not in passCount since they have NO_STATUS
    Set<ConfiguredTarget> testTargetsSet = new HashSet<>(testTargets);
    Set<ConfiguredTarget> skippedTargetsSet = new HashSet<>(skippedTargets);

    return passCount == Sets.difference(testTargetsSet, skippedTargetsSet).size()
        ? DetailedExitCode.success()
        : TESTS_FAILED_DETAILED_CODE;
  }

  private static ConfiguredTargetKey asKey(ConfiguredTarget target) {
    return ConfiguredTargetKey.builder()
        .setLabel(target.getLabel())
        .setConfigurationKey(target.getActual().getConfigurationKey())
        .build();
  }
}
