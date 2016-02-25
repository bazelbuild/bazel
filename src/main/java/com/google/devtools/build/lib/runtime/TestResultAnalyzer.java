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

import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.build.lib.rules.test.TestProvider;
import com.google.devtools.build.lib.rules.test.TestResult;
import com.google.devtools.build.lib.runtime.TerminalTestResultNotifier.TestSummaryOptions;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Prints results to the terminal, showing the results of each test target.
 */
@ThreadCompatible
public class TestResultAnalyzer {
  private final Path execRoot;
  private final TestSummaryOptions summaryOptions;
  private final ExecutionOptions executionOptions;
  private final EventBus eventBus;

  /**
   * @param summaryOptions Parsed test summarization options.
   * @param executionOptions Parsed build/test execution options.
   * @param eventBus For reporting failed to build and cached tests.
   */
  public TestResultAnalyzer(Path execRoot,
                            TestSummaryOptions summaryOptions,
                            ExecutionOptions executionOptions,
                            EventBus eventBus) {
    this.execRoot = execRoot;
    this.summaryOptions = summaryOptions;
    this.executionOptions = executionOptions;
    this.eventBus = eventBus;
  }

  /**
   * Prints out the results of the given tests, and returns true if they all passed.
   * Posts any targets which weren't already completed by the listener to the EventBus.
   * Reports all targets on the console via the given notifier.
   * Run at the end of the build, run only once.
   *
   * @param testTargets The list of targets being run
   * @param listener An aggregating listener with intermediate results
   * @param notifier A console notifier to echo results to.
   * @return true if all the tests passed, else false
   */
  public boolean differentialAnalyzeAndReport(
      Collection<ConfiguredTarget> testTargets,
      AggregatingTestListener listener,
      TestResultNotifier notifier) {

    Preconditions.checkNotNull(testTargets);
    Preconditions.checkNotNull(listener);
    Preconditions.checkNotNull(notifier);

    // The natural ordering of the summaries defines their output order.
    Set<TestSummary> summaries = Sets.newTreeSet();

    int totalRun = 0; // Number of targets running at least one non-cached test.
    int passCount = 0;

    for (ConfiguredTarget testTarget : testTargets) {
      TestSummary summary = aggregateAndReportSummary(testTarget, listener).build();
      summaries.add(summary);

      // Finished aggregating; build the final console output.
      if (summary.actionRan()) {
        totalRun++;
      }

      if (TestResult.isBlazeTestStatusPassed(summary.getStatus())) {
        passCount++;
      }
    }

    Preconditions.checkState(summaries.size() == testTargets.size());

    notifier.notify(summaries, totalRun);
    return passCount == testTargets.size();
  }

  private static BlazeTestStatus aggregateStatus(BlazeTestStatus status, BlazeTestStatus other) {
    return status.ordinal() > other.ordinal() ? status : other;
  }

  /**
   * Helper for differential analysis which aggregates the TestSummary
   * for an individual target, reporting runs on the EventBus if necessary.
   */
  private TestSummary.Builder aggregateAndReportSummary(
      ConfiguredTarget testTarget,
      AggregatingTestListener listener) {

    // If already reported by the listener, no work remains for this target.
    TestSummary.Builder summary = listener.getCurrentSummary(testTarget);
    Label testLabel = testTarget.getLabel();
    Preconditions.checkNotNull(summary,
        "%s did not complete test filtering, but has a test result", testLabel);
    if (listener.targetReported(testTarget)) {
      return summary;
    }

    Collection<Artifact> incompleteRuns = listener.getIncompleteRuns(testTarget);
    Map<Artifact, TestResult> statusMap = listener.getStatusMap();

    // We will get back multiple TestResult instances if test had to be retried several
    // times before passing. Sharding and multiple runs of the same test without retries
    // will be represented by separate artifacts and will produce exactly one TestResult.
    for (Artifact testStatus : TestProvider.getTestStatusArtifacts(testTarget)) {
      // When a build is interrupted ( eg. a broken target with --nokeep_going ) runResult could
      // be null for an unrelated test because we were not able to even try to execute the test.
      // In that case, for tests that were previously passing we return null ( == NO STATUS),
      // because checking if the cached test target is up-to-date would require running the
      // dependency checker transitively.
      TestResult runResult = statusMap.get(testStatus);
      boolean isIncompleteRun = incompleteRuns.contains(testStatus);
      if (runResult == null) {
        summary = markIncomplete(summary);
      } else if (isIncompleteRun) {
        // Only process results which were not recorded by the listener.

        boolean newlyFetched = !statusMap.containsKey(testStatus);
        summary = incrementalAnalyze(summary, runResult);
        if (newlyFetched) {
          eventBus.post(runResult);
        }
        Preconditions.checkState(
            listener.getIncompleteRuns(testTarget).contains(testStatus) == isIncompleteRun,
            "TestListener changed in differential analysis. Ensure it isn't still registered.");
      }
    }

    // The target was not posted by the listener and must be posted now.
    eventBus.post(summary.build());
    return summary;
  }

  /**
   * Incrementally updates a TestSummary given an existing summary
   * and a new TestResult. Only call on built targets.
   *
   * @param summaryBuilder Existing unbuilt test summary associated with a target.
   * @param result New test result to aggregate into the summary.
   * @return The updated TestSummary.
   */
  public TestSummary.Builder incrementalAnalyze(TestSummary.Builder summaryBuilder,
                                                TestResult result) {
    // Cache retrieval should have been performed already.
    Preconditions.checkNotNull(result);
    Preconditions.checkNotNull(summaryBuilder);
    TestSummary existingSummary = Preconditions.checkNotNull(summaryBuilder.peek());

    TransitiveInfoCollection target = existingSummary.getTarget();
    Preconditions.checkNotNull(
        target, "The existing TestSummary must be associated with a target");

    BlazeTestStatus status = existingSummary.getStatus();
    int numCached = existingSummary.numCached();
    int numLocalActionCached = existingSummary.numLocalActionCached();

    // If a test was neither cached locally nor remotely we say action was taken.
    if (!(result.isCached() || result.getData().getRemotelyCached())) {
      summaryBuilder.setActionRan(true);
    } else {
      numCached++;
    }
    
    if (result.isCached()) {
      numLocalActionCached++;
    }
    
    PathFragment coverageData = result.getCoverageData();
    if (coverageData != null) {
      summaryBuilder.addCoverageFiles(
          Collections.singletonList(execRoot.getRelative(coverageData)));
    }

    if (!executionOptions.runsPerTestDetectsFlakes) {
      status = aggregateStatus(status, result.getData().getStatus());
    } else {
      int shardNumber = result.getShardNum();
      int runsPerTestForLabel = target.getProvider(TestProvider.class).getTestParams().getRuns();
      List<BlazeTestStatus> singleShardStatuses = summaryBuilder.addShardStatus(
          shardNumber, result.getData().getStatus());
      if (singleShardStatuses.size() == runsPerTestForLabel) {
        BlazeTestStatus shardStatus = BlazeTestStatus.NO_STATUS;
        int passes = 0;
        for (BlazeTestStatus runStatusForShard : singleShardStatuses) {
          shardStatus = aggregateStatus(shardStatus, runStatusForShard);
          if (TestResult.isBlazeTestStatusPassed(runStatusForShard)) {
            passes++;
          }
        }
        // Under the RunsPerTestDetectsFlakes option, return flaky if 1 <= p < n shards pass.
        // If all results pass or fail, aggregate the passing/failing shardStatus.
        if (passes == 0 || passes == runsPerTestForLabel) {
          status = aggregateStatus(status, shardStatus);
        } else {
          status = aggregateStatus(status, BlazeTestStatus.FLAKY);
        }
      }
    }

    List<Path> passed = new ArrayList<>();
    if (result.getData().hasPassedLog()) {
      passed.add(result.getTestAction().getTestLog().getPath().getRelative(
          result.getData().getPassedLog()));
    }

    List<Path> failed = new ArrayList<>();
    for (String path : result.getData().getFailedLogsList()) {
      failed.add(result.getTestAction().getTestLog().getPath().getRelative(path));
    }

    summaryBuilder
        .addTestTimes(result.getData().getTestTimesList())
        .addPassedLogs(passed)
        .addFailedLogs(failed)
        .addWarnings(result.getData().getWarningList())
        .collectFailedTests(result.getData().getTestCase())
        .setRanRemotely(result.getData().getIsRemoteStrategy());

    List<String> warnings = new ArrayList<>();
    if (status == BlazeTestStatus.PASSED
        && shouldEmitTestSizeWarningInSummary(
            summaryOptions.testVerboseTimeoutWarnings,
            warnings,
            result.getData().getTestProcessTimesList(),
            target)) {
      summaryBuilder.setWasUnreportedWrongSize(true);
    }

    return summaryBuilder
        .setStatus(status)
        .setNumCached(numCached)
        .setNumLocalActionCached(numLocalActionCached)
        .addWarnings(warnings);
  }

  private TestSummary.Builder markIncomplete(TestSummary.Builder summaryBuilder) {
    // TODO(bazel-team): (2010) Make NotRunTestResult support both tests failed to built and
    // tests with no status and post it here.
    TestSummary summary = summaryBuilder.peek();
    BlazeTestStatus status = summary.getStatus();
    if (status != BlazeTestStatus.NO_STATUS) {
      status = aggregateStatus(status, BlazeTestStatus.INCOMPLETE);
    }

    return summaryBuilder.setStatus(status);
  }

  TestSummary.Builder markUnbuilt(
      TestSummary.Builder summary, boolean blazeHalted, boolean stopOnFirstFailure) {
    // stopOnFirstFailure = true means that at least on of the options keep_going and
    // test_keep_going is set to false.
    // Consequently, we mark all unbuilt targets with NO_STATUS instead of FAILED_TO_BUILD in 
    // order to indicate that Blaze has skipped these targets.
    BlazeTestStatus runStatus =
        blazeHalted 
            ? BlazeTestStatus.BLAZE_HALTED_BEFORE_TESTING : (
                executionOptions.testCheckUpToDate || stopOnFirstFailure
                    ? BlazeTestStatus.NO_STATUS : BlazeTestStatus.FAILED_TO_BUILD);

    return summary.setStatus(runStatus);
  }

  /**
   * Checks whether the specified test timeout could have been smaller and adds
   * a warning message if verbose is true.
   *
   * <p>Returns true if there was a test with the wrong timeout, but if was not
   * reported.
   */
  private static boolean shouldEmitTestSizeWarningInSummary(boolean verbose,
      List<String> warnings, List<Long> testTimes, TransitiveInfoCollection target) {

    TestTimeout specifiedTimeout =
        target.getProvider(TestProvider.class).getTestParams().getTimeout();
    long maxTimeOfShard = 0;

    for (Long shardTime : testTimes) {
      if (shardTime != null) {
        maxTimeOfShard = Math.max(maxTimeOfShard, shardTime);
      }
    }

    int maxTimeInSeconds = (int) (maxTimeOfShard / 1000);

    if (!specifiedTimeout.isInRangeFuzzy(maxTimeInSeconds)) {
      TestTimeout expectedTimeout = TestTimeout.getSuggestedTestTimeout(maxTimeInSeconds);
      TestSize expectedSize = TestSize.getTestSize(expectedTimeout);
      if (verbose) {
        StringBuilder builder = new StringBuilder(String.format(
            "%s: Test execution time (%.1fs excluding execution overhead) outside of "
            + "range for %s tests. Consider setting timeout=\"%s\"",
            target.getLabel(),
            maxTimeOfShard / 1000.0,
            specifiedTimeout.prettyPrint(),
            expectedTimeout));
        if (expectedSize != null) {
          builder.append(" or size=\"").append(expectedSize).append("\"");
        }
        builder.append(".");
        warnings.add(builder.toString());
        return false;
      }
      return true;
    } else {
      return false;
    }
  }
}
