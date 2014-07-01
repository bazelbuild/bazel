// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.blaze;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.exec.TestLogHelper;
import com.google.devtools.build.lib.exec.TestStrategy.TestOutputFormat;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.io.AnsiTerminalPrinter;
import com.google.devtools.build.lib.util.io.AnsiTerminalPrinter.Mode;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.FilesToRunProvider;
import com.google.devtools.build.lib.view.test.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestResult;
import com.google.devtools.build.lib.view.test.TestResult.FailedTestCaseDetails;
import com.google.devtools.build.lib.view.test.TestResult.FailedTestCaseDetailsStatus;
import com.google.devtools.build.lib.view.test.TestResult.TestCaseStatus;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;

/**
 * Test summary entry. Stores summary information for a single test rule.
 * Also used to sort summary output by status.
 *
 * <p>Invariant:
 * All TestSummary mutations should be performed through the Builder.
 * No direct TestSummary methods (except the constructor) may mutate the object.
 */
@VisibleForTesting // Ideally package-scoped.
public class TestSummary implements Comparable<TestSummary> {
  /**
   * Builder class responsible for creating and altering TestSummary objects.
   */
  public static class Builder {
    private TestSummary summary;
    private boolean built;

    private Builder() {
      summary = new TestSummary();
      built = false;
    }

    private void mergeFrom(TestSummary existingSummary) {
      // Yuck, manually fill in fields.
      summary.shardRunStatuses = ArrayListMultimap.create(existingSummary.shardRunStatuses);
      setTarget(existingSummary.target);
      setStatus(existingSummary.status);
      addCoverageFiles(existingSummary.coverageFiles);
      addPassedLogs(existingSummary.passedLogs);
      addFailedLogs(existingSummary.failedLogs);

      if (existingSummary.failedTestCases != null) {
        addFailedTestCases(existingSummary.failedTestCases);
      }

      addTestTimes(existingSummary.testTimes);
      addWarnings(existingSummary.warnings);
      setActionRan(existingSummary.actionRan);
      setNumCached(existingSummary.numCached);
      setRanRemotely(existingSummary.ranRemotely);
      setWasUnreportedWrongSize(existingSummary.wasUnreportedWrongSize);
    }

    // Implements copy on write logic, allowing reuse of the same builder.
    private void checkMutation() {
      // If mutating the builder after an object was built, create another copy.
      if (built) {
        built = false;
        TestSummary lastSummary = summary;
        summary = new TestSummary();
        mergeFrom(lastSummary);
      }
    }

    // This used to return a reference to the value on success.
    // However, since it can alter the summary member, inlining it in an
    // assignment to a property of summary was unsafe.
    private void checkMutation(Object value) {
      Preconditions.checkNotNull(value);
      checkMutation();
    }

    public Builder setTarget(ConfiguredTarget target) {
      checkMutation(target);
      summary.target = target;
      return this;
    }

    public Builder setStatus(BlazeTestStatus status) {
      checkMutation(status);
      summary.status = status;
      return this;
    }

    public Builder addCoverageFiles(List<Path> coverageFiles) {
      checkMutation(coverageFiles);
      summary.coverageFiles.addAll(coverageFiles);
      return this;
    }

    public Builder addPassedLogs(List<Path> passedLogs) {
      checkMutation(passedLogs);
      summary.passedLogs.addAll(passedLogs);
      return this;
    }

    public Builder addFailedLogs(List<Path> failedLogs) {
      checkMutation(failedLogs);
      summary.failedLogs.addAll(failedLogs);
      return this;
    }

    public Builder addFailedTestCases(FailedTestCaseDetails failedTestCases) {
      checkMutation(failedTestCases);

      if (summary.failedTestCases == null) {
        summary.failedTestCases = new FailedTestCaseDetails(failedTestCases.getStatus());
      }
      summary.failedTestCases.mergeFrom(failedTestCases);
      return this;
    }

    public Builder addTestTimes(List<Long> testTimes) {
      checkMutation(testTimes);
      summary.testTimes.addAll(testTimes);
      return this;
    }

    public Builder addWarnings(List<String> warnings) {
      checkMutation(warnings);
      summary.warnings.addAll(warnings);
      return this;
    }

    public Builder setActionRan(boolean actionRan) {
      checkMutation();
      summary.actionRan = actionRan;
      return this;
    }

    public Builder setNumCached(int numCached) {
      checkMutation();
      summary.numCached = numCached;
      return this;
    }

    public Builder setNumLocalActionCached(int numLocalActionCached) {
      checkMutation();
      summary.numLocalActionCached = numLocalActionCached;
      return this;
    }

    public Builder setRanRemotely(boolean ranRemotely) {
      checkMutation();
      summary.ranRemotely = ranRemotely;
      return this;
    }

    public Builder setWasUnreportedWrongSize(boolean wasUnreportedWrongSize) {
      checkMutation();
      summary.wasUnreportedWrongSize = wasUnreportedWrongSize;
      return this;
    }

    /**
     * Records a new result for the given shard of the test.
     *
     * @return an immutable view of the statuses associated with the shard, with the new element.
     */
    public List<BlazeTestStatus> addShardStatus(int shardNumber, BlazeTestStatus status) {
      Preconditions.checkState(summary.shardRunStatuses.put(shardNumber, status),
          "shardRunStatuses must allow duplicate statuses");
      return ImmutableList.copyOf(summary.shardRunStatuses.get(shardNumber));
    }

    /**
     * Returns the created TestSummary object.
     * Any actions following a build() will create another copy of the same values.
     * Since no mutators are provided directly by TestSummary, a copy will not
     * be produced if two builds are invoked in a row without calling a setter.
     */
    public TestSummary build() {
      peek();
      if (!built) {
        makeSummaryImmutable();
        // else: it is already immutable.
      }

      Preconditions.checkState(built, "Built flag was not set");
      return summary;
    }

    /**
     * Within-package, it is possible to read directly from an
     * incompletely-built TestSummary. Used to pass Builders around directly.
     */
    TestSummary peek() {
      Preconditions.checkNotNull(summary.target, "Target cannot be null");
      Preconditions.checkNotNull(summary.status, "Status cannot be null");
      return summary;
    }

    private void makeSummaryImmutable() {
      // Once finalized, the list types are immutable.
      summary.passedLogs = Collections.unmodifiableList(summary.passedLogs);
      summary.failedLogs = Collections.unmodifiableList(summary.failedLogs);
      summary.warnings = Collections.unmodifiableList(summary.warnings);
      summary.coverageFiles = Collections.unmodifiableList(summary.coverageFiles);
      summary.testTimes = Collections.unmodifiableList(summary.testTimes);

      built = true;
    }
  }

  private ConfiguredTarget target;
  private BlazeTestStatus status;
  // Currently only populated if --runs_per_test_detects_flakes is enabled.
  private Multimap<Integer, BlazeTestStatus> shardRunStatuses = ArrayListMultimap.create();
  private int numCached;
  private int numLocalActionCached;
  private boolean actionRan;
  private boolean ranRemotely;
  private boolean wasUnreportedWrongSize;
  private FailedTestCaseDetails failedTestCases;
  private List<Path> passedLogs = new ArrayList<>();
  private List<Path> failedLogs = new ArrayList<>();
  private List<String> warnings = new ArrayList<>();
  private List<Path> coverageFiles = new ArrayList<>();
  private List<Long> testTimes = new ArrayList<>();

  // Don't allow public instantiation; go through the Builder.
  private TestSummary() {
  }

  /**
   * Creates a new Builder allowing construction of a new TestSummary object.
   */
  public static Builder newBuilder() {
    return new Builder();
  }

  /**
   * Creates a new Builder initialized with a copy of the existing object's values.
   */
  public static Builder newBuilderFromExisting(TestSummary existing) {
    Builder builder = new Builder();
    builder.mergeFrom(existing);
    return builder;
  }

  public ConfiguredTarget getTarget() {
    return target;
  }

  public BlazeTestStatus getStatus() {
    return status;
  }

  public boolean isCached() {
    return numCached > 0;
  }

  public boolean isLocalActionCached() {
    return numLocalActionCached > 0;
  }

  public int numLocalActionCached() {
    return numLocalActionCached;
  }

  public int numCached() {
    return numCached;
  }

  private int numUncached() {
    return totalRuns() - numCached;
  }

  public boolean actionRan() {
    return actionRan;
  }

  public boolean ranRemotely() {
    return ranRemotely;
  }

  public boolean wasUnreportedWrongSize() {
    return wasUnreportedWrongSize;
  }

  /**
   * Returns an immutable view of the warnings associated with this test.
   */
  public List<String> getWarnings() {
    return Collections.unmodifiableList(warnings);
  }

  @Override
  public int compareTo(TestSummary that) {
    if (this.isCached() != that.isCached()) {
      return this.isCached() ? -1 : 1;
    } else if ((this.isCached() && that.isCached()) && (this.numUncached() != that.numUncached())) {
      return this.numUncached() - that.numUncached();
    } else if (this.status != that.status) {
      return this.status.getSortKey() - that.status.getSortKey();
    } else {
      Artifact thisExecutable = this.target.getProvider(FilesToRunProvider.class).getExecutable();
      Artifact thatExecutable = that.target.getProvider(FilesToRunProvider.class).getExecutable();
      return thisExecutable.getPath().compareTo(thatExecutable.getPath());
    }
  }

  private String getTimeSummary() {
    if (testTimes.isEmpty()) {
      return "";
    } else if (testTimes.size() == 1) {
      return " in " + timeInSec(testTimes.get(0), TimeUnit.MILLISECONDS);
    } else {
      // We previously used com.google.math for this, which added about 1 MB of deps to the total
      // size. If we re-introduce a dependency on that package, we could revert this change.
      long min = testTimes.get(0).longValue(), max = min, sum = 0;
      double sumOfSquares = 0.0;
      for (Long l : testTimes) {
        long value = l.longValue();
        min = value < min ? value : min;
        max = value > max ? value : max;
        sum += value;
        sumOfSquares += ((double) value) * (double) value;
      }
      double mean = ((double) sum) / testTimes.size();
      double stddev = Math.sqrt((sumOfSquares - sum * mean) / testTimes.size());
      // For sharded tests, we print the max time on the same line as
      // the test, and then print more detailed info about the
      // distribution of times on the next line.
      String maxTime = timeInSec(max, TimeUnit.MILLISECONDS);
      return String.format(
          " in %s\n  Stats over %d runs: max = %s, min = %s, avg = %s, dev = %s",
          maxTime,
          testTimes.size(),
          maxTime,
          timeInSec(min, TimeUnit.MILLISECONDS),
          timeInSec((long) mean, TimeUnit.MILLISECONDS),
          timeInSec((long) stddev, TimeUnit.MILLISECONDS));
    }
  }

  /**
   * Return the given time in seconds, to 1 decimal place,
   * ie "32.1s".
   */
  private static String timeInSec(long time, TimeUnit unit) {
    double ms = TimeUnit.MILLISECONDS.convert(time, unit);
    return String.format("%.1fs", ms / 1000.0);
  }

  private String getAttemptSummary() {
    int attempts = passedLogs.size() + failedLogs.size();
    if (attempts > 1) {
      // Print number of failed runs for failed tests if testing was completed.
      if (status == BlazeTestStatus.FLAKY) {
        return ", failed in " + failedLogs.size() + " out of " + attempts;
      }
      if (status == BlazeTestStatus.TIMEOUT || status == BlazeTestStatus.FAILED) {
        return " in " + failedLogs.size() + " out of " + attempts;
      }
    }
    return "";
  }

  /**
   * Prints the result of an individual test case. It is assumed not to have
   * passed, since passed test cases are not reported.
   */
  private void printTestCase(
      AnsiTerminalPrinter terminalPrinter, TestResult.TestCaseDetail testCase) {
    String timeSummary;
    if (testCase.getRunDurationMillis() != null) {
      timeSummary = " (" +
          timeInSec(testCase.getRunDurationMillis(), TimeUnit.MILLISECONDS) +
          ")";
    } else {
      timeSummary = "";
    }

    terminalPrinter.print(
        "    "
        + Mode.ERROR
        + Strings.padEnd(testCase.getStatus().toString(), 8, ' ')
        + Mode.DEFAULT
        + testCase.getName()
        + timeSummary
        + "\n");
  }

  private String getCacheMessage() {
    if (numCached == 0 || status == BlazeTestStatus.INCOMPLETE) {
      return "";
    } else if (numCached == totalRuns()) {
      return "(cached) ";
    } else {
      return String.format("(%d/%d cached) ", numCached, totalRuns());
    }
  }

  public int totalRuns() {
    return testTimes.size();
  }

  /**
   * Prints summary status for a single test.
   * @param terminalPrinter The printer to print to
   */
  public void print(AnsiTerminalPrinter terminalPrinter,
      boolean verboseSummary, boolean printFailedTestCases) {
    // Skip output for tests that failed to build.
    if (status == BlazeTestStatus.FAILED_TO_BUILD) {
      return;
    }
    String message = getCacheMessage() + status;
    terminalPrinter.print(
        Strings.padEnd(target.getLabel().toString(), 78 - message.length(), ' ')
        + " " + getStatusMode(status) + message + Mode.DEFAULT
        + (verboseSummary ? getAttemptSummary() + getTimeSummary() : "") + "\n");

    if (printFailedTestCases && status == BlazeTestStatus.FAILED) {
      if (failedTestCases.getStatus() == FailedTestCaseDetailsStatus.NOT_AVAILABLE) {
        terminalPrinter.print(
            Mode.WARNING + "    (individual test case information not available) " +
            Mode.DEFAULT + "\n");
      } else {
        for (TestResult.TestCaseDetail testCase : failedTestCases.getDetails()) {
          if (testCase.getStatus() != TestCaseStatus.PASSED) {
            printTestCase(terminalPrinter, testCase);
          }
        }

        if (failedTestCases.getStatus() != FailedTestCaseDetailsStatus.FULL) {
          terminalPrinter.print(
              Mode.WARNING
              + "    (some shards did not report details, list of failed test"
              + " cases incomplete)\n"
              + Mode.DEFAULT);
        }
      }
    }

    if (printFailedTestCases) {
      // In this mode, test output and coverage files would just clutter up
      // the output.
      return;
    }

    for (String warning : warnings) {
      terminalPrinter.print("  " + AnsiTerminalPrinter.Mode.WARNING +
          "WARNING: " + AnsiTerminalPrinter.Mode.DEFAULT + warning + "\n");
    }

    for (Path path : failedLogs) {
      if (path.exists()) {
        // Don't use getPrettyPath() here - we want to print the absolute path,
        // so that it cut and paste into a different terminal, and we don't
        // want to use the blaze-bin etc. symlinks because they could be changed
        // by a subsequent build with different options.
        terminalPrinter.print("  " + path.getPathString() + "\n");
      }
    }
    for (Path path : coverageFiles) {
      // Print only non-trivial coverage files.
      try {
        if (path.exists() && path.getFileSize() > 0) {
          terminalPrinter.print("  " + path.getPathString() + "\n");
        }
      } catch (IOException e) {
        LoggingUtil.logToRemote(Level.WARNING, "Error while reading coverage data file size",
            e);
      }
    }
  }

  private static Mode getStatusMode(BlazeTestStatus status) {
    return status == BlazeTestStatus.PASSED
        ? Mode.INFO
        : (status == BlazeTestStatus.FLAKY ? Mode.WARNING : Mode.ERROR);
  }

  /**
   * Print the cached test log to the given printer.
   */
  public void printCachedOutput(TestOutputFormat testOutput,
      AnsiTerminalPrinter printer) {

    String testName = target.getLabel().toString();
    List<String> allLogs = new ArrayList<>();
    for (Path path : failedLogs) {
      allLogs.add(path.getPathString());
    }
    for (Path path : passedLogs) {
      allLogs.add(path.getPathString());
    }
    printer.printLn("" + getStatusMode(status) + status + ": " + Mode.DEFAULT +
        testName + " (see " + Joiner.on(' ').join(allLogs) + ")");
    printer.printLn(Mode.INFO + "INFO: " + Mode.DEFAULT + "From Testing " + testName);

    // Whether to output the target at all was checked by the caller.
    // Now check whether to output failing shards.
    if (TestLogHelper.shouldOutputTestLog(testOutput, false)) {
      for (Path path : failedLogs) {
        try {
          TestLogHelper.writeTestLog(path, testName, printer.getOutputStream());
        } catch (IOException e) {
          printer.printLn("==================== Could not read test output for " + testName);
          LoggingUtil.logToRemote(Level.WARNING, "Error while reading test log", e);
        }
      }
    }

    // And passing shards, independently.
    if (TestLogHelper.shouldOutputTestLog(testOutput, true)) {
      for (Path path : passedLogs) {
        try {
          TestLogHelper.writeTestLog(path, testName, printer.getOutputStream());
        } catch (Exception e) {
          printer.printLn("==================== Could not read test output for " + testName);
          LoggingUtil.logToRemote(Level.WARNING, "Error while reading test log", e);
        }
      }
    }
  }
}
