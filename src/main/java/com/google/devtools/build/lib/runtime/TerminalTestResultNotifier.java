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

import com.google.devtools.build.lib.analysis.test.TestResult;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.TestLogHelper;
import com.google.devtools.build.lib.exec.TestStrategy.TestOutputFormat;
import com.google.devtools.build.lib.exec.TestStrategy.TestSummaryFormat;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.util.io.AnsiTerminalPrinter;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Prints the test results to a terminal.
 */
public class TerminalTestResultNotifier implements TestResultNotifier {
  private static class TestResultStats {
    int numberOfTargets;
    int passCount;
    int failedToBuildCount;
    int failedCount;
    int failedRemotelyCount;
    int failedLocallyCount;
    int noStatusCount;
    int numberOfExecutedTargets;
    boolean wasUnreportedWrongSize;
  }

  /**
   * Flags specific to test summary reporting.
   */
  public static class TestSummaryOptions extends OptionsBase {
    @Option(
      name = "verbose_test_summary",
      defaultValue = "true",
      category = "verbosity",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "If true, print additional information (timing, number of failed runs, etc) in the"
              + " test summary."
    )
    public boolean verboseSummary;

    @Option(
      name = "test_verbose_timeout_warnings",
      defaultValue = "false",
      category = "verbosity",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "If true, print additional warnings when the actual test execution time does not "
              + "match the timeout defined by the test (whether implied or explicit)."
    )
    public boolean testVerboseTimeoutWarnings;
  }

  private final AnsiTerminalPrinter printer;
  private final OptionsProvider options;
  private final TestSummaryOptions summaryOptions;

  /**
   * @param printer The terminal to print to
   */
  public TerminalTestResultNotifier(AnsiTerminalPrinter printer, OptionsProvider options) {
    this.printer = printer;
    this.options = options;
    this.summaryOptions = options.getOptions(TestSummaryOptions.class);
  }

  /**
   * Decide if two tests with the same label are contained in the set of test summaries
   */
  private boolean duplicateLabels(Set<TestSummary> summaries) {
    Set<Label> labelsSeen = new HashSet<>();
    for (TestSummary summary : summaries) {
      if (labelsSeen.contains(summary.getTarget().getLabel())) {
        return true;
      }
      labelsSeen.add(summary.getTarget().getLabel());
    }
    return false;
  }

  /**
   * Prints a test result summary that contains only failed tests.
   */
  private void printDetailedTestResultSummary(Set<TestSummary> summaries) {
    boolean withConfig = duplicateLabels(summaries);
    for (TestSummary summary : summaries) {
      if (summary.getStatus() != BlazeTestStatus.PASSED) {
        TestSummaryPrinter.print(summary, printer, summaryOptions.verboseSummary, true, withConfig);
      }
    }
  }

  /**
   * Prints a full test result summary.
   */
  private void printShortSummary(Set<TestSummary> summaries, boolean showPassingTests) {
    boolean withConfig = duplicateLabels(summaries);
    for (TestSummary summary : summaries) {
      if ((summary.getStatus() != BlazeTestStatus.PASSED
              && summary.getStatus() != BlazeTestStatus.NO_STATUS)
          || showPassingTests) {
        TestSummaryPrinter.print(summary, printer, summaryOptions.verboseSummary, false,
            withConfig);
      }
    }
  }

  /**
   * Returns true iff the --check_tests_up_to_date option is enabled.
   */
  private boolean optionCheckTestsUpToDate() {
    return options.getOptions(ExecutionOptions.class).testCheckUpToDate;
  }


  /**
   * Prints a test summary information for all tests to the terminal.
   *
   * @param summaries Summary of all targets that were ran
   * @param numberOfExecutedTargets the number of targets that were actually ran
   */
  @Override
  public void notify(Set<TestSummary> summaries, int numberOfExecutedTargets) {
    TestResultStats stats = new TestResultStats();
    stats.numberOfTargets = summaries.size();
    stats.numberOfExecutedTargets = numberOfExecutedTargets;

    TestOutputFormat testOutput = options.getOptions(ExecutionOptions.class).testOutput;

    for (TestSummary summary : summaries) {
      if (summary.isLocalActionCached()
          && TestLogHelper.shouldOutputTestLog(testOutput,
              TestResult.isBlazeTestStatusPassed(summary.getStatus()))) {
        TestSummaryPrinter.printCachedOutput(summary, testOutput, printer);
      }
    }

    for (TestSummary summary : summaries) {
      if (TestResult.isBlazeTestStatusPassed(summary.getStatus())) {
        stats.passCount++;
      } else if (summary.getStatus() == BlazeTestStatus.NO_STATUS) {
        stats.noStatusCount++;
      } else if (summary.getStatus() == BlazeTestStatus.FAILED_TO_BUILD) {
        stats.failedToBuildCount++;
      } else if (summary.ranRemotely()) {
        stats.failedRemotelyCount++;
      } else {
        stats.failedLocallyCount++;
      }

      if (summary.wasUnreportedWrongSize()) {
        stats.wasUnreportedWrongSize = true;
      }
    }

    stats.failedCount = summaries.size() - stats.passCount;

    TestSummaryFormat testSummaryFormat = options.getOptions(ExecutionOptions.class).testSummary;
    switch (testSummaryFormat) {
      case DETAILED:
        printDetailedTestResultSummary(summaries);
        break;

      case SHORT:
        printShortSummary(summaries, /* showPassingTests= */ true);
        break;

      case TERSE:
        printShortSummary(summaries, /* showPassingTests= */ false);
        break;

      case NONE:
        break;
    }

    printStats(stats);
  }

  private void addFailureToErrorList(List<String> list, String failureDescription, int count) {
    addToErrorList(list, "fails", "fail", failureDescription, count);
  }

  private void addToErrorList(
      List<String> list, String singularPrefix, String pluralPrefix, String message, int count) {
    if (count > 0) {
      list.add(String.format("%s%d %s %s%s", AnsiTerminalPrinter.Mode.ERROR, count,
          count == 1 ? singularPrefix : pluralPrefix, message, AnsiTerminalPrinter.Mode.DEFAULT));
    }
  }

  private void printStats(TestResultStats stats) {
    if (!optionCheckTestsUpToDate()) {
      List<String> results = new ArrayList<>();
      if (stats.passCount == 1) {
        results.add(stats.passCount + " test passes");
      } else if (stats.passCount > 0) {
        results.add(stats.passCount + " tests pass");
      }
      addFailureToErrorList(results, "to build", stats.failedToBuildCount);
      addFailureToErrorList(results, "locally", stats.failedLocallyCount);
      addFailureToErrorList(results, "remotely", stats.failedRemotelyCount);
      addToErrorList(results, "was", "were", "skipped", stats.noStatusCount);
      printer.print(String.format("\nExecuted %d out of %d %s: %s.\n",
              stats.numberOfExecutedTargets,
              stats.numberOfTargets,
              stats.numberOfTargets == 1 ? "test" : "tests",
              StringUtil.joinEnglishList(results, "and")));
    } else {
      int failingUpToDateCount = stats.failedCount - stats.noStatusCount;
      printer.print(String.format(
          "\nFinished with %d passing and %s%d failing%s tests up to date, %s%d out of date.%s\n",
          stats.passCount,
          failingUpToDateCount > 0 ? AnsiTerminalPrinter.Mode.ERROR : "",
          failingUpToDateCount,
          AnsiTerminalPrinter.Mode.DEFAULT,
          stats.noStatusCount > 0 ? AnsiTerminalPrinter.Mode.ERROR : "",
          stats.noStatusCount,
          AnsiTerminalPrinter.Mode.DEFAULT));
    }

    if (stats.wasUnreportedWrongSize) {
       printer.print("There were tests whose specified size is too big. Use the "
           + "--test_verbose_timeout_warnings command line option to see which "
           + "ones these are.\n");
     }
  }
}
