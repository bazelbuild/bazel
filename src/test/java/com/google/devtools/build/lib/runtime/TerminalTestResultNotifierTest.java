// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.ExecutionOptions.TestSummaryFormat;
import com.google.devtools.build.lib.util.io.AnsiTerminalPrinter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.TestCase;
import com.google.devtools.build.lib.view.test.TestStatus.TestCase.Status;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;

/** Tests for {@link TerminalTestResultNotifier}. */
@RunWith(JUnit4.class)
public final class TerminalTestResultNotifierTest {

  private static final String SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER =
      "some targets did not have test case information";

  private final OptionsParsingResult optionsParsingResult = mock(OptionsParsingResult.class);
  private final AnsiTerminalPrinter ansiTerminalPrinter = mock(AnsiTerminalPrinter.class);

  @Test
  public void testCaseOption_allPass() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.PASSED,
      /*failedTestCases=*/ 0,
      /*skippedTestCases=*/ 0,
      /*unknownTestCases=*/ 0,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.TESTCASE);

    String printed = getPrintedMessage();
    assertThat(printed).contains(info("10 passing"));
    assertThat(printed).contains("0 failing");
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.ERROR.toString());
  }

  @Test
  public void testCaseOption_allPassButTargetFails() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.FAILED,
      /*failedTestCases=*/ 0,
      /*skippedTestCases=*/ 0,
      /*unknownTestCases=*/ 10,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.TESTCASE);

    String printed = getPrintedMessage();
    assertThat(printed).contains("0 passing");
    assertThat(printed).contains("0 failing");
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).contains(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.ERROR.toString());
  }

  @Test
  public void testCaseOption_someFail() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.FAILED,
      /*failedTestCases=*/ 2,
      /*skippedTestCases=*/ 0,
      /*unknownTestCases=*/ 0,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.TESTCASE);

    String printed = getPrintedMessage();
    assertThat(printed).contains(info("8 passing"));
    assertThat(printed).contains(error("2 failing"));
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
  }

  @Test
  public void shortOption_someFailToBuild() throws Exception {
    int numFailedToBuildTestCases = TerminalTestResultNotifier.NUM_FAILED_TO_BUILD + 1;

    printFailedToBuildSummaries(ExecutionOptions.TestSummaryFormat.SHORT);

    String skippedMessage = getPrintedMessage();
    assertThat(skippedMessage).isEqualTo("(Skipping other failed to build tests)");

    ArgumentCaptor<String> messageCaptor = ArgumentCaptor.forClass(String.class);
    verify(ansiTerminalPrinter, times(numFailedToBuildTestCases)).print(messageCaptor.capture());
    List<String> values = messageCaptor.getAllValues();

    for (int i = 0; i < numFailedToBuildTestCases - 1; i++) {
      String message = values.get(i);
      assertThat(message).contains("//foo/bar:baz");
      assertThat(message).contains(BlazeTestStatus.FAILED_TO_BUILD.toString().replace('_', ' '));
    }

    String last = values.get(numFailedToBuildTestCases - 1);
    assertThat(last).contains("Executed 0 out of 6 tests");
    assertThat(last).contains(numFailedToBuildTestCases + " fail to build");
  }

  @Test
  public void shortUncachedOption_someFailToBuild() throws Exception {
    int numFailedToBuildTestCases = TerminalTestResultNotifier.NUM_FAILED_TO_BUILD + 1;

    printFailedToBuildSummaries(ExecutionOptions.TestSummaryFormat.SHORT_UNCACHED);

    String skippedMessage = getPrintedMessage();
    assertThat(skippedMessage).isEqualTo("(Skipping other failed to build tests)");

    ArgumentCaptor<String> messageCaptor = ArgumentCaptor.forClass(String.class);
    verify(ansiTerminalPrinter, times(numFailedToBuildTestCases)).print(messageCaptor.capture());// 1 but should all be printed
    List<String> values = messageCaptor.getAllValues();

    for (int i = 0; i < numFailedToBuildTestCases - 1; i++) {
      String message = values.get(i);
      assertThat(message).contains("//foo/bar:baz");
      assertThat(message).contains(BlazeTestStatus.FAILED_TO_BUILD.toString().replace('_', ' '));
    }

    String last = values.get(numFailedToBuildTestCases - 1);
    assertThat(last).contains("Executed 0 out of 6 tests");
    assertThat(last).contains(numFailedToBuildTestCases + " fail to build");
  }

  @Test
  public void testCaseOption_allFail() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.FAILED,
      /*failedTestCases=*/ 10,
      /*skippedTestCases=*/ 0,
      /*unknownTestCases=*/ 0,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.TESTCASE);

    String printed = getPrintedMessage();
    assertThat(printed).contains("0 passing");
    assertThat(printed).contains(error("10 failing"));
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.INFO.toString());
  }

  @Test
  public void detailedOption_allPass() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.PASSED,
      /*failedTestCases=*/ 0,
      /*skippedTestCases=*/ 0,
      /*unknownTestCases=*/ 0,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.DETAILED);

    String printed = getPrintedMessage();
    assertThat(printed).contains(info("10 passing"));
    assertThat(printed).contains("0 failing");
    assertThat(printed).contains("0 skipped");
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.ERROR.toString());
  }

  @Test
  public void detailedUncachedOption_allPassUncached() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.PASSED,
      /*failedTestCases=*/ 0,
      /*skippedTestCases=*/ 0,
      /*unknownTestCases=*/ 0,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.DETAILED_UNCACHED);

    String printed = getPrintedMessage();
    assertThat(printed).contains(info("10 passing"));
    assertThat(printed).contains("0 failing");
    assertThat(printed).contains("0 skipped");
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.ERROR.toString());
  }

  @Test
  public void detailedUncachedOption_allPassCached() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.PASSED,
      /*failedTestCases=*/ 0,
      /*skippedTestCases=*/ 0,
      /*unknownTestCases=*/ 0,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ false
    ), ExecutionOptions.TestSummaryFormat.DETAILED_UNCACHED);

    String printed = getPrintedMessage();
    assertThat(printed).contains(info("10 passing"));
    assertThat(printed).contains("0 failing");
    assertThat(printed).contains("0 skipped");
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.ERROR.toString());
  }

  @Test
  public void detailedOption_allPassButSomeSkipped() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.PASSED,
      /*failedTestCases=*/ 0,
      /*skippedTestCases=*/ 2,
      /*unknownTestCases=*/ 0,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.DETAILED);

    String printed = getPrintedMessage();
    assertThat(printed).contains(info("8 passing"));
    assertThat(printed).contains("0 failing");
    assertThat(printed).contains(warn("2 skipped"));
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.ERROR.toString());
  }

  @Test
  public void detailedUncachedOption_allPassUncachedButSomeSkipped() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.PASSED,
      /*failedTestCases=*/ 0,
      /*skippedTestCases=*/ 2,
      /*unknownTestCases=*/ 0,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.DETAILED_UNCACHED);

    String printed = getPrintedMessage();
    assertThat(printed).contains(info("8 passing"));
    assertThat(printed).contains("0 failing");
    assertThat(printed).contains(warn("2 skipped"));
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.ERROR.toString());
  }

  @Test
  public void detailedUncachedOption_allPassCachedButSomeSkipped() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.PASSED,
      /*failedTestCases=*/ 0,
      /*skippedTestCases=*/ 2,
      /*unknownTestCases=*/ 0,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ false
    ), ExecutionOptions.TestSummaryFormat.DETAILED_UNCACHED);

    String printed = getPrintedMessage();
    assertThat(printed).contains(info("8 passing"));
    assertThat(printed).contains("0 failing");
    assertThat(printed).contains(warn("2 skipped"));
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.ERROR.toString());
  }

  @Test
  public void detailedOption_allPassButTargetFails() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.FAILED,
      /*failedTestCases=*/ 0,
      /*skippedTestCases=*/ 0,
      /*unknownTestCases=*/ 10,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.DETAILED);

    String printed = getPrintedMessage();
    assertThat(printed).contains("0 passing");
    assertThat(printed).contains("0 failing");
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).contains(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.ERROR.toString());
  }

  @Test
  public void detailedUncachedOption_allPassButTargetFails() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.FAILED,
      /*failedTestCases=*/ 0,
      /*skippedTestCases=*/ 0,
      /*unknownTestCases=*/ 10,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.DETAILED_UNCACHED);

    String printed = getPrintedMessage();
    assertThat(printed).contains("0 passing");
    assertThat(printed).contains("0 failing");
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).contains(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.ERROR.toString());
  }

  @Test
  public void detailedOption_someFail() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.FAILED,
      /*failedTestCases=*/ 2,
      /*skippedTestCases=*/ 0,
      /*unknownTestCases=*/ 0,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.DETAILED);

    String printed = getPrintedMessage();
    assertThat(printed).contains(info("8 passing"));
    assertThat(printed).contains(error("2 failing"));
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
  }

  @Test
  public void detailedUncachedOption_someFail() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.FAILED,
      /*failedTestCases=*/ 2,
      /*skippedTestCases=*/ 0,
      /*unknownTestCases=*/ 0,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.DETAILED_UNCACHED);

    String printed = getPrintedMessage();
    assertThat(printed).contains(info("8 passing"));
    assertThat(printed).contains(error("2 failing"));
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
  }

  @Test
  public void detailedOption_allFail() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.FAILED,
      /*failedTestCases=*/ 10,
      /*skippedTestCases=*/ 0,
      /*unknownTestCases=*/ 0,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.DETAILED);

    String printed = getPrintedMessage();
    assertThat(printed).contains("0 passing");
    assertThat(printed).contains(error("10 failing"));
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.INFO.toString());
  }

  @Test
  public void detailedUncachedOption_allFail() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.FAILED,
      /*failedTestCases=*/ 10,
      /*skippedTestCases=*/ 0,
      /*unknownTestCases=*/ 0,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.DETAILED_UNCACHED);

    String printed = getPrintedMessage();
    assertThat(printed).contains("0 passing");
    assertThat(printed).contains(error("10 failing"));
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.INFO.toString());
  }

  @Test
  public void shortOption_noSummaryPrinted() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.FAILED,
      /*failedTestCases=*/ 2,
      /*skippedTestCases=*/ 0,
      /*unknownTestCases=*/ 0,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.SHORT);

    verifyNoSummaryPrinted();
  }

  @Test
  public void shortUncachedOption_noSummaryPrinted() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.FAILED,
      /*failedTestCases=*/ 2,
      /*skippedTestCases=*/ 0,
      /*unknownTestCases=*/ 0,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.SHORT_UNCACHED);

    verifyNoSummaryPrinted();
  }

  @Test
  public void terseOption_noSummaryPrinted() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.FAILED,
      /*failedTestCases=*/ 2,
      /*skippedTestCases=*/ 0,
      /*unknownTestCases=*/ 0,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.TERSE);

    verifyNoSummaryPrinted();
  }

  @Test
  public void noneOption_noSummaryPrinted() throws Exception {
    printTestCaseSummary(new TestSummarySpec(
      BlazeTestStatus.FAILED,
      /*failedTestCases=*/ 2,
      /*skippedTestCases=*/ 0,
      /*unknownTestCases=*/ 0,
      /*totalTestCases=*/ 10,
      /*actionRan=*/ true
    ), ExecutionOptions.TestSummaryFormat.NONE);

    verifyNoSummaryPrinted();
  }

  // A record that is used to generate a TestSummary mock object for testing.
  private static record TestSummarySpec(
    BlazeTestStatus status,
    int failedTestCases,
    int skippedTestCases,
    int unknownTestCases,
    int totalTestCases,
    boolean actionRan) {

      TestSummary build() throws LabelSyntaxException {
        TestSummary testSummary = mock(TestSummary.class);
        when(testSummary.getTotalTestCases()).thenReturn(totalTestCases);
        when(testSummary.getUnknownTestCases()).thenReturn(unknownTestCases);
        when(testSummary.getStatus()).thenReturn(status);
        when(testSummary.actionRan()).thenReturn(actionRan);
        
        TestCase failedTestCase = TestCase.newBuilder().setStatus(Status.FAILED).build();
        List<TestCase> failedTestCasesList = Collections.nCopies(failedTestCases, failedTestCase);
        when(testSummary.getFailedTestCases()).thenReturn(failedTestCasesList);

        TestCase skippedTestCase = TestCase.newBuilder().setStatus(Status.SKIPPED).build();
        List<TestCase> skippedTestCasesList = Collections.nCopies(skippedTestCases, skippedTestCase);
        when(testSummary.getSkippedTestCases()).thenReturn(skippedTestCasesList);

        Label label = Label.parseCanonical("//foo:bar");
        when(testSummary.getLabel()).thenReturn(label);

        return testSummary;
      }
  }

  private void printFailedToBuildSummaries(TestSummaryFormat testSummaryFormat) throws LabelSyntaxException {
    ExecutionOptions executionOptions = ExecutionOptions.DEFAULTS;
    executionOptions.testSummary = testSummaryFormat;
    when(optionsParsingResult.getOptions(ExecutionOptions.class)).thenReturn(executionOptions);
    TestSummaryOptions testSummaryOptions = new TestSummaryOptions();
    testSummaryOptions.verboseSummary = true;
    when(optionsParsingResult.getOptions(TestSummaryOptions.class)).thenReturn(testSummaryOptions);

    ImmutableSortedSet.Builder<TestSummary> builder =
        ImmutableSortedSet.orderedBy(Comparator.comparing(o -> o.getLabel().getName()));
    for (int i = 0; i < TerminalTestResultNotifier.NUM_FAILED_TO_BUILD + 1; i++) {
      TestSummary testSummary = mock(TestSummary.class);
      when(testSummary.getTotalTestCases()).thenReturn(0);

      Label labelA = Label.parseCanonical("//foo/bar:baz" + i);
      when(testSummary.getFailedTestCases()).thenReturn(ImmutableList.of());
      when(testSummary.getStatus()).thenReturn(BlazeTestStatus.FAILED_TO_BUILD);
      when(testSummary.actionRan()).thenReturn(false);
      when(testSummary.getLabel()).thenReturn(labelA);

      builder.add(testSummary);
    }

    TerminalTestResultNotifier terminalTestResultNotifier =
        new TerminalTestResultNotifier(
            ansiTerminalPrinter,
            Path::getPathString,
            optionsParsingResult,
            RepositoryMapping.EMPTY);
    terminalTestResultNotifier.notify(builder.build(), 0);
  }

  private void printTestCaseSummary(TestSummarySpec testSummarySpec, TestSummaryFormat testSummaryFormat) throws LabelSyntaxException {
    ExecutionOptions executionOptions = ExecutionOptions.DEFAULTS;
    executionOptions.testSummary = testSummaryFormat;
    when(optionsParsingResult.getOptions(ExecutionOptions.class)).thenReturn(executionOptions);
    TestSummaryOptions testSummaryOptions = new TestSummaryOptions();
    testSummaryOptions.verboseSummary = true;
    when(optionsParsingResult.getOptions(TestSummaryOptions.class)).thenReturn(testSummaryOptions);

    TerminalTestResultNotifier terminalTestResultNotifier =
        new TerminalTestResultNotifier(
            ansiTerminalPrinter,
            Path::getPathString,
            optionsParsingResult,
            RepositoryMapping.EMPTY);
    terminalTestResultNotifier.notify(ImmutableSet.of(testSummarySpec.build()), 1);
  }

  private String getPrintedMessage() {
    ArgumentCaptor<String> messageCaptor = ArgumentCaptor.forClass(String.class);
    verify(ansiTerminalPrinter).printLn(messageCaptor.capture());
    return messageCaptor.getValue();
  }

  private void verifyNoSummaryPrinted() {
    verify(ansiTerminalPrinter, never()).printLn(any());
  }

  private static String info(String message) {
    return AnsiTerminalPrinter.Mode.INFO + message + AnsiTerminalPrinter.Mode.DEFAULT;
  }

  private static String warn(String message) {
    return AnsiTerminalPrinter.Mode.WARNING + message + AnsiTerminalPrinter.Mode.DEFAULT;
  }

  private static String error(String message) {
    return AnsiTerminalPrinter.Mode.ERROR + message + AnsiTerminalPrinter.Mode.DEFAULT;
  }
}
