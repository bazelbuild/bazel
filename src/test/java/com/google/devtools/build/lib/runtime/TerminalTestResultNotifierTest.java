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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.ExecutionOptions.TestSummaryFormat;
import com.google.devtools.build.lib.runtime.TerminalTestResultNotifier.TestSummaryOptions;
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

  private BlazeTestStatus targetStatus;
  private int numFailedTestCases;
  private int numUnknownTestCases;
  private int numTotalTestCases;
  private TestSummaryFormat testSummaryFormat;

  @Test
  public void testCaseOption_allPass() throws Exception {
    testSummaryFormat = ExecutionOptions.TestSummaryFormat.TESTCASE;
    numFailedTestCases = 0;
    numTotalTestCases = 10;
    targetStatus = BlazeTestStatus.PASSED;

    printTestCaseSummary();

    String printed = getPrintedMessage();
    assertThat(printed).contains(info("10 passing"));
    assertThat(printed).contains("0 failing");
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.ERROR.toString());
  }

  @Test
  public void testCaseOption_allPassButTargetFails() throws Exception {
    testSummaryFormat = ExecutionOptions.TestSummaryFormat.TESTCASE;
    numFailedTestCases = 0;
    numUnknownTestCases = 10;
    numTotalTestCases = 10;
    targetStatus = BlazeTestStatus.FAILED;

    printTestCaseSummary();

    String printed = getPrintedMessage();
    assertThat(printed).contains("0 passing");
    assertThat(printed).contains("0 failing");
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).contains(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.ERROR.toString());
  }

  @Test
  public void testCaseOption_someFail() throws Exception {
    testSummaryFormat = ExecutionOptions.TestSummaryFormat.TESTCASE;
    numFailedTestCases = 2;
    numUnknownTestCases = 0;
    numTotalTestCases = 10;
    targetStatus = BlazeTestStatus.FAILED;

    printTestCaseSummary();

    String printed = getPrintedMessage();
    assertThat(printed).contains(info("8 passing"));
    assertThat(printed).contains(error("2 failing"));
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
  }

  @Test
  public void shortOption_someFailToBuild() throws Exception {
    testSummaryFormat = ExecutionOptions.TestSummaryFormat.SHORT;
    numFailedTestCases = 0;
    int numFailedToBuildTestCases = TerminalTestResultNotifier.NUM_FAILED_TO_BUILD + 1;
    numUnknownTestCases = 0;
    numTotalTestCases = 10;
    targetStatus = BlazeTestStatus.FAILED_TO_BUILD;

    printFailedToBuildSummaries();

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
  public void testCaseOption_allFail() throws Exception {
    testSummaryFormat = ExecutionOptions.TestSummaryFormat.TESTCASE;
    numFailedTestCases = 10;
    numUnknownTestCases = 0;
    numTotalTestCases = 10;
    targetStatus = BlazeTestStatus.FAILED;

    printTestCaseSummary();

    String printed = getPrintedMessage();
    assertThat(printed).contains("0 passing");
    assertThat(printed).contains(error("10 failing"));
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.INFO.toString());
  }

  @Test
  public void detailedOption_allPass() throws Exception {
    testSummaryFormat = ExecutionOptions.TestSummaryFormat.DETAILED;
    numFailedTestCases = 0;
    numUnknownTestCases = 0;
    numTotalTestCases = 10;
    targetStatus = BlazeTestStatus.PASSED;

    printTestCaseSummary();

    String printed = getPrintedMessage();
    assertThat(printed).contains(info("10 passing"));
    assertThat(printed).contains("0 failing");
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.ERROR.toString());
  }

  @Test
  public void detailedOption_allPassButTargetFails() throws Exception {
    testSummaryFormat = ExecutionOptions.TestSummaryFormat.DETAILED;
    numFailedTestCases = 0;
    numUnknownTestCases = 10;
    numTotalTestCases = 10;
    targetStatus = BlazeTestStatus.FAILED;

    printTestCaseSummary();

    String printed = getPrintedMessage();
    assertThat(printed).contains("0 passing");
    assertThat(printed).contains("0 failing");
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).contains(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.ERROR.toString());
  }

  @Test
  public void detailedOption_someFail() throws Exception {
    testSummaryFormat = ExecutionOptions.TestSummaryFormat.DETAILED;
    numFailedTestCases = 2;
    numUnknownTestCases = 0;
    numTotalTestCases = 10;
    targetStatus = BlazeTestStatus.FAILED;

    printTestCaseSummary();

    String printed = getPrintedMessage();
    assertThat(printed).contains(info("8 passing"));
    assertThat(printed).contains(error("2 failing"));
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
  }

  @Test
  public void detailedOption_allFail() throws Exception {
    testSummaryFormat = ExecutionOptions.TestSummaryFormat.DETAILED;
    numFailedTestCases = 10;
    numUnknownTestCases = 0;
    numTotalTestCases = 10;
    targetStatus = BlazeTestStatus.FAILED;

    printTestCaseSummary();

    String printed = getPrintedMessage();
    assertThat(printed).contains("0 passing");
    assertThat(printed).contains(error("10 failing"));
    assertThat(printed).contains("out of 10 test cases");
    assertThat(printed).doesNotContain(SOME_TARGETS_ARE_MISSING_TEST_CASES_DISCLAIMER);
    assertThat(printed).doesNotContain(AnsiTerminalPrinter.Mode.INFO.toString());
  }

  @Test
  public void shortOption_noSummaryPrinted() throws Exception {
    testSummaryFormat = ExecutionOptions.TestSummaryFormat.SHORT;
    numFailedTestCases = 2;
    numUnknownTestCases = 0;
    numTotalTestCases = 10;
    targetStatus = BlazeTestStatus.FAILED;

    printTestCaseSummary();

    verifyNoSummaryPrinted();
  }

  @Test
  public void terseOption_noSummaryPrinted() throws Exception {
    testSummaryFormat = ExecutionOptions.TestSummaryFormat.TERSE;
    numFailedTestCases = 2;
    numUnknownTestCases = 0;
    numTotalTestCases = 10;
    targetStatus = BlazeTestStatus.FAILED;

    printTestCaseSummary();

    verifyNoSummaryPrinted();
  }

  @Test
  public void noneOption_noSummaryPrinted() throws Exception {
    testSummaryFormat = ExecutionOptions.TestSummaryFormat.NONE;
    numFailedTestCases = 2;
    numUnknownTestCases = 0;
    numTotalTestCases = 10;
    targetStatus = BlazeTestStatus.FAILED;

    printTestCaseSummary();

    verifyNoSummaryPrinted();
  }

  private void printFailedToBuildSummaries() throws LabelSyntaxException {
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

      Label labelA = Label.parseAbsolute("//foo/bar:baz" + i, ImmutableMap.of());
      when(testSummary.getFailedTestCases()).thenReturn(ImmutableList.of());
      when(testSummary.getStatus()).thenReturn(BlazeTestStatus.FAILED_TO_BUILD);
      when(testSummary.getLabel()).thenReturn(labelA);

      builder.add(testSummary);
    }

    TerminalTestResultNotifier terminalTestResultNotifier =
        new TerminalTestResultNotifier(
            ansiTerminalPrinter, Path::getPathString, optionsParsingResult);
    terminalTestResultNotifier.notify(builder.build(), 0);
  }

  private void printTestCaseSummary() throws LabelSyntaxException {
    ExecutionOptions executionOptions = ExecutionOptions.DEFAULTS;
    executionOptions.testSummary = testSummaryFormat;
    when(optionsParsingResult.getOptions(ExecutionOptions.class)).thenReturn(executionOptions);
    TestSummaryOptions testSummaryOptions = new TestSummaryOptions();
    testSummaryOptions.verboseSummary = true;
    when(optionsParsingResult.getOptions(TestSummaryOptions.class)).thenReturn(testSummaryOptions);

    TestSummary testSummary = mock(TestSummary.class);
    when(testSummary.getTotalTestCases()).thenReturn(numTotalTestCases);
    when(testSummary.getUnkownTestCases()).thenReturn(numUnknownTestCases);
    TestCase failedTestCase = TestCase.newBuilder().setStatus(Status.FAILED).build();
    List<TestCase> failedTestCases = Collections.nCopies(numFailedTestCases, failedTestCase);

    Label labelA = Label.parseAbsolute("//foo/bar:baz", ImmutableMap.of());
    when(testSummary.getFailedTestCases()).thenReturn(failedTestCases);
    when(testSummary.getStatus()).thenReturn(targetStatus);
    when(testSummary.getLabel()).thenReturn(labelA);

    TerminalTestResultNotifier terminalTestResultNotifier =
        new TerminalTestResultNotifier(
            ansiTerminalPrinter, Path::getPathString, optionsParsingResult);
    terminalTestResultNotifier.notify(ImmutableSet.of(testSummary), 1);
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

  private static String error(String message) {
    return AnsiTerminalPrinter.Mode.ERROR + message + AnsiTerminalPrinter.Mode.DEFAULT;
  }
}
