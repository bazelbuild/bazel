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

import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.TestStrategy.TestSummaryFormat;
import com.google.devtools.build.lib.runtime.TerminalTestResultNotifier.TestSummaryOptions;
import com.google.devtools.build.lib.util.io.AnsiTerminalPrinter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.TestCase;
import com.google.devtools.build.lib.view.test.TestStatus.TestCase.Status;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests {@link TerminalTestResultNotifier}. */
@RunWith(JUnit4.class)
public class TerminalTestResultNotifierTest {

  private final OptionsParsingResult optionsParsingResult =
      Mockito.mock(OptionsParsingResult.class);
  private final AnsiTerminalPrinter ansiTerminalPrinter = Mockito.mock(AnsiTerminalPrinter.class);
  private final Random random = new Random();

  private void givenExecutionOption(TestSummaryFormat format) {
    ExecutionOptions executionOptions = ExecutionOptions.DEFAULTS;
    executionOptions.testSummary = format;
    when(optionsParsingResult.getOptions(ExecutionOptions.class)).thenReturn(executionOptions);

    TestSummaryOptions testSummaryOptions = new TestSummaryOptions();
    testSummaryOptions.verboseSummary = true;
    when(optionsParsingResult.getOptions(TestSummaryOptions.class)).thenReturn(testSummaryOptions);
  }

  private void runTest(Boolean shouldPrintTestCaseSummary) throws Exception {
    int numOfTotalTestCases = random.nextInt(10) + 1;
    int numOfFailedCases = random.nextInt(numOfTotalTestCases);
    int numOfSuccessfulTestCases = numOfTotalTestCases - numOfFailedCases;

    TerminalTestResultNotifier terminalTestResultNotifier = new TerminalTestResultNotifier(
        ansiTerminalPrinter, Path::getPathString, optionsParsingResult);

    TestSummary testSummary = Mockito.mock(TestSummary.class);
    when(testSummary.getTotalTestCases()).thenReturn(numOfTotalTestCases);
    TestCase failedTestCase = TestCase.newBuilder().setStatus(Status.FAILED).build();
    ArrayList<TestCase> testCases =
        new ArrayList<>(Collections.nCopies(numOfFailedCases, failedTestCase));

    Label labelA = Label.parseAbsolute("//foo/bar:baz", ImmutableMap.of());
    when(testSummary.getFailedTestCases()).thenReturn(testCases);
    when(testSummary.getStatus()).thenReturn(BlazeTestStatus.FAILED);
    when(testSummary.getLabel()).thenReturn(labelA);

    HashSet<TestSummary> testSummaries = new HashSet<>();
    testSummaries.add(testSummary);
    terminalTestResultNotifier.notify(testSummaries, 1);

    String summaryMessage =
        String.format(
            "Test cases: finished with %s%d passing%s and %s%d failing%s out of %d test cases",
            numOfSuccessfulTestCases > 0 ? AnsiTerminalPrinter.Mode.INFO : "",
            numOfSuccessfulTestCases,
            AnsiTerminalPrinter.Mode.DEFAULT,
            numOfFailedCases > 0 ? AnsiTerminalPrinter.Mode.ERROR : "",
            numOfFailedCases,
            AnsiTerminalPrinter.Mode.DEFAULT,
            numOfTotalTestCases);

    if (shouldPrintTestCaseSummary) {
      verify(ansiTerminalPrinter).printLn(summaryMessage);
    } else {
      verify(ansiTerminalPrinter, never()).printLn(summaryMessage);
    }
  }

  @Test
  public void testCasesDataVisibleInTestCaseOption() throws Exception {
    givenExecutionOption(TestSummaryFormat.TESTCASE);
    runTest(true);
  }

  @Test
  public void testCasesDataVisibleInDetailedOption() throws Exception {
    givenExecutionOption(TestSummaryFormat.DETAILED);
    runTest(true);
  }

  @Test
  public void testCasesDataInVisibleInShortOption() throws Exception {
    givenExecutionOption(TestSummaryFormat.SHORT);
    runTest(false);
  }

  @Test
  public void testCasesDataInVisibleInTerseOption() throws Exception {
    givenExecutionOption(TestSummaryFormat.TERSE);
    runTest(false);
  }

  @Test
  public void testCasesDataInVisibleInNoneOption() throws Exception {
    givenExecutionOption(TestSummaryFormat.NONE);
    runTest(false);
  }
}
