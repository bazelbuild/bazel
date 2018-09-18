// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.coverageoutputgenerator;

import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.assertMergedFunctionsExecution;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.assertMergedLineNumbers;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.assertMergedLines;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.assertMergedSourceFile;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.assertTracefile1;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.createLinesExecution1;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.createLinesExecution2;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.createSourceFile1;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.createSourceFile2;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link SourceFileCoverage}. */
@RunWith(JUnit4.class)
public class SourceFileCoverageTest {

  private int[] linesExecution1;
  private int[] linesExecution2;
  private SourceFileCoverage sourceFile1;
  private SourceFileCoverage sourceFile2;

  @Before
  public void initializeExecutionCountTracefiles() {
    linesExecution1 = createLinesExecution1();
    linesExecution2 = createLinesExecution2();

    sourceFile1 = createSourceFile1(linesExecution1);
    sourceFile2 = createSourceFile2(linesExecution2);
  }

  @Test
  public void testCopyConstructor() {
    assertTracefile1(new SourceFileCoverage(sourceFile1));
  }

  @Test
  public void testMergeFunctionNameToLineNumber() {
    assertMergedLineNumbers(SourceFileCoverage.mergeLineNumbers(sourceFile1, sourceFile2));
  }

  @Test
  public void testMergeFunctionNameToExecutionCount() {
    assertMergedFunctionsExecution(
        SourceFileCoverage.mergeFunctionsExecution(sourceFile1, sourceFile2));
  }

  @Test
  public void testMergeLineNumberToLineExecution() {
    assertMergedLines(
        SourceFileCoverage.mergeLines(sourceFile1, sourceFile2), linesExecution1, linesExecution2);
  }

  @Test
  public void testMerge() {
    assertMergedSourceFile(
        SourceFileCoverage.merge(sourceFile1, sourceFile2), linesExecution1, linesExecution2);
  }
}
