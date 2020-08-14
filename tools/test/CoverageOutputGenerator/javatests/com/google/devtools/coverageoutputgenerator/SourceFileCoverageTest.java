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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.assertMergedFunctionsExecution;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.assertMergedLineNumbers;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.assertMergedLines;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.assertMergedSourceFile;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.assertTracefile1;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.createLinesExecution1;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.createLinesExecution2;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.createSourceFile1;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.createSourceFile2;
import static org.junit.Assert.assertThrows;

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
  public void testMerge() throws Exception {
    assertMergedSourceFile(
        SourceFileCoverage.merge(sourceFile1, sourceFile2), linesExecution1, linesExecution2);
  }

  @Test
  public void testIncompatibleBaBranchMergeThrows() throws Exception {
    sourceFile1.addBranch(800, BranchCoverage.create(800, 2));
    sourceFile1.addBranch(800, BranchCoverage.create(800, 1));
    sourceFile2.addBranch(800, BranchCoverage.create(800, 2));
    sourceFile2.addBranch(800, BranchCoverage.create(800, 2));
    sourceFile2.addBranch(800, BranchCoverage.create(800, 1));
    assertThrows(
        IncompatibleMergeException.class, () -> SourceFileCoverage.merge(sourceFile1, sourceFile2));
  }

  @Test
  public void testIncompatibleBrdaBranchMergeThrows() throws Exception {
    sourceFile1.addBranch(800, BranchCoverage.createWithBlockAndBranch(800, "0", "0", true, 1));
    sourceFile1.addBranch(800, BranchCoverage.createWithBlockAndBranch(800, "0", "1", true, 0));
    sourceFile2.addBranch(800, BranchCoverage.createWithBlockAndBranch(800, "1", "0", true, 3));
    sourceFile2.addBranch(800, BranchCoverage.createWithBlockAndBranch(800, "1", "1", true, 4));
    assertThrows(
        IncompatibleMergeException.class, () -> SourceFileCoverage.merge(sourceFile1, sourceFile2));
  }

  @Test
  public void testDifferentLinesReportedAreMergeable() throws Exception {
    sourceFile1 = new SourceFileCoverage("source");
    sourceFile2 = new SourceFileCoverage("source");
    sourceFile1.addBranch(1, BranchCoverage.createWithBlockAndBranch(1, "0", "0", true, 1));
    sourceFile1.addBranch(1, BranchCoverage.createWithBlockAndBranch(1, "0", "1", true, 1));
    sourceFile1.addLine(1, LineCoverage.create(1, 2, ""));
    sourceFile1.addLine(2, LineCoverage.create(2, 1, ""));
    sourceFile1.addLine(3, LineCoverage.create(3, 1, ""));

    sourceFile2.addBranch(30, BranchCoverage.createWithBlockAndBranch(30, "0", "0", true, 3));
    sourceFile2.addBranch(30, BranchCoverage.createWithBlockAndBranch(30, "0", "1", true, 0));
    sourceFile2.addBranch(30, BranchCoverage.createWithBlockAndBranch(30, "0", "2", true, 1));
    sourceFile2.addLine(30, LineCoverage.create(30, 4, ""));
    sourceFile2.addLine(31, LineCoverage.create(31, 3, ""));
    sourceFile2.addLine(32, LineCoverage.create(32, 0, ""));
    sourceFile2.addLine(33, LineCoverage.create(33, 1, ""));

    SourceFileCoverage merged = SourceFileCoverage.merge(sourceFile1, sourceFile2);
    assertThat(merged.getAllBranches())
        .containsExactly(
            BranchCoverage.createWithBlockAndBranch(1, "0", "0", true, 1),
            BranchCoverage.createWithBlockAndBranch(1, "0", "1", true, 1),
            BranchCoverage.createWithBlockAndBranch(30, "0", "0", true, 3),
            BranchCoverage.createWithBlockAndBranch(30, "0", "1", true, 0),
            BranchCoverage.createWithBlockAndBranch(30, "0", "2", true, 1));

    assertThat(merged.getAllLineExecution())
        .containsExactly(
            LineCoverage.create(1, 2, ""),
            LineCoverage.create(2, 1, ""),
            LineCoverage.create(3, 1, ""),
            LineCoverage.create(30, 4, ""),
            LineCoverage.create(31, 3, ""),
            LineCoverage.create(32, 0, ""),
            LineCoverage.create(33, 1, ""));
  }
}
