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

import java.util.List;
import java.util.TreeMap;

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

    SourceFileCoverage c4 = new SourceFileCoverage("");
    c4.addBranch(1, BranchCoverage.createWithBlockAndBranch(1, "1", "1", -1));
    SourceFileCoverage c5 = new SourceFileCoverage("");
    c5.addBranch(1, BranchCoverage.createWithBlockAndBranch(1, "1", "1", -1));
    SourceFileCoverage c6 = new SourceFileCoverage("");
    c6.addBranch(1, BranchCoverage.createWithBlockAndBranch(1, "1", "1", 0));
    SourceFileCoverage c7 = new SourceFileCoverage("");
    c7.addBranch(1, BranchCoverage.createWithBlockAndBranch(1, "1", "1", 1));
    SourceFileCoverage c8 = new SourceFileCoverage("");
    c8.addBranch(1, BranchCoverage.createWithBlockAndBranch(1, "1", "1", 4));

    assertThat(SourceFileCoverage.merge(c4, c5).nrBranchesHit()).isEqualTo(0);
    assertThat(SourceFileCoverage.merge(c4, c6).nrBranchesHit()).isEqualTo(0);
    assertThat(SourceFileCoverage.merge(c4, c7).nrBranchesHit()).isEqualTo(1);
    assertThat(SourceFileCoverage.merge(c7, c8).nrBranchesHit()).isEqualTo(1);
  }

  @Test
  public void testMergeBranches() {
    SourceFileCoverage c1 = new SourceFileCoverage("");
    c1.addBranch(1, BranchCoverage.createWithBlockAndBranch(1, "1", "1", 0));

    SourceFileCoverage c2 = new SourceFileCoverage("");
    c2.addBranch(1, BranchCoverage.createWithBlockAndBranch(1, "1", "2", 1));
    c2.addBranch(1, BranchCoverage.createWithBlockAndBranch(1, "1", "2", 2));

    SourceFileCoverage c3 = new SourceFileCoverage("");
    c3.addBranch(2, BranchCoverage.createWithBlockAndBranch(2, "2", "1", 2));

    assertThat(SourceFileCoverage.mergeBranches(c1, c2).size()).isEqualTo(1);
    assertThat(SourceFileCoverage.mergeBranches(c1, c3).size()).isEqualTo(2);
  }

  @Test
  public void testNrBranchesHit() {
    SourceFileCoverage c1 = new SourceFileCoverage("");
    c1.addBranch(1, BranchCoverage.createWithBlockAndBranch(1, "1", "1", 0));
    assertThat(c1.nrBranchesHit()).isEqualTo(0);
    c1.addBranch(1, BranchCoverage.createWithBlockAndBranch(1, "1", "2", 0));
    assertThat(c1.nrBranchesHit()).isEqualTo(0);
    c1.addBranch(1, BranchCoverage.createWithBlockAndBranch(1, "1", "3", 1));
    assertThat(c1.nrBranchesHit()).isEqualTo(1);

    SourceFileCoverage c2 = new SourceFileCoverage("");
    c2.addBranch(1, BranchCoverage.createWithBlockAndBranch(1, "1", "2", 1));
    c2.addBranch(1, BranchCoverage.createWithBlockAndBranch(1, "1", "2", 2));
    assertThat(c2.nrBranchesHit()).isEqualTo(1);

    SourceFileCoverage c3 = new SourceFileCoverage("");
    c3.addBranch(2, BranchCoverage.createWithBlockAndBranch(2, "2", "1", 2));
    c3.addBranch(2, BranchCoverage.createWithBlockAndBranch(2, "2", "2", 0));
    assertThat(c3.nrBranchesHit()).isEqualTo(1);

    c3.addBranch(2, BranchCoverage.createWithBlockAndBranch(2, "2", "2", 1));
    assertThat(c3.nrBranchesHit()).isEqualTo(2);
  }
}
