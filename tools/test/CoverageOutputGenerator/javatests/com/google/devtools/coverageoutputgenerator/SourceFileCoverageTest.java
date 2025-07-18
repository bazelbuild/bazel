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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link SourceFileCoverage}. */
@RunWith(JUnit4.class)
public class SourceFileCoverageTest {

  @Test
  public void testCopyConstructor() {
    SourceFileCoverage sourceFile = new SourceFileCoverage("src.foo");
    sourceFile.addFunctionLineNumber("foo", 3);
    sourceFile.addLine(3, 2);
    sourceFile.addLine(4, 1);
    sourceFile.addLine(5, 0);
    sourceFile.addBranch(3, BranchCoverage.create(3, "0", "0", true, 2));
    sourceFile.addBranch(3, BranchCoverage.create(3, "0", "1", true, 0));
    sourceFile.addBranch(5, BranchCoverage.create(5, "7", "0", false, 0));
    sourceFile.addBranch(5, BranchCoverage.create(5, "7", "1", false, 0));

    SourceFileCoverage copy = new SourceFileCoverage(sourceFile);

    assertThat(copy.getLines()).isEqualTo(sourceFile.getLines());
    assertThat(copy.getFunctionLineNumbers()).isEqualTo(sourceFile.getFunctionLineNumbers());
    assertThat(copy.getAllBranches())
        .containsExactly(
            BranchCoverage.create(3, "0", "0", true, 2),
            BranchCoverage.create(3, "0", "1", true, 0),
            BranchCoverage.create(5, "7", "0", false, 0),
            BranchCoverage.create(5, "7", "1", false, 0));
  }

  @Test
  public void testMergeFunctionNameToLineNumber() {
    SourceFileCoverage sourceFile1 = new SourceFileCoverage("src.foo");
    SourceFileCoverage sourceFile2 = new SourceFileCoverage("src.foo");
    sourceFile1.addFunctionLineNumber("foo", 3);
    sourceFile1.addFunctionLineNumber("bar", 10);
    sourceFile2.addFunctionLineNumber("foo", 3);
    sourceFile2.addFunctionLineNumber("bar", 10);

    SourceFileCoverage merged = SourceFileCoverage.merge(sourceFile1, sourceFile2);

    assertThat(merged.getFunctionLineNumbers()).containsExactly("foo", 3, "bar", 10);
  }

  @Test
  public void testMergeFunctionNameToExecutionCount() {
    SourceFileCoverage sourceFile1 = new SourceFileCoverage("src.foo");
    SourceFileCoverage sourceFile2 = new SourceFileCoverage("src.foo");
    sourceFile1.addFunctionLineNumber("foo", 3);
    sourceFile1.addFunctionExecution("foo", 5L);
    sourceFile2.addFunctionLineNumber("foo", 3);
    sourceFile2.addFunctionExecution("foo", 7L);

    SourceFileCoverage merged = SourceFileCoverage.merge(sourceFile1, sourceFile2);

    assertThat(merged.getFunctionsExecution()).containsExactly("foo", 12L);
  }

  @Test
  public void testMergeLineNumberToLineExecution() {
    SourceFileCoverage sourceFile1 = new SourceFileCoverage("src.foo");
    SourceFileCoverage sourceFile2 = new SourceFileCoverage("src.foo");
    sourceFile1.addLine(4, 3);
    sourceFile1.addLine(5, 4);
    sourceFile1.addLine(10, 0);
    sourceFile2.addLine(4, 5);
    sourceFile2.addLine(5, 0);
    sourceFile2.addLine(10, 3);

    SourceFileCoverage merged = SourceFileCoverage.merge(sourceFile1, sourceFile2);

    assertThat(merged.getLines())
        .containsExactly(
            4, 8L,
            5, 4L,
            10, 3L);
  }

  @Test
  public void testMergeBranches() {
    SourceFileCoverage sourceFile1 = new SourceFileCoverage("src.foo");
    SourceFileCoverage sourceFile2 = new SourceFileCoverage("src.foo");
    sourceFile1.addNewBranch(1, "0", "0", true, 1);
    sourceFile1.addNewBranch(1, "0", "1", true, 0);
    sourceFile1.addNewBranch(1, "0", "2", true, 0);
    sourceFile1.addNewBranch(1, "0", "0", true, 0);
    sourceFile1.addNewBranch(1, "1", "0", true, 3);
    sourceFile1.addNewBranch(1, "1", "1", true, 4);
    sourceFile2.addNewBranch(1, "0", "1", true, 0);
    sourceFile2.addNewBranch(1, "0", "2", true, 1);
    sourceFile2.addNewBranch(1, "1", "0", true, 7);
    sourceFile2.addNewBranch(1, "1", "1", true, 8);

    SourceFileCoverage merged = SourceFileCoverage.merge(sourceFile1, sourceFile2);

    assertThat(merged.getAllBranches())
        .containsExactly(
            BranchCoverage.create(1, "0", "0", true, 1),
            BranchCoverage.create(1, "0", "1", true, 0),
            BranchCoverage.create(1, "0", "2", true, 1),
            BranchCoverage.create(1, "1", "0", true, 10),
            BranchCoverage.create(1, "1", "1", true, 12));
  }

  @Test
  public void testMismatchedBranchMerge() throws Exception {
    SourceFileCoverage sourceFile1 = new SourceFileCoverage("source");
    SourceFileCoverage sourceFile2 = new SourceFileCoverage("source");
    sourceFile1.addNewBranch(800, "0", "0", true, 1);
    sourceFile1.addNewBranch(800, "0", "1", true, 0);
    sourceFile1.addNewBranch(800, "1", "0", true, 1);
    sourceFile2.addNewBranch(800, "1", "0", true, 3);
    sourceFile2.addNewBranch(800, "1", "1", true, 4);

    // Check the results are the same no matter the order of the merge.
    SourceFileCoverage merged1 = SourceFileCoverage.merge(sourceFile1, sourceFile2);
    SourceFileCoverage merged2 = SourceFileCoverage.merge(sourceFile2, sourceFile1);

    assertThat(merged1.getAllBranches())
        .containsExactly(
            BranchCoverage.create(800, "0", "0", true, 1),
            BranchCoverage.create(800, "0", "1", true, 0),
            BranchCoverage.create(800, "1", "0", true, 4),
            BranchCoverage.create(800, "1", "1", true, 4));
    assertThat(merged2.getAllBranches())
        .containsExactly(
            BranchCoverage.create(800, "0", "0", true, 1),
            BranchCoverage.create(800, "0", "1", true, 0),
            BranchCoverage.create(800, "1", "0", true, 4),
            BranchCoverage.create(800, "1", "1", true, 4));
  }

  @Test
  public void testDifferentLinesReportedAreMergeable() throws Exception {
    SourceFileCoverage sourceFile1 = new SourceFileCoverage("source");
    SourceFileCoverage sourceFile2 = new SourceFileCoverage("source");
    sourceFile1.addNewBranch(1, "0", "0", true, 1);
    sourceFile1.addNewBranch(1, "0", "1", true, 1);
    sourceFile1.addLine(1, 2);
    sourceFile1.addLine(2, 1);
    sourceFile1.addLine(3, 1);

    sourceFile2.addNewBranch(30, "0", "0", true, 3);
    sourceFile2.addNewBranch(30, "0", "1", true, 0);
    sourceFile2.addNewBranch(30, "0", "2", true, 1);
    sourceFile2.addLine(30, 4);
    sourceFile2.addLine(31, 3);
    sourceFile2.addLine(32, 0);
    sourceFile2.addLine(33, 1);

    SourceFileCoverage merged = SourceFileCoverage.merge(sourceFile1, sourceFile2);
    assertThat(merged.getAllBranches())
        .containsExactly(
            BranchCoverage.create(1, "0", "0", true, 1),
            BranchCoverage.create(1, "0", "1", true, 1),
            BranchCoverage.create(30, "0", "0", true, 3),
            BranchCoverage.create(30, "0", "1", true, 0),
            BranchCoverage.create(30, "0", "2", true, 1));
    assertThat(merged.getLines())
        .containsExactly(1, 2L, 2, 1L, 3, 1L, 30, 4L, 31, 3L, 32, 0L, 33, 1L);
  }

  @Test
  public void testRepeatedBranchesAreMerged() {
    SourceFileCoverage sourceFile = new SourceFileCoverage("source");
    sourceFile.addNewBranch(1, "0", "0", false, 0);
    sourceFile.addNewBranch(1, "0", "1", false, 0);
    sourceFile.addNewBranch(1, "0", "0", true, 1);
    sourceFile.addNewBranch(1, "0", "1", true, 1);
    sourceFile.addNewBranch(1, "0", "0", true, 2);
    sourceFile.addNewBranch(1, "0", "1", true, 2);

    assertThat(sourceFile.getAllBranches())
        .containsExactly(
            BranchCoverage.create(1, "0", "0", true, 3),
            BranchCoverage.create(1, "0", "1", true, 3));
  }
}
