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
import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@BranchCoverageData}. */
@RunWith(JUnit4.class)
public class BranchCoverageTest {

  private static final int BRANCH1_LINE_NR = 10;
  private static final String BRANCH1_BLOCK_NR = "3";
  private static final String BRANCH1_BRANCH_NR = "2";
  private static final int BRANCH1_NR_EXECUTIONS = 0;

  private static final int BRANCH1_OTHER_TRACEFILE_NR_EXECUTIONS = 5;

  private static final int BRANCH2_LINE_NR = 20;
  private static final String BRANCH2_BLOCK_NR = "7";
  private static final String BRANCH2_BRANCH_NR = "2";
  private static final int BRANCH2_NR_EXECUTIONS = 0;

  private static final int BRANCH2_OTHER_TRACEFILE_NR_EXECUTIONS = 0;

  static final BranchCoverage getBranch1CoverageData() {
    return BranchCoverage.createWithBlockAndBranch(
        BRANCH1_LINE_NR, BRANCH1_BLOCK_NR, BRANCH1_BRANCH_NR, BRANCH1_NR_EXECUTIONS);
  }

  static final BranchCoverage getBranch2CoverageData() {
    return BranchCoverage.createWithBlockAndBranch(
        BRANCH2_LINE_NR, BRANCH2_BLOCK_NR, BRANCH2_BRANCH_NR, BRANCH2_NR_EXECUTIONS);
  }

  static final BranchCoverage getBranch1OtherTracefileCoverageData() {
    return BranchCoverage.createWithBlockAndBranch(
        BRANCH1_LINE_NR,
        BRANCH1_BLOCK_NR,
        BRANCH1_BRANCH_NR,
        BRANCH1_OTHER_TRACEFILE_NR_EXECUTIONS);
  }

  static final BranchCoverage getBranch2OtherTracefileCoverageData() {
    return BranchCoverage.createWithBlockAndBranch(
        BRANCH2_LINE_NR,
        BRANCH2_BLOCK_NR,
        BRANCH2_BRANCH_NR,
        BRANCH2_OTHER_TRACEFILE_NR_EXECUTIONS);
  }

  @Test
  public void testMergeBranch1() {
    BranchCoverage branch1 = getBranch1CoverageData();
    BranchCoverage branch1OtherTracefile = getBranch1OtherTracefileCoverageData();
    BranchCoverage merged = BranchCoverage.merge(branch1, branch1OtherTracefile);
    assertThat(merged.lineNumber()).isEqualTo(branch1.lineNumber());
    assertThat(merged.blockNumber()).isEqualTo(branch1.blockNumber());
    assertThat(merged.branchNumber()).isEqualTo(branch1.branchNumber());
    assertThat(merged.wasExecuted()).isTrue();
    assertThat(merged.nrOfExecutions())
        .isEqualTo(branch1.nrOfExecutions() + branch1OtherTracefile.nrOfExecutions());
  }

  @Test
  public void testMergeBranch2() {
    BranchCoverage branch2 = getBranch2CoverageData();
    BranchCoverage branch2OtherTracefile = getBranch2OtherTracefileCoverageData();
    BranchCoverage merged = BranchCoverage.merge(branch2, branch2OtherTracefile);
    assertThat(merged.lineNumber()).isEqualTo(branch2.lineNumber());
    assertThat(merged.blockNumber()).isEqualTo(branch2.blockNumber());
    assertThat(merged.branchNumber()).isEqualTo(branch2.branchNumber());
    assertThat(merged.wasExecuted()).isFalse();
    assertThat(merged.nrOfExecutions()).isEqualTo(0);
  }

  @Test
  public void testMergeBranch1Branch2AssertationError() {
    BranchCoverage branch1 = getBranch1CoverageData();
    BranchCoverage branch2 = getBranch2CoverageData();
    try {
      BranchCoverage.merge(branch1, branch2);
    } catch (AssertionError er) {
      return;
    }
    fail();
  }
}
