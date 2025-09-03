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
import static org.junit.Assert.assertThrows;

import com.google.common.base.VerifyException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@BranchCoverageData}. */
@RunWith(JUnit4.class)
public class BranchCoverageTest {

  @Test
  public void testMergesBranchesEvaluated() {
    BranchCoverage b1 = BranchCoverage.create(1, "3", "2", true, 3);
    BranchCoverage b2 = BranchCoverage.create(1, "3", "2", true, 0);
    BranchCoverage b3 = BranchCoverage.create(1, "3", "2", true, 2);

    BranchCoverage m1 = BranchCoverage.merge(b1, b2);
    BranchCoverage m2 = BranchCoverage.merge(m1, b3);

    assertThat(m1.lineNumber()).isEqualTo(1);
    assertThat(m1.blockNumber()).isEqualTo("3");
    assertThat(m1.branchNumber()).isEqualTo("2");
    assertThat(m1.evaluated()).isTrue();
    assertThat(m1.nrOfExecutions()).isEqualTo(3);
    assertThat(m2.evaluated()).isTrue();
    assertThat(m2.nrOfExecutions()).isEqualTo(5);
  }

  @Test
  public void testMergeBranchesNotEvaluated() {
    BranchCoverage b1 = BranchCoverage.create(1, "2", "0", false, 0);
    BranchCoverage b2 = BranchCoverage.create(1, "2", "0", false, 0);

    BranchCoverage merged = BranchCoverage.merge(b1, b2);

    assertThat(merged.lineNumber()).isEqualTo(1);
    assertThat(merged.blockNumber()).isEqualTo("2");
    assertThat(merged.branchNumber()).isEqualTo("0");
    assertThat(merged.evaluated()).isFalse();
    assertThat(merged.nrOfExecutions()).isEqualTo(0);
  }

  @Test
  public void testMergeBranchesMixedEvaluated() {
    BranchCoverage b1 = BranchCoverage.create(1, "2", "0", false, 0);
    BranchCoverage b2 = BranchCoverage.create(1, "2", "0", true, 0);

    BranchCoverage merged = BranchCoverage.merge(b1, b2);

    assertThat(merged.lineNumber()).isEqualTo(1);
    assertThat(merged.blockNumber()).isEqualTo("2");
    assertThat(merged.branchNumber()).isEqualTo("0");
    assertThat(merged.evaluated()).isTrue();
    assertThat(merged.nrOfExecutions()).isEqualTo(0);
  }

  @Test
  public void testDifferentLineNumbersFail() {
    BranchCoverage b1 = BranchCoverage.create(2, "0", "0", true, 1);
    BranchCoverage b2 = BranchCoverage.create(3, "0", "0", true, 2);
    assertThrows(VerifyException.class, () -> BranchCoverage.merge(b1, b2));
  }

  @Test
  public void testDifferentBlockNumbersFail() {
    BranchCoverage b1 = BranchCoverage.create(1, "3", "2", true, 1);
    BranchCoverage b2 = BranchCoverage.create(1, "2", "2", true, 1);
    assertThrows(VerifyException.class, () -> BranchCoverage.merge(b1, b2));
  }

  @Test
  public void testDifferentBranchNumbersFail() {
    BranchCoverage b1 = BranchCoverage.create(1, "3", "2", true, 1);
    BranchCoverage b2 = BranchCoverage.create(1, "3", "3", true, 1);
    assertThrows(VerifyException.class, () -> BranchCoverage.merge(b1, b2));
  }
}
