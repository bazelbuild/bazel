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

/** Unit tests for {@LineExecutionCoverageData}. */
@RunWith(JUnit4.class)
public class LineCoverageTest {
  private static final int LINE1_NR = 10;
  private static final int LINE1_EXECUTION_COUNT = 10;
  private static final String LINE1_CHECKSUM = "90345njksdf2";
  private static final int LINE1_OTHER_TRACEFILE_EXECUTION_COUNT = 5;

  private static final int LINE2_NR = 20;
  private static final int LINE2_EXECUTION_COUNT = 3;
  private static final String LINE2_CHECKSUM = null;
  private static final int LINE2_OTHER_TRACEFILE_EXECUTION_COUNT = 5;

  static LineCoverage getLine1CoverageData() {
    return LineCoverage.create(LINE1_NR, LINE1_EXECUTION_COUNT, LINE1_CHECKSUM);
  }

  static LineCoverage getLine1CoverageDataDifferentChecksum() {
    return LineCoverage.create(LINE1_NR, LINE1_EXECUTION_COUNT, LINE2_CHECKSUM);
  }

  static LineCoverage getLine1CoverageDataOtherTracefile() {
    return LineCoverage.create(LINE1_NR, LINE1_OTHER_TRACEFILE_EXECUTION_COUNT, LINE1_CHECKSUM);
  }

  static LineCoverage getLine2CoverageData() {
    return LineCoverage.create(LINE2_NR, LINE2_EXECUTION_COUNT, LINE2_CHECKSUM);
  }

  static LineCoverage getLine2CoverageDataOtherTracefile() {
    return LineCoverage.create(LINE2_NR, LINE2_OTHER_TRACEFILE_EXECUTION_COUNT, LINE2_CHECKSUM);
  }

  @Test
  public void testMergeLine1() {
    LineCoverage line1 = getLine1CoverageData();
    LineCoverage line1OtherTracefile = getLine1CoverageDataOtherTracefile();
    LineCoverage merged = LineCoverage.merge(line1, line1OtherTracefile);

    assertThat(merged.lineNumber()).isEqualTo(LINE1_NR);
    assertThat(merged.executionCount())
        .isEqualTo(LINE1_EXECUTION_COUNT + LINE1_OTHER_TRACEFILE_EXECUTION_COUNT);
    assertThat(merged.checksum()).isEqualTo(LINE1_CHECKSUM);
  }

  @Test
  public void testMergeLine2() {
    LineCoverage line2 = getLine2CoverageData();
    LineCoverage line2OtherTracefile = getLine2CoverageDataOtherTracefile();
    LineCoverage merged = LineCoverage.merge(line2, line2OtherTracefile);

    assertThat(merged.lineNumber()).isEqualTo(LINE2_NR);
    assertThat(merged.executionCount())
        .isEqualTo(LINE2_EXECUTION_COUNT + LINE2_OTHER_TRACEFILE_EXECUTION_COUNT);
    assertThat(merged.checksum()).isEqualTo(null);
  }

  @Test
  public void testMergeLine1WithLine2() {
    LineCoverage line1 = getLine1CoverageData();
    LineCoverage line2 = getLine2CoverageData();
    try {
      LineCoverage.merge(line1, line2);
    } catch (AssertionError er) {
      return;
    }
    fail();
  }

  @Test
  public void testMergeLine1DifferentChecksum() {
    LineCoverage line1 = getLine1CoverageData();
    LineCoverage line1DiffrentChecksum = getLine1CoverageDataDifferentChecksum();
    try {
      LineCoverage.merge(line1, line1DiffrentChecksum);
    } catch (AssertionError er) {
      return;
    }
    fail();
  }
}
