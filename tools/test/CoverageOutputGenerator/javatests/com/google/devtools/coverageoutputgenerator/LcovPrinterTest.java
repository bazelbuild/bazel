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
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.TRACEFILE1;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.TRACEFILE1_DIFFERENT_NAME;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.createLinesExecution1;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.createSourceFile1;

import com.google.common.base.Splitter;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link LcovPrinter}. */
@RunWith(JUnit4.class)
public class LcovPrinterTest {

  private SourceFileCoverage sourceFileCoverage1;
  private SourceFileCoverage sourceFileCoverage2;
  private ByteArrayOutputStream byteOutputStream;
  private Coverage coverage;

  @Before
  public void init() {
    sourceFileCoverage1 = createSourceFile1(createLinesExecution1());
    sourceFileCoverage2 =
        LcovMergerTestUtils.createSourceFile1(
            TRACEFILE1_DIFFERENT_NAME.get(0).substring(3), createLinesExecution1());
    byteOutputStream = new ByteArrayOutputStream();
    coverage = new Coverage();
  }

  @Test
  public void testPrintTwoFiles() throws Exception {
    coverage.add(sourceFileCoverage1);
    coverage.add(sourceFileCoverage2);

    LcovPrinter.print(byteOutputStream, coverage);
    byteOutputStream.close();

    Iterable<String> fileLines = Splitter.on('\n').split(byteOutputStream.toString());

    List<String> tracefiles = new ArrayList<>();
    tracefiles.addAll(TRACEFILE1_DIFFERENT_NAME);
    tracefiles.addAll(TRACEFILE1);

    // Last line of the file will always be a newline.
    assertThat(fileLines).hasSize(tracefiles.size() + 1);
    int lineIndex = 0;
    for (String line : fileLines) {
      if (lineIndex == tracefiles.size()) {
        break;
      }
      assertThat(line).isEqualTo(tracefiles.get(lineIndex++));
    }
  }

  @Test
  public void testPrintOneFile() throws Exception {
    coverage.add(sourceFileCoverage1);
    LcovPrinter.print(byteOutputStream, coverage);
    byteOutputStream.close();
    Iterable<String> fileLines = Splitter.on('\n').split(byteOutputStream.toString());
    // Last line of the file will always be a newline.
    assertThat(fileLines).hasSize(TRACEFILE1.size() + 1);
    int lineIndex = 0;
    for (String line : fileLines) {
      if (lineIndex == TRACEFILE1.size()) {
        break;
      }
      assertThat(line).isEqualTo(TRACEFILE1.get(lineIndex++));
    }
  }

  @Test
  public void testPrintBrdaLines() throws Exception {
    SourceFileCoverage sourceFile = new SourceFileCoverage("foo");
    sourceFile.addBranch(3, BranchCoverage.createWithBlockAndBranch(3, "0", "0", true, 1));
    sourceFile.addBranch(3, BranchCoverage.createWithBlockAndBranch(3, "0", "1", true, 0));
    sourceFile.addBranch(7, BranchCoverage.createWithBlockAndBranch(7, "0", "0", false, 0));
    sourceFile.addBranch(7, BranchCoverage.createWithBlockAndBranch(7, "0", "1", false, 0));
    coverage.add(sourceFile);

    LcovPrinter.print(byteOutputStream, coverage);
    Iterable<String> fileLines = Splitter.on('\n').split(byteOutputStream.toString());
    assertThat(fileLines)
        .containsExactly(
            "SF:foo",
            "FNF:0",
            "FNH:0",
            "BRDA:3,0,0,1",
            "BRDA:3,0,1,0",
            "BRDA:7,0,0,-",
            "BRDA:7,0,1,-",
            "BRF:4",
            "BRH:1",
            "LH:0",
            "LF:0",
            "end_of_record",
            "");
  }
}
