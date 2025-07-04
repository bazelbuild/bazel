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
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Splitter;
import java.io.ByteArrayOutputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link LcovPrinter}. */
@RunWith(JUnit4.class)
public class LcovPrinterTest {

  @Test
  public void testPrintTwoFiles() throws Exception {
    Coverage coverage = new Coverage();
    SourceFileCoverage sourceFileCoverage1 = new SourceFileCoverage("src1.foo");
    SourceFileCoverage sourceFileCoverage2 = new SourceFileCoverage("src2.foo");
    sourceFileCoverage1.addLineNumber("foo", 2);
    sourceFileCoverage1.addLineNumber("bar", 4);
    sourceFileCoverage1.addFunctionExecution("foo", 3L);
    sourceFileCoverage1.addFunctionExecution("bar", 0L);
    sourceFileCoverage1.addLine(2, 3);
    sourceFileCoverage1.addLine(4, 0);
    sourceFileCoverage2.addLineNumber("foo", 3);
    sourceFileCoverage2.addFunctionExecution("foo", 1L);
    sourceFileCoverage2.addLine(3, 1);
    sourceFileCoverage2.addLine(4, 1);
    coverage.add(sourceFileCoverage1);
    coverage.add(sourceFileCoverage2);

    ByteArrayOutputStream byteOutputStream = new ByteArrayOutputStream();
    LcovPrinter.print(byteOutputStream, coverage);
    byteOutputStream.close();
    Iterable<String> fileLines = Splitter.on('\n').split(byteOutputStream.toString(UTF_8).strip());

    assertThat(fileLines)
        .containsExactly(
            "SF:src1.foo",
            "FN:4,bar",
            "FN:2,foo",
            "FNDA:0,bar",
            "FNDA:3,foo",
            "FNF:2",
            "FNH:1",
            "DA:2,3",
            "DA:4,0",
            "LH:1",
            "LF:2",
            "end_of_record",
            "SF:src2.foo",
            "FN:3,foo",
            "FNDA:1,foo",
            "FNF:1",
            "FNH:1",
            "DA:3,1",
            "DA:4,1",
            "LH:2",
            "LF:2",
            "end_of_record");
  }

  @Test
  public void testPrintOneFile() throws Exception {
    Coverage coverage = new Coverage();
    SourceFileCoverage sourceFileCoverage1 = new SourceFileCoverage("src1.foo");
    sourceFileCoverage1.addLineNumber("foo", 2);
    sourceFileCoverage1.addLineNumber("bar", 4);
    sourceFileCoverage1.addFunctionExecution("foo", 3L);
    sourceFileCoverage1.addFunctionExecution("bar", 0L);
    sourceFileCoverage1.addLine(2, 3);
    sourceFileCoverage1.addLine(4, 0);
    coverage.add(sourceFileCoverage1);

    ByteArrayOutputStream byteOutputStream = new ByteArrayOutputStream();
    LcovPrinter.print(byteOutputStream, coverage);
    byteOutputStream.close();
    Iterable<String> fileLines = Splitter.on('\n').split(byteOutputStream.toString(UTF_8).strip());

    assertThat(fileLines)
        .containsExactly(
            "SF:src1.foo",
            "FN:4,bar",
            "FN:2,foo",
            "FNDA:0,bar",
            "FNDA:3,foo",
            "FNF:2",
            "FNH:1",
            "DA:2,3",
            "DA:4,0",
            "LH:1",
            "LF:2",
            "end_of_record");
  }

  @Test
  public void testPrintBrdaLines() throws Exception {
    SourceFileCoverage sourceFile = new SourceFileCoverage("foo");
    sourceFile.addNewBrdaBranch(3, "0", "0", true, 1);
    sourceFile.addNewBrdaBranch(3, "0", "1", true, 0);
    sourceFile.addNewBrdaBranch(7, "0", "0", false, 0);
    sourceFile.addNewBrdaBranch(7, "0", "1", false, 0);
    Coverage coverage = new Coverage();
    coverage.add(sourceFile);

    ByteArrayOutputStream byteOutputStream = new ByteArrayOutputStream();
    LcovPrinter.print(byteOutputStream, coverage);
    Iterable<String> fileLines = Splitter.on('\n').split(byteOutputStream.toString(UTF_8));

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

  @Test
  public void testPrintBaLines() throws Exception {
    Coverage coverage = new Coverage();
    SourceFileCoverage sourceFile = new SourceFileCoverage("foo");
    sourceFile.addNewBranch(3, 2);
    sourceFile.addNewBranch(3, 1);
    sourceFile.addNewBranch(7, 0);
    sourceFile.addNewBranch(7, 0);
    coverage.add(sourceFile);

    ByteArrayOutputStream byteOutputStream = new ByteArrayOutputStream();
    LcovPrinter.print(byteOutputStream, coverage);
    Iterable<String> fileLines = Splitter.on('\n').split(byteOutputStream.toString(UTF_8));

    assertThat(fileLines)
        .containsExactly(
            "SF:foo",
            "FNF:0",
            "FNH:0",
            "BA:3,2",
            "BA:3,1",
            "BA:7,0",
            "BA:7,0",
            "BRF:4",
            "BRH:1",
            "LH:0",
            "LF:0",
            "end_of_record",
            "");
  }
}
