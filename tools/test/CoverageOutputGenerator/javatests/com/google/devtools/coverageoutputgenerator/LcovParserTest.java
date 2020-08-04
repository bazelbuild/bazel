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
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.TRACEFILE2;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.assertTracefile1;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.assertTracefile2;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@LcovParser}. */
@RunWith(JUnit4.class)
public class LcovParserTest {

  @Test
  public void testParseInvalidTracefile() throws IOException {
    List<SourceFileCoverage> sourceFiles =
        LcovParser.parse(new ByteArrayInputStream("Invalid lcov tracefile".getBytes(UTF_8)));
    assertThat(sourceFiles).isEmpty();
  }

  @Test
  public void testParseTracefileWithOneSourcefile() throws IOException {
    List<SourceFileCoverage> sourceFiles =
        LcovParser.parse(
            new ByteArrayInputStream(Joiner.on("\n").join(TRACEFILE1).getBytes(UTF_8)));
    assertThat(sourceFiles).hasSize(1);
    assertTracefile1(sourceFiles.get(0));
  }

  @Test
  public void testParseTracefileWithTwoSourcefiles() throws IOException {
    List<String> tracefile2ModifiedLines = new ArrayList<>();
    tracefile2ModifiedLines.addAll(TRACEFILE2);
    tracefile2ModifiedLines.set(0, "SF:BSOME_OTHER_FILE_THAT_IS_NOT_MERGED");

    List<String> tracefileLines = new ArrayList<>();
    tracefileLines.addAll(TRACEFILE1);
    tracefileLines.addAll(tracefile2ModifiedLines);

    InputStream inputStream =
        new ByteArrayInputStream(Joiner.on("\n").join(tracefileLines).getBytes(UTF_8));
    List<SourceFileCoverage> sourceFiles = LcovParser.parse(inputStream);

    assertThat(sourceFiles).hasSize(2);
    assertTracefile1(sourceFiles.get(0));
    assertTracefile2(sourceFiles.get(1));
  }

  @Test
  public void testParseTracefileWithLargeCounts() throws IOException {
      List<String> tracefile = ImmutableList.of(
          "SF:SOURCE_FILENAME",
          "FN:4,file1-func1",
          "FNDA:1000000000000,file1-func1",
          "FNF:1",
          "FNH:1",
          "DA:4,1000000000000",
          "DA:5,1000000000000",
          "LH:2",
          "LF:2",
          "end_of_record");

      List<SourceFileCoverage> sourceFiles =
          LcovParser.parse(
              new ByteArrayInputStream(Joiner.on("\n").join(tracefile).getBytes(UTF_8)));
      SourceFileCoverage sourceFile = sourceFiles.get(0);

      Map<String, Long> functions = sourceFile.getFunctionsExecution();
      assertThat(functions.get("file1-func1")).isEqualTo(1000000000000L);

      Map<Integer, LineCoverage> lines = sourceFile.getLines();
      assertThat(lines.get(4).executionCount()).isEqualTo(1000000000000L);
      assertThat(lines.get(5).executionCount()).isEqualTo(1000000000000L);
  }

}
