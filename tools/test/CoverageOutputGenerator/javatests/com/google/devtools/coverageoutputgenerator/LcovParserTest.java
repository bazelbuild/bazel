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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link LcovParser}. */
@RunWith(JUnit4.class)
public class LcovParserTest {

  @Test
  public void testParseInvalidTracefile() throws IOException {
    List<SourceFileCoverage> sourceFiles =
        LcovParser.parse(new ByteArrayInputStream("Invalid lcov tracefile".getBytes(UTF_8)));
    assertThat(sourceFiles).isEmpty();
  }

  @Test
  public void testParseTracefile() throws IOException {
    ImmutableList<String> lcovLines =
        ImmutableList.of(
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

    List<SourceFileCoverage> sourceFiles =
        LcovParser.parse(new ByteArrayInputStream(Joiner.on("\n").join(lcovLines).getBytes(UTF_8)));

    assertThat(sourceFiles).hasSize(2);
    assertThat(sourceFiles.get(0).sourceFileName()).isEqualTo("src1.foo");
    assertThat(sourceFiles.get(1).sourceFileName()).isEqualTo("src2.foo");
    assertThat(sourceFiles.get(0).getLines())
        .containsExactly(
            2, 3L,
            4, 0L);
    assertThat(sourceFiles.get(1).getLines()).containsExactly(3, 1L, 4, 1L);
    assertThat(sourceFiles.get(0).getLineNumbers()).containsExactly("bar", 4, "foo", 2);
    assertThat(sourceFiles.get(1).getLineNumbers()).containsExactly("foo", 3);
    assertThat(sourceFiles.get(0).getFunctionsExecution()).containsExactly("bar", 0L, "foo", 3L);
    assertThat(sourceFiles.get(1).getFunctionsExecution()).containsExactly("foo", 1L);
    assertThat(sourceFiles.get(0).getAllBranches()).isEmpty();
    assertThat(sourceFiles.get(1).getAllBranches()).isEmpty();
  }

  @Test
  public void testParseTracefileWithLargeCounts() throws IOException {
    List<String> tracefile =
        ImmutableList.of(
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
        LcovParser.parse(new ByteArrayInputStream(Joiner.on("\n").join(tracefile).getBytes(UTF_8)));
    SourceFileCoverage sourceFile = sourceFiles.get(0);

    Map<String, Long> functions = sourceFile.getFunctionsExecution();
    assertThat(functions).containsEntry("file1-func1", 1000000000000L);

    assertThat(sourceFile.getLines()).containsExactly(4, 1000000000000L, 5, 1000000000000L);
  }

  @Test
  public void testParseBrdaBranches() throws IOException {
    List<String> traceFile =
        ImmutableList.of(
            "SF:SOURCE_FILE",
            "FN:2,func",
            "FNDA:1,func",
            "DA:2,1",
            "DA:3,1",
            "DA:4,1",
            "DA:5,1",
            "DA:6,1",
            "BRDA:6,0,0,1",
            "BRDA:6,0,1,0",
            "DA:7,13",
            "BRDA:7,0,0,12",
            "BRDA:7,0,1,1",
            "DA:8,12",
            "DA:10,1",
            "DA:12,0",
            "BRDA:12,0,0,-",
            "BRDA:12,0,1,-",
            "DA:13,0",
            "DA:14.0",
            "DA:16,0",
            "end_of_record");
    List<SourceFileCoverage> sourceFiles =
        LcovParser.parse(new ByteArrayInputStream(Joiner.on("\n").join(traceFile).getBytes(UTF_8)));
    SourceFileCoverage sourceFile = sourceFiles.get(0);

    List<BranchCoverage> branches =
        sourceFile.getAllBranches().stream().collect(Collectors.toList());
    assertThat(branches)
        .containsExactly(
            BranchCoverage.createWithBlockAndBranch(6, "0", "0", true, 1),
            BranchCoverage.createWithBlockAndBranch(6, "0", "1", true, 0),
            BranchCoverage.createWithBlockAndBranch(7, "0", "0", true, 12),
            BranchCoverage.createWithBlockAndBranch(7, "0", "1", true, 1),
            BranchCoverage.createWithBlockAndBranch(12, "0", "0", false, 0),
            BranchCoverage.createWithBlockAndBranch(12, "0", "1", false, 0));
  }

  @Test
  public void testParseBaBranches() throws IOException {
    List<String> traceFile =
        ImmutableList.of(
            "SF:SOURCE_FILE",
            "FN:2,func",
            "FNDA:1,func",
            "DA:1,5",
            "BA:2,1",
            "BA:2,2",
            "DA:3,0",
            "BA:4,0",
            "BA:4,0",
            "DA:5,0",
            "DA:6,5",
            "BA:7,2",
            "BA:7,1",
            "BA:7,2",
            "DA:8,1",
            "DA:9,0",
            "DA:10,4",
            "end_of_record");
    List<SourceFileCoverage> sourceFiles =
        LcovParser.parse(new ByteArrayInputStream(Joiner.on("\n").join(traceFile).getBytes(UTF_8)));
    SourceFileCoverage sourceFile = sourceFiles.get(0);

    List<BranchCoverage> branches =
        sourceFile.getAllBranches().stream().collect(Collectors.toList());
    assertThat(branches)
        .containsExactly(
            BranchCoverage.create(2, 0, 1),
            BranchCoverage.create(2, 1, 2),
            BranchCoverage.create(4, 0, 0),
            BranchCoverage.create(4, 1, 0),
            BranchCoverage.create(7, 0, 2),
            BranchCoverage.create(7, 1, 1),
            BranchCoverage.create(7, 2, 2));
  }

  @Test
  public void testParseFnWithEnd() throws IOException {
    List<String> traceFile = ImmutableList.of("SF:SOURCE_FILE", "FN:2,3,func", "end_of_record");
    List<SourceFileCoverage> sourceFiles =
        LcovParser.parse(new ByteArrayInputStream(Joiner.on("\n").join(traceFile).getBytes(UTF_8)));
    SourceFileCoverage sourceFile = sourceFiles.get(0);

    assertThat(sourceFile.getAllLineNumbers()).containsExactly(Map.entry("func", 2));
  }

  @Test
  public void testParseLineWithHash() throws IOException {
    ImmutableList<String> traceFile =
        ImmutableList.of("SF:src.foo", "DA:1,1,hash", "end_of_record");

    List<SourceFileCoverage> sourceFiles =
        LcovParser.parse(new ByteArrayInputStream(Joiner.on("\n").join(traceFile).getBytes(UTF_8)));

    assertThat(sourceFiles.get(0).getLines()).containsExactly(1, 1L);
  }
}
