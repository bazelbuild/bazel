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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@GcovParser}. */
@RunWith(JUnit4.class)
public class GcovParserTest {

  private static final ImmutableList<String> GCOV_INFO_FILE =
      ImmutableList.of(
          "version: 8.1.0 20180103",
          "cwd:/home/gcc/testcase",
          "file:tmp.cpp",
          "function:7,7,0,_ZN3FooIcEC2Ev",
          "function:7,7,1,_ZN3FooIiEC2Ev",
          "function:8,8,0,_ZN3FooIcE3incEv",
          "function:8,8,2,_ZN3FooIiE3incEv",
          "function:18,37,1,main",
          "lcount:7,0,1",
          "lcount:7,1,0",
          "lcount:8,0,1",
          "lcount:8,2,0",
          "lcount:18,1,0",
          "lcount:21,1,0",
          "branch:21,taken",
          "branch:21,nottaken",
          "lcount:23,1,0",
          "branch:23,taken",
          "branch:23,nottaken",
          "lcount:24,1,0",
          "branch:24,taken",
          "branch:24,nottaken",
          "lcount:25,1,0",
          "lcount:27,11,0",
          "branch:27,taken",
          "branch:27,taken",
          "lcount:28,10,0",
          "lcount:30,1,1",
          "branch:30,nottaken",
          "branch:30,taken",
          "lcount:32,1,0",
          "branch:32,nottaken",
          "branch:32,taken",
          "lcount:33,0,1",
          "branch:33,notexec",
          "branch:33,notexec",
          "lcount:35,1,0",
          "branch:35,taken",
          "branch:35,nottaken",
          "lcount:36,1,0");

  private static final ImmutableList<String> GCOV_INFO_FILE2 =
      ImmutableList.of(
          "file:tmp.cpp",
          "function:7,0,_ZN3FooIcEC2Ev",
          "function:7,1,_ZN3FooIiEC2Ev",
          "function:8,0,_ZN3FooIcE3incEv",
          "function:8,2,_ZN3FooIiE3incEv",
          "function:18,1,main",
          "lcount:7,0",
          "lcount:7,1",
          "lcount:8,0",
          "lcount:8,2",
          "lcount:18,1",
          "lcount:21,1",
          "branch:21,taken",
          "branch:21,nottaken",
          "lcount:23,1",
          "branch:23,taken",
          "branch:23,nottaken",
          "lcount:24,1",
          "branch:24,taken",
          "branch:24,nottaken",
          "lcount:25,1",
          "lcount:27,11",
          "branch:27,taken",
          "branch:27,taken",
          "lcount:28,10",
          "lcount:30,1",
          "branch:30,nottaken",
          "branch:30,taken",
          "lcount:32,1",
          "branch:32,nottaken",
          "branch:32,taken",
          "lcount:33,0",
          "branch:33,notexec",
          "branch:33,notexec",
          "lcount:35,1",
          "branch:35,taken",
          "branch:35,nottaken",
          "lcount:36,1");

  @Test
  public void testParseInvalidFile() throws IOException {
    assertThat(GcovParser.parse(new ByteArrayInputStream("Invalid gcov file".getBytes(UTF_8))))
        .isEmpty();
  }

  @Test
  public void testParseTracefileWithOneSourcefile() throws IOException {

    List<SourceFileCoverage> sourceFiles =
        GcovParser.parse(
            new ByteArrayInputStream(Joiner.on("\n").join(GCOV_INFO_FILE).getBytes(UTF_8)));
    assertThat(sourceFiles).hasSize(1);
    assertGcovInfoFile(sourceFiles.get(0));
  }

  @Test
  public void testParseTracefilWithDifferentFormat() throws IOException {
    List<SourceFileCoverage> sourceFiles =
        GcovParser.parse(
            new ByteArrayInputStream(Joiner.on("\n").join(GCOV_INFO_FILE2).getBytes(UTF_8)));
    assertThat(sourceFiles).hasSize(1);
    assertGcovInfoFile(sourceFiles.get(0));
  }

  private void assertGcovInfoFile(SourceFileCoverage sourceFileCoverage) {
    assertThat(sourceFileCoverage.sourceFileName()).isEqualTo("tmp.cpp");

    assertThat(sourceFileCoverage.nrFunctionsFound()).isEqualTo(5);
    assertThat(sourceFileCoverage.nrFunctionsHit()).isEqualTo(3);
    assertThat(sourceFileCoverage.nrOfInstrumentedLines()).isEqualTo(14);
    assertThat(sourceFileCoverage.nrOfLinesWithNonZeroExecution()).isEqualTo(13);
    assertThat(sourceFileCoverage.nrBranchesFound()).isEqualTo(8);
    assertThat(sourceFileCoverage.nrBranchesHit()).isEqualTo(7);

    assertThat(sourceFileCoverage.getAllLineExecution())
        .containsExactly(
            LineCoverage.create(7, 1, null),
            LineCoverage.create(8, 2, null),
            LineCoverage.create(18, 1, null),
            LineCoverage.create(21, 1, null),
            LineCoverage.create(23, 1, null),
            LineCoverage.create(24, 1, null),
            LineCoverage.create(25, 1, null),
            LineCoverage.create(27, 11, null),
            LineCoverage.create(28, 10, null),
            LineCoverage.create(30, 1, null),
            LineCoverage.create(32, 1, null),
            LineCoverage.create(33, 0, null),
            LineCoverage.create(35, 1, null),
            LineCoverage.create(36, 1, null));

    assertThat(sourceFileCoverage.getAllBranches())
        .containsExactly(
            BranchCoverage.create(21, 2),
            BranchCoverage.create(23, 2),
            BranchCoverage.create(24, 2),
            BranchCoverage.create(27, 2),
            BranchCoverage.create(30, 2),
            BranchCoverage.create(32, 2),
            BranchCoverage.create(33, 0),
            BranchCoverage.create(35, 2));
  }
}
