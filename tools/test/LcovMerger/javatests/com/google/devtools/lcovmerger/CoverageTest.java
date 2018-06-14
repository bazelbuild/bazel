// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.lcovmerger;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.lcovmerger.LcovMergerTestUtils.assertMergedSourceFile;
import static com.google.devtools.lcovmerger.LcovMergerTestUtils.assertTracefile1;
import static com.google.devtools.lcovmerger.LcovMergerTestUtils.createLinesExecution1;
import static com.google.devtools.lcovmerger.LcovMergerTestUtils.createLinesExecution2;
import static com.google.devtools.lcovmerger.LcovMergerTestUtils.createSourceFile1;
import static com.google.devtools.lcovmerger.LcovMergerTestUtils.createSourceFile2;

import com.google.common.collect.Iterables;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for LcovMerger.
 */
@RunWith(JUnit4.class)
public class CoverageTest {

  private Coverage coverage;

  @Before
  public void initializeCoverage() {
    coverage = new Coverage();
  }

  @Test
  public void testOneTracefile() {
    SourceFileCoverage sourceFileCoverage =
        createSourceFile1(createLinesExecution1());
    coverage.add(sourceFileCoverage);
    assertThat(coverage.getAllSourceFiles()).hasSize(1);
    assertTracefile1(Iterables.get(coverage.getAllSourceFiles(), 0));
  }

  @Test
  public void testTwoOverlappingTracefiles() {
    int[] linesExecution1 = createLinesExecution1();
    int[] linesExecution2 = createLinesExecution2();
    SourceFileCoverage sourceFileCoverage1 = createSourceFile1(linesExecution1);
    SourceFileCoverage sourceFileCoverage2 = createSourceFile2(linesExecution2);

    coverage.add(sourceFileCoverage1);
    coverage.add(sourceFileCoverage2);

    assertThat(coverage.getAllSourceFiles()).hasSize(1);
    SourceFileCoverage merged = Iterables.get(coverage.getAllSourceFiles(), 0);
    assertMergedSourceFile(merged, linesExecution1, linesExecution2);
  }

  @Test
  public void testTwoTracefiles() {
    SourceFileCoverage sourceFileCoverage1 =
        createSourceFile1(createLinesExecution1());
    SourceFileCoverage sourceFileCoverage2 = createSourceFile1(
        "SOME_OTHER_FILENAME", createLinesExecution1());

    coverage.add(sourceFileCoverage1);
    coverage.add(sourceFileCoverage2);
    assertThat(coverage.getAllSourceFiles()).hasSize(2);
    assertTracefile1(Iterables.get(coverage.getAllSourceFiles(), 0));
    assertTracefile1(Iterables.get(coverage.getAllSourceFiles(), 1));
  }
}
