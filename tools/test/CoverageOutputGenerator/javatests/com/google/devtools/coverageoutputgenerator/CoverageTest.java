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

package com.google.devtools.coverageoutputgenerator;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.assertMergedSourceFile;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.assertTracefile1;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.createLinesExecution1;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.createLinesExecution2;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.createSourceFile1;
import static com.google.devtools.coverageoutputgenerator.LcovMergerTestUtils.createSourceFile2;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for LcovMerger. */
@RunWith(JUnit4.class)
public class CoverageTest {

  private Coverage coverage;

  @Before
  public void initializeCoverage() {
    coverage = new Coverage();
  }

  @Test
  public void testOneTracefile() {
    SourceFileCoverage sourceFileCoverage = createSourceFile1(createLinesExecution1());
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
    SourceFileCoverage sourceFileCoverage1 = createSourceFile1(createLinesExecution1());
    SourceFileCoverage sourceFileCoverage2 =
        createSourceFile1("SOME_OTHER_FILENAME", createLinesExecution1());

    coverage.add(sourceFileCoverage1);
    coverage.add(sourceFileCoverage2);
    assertThat(coverage.getAllSourceFiles()).hasSize(2);
    assertTracefile1(Iterables.get(coverage.getAllSourceFiles(), 0));
    assertTracefile1(Iterables.get(coverage.getAllSourceFiles(), 1));
  }

  @Test
  public void testFilterSources() {
    Coverage coverage = new Coverage();

    coverage.add(new SourceFileCoverage("/filterOut/package/file1.c"));
    coverage.add(new SourceFileCoverage("/filterOut/package/file2.c"));
    SourceFileCoverage validSource1 = new SourceFileCoverage("/valid/package/file3.c");
    coverage.add(validSource1);
    SourceFileCoverage validSource2 = new SourceFileCoverage("/valid/package/file4.c");
    coverage.add(validSource2);
    Collection<SourceFileCoverage> filteredSources =
        Coverage.filterOutMatchingSources(coverage, ImmutableList.of("/filterOut/package/.+"))
            .getAllSourceFiles();

    assertThat(filteredSources).containsExactly(validSource1, validSource2);
  }

  @Test
  public void testFilterSourcesEmptyResult() {
    Coverage coverage = new Coverage();

    coverage.add(new SourceFileCoverage("/filterOut/package/file1.c"));
    coverage.add(new SourceFileCoverage("/filterOut/package/file2.c"));
    Collection<SourceFileCoverage> filteredSources =
        Coverage.filterOutMatchingSources(coverage, ImmutableList.of("/filterOut/package/.+"))
            .getAllSourceFiles();

    assertThat(filteredSources).isEmpty();
  }

  @Test
  public void testFilterSourcesNoMatches() {
    Coverage coverage = new Coverage();

    SourceFileCoverage validSource1 = new SourceFileCoverage("/valid/package/file3.c");
    coverage.add(validSource1);
    SourceFileCoverage validSource2 = new SourceFileCoverage("/valid/package/file4.c");
    coverage.add(validSource2);
    Collection<SourceFileCoverage> filteredSources =
        Coverage.filterOutMatchingSources(coverage, ImmutableList.of("/something/else/.+"))
            .getAllSourceFiles();

    assertThat(filteredSources).containsExactly(validSource1, validSource2);
  }

  @Test
  public void testFilterSourcesMultipleRegex() {
    Coverage coverage = new Coverage();

    coverage.add(new SourceFileCoverage("/filterOut/package/file1.c"));
    coverage.add(new SourceFileCoverage("/filterOut/package/file2.c"));
    coverage.add(new SourceFileCoverage("/repo/external/p.c"));
    SourceFileCoverage validSource1 = new SourceFileCoverage("/valid/package/file3.c");
    coverage.add(validSource1);
    SourceFileCoverage validSource2 = new SourceFileCoverage("/valid/package/file4.c");
    coverage.add(validSource2);
    Collection<SourceFileCoverage> filteredSources =
        Coverage.filterOutMatchingSources(
                coverage, ImmutableList.of("/filterOut/package/.+", ".+external.+"))
            .getAllSourceFiles();

    assertThat(filteredSources).containsExactly(validSource1, validSource2);
  }

  @Test
  public void testFilterSourcesNoFilter() {
    Coverage coverage = new Coverage();

    SourceFileCoverage validSource1 = new SourceFileCoverage("/valid/package/file3.c");
    coverage.add(validSource1);
    SourceFileCoverage validSource2 = new SourceFileCoverage("/valid/package/file4.c");
    coverage.add(validSource2);
    Collection<SourceFileCoverage> filteredSources =
        Coverage.filterOutMatchingSources(coverage, ImmutableList.of()).getAllSourceFiles();

    assertThat(filteredSources).containsExactly(validSource1, validSource2);
  }

  @Test(expected = IllegalArgumentException.class)
  public void testFilterSourcesNullCoverage() {
    Coverage.filterOutMatchingSources(null, ImmutableList.of());
  }

  @Test(expected = IllegalArgumentException.class)
  public void testFilterSourcesNullRegex() {
    Coverage.filterOutMatchingSources(new Coverage(), null);
  }

  private List<String> getSourceFileNames(
      Collection<SourceFileCoverage> sourceFileCoverageCollection) {
    ImmutableList.Builder<String> sourceFilenames = ImmutableList.builder();
    for (SourceFileCoverage sourceFileCoverage : sourceFileCoverageCollection) {
      sourceFilenames.add(sourceFileCoverage.sourceFileName());
    }
    return sourceFilenames.build();
  }

  @Test
  public void testGetOnlyTheseSources() {
    Coverage coverage = new Coverage();
    coverage.add(new SourceFileCoverage("source/common/protobuf/utility.cc"));
    coverage.add(new SourceFileCoverage("source/common/grpc/common.cc"));
    coverage.add(new SourceFileCoverage("source/server/options.cc"));
    coverage.add(new SourceFileCoverage("source/server/manager.cc"));

    Set<String> sourcesToKeep = new HashSet<>();
    sourcesToKeep.add("source/common/protobuf/utility.cc");
    sourcesToKeep.add("source/common/grpc/common.cc");

    assertThat(
            getSourceFileNames(
                Coverage.getOnlyTheseCcSources(coverage, sourcesToKeep).getAllSourceFiles()))
        .containsExactly("source/common/protobuf/utility.cc", "source/common/grpc/common.cc");
  }

  @Test(expected = IllegalArgumentException.class)
  public void testGetOnlyTheseSourcesNullCoverage() {
    Coverage.getOnlyTheseCcSources(null, new HashSet<>());
  }

  @Test(expected = IllegalArgumentException.class)
  public void testGetOnlyTheseSourcesNullSources() {
    Coverage.getOnlyTheseCcSources(new Coverage(), null);
  }

  @Test
  public void testGetOnlyTheseSourcesEmptySources() {
    Coverage coverage = new Coverage();
    coverage.add(new SourceFileCoverage("source/common/protobuf/utility.cc"));
    coverage.add(new SourceFileCoverage("source/common/grpc/common.cc"));
    coverage.add(new SourceFileCoverage("source/server/options.cc"));
    coverage.add(new SourceFileCoverage("source/server/manager.cc"));

    assertThat(Coverage.getOnlyTheseCcSources(coverage, new HashSet<>()).getAllSourceFiles())
        .isEmpty();
  }
}
