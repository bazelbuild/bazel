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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
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
  public void testOneTracefile() throws Exception {
    SourceFileCoverage sourceFileCoverage = new SourceFileCoverage("src.foo");
    sourceFileCoverage.addLine(1, 1);
    sourceFileCoverage.addLine(2, 1);

    coverage.add(sourceFileCoverage);

    assertThat(coverage.getAllSourceFiles()).hasSize(1);
    assertThat(Iterables.get(coverage.getAllSourceFiles(), 0).getLines())
        .containsExactly(1, 1L, 2, 1L);
  }

  @Test
  public void testOverlappingTracefilesMerge() throws Exception {
    SourceFileCoverage sourceFileCoverage1 = new SourceFileCoverage("src.foo");
    SourceFileCoverage sourceFileCoverage2 = new SourceFileCoverage("src.foo");
    sourceFileCoverage1.addLine(1, 2);
    sourceFileCoverage1.addLine(2, 1);
    sourceFileCoverage1.addLine(3, 2);
    sourceFileCoverage1.addBranch(1, BranchCoverage.create(1, 0, 2));
    sourceFileCoverage1.addBranch(1, BranchCoverage.create(1, 1, 1));
    sourceFileCoverage2.addLine(1, 3);
    sourceFileCoverage2.addLine(2, 3);
    sourceFileCoverage2.addLine(3, 0);
    sourceFileCoverage2.addBranch(1, BranchCoverage.create(1, 0, 1));
    sourceFileCoverage2.addBranch(1, BranchCoverage.create(1, 1, 2));

    coverage.add(sourceFileCoverage1);
    coverage.add(sourceFileCoverage2);

    assertThat(coverage.getAllSourceFiles()).hasSize(1);
    assertThat(Iterables.get(coverage.getAllSourceFiles(), 0).getLines())
        .containsExactly(1, 5L, 2, 4L, 3, 2L);
    assertThat(Iterables.get(coverage.getAllSourceFiles(), 0).getAllBranches())
        .containsExactly(BranchCoverage.create(1, 0, 2), BranchCoverage.create(1, 1, 2));
  }

  @Test
  public void testDistinctTracefiles() throws Exception {
    SourceFileCoverage sourceFileCoverage1 = new SourceFileCoverage("src_1.foo");
    SourceFileCoverage sourceFileCoverage2 = new SourceFileCoverage("src_2.foo");
    sourceFileCoverage1.addLine(1, 1L);
    sourceFileCoverage1.addLine(2, 1L);
    sourceFileCoverage2.addLine(1, 3L);
    sourceFileCoverage2.addLine(2, 3L);

    coverage.add(sourceFileCoverage1);
    coverage.add(sourceFileCoverage2);

    assertThat(coverage.getAllSourceFiles()).hasSize(2);
    assertThat(Iterables.get(coverage.getAllSourceFiles(), 0).sourceFileName())
        .isEqualTo("src_1.foo");
    assertThat(Iterables.get(coverage.getAllSourceFiles(), 1).sourceFileName())
        .isEqualTo("src_2.foo");
    assertThat(Iterables.get(coverage.getAllSourceFiles(), 0).getLines())
        .containsExactly(1, 1L, 2, 1L);
    assertThat(Iterables.get(coverage.getAllSourceFiles(), 1).getLines())
        .containsExactly(1, 3L, 2, 3L);
  }

  @Test
  public void testFilterSources() throws Exception {
    Coverage coverage = new Coverage();

    coverage.add(new SourceFileCoverage("/filterOut/package/file1.c"));
    coverage.add(new SourceFileCoverage("/filterOut/package/file2.c"));
    SourceFileCoverage validSource1 = new SourceFileCoverage("/valid/package/file3.c");
    coverage.add(validSource1);
    SourceFileCoverage validSource2 = new SourceFileCoverage("/valid/package/file4.c");
    coverage.add(validSource2);
    Collection<SourceFileCoverage> filteredSources =
        Coverage.filterOutMatchingSources(
                coverage, ImmutableList.of(Pattern.compile("/filterOut/package/.+")))
            .getAllSourceFiles();

    assertThat(filteredSources).containsExactly(validSource1, validSource2);
  }

  @Test
  public void testFilterSourcesEmptyResult() throws Exception {
    Coverage coverage = new Coverage();

    coverage.add(new SourceFileCoverage("/filterOut/package/file1.c"));
    coverage.add(new SourceFileCoverage("/filterOut/package/file2.c"));
    Collection<SourceFileCoverage> filteredSources =
        Coverage.filterOutMatchingSources(
                coverage, ImmutableList.of(Pattern.compile("/filterOut/package/.+")))
            .getAllSourceFiles();

    assertThat(filteredSources).isEmpty();
  }

  @Test
  public void testFilterSourcesNoMatches() throws Exception {
    Coverage coverage = new Coverage();

    SourceFileCoverage validSource1 = new SourceFileCoverage("/valid/package/file3.c");
    coverage.add(validSource1);
    SourceFileCoverage validSource2 = new SourceFileCoverage("/valid/package/file4.c");
    coverage.add(validSource2);
    Collection<SourceFileCoverage> filteredSources =
        Coverage.filterOutMatchingSources(
                coverage, ImmutableList.of(Pattern.compile("/something/else/.+")))
            .getAllSourceFiles();

    assertThat(filteredSources).containsExactly(validSource1, validSource2);
  }

  @Test
  public void testFilterSourcesMultipleRegex() throws Exception {
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
                coverage,
                ImmutableList.of(
                    Pattern.compile("/filterOut/package/.+"), Pattern.compile(".+external.+")))
            .getAllSourceFiles();

    assertThat(filteredSources).containsExactly(validSource1, validSource2);
  }

  @Test
  public void testFilterSourcesNoFilter() throws Exception {
    Coverage coverage = new Coverage();

    SourceFileCoverage validSource1 = new SourceFileCoverage("/valid/package/file3.c");
    coverage.add(validSource1);
    SourceFileCoverage validSource2 = new SourceFileCoverage("/valid/package/file4.c");
    coverage.add(validSource2);
    Collection<SourceFileCoverage> filteredSources =
        Coverage.filterOutMatchingSources(coverage, ImmutableList.of()).getAllSourceFiles();

    assertThat(filteredSources).containsExactly(validSource1, validSource2);
  }

  @Test
  public void testFilterSourcesNullCoverage() {
    assertThrows(
        IllegalArgumentException.class,
        () -> Coverage.filterOutMatchingSources(null, ImmutableList.of()));
  }

  @Test
  public void testFilterSourcesNullRegex() {
    assertThrows(
        IllegalArgumentException.class,
        () -> Coverage.filterOutMatchingSources(new Coverage(), null));
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
  public void testGetOnlyTheseSources() throws Exception {
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
                Coverage.getOnlyTheseSources(coverage, sourcesToKeep).getAllSourceFiles()))
        .containsExactly("source/common/protobuf/utility.cc", "source/common/grpc/common.cc");
  }

  @Test
  public void testGetOnlyTheseSourcesNullCoverage() {
    assertThrows(
        IllegalArgumentException.class, () -> Coverage.getOnlyTheseSources(null, new HashSet<>()));
  }

  @Test
  public void testGetOnlyTheseSourcesNullSources() {
    assertThrows(
        IllegalArgumentException.class, () -> Coverage.getOnlyTheseSources(new Coverage(), null));
  }

  @Test
  public void testGetOnlyTheseSourcesEmptySources() throws Exception {
    Coverage coverage = new Coverage();
    coverage.add(new SourceFileCoverage("source/common/protobuf/utility.cc"));
    coverage.add(new SourceFileCoverage("source/common/grpc/common.cc"));
    coverage.add(new SourceFileCoverage("source/server/options.cc"));
    coverage.add(new SourceFileCoverage("source/server/manager.cc"));

    assertThat(Coverage.getOnlyTheseSources(coverage, new HashSet<>()).getAllSourceFiles())
        .isEmpty();
  }
}
