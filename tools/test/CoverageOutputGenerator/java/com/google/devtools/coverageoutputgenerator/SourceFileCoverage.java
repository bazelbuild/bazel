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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.MultimapBuilder;
import com.google.common.collect.Sets;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/** Stores coverage information for a specific source file. */
class SourceFileCoverage {

  private String sourceFileName;
  private final TreeMap<String, Integer> lineNumbers; // function name to line numbers
  private final TreeMap<String, Long> functionsExecution; // function name to execution count
  private final ListMultimap<Integer, BranchCoverage> branches; // line number to branches
  private final TreeMap<Integer, LineCoverage> lines; // line number to line execution

  SourceFileCoverage(String sourcefile) {
    this.sourceFileName = sourcefile;
    this.functionsExecution = new TreeMap<>();
    this.lineNumbers = new TreeMap<>();
    this.lines = new TreeMap<>();
    this.branches = MultimapBuilder.treeKeys().arrayListValues().build();
  }

  SourceFileCoverage(SourceFileCoverage other) {
    this.sourceFileName = other.sourceFileName;

    this.functionsExecution = new TreeMap<>();
    this.lineNumbers = new TreeMap<>();
    this.lines = new TreeMap<>();
    this.branches = MultimapBuilder.treeKeys().arrayListValues().build();

    this.lineNumbers.putAll(other.lineNumbers);
    this.functionsExecution.putAll(other.functionsExecution);
    this.branches.putAll(other.branches);
    this.lines.putAll(other.lines);
  }

  void changeSourcefileName(String newSourcefileName) {
    this.sourceFileName = newSourcefileName;
  }

  /** Returns the merged functions found in the two given {@code SourceFileCoverage}s. */
  @VisibleForTesting
  static TreeMap<String, Integer> mergeLineNumbers(SourceFileCoverage s1, SourceFileCoverage s2) {
    TreeMap<String, Integer> merged = new TreeMap<>();
    merged.putAll(s1.lineNumbers);
    merged.putAll(s2.lineNumbers);
    return merged;
  }

  /** Returns the merged execution count found in the two given {@code SourceFileCoverage}s. */
  @VisibleForTesting
  static TreeMap<String, Long> mergeFunctionsExecution(
      SourceFileCoverage s1, SourceFileCoverage s2) {
    return Stream.of(s1.functionsExecution, s2.functionsExecution)
        .map(Map::entrySet)
        .flatMap(Collection::stream)
        .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, Long::sum, TreeMap::new));
  }

  /** Returns the merged branches found in the two given {@code SourceFileCoverage}s. */
  @VisibleForTesting
  static ListMultimap<Integer, BranchCoverage> mergeBranches(
      SourceFileCoverage s1, SourceFileCoverage s2) throws IncompatibleMergeException {

    ListMultimap<Integer, BranchCoverage> merged =
        MultimapBuilder.treeKeys().arrayListValues().build();

    for (int line : Sets.union(s1.branches.keySet(), s2.branches.keySet())) {
      Collection<BranchCoverage> s1Branches = s1.branches.get(line);
      Collection<BranchCoverage> s2Branches = s2.branches.get(line);

      if (s1Branches.isEmpty()) {
        merged.putAll(line, s2Branches);
      } else if (s2Branches.isEmpty()) {
        merged.putAll(line, s1Branches);
      } else if (s1Branches.size() != s2Branches.size()) {
        throw new IncompatibleMergeException(
            String.format(
                "Different number of branches found at line %d for %s", line, s1.sourceFileName));
      } else {
        Iterator<BranchCoverage> it1 = s1Branches.iterator();
        Iterator<BranchCoverage> it2 = s2Branches.iterator();
        while (it1.hasNext() && it2.hasNext()) {
          BranchCoverage b1 = it1.next();
          BranchCoverage b2 = it2.next();
          if (b1.lineNumber() != b2.lineNumber()
              || !b1.blockNumber().equals(b2.blockNumber())
              || !b1.branchNumber().equals(b2.branchNumber())) {
            throw new IncompatibleMergeException(
                String.format(
                    "Branches for %s do not align for source lines %d and %d",
                    s1.sourceFileName, b1.lineNumber(), b2.lineNumber()));
          }
          BranchCoverage branch = BranchCoverage.merge(b1, b2);
          merged.put(line, branch);
        }
      }
    }
    return merged;
  }

  static int getNumberOfBranchesHit(SourceFileCoverage sourceFileCoverage) {
    return (int)
        sourceFileCoverage.branches.values().stream().filter(BranchCoverage::wasExecuted).count();
  }

  /** Returns the merged line execution found in the two given {@code SourceFileCoverage}s. */
  @VisibleForTesting
  static TreeMap<Integer, LineCoverage> mergeLines(SourceFileCoverage s1, SourceFileCoverage s2) {
    return Stream.of(s1.lines, s2.lines)
        .map(Map::entrySet)
        .flatMap(Collection::stream)
        .collect(
            Collectors.toMap(
                Map.Entry::getKey, Map.Entry::getValue, LineCoverage::merge, TreeMap::new));
  }

  private static int getNumberOfExecutedLines(SourceFileCoverage sourceFileCoverage) {
    return (int)
        sourceFileCoverage.lines.entrySet().stream()
            .filter(line -> line.getValue().executionCount() > 0)
            .count();
  }

  /**
   * Merges all the fields of {@code other} with the current {@link SourceFileCoverage} into a new
   * {@link SourceFileCoverage}
   *
   * <p>Assumes both the current and the given {@link SourceFileCoverage} have the same {@code
   * sourceFileName}.
   *
   * @return a new {@link SourceFileCoverage} that contains the merged coverage.
   */
  static SourceFileCoverage merge(SourceFileCoverage source1, SourceFileCoverage source2)
      throws IncompatibleMergeException {
    assert source1.sourceFileName.equals(source2.sourceFileName);
    SourceFileCoverage merged = new SourceFileCoverage(source2.sourceFileName);

    merged.addAllLineNumbers(mergeLineNumbers(source1, source2));
    merged.addAllFunctionsExecution(mergeFunctionsExecution(source1, source2));
    merged.addAllBranches(mergeBranches(source1, source2));
    merged.addAllLines(mergeLines(source1, source2));
    return merged;
  }

  String sourceFileName() {
    return sourceFileName;
  }

  int nrFunctionsFound() {
    return functionsExecution.size();
  }

  int nrFunctionsHit() {
    return (int)
        functionsExecution.entrySet().stream().filter(function -> function.getValue() > 0).count();
  }

  int nrBranchesFound() {
    return branches.size();
  }

  int nrBranchesHit() {
    return getNumberOfBranchesHit(this);
  }

  int nrOfLinesWithNonZeroExecution() {
    return getNumberOfExecutedLines(this);
  }

  int nrOfInstrumentedLines() {
    return this.lines.size();
  }

  Collection<LineCoverage> getAllLineExecution() {
    return lines.values();
  }

  @VisibleForTesting
  TreeMap<String, Integer> getLineNumbers() {
    return lineNumbers;
  }

  Set<Entry<String, Integer>> getAllLineNumbers() {
    return lineNumbers.entrySet();
  }

  @VisibleForTesting
  TreeMap<String, Long> getFunctionsExecution() {
    return functionsExecution;
  }

  Set<Entry<String, Long>> getAllExecutionCount() {
    return functionsExecution.entrySet();
  }

  Collection<BranchCoverage> getAllBranches() {
    return branches.values();
  }

  @VisibleForTesting
  Map<Integer, LineCoverage> getLines() {
    return lines;
  }

  void addLineNumber(String functionName, Integer lineNumber) {
    this.lineNumbers.put(functionName, lineNumber);
  }

  void addAllLineNumbers(TreeMap<String, Integer> lineNumber) {
    this.lineNumbers.putAll(lineNumber);
  }

  void addFunctionExecution(String functionName, Long executionCount) {
    this.functionsExecution.put(functionName, executionCount);
  }

  void addAllFunctionsExecution(TreeMap<String, Long> functionsExecution) {
    this.functionsExecution.putAll(functionsExecution);
  }

  void addBranch(Integer lineNumber, BranchCoverage branch) {
    branches.put(lineNumber, branch);
  }

  void addAllBranches(ListMultimap<Integer, BranchCoverage> branches) {
    this.branches.putAll(branches);
  }

  void addLine(Integer lineNumber, LineCoverage line) {
    if (this.lines.get(lineNumber) != null) {
      this.lines.put(lineNumber, LineCoverage.merge(line, this.lines.get(lineNumber)));
      return;
    }
    this.lines.put(lineNumber, line);
  }

  void addAllLines(TreeMap<Integer, LineCoverage> lines) {
    this.lines.putAll(lines);
  }
}
