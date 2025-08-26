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
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/** Stores coverage information for a specific source file. */
class SourceFileCoverage {

  private String sourceFileName;
  private final SortedMap<String, Integer> functionLineNumbers; // function name to line numbers
  private final SortedMap<String, Long> functionsExecution; // function name to execution count
  private final ListMultimap<Integer, BranchCoverage> branches; // line number to branches
  private final SortedMap<Integer, Long> lines; // line number to line execution

  SourceFileCoverage(String sourcefile) {
    this.sourceFileName = sourcefile;
    this.functionsExecution = new TreeMap<>();
    this.functionLineNumbers = new TreeMap<>();
    this.lines = new TreeMap<>();
    this.branches = MultimapBuilder.treeKeys().arrayListValues().build();
  }

  SourceFileCoverage(SourceFileCoverage other) {
    this.sourceFileName = other.sourceFileName;

    this.functionsExecution = new TreeMap<>();
    this.functionLineNumbers = new TreeMap<>();
    this.lines = new TreeMap<>();
    this.branches = MultimapBuilder.treeKeys().arrayListValues().build();

    this.functionLineNumbers.putAll(other.functionLineNumbers);
    this.functionsExecution.putAll(other.functionsExecution);
    this.branches.putAll(other.branches);
    this.lines.putAll(other.lines);
  }

  void changeSourcefileName(String newSourcefileName) {
    this.sourceFileName = newSourcefileName;
  }

  /** Returns the merged functions found in the two given {@code SourceFileCoverage}s. */
  @VisibleForTesting
  static SortedMap<String, Integer> mergeFunctionLineNumbers(
      SourceFileCoverage s1, SourceFileCoverage s2) {
    SortedMap<String, Integer> merged = new TreeMap<>();
    merged.putAll(s1.functionLineNumbers);
    merged.putAll(s2.functionLineNumbers);
    return merged;
  }

  /** Returns the merged execution count found in the two given {@code SourceFileCoverage}s. */
  @VisibleForTesting
  static SortedMap<String, Long> mergeFunctionsExecution(
      SourceFileCoverage s1, SourceFileCoverage s2) {
    return Stream.of(s1.functionsExecution, s2.functionsExecution)
        .map(Map::entrySet)
        .flatMap(Collection::stream)
        .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, Long::sum, TreeMap::new));
  }

  static int getNumberOfBranchesHit(SourceFileCoverage sourceFileCoverage) {
    return (int)
        sourceFileCoverage.branches.values().stream().filter(BranchCoverage::wasExecuted).count();
  }

  /** Returns the merged line execution found in the two given {@code SourceFileCoverage}s. */
  @VisibleForTesting
  static SortedMap<Integer, Long> mergeLines(SourceFileCoverage s1, SourceFileCoverage s2) {
    SortedMap<Integer, Long> merged = new TreeMap<>();
    merged.putAll(s1.lines);
    for (Entry<Integer, Long> entry : s2.lines.entrySet()) {
      Long value = entry.getValue();
      Long old = merged.get(entry.getKey());
      if (old != null) {
        value = old + value;
      }
      merged.put(entry.getKey(), value);
    }
    return merged;
  }

  private static int getNumberOfExecutedLines(SourceFileCoverage sourceFileCoverage) {
    return (int)
        sourceFileCoverage.lines.entrySet().stream().filter(line -> line.getValue() > 0).count();
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
  static SourceFileCoverage merge(SourceFileCoverage source1, SourceFileCoverage source2) {
    assert source1.sourceFileName.equals(source2.sourceFileName);
    SourceFileCoverage merged = new SourceFileCoverage(source2.sourceFileName);

    merged.addAllFunctionLineNumbers(mergeFunctionLineNumbers(source1, source2));
    merged.addAllFunctionsExecution(mergeFunctionsExecution(source1, source2));
    merged.addAllBranches(source1.branches);
    merged.addAllBranches(source2.branches);
    merged.addAllLines(source1.lines);
    merged.addAllLines(source2.lines);
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

  @VisibleForTesting
  SortedMap<String, Integer> getFunctionLineNumbers() {
    return functionLineNumbers;
  }

  Set<Entry<String, Integer>> getAllFunctionLineNumbers() {
    return functionLineNumbers.entrySet();
  }

  @VisibleForTesting
  SortedMap<String, Long> getFunctionsExecution() {
    return functionsExecution;
  }

  Set<Entry<String, Long>> getAllExecutionCount() {
    return functionsExecution.entrySet();
  }

  Collection<BranchCoverage> getAllBranches() {
    return branches.values();
  }

  List<BranchCoverage> getBranches(int lineNumber) {
    return branches.get(lineNumber);
  }

  @VisibleForTesting
  Map<Integer, Long> getLines() {
    return lines;
  }

  Set<Entry<Integer, Long>> getAllLines() {
    return lines.entrySet();
  }

  void addFunctionLineNumber(String functionName, Integer lineNumber) {
    this.functionLineNumbers.put(functionName, lineNumber);
  }

  void addAllFunctionLineNumbers(SortedMap<String, Integer> lineNumber) {
    this.functionLineNumbers.putAll(lineNumber);
  }

  void addFunctionExecution(String functionName, Long executionCount) {
    this.functionsExecution.put(functionName, executionCount);
  }

  void addAllFunctionsExecution(SortedMap<String, Long> functionsExecution) {
    this.functionsExecution.putAll(functionsExecution);
  }

  /** Creates and adds a new branch to the source file. */
  void addNewBranch(
      int lineNumber,
      String blockNumber,
      String branchNumber,
      boolean evaluated,
      long executionCount) {
    BranchCoverage branch =
        BranchCoverage.create(lineNumber, blockNumber, branchNumber, evaluated, executionCount);
    addBranch(lineNumber, branch);
  }

  void addBranch(Integer lineNumber, BranchCoverage branch) {
    // if a line was already given for the same block and branch, merge it with the new one.
    for (int i = 0; i < branches.get(lineNumber).size(); i++) {
      BranchCoverage original = branches.get(lineNumber).get(i);
      if (original.blockNumber().equals(branch.blockNumber())
          && original.branchNumber().equals(branch.branchNumber())) {
        BranchCoverage merged = BranchCoverage.merge(original, branch);
        branches.get(lineNumber).set(i, merged);
        return;
      }
    }
    branches.put(lineNumber, branch);
  }

  void addAllBranches(ListMultimap<Integer, BranchCoverage> branches) {
    for (Entry<Integer, BranchCoverage> entry : branches.entries()) {
      addBranch(entry.getKey(), entry.getValue());
    }
  }

  void addLine(int lineNumber, long executionCount) {
    addLine(Integer.valueOf(lineNumber), Long.valueOf(executionCount));
  }

  void addLine(Integer lineNumber, Long executionCount) {
    Long old = lines.get(lineNumber);
    if (old != null) {
      executionCount = executionCount + old;
    }
    lines.put(lineNumber, executionCount);
  }

  void addAllLines(SortedMap<Integer, Long> lines) {
    for (Entry<Integer, Long> entry : lines.entrySet()) {
      addLine(entry.getKey(), entry.getValue());
    }
  }
}
