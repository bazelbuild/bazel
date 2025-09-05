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
  private final LineCoverage lineCoverage;

  public SourceFileCoverage(String sourcefile) {
    this.sourceFileName = sourcefile;
    this.functionsExecution = new TreeMap<>();
    this.functionLineNumbers = new TreeMap<>();
    this.lineCoverage = LineCoverage.create();
    this.branches = MultimapBuilder.treeKeys().arrayListValues().build();
  }

  SourceFileCoverage(SourceFileCoverage other) {
    this.sourceFileName = other.sourceFileName;

    this.functionsExecution = new TreeMap<>();
    this.functionLineNumbers = new TreeMap<>();
    this.branches = MultimapBuilder.treeKeys().arrayListValues().build();

    this.functionLineNumbers.putAll(other.functionLineNumbers);
    this.functionsExecution.putAll(other.functionsExecution);
    this.branches.putAll(other.branches);
    this.lineCoverage = LineCoverage.copy(other.lineCoverage);
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

  /**
   * Merges all the fields of {@code other} with the current {@link SourceFileCoverage} into a new
   * {@link SourceFileCoverage}
   *
   * <p>Assumes both the current and the given {@link SourceFileCoverage} have the same {@code
   * sourceFileName}.
   *
   * @return a new {@link SourceFileCoverage} that contains the merged coverage.
   */
  public static SourceFileCoverage merge(SourceFileCoverage source1, SourceFileCoverage source2) {
    assert source1.sourceFileName.equals(source2.sourceFileName);
    SourceFileCoverage merged = new SourceFileCoverage(source1);

    merged.addAllFunctionLineNumbers(source2.functionLineNumbers);
    merged.addAllFunctionsExecution(source2.functionsExecution);
    merged.addAllBranches(source2.branches);
    merged.lineCoverage.add(source2.lineCoverage);
    return merged;
  }

  public String sourceFileName() {
    return sourceFileName;
  }

  public int nrFunctionsFound() {
    return functionsExecution.size();
  }

  public int nrFunctionsHit() {
    return (int)
        functionsExecution.entrySet().stream().filter(function -> function.getValue() > 0).count();
  }

  public int nrBranchesFound() {
    return branches.size();
  }

  public int nrBranchesHit() {
    return (int) branches.values().stream().filter(BranchCoverage::wasExecuted).count();
  }

  public int nrOfLinesWithNonZeroExecution() {
    return lineCoverage.numberOfExecutedLines();
  }

  public int nrOfInstrumentedLines() {
    return lineCoverage.numberOfInstrumentedLines();
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
    TreeMap<Integer, Long> result = new TreeMap<>();
    for (Entry<Integer, Long> entry : lineCoverage) {
      result.put(entry.getKey(), entry.getValue());
    }
    return result;
  }

  public Iterable<Entry<Integer, Long>> getAllLines() {
    return lineCoverage;
  }

  void addFunctionLineNumber(String functionName, Integer lineNumber) {
    this.functionLineNumbers.put(functionName, lineNumber);
  }

  void addAllFunctionLineNumbers(SortedMap<String, Integer> lineNumber) {
    this.functionLineNumbers.putAll(lineNumber);
  }

  public void addFunctionExecution(String functionName, Long executionCount) {
    long value = functionsExecution.getOrDefault(functionName, 0L) + executionCount;
    this.functionsExecution.put(functionName, value);
  }

  private void addAllFunctionsExecution(SortedMap<String, Long> functionsExecution) {
    for (Entry<String, Long> entry : functionsExecution.entrySet()) {
      addFunctionExecution(entry.getKey(), entry.getValue());
    }
  }

  /**
   * Creates and adds a new branch to the source file. If the branch already exists, the execution
   * count and evaluated status are combined with the existing one.
   *
   * @param lineNumber The line number the branch is on
   * @param blockNumber ID for the block containing the branch
   * @param branchNumber ID for the specific branch at this line
   * @param evaluated Whether branches for this line were ever evaluated
   * @param executionCount How many times this particular branch was taken
   */
  public void addNewBranch(
      int lineNumber,
      String blockNumber,
      String branchNumber,
      boolean evaluated,
      long executionCount) {
    BranchCoverage branch =
        BranchCoverage.create(lineNumber, blockNumber, branchNumber, evaluated, executionCount);
    addBranch(lineNumber, branch);
  }

  /**
   * Adds the given branch to the source file. If the branch already exists, it is merged with the
   * existing one.
   */
  public void addBranch(Integer lineNumber, BranchCoverage branch) {
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

  private void addAllBranches(ListMultimap<Integer, BranchCoverage> branches) {
    for (Entry<Integer, BranchCoverage> entry : branches.entries()) {
      addBranch(entry.getKey(), entry.getValue());
    }
  }

  public void addLine(int lineNumber, long executionCount) {
    lineCoverage.addLine(lineNumber, executionCount);
  }
}
