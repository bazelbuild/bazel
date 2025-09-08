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
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
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
  private final Map<BranchCoverageKey, Long> branchExecutions; // branch to execution count
  private final Map<Integer, Boolean> branchEvaluations; // line to branch evaluations
  private final LineCoverage lineCoverage;

  public SourceFileCoverage(String sourcefile) {
    this.sourceFileName = sourcefile;
    this.functionsExecution = new TreeMap<>();
    this.functionLineNumbers = new TreeMap<>();
    this.lineCoverage = LineCoverage.create();
    this.branchExecutions = new HashMap<>();
    this.branchEvaluations = new HashMap<>();
  }

  SourceFileCoverage(SourceFileCoverage other) {
    this.sourceFileName = other.sourceFileName;

    this.functionsExecution = new TreeMap<>();
    this.functionLineNumbers = new TreeMap<>();
    this.branchExecutions = new HashMap<>();
    this.branchEvaluations = new HashMap<>();

    this.functionLineNumbers.putAll(other.functionLineNumbers);
    this.functionsExecution.putAll(other.functionsExecution);
    this.branchExecutions.putAll(other.branchExecutions);
    this.branchEvaluations.putAll(other.branchEvaluations);
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
    merged.addAllBranches(source2.branchExecutions, source2.branchEvaluations);
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
    return branchExecutions.size();
  }

  public int nrBranchesHit() {
    return (int) branchExecutions.values().stream().filter(x -> x > 0).count();
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

  @VisibleForTesting
  Map<BranchCoverageKey, Long> getBranchExecutions() {
    return branchExecutions;
  }

  public Set<Entry<String, Long>> getAllExecutionCount() {
    return functionsExecution.entrySet();
  }

  public Collection<BranchCoverageItem> getAllBranches() {
    // this is not efficient, but should only ever be called when printing out lcov
    ImmutableList.Builder<BranchCoverageItem> builder = ImmutableList.builder();
    ArrayList<BranchCoverageKey> sortedKeys = new ArrayList<>(branchExecutions.keySet());
    Collections.sort(sortedKeys);
    for (BranchCoverageKey branch : sortedKeys) {
      long executionCount = branchExecutions.get(branch);
      boolean evaluated = branchEvaluations.getOrDefault(branch.lineNumber(), false);
      builder.add(
          BranchCoverageItem.create(
              branch.lineNumber(),
              branch.blockNumber(),
              branch.branchNumber(),
              evaluated,
              executionCount));
    }
    return builder.build();
  }

  public long getBranchExecutionCount(BranchCoverageKey branch) {
    return branchExecutions.getOrDefault(branch, 0L);
  }

  public boolean isBranchEvaluated(int lineNumber) {
    return branchEvaluations.getOrDefault(lineNumber, false);
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
   * Adds a new branch to the source file. If the branch already exists, the execution count and
   * evaluated status are combined with the existing one.
   *
   * @param lineNumber The line number the branch is on
   * @param blockNumber ID for the block containing the branch
   * @param branchNumber ID for the specific branch at this line
   * @param evaluated Whether branches for this line were ever evaluated
   * @param executionCount How many times this particular branch was taken
   */
  public void addBranch(
      int lineNumber,
      String blockNumber,
      String branchNumber,
      boolean evaluated,
      long executionCount) {
    BranchCoverageKey branch = BranchCoverageKey.create(lineNumber, blockNumber, branchNumber);
    addBranchFromKey(branch, evaluated, executionCount);
  }

  private void addBranchFromKey(BranchCoverageKey branch, boolean evaluated, long executionCount) {
    long prevCount = branchExecutions.getOrDefault(branch, 0L);
    branchExecutions.put(branch, prevCount + executionCount);
    branchEvaluations.put(
        branch.lineNumber(),
        evaluated || branchEvaluations.getOrDefault(branch.lineNumber(), false));
  }

  private void addAllBranches(
      Map<BranchCoverageKey, Long> branchExecutions, Map<Integer, Boolean> branchEvaluations) {
    for (Entry<BranchCoverageKey, Long> entry : branchExecutions.entrySet()) {
      BranchCoverageKey branch = entry.getKey();
      long executionCount = entry.getValue();
      boolean evaluated = branchEvaluations.getOrDefault(branch.lineNumber(), false);
      addBranchFromKey(branch, evaluated, executionCount);
    }
  }

  public void addLine(int lineNumber, long executionCount) {
    lineCoverage.addLine(lineNumber, executionCount);
  }
}
