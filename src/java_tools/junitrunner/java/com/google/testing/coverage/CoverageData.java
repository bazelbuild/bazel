// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.coverage;

import static com.google.common.collect.ImmutableSortedSet.toImmutableSortedSet;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Maps;
import com.google.common.collect.Ordering;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.NoSuchElementException;

/** Coverage data from the analysis of either a class or a source file. */
public final class CoverageData {
  private final ImmutableMap<Integer, BranchData> branches; // map of line-number to BranchData
  private final BitField linesExecuted;
  private final BitField linesInstrumented;
  private final ImmutableMap<String, Integer> methodStartLines;
  private final ImmutableMap<String, Boolean> methodExecuted;

  private static final BranchData NULL_BRANCH_DATA = BranchData.builder().build();

  private CoverageData(
      ImmutableMap<Integer, BranchData> branches,
      BitField linesExecuted,
      BitField linesInstrumented,
      ImmutableMap<String, Integer> methodStartLines,
      ImmutableMap<String, Boolean> methodExecuted) {
    this.branches = branches;
    this.linesExecuted = linesExecuted;
    this.linesInstrumented = linesInstrumented;
    this.methodStartLines = methodStartLines;
    this.methodExecuted = methodExecuted;
  }

  public static Builder builder() {
    return new Builder();
  }

  /** Returns true if there is no coverage data. */
  public boolean isEmpty() {
    return branches.isEmpty()
        && linesInstrumented.isEmpty()
        && methodStartLines.isEmpty()
        && methodExecuted.isEmpty();
  }

  /** Returns true if the line has branches. */
  public boolean hasBranches(Integer line) {
    return branches.containsKey(line);
  }

  /** Gets BranchData for this line */
  public BranchData getBranches(Integer line) {
    if (!hasBranches(line)) {
      return NULL_BRANCH_DATA;
    }
    return branches.get(line);
  }

  /** Returns line numbers where more than one branch is present. */
  public ImmutableSortedSet<Integer> linesWithBranches() {
    return branches.entrySet().stream()
        // Omit branches that are not actually branches (unconditional jumps)
        .filter(entry -> entry.getValue().size() > 1)
        .map(Map.Entry::getKey)
        .collect(toImmutableSortedSet(Ordering.natural()));
  }

  /** Returns the names of all methods in this coverage data. */
  public ImmutableList<String> getMethods() {
    return ImmutableList.copyOf(methodStartLines.keySet());
  }

  /** Returns the line number for the first line of the given method. */
  public int getMethodLine(String method) {
    return methodStartLines.get(method);
  }

  /** Returns if the given method was executed. */
  public boolean isMethodExecuted(String method) {
    return methodExecuted.get(method);
  }

  /** Returns if the given line was executed. */
  public boolean isLineExecuted(int line) {
    return linesExecuted.isBitSet(line);
  }

  /** Were any branches on this line executed. */
  public boolean isBranchExecuted(int line) {
    return branches.containsKey(line) && getBranches(line).anyBranchTaken();
  }

  /** Returns the line number for all instrumented lines. */
  public ImmutableList<Integer> getInstrumentedLines() {
    return ImmutableList.copyOf(new InstrumentedLinesIterator(linesInstrumented));
  }

  /** Iterator for instrumented lines. */
  private static class InstrumentedLinesIterator implements Iterator<Integer> {

    private final BitField linesInstrumented;
    private int idx;
    private final int maxIdx;

    InstrumentedLinesIterator(BitField linesInstrumented) {
      this.linesInstrumented = linesInstrumented;
      maxIdx = linesInstrumented.sizeInBits();

      // Move idx to point to the first populated bit.
      idx = -1;
      moveToNextPopulated();
    }

    @Override
    public boolean hasNext() {
      return idx < maxIdx;
    }

    @Override
    public Integer next() {
      // if idx < maxIdx then it points to a valid element
      if (idx >= maxIdx) {
        throw new NoSuchElementException();
      }
      int v = idx;
      moveToNextPopulated();
      return v;
    }

    private void moveToNextPopulated() {
      do {
        idx++;
      } while (idx < maxIdx && !linesInstrumented.isBitSet(idx));
    }
  }

  /** Builder for CoverageData. */
  public static final class Builder {
    private final Map<Integer, BranchData.Builder> branches = new HashMap<>();
    private final BitField linesExecuted = new BitField();
    private final BitField linesInstrumented = new BitField();
    private final Map<String, Integer> methodStartLines = new LinkedHashMap<>();
    private final Map<String, Boolean> methodExecuted = new LinkedHashMap<>();

    /**
     * Adds a method to the coverage data.
     *
     * <p>This overwrites any values for an existing method name.
     */
    @CanIgnoreReturnValue
    public Builder addMethod(String method, int line, boolean executed) {
      methodStartLines.put(method, line);
      methodExecuted.put(method, executed);
      return this;
    }

    /**
     * Adds a line to the coverage data.
     *
     * <p>This overwrites any values for an existing line.
     */
    @CanIgnoreReturnValue
    public Builder addLine(int line, boolean executed) {
      linesInstrumented.setBit(line);
      if (executed) {
        linesExecuted.setBit(line);
      }
      return this;
    }

    /**
     * Adds a branch to the coverage data for the given line.
     *
     * <p>Branches for a line are stored in the order they were added by this method.
     */
    @CanIgnoreReturnValue
    public Builder addBranch(int line, boolean taken) {
      BranchData.Builder branchBuilder =
          branches.computeIfAbsent(line, unused -> BranchData.builder());
      branchBuilder.addBranch(taken);
      return this;
    }

    /** Returns true if no coverage data has been added. */
    public boolean isEmpty() {
      return branches.isEmpty()
          && linesInstrumented.isEmpty()
          && methodStartLines.isEmpty()
          && methodExecuted.isEmpty();
    }

    /** Returns a new CoverageData from this builder. */
    public CoverageData build() {
      return new CoverageData(
          ImmutableMap.copyOf(Maps.transformValues(branches, BranchData.Builder::build)),
          linesExecuted,
          linesInstrumented,
          ImmutableMap.copyOf(methodStartLines),
          ImmutableMap.copyOf(methodExecuted));
    }
  }

  /** Data for a set of branches on a single line. */
  public static final class BranchData {
    private final BitField branchBits;
    private final int count;

    private BranchData(BitField branchBits, int count) {
      this.branchBits = branchBits;
      this.count = count;
    }

    /** Creates a new Builder for BranchData. */
    public static Builder builder() {
      return new Builder();
    }

    /** Returns true if the branch at the given index was taken. */
    public boolean isBranchTaken(int idx) {
      if (idx >= count) {
        throw new IndexOutOfBoundsException(
            String.format("Branch index %d is out of bounds [0, %d)", idx, count));
      }
      return branchBits.isBitSet(idx);
    }

    /** Returns true if any branch was taken. */
    public boolean anyBranchTaken() {
      return branchBits.any();
    }

    /** Returns the coverage for each branch on this line. */
    public ImmutableList<Boolean> getBranchValues() {
      ImmutableList.Builder<Boolean> result = ImmutableList.builder();
      for (int i = 0; i < count; i++) {
        result.add(branchBits.isBitSet(i));
      }
      return result.build();
    }

    /** Returns the number of branches on this line. */
    public int size() {
      return count;
    }

    /** Returns true if there are no branches on this line. */
    public boolean isEmpty() {
      return count == 0;
    }

    /** Builder for BranchData. */
    public static class Builder {
      private final BitField branchBits;
      private int count;

      public Builder() {
        branchBits = new BitField();
      }

      /**
       * Adds a branch to the BranchData.
       *
       * <p>Branches are stored in the order they were added by this method.
       */
      @CanIgnoreReturnValue
      public Builder addBranch(boolean taken) {
        if (taken) {
          branchBits.setBit(count);
        }
        count++;
        return this;
      }

      /** Returns a new BranchData from this builder. */
      public BranchData build() {
        return new BranchData(branchBits, count);
      }
    }
  }
}
