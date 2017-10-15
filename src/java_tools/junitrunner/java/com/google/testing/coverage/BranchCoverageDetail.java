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

import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/** Details of branch coverage information. */
public class BranchCoverageDetail {
  private final Map<Integer, BitField> branchTaken;
  private final Map<Integer, Integer> branches;

  public BranchCoverageDetail() {
    branchTaken = new TreeMap<Integer, BitField>();
    branches = new TreeMap<Integer, Integer>();
  }

  private BitField getBranchForLine(int line) {
    BitField value = branchTaken.get(line);
    if (value != null) {
      return value;
    }
    value = new BitField();
    branchTaken.put(line, value);
    return value;
  }

  /** Returns true if the line has branches. */
  public boolean hasBranches(int line) {
    return branches.containsKey(line);
  }

  /** Sets the number of branches entry. */
  public void setBranches(int line, int n) {
    branches.put(line, n);
  }

  /** Gets the number of branches in the line, returns 0 if there is no branch. */
  public int getBranches(int line) {
    Integer value = branches.get(line);
    if (value == null) {
      return 0;
    }
    return value;
  }

  /** Sets the taken bit of the given line for the given branch index. */
  public void setTakenBit(int line, int branchIdx) {
    getBranchForLine(line).setBit(branchIdx);
  }

  public boolean getTakenBit(int line, int branchIdx) {
    return getBranchForLine(line).isBitSet(branchIdx);
  }

  /** Calculate executed bit using heuristics. */
  public boolean getExecutedBit(int line) {
    // If any of the branch is taken, the branch must have executed. Otherwise assume it is not.
    return getBranchForLine(line).any();
  }

  /** Returns line numbers where more than one branch is present. */
  public Set<Integer> linesWithBranches() {
    Set<Integer> result = new TreeSet<Integer>();
    for (int i : branches.keySet()) {
      if (branches.get(i) > 1) {
        result.add(i);
      }
    }
    return result;
  }
}
