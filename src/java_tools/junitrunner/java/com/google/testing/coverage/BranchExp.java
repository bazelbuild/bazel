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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** A branch coverage that must be evaluated as a combination of probes. */
public class BranchExp implements CovExp {
  private final List<CovExp> branches;

  private boolean hasValue;
  private boolean[] probesUsed;
  private boolean value;

  public BranchExp(List<CovExp> branches) {
    this.branches = branches;
    hasValue = false;
  }

  /** Create a new BranchExp using this CovExp as the only branch. */
  public BranchExp(CovExp exp) {
    branches = new ArrayList<CovExp>();
    branches.add(exp);
    hasValue = false;
  }

  /** Gets the expressions for the branches. */
  public List<CovExp> getBranches() {
    return branches;
  }

  /**
   * Add an expression to a branch expression.
   *
   * @return the index of the newly added branch.
   */
  public int add(CovExp exp) {
    branches.add(exp);
    return branches.size() - 1;
  }

  /** Update an existing branch expression. */
  public void update(int idx, CovExp exp) {
    branches.set(idx, exp);
  }

  /** Make a new BranchExp representing the concatenation of branches in inputs. */
  public static BranchExp concatenate(BranchExp first, BranchExp second) {
    List<CovExp> branches = new ArrayList<>(first.branches);
    branches.addAll(second.branches);
    return new BranchExp(branches);
  }

  /** Make a new BranchExp representing the pairwise union of branches in inputs */
  public static BranchExp zip(BranchExp left, BranchExp right) {
    List<CovExp> zippedBranches = new ArrayList<>();
    int leftSize = left.branches.size();
    int rightSize = right.branches.size();
    int i;
    for (i = 0; i < leftSize && i < rightSize; i++) {
      List<CovExp> branches = Arrays.asList(left.branches.get(i), right.branches.get(i));
      zippedBranches.add(new BranchExp(branches));
    }
    List<CovExp> remainder = leftSize < rightSize ? right.branches : left.branches;
    for (; i < remainder.size(); i++) {
      zippedBranches.add(new BranchExp(remainder.get(i)));
    }
    return new BranchExp(zippedBranches);
  }

  /** Wraps a CovExp in a BranchExp if it isn't one already. */
  public static BranchExp ensureIsBranchExp(CovExp exp) {
    return exp instanceof BranchExp ? (BranchExp) exp : new BranchExp(exp);
  }

  @Override
  public boolean eval(final boolean[] probes) {
    if (hasValue && probes == probesUsed) {
      return value;
    }
    value = false;
    for (CovExp exp : branches) {
      value = exp.eval(probes);
      if (value) {
        break;
      }
    }
    hasValue = value; // The value is cached.
    probesUsed = probes;
    return value;
  }
}
