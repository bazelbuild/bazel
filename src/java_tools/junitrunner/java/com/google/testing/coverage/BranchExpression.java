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
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/** A branch coverage that must be evaluated as a combination of probes. */
public class BranchExpression implements CoverageExpression {
  private final List<CoverageExpression> branches;

  // Cache the evaluation result to avoid reevaluating the expression with the same probes.
  private boolean[] probesUsed;
  private boolean value = false;

  private BranchExpression(List<CoverageExpression> branches) {
    this.branches = branches;
  }

  /** Create a new BranchExpression using this CoverageExpression as the only branch. */
  private BranchExpression(CoverageExpression exp) {
    branches = new ArrayList<>();
    branches.add(exp);
  }

  /** Make a new BranchExpression with a branch for each of the given expressions. */
  public static BranchExpression create(CoverageExpression... expressions) {
    return new BranchExpression(new ArrayList<>(Arrays.asList(expressions)));
  }

  /** Make a new BranchExpression with a branch for each of the given expressions. */
  public static BranchExpression create(List<CoverageExpression> expressions) {
    return new BranchExpression(new ArrayList<>(expressions));
  }

  /** Returns true if any branches been set for this BranchExpression. */
  public boolean hasBranches() {
    return branches.stream().anyMatch(exp -> !exp.equals(NullExpression.NULL_EXP));
  }

  /**
   * Returns the expressions for the logical branches.
   *
   * <p>Expressions that have not been set are omitted.
   */
  public List<CoverageExpression> getBranches() {
    return branches.stream()
        .filter(exp -> !exp.equals(NullExpression.NULL_EXP))
        .collect(Collectors.toList());
  }

  /** Set the expression at a given index for this branch. */
  public void setBranchAtIndex(int index, CoverageExpression exp) {
    extendBranches(index + 1);
    branches.set(index, exp);
    invalidateEvalCache();
  }

  /** Returns the expression at a given index for this branch. */
  public CoverageExpression getBranchAtIndex(int index) {
    return branches.get(index);
  }

  /** Expands the current branch set to the new size */
  private void extendBranches(int size) {
    if (branches.size() < size) {
      // This preserves the cached eval value so no need to invalidate.
      branches.addAll(Collections.nCopies(size - branches.size(), NullExpression.NULL_EXP));
    }
  }

  /**
   * Add an expression to a branch expression.
   *
   * @return the index of the newly added branch.
   */
  public int add(CoverageExpression exp) {
    branches.add(exp);
    invalidateEvalCache();
    return branches.size() - 1;
  }

  /** Make a new BranchExpression representing the concatenation of branches in inputs. */
  public static BranchExpression concatenate(BranchExpression first, BranchExpression second) {
    List<CoverageExpression> branches = new ArrayList<>(first.branches);
    branches.addAll(second.branches);
    return new BranchExpression(branches);
  }

  /** Make a new BranchExpression representing the pairwise union of branches in inputs */
  public static BranchExpression zip(BranchExpression left, BranchExpression right) {
    List<CoverageExpression> zippedBranches = new ArrayList<>();
    int leftSize = left.branches.size();
    int rightSize = right.branches.size();
    int i;
    for (i = 0; i < leftSize && i < rightSize; i++) {
      List<CoverageExpression> branches =
          Arrays.asList(left.branches.get(i), right.branches.get(i));
      zippedBranches.add(new BranchExpression(branches));
    }
    List<CoverageExpression> remainder = leftSize < rightSize ? right.branches : left.branches;
    for (; i < remainder.size(); i++) {
      zippedBranches.add(new BranchExpression(remainder.get(i)));
    }
    return new BranchExpression(zippedBranches);
  }

  /** Wraps a CoverageExpression in a BranchExpression if it isn't one already. */
  public static BranchExpression ensureIsBranchExpression(CoverageExpression exp) {
    return exp instanceof BranchExpression ? (BranchExpression) exp : new BranchExpression(exp);
  }

  private void invalidateEvalCache() {
    probesUsed = null;
  }

  @Override
  public boolean eval(final boolean[] probes) {
    if (probes == probesUsed) {
      return value;
    }
    value = false;
    for (CoverageExpression exp : branches) {
      value = exp.eval(probes);
      if (value) {
        break;
      }
    }
    probesUsed = probes;
    return value;
  }
}
