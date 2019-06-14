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

  /** Make a union of the branches of two BranchExp. */
  public void merge(BranchExp other) {
    branches.addAll(other.branches);
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
