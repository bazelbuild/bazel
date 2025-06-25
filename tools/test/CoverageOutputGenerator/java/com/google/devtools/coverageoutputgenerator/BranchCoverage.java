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

import static com.google.common.base.Verify.verify;
import static java.lang.Math.max;

import com.google.auto.value.AutoValue;

/**
 * Stores branch coverage information.
 *
 * <p>Corresponds to either a BRDA or BA (Google only) line in an lcov report.
 *
 * <p>BA lines correspond to instances where blockNumber and branchNumber are set to empty Strings
 * and have the form:
 *
 * <pre>BA:[line_number],[taken]</pre>
 *
 * In this case, nrOfExecutions() actually refers to the "taken" value where:
 *
 * <ul>
 *   <li>0 = Branch was never evaluated (evaluated() == false)
 *   <li>1 = Branch was evaluated but never taken
 *   <li>2 = Branch was taken
 * </ul>
 *
 * BRDA lines set have the form
 *
 * <pre>BRDA:[line_number],[block_number],[branch_number],[taken]</pre>
 *
 * where the block and branch numbers are internal identifiers, and taken is either "-" if the
 * branch condition was never evaluated or a number indicating how often the branch was taken (which
 * may be 0).
 */
@AutoValue
abstract class BranchCoverage {

  /**
   * Create a BranchCoverage object corresponding to a BA line
   *
   * <pre>BA:[line_number],[taken]</pre>
   *
   * <p>The branch number is not part of the BA line so must be calculated by the caller. It is used
   * for reconciling branches between reports when merging.
   *
   * @param lineNumber line number the branch comes from
   * @param branchNumber the index of this branch in the line; only used for merging reports
   * @param value the taken value, 0, 1, 2
   * @return corresponding BranchCoverage
   */
  static BranchCoverage create(int lineNumber, int branchNumber, long value) {
    verify(0 <= value && value < 3, "Taken value must be one of {0, 1, 2}");
    return new AutoValue_BranchCoverage(
        lineNumber, /* blockNumber= */ "", Integer.toString(branchNumber), value > 0, value);
  }

  /**
   * Create a BranchCoverage object corresponding to a BRDA line with a dummy block number
   *
   * <pre>BRDA:[line_number],[block_number=0],[branch_number],[taken]</pre>
   *
   * @param lineNumber line number the branch comes from
   * @param branchNumber id for the specific branch at this line
   * @param evaluated if this branch was evaluated (taken != "-")
   * @param nrOfExecutions how many times the branch was taken (the value of taken if taken != "-")
   * @return corresponding BranchCoverage
   */
  static BranchCoverage createWithDummyBlock(
      int lineNumber, String branchNumber, boolean evaluated, long nrOfExecutions) {
    return new AutoValue_BranchCoverage(
        lineNumber, /*blockNumber=*/ "0", branchNumber, evaluated, nrOfExecutions);
  }

  /**
   * Create a BranchCoverage object corresponding to a BRDA line
   *
   * <pre>BRDA:[line_number],[block_number],[branch_number],[taken]</pre>
   *
   * @param lineNumber line number the branch comes from
   * @param blockNumber GCC internal block id
   * @param branchNumber id for the specific branch at this line
   * @param evaluated if this branch was evaluated (taken != "-")
   * @param nrOfExecutions how many times the branch was taken (the value of taken if taken != "-")
   * @return corresponding BranchCoverage
   */
  static BranchCoverage createWithBlockAndBranch(
      int lineNumber,
      String blockNumber,
      String branchNumber,
      boolean evaluated,
      long nrOfExecutions) {
    return new AutoValue_BranchCoverage(
        lineNumber, blockNumber, branchNumber, evaluated, nrOfExecutions);
  }

  /**
   * Merges two given instances of {@link BranchCoverage}.
   *
   * <p>Calling {@code lineNumber()}, {@code blockNumber()} and {@code branchNumber()} must return
   * the same values for {@code first} and {@code second}.
   */
  static BranchCoverage merge(BranchCoverage first, BranchCoverage second) {
    verify(first.lineNumber() == second.lineNumber(), "Branch line numbers must match");
    verify(first.blockNumber().equals(second.blockNumber()), "Branch block numbers must match");
    verify(first.branchNumber().equals(second.branchNumber()), "Branch branch numbers must match");
    return first.blockNumber().isEmpty()
        ? mergeWithNoBlockAndBranch(first, second)
        : mergeWithBlockAndBranch(first, second);
  }

  private static BranchCoverage mergeWithBlockAndBranch(
      BranchCoverage first, BranchCoverage second) {
    return createWithBlockAndBranch(
        first.lineNumber(),
        first.blockNumber(),
        first.branchNumber(),
        first.evaluated() || second.evaluated(),
        first.nrOfExecutions() + second.nrOfExecutions());
  }

  private static BranchCoverage mergeWithNoBlockAndBranch(
      BranchCoverage first, BranchCoverage second) {
    long value = max(first.nrOfExecutions(), second.nrOfExecutions());
    verify(0 <= value && value < 3, "Taken value must be one of {0, 1, 2}");
    return createWithBlockAndBranch(
        first.lineNumber(), first.blockNumber(), first.branchNumber(), value > 0, value);
  }

  abstract int lineNumber();

  /**
   * Internal GCC ID for the branch.
   *
   * <p>Empty for BA lines.
   */
  abstract String blockNumber();

  /** Either the internal GCC ID for the branch or an increasing counter for BA lines. */
  abstract String branchNumber();

  abstract boolean evaluated();

  abstract long nrOfExecutions();

  boolean wasExecuted() {
    // if there's no block number then this is a BA branch so only taken if the "nrOfExecutions"
    // value == 2 (since it refers to the BA taken value)
    // otherwise it really is an execution count, so a count > 0 means the branch was executed
    return blockNumber().isEmpty() ? nrOfExecutions() == 2 : nrOfExecutions() > 0;
  }
}
