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

import com.google.auto.value.AutoValue;

/**
 * An identifier (line, block, branch) for a particular branch.
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
abstract class BranchCoverageKey implements Comparable<BranchCoverageKey> {

  /**
   * Creates an instance of a BranchCoverage.
   *
   * @param lineNumber The line number the branch is on
   * @param blockNumber GCC internal ID - often "0"
   * @param branchNumber ID for the specific branch at this line
   */
  static BranchCoverageKey create(int lineNumber, String blockNumber, String branchNumber) {
    return new AutoValue_BranchCoverageKey(lineNumber, blockNumber, branchNumber);
  }

  abstract int lineNumber();

  /** Internal GCC ID for the branch. */
  abstract String blockNumber();

  /** Either the internal GCC ID for the branch or an increasing counter for BA lines. */
  abstract String branchNumber();

  @Override
  public int compareTo(BranchCoverageKey other) {
    int lineDiff = this.lineNumber() - other.lineNumber();
    if (lineDiff != 0) {
      return lineDiff;
    }
    int blockDiff = this.blockNumber().compareTo(other.blockNumber());
    if (blockDiff != 0) {
      return blockDiff;
    }
    return this.branchNumber().compareTo(other.branchNumber());
  }
}
